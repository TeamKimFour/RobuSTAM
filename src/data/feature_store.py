"""Feature Store 입출력 — 가공된 187차원 피처 행렬의 저장·조회 계층.

저장소는 두 부분으로 구성된다 (CLAUDE.md 스택: Apache Arrow(Parquet) + SQLite):
  - Parquet: 실제 피처 행렬을 fold/split 파티션으로 저장 (wide 포맷, 컬럼 = schema.feature_names).
  - SQLite 메타DB: fold 경계·컬럼맵·정규화 통계·run 정보. 데이터 산출물은 gitignore되므로
    "어떤 config·통계로 만들어졌나"의 감사 기록이자 추론(도현) 재사용의 근거가 된다.

실제 피처는 2주차 지표·정규화 후 채워진다. 본 모듈은 입출력 골격으로, 합성 데이터로 검증된다.

파티션 레이아웃:
    data/feature_store/fold=<id>/split=<train|valid|test>/part.parquet
"""

import json
import sqlite3
from pathlib import Path

import pandas as pd

SPLITS = ("train", "valid", "test")


# ── Parquet 입출력 ──
def partition_path(out_dir: str, fold_id: int, split: str) -> Path:
    """fold/split 파티션의 Parquet 경로."""
    if split not in SPLITS:
        raise ValueError(f"split은 {SPLITS} 중 하나여야 합니다: {split!r}")
    return Path(out_dir) / f"fold={fold_id}" / f"split={split}" / "part.parquet"


def write_features(
    df: pd.DataFrame,
    out_dir: str,
    fold_id: int,
    split: str,
    expected_columns: list[str] | None = None,
) -> Path:
    """피처 행렬을 fold/split 파티션 Parquet으로 저장한다.

    df: index=date, columns=schema.feature_names(W) (wide 포맷).
    expected_columns가 주어지면 컬럼 규격 일치를 검증한다(차원 누락·순서 오류 방어).
    """
    if expected_columns is not None and list(df.columns) != list(expected_columns):
        raise ValueError(
            f"컬럼 규격 불일치: {len(df.columns)}개 (기대 {len(expected_columns)}개). "
            "schema.feature_names와 일치해야 합니다."
        )
    path = partition_path(out_dir, fold_id, split)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path)
    return path


def load_features(out_dir: str, fold_id: int, split: str) -> pd.DataFrame:
    """fold/split 파티션 Parquet을 읽는다."""
    path = partition_path(out_dir, fold_id, split)
    if not path.is_file():
        raise FileNotFoundError(f"Feature Store 파티션이 없습니다: {path}")
    return pd.read_parquet(path)


# ── SQLite 메타DB ──
def init_meta_db(db_path: str) -> None:
    """메타DB 스키마를 생성한다(존재하면 무시)."""
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db_path) as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS runs (
                run_id TEXT PRIMARY KEY,
                created_at TEXT,
                config_hash TEXT,
                window_w INTEGER,
                state_dim INTEGER,
                asset_order TEXT
            );
            CREATE TABLE IF NOT EXISTS folds (
                run_id TEXT, fold_id INTEGER, mode TEXT,
                train_start TEXT, train_end TEXT,
                valid_start TEXT, valid_end TEXT,
                test_start TEXT, test_end TEXT,
                embargo_days INTEGER,
                PRIMARY KEY (run_id, fold_id)
            );
            CREATE TABLE IF NOT EXISTS feature_columns (
                run_id TEXT, col_index INTEGER, col_name TEXT,
                PRIMARY KEY (run_id, col_index)
            );
            CREATE TABLE IF NOT EXISTS scaler_stats (
                run_id TEXT, fold_id INTEGER, feature_name TEXT,
                mean REAL, std REAL,
                PRIMARY KEY (run_id, fold_id, feature_name)
            );
            """
        )


def write_run(
    db_path: str,
    run_id: str,
    created_at: str,
    config_hash: str,
    window_w: int,
    state_dim: int,
    asset_order: list[str],
) -> None:
    """run 메타(설정 스냅샷)를 기록한다."""
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            "INSERT OR REPLACE INTO runs VALUES (?,?,?,?,?,?)",
            (run_id, created_at, config_hash, window_w, state_dim, json.dumps(asset_order)),
        )


def write_feature_columns(db_path: str, run_id: str, columns: list[str]) -> None:
    """187개 컬럼명·인덱스를 영속화한다(차원 검증·인덱스맵 감사용)."""
    with sqlite3.connect(db_path) as conn:
        conn.executemany(
            "INSERT OR REPLACE INTO feature_columns VALUES (?,?,?)",
            [(run_id, i, name) for i, name in enumerate(columns)],
        )


def write_fold(db_path: str, run_id: str, fold: dict) -> None:
    """fold 경계를 기록한다. fold dict 키: fold_id, mode, train/valid/test_start·end, embargo_days."""
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            "INSERT OR REPLACE INTO folds VALUES (?,?,?,?,?,?,?,?,?,?)",
            (
                run_id, fold["fold_id"], fold.get("mode", "expanding"),
                fold.get("train_start"), fold.get("train_end"),
                fold.get("valid_start"), fold.get("valid_end"),
                fold.get("test_start"), fold.get("test_end"),
                fold.get("embargo_days"),
            ),
        )


def read_feature_columns(db_path: str, run_id: str) -> list[str]:
    """기록된 컬럼명을 인덱스 순서로 반환한다."""
    with sqlite3.connect(db_path) as conn:
        rows = conn.execute(
            "SELECT col_name FROM feature_columns WHERE run_id=? ORDER BY col_index", (run_id,)
        ).fetchall()
    return [r[0] for r in rows]


def read_folds(db_path: str, run_id: str) -> list[dict]:
    """기록된 fold 목록을 반환한다."""
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT * FROM folds WHERE run_id=? ORDER BY fold_id", (run_id,)
        ).fetchall()
    return [dict(r) for r in rows]


def config_hash(cfg: dict) -> str:
    """config의 핵심값 해시 — 차원·자산순서 변경을 탐지(저장 데이터와 코드 정합성)."""
    import hashlib

    key = json.dumps(
        {"assets": cfg.get("assets"), "window": cfg.get("window")}, sort_keys=True
    )
    return hashlib.sha1(key.encode("utf-8")).hexdigest()[:12]
