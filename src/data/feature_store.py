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
                asset_order TEXT,
                combo TEXT
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
    combo: str | None = None,
) -> None:
    """run 메타(설정 스냅샷)를 기록한다.

    combo: 이 run을 만든 피처 콤보 이름(M0~M3 등, 감사용). 콤보별로 meta.sqlite 자체가
    분리되므로(config_loader.get_meta_db) 필수는 아니지만, 어느 콤보의 빌드였는지 한눈에
    보이도록 남긴다. 생략(None)해도 기존 호출부는 그대로 동작한다.
    """
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            "INSERT OR REPLACE INTO runs VALUES (?,?,?,?,?,?,?)",
            (run_id, created_at, config_hash, window_w, state_dim, json.dumps(asset_order), combo),
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


def latest_run_id_for_fold(db_path: str, fold_id: int) -> str | None:
    """해당 fold를 가장 최근에 빌드한 build run_id를 반환한다(없으면 None).

    Parquet 파티션(`fold=<id>/split=...`)은 build()를 다시 돌릴 때마다 최신 run이
    덮어쓰므로, 디스크의 그 fold 데이터는 항상 **가장 최근 build**의 산출물이다. 학습
    provenance(어느 정규화 통계로 정규화됐나)는 그 최신 run_id를 가리켜야 하므로,
    folds×runs를 조인해 `runs.created_at` 최신 행의 run_id를 고른다.

    메타DB나 테이블이 아직 없으면(빌드 전) None을 돌려준다 — 학습이 이를 provenance
    미상으로 처리할 수 있게 한다.
    """
    if not Path(db_path).is_file():
        return None
    with sqlite3.connect(db_path) as conn:
        try:
            row = conn.execute(
                "SELECT f.run_id FROM folds f JOIN runs r ON f.run_id = r.run_id "
                "WHERE f.fold_id=? ORDER BY r.created_at DESC, f.run_id DESC LIMIT 1",
                (fold_id,),
            ).fetchone()
        except sqlite3.OperationalError:
            # folds/runs 테이블이 아직 생성되지 않은 경우(init 전) → provenance 미상.
            return None
    return row[0] if row else None


def write_scaler_stats(
    db_path: str, run_id: str, fold_id: int, rows: list[tuple[str, float, float]]
) -> None:
    """정규화 통계(feature_name, mean, std)를 fold 단위로 저장한다(추론 재사용·감사용)."""
    with sqlite3.connect(db_path) as conn:
        conn.executemany(
            "INSERT OR REPLACE INTO scaler_stats VALUES (?,?,?,?,?)",
            [(run_id, fold_id, name, float(mean), float(std)) for name, mean, std in rows],
        )


def latest_run_id_for_config(db_path: str, config_hash: str) -> str | None:
    """해당 `config_hash`로 만들어진 build run 중 가장 최근 run_id를 반환한다(없으면 None).

    `config.inference.scaler_run_id`를 비워둔 채로 배포할 때 쓰는 해석기다. build run_id는
    타임스탬프를 포함해 머신 간 재현이 불가능하지만, **같은 config로 재빌드하면 정규화 통계는
    사실상 동일**하다(실측: 182개 중 최대 상대오차 7.4e-4 — 조정가 반올림 노이즈 수준).
    따라서 통계 파일을 옮기는 대신 각자 build해서 자기 run을 쓰면 된다.

    `config_hash`로 좁히는 이유: 그냥 "최신 run"을 쓰면 `window`·자산구성이 다른 빌드의
    통계를 조용히 집어 차원·의미가 어긋날 수 있다. 해시가 같아야 최소한 State 규격이 같다.

    `config_hash`는 `assets·window·features·normalize·split`을 해싱한다 — 즉 **통계 값을
    바꾸는 설정이 모두 포함**되므로, 지표 파라미터나 fold 경계를 바꾼 build는 해시가 달라져
    자동 선택에서 자연히 걸러진다(그 경우 "build run이 없습니다"로 크게 실패한다).
    엄밀한 감사 추적이 필요하면 여전히 `scaler_run_id`를 명시해 고정하면 된다.
    """
    if not Path(db_path).is_file():
        return None
    with sqlite3.connect(db_path) as conn:
        try:
            row = conn.execute(
                "SELECT run_id FROM runs WHERE config_hash=? "
                "ORDER BY created_at DESC, run_id DESC LIMIT 1",
                (config_hash,),
            ).fetchone()
        except sqlite3.OperationalError:
            return None
    return row[0] if row else None


def read_scaler_stats(db_path: str, run_id: str, fold_id: int) -> dict[str, tuple[float, float]]:
    """fold의 정규화 통계를 {feature_name: (mean, std)}로 반환한다."""
    with sqlite3.connect(db_path) as conn:
        rows = conn.execute(
            "SELECT feature_name, mean, std FROM scaler_stats WHERE run_id=? AND fold_id=?",
            (run_id, fold_id),
        ).fetchall()
    return {name: (mean, std) for name, mean, std in rows}


# ── 원시 익일 수익률(targets) — 보상·백테스트용 (정규화 안 함) ──
def targets_path(out_dir: str, fold_id: int, split: str) -> Path:
    """fold/split 파티션의 targets(익일 수익률) Parquet 경로."""
    return partition_path(out_dir, fold_id, split).with_name("targets.parquet")


def write_targets(df: pd.DataFrame, out_dir: str, fold_id: int, split: str) -> Path:
    """익일 수익률 targets를 State와 같은 파티션에 저장한다(컬럼 fwd_ret_<asset>)."""
    path = targets_path(out_dir, fold_id, split)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path)
    return path


def load_targets(out_dir: str, fold_id: int, split: str) -> pd.DataFrame:
    """fold/split 파티션의 targets를 읽는다."""
    path = targets_path(out_dir, fold_id, split)
    if not path.is_file():
        raise FileNotFoundError(f"targets 파티션이 없습니다: {path}")
    return pd.read_parquet(path)


# 해시 대상 — **산출되는 State·정규화 통계를 바꾸는** config 섹션만 포함한다.
# 여기 빠진 항목이 바뀌면 해시가 그대로라, 자동 run 해석이 비호환 build를 조용히 고른다.
_HASH_KEYS = ("assets", "window", "features", "normalize", "split")


def config_hash(cfg: dict) -> str:
    """build 산출물의 **호환성 지문**. 같은 해시 = 같은 규격·같은 통계 정의.

    쓰임새가 두 가지다.
      ① `build.py`의 `run_id` 접두어 — 어떤 설정으로 만든 데이터인지 감사.
      ② `latest_run_id_for_config`의 **자동 run 해석 필터** — `inference.scaler_run_id`를
         비웠을 때 "이 build를 써도 되는가"를 판정하는 **유일한 안전장치**(이슈 #33·PR #36).

    ②가 생기면서 해시 범위가 곧 안전성이 됐다. 그래서 통계 값을 바꾸는 설정을 모두 넣는다.
      - `features` — 지표 파라미터(RSI 길이·MACD·bbands 등)가 바뀌면 지표 값 자체가 달라진다.
      - `normalize` — scope·eps가 바뀌면 μ/σ의 정의가 달라진다.
      - `split` — anchor·test_blocks·embargo가 바뀌면 **fold train 구간**이 달라져 통계가 달라진다.

    반대로 아래는 **일부러 제외**한다(통계를 바꾸지 않는데 해시만 흔들려 불필요한 재빌드를 부른다).
      - `data.start`/`end` — fold 경계는 anchor·test_blocks로 정해지고 신규 데이터는 뒤에만
        붙으므로, 기간을 늘려도 각 fold의 train 구간·통계는 그대로다.
      - `transaction_cost` — 환경·보상의 값이지 데이터 산출물과 무관하다.
      - `inference`·`model` — 소비 측 설정이라 build 산출물에 영향이 없다.

    호환성 메모: 해시 범위를 넓히면 **이전 run_id는 더 이상 매칭되지 않는다**. 명시적으로
    고정한 `scaler_run_id`는 그대로 동작하고, 자동 해석은 재빌드 후 새 run을 집는다
    (못 찾으면 "build run이 없습니다"로 조용히가 아니라 크게 실패한다).
    """
    import hashlib

    key = json.dumps(
        {k: cfg.get(k) for k in _HASH_KEYS}, sort_keys=True, default=str
    )
    return hashlib.sha1(key.encode("utf-8")).hexdigest()[:12]
