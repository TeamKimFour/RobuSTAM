"""feature_store.py 입출력 테스트.

Parquet round-trip은 pandas/pyarrow 필요(importorskip 가드). SQLite 메타는 표준 라이브러리.
실데이터(2주차) 없이 합성 187차원 행렬로 입출력 골격을 검증한다.
"""

import pytest

pd = pytest.importorskip("pandas")
np = pytest.importorskip("numpy")

from src.data import feature_store as fs
from src.data import schema as s


def _synthetic_matrix(n_rows=12, W=30):
    """schema.feature_names 규격의 합성 피처 행렬."""
    cols = s.feature_names(W)
    idx = pd.date_range("2020-01-01", periods=n_rows, freq="B")
    data = np.arange(n_rows * len(cols), dtype="float64").reshape(n_rows, len(cols))
    return pd.DataFrame(data, index=idx, columns=cols)


# ── Parquet 입출력 ──
def test_parquet_roundtrip(tmp_path):
    df = _synthetic_matrix()
    out = str(tmp_path / "fstore")
    fs.write_features(df, out, fold_id=1, split="train")
    loaded = fs.load_features(out, fold_id=1, split="train")
    # Parquet은 인덱스 freq 메타를 보존하지 않으므로 freq 비교 제외(값·날짜는 동일해야 함)
    pd.testing.assert_frame_equal(df, loaded, check_freq=False)


def test_partition_path_format(tmp_path):
    p = fs.partition_path(str(tmp_path), fold_id=2, split="test")
    assert p.parts[-3:] == ("fold=2", "split=test", "part.parquet")


def test_invalid_split_rejected(tmp_path):
    with pytest.raises(ValueError):
        fs.partition_path(str(tmp_path), fold_id=1, split="holdout")


def test_write_validates_columns(tmp_path):
    df = _synthetic_matrix()
    expected = s.feature_names(30)
    # 컬럼 하나를 바꿔 규격 위반 유도
    bad = df.rename(columns={df.columns[0]: "WRONG"})
    with pytest.raises(ValueError):
        fs.write_features(bad, str(tmp_path), 1, "train", expected_columns=expected)


def test_load_missing_partition_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        fs.load_features(str(tmp_path), fold_id=9, split="valid")


# ── SQLite 메타 ──
def test_meta_roundtrip(tmp_path):
    db = str(tmp_path / "meta.sqlite")
    fs.init_meta_db(db)

    cols = s.feature_names(30)
    fs.write_run(db, "run1", "2026-06-28T00:00:00", "abc123", 30, s.state_dim(30), list(s.ASSETS))
    fs.write_feature_columns(db, "run1", cols)
    fs.write_fold(db, "run1", {
        "fold_id": 1, "mode": "expanding",
        "train_start": "2010-01-01", "train_end": "2019-12-31",
        "valid_start": "2019-01-01", "valid_end": "2019-12-31",
        "test_start": "2020-01-01", "test_end": "2021-12-31",
        "embargo_days": 34,
    })

    read_cols = fs.read_feature_columns(db, "run1")
    assert read_cols == cols
    assert len(read_cols) == s.state_dim(30) == 187

    folds = fs.read_folds(db, "run1")
    assert len(folds) == 1
    assert folds[0]["test_start"] == "2020-01-01"
    assert folds[0]["embargo_days"] == 34


def test_init_idempotent(tmp_path):
    db = str(tmp_path / "meta.sqlite")
    fs.init_meta_db(db)
    fs.init_meta_db(db)  # 재호출해도 에러 없어야 함


def test_init_migrates_legacy_runs_table_without_combo_column(tmp_path):
    """콤보 시스템 이전(v0.19 이하) meta.sqlite도 init_meta_db 재호출로 combo 컬럼을 받아야 한다.

    CREATE TABLE IF NOT EXISTS는 기존 테이블을 안 건드리므로, 구버전 6컬럼 runs 테이블에
    write_run이 7번째 값(combo)을 넣으려다 'table runs has 6 columns but 7 values were
    supplied'로 깨지는 걸 막는 마이그레이션 회귀 테스트.
    """
    import sqlite3

    db = str(tmp_path / "meta.sqlite")
    with sqlite3.connect(db) as conn:
        conn.execute(
            """CREATE TABLE runs (
                run_id TEXT PRIMARY KEY, created_at TEXT, config_hash TEXT,
                window_w INTEGER, state_dim INTEGER, asset_order TEXT
            )"""
        )  # 구버전 6컬럼 스키마 재현

    fs.init_meta_db(db)  # 마이그레이션 수행돼야 함
    fs.write_run(db, "run1", "2026-08-09T00:00:00", "abc123", 30, 187, list(s.ASSETS), combo="full")


# ── build run_id provenance (이슈 #27) ──
def _write_run_with_fold(db, run_id, created_at, fold_id):
    fs.write_run(db, run_id, created_at, "abc123", 30, s.state_dim(30), list(s.ASSETS))
    fs.write_fold(db, run_id, {
        "fold_id": fold_id, "mode": "expanding",
        "train_start": "2010-01-01", "train_end": "2019-12-31",
        "valid_start": "2019-01-01", "valid_end": "2019-12-31",
        "test_start": "2020-01-01", "test_end": "2021-12-31",
        "embargo_days": 34,
    })


def test_latest_run_id_picks_newest_build(tmp_path):
    db = str(tmp_path / "meta.sqlite")
    fs.init_meta_db(db)
    # 같은 fold=1을 두 번 빌드 — 최신 created_at의 run이 디스크 파티션의 주인이다.
    _write_run_with_fold(db, "old-run", "2026-06-01T00:00:00", fold_id=1)
    _write_run_with_fold(db, "new-run", "2026-06-28T00:00:00", fold_id=1)

    assert fs.latest_run_id_for_fold(db, 1) == "new-run"


def test_latest_run_id_none_when_fold_absent(tmp_path):
    db = str(tmp_path / "meta.sqlite")
    fs.init_meta_db(db)
    _write_run_with_fold(db, "run1", "2026-06-28T00:00:00", fold_id=1)
    # 기록되지 않은 fold는 None.
    assert fs.latest_run_id_for_fold(db, 2) is None


def test_latest_run_id_none_when_db_missing(tmp_path):
    # 빌드 전(DB 파일 없음)이면 예외 없이 None.
    assert fs.latest_run_id_for_fold(str(tmp_path / "nope.sqlite"), 1) is None


def _hash_cfg(**over):
    """해시 대상 섹션을 모두 갖춘 최소 config(테스트용)."""
    cfg = {
        "assets": list(s.ASSETS),
        "window": 30,
        "features": {"params": {"rsi_length": 14, "macd": [12, 26, 9]}},
        "normalize": {"returns_scope": "per_asset", "feature_scope": "per_column", "eps": 1e-8},
        "split": {"anchor_start": "2010-01-01", "valid_days": 252, "embargo_days": 34},
        # 아래는 해시 비대상 — 바뀌어도 통계가 달라지지 않는다
        "transaction_cost": 0.001,
        "data": {"start": "2009-10-01", "end": "2025-12-31"},
    }
    cfg.update(over)
    return cfg


def test_config_hash_detects_change():
    assert fs.config_hash(_hash_cfg()) == fs.config_hash(_hash_cfg())  # 결정적
    assert fs.config_hash(_hash_cfg()) != fs.config_hash(_hash_cfg(window=20))


def test_config_hash_covers_stat_changing_sections():
    """통계 값을 바꾸는 설정은 모두 해시에 반영돼야 한다.

    자동 run 해석(latest_run_id_for_config)의 유일한 안전장치라, 여기 빠지면
    비호환 build를 조용히 집는다.
    """
    base = fs.config_hash(_hash_cfg())

    # 지표 파라미터 — 지표 값 자체가 달라짐
    assert base != fs.config_hash(
        _hash_cfg(features={"params": {"rsi_length": 21, "macd": [12, 26, 9]}})
    )
    # 정규화 스코프 — μ/σ 정의가 달라짐
    assert base != fs.config_hash(
        _hash_cfg(normalize={"returns_scope": "per_column",
                             "feature_scope": "per_column", "eps": 1e-8})
    )
    # fold 경계 — train 구간이 달라져 통계가 달라짐
    assert base != fs.config_hash(
        _hash_cfg(split={"anchor_start": "2012-01-01", "valid_days": 252, "embargo_days": 34})
    )
    # 자산 구성
    assert base != fs.config_hash(_hash_cfg(assets=["SPY", "EWY", "TLT", "GLD", "IEF"]))


def test_config_hash_ignores_non_stat_settings():
    """통계와 무관한 설정은 해시를 흔들지 않는다(불필요한 재빌드 방지)."""
    base = fs.config_hash(_hash_cfg())
    # 수집 기간: fold 경계는 anchor·test_blocks가 정하고 신규 데이터는 뒤에만 붙는다
    assert base == fs.config_hash(_hash_cfg(data={"start": "2009-10-01", "end": "2026-12-31"}))
    # 거래비용: 환경·보상의 값이지 데이터 산출물과 무관
    assert base == fs.config_hash(_hash_cfg(transaction_cost=0.002))
    # 소비 측 설정
    assert base == fs.config_hash(_hash_cfg(inference={"model_path": "x.zip"}))


# ── 피처 콤보(M0~M3) — resolve_combo로 반영된 features가 해시에 실제로 반영되는지 ──


def test_config_hash_differs_across_resolved_combos():
    """resolve_combo(cfg, combo)로 반영된 cfg를 해시하면 콤보마다 다른 해시가 나와야 한다.

    build.py는 원본 cfg가 아니라 resolved_cfg를 config_hash에 넘긴다 — 그래야
    latest_run_id_for_config가 다른 콤보의 빌드를 잘못 재사용하지 않는다.
    """
    base = _hash_cfg(feature_combos={
        "full": {"asset": list(s.ASSET_FEATURES), "market": list(s.MARKET_FEATURES)},
        "M0": {"asset": [], "market": ["Equity_Bond_Ratio"]},
        "M1": {"asset": ["MACD_Hist", "Rolling_Vol_20"], "market": ["Equity_Bond_Ratio"]},
    }, active_combo="full")

    from src import config_loader as cl

    hashes = {name: fs.config_hash(cl.resolve_combo(base, name)) for name in ("full", "M0", "M1")}
    assert len(set(hashes.values())) == 3  # 셋 다 달라야 함


def test_config_hash_reacts_to_in_place_combo_edit():
    """콤보 이름은 안 바뀌었어도 그 콤보의 실제 피처 리스트가 바뀌면 해시도 바뀌어야 한다."""
    from src import config_loader as cl

    cfg_a = _hash_cfg(
        feature_combos={"M1": {"asset": ["MACD_Hist"], "market": ["Equity_Bond_Ratio"]}},
        active_combo="M1",
    )
    cfg_b = _hash_cfg(
        feature_combos={"M1": {"asset": ["MACD_Hist", "RSI_28"], "market": ["Equity_Bond_Ratio"]}},
        active_combo="M1",
    )
    assert fs.config_hash(cl.resolve_combo(cfg_a)) != fs.config_hash(cl.resolve_combo(cfg_b))
