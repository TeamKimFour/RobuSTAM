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


def test_config_hash_detects_change():
    base = {"assets": list(s.ASSETS), "window": 30}
    same = {"assets": list(s.ASSETS), "window": 30}
    changed = {"assets": list(s.ASSETS), "window": 20}
    assert fs.config_hash(base) == fs.config_hash(same)
    assert fs.config_hash(base) != fs.config_hash(changed)
