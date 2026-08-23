"""screen_splits 테스트 — 분할(split) 후보 비교의 순수 계산 부분 (실데이터 없이 합성 state_all로).

`_build_state_all`(실데이터 파이프라인 호출)은 제외하고, `evaluate_candidate`/`_fold_drift`/
`summarize`/`CANDIDATE_SPLITS` 정의가 옳은지만 검증한다.
"""

from __future__ import annotations

import numpy as np
import pytest

pd = pytest.importorskip("pandas")

from src.data import schema
from src.data import screen_splits as ss

ASSETS = list(schema.ASSETS)
W = 30


def _base_cfg(rng_seed=0):
    return {
        "assets": ASSETS,
        "window": W,
        "transaction_cost": 0.001,
        "normalize": {"returns_scope": "per_asset", "feature_scope": "per_column", "eps": 1e-8},
    }


def _synthetic_state_all(n_rows: int, start="2015-01-01", seed=0) -> pd.DataFrame:
    """schema.feature_names(W) 규격의 합성 state_all(비정규화, 실수 값)."""
    cols = schema.feature_names(W)
    idx = pd.bdate_range(start, periods=n_rows)
    rng = np.random.default_rng(seed)
    data = rng.normal(size=(n_rows, len(cols)))
    return pd.DataFrame(data, index=idx, columns=cols)


# ── _fold_drift ──────────────────────────────────────────────────────

def test_fold_drift_zero_for_identical_train_test_distribution():
    """train과 test가 같은 분포(사실상 같은 데이터)면 드리프트가 0에 가까워야 한다."""
    cfg = _base_cfg()
    train = _synthetic_state_all(500, seed=1)
    test = _synthetic_state_all(100, start="2017-01-01", seed=1)  # 같은 seed → 같은 분포
    d = ss._fold_drift(train, test, cfg, W)
    assert abs(d["mean_abs_test_mean"]) < 0.5
    assert d["max_abs_z"] < 10  # 같은 정규분포에서 뽑았으니 극단값이 작아야 함


def test_fold_drift_large_for_shifted_test_distribution():
    """test가 train과 완전히 다른 분포(평균 이동)면 드리프트가 크게 나와야 한다."""
    cfg = _base_cfg()
    train = _synthetic_state_all(500, seed=2)
    test = _synthetic_state_all(50, start="2017-01-01", seed=2)
    shifted_cols = [c for c in test.columns if c.startswith(("feat_", "mkt_"))]
    test = test.copy()
    test[shifted_cols] = test[shifted_cols] + 10.0  # 평균을 10 표준편차만큼 이동
    d = ss._fold_drift(train, test, cfg, W)
    assert d["mean_abs_test_mean"] > 5.0
    assert d["max_abs_z"] > 5.0


def test_fold_drift_handles_empty_test():
    cfg = _base_cfg()
    train = _synthetic_state_all(500, seed=3)
    test = train.iloc[0:0]
    d = ss._fold_drift(train, test, cfg, W)
    assert np.isnan(d["max_abs_z"])


# ── evaluate_candidate ───────────────────────────────────────────────

def test_evaluate_candidate_matches_fold_count():
    state_all = _synthetic_state_all(2000, start="2015-01-01", seed=4)
    base_cfg = _base_cfg()
    split_cfg = {
        "mode": "expanding",
        "anchor_start": "2015-01-01",
        "test_blocks": [["2019-01-01", "2019-06-30"]],
        "valid_days": 60,
        "embargo_days": 5,
    }
    df = ss.evaluate_candidate(state_all, base_cfg, split_cfg)
    assert len(df) == 1
    assert set(df.columns) >= {
        "fold_id", "train_start", "train_end", "test_start", "test_end",
        "n_train", "n_valid", "n_test", "mean_abs_test_mean", "mean_abs_test_std_m1", "max_abs_z",
    }
    assert df.iloc[0]["n_valid"] == 60


def test_evaluate_candidate_rolling_has_constant_train_size():
    state_all = _synthetic_state_all(3000, start="2012-01-01", seed=5)
    base_cfg = _base_cfg()
    split_cfg = {
        "mode": "rolling",
        "rolling_train_days": 500,
        "test_blocks": [
            ["2018-01-01", "2018-06-30"],
            ["2019-01-01", "2019-06-30"],
        ],
        "valid_days": 60,
        "embargo_days": 5,
    }
    df = ss.evaluate_candidate(state_all, base_cfg, split_cfg)
    assert len(df) == 2
    assert df["n_train"].nunique() == 1  # rolling → fold마다 동일 크기


# ── summarize ────────────────────────────────────────────────────────

def test_summarize_aggregates_across_candidates():
    df1 = pd.DataFrame({
        "n_train": [100, 200], "max_abs_z": [5.0, 7.0],
        "mean_abs_test_mean": [0.1, 0.2], "mean_abs_test_std_m1": [0.0, 0.0],
    })
    df2 = pd.DataFrame({
        "n_train": [150, 150], "max_abs_z": [3.0, 3.0],
        "mean_abs_test_mean": [0.05, 0.05], "mean_abs_test_std_m1": [0.0, 0.0],
    })
    summary = ss.summarize({"cand1": df1, "cand2": df2})
    assert list(summary.index) == ["cand1", "cand2"]
    assert summary.loc["cand1", "n_folds"] == 2
    assert summary.loc["cand1", "train_size_range"] == 100
    assert summary.loc["cand2", "train_size_range"] == 0
    assert summary.loc["cand2", "mean_max_abs_z"] == pytest.approx(3.0)


# ── CANDIDATE_SPLITS 정의 자체의 유효성 ──────────────────────────────

def test_candidate_splits_are_valid_configs():
    """정의된 후보들이 splits.make_folds가 받아들이는 형태인지(모드·필수키) 확인."""
    for name, split_cfg in ss.CANDIDATE_SPLITS.items():
        assert split_cfg["mode"] in ("expanding", "rolling"), name
        assert "test_blocks" in split_cfg and len(split_cfg["test_blocks"]) > 0, name
        assert "valid_days" in split_cfg and "embargo_days" in split_cfg, name
        if split_cfg["mode"] == "expanding":
            assert "anchor_start" in split_cfg, name
        else:
            assert "rolling_train_days" in split_cfg, name


def test_candidate_splits_actually_produce_folds_on_real_index():
    """합성 인덱스로도 후보 전부가 make_folds를 통과해야 한다(설정 오타 회귀 방지)."""
    from src.data.splits import make_folds

    idx = pd.bdate_range("2009-10-01", "2025-12-31")
    base_cfg = _base_cfg()
    for name, split_cfg in ss.CANDIDATE_SPLITS.items():
        cfg = {**base_cfg, "split": split_cfg}
        folds = make_folds(idx, cfg)
        assert len(folds) == len(split_cfg["test_blocks"]), name
