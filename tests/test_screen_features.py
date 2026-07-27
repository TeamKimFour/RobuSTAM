"""screen_features 테스트 — 대리모델이 심은 신호를 잡는지, 컬럼 선택이 맞는지.

(pandas·pyarrow·scikit-learn 필요) 합성 Feature Store에 알려진 선형 신호를 심고 대리모델
rank-IC가 높게 나오는지 확인한다. scikit-learn은 dev 전용이라 importorskip 가드.
"""

import numpy as np
import pytest

pd = pytest.importorskip("pandas")
pytest.importorskip("pyarrow")
pytest.importorskip("sklearn")

from src.data import feature_store as fs
from src.data import schema
from src.data import screen_features as sfz

ASSETS = list(schema.ASSETS)
W = 30


def _write_fold(out_dir, fold, plant_signal: bool, n_train=80, n_test=40, seed=0):
    """fold의 train/test 파티션을 합성으로 쓴다. plant_signal이면 MACD_Hist=3*익일수익률."""
    cols = schema.feature_names(W)
    rng = np.random.default_rng(seed + fold)
    for split, n in (("train", n_train), ("test", n_test)):
        idx = pd.bdate_range("2015-01-01", periods=n) if split == "train" \
            else pd.bdate_range("2019-01-01", periods=n)
        feats = pd.DataFrame(rng.normal(size=(n, len(cols))), index=idx, columns=cols)
        tgt_idx = idx[:-1]  # targets는 마지막 행 없음
        tgts = pd.DataFrame(
            {f"fwd_ret_{a}": rng.normal(0, 0.01, len(tgt_idx)) for a in ASSETS},
            index=tgt_idx,
        )
        if plant_signal:
            for a in ASSETS:  # 공통 인덱스에 선형 신호 주입
                feats.loc[tgt_idx, f"feat_{a}_MACD_Hist"] = tgts[f"fwd_ret_{a}"].to_numpy() * 3.0
        fs.write_features(feats, str(out_dir), fold, split)
        fs.write_targets(tgts, str(out_dir), fold, split)


def test_xy_selects_correct_columns(tmp_path):
    _write_fold(tmp_path, 1, plant_signal=False)
    x, y = sfz._xy(str(tmp_path), 1, "train", ["MACD_Hist"], ["Equity_Bond_Ratio"])
    # 자산 5종 × (MACD 1 + 시장 1) → 컬럼 2, 행 = 5자산 × 공통일수
    assert x.shape[1] == 2
    assert len(y) == x.shape[0]


def test_surrogate_recovers_planted_signal(tmp_path):
    """MACD_Hist에 선형 신호를 심으면 대리모델 rank-IC가 높아야 한다."""
    for f in sfz.FOLDS:
        _write_fold(tmp_path, f, plant_signal=True)
    score = sfz.surrogate_score(str(tmp_path), ["MACD_Hist"], [])
    assert score["ridge_ic"] > 0.8  # 선형 신호 → Ridge가 강하게 잡음


def test_surrogate_near_zero_on_noise(tmp_path):
    """신호 없는(무작위) 피처면 rank-IC가 0 근방이어야 한다."""
    for f in sfz.FOLDS:
        _write_fold(tmp_path, f, plant_signal=False)
    score = sfz.surrogate_score(str(tmp_path), ["RSI_14"], [])
    assert abs(score["ridge_ic"]) < 0.3  # 무신호


def test_compare_featuresets_returns_all_rows(tmp_path):
    for f in sfz.FOLDS:
        _write_fold(tmp_path, f, plant_signal=False)
    df = sfz.compare_featuresets(str(tmp_path))
    assert set(df.index) == set(sfz.FEATURESETS)
    assert list(df.columns) == ["ridge_ic", "ridge_r2", "gbm_ic", "gbm_r2"]
