"""analyze_features 테스트 — IC·분포 드리프트 계산의 정확성 (pandas·pyarrow 필요).

합성 Feature Store를 만들어, 알려진 관계를 심고 분석기가 그대로 뽑아내는지 검증한다.
scipy 없이 도는지도 함께 확인한다(_spearman = 순위 피어슨).
"""

import numpy as np
import pytest

pd = pytest.importorskip("pandas")
pytest.importorskip("pyarrow")

from src.data import analyze_features as af
from src.data import feature_store as fs
from src.data import schema

ASSETS = list(schema.ASSETS)
W = 30


def _write_partition(out_dir, fold, split, feats: pd.DataFrame, tgts: pd.DataFrame):
    fs.write_features(feats, str(out_dir), fold, split)
    fs.write_targets(tgts, str(out_dir), fold, split)


def _frame(index, cols, filler):
    return pd.DataFrame({c: filler(c, i) for i, c in enumerate(cols)}, index=index)


def test_spearman_no_scipy_matches_known_monotonic():
    """단조 증가 관계면 스피어만 = +1 (scipy 없이도)."""
    x = pd.Series([1.0, 2, 3, 4, 5])
    y = pd.Series([10.0, 20, 33, 44, 500])  # 비선형이지만 단조 증가
    assert af._spearman(x, y) == pytest.approx(1.0, abs=1e-9)
    assert af._spearman(x, -y) == pytest.approx(-1.0, abs=1e-9)


def test_feature_ic_recovers_planted_signal(tmp_path):
    """feat_SPY_RSI_14를 SPY 익일수익률과 완전 단조로 심으면 IC≈+1로 잡힌다."""
    n = 40
    idx = pd.bdate_range("2020-01-01", periods=n)
    cols = schema.feature_names(W)
    feats = _frame(idx, cols, lambda c, i: np.random.default_rng(i).normal(size=n))
    tgt_idx = idx[:-1]  # targets는 마지막 행 없음
    tgts = pd.DataFrame(
        {f"fwd_ret_{a}": np.random.default_rng(100 + j).normal(size=n - 1)
         for j, a in enumerate(ASSETS)},
        index=tgt_idx,
    )
    # SPY RSI를 SPY 익일수익률과 단조 일치하게 심는다(공통 인덱스 기준)
    feats.loc[tgt_idx, "feat_SPY_RSI_14"] = tgts["fwd_ret_SPY"].to_numpy() * 3.0

    _write_partition(tmp_path, 1, "test", feats, tgts)
    df, _ = af.feature_ic(str(tmp_path), 1, "test")
    assert df.loc["RSI_14", "SPY"] == pytest.approx(1.0, abs=1e-6)


def test_distribution_drift_detects_shift(tmp_path):
    """train은 표준정규(≈0,1), test는 평균 이동 → test_mean이 그 이동을 잡는다."""
    cols = schema.feature_names(W)
    tr_idx = pd.bdate_range("2015-01-01", periods=50)
    te_idx = pd.bdate_range("2020-01-01", periods=20)
    train = _frame(tr_idx, cols, lambda c, i: np.random.default_rng(i).normal(0, 1, len(tr_idx)))
    test = _frame(te_idx, cols, lambda c, i: np.full(len(te_idx), 5.0))  # 5σ 이동

    fs.write_features(train, str(tmp_path), 2, "train")
    fs.write_features(test, str(tmp_path), 2, "test")
    d = af.distribution_drift(str(tmp_path), 2)
    # feat_/mkt_ 컬럼만 대상 — test 평균이 5 근방
    assert d["test_mean"].abs().min() > 4.0
