"""normalize.py 테스트 — 룩어헤드(train만 fit)·스코프·std=0 가드 (pandas·numpy 필요)."""

import numpy as np
import pytest

pd = pytest.importorskip("pandas")

from src.data import schema
from src.data.normalize import ZScoreScaler

W = 30


def _state(n=300, seed=0):
    idx = pd.bdate_range("2015-01-01", periods=n)
    rng = np.random.default_rng(seed)
    names = schema.feature_names(W)
    df = pd.DataFrame(rng.normal(0, 1, (n, len(names))), index=idx, columns=names)
    # prev_weight은 0
    for a in schema.ASSETS:
        df[f"prevw_{a}"] = 0.0
    return df


def test_train_transformed_is_standardized():
    df = _state()
    scaler = ZScoreScaler().fit(df, W)
    out = scaler.transform(df)
    c = "feat_SPY_RSI_14"
    assert abs(out[c].mean()) < 1e-9
    assert abs(out[c].std(ddof=0) - 1.0) < 1e-6


def test_per_asset_returns_share_stats():
    """수익률 30칸은 자산당 단일 μ/σ(per_asset)."""
    scaler = ZScoreScaler(returns_scope="per_asset").fit(_state(), W)
    spy = {scaler.stats_[f"ret_SPY_lag{k}"] for k in range(W)}
    assert len(spy) == 1  # 모두 동일


def test_per_column_features_distinct():
    scaler = ZScoreScaler().fit(_state(), W)
    a = scaler.stats_["feat_SPY_RSI_14"]
    b = scaler.stats_["feat_EWY_RSI_14"]
    assert a != b


def test_fit_only_on_train_valid_outlier_ignored():
    """valid에 극단 outlier가 있어도 통계는 train fit 이후 불변이어야 한다 (룩어헤드 차단)."""
    train = _state(seed=1)
    scaler = ZScoreScaler().fit(train, W)
    stats_before = dict(scaler.stats_)

    valid = _state(n=50, seed=2)
    valid.iloc[0, valid.columns.get_loc("feat_SPY_RSI_14")] = 1e6  # 극단 오염
    scaler.transform(valid)  # transform은 재fit하지 않음

    assert scaler.stats_ == stats_before  # 통계 불변


def test_constant_column_std_guard():
    """상수 컬럼(SHV류)은 σ=1 가드로 inf/NaN 없이 0 출력."""
    df = _state()
    df["feat_SHV_Rolling_Vol_20"] = 5.0  # 상수
    scaler = ZScoreScaler(eps=1e-8).fit(df, W)
    out = scaler.transform(df)
    col = out["feat_SHV_Rolling_Vol_20"]
    assert np.isfinite(col.to_numpy()).all()
    assert (col == 0.0).all()  # (상수-평균)/1 = 0


def test_prev_weight_untouched():
    df = _state()
    out = ZScoreScaler().fit(df, W).transform(df)
    for a in schema.ASSETS:
        assert (out[f"prevw_{a}"] == 0.0).all()


def test_stats_roundtrip():
    scaler = ZScoreScaler().fit(_state(), W)
    rows = scaler.to_stats_rows()
    restored = ZScoreScaler.from_stats_rows(rows)
    assert restored.stats_ == scaler.stats_
