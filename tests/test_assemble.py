"""assemble.py 테스트 — 187 차원·인덱스맵·시장 단일값·prev_weight (pandas 필요)."""

import numpy as np
import pytest

pd = pytest.importorskip("pandas")

from src.data import schema
from src.data.assemble import assemble_state_matrix

ASSETS = list(schema.ASSETS)
W = 30


def _cfg():
    return {"assets": ASSETS, "window": W}


def _inputs(n=120, seed=1):
    idx = pd.bdate_range("2015-01-01", periods=n)
    rng = np.random.default_rng(seed)
    logret = pd.DataFrame(rng.normal(0, 0.01, (n, 5)), index=idx, columns=ASSETS)

    # 지표: 앞 40행은 warm-up으로 없다고 가정
    feat_idx = idx[40:]
    asset_cols = [f"feat_{a}_{f}" for a in ASSETS for f in schema.ASSET_FEATURES]
    asset_feat = pd.DataFrame(
        rng.normal(0, 1, (len(feat_idx), len(asset_cols))), index=feat_idx, columns=asset_cols
    )
    market = pd.DataFrame(
        rng.normal(0, 1, (len(feat_idx), 2)),
        index=feat_idx,
        columns=["mkt_Equity_Bond_Ratio", "mkt_Gold_Vol_Ratio"],
    )
    return logret, asset_feat, market


def test_shape_is_187_and_columns_match():
    logret, asset_feat, market = _inputs()
    state = assemble_state_matrix(logret, asset_feat, market, _cfg())
    assert state.shape[1] == schema.state_dim(W) == 187
    assert list(state.columns) == schema.feature_names(W)
    assert not state.isna().any().any()


def test_prev_weight_block_is_zero():
    logret, asset_feat, market = _inputs()
    state = assemble_state_matrix(logret, asset_feat, market, _cfg())
    prevw = state[[f"prevw_{a}" for a in ASSETS]]
    assert (prevw == 0.0).all().all()


def test_returns_lag0_equals_same_day_return():
    logret, asset_feat, market = _inputs()
    state = assemble_state_matrix(logret, asset_feat, market, _cfg())
    t = state.index[-1]
    for a in ASSETS:
        assert np.isclose(state.loc[t, f"ret_{a}_lag0"], logret.loc[t, a])


def test_returns_window_no_future_leak():
    """lag_k는 k일 전 값 — 미래 수익률이 윈도우에 섞이지 않아야 한다."""
    logret, asset_feat, market = _inputs()
    state = assemble_state_matrix(logret, asset_feat, market, _cfg())
    t_pos = logret.index.get_loc(state.index[-1])
    t = logret.index[t_pos]
    # lag1 == 전날 수익률
    assert np.isclose(state.loc[t, "ret_SPY_lag1"], logret["SPY"].iloc[t_pos - 1])


def test_market_single_value_not_replicated():
    """시장지표 슬라이스는 정확히 2칸이고 자산 지표와 독립이어야 한다."""
    logret, asset_feat, market = _inputs()
    state = assemble_state_matrix(logret, asset_feat, market, _cfg())
    sl = schema.market_feature_slice(W)
    assert sl.stop - sl.start == 2
    mkt_cols = [c for c in state.columns if c.startswith("mkt_")]
    assert len(mkt_cols) == 2
