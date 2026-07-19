"""features.py 테스트 — 지표 인과성·warm-up·시장 단일값 (pandas·pandas_ta 필요)."""

import numpy as np
import pytest

pd = pytest.importorskip("pandas")
pytest.importorskip("pandas_ta")

from src.data import features as F
from src.data import schema

ASSETS = list(schema.ASSETS)


def _synthetic(n=400, seed=0):
    """합성 가격·로그수익률 (5자산)."""
    idx = pd.bdate_range("2015-01-01", periods=n)
    rng = np.random.default_rng(seed)
    steps = rng.normal(0.0003, 0.01, (n, 5))
    prices = pd.DataFrame(100 * np.exp(np.cumsum(steps, axis=0)), index=idx, columns=ASSETS)
    logret = np.log(prices / prices.shift(1)).iloc[1:]
    close = prices.loc[logret.index]
    return close, logret


def _cfg():
    return {
        "features": {
            "params": {
                "sma_fast": 5, "sma_slow": 20, "rsi_length": 14, "macd": [12, 26, 9],
                "vol_window": 20, "bbands": [20, 2], "roc_length": 10,
                "annualize_vol": False, "ebr_ma_window": 20, "gvr_long_window": 60,
            }
        }
    }


def test_output_shape_and_columns():
    close, logret = _synthetic()
    asset_wide, market = F.compute_features(close, logret, _cfg())
    assert asset_wide.shape[1] == 30  # 6지표 × 5자산
    assert list(market.columns) == ["mkt_Equity_Bond_Ratio", "mkt_Gold_Vol_Ratio"]
    # 컬럼명이 schema.feature_names 규격과 일치
    assert "feat_SPY_MA_Cross_5_20" in asset_wide.columns
    assert not asset_wide.isna().any().any()
    assert not market.isna().any().any()


def test_market_is_single_series_not_per_asset():
    """시장지표는 2컬럼 단일값 — 자산별 복제 금지(1,355 오류 회귀)."""
    close, logret = _synthetic()
    _, market = F.compute_features(close, logret, _cfg())
    assert market.shape[1] == 2


def test_warmup_dropped():
    """warm-up NaN은 메우지 않고 drop → 앞부분 행이 잘려야 한다."""
    close, logret = _synthetic()
    asset_wide, _ = F.compute_features(close, logret, _cfg())
    # 최장 lookback(gvr_long_window=60 등)만큼 앞이 잘림
    assert asset_wide.index.min() > logret.index.min()


def test_feature_causality():
    """미래 가격을 바꿔도 과거 시점의 지표는 불변이어야 한다 (룩어헤드 차단)."""
    close, logret = _synthetic()
    asset_wide, market = F.compute_features(close, logret, _cfg())

    # 미래(마지막 30행) 가격을 오염
    close2 = close.copy()
    logret2 = logret.copy()
    close2.iloc[-30:] *= 1.5
    logret2.iloc[-30:] = 0.02
    asset_wide2, market2 = F.compute_features(close2, logret2, _cfg())

    # 공통 인덱스의 앞부분(오염 이전)은 동일해야 함
    common = asset_wide.index.intersection(asset_wide2.index)
    cutoff = common[-60]  # 오염 구간보다 충분히 이전
    past = common[common <= cutoff]
    pd.testing.assert_frame_equal(asset_wide.loc[past], asset_wide2.loc[past])
    pd.testing.assert_frame_equal(market.loc[past], market2.loc[past])


def test_short_input_raises_clear_error():
    """데이터 부족 시 pandas-ta None이 흘러가 TypeError로 터지지 않고, 원인이 보이는 에러가 난다.

    (daily.yml 장애: 수집 0행 → `TypeError: NoneType - NoneType`으로 원인에서 멀리 떨어져 실패)
    """
    idx = pd.bdate_range("2020-01-01", periods=3)
    close = pd.Series([100.0, 101.0, 102.0], index=idx, name="SPY")
    logret = pd.Series([0.0, 0.01, 0.01], index=idx, name="SPY")
    with pytest.raises(ValueError, match="입력이"):
        F._asset_feature("MA_Cross_5_20", close, logret, _cfg()["features"]["params"])


def test_rsi_high_on_uptrend():
    """단조 상승 가격이면 RSI가 높아야 한다 (sanity)."""
    idx = pd.bdate_range("2015-01-01", periods=200)
    up = pd.Series(np.linspace(100, 200, 200), index=idx)
    close = pd.DataFrame({a: up for a in ASSETS})
    logret = np.log(close / close.shift(1)).iloc[1:]
    close = close.loc[logret.index]
    asset_wide, _ = F.compute_features(close, logret, _cfg())
    assert asset_wide["feat_SPY_RSI_14"].iloc[-1] > 70
