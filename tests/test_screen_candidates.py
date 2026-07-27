"""screen_candidates 테스트 — 후보 계산의 인과성·IC 산출을 합성 데이터로 검증.

(pandas·pandas_ta 필요) 실데이터·네트워크 없이 tmp raw로 스크리닝을 돌린다.
"""

import numpy as np
import pytest

pd = pytest.importorskip("pandas")
pytest.importorskip("pandas_ta")

from src.data import schema
from src.data import screen_candidates as sc

ASSETS = list(schema.ASSETS)


def _cfg(tmp_path, n=900):
    raw = tmp_path / "raw"
    raw.mkdir()
    idx = pd.bdate_range("2015-01-01", periods=n)
    rng = np.random.default_rng(0)
    prices = pd.DataFrame(
        100 * np.exp(np.cumsum(rng.normal(0.0002, 0.01, (n, 5)), axis=0)),
        index=idx, columns=ASSETS,
    )
    prices.index.name = "date"
    prices.to_parquet(raw / "prices_raw.parquet")
    end = idx[-1]
    mid = idx[n // 2]
    return {
        "data": {"raw_dir": str(raw)},
        "split": {"test_blocks": [
            [str(idx[n // 3].date()), str(mid.date())],
            [str(mid.date()), str(end.date())],
            [str(idx[2 * n // 3].date()), str(end.date())],
        ]},
    }


def test_spearman_no_scipy():
    x = pd.Series([1.0, 2, 3, 4, 5])
    assert sc._spearman(x, x * 2) == pytest.approx(1.0, abs=1e-9)


def test_screen_returns_expected_shape(tmp_path):
    df = sc.screen(_cfg(tmp_path))
    # 후보 + 대조군이 모두 표에 있고 필수 컬럼을 가진다
    assert {"RS_SPY_TLT", "RSI_SPY_28", "Regime_Drawdown_SPY", "baseline_RSI_14"} <= set(df.index)
    for col in ("mean_abs_IC", "IC_f1", "IC_f2", "IC_f3", "sign_stable"):
        assert col in df.columns
    assert not df["mean_abs_IC"].isna().any()


def test_candidates_are_causal(tmp_path):
    """후보는 과거만 참조해야 한다 — 미래 가격을 바꿔도 과거 시점 값은 불변(룩어헤드 차단)."""
    cfg = _cfg(tmp_path)
    from src.data.collect import load_raw
    from src.data.returns import log_returns

    prices = load_raw(cfg["data"]["raw_dir"])
    logret = log_returns(prices)
    close = prices.loc[logret.index]
    cand = sc._candidates(close, logret)

    prices2 = prices.copy()
    prices2.iloc[-20:] *= 1.5  # 미래 20일 오염
    logret2 = log_returns(prices2)
    close2 = prices2.loc[logret2.index]
    cand2 = sc._candidates(close2, logret2)

    cutoff = close.index[-60]  # 오염 이전
    for name in ("RS_SPY_TLT", "Regime_Drawdown_SPY", "RSI_SPY_28"):
        s1 = cand[name][0].loc[:cutoff].dropna()
        s2 = cand2[name][0].loc[:cutoff].dropna()
        common = s1.index.intersection(s2.index)
        np.testing.assert_allclose(s1.loc[common], s2.loc[common], rtol=1e-9, atol=1e-9)
