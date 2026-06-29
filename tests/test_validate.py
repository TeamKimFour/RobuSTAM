"""validate.py 테스트 — 합성 정상/이상 케이스로 품질 검사 검출 확인 (pandas 필요)."""

import pytest

pd = pytest.importorskip("pandas")
np = pytest.importorskip("numpy")

from src.data import validate as v

ASSETS = ["SPY", "EWY", "TLT", "GLD", "SHV"]


def _good_prices(n=300):
    idx = pd.bdate_range("2015-01-01", periods=n)
    rng = np.random.default_rng(0)
    # 완만한 랜덤워크(양수 가격)
    steps = rng.normal(0, 0.01, (n, 5))
    prices = 100 * np.exp(np.cumsum(steps, axis=0))
    return pd.DataFrame(prices, index=idx, columns=ASSETS)


def test_good_data_passes():
    assert v.validate_prices(_good_prices(), ASSETS) == []


def test_detects_nan():
    df = _good_prices()
    df.iloc[10, 2] = np.nan
    issues = v.validate_prices(df, ASSETS)
    assert any("결측" in i for i in issues)


def test_detects_nonpositive():
    df = _good_prices()
    df.iloc[5, 0] = -1.0
    issues = v.validate_prices(df, ASSETS)
    assert any("0 이하" in i for i in issues)


def test_detects_unsorted_index():
    df = _good_prices()
    df = df.iloc[::-1]  # 역순
    issues = v.validate_prices(df, ASSETS)
    assert any("정렬" in i for i in issues)


def test_detects_duplicate_index():
    df = _good_prices()
    df = pd.concat([df, df.iloc[[0]]])
    issues = v.validate_prices(df, ASSETS)
    assert any("중복" in i for i in issues)


def test_detects_wrong_asset_order():
    df = _good_prices()[["EWY", "SPY", "TLT", "GLD", "SHV"]]
    issues = v.validate_prices(df, ASSETS)
    assert any("자산" in i for i in issues)


def test_detects_extreme_jump():
    df = _good_prices()
    df.iloc[100, 0] = df.iloc[99, 0] * 3.0  # +200% 점프
    issues = v.validate_prices(df, ASSETS)
    assert any("극단" in i for i in issues)


def test_min_rows():
    df = _good_prices(n=10)
    issues = v.validate_prices(df, ASSETS, min_rows=252)
    assert any("행 수" in i for i in issues)


def test_summarize_keys():
    s = v.summarize_prices(_good_prices())
    for k in ("shape", "period", "correlation", "ret_annual_vol", "extreme_moves"):
        assert k in s
    assert s["nan_count"] == 0
