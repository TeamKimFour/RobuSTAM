"""collect.py 정렬 로직 테스트 — 네트워크 없이 합성 데이터로 검증.

실제 yfinance 다운로드(fetch_prices)는 네트워크 의존이라 여기서 테스트하지 않는다.
순수 정렬 로직 _align만 검증한다(pandas 필요).
"""

import pytest

pd = pytest.importorskip("pandas")
np = pytest.importorskip("numpy")

from src.data import collect as c


def _close_with_gap():
    """B자산에 하루 결측이 있는 종가 표."""
    idx = pd.date_range("2020-01-01", periods=4, freq="D")
    return pd.DataFrame(
        {
            "SPY": [100, 101, 102, 103],
            "EWY": [50, 51, 52, 53],
            "TLT": [90, 91, 92, 93],
            "GLD": [180, 181, 182, 183],
            "SHV": [110, np.nan, 110, 110],  # 둘째 날 결측
        },
        index=idx,
    )


ASSETS = ["SPY", "EWY", "TLT", "GLD", "SHV"]


def test_align_drops_incomplete_days_no_ffill():
    """결측일은 drop되고, ffill로 채워지지 않아야 한다."""
    aligned = c._align(_close_with_gap(), ASSETS)
    # 4일 중 결측 있던 1일이 빠져 3일
    assert len(aligned) == 3
    # 결측치 없음
    assert not aligned.isna().any().any()
    # ffill 안 됨: 둘째 날이 통째로 사라졌고, SHV가 가짜로 채워지지 않음
    assert pd.Timestamp("2020-01-02") not in aligned.index


def test_align_preserves_asset_order():
    """컬럼이 자산 고정 순서로 정렬되어야 한다."""
    shuffled = _close_with_gap()[["GLD", "SHV", "SPY", "TLT", "EWY"]]
    aligned = c._align(shuffled, ASSETS)
    assert list(aligned.columns) == ASSETS


def test_align_raises_on_missing_asset():
    incomplete = _close_with_gap().drop(columns=["TLT"])
    with pytest.raises(ValueError):
        c._align(incomplete, ASSETS)
