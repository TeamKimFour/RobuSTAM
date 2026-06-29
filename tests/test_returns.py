"""returns.py 테스트 — pandas 필요(importorskip 가드)."""

import math

import pytest

pd = pytest.importorskip("pandas")
np = pytest.importorskip("numpy")

from src.data import returns as r


def _prices():
    # 2자산, 단순한 가격열
    idx = pd.date_range("2020-01-01", periods=5, freq="D")
    return pd.DataFrame(
        {"A": [100, 110, 121, 121, 100], "B": [50, 50, 55, 60, 60]}, index=idx
    )


def test_log_returns_values_and_first_row_dropped():
    rets = r.log_returns(_prices())
    # 첫 행(NaN) 제거 → 5행 → 4행
    assert len(rets) == 4
    # ln(110/100) ≈ 0.0953
    assert math.isclose(rets["A"].iloc[0], math.log(110 / 100), rel_tol=1e-9)
    # 무한대·NaN 없음
    assert not rets.isna().any().any()
    assert np.isfinite(rets.to_numpy()).all()


def test_return_window_shape_and_bounds():
    rets = r.log_returns(_prices())  # 4행, 2자산
    W = 2
    win = r.return_window(rets, t=3, W=W)
    assert win.shape == (W, 2)
    # 마지막 행은 위치 3의 수익률과 일치
    assert np.allclose(win[-1], rets.iloc[3].to_numpy())


def test_return_window_excludes_future():
    """t+1 이후 값을 바꿔도 위치 t의 윈도우는 불변이어야 한다 (룩어헤드 차단)."""
    rets = r.log_returns(_prices())
    W = 2
    before = r.return_window(rets, t=2, W=W).copy()
    # 미래(위치 3)를 오염시킴
    rets.iloc[3] = 999.0
    after = r.return_window(rets, t=2, W=W)
    assert np.array_equal(before, after)


def test_return_window_insufficient_history():
    rets = r.log_returns(_prices())
    with pytest.raises(ValueError):
        r.return_window(rets, t=0, W=3)  # 이전 데이터 부족
