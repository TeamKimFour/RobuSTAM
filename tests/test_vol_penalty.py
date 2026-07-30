"""위험조정 보상 페널티(κ·σ_recent) 계약 검증 — 이슈 #34 개선안 D.

κ(`model.vol_penalty_coef`)는 λ와 마찬가지로 **학습 보상에만** 거는 셰이핑이다. 이 파일이
지키는 계약:

    - 보상에서 κ·σ_recent가 정확히 깎인다 (R = log_return − λ·c·turnover − κ·σ_recent)
    - σ_recent는 **직전 스텝까지** 실현된 포트폴리오 로그수익률의 표준편차다(인과성 —
      행동 시점에 알 수 없는 당일 수익률을 넣으면 룩어헤드가 된다)
    - 기본값 κ=0.0에서는 기존 동작과 완전히 동일하다(페널티 없음)
    - reset()은 롤링 버퍼를 비운다(에피소드 간 오염 방지)

λ 셰이핑(test_cost_shaping.py)과 직교한다 — 둘 다 걸려도 서로 간섭하지 않는다.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data import schema

pytest.importorskip("gymnasium")

from src.env.portfolio_env import PortfolioEnv  # noqa: E402

ASSETS = ["SPY", "EWY", "TLT", "GLD", "SHV"]
FWD_RET_COLS = [f"fwd_ret_{a}" for a in ASSETS]

# 매 스텝 SPY 100%로 몰아가는 로짓 — 포트폴리오 로그수익률 = 그 스텝 SPY fwd 로그수익률이 된다.
SPY_ALL_LOGITS = np.array([1e6, -1e6, -1e6, -1e6, -1e6], dtype=np.float64)


def _cfg(window: int = 30, c: float = 0.001, vol_window: int | None = None) -> dict:
    cfg: dict = {"assets": ASSETS, "window": window, "transaction_cost": c}
    if vol_window is not None:
        cfg["model"] = {"vol_penalty_window": vol_window}
    return cfg


def _state_df(window: int, n_rows: int) -> pd.DataFrame:
    cols = schema.feature_names(window)
    dates = pd.date_range("2020-01-01", periods=n_rows, freq="B")
    data = np.arange(n_rows * len(cols), dtype=np.float32).reshape(n_rows, len(cols))
    return pd.DataFrame(data, index=dates, columns=cols)


def _returns_df(state_df: pd.DataFrame, spy_series: np.ndarray) -> pd.DataFrame:
    """SPY 열만 지정 시계열(로그수익률)로 채우고 나머지는 0."""
    arr = np.zeros((len(state_df), 5), dtype=np.float64)
    arr[:, ASSETS.index("SPY")] = spy_series
    return pd.DataFrame(arr, index=state_df.index, columns=FWD_RET_COLS)


# SPY 로그수익률 시계열 (스텝별로 다르게 — 변동성이 실제로 생기도록)
SPY_LOG_RETS = np.array([0.010, -0.020, 0.030, 0.000, 0.015, -0.005, 0.025, 0.008, -0.012])


def _env(vol_penalty_coef: float, *, c: float = 0.001, vol_window: int | None = None,
         cost_multiplier: float = 1.0) -> PortfolioEnv:
    W = 30
    n = len(SPY_LOG_RETS) + 1  # step은 len-1까지 도므로 시계열 전체를 소진하려면 +1행
    s = _state_df(W, n)
    r = _returns_df(s, np.append(SPY_LOG_RETS, 0.0))
    return PortfolioEnv(
        s, r, cfg=_cfg(W, c, vol_window),
        cost_multiplier=cost_multiplier, vol_penalty_coef=vol_penalty_coef,
    )


# ── 기본값 κ=0 → 기존 동작 보존 ──────────────────────────────────────

def test_default_coef_is_zero_and_no_penalty():
    """κ 미지정 시 0.0 — 보상에 페널티가 붙지 않고 vol_penalty=0."""
    env = _env(vol_penalty_coef=0.0)
    assert env.vol_penalty_coef == 0.0
    env.reset()
    total_pen = 0.0
    terminated = False
    while not terminated:
        _, reward, terminated, _, info = env.step(SPY_ALL_LOGITS)
        total_pen += info["vol_penalty"]
        # κ=0이라 보상은 순전히 log_return − 실비용(λ=1)만으로 설명된다.
        assert reward == pytest.approx(info["log_return"] - info["cost"], abs=1e-12)
    assert total_pen == 0.0


def test_omitting_coef_matches_explicit_zero():
    """κ 인자를 생략해도 0.0을 명시한 것과 동일(기본값 회귀 방어)."""
    W, n = 30, len(SPY_LOG_RETS) + 1
    s = _state_df(W, n)
    r = _returns_df(s, np.append(SPY_LOG_RETS, 0.0))
    a = PortfolioEnv(s, r, cfg=_cfg(W))
    b = PortfolioEnv(s, r, cfg=_cfg(W), vol_penalty_coef=0.0)
    a.reset()
    b.reset()
    _, ra, _, _, _ = a.step(SPY_ALL_LOGITS)
    _, rb, _, _, _ = b.step(SPY_ALL_LOGITS)
    assert ra == pytest.approx(rb, abs=1e-12)


# ── σ_recent는 직전 스텝까지의 실현 변동성(인과성) ──────────────────

def test_recent_vol_is_causal_std_of_prior_returns():
    """info["recent_vol"]는 이번 스텝 수익률을 제외한, 직전까지 실현 로그수익률의 표준편차다."""
    env = _env(vol_penalty_coef=0.0)
    env.reset()
    realized: list[float] = []
    for i in range(len(SPY_LOG_RETS)):
        # 이번 스텝 진입 시점의 recent_vol은 지금까지 쌓인 realized로만 계산돼야 한다.
        expected = float(np.std(realized)) if len(realized) >= 2 else 0.0
        _, _, terminated, _, info = env.step(SPY_ALL_LOGITS)
        assert info["recent_vol"] == pytest.approx(expected, abs=1e-12), f"step {i}"
        # 포트폴리오 로그수익률(=SPY 로그수익률)이 사후에 버퍼로 들어간다.
        realized.append(info["log_return"])
        assert info["log_return"] == pytest.approx(SPY_LOG_RETS[i], abs=1e-9)
        if terminated:
            break


def test_first_two_steps_have_zero_vol():
    """표본이 2개 미만이면 표준편차가 정의되지 않으므로 페널티 면제(recent_vol=0)."""
    env = _env(vol_penalty_coef=5.0)
    env.reset()
    _, _, _, _, i0 = env.step(SPY_ALL_LOGITS)  # 버퍼 비어 있음
    _, _, _, _, i1 = env.step(SPY_ALL_LOGITS)  # 표본 1개
    assert i0["recent_vol"] == 0.0 and i0["vol_penalty"] == 0.0
    assert i1["recent_vol"] == 0.0 and i1["vol_penalty"] == 0.0


# ── κ>0: 보상에서 정확히 κ·σ가 깎인다 ────────────────────────────────

def test_reward_subtracts_kappa_times_recent_vol():
    kappa = 4.0
    env = _env(vol_penalty_coef=kappa)
    env.reset()
    terminated = False
    while not terminated:
        _, reward, terminated, _, info = env.step(SPY_ALL_LOGITS)
        assert info["vol_penalty"] == pytest.approx(kappa * info["recent_vol"], abs=1e-12)
        assert reward == pytest.approx(
            info["log_return"] - info["cost"] - kappa * info["recent_vol"], abs=1e-12
        )


def test_kappa_orthogonal_to_lambda():
    """λ와 κ는 서로 독립 — 동시에 걸면 두 페널티가 각각 그대로 반영된다."""
    lam, kappa, c = 3.0, 2.0, 0.001
    env = _env(vol_penalty_coef=kappa, c=c, cost_multiplier=lam)
    env.reset()
    env.step(SPY_ALL_LOGITS)  # 버퍼 워밍업
    env.step(SPY_ALL_LOGITS)
    _, reward, _, _, info = env.step(SPY_ALL_LOGITS)  # 이제 recent_vol>0
    assert info["recent_vol"] > 0
    # info["cost"]는 실비용(λ 무관), 보상엔 λ·c·turnover와 κ·σ가 함께 반영된다.
    expected = info["log_return"] - lam * c * info["turnover"] - kappa * info["recent_vol"]
    assert reward == pytest.approx(expected, abs=1e-12)
    assert info["shaped_cost"] == pytest.approx(lam * info["cost"], abs=1e-12)


# ── reset이 버퍼를 비운다 ─────────────────────────────────────────────

def test_reset_clears_return_history():
    env = _env(vol_penalty_coef=5.0)
    env.reset()
    for _ in range(4):
        env.step(SPY_ALL_LOGITS)
    assert len(env._return_history) > 0
    env.reset()
    assert len(env._return_history) == 0
    # 리셋 직후 첫 스텝은 다시 표본 0 → 페널티 없음
    _, _, _, _, info = env.step(SPY_ALL_LOGITS)
    assert info["recent_vol"] == 0.0


# ── 입력 검증 ─────────────────────────────────────────────────────────

def test_negative_coef_rejected():
    with pytest.raises(ValueError, match="vol_penalty_coef"):
        _env(vol_penalty_coef=-0.5)


def test_window_below_two_rejected():
    with pytest.raises(ValueError, match="vol_penalty_window"):
        _env(vol_penalty_coef=1.0, vol_window=1)
