"""거래비용 보상 셰이핑(λ) 계약 검증 — 이슈 #34 개선안 A.

λ(`model.train_cost_multiplier`)는 **학습 보상에만** 거는 가중치다. 이 파일이 지키는 것은
"셰이핑이 성과 측정 축으로 새지 않는가" 하나다:

    - 보상에는 λ가 걸린다 (R = log_return − λ·c·turnover)
    - `info["cost"]`는 λ와 무관하게 **실비용**(c·turnover)을 보고한다
    - 기본값 λ=1.0에서는 기존 동작과 완전히 동일하다

이 분리가 깨지면 정책이 벤치마크와 다른 자로 측정되는데, 수치는 그럴듯하게 나와서
조용히 틀린다(이슈 #34에서 lr1e-6이 "샤프 0.996"으로 성공처럼 보였던 것과 같은 함정).
그래서 계약을 테스트로 박아둔다.
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

# SHV 100% → SPY 100% 완전 스왑을 유도하는 로짓 (turnover=2)
FULL_SWAP_LOGITS = np.array([1e6, -1e6, -1e6, -1e6, -1e6], dtype=np.float64)


def _cfg(window: int = 30, c: float = 0.001) -> dict:
    return {"assets": ASSETS, "window": window, "transaction_cost": c}


def _state_df(window: int, n_rows: int = 10) -> pd.DataFrame:
    cols = schema.feature_names(window)
    dates = pd.date_range("2020-01-01", periods=n_rows, freq="B")
    data = np.arange(n_rows * len(cols), dtype=np.float32).reshape(n_rows, len(cols))
    return pd.DataFrame(data, index=dates, columns=cols)


def _zero_returns_df(state_df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        np.zeros((len(state_df), 5), dtype=np.float32),
        index=state_df.index,
        columns=FWD_RET_COLS,
    )


def _env(cost_multiplier: float, c: float = 0.001) -> PortfolioEnv:
    W = 30
    s = _state_df(W)
    return PortfolioEnv(s, _zero_returns_df(s), cfg=_cfg(W, c), cost_multiplier=cost_multiplier)


# ── 기본값은 기존 동작과 동일해야 한다 ──────────────────────────────

def test_default_multiplier_is_one_and_preserves_legacy_contract():
    """λ 미지정 시 1.0 — 보상·실비용이 일치하던 기존 계약이 그대로 유지된다."""
    env = _env(cost_multiplier=1.0)
    assert env.cost_multiplier == 1.0
    assert env.c == env.c_real

    env.reset()
    _, reward, _, _, info = env.step(FULL_SWAP_LOGITS)
    # 수익률 0인 구간이라 보상은 순전히 -비용
    assert reward == pytest.approx(info["log_return"] - info["cost"], abs=1e-12)
    assert info["shaped_cost"] == pytest.approx(info["cost"], abs=1e-12)


def test_explicit_multiplier_one_matches_no_argument():
    """λ=1.0을 명시해도 인자를 생략한 것과 완전히 같다(기본값 회귀 방어)."""
    a = _env(cost_multiplier=1.0)
    W, s = 30, _state_df(30)
    b = PortfolioEnv(s, _zero_returns_df(s), cfg=_cfg(W, 0.001))
    a.reset()
    b.reset()
    _, ra, _, _, ia = a.step(FULL_SWAP_LOGITS)
    _, rb, _, _, ib = b.step(FULL_SWAP_LOGITS)
    assert ra == pytest.approx(rb, abs=1e-12)
    assert ia["cost"] == pytest.approx(ib["cost"], abs=1e-12)


# ── λ>1: 보상만 커지고 실비용 보고는 그대로 ────────────────────────

@pytest.mark.parametrize("lam", [2.0, 5.0, 10.0])
def test_reward_penalty_scales_with_lambda(lam):
    """보상의 비용 항은 정확히 λ배가 된다 — R = log_return − λ·c·turnover."""
    c = 0.001
    env = _env(cost_multiplier=lam, c=c)
    env.reset()
    _, reward, _, _, info = env.step(FULL_SWAP_LOGITS)

    turnover = info["turnover"]
    assert turnover == pytest.approx(2.0, abs=1e-6)  # 완전 스왑
    assert reward == pytest.approx(info["log_return"] - lam * c * turnover, abs=1e-12)


@pytest.mark.parametrize("lam", [2.0, 5.0, 10.0])
def test_info_cost_reports_real_money_not_shaped(lam):
    """λ를 걸어도 `info["cost"]`는 실제로 나간 돈(c·turnover)이다.

    진단·집계가 셰이핑된 값을 실비용으로 오독하면 "비용이 λ배로 늘었다"는 착시가 생긴다.
    """
    c = 0.001
    env = _env(cost_multiplier=lam, c=c)
    env.reset()
    _, _, _, _, info = env.step(FULL_SWAP_LOGITS)

    assert info["cost"] == pytest.approx(c * info["turnover"], abs=1e-12)
    assert info["shaped_cost"] == pytest.approx(lam * info["cost"], abs=1e-12)


def test_turnover_is_untouched_by_lambda():
    """turnover는 셰이핑과 무관한 물리량 — λ가 달라도 동일해야 한다."""
    base = _env(cost_multiplier=1.0)
    shaped = _env(cost_multiplier=10.0)
    base.reset()
    shaped.reset()
    _, _, _, _, ib = base.step(FULL_SWAP_LOGITS)
    _, _, _, _, is_ = shaped.step(FULL_SWAP_LOGITS)
    assert ib["turnover"] == pytest.approx(is_["turnover"], abs=1e-12)


def test_invalid_multiplier_rejected():
    """0·음수 λ는 비용을 보상으로 바꿔버리므로 생성 시점에 막는다."""
    for bad in (0.0, -1.0):
        with pytest.raises(ValueError, match="cost_multiplier"):
            _env(cost_multiplier=bad)
