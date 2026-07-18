"""DiscretePortfolioEnv 테스트 — DQN용 이산 행동 어댑터의 계약 검증.

원본 PortfolioEnv 계약(관측 shape, ∑w=1, reward 위임)이 어댑터를 거쳐도 그대로
유지되는지, 그리고 이산 action → 비중 이동 매핑이 SHV 상대 교환 규칙(옵션 B)을
정확히 따르는지 검증한다.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data import schema

pytest.importorskip("gymnasium")

from src.env.discrete_env import DiscretePortfolioEnv  # noqa: E402
from src.env.portfolio_env import PortfolioEnv  # noqa: E402

ASSETS = ["SPY", "EWY", "TLT", "GLD", "SHV"]
FWD_RET_COLS = [f"fwd_ret_{a}" for a in ASSETS]
SHV_IDX = 4
TRADABLE_IDX = [0, 1, 2, 3]  # SPY, EWY, TLT, GLD
HOLD_ACTION = 8               # 2·A_tr = 8


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


def _mk(delta: float = 0.1, n_rows: int = 10, c: float = 0.001) -> DiscretePortfolioEnv:
    W = 30
    s = _state_df(W, n_rows)
    r = _zero_returns_df(s)  # 이동 계산 검증엔 return=0이 편함(비중만 이동)
    base = PortfolioEnv(s, r, cfg=_cfg(W, c))
    return DiscretePortfolioEnv(base, delta=delta)


# ── action/observation space ─────────────────────────────────────────

def test_action_space_is_discrete_9():
    env = _mk()
    assert env.action_space.n == 9  # 2·A_tr + 1 = 9


def test_observation_space_unchanged_after_wrap():
    env = _mk()
    assert env.observation_space.shape == (187,)


def test_reset_preserves_shv100_initial_weight():
    env = _mk()
    obs, info = env.reset()
    assert obs.shape == (187,)
    prev_w = obs[schema.prev_weight_slice(30)]
    np.testing.assert_array_equal(
        prev_w, np.array([0.0, 0.0, 0.0, 0.0, 1.0], dtype=np.float32)
    )
    assert info == {}


# ── 생성자 검증 ────────────────────────────────────────────────────────

def test_delta_out_of_range_raises():
    W = 30
    s = _state_df(W)
    r = _zero_returns_df(s)
    base = PortfolioEnv(s, r, cfg=_cfg(W))
    with pytest.raises(ValueError, match="delta"):
        DiscretePortfolioEnv(base, delta=0.0)
    with pytest.raises(ValueError, match="delta"):
        DiscretePortfolioEnv(base, delta=1.5)


def test_invalid_action_index_raises():
    env = _mk()
    env.reset()
    with pytest.raises(ValueError, match="이산 action"):
        env.step(9)  # 유효 범위 [0, 9)
    with pytest.raises(ValueError, match="이산 action"):
        env.step(-1)


# ── hold action (=8) ─────────────────────────────────────────────────

def test_hold_action_keeps_weights_unchanged():
    env = _mk()
    env.reset()
    _, _, _, _, info = env.step(HOLD_ACTION)
    # 초기 SHV 100%가 유지되어야 함
    np.testing.assert_allclose(
        info["weights"], np.array([0.0, 0.0, 0.0, 0.0, 1.0]), atol=1e-6
    )


# ── +Δ actions (0..3): asset ← SHV ────────────────────────────────────

@pytest.mark.parametrize("action_idx, asset_idx", list(enumerate(TRADABLE_IDX)))
def test_plus_delta_moves_from_shv_to_asset(action_idx, asset_idx):
    delta = 0.1
    env = _mk(delta=delta)
    env.reset()
    _, _, _, _, info = env.step(action_idx)
    expected = np.zeros(5)
    expected[asset_idx] = delta
    expected[SHV_IDX] = 1.0 - delta
    np.testing.assert_allclose(info["weights"], expected, atol=1e-6)


def test_plus_delta_clipped_when_shv_insufficient():
    """SHV 여유분보다 큰 Δ가 들어오면 여유분만큼만 이동한다."""
    delta = 0.4  # SHV가 0.3까지 줄어들면 다음 +0.4는 0.3만 이동
    env = _mk(delta=delta)
    env.reset()
    # 3번 연속 +Δ_SPY → SHV: 1 - 3·0.4 는 음수 → 실제로는 0.4 + 0.4 + 0.2 만 이동
    for _ in range(3):
        _, _, _, _, info = env.step(0)  # +Δ SPY
    # SHV = 0에 도달, SPY = 1.0
    np.testing.assert_allclose(info["weights"][SHV_IDX], 0.0, atol=1e-6)
    np.testing.assert_allclose(info["weights"][0], 1.0, atol=1e-6)


# ── -Δ actions (4..7): asset → SHV ────────────────────────────────────

def test_minus_delta_no_op_when_asset_is_zero():
    """자산 비중이 0인 상태에서 -Δ가 들어오면 이전량 0이라 no-op."""
    env = _mk(delta=0.1)
    env.reset()  # SPY=0에서 시작
    _, _, _, _, info = env.step(4)  # -Δ SPY (asset 0)
    np.testing.assert_allclose(
        info["weights"], np.array([0.0, 0.0, 0.0, 0.0, 1.0]), atol=1e-6
    )


def test_minus_delta_moves_from_asset_to_shv():
    """+Δ로 자산을 쌓은 뒤 -Δ로 되돌리면 정확히 원위치로 돌아온다."""
    delta = 0.2
    env = _mk(delta=delta)
    env.reset()
    env.step(1)  # +Δ EWY → EWY=0.2, SHV=0.8
    _, _, _, _, info = env.step(5)  # -Δ EWY (asset idx 1)
    np.testing.assert_allclose(
        info["weights"], np.array([0.0, 0.0, 0.0, 0.0, 1.0]), atol=1e-6
    )


# ── sum-to-1 불변식 (임의 시퀀스) ─────────────────────────────────────

def test_weights_sum_to_1_after_arbitrary_sequence():
    env = _mk(delta=0.1, n_rows=20)
    env.reset()
    rng = np.random.default_rng(0)
    for _ in range(15):
        a = int(rng.integers(0, 9))
        _, _, terminated, _, info = env.step(a)
        assert np.isclose(info["weights"].sum(), 1.0, atol=1e-5)
        assert (info["weights"] >= -1e-8).all()
        if terminated:
            break


# ── obs/step 계약 통과 ────────────────────────────────────────────────

def test_step_returns_gymnasium_5tuple():
    env = _mk()
    env.reset()
    obs, reward, terminated, truncated, info = env.step(0)
    assert obs.shape == (187,)
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    for key in ("portfolio_return", "log_return", "turnover", "cost", "weights"):
        assert key in info


# ── SB3 DQN 스모크: 환경-에이전트 연결 검증 (3주차 완료 기준) ────────

def test_sb3_dqn_learns_a_few_steps():
    """SB3 DQN이 어댑터에 붙어 예외 없이 몇 스텝 학습된다.

    3주차 완료 기준의 "환경-에이전트 연결 확인"에 해당한다. 학습 성능이 아닌
    파이프라인 결선(action_space 호환·reset/step 사이클·reward 스칼라)만 본다.
    """
    sb3 = pytest.importorskip("stable_baselines3")

    W = 30
    n_rows = 60
    s = _state_df(W, n_rows)
    # 학습에 신호가 필요하므로 결정적 소량 수익률
    rng = np.random.default_rng(0)
    r_arr = rng.normal(0.0, 0.005, size=(n_rows, 5)).astype(np.float32)
    r = pd.DataFrame(r_arr, index=s.index, columns=FWD_RET_COLS)
    env = DiscretePortfolioEnv(PortfolioEnv(s, r, cfg=_cfg(W)), delta=0.1)

    # DQN은 replay buffer가 learning_starts만큼 채워진 뒤 학습을 시작한다.
    # 스모크라서 buffer/batch를 매우 작게 잡아 CPU에서 수 초 안에 끝나게 한다.
    model = sb3.DQN(
        "MlpPolicy",
        env,
        learning_starts=32,
        buffer_size=256,
        batch_size=16,
        train_freq=8,
        target_update_interval=32,
        verbose=0,
        seed=0,
    )
    model.learn(total_timesteps=100)  # 예외 없이 완주하면 통과
