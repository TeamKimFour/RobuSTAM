"""PortfolioEnv 골격 테스트 — 관측/행동 shape, reset 초기 비중, 스키마 검증.

Softmax·거래비용·보상 로직은 후속 PR에서 채우므로 여기서는 검증하지 않는다.
gymnasium 미설치 환경에서도 파이썬 임포트가 실패하지 않도록 importorskip으로 가드한다.
"""

import numpy as np
import pandas as pd
import pytest

from src.data import schema

pytest.importorskip("gymnasium")

from src.env.portfolio_env import PortfolioEnv  # noqa: E402


def _fake_cfg(window: int = 30) -> dict:
    return {
        "assets": ["SPY", "EWY", "TLT", "GLD", "SHV"],
        "window": window,
        "transaction_cost": 0.001,
    }


def _fake_state_df(window: int, n_rows: int = 10) -> pd.DataFrame:
    cols = schema.feature_names(window)
    dates = pd.date_range("2020-01-01", periods=n_rows, freq="B")
    data = np.arange(n_rows * len(cols), dtype=np.float32).reshape(n_rows, len(cols))
    return pd.DataFrame(data, index=dates, columns=cols)


def test_observation_space_W30():
    env = PortfolioEnv(_fake_state_df(30), cfg=_fake_cfg(30))
    assert env.observation_space.shape == (187,)


def test_observation_shape_scales_with_window():
    """187을 하드코딩하지 않고 W에서 산출되어야 한다 (state_spec.md §5③)."""
    for w, expected in [(20, 137), (30, 187), (60, 337)]:
        env = PortfolioEnv(_fake_state_df(w), cfg=_fake_cfg(w))
        assert env.observation_space.shape == (expected,), f"W={w}"


def test_action_space_matches_n_assets():
    env = PortfolioEnv(_fake_state_df(30), cfg=_fake_cfg(30))
    assert env.action_space.shape == (5,)


def test_reset_returns_obs_and_empty_info():
    env = PortfolioEnv(_fake_state_df(30), cfg=_fake_cfg(30))
    obs, info = env.reset()
    assert obs.shape == (187,)
    assert obs.dtype == np.float32
    assert info == {}


def test_reset_initial_weight_is_shv_100pct():
    """무위험 자산에서 출발 — reset 후 prev_weight 블록은 SHV만 1.0."""
    W = 30
    env = PortfolioEnv(_fake_state_df(W), cfg=_fake_cfg(W))
    obs, _ = env.reset()
    prev_w = obs[schema.prev_weight_slice(W)]
    expected = np.array([0.0, 0.0, 0.0, 0.0, 1.0], dtype=np.float32)
    np.testing.assert_array_equal(prev_w, expected)


def test_step_returns_gymnasium_5_tuple():
    env = PortfolioEnv(_fake_state_df(30, n_rows=5), cfg=_fake_cfg(30))
    env.reset()
    obs, reward, terminated, truncated, info = env.step(np.zeros(5, dtype=np.float32))
    assert obs.shape == (187,)
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert info == {}


def test_episode_terminates_at_last_row():
    n = 4
    env = PortfolioEnv(_fake_state_df(30, n_rows=n), cfg=_fake_cfg(30))
    env.reset()
    action = np.zeros(5, dtype=np.float32)
    for _ in range(n - 2):
        _, _, terminated, _, _ = env.step(action)
        assert not terminated
    _, _, terminated, _, _ = env.step(action)
    assert terminated


def test_column_schema_mismatch_raises():
    W = 30
    df = _fake_state_df(W)
    bad = df.rename(columns={df.columns[0]: "wrong"})
    with pytest.raises(ValueError, match="schema.feature_names"):
        PortfolioEnv(bad, cfg=_fake_cfg(W))


def test_too_few_rows_raises():
    W = 30
    df = _fake_state_df(W, n_rows=1)
    with pytest.raises(ValueError, match="최소 2행"):
        PortfolioEnv(df, cfg=_fake_cfg(W))


def test_transaction_cost_loaded_from_cfg():
    """거래비용은 config에서 읽어야 하며 코드에 박히면 안 된다."""
    cfg = _fake_cfg(30)
    cfg["transaction_cost"] = 0.0025
    env = PortfolioEnv(_fake_state_df(30), cfg=cfg)
    assert env.c == 0.0025
