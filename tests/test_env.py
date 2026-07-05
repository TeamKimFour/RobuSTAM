"""PortfolioEnv 테스트 — 관측/행동 shape 및 스키마 검증.

Softmax·거래비용·보상 로직 세부 검증은 후속 커밋에서 추가한다.
gymnasium 미설치 환경에서도 파이썬 임포트가 실패하지 않도록 importorskip으로 가드한다.
"""

import numpy as np
import pandas as pd
import pytest

from src.data import schema

pytest.importorskip("gymnasium")

from src.env.portfolio_env import PortfolioEnv  # noqa: E402

ASSETS = ["SPY", "EWY", "TLT", "GLD", "SHV"]


def _fake_cfg(window: int = 30, c: float = 0.001) -> dict:
    return {
        "assets": ASSETS,
        "window": window,
        "transaction_cost": c,
    }


def _fake_state_df(window: int, n_rows: int = 10) -> pd.DataFrame:
    cols = schema.feature_names(window)
    dates = pd.date_range("2020-01-01", periods=n_rows, freq="B")
    data = np.arange(n_rows * len(cols), dtype=np.float32).reshape(n_rows, len(cols))
    return pd.DataFrame(data, index=dates, columns=cols)


def _fake_returns_df(state_df: pd.DataFrame, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    data = rng.normal(0.0, 0.01, size=(len(state_df), 5)).astype(np.float32)
    return pd.DataFrame(data, index=state_df.index, columns=ASSETS)


def _mk_env(W: int = 30, n_rows: int = 10, c: float = 0.001) -> PortfolioEnv:
    s = _fake_state_df(W, n_rows)
    r = _fake_returns_df(s)
    return PortfolioEnv(s, r, cfg=_fake_cfg(W, c))


def test_observation_space_W30():
    env = _mk_env()
    assert env.observation_space.shape == (187,)


def test_observation_shape_scales_with_window():
    """187을 하드코딩하지 않고 W에서 산출되어야 한다 (state_spec.md §5③)."""
    for w, expected in [(20, 137), (30, 187), (60, 337)]:
        env = _mk_env(W=w)
        assert env.observation_space.shape == (expected,), f"W={w}"


def test_action_space_matches_n_assets():
    env = _mk_env()
    assert env.action_space.shape == (5,)


def test_reset_returns_obs_and_empty_info():
    env = _mk_env()
    obs, info = env.reset()
    assert obs.shape == (187,)
    assert obs.dtype == np.float32
    assert info == {}


def test_reset_initial_weight_is_shv_100pct():
    """무위험 자산에서 출발 — reset 후 prev_weight 블록은 SHV만 1.0."""
    W = 30
    env = _mk_env(W=W)
    obs, _ = env.reset()
    prev_w = obs[schema.prev_weight_slice(W)]
    expected = np.array([0.0, 0.0, 0.0, 0.0, 1.0], dtype=np.float32)
    np.testing.assert_array_equal(prev_w, expected)


def test_column_schema_mismatch_raises():
    W = 30
    s = _fake_state_df(W)
    r = _fake_returns_df(s)
    bad = s.rename(columns={s.columns[0]: "wrong"})
    with pytest.raises(ValueError, match="schema.feature_names"):
        PortfolioEnv(bad, r, cfg=_fake_cfg(W))


def test_too_few_rows_raises():
    W = 30
    s = _fake_state_df(W, n_rows=1)
    r = _fake_returns_df(s)
    with pytest.raises(ValueError, match="최소 2행"):
        PortfolioEnv(s, r, cfg=_fake_cfg(W))


def test_transaction_cost_loaded_from_cfg():
    env = _mk_env(c=0.0025)
    assert env.c == 0.0025


def test_step_returns_gymnasium_5_tuple():
    env = _mk_env(n_rows=5)
    env.reset()
    obs, reward, terminated, truncated, info = env.step(np.zeros(5, dtype=np.float32))
    assert obs.shape == (187,)
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)


def test_episode_terminates_at_last_row():
    n = 4
    env = _mk_env(n_rows=n)
    env.reset()
    action = np.zeros(5, dtype=np.float32)
    for _ in range(n - 2):
        _, _, terminated, _, _ = env.step(action)
        assert not terminated
    _, _, terminated, _, _ = env.step(action)
    assert terminated
