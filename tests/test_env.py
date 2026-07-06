"""PortfolioEnv 테스트 — 관측/행동 shape, reset, 도현/민지 API 정합 (기본).

Softmax·로그→산술 변환·상세 보상 위임 검증은 후속 커밋에서 확장한다.
"""

import numpy as np
import pandas as pd
import pytest

from src.data import schema

pytest.importorskip("gymnasium")

from src.env.portfolio_env import PortfolioEnv  # noqa: E402

ASSETS = ["SPY", "EWY", "TLT", "GLD", "SHV"]
FWD_RET_COLS = [f"fwd_ret_{a}" for a in ASSETS]


def _fake_cfg(window: int = 30, c: float = 0.001) -> dict:
    return {"assets": ASSETS, "window": window, "transaction_cost": c}


def _fake_state_df(window: int, n_rows: int = 10) -> pd.DataFrame:
    cols = schema.feature_names(window)
    dates = pd.date_range("2020-01-01", periods=n_rows, freq="B")
    data = np.arange(n_rows * len(cols), dtype=np.float32).reshape(n_rows, len(cols))
    return pd.DataFrame(data, index=dates, columns=cols)


def _fake_log_returns_df(state_df: pd.DataFrame, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    data = rng.normal(0.0, 0.01, size=(len(state_df), 5)).astype(np.float32)
    return pd.DataFrame(data, index=state_df.index, columns=FWD_RET_COLS)


def _mk_env(W: int = 30, n_rows: int = 10, c: float = 0.001) -> PortfolioEnv:
    s = _fake_state_df(W, n_rows)
    r = _fake_log_returns_df(s)
    return PortfolioEnv(s, r, cfg=_fake_cfg(W, c))


def test_observation_space_W30():
    env = _mk_env()
    assert env.observation_space.shape == (187,)


def test_observation_shape_scales_with_window():
    """187을 하드코딩하지 않고 W에서 산출되어야 한다."""
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
    """팀 회의 확정 옵션 a: SHV 100%에서 출발."""
    W = 30
    env = _mk_env(W=W)
    obs, _ = env.reset()
    prev_w = obs[schema.prev_weight_slice(W)]
    expected = np.array([0.0, 0.0, 0.0, 0.0, 1.0], dtype=np.float32)
    np.testing.assert_array_equal(prev_w, expected)


def test_state_column_schema_mismatch_raises():
    W = 30
    s = _fake_state_df(W)
    r = _fake_log_returns_df(s)
    bad = s.rename(columns={s.columns[0]: "wrong"})
    with pytest.raises(ValueError, match="schema.feature_names"):
        PortfolioEnv(bad, r, cfg=_fake_cfg(W))


def test_returns_df_must_use_fwd_ret_prefix():
    """민지 feature_store.load_targets 출력과 정합."""
    W = 30
    s = _fake_state_df(W)
    bad = _fake_log_returns_df(s).rename(columns={"fwd_ret_SPY": "SPY"})
    with pytest.raises(ValueError, match="fwd_ret_"):
        PortfolioEnv(s, bad, cfg=_fake_cfg(W))


def test_returns_df_index_must_match_state_df():
    W = 30
    s = _fake_state_df(W)
    bad = _fake_log_returns_df(s)
    bad.index = pd.date_range("2010-01-01", periods=len(bad), freq="B")
    with pytest.raises(ValueError, match="인덱스"):
        PortfolioEnv(s, bad, cfg=_fake_cfg(W))


def test_too_few_rows_raises():
    W = 30
    s = _fake_state_df(W, n_rows=1)
    r = _fake_log_returns_df(s)
    with pytest.raises(ValueError, match="최소 2행"):
        PortfolioEnv(s, r, cfg=_fake_cfg(W))


def test_transaction_cost_loaded_from_cfg():
    env = _mk_env(c=0.0025)
    assert env.c == 0.0025


def test_step_returns_gymnasium_5_tuple():
    env = _mk_env(n_rows=5)
    env.reset()
    obs, reward, terminated, truncated, info = env.step(np.zeros(5, dtype=np.float64))
    assert obs.shape == (187,)
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)


def test_episode_terminates_at_last_row():
    n = 4
    env = _mk_env(n_rows=n)
    env.reset()
    action = np.zeros(5, dtype=np.float64)
    for _ in range(n - 2):
        _, _, terminated, _, _ = env.step(action)
        assert not terminated
    _, _, terminated, _, _ = env.step(action)
    assert terminated
