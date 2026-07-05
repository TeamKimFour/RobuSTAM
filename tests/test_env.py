"""PortfolioEnv 테스트 — 관측/행동 shape, Softmax·거래비용·보상 로직 검증.

슬리피지와 확장 보상(Differential Sharpe) 검증은 후속 PR-3에서 추가한다.
gymnasium 미설치 환경에서도 파이썬 임포트가 실패하지 않도록 importorskip으로 가드한다.
"""

import numpy as np
import pandas as pd
import pytest

from src.data import schema

pytest.importorskip("gymnasium")

from src.env.portfolio_env import PortfolioEnv, _softmax  # noqa: E402

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


def _zero_returns_df(state_df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        np.zeros((len(state_df), 5), dtype=np.float32),
        index=state_df.index,
        columns=ASSETS,
    )


def _mk_env(W: int = 30, n_rows: int = 10, c: float = 0.001) -> PortfolioEnv:
    s = _fake_state_df(W, n_rows)
    r = _fake_returns_df(s)
    return PortfolioEnv(s, r, cfg=_fake_cfg(W, c))


# ── 관측/행동 shape 계약 ────────────────────────────────────────────────

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


# ── reset ──────────────────────────────────────────────────────────────

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


# ── 생성자 계약 (state_df + returns_df 검증) ──────────────────────────

def test_column_schema_mismatch_raises():
    W = 30
    s = _fake_state_df(W)
    r = _fake_returns_df(s)
    bad = s.rename(columns={s.columns[0]: "wrong"})
    with pytest.raises(ValueError, match="schema.feature_names"):
        PortfolioEnv(bad, r, cfg=_fake_cfg(W))


def test_returns_df_columns_must_match_assets_in_order():
    W = 30
    s = _fake_state_df(W)
    bad = _fake_returns_df(s).rename(columns={"SPY": "AAPL"})
    with pytest.raises(ValueError, match="assets 순서"):
        PortfolioEnv(s, bad, cfg=_fake_cfg(W))


def test_returns_df_index_must_match_state_df():
    W = 30
    s = _fake_state_df(W)
    bad = _fake_returns_df(s)
    bad.index = pd.date_range("2010-01-01", periods=len(bad), freq="B")
    with pytest.raises(ValueError, match="인덱스"):
        PortfolioEnv(s, bad, cfg=_fake_cfg(W))


def test_too_few_rows_raises():
    W = 30
    s = _fake_state_df(W, n_rows=1)
    r = _fake_returns_df(s)
    with pytest.raises(ValueError, match="최소 2행"):
        PortfolioEnv(s, r, cfg=_fake_cfg(W))


def test_transaction_cost_loaded_from_cfg():
    env = _mk_env(c=0.0025)
    assert env.c == 0.0025


# ── Softmax 정규화 ─────────────────────────────────────────────────────

def test_softmax_output_sums_to_1_and_non_negative():
    for logits in [
        np.array([1.0, 1.0, 1.0, 1.0, 1.0]),
        np.array([-2.0, 3.0, 0.0, 1.5, -0.5]),
        np.array([100.0, 0.0, 0.0, 0.0, 0.0]),
        np.array([1e5, 1e5, 1e5, 1e5, 1e5 + 1.0]),  # 오버플로 위험 로짓
    ]:
        w = _softmax(logits.astype(np.float32))
        assert np.isclose(w.sum(), 1.0, atol=1e-6), f"logits={logits}"
        assert (w >= 0).all()
        assert np.isfinite(w).all()


def test_step_updates_prev_weight_block_with_softmax_output():
    """step 이후 다음 관측의 직전비중 블록은 방금 선택한 w_new(Softmax 출력)와 일치."""
    W = 30
    env = _mk_env(W=W)
    env.reset()
    logits = np.array([1.0, -1.0, 0.5, 0.0, 2.0], dtype=np.float32)
    obs, _, _, _, info = env.step(logits)
    w_expected = _softmax(logits)
    np.testing.assert_allclose(info["weights"], w_expected, atol=1e-6)
    np.testing.assert_allclose(
        obs[schema.prev_weight_slice(W)], w_expected, atol=1e-6
    )


def test_action_wrong_shape_raises():
    env = _mk_env()
    env.reset()
    with pytest.raises(ValueError, match="action shape"):
        env.step(np.zeros(3, dtype=np.float32))


# ── 거래비용 ──────────────────────────────────────────────────────────

def test_zero_turnover_yields_zero_cost():
    """action이 직전 비중과 동일하면(w_new=w_prev) 거래비용은 0."""
    env = _mk_env()
    env.reset()  # w_prev = SHV 100%
    # softmax([-큰수]*4 + [+큰수]) ≈ [0,0,0,0,1] = w_prev
    logits = np.array([-1e6, -1e6, -1e6, -1e6, 100.0], dtype=np.float32)
    _, _, _, _, info = env.step(logits)
    assert info["turnover"] == pytest.approx(0.0, abs=1e-4)
    assert info["cost"] == pytest.approx(0.0, abs=1e-6)


def test_full_switch_turnover_is_2():
    """SHV 100% → SPY 100%는 총 회전율 2 (매도1 + 매수1)."""
    W = 30
    s = _fake_state_df(W)
    r = _zero_returns_df(s)  # zero returns → reward = -cost
    env = PortfolioEnv(s, r, cfg=_fake_cfg(W, c=0.01))
    env.reset()
    logits = np.array([100.0, -1e6, -1e6, -1e6, -1e6], dtype=np.float32)  # SPY 100%
    _, reward, _, _, info = env.step(logits)
    assert info["turnover"] == pytest.approx(2.0, abs=1e-4)
    # reward = 0 - 0.01*2 = -0.02
    assert reward == pytest.approx(-0.02, abs=1e-6)


def test_transaction_cost_scales_linearly_with_c():
    """c를 10배 올리면 동일 로짓·zero-return 조건에서 비용도 10배."""
    W = 30
    s = _fake_state_df(W)
    r = _zero_returns_df(s)
    logits = np.array([1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float32)
    env1 = PortfolioEnv(s, r, cfg=_fake_cfg(W, c=0.001))
    env2 = PortfolioEnv(s, r, cfg=_fake_cfg(W, c=0.010))
    env1.reset()
    env2.reset()
    _, _, _, _, info1 = env1.step(logits)
    _, _, _, _, info2 = env2.step(logits)
    assert info2["cost"] == pytest.approx(10 * info1["cost"], rel=1e-5)


# ── 보상 계산 ─────────────────────────────────────────────────────────

def test_reward_matches_manual_calculation():
    """수동 계산과 보상이 정확히 일치 — 룩어헤드 없음(다음 기간 returns 사용) 포함 검증."""
    W = 30
    n = 5
    s = _fake_state_df(W, n_rows=n)
    # 결정적 returns: r[t] = 0.01 * t (모든 자산 동일)
    returns_arr = np.tile(
        np.arange(n, dtype=np.float32).reshape(-1, 1) * 0.01, (1, 5)
    )
    r = pd.DataFrame(returns_arr, index=s.index, columns=ASSETS)
    env = PortfolioEnv(s, r, cfg=_fake_cfg(W, c=0.001))
    env.reset()
    logits = np.array([1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float32)  # 균등 0.2씩
    _, reward, _, _, info = env.step(logits)

    w_new = _softmax(logits)  # [0.2]*5
    r_next = returns_arr[1]   # t=1 시점 실현: 0.01 (룩어헤드 없음 — t=0 아님)
    expected_ret = float(np.dot(w_new, r_next))
    # turnover: |0.2-0|*4 + |0.2-1| = 0.8 + 0.8 = 1.6
    expected_turnover = 4 * 0.2 + 0.8
    expected_cost = 0.001 * expected_turnover

    assert info["portfolio_return"] == pytest.approx(expected_ret, abs=1e-6)
    assert info["turnover"] == pytest.approx(expected_turnover, abs=1e-5)
    assert reward == pytest.approx(expected_ret - expected_cost, abs=1e-6)


def test_reward_reduces_to_negative_cost_under_zero_returns():
    """수익률이 0이면 reward = -cost. 과매매를 억제하는지 확인."""
    W = 30
    s = _fake_state_df(W)
    r = _zero_returns_df(s)
    env = PortfolioEnv(s, r, cfg=_fake_cfg(W, c=0.001))
    env.reset()
    logits = np.array([2.0, -1.0, 0.5, 0.0, -0.5], dtype=np.float32)
    _, reward, _, _, info = env.step(logits)
    assert info["portfolio_return"] == pytest.approx(0.0, abs=1e-6)
    assert reward == pytest.approx(-info["cost"], abs=1e-6)
    assert reward < 0  # 회전이 있으면 페널티


# ── 에피소드 진행 ──────────────────────────────────────────────────────

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
