"""PortfolioEnv 테스트 — 관측/행동 shape, Softmax, 도현/민지 API 정합.

보상 수식 자체의 검증은 tests/test_reward.py가 담당한다.
여기서는 env가 어댑터 역할(로그→산술 변환, 컬럼 스키마, reward 위임)을 정확히
수행하는지에 집중한다.
"""

import math

import numpy as np
import pandas as pd
import pytest

from src.data import schema
from src.reward import calculate_reward

pytest.importorskip("gymnasium")

from src.env.portfolio_env import PortfolioEnv, _softmax  # noqa: E402

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


def _zero_log_returns_df(state_df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        np.zeros((len(state_df), 5), dtype=np.float32),
        index=state_df.index,
        columns=FWD_RET_COLS,
    )


def _mk_env(W: int = 30, n_rows: int = 10, c: float = 0.001) -> PortfolioEnv:
    s = _fake_state_df(W, n_rows)
    r = _fake_log_returns_df(s)
    return PortfolioEnv(s, r, cfg=_fake_cfg(W, c))


# ── 관측/행동 shape (state_spec §5③ 준수) ─────────────────────────────

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


# ── reset (초기 비중 정책: SHV 100%) ──────────────────────────────────

def test_reset_returns_obs_and_empty_info():
    env = _mk_env()
    obs, info = env.reset()
    assert obs.shape == (187,)
    assert obs.dtype == np.float32
    assert info == {}


def test_reset_initial_weight_is_shv_100pct():
    """팀 회의 확정 옵션 a: 무위험 자산 100%에서 출발."""
    W = 30
    env = _mk_env(W=W)
    obs, _ = env.reset()
    prev_w = obs[schema.prev_weight_slice(W)]
    expected = np.array([0.0, 0.0, 0.0, 0.0, 1.0], dtype=np.float32)
    np.testing.assert_array_equal(prev_w, expected)


# ── 생성자 계약 검증 ──────────────────────────────────────────────────

def test_state_column_schema_mismatch_raises():
    W = 30
    s = _fake_state_df(W)
    r = _fake_log_returns_df(s)
    bad = s.rename(columns={s.columns[0]: "wrong"})
    with pytest.raises(ValueError, match="schema.feature_names"):
        PortfolioEnv(bad, r, cfg=_fake_cfg(W))


def test_returns_df_must_use_fwd_ret_prefix():
    """민지 feature_store.load_targets 출력과 정확히 정합."""
    W = 30
    s = _fake_state_df(W)
    bad = _fake_log_returns_df(s).rename(columns={"fwd_ret_SPY": "SPY"})
    with pytest.raises(ValueError, match="fwd_ret_"):
        PortfolioEnv(s, bad, cfg=_fake_cfg(W))


def test_returns_df_column_order_must_match_assets():
    W = 30
    s = _fake_state_df(W)
    bad = _fake_log_returns_df(s)[[FWD_RET_COLS[1], FWD_RET_COLS[0], *FWD_RET_COLS[2:]]]
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


# ── Softmax ─────────────────────────────────────────────────────────

def test_softmax_output_sums_to_1_and_non_negative():
    for logits in [
        np.array([1.0, 1.0, 1.0, 1.0, 1.0]),
        np.array([-2.0, 3.0, 0.0, 1.5, -0.5]),
        np.array([1e5, 1e5, 1e5, 1e5, 1e5 + 1.0]),  # 오버플로 위험
    ]:
        w = _softmax(logits.astype(np.float64))
        assert np.isclose(w.sum(), 1.0, atol=1e-9)
        assert (w >= 0).all()
        assert np.isfinite(w).all()


def test_step_updates_prev_weight_block_with_softmax_output():
    W = 30
    env = _mk_env(W=W)
    env.reset()
    logits = np.array([1.0, -1.0, 0.5, 0.0, 2.0], dtype=np.float64)
    obs, _, _, _, info = env.step(logits)
    w_expected = _softmax(logits)
    np.testing.assert_allclose(info["weights"], w_expected, atol=1e-6)
    np.testing.assert_allclose(
        obs[schema.prev_weight_slice(W)], w_expected.astype(np.float32), atol=1e-6
    )


def test_action_wrong_shape_raises():
    env = _mk_env()
    env.reset()
    with pytest.raises(ValueError, match="action shape"):
        env.step(np.zeros(3, dtype=np.float64))


# ── 도현 reward 위임 & 선택지 α 변환 검증 ────────────────────────────

def test_reward_delegates_to_calculate_reward_with_arithmetic_returns():
    """env 보상 = calculate_reward(prev, w_new, expm1(r_log), c) 와 정확히 일치.

    이 테스트가 통과하면 env가 (1) reward 계산을 직접 하지 않고 (2) 로그→산술
    변환을 정확히 수행하며 (3) 민지 fwd_ret 규약(index t=next-day)을 정확히 따른다는
    세 계약을 모두 만족한다는 뜻이다.
    """
    W = 30
    n = 5
    s = _fake_state_df(W, n_rows=n)
    # 결정적 log returns (민지 규약: fwd_ret[t] = log_ret[t+1] 저장): r_log[t] = 0.01 * (t+1)
    r_log_arr = np.tile(
        (np.arange(n, dtype=np.float64) + 1.0).reshape(-1, 1) * 0.01, (1, 5)
    )
    r = pd.DataFrame(r_log_arr, index=s.index, columns=FWD_RET_COLS)
    env = PortfolioEnv(s, r, cfg=_fake_cfg(W, c=0.001))
    env.reset()
    logits = np.array([1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float64)
    _, reward, _, _, info = env.step(logits)

    w_new = _softmax(logits)  # [0.2]*5
    w_prev = np.array([0.0, 0.0, 0.0, 0.0, 1.0])
    # 민지 fwd_ret 규약: env는 t=0 위치 값을 그대로 사용 (이미 next-day)
    r_arith_next = np.expm1(r_log_arr[0])
    expected = calculate_reward(w_prev, w_new, r_arith_next, transaction_cost_rate=0.001)
    assert reward == pytest.approx(expected, abs=1e-12)


def test_first_step_uses_returns_at_index_0_not_1():
    """민지 fwd_ret 규약 회귀 방지 — 한 칸 밀림 사고 재발 방지.

    fwd_ret[t]가 이미 t→t+1 수익률이므로 첫 스텝은 index 0을 써야 한다.
    만약 코드가 예전처럼 index 1을 참조하면 이 테스트가 실패한다.
    """
    W = 30
    n = 4
    s = _fake_state_df(W, n_rows=n)
    # 각 자산 t 위치를 고유값으로 → 어느 인덱스에서 읽었는지 역추적 가능
    r_log_arr = np.zeros((n, 5), dtype=np.float64)
    r_log_arr[0] = 0.02   # 이 값이 첫 step의 log_return이어야 함
    r_log_arr[1] = 0.99   # 만약 index 1을 참조했다면 log_return이 log(1+expm1(0.99))=0.99가 됨
    r = pd.DataFrame(r_log_arr, index=s.index, columns=FWD_RET_COLS)
    env = PortfolioEnv(s, r, cfg=_fake_cfg(W, c=0.0))
    env.reset()
    # 균등 로짓 → 균등 비중 → log_return = log(1 + expm1(r_log[?])) = r_log[?]
    _, _, _, _, info = env.step(np.array([1.0] * 5, dtype=np.float64))
    assert info["log_return"] == pytest.approx(0.02, abs=1e-9)


def test_log_return_conversion_not_identity():
    """선택지 α가 실제로 적용됐는지 — 로그 그대로 넘겼을 경우와 다른 값이 나온다."""
    W = 30
    n = 3
    s = _fake_state_df(W, n_rows=n)
    # 큰 수익률 (근사 오차가 커지도록)
    r_log = np.full((n, 5), 0.10, dtype=np.float64)
    r = pd.DataFrame(r_log, index=s.index, columns=FWD_RET_COLS)
    env = PortfolioEnv(s, r, cfg=_fake_cfg(W, c=0.0))
    env.reset()
    logits = np.array([1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float64)
    _, reward, _, _, info = env.step(logits)

    # 산술수익률 변환: exp(0.10)-1 ≈ 0.10517
    # 포트폴리오 산술: 0.2*0.10517*5 = 0.10517
    # 로그수익률: log(1+0.10517) = 0.10 ← 원래 로그와 같아야 정합
    assert info["log_return"] == pytest.approx(0.10, abs=1e-9)
    # 만약 변환 안 했다면 log(1+0.10) = 0.09531 이 됐을 것 → 다른 값
    assert reward == pytest.approx(0.10, abs=1e-9)


def test_zero_log_returns_yield_negative_cost_only():
    """log=0 → arith=0. 회전 있으면 reward = -cost."""
    W = 30
    s = _fake_state_df(W)
    r = _zero_log_returns_df(s)
    env = PortfolioEnv(s, r, cfg=_fake_cfg(W, c=0.001))
    env.reset()
    logits = np.array([1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float64)
    _, reward, _, _, info = env.step(logits)
    assert info["portfolio_return"] == pytest.approx(0.0, abs=1e-12)
    assert info["log_return"] == pytest.approx(0.0, abs=1e-12)
    assert reward == pytest.approx(-info["cost"], abs=1e-12)
    assert reward < 0


def test_info_dict_matches_calculate_reward_verbose_shape():
    """info dict 키가 도현 verbose 규격과 일치."""
    env = _mk_env()
    env.reset()
    _, _, _, _, info = env.step(np.zeros(5, dtype=np.float64))
    for key in ("portfolio_return", "log_return", "turnover", "cost", "weights"):
        assert key in info, f"info missing '{key}'"


# ── 에피소드 진행 ──────────────────────────────────────────────────────

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
