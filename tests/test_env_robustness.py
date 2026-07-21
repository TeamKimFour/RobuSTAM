"""환경 견고성 테스트 — 과매매 페널티·룩어헤드 방어 (4주차 산출물).

test_env.py / test_discrete_env.py가 계약 준수(shape·스키마·reward 위임)를
검증한다면, 이 파일은 "환경이 잘못된 학습을 유도하지 않는가"를 검증한다.
    - 과매매 페널티: turnover가 커지면 cost·reward가 정확히 그만큼 손해로 반영
    - 룩어헤드 방어: 어느 스텝에서도 미래 시점 데이터를 훔쳐보지 않음

원본 PortfolioEnv와 DQN용 DiscretePortfolioEnv 둘 다 검증한다. 이산 어댑터는
log→softmax 왕복으로 ~1e-8 수준의 부동소수 드리프트가 있어 tolerance를 소폭
완화한다(계약은 이론값과 동일, 수치만 근사).

역사적으로 "한 칸 밀림"(fix 커밋 090b573)이 발견됐던 지점이라, 여러 각도에서
재발 방지 테스트를 촘촘히 배치한다.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data import schema

pytest.importorskip("gymnasium")

from src.env.discrete_env import DiscretePortfolioEnv  # noqa: E402
from src.env.portfolio_env import PortfolioEnv, _softmax  # noqa: E402

HOLD_ACTION = 8  # DiscretePortfolioEnv: 2·A_tr = 8 (유지)

ASSETS = ["SPY", "EWY", "TLT", "GLD", "SHV"]
FWD_RET_COLS = [f"fwd_ret_{a}" for a in ASSETS]


def _cfg(window: int = 30, c: float = 0.001) -> dict:
    return {"assets": ASSETS, "window": window, "transaction_cost": c}


def _state_df(window: int, n_rows: int = 10) -> pd.DataFrame:
    cols = schema.feature_names(window)
    dates = pd.date_range("2020-01-01", periods=n_rows, freq="B")
    # 각 (t, col)에 고유값을 넣어 어느 행을 읽었는지 역추적 가능하게 한다.
    data = np.arange(n_rows * len(cols), dtype=np.float32).reshape(n_rows, len(cols))
    return pd.DataFrame(data, index=dates, columns=cols)


def _zero_returns_df(state_df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        np.zeros((len(state_df), 5), dtype=np.float32),
        index=state_df.index,
        columns=FWD_RET_COLS,
    )


# ── 과매매 페널티 (turnover × c 정확도) ─────────────────────────────

def test_hold_action_yields_zero_turnover_and_zero_cost():
    """같은 비중을 유지하면 turnover=0, cost=0, reward=포트폴리오 수익."""
    W = 30
    s = _state_df(W)
    r = _zero_returns_df(s)
    env = PortfolioEnv(s, r, cfg=_cfg(W, c=0.001))
    env.reset()  # SHV 100% 출발
    # 극단 로짓으로 SHV 100%를 유지 → 이전 비중과 정확히 동일
    logits = np.array([-1e6, -1e6, -1e6, -1e6, 1e6], dtype=np.float64)
    _, reward, _, _, info = env.step(logits)
    assert info["turnover"] == pytest.approx(0.0, abs=1e-6)
    assert info["cost"] == pytest.approx(0.0, abs=1e-9)
    assert reward == pytest.approx(0.0, abs=1e-9)


def test_full_swap_yields_cost_2c():
    """SHV 100% → SPY 100% 완전 스왑: turnover=2, cost=2c."""
    W = 30
    c = 0.001
    s = _state_df(W)
    r = _zero_returns_df(s)
    env = PortfolioEnv(s, r, cfg=_cfg(W, c))
    env.reset()
    # SPY에 몰빵 (softmax [1e6,-1e6,...] ≈ [1,0,0,0,0])
    logits = np.array([1e6, -1e6, -1e6, -1e6, -1e6], dtype=np.float64)
    _, reward, _, _, info = env.step(logits)
    assert info["turnover"] == pytest.approx(2.0, abs=1e-5)
    assert info["cost"] == pytest.approx(2 * c, abs=1e-8)
    assert reward == pytest.approx(-2 * c, abs=1e-8)


def test_cost_scales_linearly_with_turnover():
    """turnover가 2배면 cost·reward 페널티도 정확히 2배."""
    W = 30
    c = 0.001
    s = _state_df(W)
    r = _zero_returns_df(s)

    # 부분 스왑 (SHV 100% → SPY 50% + SHV 50%): turnover = 0.5+0.5 = 1
    env1 = PortfolioEnv(s, r, cfg=_cfg(W, c))
    env1.reset()
    logits_half = np.log(np.array([0.5, 1e-9, 1e-9, 1e-9, 0.5]))  # softmax ≈ [0.5,0,0,0,0.5]
    _, _, _, _, info1 = env1.step(logits_half)

    # 완전 스왑 (SHV → SPY): turnover = 2
    env2 = PortfolioEnv(s, r, cfg=_cfg(W, c))
    env2.reset()
    logits_full = np.array([1e6, -1e6, -1e6, -1e6, -1e6], dtype=np.float64)
    _, _, _, _, info2 = env2.step(logits_full)

    assert info2["turnover"] == pytest.approx(2 * info1["turnover"], rel=1e-4)
    assert info2["cost"] == pytest.approx(2 * info1["cost"], rel=1e-4)


def test_cumulative_cost_matches_per_step_theory_over_episode():
    """N번 완전 스왑을 반복하면 누적 cost = N × 2c 와 정확히 일치."""
    W = 30
    n_rows = 8
    c = 0.001
    s = _state_df(W, n_rows=n_rows)
    r = _zero_returns_df(s)
    env = PortfolioEnv(s, r, cfg=_cfg(W, c))
    env.reset()

    total_cost = 0.0
    steps = 0
    # SPY ↔ SHV 를 매 스텝 완전 스왑
    for t in range(n_rows - 1):
        if t % 2 == 0:
            logits = np.array([1e6, -1e6, -1e6, -1e6, -1e6], dtype=np.float64)
        else:
            logits = np.array([-1e6, -1e6, -1e6, -1e6, 1e6], dtype=np.float64)
        _, _, terminated, _, info = env.step(logits)
        total_cost += info["cost"]
        steps += 1
        if terminated:
            break

    # 첫 스텝은 SHV→SPY(turnover=2), 이후 매 스텝 완전 스왑(turnover=2) → cost=2c 균일
    assert total_cost == pytest.approx(steps * 2 * c, abs=1e-6)


# ── 룩어헤드 방어 (multi-step, obs, 회귀 다각화) ─────────────────────

def test_each_step_uses_fwd_ret_at_current_index_only():
    """모든 t에서 log_return이 정확히 returns_df.iloc[t]와 일치.

    회귀 방지: test_env.py::test_first_step_uses_returns_at_index_0_not_1 는
    첫 스텝만 검증한다. 이 테스트는 여러 스텝을 반복해 어느 지점에서도
    한 칸 밀림 사고가 재발하지 않는지 확인한다.
    """
    W = 30
    n = 6
    s = _state_df(W, n_rows=n)
    # 각 t 위치에 고유 값 → info["log_return"]으로 어느 인덱스에서 읽었는지 역추적
    r_arr = np.zeros((n, 5), dtype=np.float64)
    for t in range(n):
        r_arr[t] = (t + 1) * 0.001  # 0.001, 0.002, ..., 0.006
    r = pd.DataFrame(r_arr, index=s.index, columns=FWD_RET_COLS)
    env = PortfolioEnv(s, r, cfg=_cfg(W, c=0.0))
    env.reset()

    # 균등 비중 유지: portfolio_arith = expm1(r_log), log_return = log(1+expm1(r_log)) = r_log
    logits = np.array([1.0] * 5, dtype=np.float64)
    for t in range(n - 1):
        _, _, terminated, _, info = env.step(logits)
        expected = (t + 1) * 0.001
        assert info["log_return"] == pytest.approx(expected, abs=1e-9), (
            f"t={t}: log_return이 index {t}의 값 {expected}와 다름 → 룩어헤드 회귀 의심"
        )
        if terminated:
            break


def test_observation_carries_only_current_row_state():
    """obs[t]의 정적 부분(직전 비중 슬라이스 제외)이 state_df.iloc[t]와 정확히 일치.

    회귀 방지: env가 실수로 state_df.iloc[t+1]이나 t-1을 관측에 노출하면 실패.
    직전 비중 슬라이스는 런타임 값이므로 비교에서 제외한다.
    """
    W = 30
    n = 6
    s = _state_df(W, n_rows=n)
    r = _zero_returns_df(s)
    env = PortfolioEnv(s, r, cfg=_cfg(W))
    prev_w_slice = schema.prev_weight_slice(W)

    obs, _ = env.reset()
    # reset 직후 obs는 state_df.iloc[0]
    row0 = s.iloc[0].to_numpy(dtype=np.float32).copy()
    for i in range(len(row0)):
        if prev_w_slice.start <= i < prev_w_slice.stop:
            continue
        assert obs[i] == pytest.approx(row0[i]), f"reset: obs[{i}] != state_df.iloc[0][{i}]"

    logits = np.array([1.0] * 5, dtype=np.float64)
    for t in range(1, n - 1):
        obs, _, terminated, _, _ = env.step(logits)
        expected_row = s.iloc[t].to_numpy(dtype=np.float32)
        for i in range(len(expected_row)):
            if prev_w_slice.start <= i < prev_w_slice.stop:
                continue
            assert obs[i] == pytest.approx(expected_row[i]), (
                f"t={t}: obs[{i}]={obs[i]} != state_df.iloc[{t}][{i}]={expected_row[i]}"
                " → 관측이 미래·과거 행을 참조하는 룩어헤드 의심"
            )
        if terminated:
            break


def test_reordering_returns_changes_reward_exactly_as_expected():
    """returns_df를 시프트하면 reward가 정확히 시프트된 값으로 바뀐다.

    회귀 방지: 만약 env가 하드코딩된 오프셋(예: iloc[t+1])을 쓴다면 시프트가
    예상대로 반영되지 않는다. 순수한 룩어헤드 계약(iloc[t]) 준수 여부를
    행동/시간 대신 데이터 축을 뒤흔들어 교차검증한다.
    """
    W = 30
    n = 4
    s = _state_df(W, n_rows=n)
    r_base = np.zeros((n, 5), dtype=np.float64)
    r_base[0] = 0.02
    r_base[1] = 0.05
    r_base[2] = 0.10
    r_shifted = np.roll(r_base, shift=1, axis=0)  # 값이 한 칸 뒤로 이동

    r_a = pd.DataFrame(r_base, index=s.index, columns=FWD_RET_COLS)
    r_b = pd.DataFrame(r_shifted, index=s.index, columns=FWD_RET_COLS)

    logits = np.array([1.0] * 5, dtype=np.float64)

    env_a = PortfolioEnv(s, r_a, cfg=_cfg(W, c=0.0))
    env_a.reset()
    _, _, _, _, info_a0 = env_a.step(logits)  # 기대: r_base[0] = 0.02

    env_b = PortfolioEnv(s, r_b, cfg=_cfg(W, c=0.0))
    env_b.reset()
    _, _, _, _, info_b0 = env_b.step(logits)  # 기대: r_shifted[0] = r_base[-1] = 0

    assert info_a0["log_return"] == pytest.approx(0.02, abs=1e-9)
    assert info_b0["log_return"] == pytest.approx(0.0, abs=1e-9)


# ── 인터랙션: 과매매 페널티가 실제 수익과 결합되어도 정합 ───────────

def _mk_discrete(delta: float = 0.1, c: float = 0.001, n_rows: int = 10, r_arr=None):
    W = 30
    s = _state_df(W, n_rows=n_rows)
    if r_arr is None:
        r = _zero_returns_df(s)
    else:
        r = pd.DataFrame(r_arr, index=s.index, columns=FWD_RET_COLS)
    return DiscretePortfolioEnv(PortfolioEnv(s, r, cfg=_cfg(W, c)), delta=delta)


# ── DiscretePortfolioEnv 견고성 (과매매·룩어헤드 통과 검증) ──────────

def test_discrete_hold_yields_zero_turnover_zero_cost():
    """유지 action(=8) → turnover≈0, cost≈0, reward≈0 (r=0).

    log→softmax 왕복이 있어도 유지는 그대로 SHV=1 근방을 지킨다.
    """
    env = _mk_discrete(delta=0.1, c=0.001)
    env.reset()  # SHV 100%
    _, reward, _, _, info = env.step(HOLD_ACTION)
    assert info["turnover"] == pytest.approx(0.0, abs=1e-6)
    assert info["cost"] == pytest.approx(0.0, abs=1e-9)
    assert reward == pytest.approx(0.0, abs=1e-9)


def test_discrete_plus_delta_turnover_equals_2delta():
    """+Δ action → SHV(1)→자산(Δ)+SHV(1-Δ). turnover=2Δ, cost=2Δc, reward=-2Δc."""
    delta = 0.1
    c = 0.001
    env = _mk_discrete(delta=delta, c=c)
    env.reset()
    _, reward, _, _, info = env.step(0)  # +Δ SPY (SHV→SPY)
    # 이론값: |Δ - 0| + 3·|~0 - 0| + |(1-Δ) - 1| = 2Δ
    assert info["turnover"] == pytest.approx(2 * delta, abs=1e-4)
    assert info["cost"] == pytest.approx(2 * delta * c, abs=1e-7)
    assert reward == pytest.approx(-2 * delta * c, abs=1e-7)


def test_discrete_cumulative_cost_matches_delta_theory():
    """+Δ와 -Δ 반복 완전 스왑 → 각 스텝 turnover≈2Δ → 누적 cost = N·2Δc."""
    delta = 0.1
    c = 0.001
    n_rows = 8
    env = _mk_discrete(delta=delta, c=c, n_rows=n_rows)
    env.reset()
    total_cost = 0.0
    steps = 0
    for t in range(n_rows - 1):
        action = 0 if t % 2 == 0 else 4  # +Δ SPY ↔ -Δ SPY
        _, _, terminated, _, info = env.step(action)
        total_cost += info["cost"]
        steps += 1
        if terminated:
            break
    assert total_cost == pytest.approx(steps * 2 * delta * c, rel=1e-3)


def test_discrete_lookahead_via_hold_action():
    """이산 어댑터를 거쳐도 각 t가 returns_df.iloc[t]만 참조.

    유지 action으로 SHV≈1을 유지하면 (자산별 수익률이 같을 때) log_return이
    정확히 r_log[t]와 일치. 어댑터가 실수로 원본 env의 시간 인덱스를 건드리면
    이 테스트가 실패한다.
    """
    n = 6
    r_arr = np.zeros((n, 5), dtype=np.float64)
    for t in range(n):
        r_arr[t] = (t + 1) * 0.001  # 각 t 고유값 (자산별로는 동일 → 비중과 무관)
    env = _mk_discrete(delta=0.1, c=0.0, n_rows=n, r_arr=r_arr)
    env.reset()
    for t in range(n - 1):
        _, _, terminated, _, info = env.step(HOLD_ACTION)
        expected = (t + 1) * 0.001
        assert info["log_return"] == pytest.approx(expected, abs=1e-6), (
            f"discrete t={t}: 룩어헤드 회귀 의심 (expected {expected}, got {info['log_return']})"
        )
        if terminated:
            break


# ── 원본 env: reward 항등식 (인터랙션 검증) ─────────────────────────

def test_reward_equals_log_portfolio_minus_cost_across_multiple_steps():
    """매 스텝 reward = log_return - cost 관계가 여러 스텝 동안 정확히 유지."""
    W = 30
    n = 5
    c = 0.0025
    s = _state_df(W, n_rows=n)
    rng = np.random.default_rng(42)
    r_arr = rng.normal(0.0, 0.01, size=(n, 5)).astype(np.float64)
    r = pd.DataFrame(r_arr, index=s.index, columns=FWD_RET_COLS)
    env = PortfolioEnv(s, r, cfg=_cfg(W, c))
    env.reset()

    logits_seq = [
        np.array([1.0, 1.0, 1.0, 1.0, 1.0]),   # 균등
        np.array([2.0, -1.0, 0.5, 0.0, 1.5]),  # 이동
        np.array([-1.0, 3.0, 0.0, 2.0, -1.0]), # 재이동
        np.array([0.0, 0.0, 0.0, 0.0, 0.0]),   # 균등 복귀
    ]
    for logits in logits_seq:
        _, reward, terminated, _, info = env.step(logits.astype(np.float64))
        # 정의상 reward = log_return - cost
        assert reward == pytest.approx(info["log_return"] - info["cost"], abs=1e-12)
        # cost >= 0
        assert info["cost"] >= 0.0
        # turnover ∈ [0, 2]
        assert 0.0 - 1e-9 <= info["turnover"] <= 2.0 + 1e-9
        if terminated:
            break
