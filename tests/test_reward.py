"""src.reward 단위 테스트 — 도현 PDF 수식·계약 준수 검증.

계약:
    - asset_returns는 산술수익률 (예: 1% 상승 = 0.01)
    - R = ln(1 + Σ w_new·r) − c · Σ|w_new − w_prev|
    - 비중 합 ≠ 1 이면 ValueError, shape 불일치도 ValueError
    - portfolio_return이 -1 이하로 내려가면 log(0 이하) 방지를 위해 클립
"""

import math

import numpy as np
import pytest

from src.reward import (
    DEFAULT_TRANSACTION_COST_RATE,
    calculate_reward,
    calculate_reward_verbose,
)

ASSETS = ("SPY", "EWY", "TLT", "GLD", "SHV")
N = len(ASSETS)


def _uniform() -> np.ndarray:
    return np.full(N, 1.0 / N, dtype=np.float64)


def _shv_only() -> np.ndarray:
    w = np.zeros(N, dtype=np.float64)
    w[ASSETS.index("SHV")] = 1.0
    return w


# ── 수식 정합성 ────────────────────────────────────────────────────────

def test_default_transaction_cost_rate_is_0001():
    """CLAUDE.md §2 확정값 편도 0.1%."""
    assert DEFAULT_TRANSACTION_COST_RATE == 0.001


def test_reward_matches_analytical_formula():
    """R = ln(1 + w·r) − c·|Δw| 수동 계산과 일치."""
    prev = _shv_only()
    new = _uniform()  # w_new = [0.2]*5
    r = np.array([0.01, -0.005, 0.002, 0.0, 0.0001], dtype=np.float64)  # 산술수익률
    c = 0.001

    expected_portfolio = float(np.dot(new, r))              # 0.2*(0.01−0.005+0.002+0+0.0001)=0.00142
    expected_log = math.log1p(expected_portfolio)
    expected_turnover = float(np.abs(new - prev).sum())     # 0.2*4 + 0.8 = 1.6
    expected_cost = c * expected_turnover                   # 0.0016
    expected_reward = expected_log - expected_cost

    got = calculate_reward(prev, new, r, transaction_cost_rate=c)
    assert got == pytest.approx(expected_reward, abs=1e-12)


def test_zero_returns_and_no_turnover_yield_zero_reward():
    """수익률=0 이고 비중 변화도 없으면 R=0."""
    w = _uniform()
    r = np.zeros(N)
    assert calculate_reward(w, w, r, transaction_cost_rate=0.001) == pytest.approx(0.0, abs=1e-15)


def test_zero_returns_yield_negative_cost_only():
    """수익률=0 → R = −cost. 과매매 억제 계약(계획서 §3-3 함정 방지)."""
    prev = _shv_only()
    new = _uniform()
    r = np.zeros(N)
    reward = calculate_reward(prev, new, r, transaction_cost_rate=0.001)
    turnover = float(np.abs(new - prev).sum())  # 1.6
    assert reward == pytest.approx(-0.001 * turnover, abs=1e-15)
    assert reward < 0


def test_transaction_cost_scales_linearly_with_c():
    prev = _shv_only()
    new = _uniform()
    r = np.zeros(N)
    r1 = calculate_reward(prev, new, r, transaction_cost_rate=0.001)
    r2 = calculate_reward(prev, new, r, transaction_cost_rate=0.010)
    assert r2 == pytest.approx(10 * r1, rel=1e-12)


def test_full_switch_turnover_is_2():
    """SHV 100% → SPY 100% 완전 교체는 turnover=2 (매도1 + 매수1)."""
    prev = _shv_only()
    new = np.zeros(N)
    new[ASSETS.index("SPY")] = 1.0
    r = np.zeros(N)
    reward = calculate_reward(prev, new, r, transaction_cost_rate=0.01)
    # r=0 → R = -c*turnover = -0.01*2 = -0.02
    assert reward == pytest.approx(-0.02, abs=1e-12)


def test_log_return_is_not_simple_dot_product():
    """산술과 로그의 미세 차이가 결과에 반영된다 (log1p이 존재해야 통과)."""
    prev = _uniform()
    new = _uniform()  # turnover=0 → 비용 없음
    r = np.full(N, 0.10, dtype=np.float64)  # 각 자산 10% (근사 오차 큼)
    reward = calculate_reward(prev, new, r, transaction_cost_rate=0.0)

    dot = float(np.dot(new, r))                       # 0.10
    log_correct = math.log1p(dot)                     # ≈ 0.09531
    assert reward == pytest.approx(log_correct, abs=1e-12)
    assert reward < dot  # log(1+x) < x for x>0 (미세 차이지만 존재)


# ── 입력 검증 ─────────────────────────────────────────────────────────

def test_shape_mismatch_raises():
    with pytest.raises(ValueError, match="shape"):
        calculate_reward(np.zeros(4), np.zeros(5), np.zeros(5))


def test_prev_weight_sum_not_1_raises():
    prev = np.array([0.5, 0.5, 0.5, 0.5, 0.5])  # 합 2.5
    with pytest.raises(ValueError, match="prev_weights"):
        calculate_reward(prev, _uniform(), np.zeros(N))


def test_new_weight_sum_not_1_raises():
    new = np.array([0.5, 0.5, 0.5, 0.5, 0.5])  # 합 2.5
    with pytest.raises(ValueError, match="new_weights"):
        calculate_reward(_uniform(), new, np.zeros(N))


def test_weight_sum_within_tolerance_ok():
    """수치오차 허용 (atol=1e-3)."""
    w = _uniform().copy()
    w[0] += 5e-4  # 합 1.0005 — 허용
    r = np.zeros(N)
    calculate_reward(w, w, r)  # 통과해야 함


# ── 로그 안전성 ────────────────────────────────────────────────────────

def test_extreme_negative_portfolio_return_is_clipped():
    """비현실적 극단(모든 자산 -100% 초과) 상황에서도 log(음수) 방지."""
    prev = _uniform()
    new = _uniform()  # turnover=0
    r = np.full(N, -1.5, dtype=np.float64)  # 산술수익률 -150% (비현실이지만 안전 요구)
    reward = calculate_reward(prev, new, r, transaction_cost_rate=0.0)
    assert math.isfinite(reward)
    assert reward < 0


# ── verbose 버전 ──────────────────────────────────────────────────────

def test_verbose_returns_expected_keys_and_matches_scalar():
    prev = _shv_only()
    new = _uniform()
    r = np.array([0.01, -0.005, 0.002, 0.0, 0.0001], dtype=np.float64)
    c = 0.001

    d = calculate_reward_verbose(prev, new, r, transaction_cost_rate=c)
    assert set(d.keys()) == {
        "portfolio_return", "log_return", "turnover", "transaction_cost",
        "vol_penalty", "reward",
    }
    assert d["reward"] == pytest.approx(calculate_reward(prev, new, r, c), abs=1e-12)
    # κ 미지정(기본 0.0)이라 vol_penalty=0 → 기존 R = log_return − transaction_cost 그대로.
    assert d["vol_penalty"] == 0.0
    assert d["log_return"] - d["transaction_cost"] == pytest.approx(d["reward"], abs=1e-12)
    assert d["turnover"] == pytest.approx(float(np.abs(new - prev).sum()), abs=1e-12)
    assert d["transaction_cost"] == pytest.approx(c * d["turnover"], abs=1e-12)


# ── 위험조정 페널티 κ (이슈 #34 개선안 D) ──────────────────────────────

def test_vol_penalty_default_off_preserves_reward():
    """κ 미지정 시 기존 보상과 완전히 동일해야 한다(기본 0.0 = 셰이핑 없음)."""
    prev = _shv_only()
    new = _uniform()
    r = np.array([0.01, -0.005, 0.002, 0.0, 0.0001], dtype=np.float64)
    base = calculate_reward(prev, new, r, transaction_cost_rate=0.001)
    # recent_vol을 줘도 κ=0이면 페널티가 붙지 않는다.
    same = calculate_reward(prev, new, r, transaction_cost_rate=0.001, recent_vol=0.05)
    assert same == pytest.approx(base, abs=1e-15)


def test_vol_penalty_subtracts_kappa_times_sigma():
    """R = (log − cost) − κ·σ_recent. κ·σ만큼 정확히 깎여야 한다."""
    prev = _uniform()
    new = _uniform()  # turnover=0 → 비용 0, 순수 페널티 효과만 본다
    r = np.zeros(N)
    kappa, sigma = 3.0, 0.02
    d = calculate_reward_verbose(
        prev, new, r, transaction_cost_rate=0.001,
        vol_penalty_coef=kappa, recent_vol=sigma,
    )
    assert d["vol_penalty"] == pytest.approx(kappa * sigma, abs=1e-15)
    assert d["reward"] == pytest.approx(
        d["log_return"] - d["transaction_cost"] - kappa * sigma, abs=1e-15
    )


def test_vol_penalty_scales_linearly_with_kappa():
    prev = _uniform()
    new = _uniform()
    r = np.zeros(N)
    d1 = calculate_reward_verbose(prev, new, r, vol_penalty_coef=1.0, recent_vol=0.03)
    d2 = calculate_reward_verbose(prev, new, r, vol_penalty_coef=4.0, recent_vol=0.03)
    assert d2["vol_penalty"] == pytest.approx(4 * d1["vol_penalty"], rel=1e-12)


def test_negative_vol_penalty_coef_raises():
    with pytest.raises(ValueError, match="vol_penalty_coef"):
        calculate_reward_verbose(_uniform(), _uniform(), np.zeros(N), vol_penalty_coef=-0.1)


# ── verbose 경로도 동일한 검증을 강제 (env.step()의 실사용 경로) ─────────

def test_verbose_shape_mismatch_raises():
    with pytest.raises(ValueError, match="shape"):
        calculate_reward_verbose(np.zeros(4), np.zeros(5), np.zeros(5))


def test_verbose_prev_weight_sum_not_1_raises():
    prev = np.array([0.5, 0.5, 0.5, 0.5, 0.5])  # 합 2.5
    with pytest.raises(ValueError, match="prev_weights"):
        calculate_reward_verbose(prev, _uniform(), np.zeros(N))


def test_verbose_new_weight_sum_not_1_raises():
    new = np.array([0.5, 0.5, 0.5, 0.5, 0.5])  # 합 2.5
    with pytest.raises(ValueError, match="new_weights"):
        calculate_reward_verbose(_uniform(), new, np.zeros(N))
