"""RobuSTAM 보상 함수(Reward Function) 모듈.

이 모듈은 PortfolioEnv와 독립적으로 설계되어 있어 env 없이도 단독 검증 가능하다.
env는 3가지 정보(prev_weights, new_weights, asset_returns)를 넘겨주는 어댑터 역할만 하고,
실제 보상 계산은 이 모듈이 담당한다. 계약 결정은 팀 회의 확정 사항이다.

수식 (계획서 §3-3 · CLAUDE.md §2 1차 버전):
    r_portfolio = Σ w_i,t · r_i,t            (산술 포트폴리오 수익률)
    r_log       = ln(1 + r_portfolio)        (로그수익률 변환 — 누적 시 합산 가능)
    cost_t      = c · Σ|w_i,t - w_i,t-1|     (L1 회전율 × 편도 거래비용)
    vol_pen_t   = κ · σ_recent               (위험조정 페널티 — 이슈 #34 개선안 D)
    R_t         = r_log - cost_t - vol_pen_t

핵심 계약:
    - vol_penalty_coef(κ) 기본 0.0 → 셰이핑 없음(기존 동작 보존). σ_recent(recent_vol)는
      최근 실현 포트폴리오 로그수익률의 표준편차로, **env가 롤링 윈도우로 계산해 넘긴다**
      (샤프는 에피소드 통계라 per-step 불가 → 롤링 변동성 근사, 이슈 #34 개선안 D).
      비용(c)과 동일하게 **학습 전용 셰이핑**이며 성과 측정 축은 건드리지 않는다.
    - asset_returns는 **산술수익률** (예: 1% 오르면 0.01).
      민지 targets.parquet은 로그수익률로 저장되므로 env가 exp변환(선택지 α)해서 넘겨준다.
    - transaction_cost_rate 기본값 0.001(편도 0.1%), 계획서 §2-1 정량목표.
    - 함수는 env·백테스트 엔진 어디서든 동일 결과가 나오도록 순수 함수로 유지한다.
"""

from __future__ import annotations

import numpy as np

# 계획서 §2-1: 편도 거래수수료 0.1% ~ 0.25% 범위, 확정값 0.1%
DEFAULT_TRANSACTION_COST_RATE = 0.001

# log(1+x) 안정성 가드: 이론상 포트폴리오 산술수익률이 -1 이하가 될 수 있어 클립한다
_MIN_PORTFOLIO_RETURN = -0.999999


def calculate_reward_verbose(
    prev_weights: np.ndarray,
    new_weights: np.ndarray,
    asset_returns: np.ndarray,
    transaction_cost_rate: float = DEFAULT_TRANSACTION_COST_RATE,
    vol_penalty_coef: float = 0.0,
    recent_vol: float = 0.0,
) -> dict:
    """한 스텝의 보상과 중간 계산값을 dict로 반환한다 (학습 진단·MLflow 로깅용).

    검증(shape 일치·비중 합≈1)과 계산이 이 함수에 모여 있어, env.step()이 이 함수를
    직접 호출해도 계약 위반을 조기에 잡을 수 있다. `calculate_reward`는 이 함수를
    감싸 스칼라 보상만 돌려주는 얇은 래퍼다.

    Parameters
    ----------
    prev_weights : np.ndarray, shape (n_assets,)
        직전 시점의 자산별 비중. 합이 1이어야 한다.
    new_weights : np.ndarray, shape (n_assets,)
        이번 시점 목표 비중 (정책망 출력에 Softmax 적용한 결과). 합이 1이어야 한다.
    asset_returns : np.ndarray, shape (n_assets,)
        이번 거래일의 자산별 **산술수익률** (예: 1% 상승 = 0.01).
    transaction_cost_rate : float, default=0.001
        편도 거래비용률. 계획서 기준 0.001 ~ 0.0025 사이 실험 예정.
    vol_penalty_coef : float, default=0.0
        위험조정 페널티 계수 κ (이슈 #34 개선안 D). 0.0이면 페널티 없음(기존 동작).
        음수는 거부한다.
    recent_vol : float, default=0.0
        최근 실현 포트폴리오 로그수익률의 표준편차 σ_recent. env가 롤링 윈도우로
        계산해 넘긴다. per-step 함수라 여기서는 스칼라로 받아 κ만 곱한다.

    Returns
    -------
    dict
        키: portfolio_return · log_return · turnover · transaction_cost ·
        vol_penalty · reward.

    Raises
    ------
    ValueError
        입력 배열 shape이 서로 다르거나, 비중의 합이 1에서 크게 벗어나거나,
        vol_penalty_coef가 음수인 경우.
    """
    if vol_penalty_coef < 0:
        raise ValueError(f"vol_penalty_coef(κ)는 음수일 수 없습니다: {vol_penalty_coef}")
    prev_weights = np.asarray(prev_weights, dtype=np.float64)
    new_weights = np.asarray(new_weights, dtype=np.float64)
    asset_returns = np.asarray(asset_returns, dtype=np.float64)

    if prev_weights.shape != new_weights.shape or prev_weights.shape != asset_returns.shape:
        raise ValueError(
            "shape이 모두 같아야 합니다. "
            f"prev_weights={prev_weights.shape}, new_weights={new_weights.shape}, "
            f"asset_returns={asset_returns.shape}"
        )
    for name, w in (("prev_weights", prev_weights), ("new_weights", new_weights)):
        total = float(w.sum())
        if not np.isclose(total, 1.0, atol=1e-3):
            raise ValueError(f"{name}의 합이 1이 아닙니다 (현재 합: {total:.6f})")

    # 1) 포트폴리오 산술 수익률
    portfolio_return = float(np.dot(new_weights, asset_returns))

    # 2) 로그수익률 변환 (log(1+x) 정의역 보호)
    portfolio_return_clipped = max(portfolio_return, _MIN_PORTFOLIO_RETURN)
    log_return = float(np.log1p(portfolio_return_clipped))

    # 3) 거래비용 페널티 = c × L1(회전율)
    turnover = float(np.abs(new_weights - prev_weights).sum())
    transaction_cost = transaction_cost_rate * turnover

    # 4) 위험조정 페널티 = κ × σ_recent (이슈 #34 개선안 D, κ=0이면 0)
    vol_penalty = vol_penalty_coef * recent_vol

    # 5) 최종 보상
    reward = log_return - transaction_cost - vol_penalty

    return {
        "portfolio_return": portfolio_return,
        "log_return": log_return,
        "turnover": turnover,
        "transaction_cost": transaction_cost,
        "vol_penalty": vol_penalty,
        "reward": reward,
    }


def calculate_reward(
    prev_weights: np.ndarray,
    new_weights: np.ndarray,
    asset_returns: np.ndarray,
    transaction_cost_rate: float = DEFAULT_TRANSACTION_COST_RATE,
    vol_penalty_coef: float = 0.0,
    recent_vol: float = 0.0,
) -> float:
    """한 스텝의 보상 R_t = ln(1 + w_new · r) − c · Σ|w_new − w_prev| − κ · σ_recent.

    검증·계산 로직은 `calculate_reward_verbose`에 모여 있으며, 이 함수는 스칼라
    보상만 필요한 호출자를 위한 얇은 래퍼다. `vol_penalty_coef`(κ) 기본 0.0이라
    미지정 호출부는 기존과 동일하게 R = log_return − cost만 받는다.
    """
    return calculate_reward_verbose(
        prev_weights=prev_weights,
        new_weights=new_weights,
        asset_returns=asset_returns,
        transaction_cost_rate=transaction_cost_rate,
        vol_penalty_coef=vol_penalty_coef,
        recent_vol=recent_vol,
    )["reward"]
