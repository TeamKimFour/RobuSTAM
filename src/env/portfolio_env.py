"""자산 5종 데일리 리밸런싱 커스텀 Gymnasium 환경.

관측/행동 공간·reset()·step()을 정의한다. 실제 보상 계산은 이 클래스가 하지 않고
`src.reward.calculate_reward_verbose`에 위임한다(팀 회의 확정: env와 reward 모듈은 분리).
verbose 버전을 쓰는 것은 스텝별 중간값(portfolio_return·turnover·cost)을 info로
반환해 학습 진단·MLflow 로깅에 활용하기 위한 것이고, 입력 검증도 verbose 쪽에 통합돼 있다.

인터페이스 계약 (docs/state_spec.md §3 · CLAUDE.md §2):
    관측 shape = (state_dim,), state_dim = config.window에서 산출 (W=30 → 187).
    관측 마지막 A칸([182:187] @ W=30)은 이 클래스가 런타임에 직전 비중으로 채운다.
    행동 shape = (A,) — 정책 신경망 로짓. step() 내부에서 Softmax로 합=1 정규화.

데이터 계약 (팀 회의 확정):
    - state_df 컬럼 = `schema.feature_names(W)` (민지 `assemble.assemble_state_matrix` 출력).
    - returns_df 컬럼 = `fwd_ret_<asset>` (민지 `feature_store.load_targets` 출력).
    - returns_df.iloc[t] = **fwd_ret[t] = log_ret[t+1]** (index t에 이미 t→t+1 기간 수익률
      저장 — 룩어헤드 방지는 민지 targets 저장 단계에서 처리됨).

리밸런싱·보상 모델:
    - 매 스텝 시작에 목표 비중으로 즉시 완전 리밸런싱 (기간 내 드리프트 무시).
    - **룩어헤드 방지:** returns_df.iloc[self._t]를 그대로 조회한다. fwd_ret 규약상
      index t 자체가 이미 next-day 수익률이므로 인덱스 조작 없이 사용한다.
    - **선택지 α (팀 회의 확정):** 민지 targets은 로그수익률로 저장되어 있으나 도현
      `calculate_reward`는 산술수익률을 기대한다. 이 어댑터 역할을 env가 담당해
      `r_arith = exp(r_log) - 1`로 변환 후 넘긴다.
    - 보상: `src.reward.calculate_reward_verbose(prev, new, r_arith, c)` (엔지니어 결정 완료).
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces

from src.config_loader import (
    get_assets,
    get_state_dim,
    get_transaction_cost,
    get_window,
    load_config,
)
from src.data import schema
from src.reward import calculate_reward_verbose


def _softmax(logits: np.ndarray) -> np.ndarray:
    """수치 안정 Softmax — 자산별 목표 비중으로 정규화 (∑w=1, CLAUDE.md §2)."""
    z = logits - logits.max()
    e = np.exp(z)
    return e / e.sum()


class PortfolioEnv(gym.Env):
    """5자산 목표 비중을 매 거래일 조절하는 포트폴리오 환경.

    Parameters
    ----------
    state_df : 조립된 wide State (`src.data.assemble.assemble_state_matrix` 출력).
        컬럼 순서는 `schema.feature_names(W)`와 정확히 일치해야 한다.
    returns_df : 익일 로그수익률 (`src.data.feature_store.load_targets` 출력).
        컬럼 = `fwd_ret_<asset>` 순서, index = state_df와 동일.
    cfg : config.yaml 사전. 생략 시 기본 경로에서 로드한다.
    cost_multiplier : 보상 계산에만 곱하는 거래비용 가중치 λ (기본 1.0 = 실비용 그대로).
        **학습 전용 셰이핑 장치**다(이슈 #34 개선안 A). 1.0이 아닌 값은 학습 env에서만 쓰고,
        평가·백테스트는 반드시 1.0(=`config.transaction_cost` 원값)을 유지해야 한다 —
        CLAUDE.md §2의 "편도 0.1%" 원칙은 성과 측정 축에 대한 것이므로, 여기를 건드리면
        벤치마크와 다른 자로 재는 셈이 된다.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        state_df: pd.DataFrame,
        returns_df: pd.DataFrame,
        cfg: dict | None = None,
        cost_multiplier: float = 1.0,
    ) -> None:
        super().__init__()
        self.cfg = cfg if cfg is not None else load_config()
        self.W = get_window(self.cfg)
        self.assets = get_assets(self.cfg)
        self.n_assets = len(self.assets)
        self.state_dim = get_state_dim(self.cfg)
        if cost_multiplier <= 0:
            raise ValueError(f"cost_multiplier는 양수여야 합니다: {cost_multiplier}")
        self.cost_multiplier = float(cost_multiplier)
        # 실비용 c_real은 그대로 보존하고, 보상에 쓰는 비율만 λ배 한다.
        self.c_real = get_transaction_cost(self.cfg)
        self.c = self.c_real * self.cost_multiplier

        expected_state_cols = schema.feature_names(self.W)
        if list(state_df.columns) != expected_state_cols:
            raise ValueError(
                "state_df 컬럼이 schema.feature_names(W)와 일치하지 않습니다. "
                "src.data.assemble.assemble_state_matrix 출력을 사용하세요."
            )

        expected_ret_cols = [f"fwd_ret_{a}" for a in self.assets]
        if list(returns_df.columns) != expected_ret_cols:
            raise ValueError(
                f"returns_df 컬럼이 fwd_ret_<asset> 순서와 다릅니다. "
                f"기대: {expected_ret_cols} — src.data.feature_store.load_targets 출력을 사용하세요."
            )
        if not returns_df.index.equals(state_df.index):
            raise ValueError("returns_df 인덱스가 state_df 인덱스와 일치해야 합니다.")
        if len(state_df) < 2:
            raise ValueError("state_df는 최소 2행 이상 필요합니다 (reset + 1 step).")

        self._state = state_df.reset_index(drop=True)
        # 민지 targets은 로그수익률(선택지 α: 여기서 산술수익률로 변환해 도현에 넘김)
        self._log_returns = returns_df.reset_index(drop=True).to_numpy(dtype=np.float64)
        self._prev_weight_slice = schema.prev_weight_slice(self.W)

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.state_dim,),
            dtype=np.float32,
        )
        self.action_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.n_assets,),
            dtype=np.float32,
        )

        self._t: int = 0
        self._current_weight = np.zeros(self.n_assets, dtype=np.float64)

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict | None = None,
    ) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        self._t = 0
        # 초기 비중: SHV(현금성) 100% — 무위험 상태 출발 (팀 회의 확정 옵션 a)
        self._current_weight = np.zeros(self.n_assets, dtype=np.float64)
        self._current_weight[self.assets.index("SHV")] = 1.0
        return self._obs(), {}

    def step(
        self,
        action: np.ndarray,
    ) -> tuple[np.ndarray, float, bool, bool, dict]:
        action = np.asarray(action, dtype=np.float64)
        if action.shape != (self.n_assets,):
            raise ValueError(
                f"action shape는 ({self.n_assets},) 이어야 합니다: {action.shape}"
            )

        # 1) Softmax로 목표 비중 확정 (∑w=1)
        w_new = _softmax(action)
        w_prev = self._current_weight

        # 2) 다음 기간(=t→t+1) 실현 수익률 조회
        #    민지 targets 규약: returns_df.iloc[t] = fwd_ret[t] = log_ret[t+1]
        #    → index t 자체가 이미 next-day 값이므로 인덱스 조작 없이 조회 (한 칸 밀림 사고 방지)
        r_log = self._log_returns[self._t]

        # 3) 선택지 α: 로그수익률 → 산술수익률 (도현 calculate_reward 계약)
        r_arith = np.expm1(r_log)

        # 4) 보상 계산은 src.reward 모듈에 위임 (env와 reward 분리 — 팀 회의 확정)
        detail = calculate_reward_verbose(
            prev_weights=w_prev,
            new_weights=w_new,
            asset_returns=r_arith,
            transaction_cost_rate=self.c,
        )
        reward = detail["reward"]

        # 5) 상태 업데이트 (완전 리밸런싱 가정) + 시간 인덱스 진행
        self._current_weight = w_new
        self._t += 1

        terminated = self._t >= len(self._state) - 1
        # info["cost"]는 항상 **실비용**(c_real × turnover)을 보고한다 — λ를 걸어도 진단·집계가
        # 실제로 나간 돈을 가리키게 하기 위함이다. 보상에 반영된 셰이핑 비용은 따로 노출한다.
        # (λ=1이면 둘이 같으므로 기존 호출부 동작은 그대로다.)
        info = {
            "portfolio_return": detail["portfolio_return"],
            "log_return": detail["log_return"],
            "turnover": detail["turnover"],
            "cost": detail["turnover"] * self.c_real,
            "shaped_cost": detail["transaction_cost"],
            "weights": w_new.astype(np.float32).copy(),
        }
        return self._obs(), reward, terminated, False, info

    def _obs(self) -> np.ndarray:
        row = self._state.iloc[self._t].to_numpy(dtype=np.float32).copy()
        row[self._prev_weight_slice] = self._current_weight.astype(np.float32)
        return row
