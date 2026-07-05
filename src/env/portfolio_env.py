"""자산 5종 데일리 리밸런싱 커스텀 Gymnasium 환경.

관측/행동 공간·reset()·step() 본체를 정의한다. 슬리피지와 확장 보상(Differential
Sharpe 등)은 후속 PR에서 채운다.

인터페이스 계약 (docs/state_spec.md §3 인덱스맵):
    관측 shape = (state_dim,), state_dim = config.window에서 산출 (W=30 → 187)
    관측 마지막 A칸([182:187] @ W=30)은 이 클래스가 런타임에 직전 비중으로 채운다
    행동 shape = (A,) — 정책 신경망 로짓. step() 내부에서 Softmax로 합=1 정규화

리밸런싱·보상 모델 (CLAUDE.md §2, 계획서 §3-3 1차 버전):
    - 매 스텝 시작 시점에 목표 비중으로 즉시 완전 리밸런싱 (기간 내 드리프트 무시)
    - 거래비용 = c × Σ|w_new - w_prev|  (편도 c, 매수·매도 합계에 비례)
    - 보상 R_t = w_new · r_next - c × turnover
    - 룩어헤드 방지: r_next는 시간 인덱스를 증가시킨 뒤 조회한다
      (관측 시점 이후 실현되는 다음 기간 수익률만 사용).
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
    returns_df : 각 자산의 로그수익률 (index=state_df와 동일, 컬럼=config.assets 순서).
        `returns_df.iloc[t]` = t 시점에 실현된 로그수익률(t-1→t 기간).
    cfg : config.yaml 사전. 생략 시 기본 경로에서 로드한다.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        state_df: pd.DataFrame,
        returns_df: pd.DataFrame,
        cfg: dict | None = None,
    ) -> None:
        super().__init__()
        self.cfg = cfg if cfg is not None else load_config()
        self.W = get_window(self.cfg)
        self.assets = get_assets(self.cfg)
        self.n_assets = len(self.assets)
        self.state_dim = get_state_dim(self.cfg)
        self.c = get_transaction_cost(self.cfg)

        expected_cols = schema.feature_names(self.W)
        if list(state_df.columns) != expected_cols:
            raise ValueError(
                "state_df 컬럼이 schema.feature_names(W)와 일치하지 않습니다. "
                "src.data.assemble.assemble_state_matrix 출력을 사용하세요."
            )
        if list(returns_df.columns) != self.assets:
            raise ValueError(
                f"returns_df 컬럼이 assets 순서와 다릅니다. 기대: {self.assets}"
            )
        if not returns_df.index.equals(state_df.index):
            raise ValueError("returns_df 인덱스가 state_df 인덱스와 일치해야 합니다.")
        if len(state_df) < 2:
            raise ValueError("state_df는 최소 2행 이상 필요합니다 (reset + 1 step).")

        self._state = state_df.reset_index(drop=True)
        self._returns = returns_df.reset_index(drop=True).to_numpy(dtype=np.float32)
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
        self._current_weight = np.zeros(self.n_assets, dtype=np.float32)

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict | None = None,
    ) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        self._t = 0
        # 초기 비중: SHV(현금성) 100% — 무위험 상태에서 출발
        self._current_weight = np.zeros(self.n_assets, dtype=np.float32)
        self._current_weight[self.assets.index("SHV")] = 1.0
        return self._obs(), {}

    def step(
        self,
        action: np.ndarray,
    ) -> tuple[np.ndarray, float, bool, bool, dict]:
        action = np.asarray(action, dtype=np.float32)
        if action.shape != (self.n_assets,):
            raise ValueError(
                f"action shape는 ({self.n_assets},) 이어야 합니다: {action.shape}"
            )

        # 1) Softmax로 목표 비중 확정 (∑w=1)
        w_new = _softmax(action)
        w_prev = self._current_weight

        # 2) 거래비용 (기간 시작 시 리밸런싱 시점에 차감)
        turnover = float(np.abs(w_new - w_prev).sum())
        cost = self.c * turnover

        # 3) 다음 기간 실현 수익률 조회 (룩어헤드 방지: t를 먼저 증가)
        self._t += 1
        r_realized = self._returns[self._t]
        portfolio_return = float(np.dot(w_new, r_realized))

        # 4) 보상 = 포트폴리오 로그수익률 - 거래비용 (CLAUDE.md §2, 계획서 §3-3 1차)
        reward = portfolio_return - cost

        # 5) 상태 업데이트 (기간 내 드리프트 무시, 다음 스텝 시작 시 다시 리밸런싱)
        self._current_weight = w_new

        terminated = self._t >= len(self._state) - 1
        info = {
            "portfolio_return": portfolio_return,
            "cost": cost,
            "turnover": turnover,
            "weights": w_new.copy(),
        }
        return self._obs(), reward, terminated, False, info

    def _obs(self) -> np.ndarray:
        row = self._state.iloc[self._t].to_numpy(dtype=np.float32).copy()
        row[self._prev_weight_slice] = self._current_weight
        return row
