"""자산 5종 데일리 리밸런싱 커스텀 Gymnasium 환경 — 골격.

이 파일은 PR-1(env-skeleton) 범위로, 관측/행동 공간·reset()·step()의 뼈대만 정의한다.
Softmax 정규화, 거래비용 차감, 슬리피지, 보상함수는 후속 PR에서 채운다.

인터페이스 계약 (docs/state_spec.md §3 인덱스맵):
    관측 shape = (state_dim,), state_dim = config.window에서 산출 (W=30 → 187)
    관측 마지막 A칸([182:187] @ W=30)은 이 클래스가 런타임에 직전 비중으로 채운다
    행동 shape = (A,) — 정책 신경망 로짓. Softmax는 step()에서 처리(CLAUDE.md §2)
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


class PortfolioEnv(gym.Env):
    """5자산 목표 비중을 매 거래일 조절하는 포트폴리오 환경.

    Parameters
    ----------
    state_df : 조립된 wide State (`src.data.assemble.assemble_state_matrix` 출력).
        컬럼 순서는 `schema.feature_names(W)`와 정확히 일치해야 한다.
    cfg : config.yaml 사전. 생략 시 기본 경로에서 로드한다.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        state_df: pd.DataFrame,
        cfg: dict | None = None,
    ) -> None:
        super().__init__()
        self.cfg = cfg if cfg is not None else load_config()
        self.W = get_window(self.cfg)
        self.assets = get_assets(self.cfg)
        self.n_assets = len(self.assets)
        self.state_dim = get_state_dim(self.cfg)
        self.c = get_transaction_cost(self.cfg)  # 후속 PR에서 사용

        expected_cols = schema.feature_names(self.W)
        if list(state_df.columns) != expected_cols:
            raise ValueError(
                "state_df 컬럼이 schema.feature_names(W)와 일치하지 않습니다. "
                "src.data.assemble.assemble_state_matrix 출력을 사용하세요."
            )
        if len(state_df) < 2:
            raise ValueError("state_df는 최소 2행 이상 필요합니다 (reset + 1 step).")

        self._state = state_df.reset_index(drop=True)
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
        # 후속 PR-2/3: action → softmax → 새 목표 비중, 로그수익률·거래비용·슬리피지 반영
        # 현재는 shape 계약 검증용 스텁이며 보상은 0으로 고정한다.
        self._t += 1
        terminated = self._t >= len(self._state) - 1
        return self._obs(), 0.0, terminated, False, {}

    def _obs(self) -> np.ndarray:
        row = self._state.iloc[self._t].to_numpy(dtype=np.float32).copy()
        row[self._prev_weight_slice] = self._current_weight
        return row
