"""DQN MVP용 이산 행동 어댑터 (3주차, docs/model_training.md §7).

원본 `PortfolioEnv`는 연속 로짓(shape=(A,)) → Softmax → A자산 비중을 계약으로 삼는다.
DQN(Deep Q-Network)은 유한한 이산 행동만 다룰 수 있으므로, 원본 env는 손대지 않고
이 어댑터가 정수 action → 로짓 벡터로 변환한다(gymnasium.ActionWrapper 패턴).

행동 설계 (3주차 확정 옵션 B — "자산 하나를 SHV와 교환하거나 유지"):
    조정 대상 자산 = SHV를 제외한 [SPY, EWY, TLT, GLD] (A_tr = 4)
    총 행동 수 = 2·A_tr + 1 = 9
        - 0..A_tr-1       : 자산 i에 +Δ (SHV에서 뺌)
        - A_tr..2·A_tr-1  : 자산 i에서 -Δ (SHV로 반환)
        - 2·A_tr          : 유지 (현재 비중 그대로)

    SHV(현금성)를 상대 계정으로 삼으면 이전량이 항상 명확하고 ∑w=1이 자동 보존된다.
    행동 개수가 9로 작아 DQN의 Q(s,a) 테이블이 좁고, 학습이 안정적이다(설계 A의 3^5=243 대비).

경계 처리:
    실제 이전량 = min(Δ, 여유분)으로 clip. 예: SHV=0.05일 때 +0.1 → 0.05만 이동.
    → 음수 비중 방지, 항상 유효한 action 유지. SHV=0에서 +Δ, 자산=0에서 -Δ는
      no-op으로 동작하지만 유지·반대방향 action은 항상 살아 있어 에피소드 진행 문제 없음.

로짓 변환 (softmax 계약 유지):
    원본 env는 내부에서 softmax(∑=1) 정규화한다. 계산한 w_new(합=1)를
    log(clip(w_new, EPS, 1))로 넘기면 softmax가 w_new를 EPS 근사오차 내에서 복원한다.
        softmax(log(w)) = exp(log(w)) / Σ exp(log(w)) = w / Σw = w   (Σw=1일 때)
    EPS(1e-8)로 잘린 0-비중 자산은 softmax 후 ~1e-8 크기로 남는다(테스트로 검증).

관측·보상 계약:
    관측 shape·보상 계산은 원본 env를 그대로 통과한다(어댑터는 action만 변환).
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from src.config_loader import get_assets
from src.env.portfolio_env import PortfolioEnv


_EPS = 1e-8


def discrete_action_to_logits(
    w_prev: np.ndarray,
    discrete_action: int,
    *,
    shv_idx: int,
    tradable_idx: list[int],
    delta: float,
) -> np.ndarray:
    """이산 action(정수)을 직전 비중 기준으로 로짓 벡터로 바꾼다 (순수 함수).

    학습(`DiscretePortfolioEnv.action`)과 백테스트(`src.backtest.policy.run_policy`)가
    **같은 함수**를 쓰게 하려고 분리했다. 두 경로가 Δ 이전 규칙을 따로 구현하면 백테스트가
    학습과 조용히 어긋난다 — policy.py 모듈 독스트링 ③이 경고하는 것과 같은 종류의 사고다.
    """
    n_tradable = len(tradable_idx)
    hold_action = 2 * n_tradable
    if not (0 <= discrete_action <= hold_action):
        raise ValueError(
            f"이산 action은 [0, {hold_action + 1}) 범위여야 합니다: {discrete_action}"
        )

    w_new = np.asarray(w_prev, dtype=np.float64).copy()
    if discrete_action == hold_action:
        pass
    elif discrete_action < n_tradable:  # +Δ to tradable asset (SHV → asset)
        asset_i = tradable_idx[discrete_action]
        move = float(min(delta, w_new[shv_idx]))
        w_new[asset_i] += move
        w_new[shv_idx] -= move
    else:  # -Δ from tradable asset (asset → SHV)
        asset_i = tradable_idx[discrete_action - n_tradable]
        move = float(min(delta, w_new[asset_i]))
        w_new[asset_i] -= move
        w_new[shv_idx] += move

    # softmax(log(w)) = w (∑w=1) — 원본 env의 softmax 계약을 그대로 유지한다.
    return np.log(np.clip(w_new, _EPS, 1.0)).astype(np.float32)


class DiscretePortfolioEnv(gym.ActionWrapper):
    """PortfolioEnv를 DQN용 이산 행동으로 감싸는 어댑터.

    Parameters
    ----------
    env : PortfolioEnv
        원본 5자산 환경. 관측·보상 계약은 그대로 보존된다.
    delta : float, default 0.1
        스텝당 자산 ↔ SHV 이전 폭. (0, 1] 범위.
    """

    _EPS = 1e-8

    def __init__(self, env: PortfolioEnv, delta: float = 0.1) -> None:
        super().__init__(env)
        if not (0.0 < delta <= 1.0):
            raise ValueError(f"delta는 (0, 1] 범위여야 합니다: {delta}")
        self.delta = float(delta)

        # SHV 인덱스 고정 (자산 순서는 CLAUDE.md §2 · config_loader.EXPECTED_ASSETS로 보장).
        assets = get_assets(env.unwrapped.cfg)
        if "SHV" not in assets:
            raise ValueError(
                "이산 어댑터는 SHV(현금성)를 상대 계정으로 사용합니다. "
                "config.assets에 SHV가 없으면 사용할 수 없습니다."
            )
        self._shv_idx = assets.index("SHV")
        self._tradable_idx = [i for i, a in enumerate(assets) if a != "SHV"]
        self._n_tradable = len(self._tradable_idx)
        self._hold_action = 2 * self._n_tradable
        self.action_space = spaces.Discrete(2 * self._n_tradable + 1)

    def action(self, discrete_action) -> np.ndarray:
        # ActionWrapper.step()이 self.env.step()을 곧바로 호출하므로 이 시점의
        # _current_weight가 w_prev와 일치한다.
        w_prev = self.env.unwrapped._current_weight.copy()
        return discrete_action_to_logits(
            w_prev,
            int(discrete_action),
            shv_idx=self._shv_idx,
            tradable_idx=self._tradable_idx,
            delta=self.delta,
        )
