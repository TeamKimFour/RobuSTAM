import numpy as np
import pandas as pd


class BacktestEngine:
    """
    백테스트 엔진
    - AI 모델이 제시한 자산 비중(action)을 받아
    - 매 거래일마다 NAV(총자산)를 갱신하고
    - 수수료를 차감하는 시뮬레이터
    """

    def __init__(self, initial_nav: float = 1_000_000, transaction_cost: float = 0.001):
        """
        Parameters
        ----------
        initial_nav      : 초기 총자산 (기본값 100만원)
        transaction_cost : 편도 거래 비용률 (기본값 0.1% = 0.001)
                           팀 확정 전까지 가변 파라미터로 유지
        """
        self.initial_nav = initial_nav
        self.transaction_cost = transaction_cost

    def calc_nav(
        self,
        prev_nav: float,
        prev_weights: np.ndarray,
        new_weights: np.ndarray,
        price_returns: np.ndarray,
    ) -> tuple[float, float]:
        """
        하루치 NAV 갱신

        Parameters
        ----------
        prev_nav      : 전날 NAV
        prev_weights  : 전날 자산 비중 벡터 (합=1)
        new_weights   : AI가 지시한 새 자산 비중 벡터 (합=1)
        price_returns : 오늘 각 자산의 수익률 벡터 (예: [0.01, -0.005, ...])

        Returns
        -------
        new_nav  : 오늘 NAV
        cost     : 오늘 차감된 수수료
        """
        # 1. 주가 변동 반영
        nav_after_return = prev_nav * (1 + np.dot(prev_weights, price_returns))

        # 2. 수수료 계산 (비중 변경폭 × 거래 비용률)
        turnover = np.sum(np.abs(new_weights - prev_weights))
        cost = nav_after_return * turnover * self.transaction_cost

        # 3. 수수료 차감 후 새 NAV
        new_nav = nav_after_return - cost

        return new_nav, cost