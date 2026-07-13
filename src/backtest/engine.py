import numpy as np
import pandas as pd

from src.config_loader import load_config, get_transaction_cost


class BacktestEngine:
    """
    백테스트 엔진
    - AI 모델이 제시한 자산 비중(action)을 받아
    - 매 거래일마다 NAV(총자산)를 갱신하고
    - 수수료를 차감하는 시뮬레이터
    """

    def __init__(self, initial_nav: float = 1_000_000, config_path: str = "config/config.yaml"):
        """
        Parameters
        ----------
        initial_nav  : 초기 총자산 (기본값 100만원)
        config_path  : config.yaml 경로
        """
        self.initial_nav = initial_nav
        self.config_path = config_path
        cfg = load_config(config_path)
        self.transaction_cost = get_transaction_cost(cfg)

    def calc_nav(
        self,
        prev_nav: float,
        prev_weights: np.ndarray,
        new_weights: np.ndarray,
        price_returns: np.ndarray,
    ) -> tuple[float, float]:
        """
        하루치 NAV 갱신 (start-of-day 리밸런싱)

        매 스텝 시작에 전날 비중에서 새 비중으로 리밸런싱하고,
        그 새 비중으로 오늘 수익을 실현한다(env와 관점 통일).

        완전 리밸런싱 가정(팀 회의 확정): env와 동일하게 기간 내 드리프트는
        무시한다. 여러 날을 순회하는 호출자는 다음 스텝의 prev_weights로
        이번 스텝의 new_weights를 드리프트 조정 없이 그대로 넘겨야 한다.

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
        # 1. 수수료 계산 (비중 변경폭 × 거래 비용률) — 리밸런싱 전 NAV 기준
        turnover = np.sum(np.abs(new_weights - prev_weights))
        cost = prev_nav * turnover * self.transaction_cost

        # 2. 수수료 차감 후 리밸런싱
        nav_after_cost = prev_nav - cost

        # 3. 새 비중으로 오늘 주가 변동 반영
        new_nav = nav_after_cost * (1 + np.dot(new_weights, price_returns))

        return new_nav, cost

    def save_results(
        self,
        nav: pd.DataFrame,
        metrics: dict,
        run_id: str | None = None,
        *,
        bucket: str | None = None,
        client=None,
    ) -> str | None:
        """백테스트 실행 결과를 S3에 저장한다 (src.backtest.s3_results에 위임).

        calc_nav()와는 무관한 별도 단계 — 백테스트 루프가 모두 끝난 뒤 호출한다.
        """
        from src.backtest.s3_results import save_results_to_s3

        return save_results_to_s3(
            nav, metrics, run_id, self.config_path, bucket=bucket, client=client
        )