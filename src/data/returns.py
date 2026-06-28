"""로그수익률 및 수익률 윈도우 구성 모듈.

State의 수익률 블록(인덱스 [0:A·W])은 "원시 수익률만 시계열" 원칙에 따라 가격 경로 자체를
신호로 사용한다(docs/state_spec.md §1). 윈도우는 현재 시점 t를 포함하되 **t+1 이후는 절대
포함하지 않는다**(룩어헤드 핵심).
"""

import numpy as np
import pandas as pd


def log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """일별 로그수익률 = ln(P_t / P_{t-1}). 첫 행(NaN)은 drop한다.

    로그수익률은 가산성(시계열 합산 용이)·정규성에서 단순수익률보다 유리하다.
    """
    rets = np.log(prices / prices.shift(1))
    return rets.iloc[1:]  # 첫 행은 이전 가격이 없어 NaN → 제거


def return_window(log_ret: pd.DataFrame, t: int, W: int) -> np.ndarray:
    """위치 t(정수 인덱스, 포함)에서 끝나는 W일 수익률 윈도우를 반환한다.

    반환 shape = (W, n_assets). 구간은 [t-W+1 .. t] (t 포함, t+1 이후 미포함).
    t-W+1 < 0이면 윈도우가 부족하므로 에러.
    """
    if t < 0 or t >= len(log_ret):
        raise IndexError(f"t는 0..{len(log_ret) - 1} 범위여야 합니다: {t}")
    if t - W + 1 < 0:
        raise ValueError(f"위치 {t}에서 W={W} 윈도우를 만들 이전 데이터가 부족합니다")
    return log_ret.iloc[t - W + 1 : t + 1].to_numpy()
