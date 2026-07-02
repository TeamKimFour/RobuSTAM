"""187차원 State 행렬 조립.

수익률 윈도우 + 자산 지표 + 시장 지표 + 직전비중(0) 을 schema 인덱스맵 위치에 배치해
wide DataFrame(컬럼 = schema.feature_names(W))을 만든다. 정규화는 여기서 하지 않는다(normalize 단계).

룩어헤드: 수익률 윈도우는 t 포함·t+1 미포함(returns.return_window와 동일 원리, shift로 벡터화).
조립 대상 날짜는 "W일 윈도우가 가능하고 지표도 존재하는 날"의 교집합.
"""

import pandas as pd

from src.config_loader import get_window
from src.data import schema


def assemble_state_matrix(
    log_ret: pd.DataFrame,
    asset_feat: pd.DataFrame,
    market_feat: pd.DataFrame,
    cfg: dict,
) -> pd.DataFrame:
    """187차원 wide State 행렬을 조립한다.

    log_ret    : 전체 로그수익률 (index=date, 컬럼=자산)
    asset_feat : 자산 지표 30컬럼 (warm-up drop됨, 컬럼=feat_{asset}_{name})
    market_feat: 시장 지표 2컬럼 (mkt_{name})
    반환: index=조립가능일, 컬럼=feature_names(W) 순서, prev_weight 블록은 0.
    """
    W = get_window(cfg)
    names = schema.feature_names(W)

    # 조립 가능일: 지표가 있고(warm-up 후) + 앞으로 W일 윈도우가 확보되는 날
    pos = pd.Series(range(len(log_ret.index)), index=log_ret.index)
    dates = asset_feat.index[asset_feat.index.isin(log_ret.index)]
    dates = dates[pos.reindex(dates).to_numpy() >= W - 1]

    out = pd.DataFrame(0.0, index=dates, columns=names)
    out.index.name = "date"

    # [0:150] 수익률 윈도우 — lag_k = k일 전 수익률 (lag0=당일). t+1 미포함.
    for asset in schema.ASSETS:
        s = log_ret[asset]
        for k in range(W):
            out[f"ret_{asset}_lag{k}"] = s.shift(k).reindex(dates).to_numpy()

    # [150:180] 자산 지표 · [180:182] 시장 지표 (시장은 단일값 — 자산별 복제 없음)
    for col in asset_feat.columns:
        out[col] = asset_feat[col].reindex(dates).to_numpy()
    for col in market_feat.columns:
        out[col] = market_feat[col].reindex(dates).to_numpy()

    # [182:187] 직전 비중 — 데이터 레이어는 0 유지(Gym 런타임이 채움)

    assert list(out.columns) == names, "컬럼 순서가 schema.feature_names와 불일치"
    return out
