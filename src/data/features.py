"""기술적 지표 계산 — 자산별 6종 + 시장 공통 2종.

State의 지표 블록([150:182])을 채운다. 모든 지표는 **인과적**(과거만 참조)이며, warm-up NaN은
메우지 않고 호출부(`compute_features`)에서 통합 drop한다. 산식·파라미터는 docs/state_spec.md §3-1과
config `features.params`가 SSOT다. 컬럼명은 schema.feature_names 규격(`feat_{asset}_{name}`,
`mkt_{name}`)과 일치시켜 assemble이 그대로 배치할 수 있게 한다.
"""

import pandas as pd

from src.data import schema


def _col_by_prefix(df: pd.DataFrame, prefix: str) -> pd.Series:
    """pandas-ta 컬럼명이 버전마다 접미사가 달라(예: BBB_20_2.0_2.0) 접두어로 찾는다."""
    matches = [c for c in df.columns if c.startswith(prefix)]
    if not matches:
        raise KeyError(f"'{prefix}'로 시작하는 컬럼이 없습니다: {list(df.columns)}")
    return df[matches[0]]


def _asset_feature(name: str, close: pd.Series, logret: pd.Series, p: dict) -> pd.Series:
    """자산 지표 1종 계산. name은 schema.ASSET_FEATURES 값."""
    import pandas_ta as ta

    def _need(obj, what: str):
        """pandas-ta는 입력이 부족하면 None을 반환한다 — 조용히 흘려보내지 않고 원인을 밝힌다.

        (수집이 빈 데이터를 넘겼을 때 `TypeError: NoneType - NoneType`처럼 원인에서
        한참 떨어진 곳에서 터지던 문제. 근본 방어는 collect의 품질 게이트다.)
        """
        if obj is None:
            raise ValueError(
                f"지표 '{name}'의 {what} 계산 실패 — 자산 {close.name!r}의 입력이 "
                f"{len(close)}행으로 부족합니다. 수집 결과가 비었거나 잘렸는지 확인하세요"
                " (src/data/collect.py 품질 게이트)."
            )
        return obj

    if name == "MA_Cross_5_20":
        sma_fast = _need(ta.sma(close, length=p["sma_fast"]), f"sma{p['sma_fast']}")
        sma_slow = _need(ta.sma(close, length=p["sma_slow"]), f"sma{p['sma_slow']}")
        return (sma_fast - sma_slow) / sma_slow
    if name == "RSI_14":
        return _need(ta.rsi(close, length=p["rsi_length"]), "rsi")
    if name == "MACD_Hist":
        fast, slow, signal = p["macd"]
        macd = _need(ta.macd(close, fast=fast, slow=slow, signal=signal), "macd")
        return _col_by_prefix(macd, "MACDh_")
    if name == "Rolling_Vol_20":
        return logret.rolling(p["vol_window"]).std()  # annualize_vol=false → 연율화 안 함
    if name == "Bollinger_Band_Width":
        length, std = p["bbands"]
        bb = _need(ta.bbands(close, length=length, std=std), "bbands")
        return _col_by_prefix(bb, "BBB_")
    if name == "ROC_10":
        return _need(ta.roc(close, length=p["roc_length"]), "roc")
    raise ValueError(f"알 수 없는 자산 지표: {name}")


def asset_indicators(close: pd.DataFrame, logret: pd.DataFrame, params: dict) -> pd.DataFrame:
    """자산별 6지표를 계산해 wide DataFrame으로 반환(컬럼 = feat_{asset}_{name}, 순서 = schema).

    warm-up NaN 미제거(호출부에서 통합 drop).
    """
    cols: dict[str, pd.Series] = {}
    for asset in schema.ASSETS:
        for feat in schema.ASSET_FEATURES:
            cols[f"feat_{asset}_{feat}"] = _asset_feature(feat, close[asset], logret[asset], params)
    return pd.DataFrame(cols)


def market_indicators(close: pd.DataFrame, logret: pd.DataFrame, params: dict) -> pd.DataFrame:
    """시장 공통 2지표(단일 시계열). 자산별로 복제하지 않는다(state_spec §3 경고)."""
    # Equity_Bond_Ratio: SPY/TLT 비율의 이동평균 대비 편차 (상대강도 추세)
    ratio = close["SPY"] / close["TLT"]
    ebr = ratio / ratio.rolling(params["ebr_ma_window"]).mean() - 1.0

    # Gold_Vol_Ratio: GLD 단기(vol_window) / 장기(gvr_long_window) 변동성 비 (변동성 레짐)
    gld_short = logret["GLD"].rolling(params["vol_window"]).std()
    gld_long = logret["GLD"].rolling(params["gvr_long_window"]).std()
    gvr = gld_short / gld_long

    return pd.DataFrame(
        {"mkt_Equity_Bond_Ratio": ebr, "mkt_Gold_Vol_Ratio": gvr}
    )


def compute_features(
    close: pd.DataFrame, logret: pd.DataFrame, cfg: dict
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """자산·시장 지표를 계산하고 warm-up NaN을 통합 drop한다.

    반환: (asset_wide[30컬럼], market[2컬럼]) — 동일 index로 정렬됨.
    """
    params = cfg["features"]["params"]
    asset_wide = asset_indicators(close, logret, params)
    market = market_indicators(close, logret, params)

    combined = pd.concat([asset_wide, market], axis=1).dropna(how="any")
    return combined[asset_wide.columns], combined[market.columns]
