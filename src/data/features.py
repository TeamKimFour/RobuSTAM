"""기술적 지표 계산 — 자산 지표·시장 지표(콤보별 부분집합 지원, M0~M3).

State의 지표 블록을 채운다. 모든 지표는 **인과적**(과거만 참조)이며, warm-up NaN은
메우지 않고 호출부(`compute_features`)에서 통합 drop한다. 산식·파라미터는 docs/state_spec.md §3-1과
config `features.params`가 SSOT다. 컬럼명은 schema.feature_names 규격(`feat_{asset}_{name}`,
`mkt_{name}`)과 일치시켜 assemble이 그대로 배치할 수 있게 한다.

카논(기본 `full` 콤보)은 자산 6종 + 시장 2종이지만, `config_loader.resolve_combo`로 활성화된
콤보(M0~M3 등)에 따라 요청되는 지표 이름 부분집합만 계산한다 — 계산되지 않는 지표는 비용을
쓰지 않는다. `KNOWN_ASSET_FEATURES`/`KNOWN_MARKET_FEATURES`는 이 모듈이 실제로 계산 가능한
전체 지표 카탈로그이며 `config_loader._validate_combos`가 이를 참조해 콤보 정의를 검증한다.
"""

from collections.abc import Sequence

import pandas as pd

from src.config_loader import get_asset_features, get_market_features
from src.data import schema

# 이 모듈이 계산 가능한 지표 카탈로그 — schema.ASSET_FEATURES/MARKET_FEATURES(카논 6+2)에
# RSI_28(M2·M3)·Drawdown(M3, 시장 단일값)을 더한 전체 집합. config_loader가 콤보 검증에 쓴다.
KNOWN_ASSET_FEATURES = (*schema.ASSET_FEATURES, "RSI_28")
KNOWN_MARKET_FEATURES = (*schema.MARKET_FEATURES, "Drawdown")


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
    if name == "RSI_28":
        return _need(ta.rsi(close, length=p["rsi_28_length"]), "rsi")
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


def _market_feature(name: str, close: pd.DataFrame, logret: pd.DataFrame, p: dict) -> pd.Series:
    """시장 공통 지표 1종 계산(전 자산 단일값). name은 KNOWN_MARKET_FEATURES 값."""
    if name == "Equity_Bond_Ratio":
        # SPY/TLT 비율의 이동평균 대비 편차 (상대강도 추세)
        ratio = close["SPY"] / close["TLT"]
        return ratio / ratio.rolling(p["ebr_ma_window"]).mean() - 1.0
    if name == "Gold_Vol_Ratio":
        # GLD 단기(vol_window) / 장기(gvr_long_window) 변동성 비 (변동성 레짐)
        gld_short = logret["GLD"].rolling(p["vol_window"]).std()
        gld_long = logret["GLD"].rolling(p["gvr_long_window"]).std()
        return gld_short / gld_long
    if name == "Drawdown":
        # SPY 최근 drawdown_lookback일 고점 대비 낙폭 (국면 신호, M3)
        return close["SPY"] / close["SPY"].rolling(p["drawdown_lookback"]).max() - 1.0
    raise ValueError(f"알 수 없는 시장 지표: {name}")


def asset_indicators(
    close: pd.DataFrame,
    logret: pd.DataFrame,
    params: dict,
    asset_features: Sequence[str] = schema.ASSET_FEATURES,
) -> pd.DataFrame:
    """요청된 자산 지표만 계산해 wide DataFrame으로 반환(컬럼 = feat_{asset}_{name}).

    asset_features 생략 시 카논 6종(`full` 콤보). 빈 리스트(M0)면 0컬럼이되 close.index로
    정확히 인덱싱된 DataFrame을 반환해 뒤의 pd.concat/dropna가 날짜 정렬을 유지하게 한다.
    warm-up NaN 미제거(호출부에서 통합 drop).
    """
    cols: dict[str, pd.Series] = {}
    for asset in schema.ASSETS:
        for feat in asset_features:
            cols[f"feat_{asset}_{feat}"] = _asset_feature(feat, close[asset], logret[asset], params)
    return pd.DataFrame(cols, index=close.index)


def market_indicators(
    close: pd.DataFrame,
    logret: pd.DataFrame,
    params: dict,
    market_features: Sequence[str] = schema.MARKET_FEATURES,
) -> pd.DataFrame:
    """요청된 시장 공통 지표만 계산한다(단일 시계열). 자산별로 복제하지 않는다(state_spec §3 경고).

    market_features 생략 시 카논 2종(`full` 콤보).
    """
    cols = {f"mkt_{name}": _market_feature(name, close, logret, params) for name in market_features}
    return pd.DataFrame(cols, index=close.index)


def compute_features(
    close: pd.DataFrame, logret: pd.DataFrame, cfg: dict
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """활성 콤보(cfg['features'])의 자산·시장 지표를 계산하고 warm-up NaN을 통합 drop한다.

    cfg는 config_loader.resolve_combo가 이미 콤보를 반영한 것이어야 한다(build.py가 넘김).
    반환: (asset_wide, market) — 동일 index로 정렬됨. 콤보에 따라 컬럼 수가 달라진다
    (예: M0은 asset_wide 0컬럼·market 1컬럼).
    """
    params = cfg["features"]["params"]
    asset_features = get_asset_features(cfg)
    market_features = get_market_features(cfg)
    asset_wide = asset_indicators(close, logret, params, asset_features)
    market = market_indicators(close, logret, params, market_features)

    combined = pd.concat([asset_wide, market], axis=1).dropna(how="any")
    return combined[asset_wide.columns], combined[market.columns]
