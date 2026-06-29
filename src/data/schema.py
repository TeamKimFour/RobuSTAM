"""State(관측공간) 인덱스맵의 코드측 단일 진실원천(SSOT).

docs/state_spec.md §3 인덱스맵을 코드로 옮긴 것이다. 형우(Gym)·도현(모델)·민지(Feature Store)가
이 모듈의 슬라이스 함수를 함께 import하여 동일 인덱스를 보장한다. 문서와 충돌하면 문서가 우선이며,
그 정합성은 tests/test_schema.py가 검증한다.

인덱스맵 (W=30 기준, docs/state_spec.md §3):
    [0:150]    수익률 윈도우   자산별 W일 로그수익률 (SPY/EWY/TLT/GLD/SHV 순)
    [150:180]  자산별 최신지표  자산당 6개 (ASSET_FEATURES 순)
    [180:182]  시장 공통지표   [Equity_Bond_Ratio, Gold_Vol_Ratio] — 시장 전체 단일값
    [182:187]  직전 비중       자산별 1칸 (Gym 환경이 채움)

일반식 (W는 하이퍼파라미터):
    수익률 블록   [0 : A·W]
    지표 블록     [A·W : A·W + K_asset·A]
    시장 블록     [A·W + K_asset·A : A·W + K_asset·A + K_market]
    직전비중 블록 [그 뒤 : state_dim]
"""

# ── 고정 상수 (절대 변경 금지 — CLAUDE.md §2, docs/state_spec.md) ──
ASSETS = ("SPY", "EWY", "TLT", "GLD", "SHV")  # 순서 고정

# 자산별 최신지표 6종 (순서 = state_spec §3 인덱스맵)
ASSET_FEATURES = (
    "MA_Cross_5_20",
    "RSI_14",
    "MACD_Hist",
    "Rolling_Vol_20",
    "Bollinger_Band_Width",
    "ROC_10",
)

# 시장 공통지표 2종 — 자산별로 곱하지 말 것 (state_spec §3 경고: 1,355안의 핵심 오류)
MARKET_FEATURES = ("Equity_Bond_Ratio", "Gold_Vol_Ratio")

N_ASSETS = len(ASSETS)                  # A = 5
N_ASSET_FEATURES = len(ASSET_FEATURES)  # K_asset = 6
N_MARKET_FEATURES = len(MARKET_FEATURES)  # K_market = 2


def state_dim(W: int) -> int:
    """State 차원을 W에서 산출한다. W=20→137, 30→187, 60→337."""
    return (N_ASSETS * W) + (N_ASSET_FEATURES * N_ASSETS) + N_MARKET_FEATURES + N_ASSETS


# ── 블록 시작 오프셋 헬퍼 ──
def _returns_block_start() -> int:
    return 0


def _asset_features_block_start(W: int) -> int:
    return N_ASSETS * W


def _market_block_start(W: int) -> int:
    return _asset_features_block_start(W) + N_ASSET_FEATURES * N_ASSETS


def _prev_weight_block_start(W: int) -> int:
    return _market_block_start(W) + N_MARKET_FEATURES


# ── 슬라이스 함수 (외부에서 사용하는 공개 API) ──
def returns_slice(asset_idx: int, W: int) -> slice:
    """asset_idx 자산의 W일 로그수익률 윈도우 구간."""
    _check_asset_idx(asset_idx)
    start = _returns_block_start() + asset_idx * W
    return slice(start, start + W)


def asset_feature_slice(asset_idx: int, W: int) -> slice:
    """asset_idx 자산의 최신지표 6칸 구간."""
    _check_asset_idx(asset_idx)
    start = _asset_features_block_start(W) + asset_idx * N_ASSET_FEATURES
    return slice(start, start + N_ASSET_FEATURES)


def market_feature_slice(W: int) -> slice:
    """시장 공통지표 2칸 구간 (전 자산 공유 단일값)."""
    start = _market_block_start(W)
    return slice(start, start + N_MARKET_FEATURES)


def prev_weight_slice(W: int) -> slice:
    """직전 보유 비중 A칸 구간 (Gym 환경이 런타임에 채움)."""
    start = _prev_weight_block_start(W)
    return slice(start, start + N_ASSETS)


def _check_asset_idx(asset_idx: int) -> None:
    if not 0 <= asset_idx < N_ASSETS:
        raise IndexError(f"asset_idx는 0..{N_ASSETS - 1} 범위여야 합니다: {asset_idx}")


def feature_names(W: int) -> list[str]:
    """state_dim 길이의 자기설명적 컬럼명 목록 — Feature Store 컬럼·디버깅·테스트용.

    예: ret_SPY_lag29 … ret_SPY_lag0, feat_SPY_RSI_14, mkt_Equity_Bond_Ratio, prevw_SPY …
    """
    names: list[str] = []
    # 수익률 블록: 자산별 t-(W-1) … t-0
    for asset in ASSETS:
        names.extend(f"ret_{asset}_lag{W - 1 - i}" for i in range(W))
    # 자산별 지표 블록
    for asset in ASSETS:
        names.extend(f"feat_{asset}_{feat}" for feat in ASSET_FEATURES)
    # 시장 공통 블록
    names.extend(f"mkt_{feat}" for feat in MARKET_FEATURES)
    # 직전 비중 블록
    names.extend(f"prevw_{asset}" for asset in ASSETS)
    return names


def index_map(W: int) -> dict[str, slice]:
    """전체 블록 슬라이스 맵 — 디버깅·테스트 가독성용."""
    block: dict[str, slice] = {}
    for i, asset in enumerate(ASSETS):
        block[f"returns_{asset}"] = returns_slice(i, W)
    for i, asset in enumerate(ASSETS):
        block[f"features_{asset}"] = asset_feature_slice(i, W)
    block["market"] = market_feature_slice(W)
    block["prev_weight"] = prev_weight_slice(W)
    return block
