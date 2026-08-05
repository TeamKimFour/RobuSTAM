"""State(관측공간) 인덱스맵의 코드측 단일 진실원천(SSOT).

docs/state_spec.md §3 인덱스맵을 코드로 옮긴 것이다. 형우(Gym)·도현(모델)·민지(Feature Store)가
이 모듈의 슬라이스 함수를 함께 import하여 동일 인덱스를 보장한다. 문서와 충돌하면 문서가 우선이며,
그 정합성은 tests/test_schema.py가 검증한다.

인덱스맵 (W=30, K_asset=6, K_market=2 기본값 기준, docs/state_spec.md §3):
    [0:150]    수익률 윈도우   자산별 W일 로그수익률 (SPY/EWY/TLT/GLD/SHV 순)
    [150:180]  자산별 최신지표  자산당 K_asset개 (ASSET_FEATURES 순)
    [180:182]  시장 공통지표   K_market개 — 시장 전체 단일값
    [182:187]  직전 비중       자산별 1칸 (Gym 환경이 채움)

일반식 (W · K_asset · K_market 는 모두 config에서 온다):
    수익률 블록   [0 : A·W]
    지표 블록     [A·W : A·W + K_asset·A]
    시장 블록     [A·W + K_asset·A : A·W + K_asset·A + K_market]
    직전비중 블록 [그 뒤 : state_dim]

가변 지표 개수를 지원하기 위해 슬라이스·차원 함수들은 `n_asset_features`와
`n_market_features` 인자를 받으며, None이면 아래 카논(6/2)을 기본값으로 쓴다.
민지 Feature Store가 config·schema를 확장하면 그 config에서 온 개수를 그대로 전달한다.
"""

from collections.abc import Sequence

# ── 고정 상수 (절대 변경 금지 — CLAUDE.md §2, docs/state_spec.md) ──
ASSETS = ("SPY", "EWY", "TLT", "GLD", "SHV")  # 순서 고정

# 자산별 최신지표 카논 6종 (순서 = state_spec §3 인덱스맵)
# 지표 종류가 바뀌면 config.features.asset 이 SSOT이며, 이 상수는 그 기본값이다.
ASSET_FEATURES = (
    "MA_Cross_5_20",
    "RSI_14",
    "MACD_Hist",
    "Rolling_Vol_20",
    "Bollinger_Band_Width",
    "ROC_10",
)

# 시장 공통지표 카논 2종 — 자산별로 곱하지 말 것 (state_spec §3 경고: 1,355안의 핵심 오류)
MARKET_FEATURES = ("Equity_Bond_Ratio", "Gold_Vol_Ratio")

N_ASSETS = len(ASSETS)                    # A = 5
N_ASSET_FEATURES = len(ASSET_FEATURES)    # 카논 K_asset = 6 (기본값)
N_MARKET_FEATURES = len(MARKET_FEATURES)  # 카논 K_market = 2 (기본값)


def _resolve_counts(
    n_asset_features: int | None,
    n_market_features: int | None,
) -> tuple[int, int]:
    """None을 카논 기본값으로 채운 뒤 유효성을 검증한다."""
    k_a = N_ASSET_FEATURES if n_asset_features is None else int(n_asset_features)
    k_m = N_MARKET_FEATURES if n_market_features is None else int(n_market_features)
    if k_a < 0 or k_m < 0:
        raise ValueError(
            f"지표 개수는 음수일 수 없습니다: n_asset_features={k_a}, n_market_features={k_m}"
        )
    return k_a, k_m


def state_dim(
    W: int,
    n_asset_features: int | None = None,
    n_market_features: int | None = None,
) -> int:
    """State 차원을 (W, K_asset, K_market)에서 산출한다. 187을 하드코딩하지 않는다.

    기본값(K_asset=6, K_market=2)에서 W=20→137, 30→187, 60→337.
    민지 Feature Store가 지표 개수를 바꿔 D=166/171/172 같은 변형이 오면
    호출부에서 n_asset_features·n_market_features를 그 값으로 넘기면 된다.
    """
    k_a, k_m = _resolve_counts(n_asset_features, n_market_features)
    return (N_ASSETS * W) + (k_a * N_ASSETS) + k_m + N_ASSETS


# ── 블록 시작 오프셋 헬퍼 ──
def _returns_block_start() -> int:
    return 0


def _asset_features_block_start(W: int) -> int:
    return N_ASSETS * W


def _market_block_start(W: int, n_asset_features: int) -> int:
    return _asset_features_block_start(W) + n_asset_features * N_ASSETS


def _prev_weight_block_start(W: int, n_asset_features: int, n_market_features: int) -> int:
    return _market_block_start(W, n_asset_features) + n_market_features


# ── 슬라이스 함수 (외부에서 사용하는 공개 API) ──
def returns_slice(asset_idx: int, W: int) -> slice:
    """asset_idx 자산의 W일 로그수익률 윈도우 구간."""
    _check_asset_idx(asset_idx)
    start = _returns_block_start() + asset_idx * W
    return slice(start, start + W)


def asset_feature_slice(
    asset_idx: int,
    W: int,
    n_asset_features: int | None = None,
) -> slice:
    """asset_idx 자산의 최신지표 K_asset칸 구간."""
    _check_asset_idx(asset_idx)
    k_a, _ = _resolve_counts(n_asset_features, 0)
    start = _asset_features_block_start(W) + asset_idx * k_a
    return slice(start, start + k_a)


def market_feature_slice(
    W: int,
    n_asset_features: int | None = None,
    n_market_features: int | None = None,
) -> slice:
    """시장 공통지표 K_market칸 구간 (전 자산 공유 단일값)."""
    k_a, k_m = _resolve_counts(n_asset_features, n_market_features)
    start = _market_block_start(W, k_a)
    return slice(start, start + k_m)


def prev_weight_slice(
    W: int,
    n_asset_features: int | None = None,
    n_market_features: int | None = None,
) -> slice:
    """직전 보유 비중 A칸 구간 (Gym 환경이 런타임에 채움)."""
    k_a, k_m = _resolve_counts(n_asset_features, n_market_features)
    start = _prev_weight_block_start(W, k_a, k_m)
    return slice(start, start + N_ASSETS)


def _check_asset_idx(asset_idx: int) -> None:
    if not 0 <= asset_idx < N_ASSETS:
        raise IndexError(f"asset_idx는 0..{N_ASSETS - 1} 범위여야 합니다: {asset_idx}")


def feature_names(
    W: int,
    asset_features: Sequence[str] | None = None,
    market_features: Sequence[str] | None = None,
) -> list[str]:
    """state_dim 길이의 자기설명적 컬럼명 목록 — Feature Store 컬럼·디버깅·테스트용.

    지표 이름 리스트를 명시하면 그 순서로 컬럼명을 만든다(민지 config.features 소비).
    None이면 카논 리스트를 쓴다(기존 187차원 계약과 동일).

    예: ret_SPY_lag29 … ret_SPY_lag0, feat_SPY_RSI_14, mkt_Equity_Bond_Ratio, prevw_SPY …
    """
    asset_feats = tuple(ASSET_FEATURES if asset_features is None else asset_features)
    market_feats = tuple(MARKET_FEATURES if market_features is None else market_features)

    names: list[str] = []
    # 수익률 블록: 자산별 t-(W-1) … t-0
    for asset in ASSETS:
        names.extend(f"ret_{asset}_lag{W - 1 - i}" for i in range(W))
    # 자산별 지표 블록
    for asset in ASSETS:
        names.extend(f"feat_{asset}_{feat}" for feat in asset_feats)
    # 시장 공통 블록
    names.extend(f"mkt_{feat}" for feat in market_feats)
    # 직전 비중 블록
    names.extend(f"prevw_{asset}" for asset in ASSETS)
    return names


def index_map(
    W: int,
    n_asset_features: int | None = None,
    n_market_features: int | None = None,
) -> dict[str, slice]:
    """전체 블록 슬라이스 맵 — 디버깅·테스트 가독성용."""
    k_a, k_m = _resolve_counts(n_asset_features, n_market_features)
    block: dict[str, slice] = {}
    for i, asset in enumerate(ASSETS):
        block[f"returns_{asset}"] = returns_slice(i, W)
    for i, asset in enumerate(ASSETS):
        block[f"features_{asset}"] = asset_feature_slice(i, W, k_a)
    block["market"] = market_feature_slice(W, k_a, k_m)
    block["prev_weight"] = prev_weight_slice(W, k_a, k_m)
    return block
