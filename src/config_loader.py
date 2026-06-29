"""config.yaml 로더 — 설정값을 읽고 파생 차원(state_dim)을 산출하는 단일 통로.

코드 어디서도 187 같은 차원을 직접 박지 않는다. 항상 config의 W에서 계산한다
(docs/state_spec.md §4·§5③). 서드파티 의존은 PyYAML 하나뿐이라 CI에서 가볍게 테스트된다.

State 차원 수식 (state_spec.md §2):
    state_dim = (A × W) + (K_asset × A) + K_market + A
      A        = 자산 수 (config.assets 길이)
      W        = 수익률 history 윈도우 (config.window)
      K_asset  = 자산별 최신지표 수 (6)
      K_market = 시장 공통지표 수 (2)
      +A       = 직전 보유 비중(Prev_Weight)
"""

from pathlib import Path

# 구조 상수 — docs/state_spec.md 인덱스맵에서 고정된 값.
# 카논 정의(지표 '이름' 목록)는 src/data/schema.py가 가지며, 테스트가 이 수치와 교차검증한다.
N_ASSET_FEATURES = 6   # [MA_Cross_5_20, RSI_14, MACD_Hist, Rolling_Vol_20, Bollinger_Band_Width, ROC_10]
N_MARKET_FEATURES = 2  # [Equity_Bond_Ratio, Gold_Vol_Ratio]
N_PREV_WEIGHT = 1      # 자산당 직전 비중 1칸 (→ 전체 A칸)

# 자산 순서는 절대 불변 (CLAUDE.md §2). 로드 시 이 순서와 일치하는지 검증한다.
EXPECTED_ASSETS = ("SPY", "EWY", "TLT", "GLD", "SHV")

DEFAULT_CONFIG_PATH = "config/config.yaml"


def load_config(path: str = DEFAULT_CONFIG_PATH) -> dict:
    """config.yaml을 읽어 dict로 반환한다. 핵심 키 존재·자산 순서를 함께 검증한다."""
    import yaml  # 함수 내부 import — 차원 계산 등 순수 함수는 PyYAML 없이도 동작하게

    cfg_path = Path(path)
    if not cfg_path.is_file():
        raise FileNotFoundError(f"config 파일을 찾을 수 없습니다: {cfg_path}")

    with cfg_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    _validate(cfg)
    return cfg


def _validate(cfg: dict) -> None:
    """필수 키와 핵심 불변식(자산 순서·W 양수)을 검증한다."""
    for key in ("assets", "window", "transaction_cost"):
        if key not in cfg:
            raise KeyError(f"config에 필수 키가 없습니다: '{key}'")

    assets = tuple(cfg["assets"])
    if assets != EXPECTED_ASSETS:
        raise ValueError(
            f"자산 순서가 고정값과 다릅니다. 기대: {EXPECTED_ASSETS}, 실제: {assets} "
            "(CLAUDE.md §2 — 순서 변경 금지)"
        )

    if not isinstance(cfg["window"], int) or cfg["window"] <= 0:
        raise ValueError(f"window(W)는 양의 정수여야 합니다: {cfg['window']!r}")


def get_window(cfg: dict) -> int:
    """수익률 history 윈도우 W를 반환한다."""
    return int(cfg["window"])


def get_assets(cfg: dict) -> list[str]:
    """자산 목록을 고정 순서대로 반환한다."""
    return list(cfg["assets"])


def get_transaction_cost(cfg: dict) -> float:
    """편도 거래비용 c를 반환한다 (기본 0.001 = 0.1%)."""
    return float(cfg["transaction_cost"])


def get_state_dim(cfg: dict) -> int:
    """config의 W·자산수에서 State 차원을 산출한다 (187을 하드코딩하지 않는다).

    W=20 → 137, W=30 → 187, W=60 → 337.
    """
    A = len(get_assets(cfg))
    W = get_window(cfg)
    return (A * W) + (N_ASSET_FEATURES * A) + N_MARKET_FEATURES + (N_PREV_WEIGHT * A)
