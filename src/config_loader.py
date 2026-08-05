"""config.yaml 로더 — 설정값을 읽고 파생 차원(state_dim)을 산출하는 단일 통로.

코드 어디서도 187 같은 차원을 직접 박지 않는다. 항상 config의 W와 features 길이에서 계산한다
(docs/state_spec.md §4·§5③). 서드파티 의존은 PyYAML 하나뿐이라 CI에서 가볍게 테스트된다.

State 차원 수식 (state_spec.md §2):
    state_dim = (A × W) + (K_asset × A) + K_market + A
      A        = 자산 수 (config.assets 길이)
      W        = 수익률 history 윈도우 (config.window)
      K_asset  = 자산별 최신지표 수 (config.features.asset 길이, 기본 6)
      K_market = 시장 공통지표 수 (config.features.market 길이, 기본 2)
      +A       = 직전 보유 비중(Prev_Weight)

민지 Feature Store가 지표 개수를 바꿔 D=166/171/172 같은 변형을 낼 때,
`config.features.asset`·`config.features.market` 길이만 그에 맞게 두면 이 로더가
자동으로 새 state_dim을 뽑아 환경·모델·API가 그대로 소비할 수 있다.
"""

from pathlib import Path

from src.data import schema

# 구조 상수 — docs/state_spec.md 인덱스맵에서 고정된 값의 **기본치**.
# 카논 정의(지표 '이름' 목록)는 src/data/schema.py가 가지며, 테스트가 이 수치와 교차검증한다.
# 실제 값은 config.features.asset·market 길이에서 우선 계산한다.
N_ASSET_FEATURES = schema.N_ASSET_FEATURES   # 카논 6 (기본값)
N_MARKET_FEATURES = schema.N_MARKET_FEATURES  # 카논 2 (기본값)
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


def get_asset_features(cfg: dict) -> list[str]:
    """자산별 지표 이름 목록을 반환한다 (config.features.asset).

    config에 features 섹션이 없으면 schema 카논 6종을 기본값으로 쓴다 —
    features 섹션 없이도 기본 187차원 계약이 그대로 유지되도록 하기 위함.
    """
    feats = cfg.get("features", {}).get("asset")
    if feats is None:
        return list(schema.ASSET_FEATURES)
    return list(feats)


def get_market_features(cfg: dict) -> list[str]:
    """시장 공통지표 이름 목록을 반환한다 (config.features.market, 기본 카논 2종)."""
    feats = cfg.get("features", {}).get("market")
    if feats is None:
        return list(schema.MARKET_FEATURES)
    return list(feats)


def get_n_asset_features(cfg: dict) -> int:
    """K_asset — 자산별 지표 수."""
    return len(get_asset_features(cfg))


def get_n_market_features(cfg: dict) -> int:
    """K_market — 시장 공통지표 수."""
    return len(get_market_features(cfg))


def get_state_dim(cfg: dict) -> int:
    """config의 W·자산수·지표 개수에서 State 차원을 산출한다.

    카논(K_asset=6, K_market=2, A=5)에서 W=20 → 137, W=30 → 187, W=60 → 337.
    features 개수가 변하면 (5W)+5·K_asset+K_market+5 으로 자동 반영된다.
    """
    A = len(get_assets(cfg))
    W = get_window(cfg)
    k_a = get_n_asset_features(cfg)
    k_m = get_n_market_features(cfg)
    return (A * W) + (k_a * A) + k_m + (N_PREV_WEIGHT * A)


def get_precompute_path(cfg: dict) -> str:
    """익일 추천 비중 파일(latest.json) 경로를 반환한다.

    민지 precompute가 이 경로에 쓰고, 도현 API(src/api/main.py)가 이 경로를 읽는다
    (docs/data_pipeline.md §8 계약).
    """
    return cfg.get("data", {}).get("precompute_path", "data/precompute/latest.json")


def get_inference(cfg: dict) -> dict:
    """추론(precompute)이 쓸 설정을 반환한다 — 모델 경로·정규화 통계 출처·버전.

    어느 policy(model_path)를 어느 build 통계(scaler_run_id/fold_id)로 정규화 재현할지
    지정한다. 이 값은 학습 provenance(도현 train.py가 기록하는 build run_id/fold_id)와
    일치해야 추론 정규화가 학습과 동일하게 재현된다(docs/data_pipeline.md §8).
    """
    return cfg.get("inference", {})
