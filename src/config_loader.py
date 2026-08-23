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

import sys
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
    """필수 키와 핵심 불변식(자산 순서·W 양수·콤보 정합성)을 검증한다."""
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

    if "feature_combos" in cfg or "active_combo" in cfg:
        _validate_combos(cfg)


def _validate_combos(cfg: dict) -> None:
    """feature_combos/active_combo 섹션의 정합성을 검증한다 (M0~M3 등 콤보 시스템).

    이 섹션이 없는 config(구 계약)는 이 함수가 아예 호출되지 않으므로 영향 없다.
    """
    from src.data.features import KNOWN_ASSET_FEATURES, KNOWN_MARKET_FEATURES

    combos = cfg.get("feature_combos")
    active = cfg.get("active_combo")
    if combos is None or active is None:
        raise KeyError("feature_combos와 active_combo는 함께 정의해야 합니다")

    if active not in combos:
        raise ValueError(f"active_combo({active!r})가 feature_combos에 없습니다: {list(combos)}")

    for name, spec in combos.items():
        if "asset" not in spec or "market" not in spec:
            raise KeyError(f"feature_combos.{name}에는 asset·market 키가 모두 있어야 합니다")
        for kind, names, known in (
            ("asset", spec["asset"], KNOWN_ASSET_FEATURES),
            ("market", spec["market"], KNOWN_MARKET_FEATURES),
        ):
            if len(set(names)) != len(names):
                raise ValueError(f"feature_combos.{name}.{kind}에 중복 지표가 있습니다: {names}")
            unknown = [n for n in names if n not in known]
            if unknown:
                raise ValueError(
                    f"feature_combos.{name}.{kind}에 알 수 없는 지표: {unknown} "
                    f"(지원: {sorted(known)})"
                )


def get_window(cfg: dict) -> int:
    """수익률 history 윈도우 W를 반환한다."""
    return int(cfg["window"])


def get_assets(cfg: dict) -> list[str]:
    """자산 목록을 고정 순서대로 반환한다."""
    return list(cfg["assets"])


def get_transaction_cost(cfg: dict) -> float:
    """편도 거래비용 c를 반환한다 (기본 0.001 = 0.1%)."""
    return float(cfg["transaction_cost"])


def get_action_delta(cfg: dict) -> float:
    """이산 행동(DQN)의 스텝당 SHV↔자산 이전폭 Δ를 반환한다 (기본 0.1, docs/env_spec.md §4-4).

    학습(DiscretePortfolioEnv)과 백테스트(src.backtest.policy)가 같은 값을 봐야 두 경로의
    비중 궤적이 일치하므로 config를 단일 출처로 둔다.
    """
    delta = float(cfg.get("model", {}).get("action_delta", 0.1))
    if not (0.0 < delta <= 1.0):
        raise ValueError(f"model.action_delta는 (0, 1] 범위여야 합니다: {delta}")
    return delta


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


def resolve_combo(cfg: dict, combo: str | None = None) -> dict:
    """콤보(생략 시 cfg['active_combo'])의 피처 리스트로 cfg['features']를 덮어쓴 새 dict를 반환한다.

    콤보 시스템(M0~M3, docs/data_pipeline.md §3-2)의 유일한 진입점 — 이후
    get_asset_features/get_market_features/get_state_dim/compute_features/
    assemble_state_matrix는 반환된 cfg만 보면 되고 콤보 개념을 몰라도 된다.
    원본 cfg는 변경하지 않는다(얕은 복사 + features만 새 dict).

    cfg에 feature_combos가 없으면(구 계약) 원본 cfg를 그대로 반환한다.
    """
    combos = cfg.get("feature_combos")
    if combos is None:
        return cfg
    name = combo if combo is not None else cfg["active_combo"]
    if name not in combos:
        raise ValueError(f"콤보 '{name}'가 feature_combos에 없습니다: {list(combos)}")
    spec = combos[name]
    resolved = dict(cfg)
    resolved["features"] = {
        **cfg.get("features", {}),
        "asset": list(spec["asset"]),
        "market": list(spec["market"]),
    }
    return resolved


def get_feature_store_dir(cfg: dict, combo: str | None = None) -> str:
    """콤보별 Feature Store 출력 디렉토리를 반환한다.

    combo가 None이거나 'full'(=active_combo 기본값)이면 기존 플랫 경로를 그대로 반환한다
    (하위호환 핵심 — env/train.py/daily.yml이 기대하는 data/feature_store/fold=*/... 를
    바꾸지 않는다). 그 외 콤보는 <base>/<combo>/ 하위 디렉토리로 분리한다.
    """
    base = cfg.get("data", {}).get("feature_store_dir", "data/feature_store")
    name = combo if combo is not None else cfg.get("active_combo")
    if name in (None, "full"):
        return base
    return str(Path(base) / name)


def get_meta_db(cfg: dict, combo: str | None = None) -> str:
    """콤보별 메타DB(sqlite) 경로를 반환한다. get_feature_store_dir와 동일한 하위호환 규칙."""
    base = cfg.get("data", {}).get("meta_db", "data/feature_store/meta.sqlite")
    name = combo if combo is not None else cfg.get("active_combo")
    if name in (None, "full"):
        return base
    p = Path(base)
    return str(p.parent / name / p.name)


def load_config_for_combo(path: str = DEFAULT_CONFIG_PATH, combo: str | None = None) -> dict:
    """config를 읽어 콤보를 반영한 dict를 반환한다 — 학습·백테스트 진입점용.

    `resolve_combo`가 `features`(지표 리스트 → state_dim)를 바꾸는 데 더해, 이 함수는
    `data.feature_store_dir`·`data.meta_db`까지 **그 콤보의 것으로 갈아끼운다.** 그래서
    하위 모듈(PortfolioEnv·load_fold_env·run_policy_on_fold·feature_store)은 콤보라는
    개념을 몰라도 되고, `cfg["data"]["feature_store_dir"]`를 읽던 기존 코드가 그대로
    올바른 콤보의 Feature Store를 읽는다.

    combo 생략 시 `active_combo`(기본 `full`)라 기존 동작과 완전히 동일하다.
    """
    return resolve_paths_for_combo(load_config(path), combo)


def resolve_paths_for_combo(cfg: dict, combo: str | None = None) -> dict:
    """이미 읽어둔 cfg에 콤보를 반영한다(`load_config_for_combo`의 dict 버전).

    원본 cfg는 변경하지 않는다 — `data`도 새 dict로 복사한 뒤 경로만 덮어쓴다.
    """
    resolved = resolve_combo(cfg, combo)
    resolved["data"] = {
        **cfg.get("data", {}),
        "feature_store_dir": get_feature_store_dir(cfg, combo),
        "meta_db": get_meta_db(cfg, combo),
    }
    return resolved


def force_utf8_stdout() -> None:
    """콘솔 출력 인코딩을 UTF-8로 고정한다 (CLI 진입점에서만 호출).

    Windows 기본 콘솔은 cp949라 MLflow가 http 트래킹 서버를 쓸 때 찍는 실행 URL 줄
    (🏃 이모지 포함)에서 UnicodeEncodeError가 난다. 이 예외가 `mlflow.end_run()` 안의
    `set_terminated` 도중에 터지면 **run이 RUNNING 상태로 남아** `select.py`의
    `status='FINISHED'` 필터에서 통째로 빠진다 — 학습은 됐는데 배포 후보로 안 잡히는
    조용한 실패라 진입점에서 미리 막는다.

    라이브러리 사용(다른 모듈이 train()을 import해 쓰는 경우)에는 전역 상태를 건드리지
    않도록 `__main__` 블록에서만 부른다.
    """
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(encoding="utf-8", errors="replace")


def get_active_combo(cfg: dict, combo: str | None = None) -> str | None:
    """이 cfg가 실제로 소비하는 콤보 이름(로깅·provenance용). 콤보 시스템이 없으면 None."""
    if cfg.get("feature_combos") is None:
        return None
    return combo if combo is not None else cfg.get("active_combo")


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
