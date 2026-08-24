"""피처 콤보(M0~M3)가 학습·백테스트 경로까지 실제로 배선됐는지 검증 (6주차 도현 2순위).

`src/data/build.py`는 콤보별 Feature Store를 만들 수 있었지만, 소비 측(train.py·policy.py·
runner.py)은 카논 6+2(=187)만 가정하고 있었다. 이 테스트는 그 경계에서 조용히 어긋날 수 있는
지점들을 박제한다.

  ① `resolve_paths_for_combo`가 features와 **data 경로**를 함께 갈아끼우는지
     (features만 바꾸면 M1 config로 full Feature Store를 읽어 차원이 어긋난다)
  ② `run_policy`의 prev_weight 슬라이스가 **콤보의 지표 개수**로 계산되는지
     — prev_weight 블록 시작 위치는 K_asset·K_market에 달려 있어, 카논 기본값으로
       잡으면 엉뚱한 칸을 덮어쓴다
  ③ 백테스트 S3 run_id가 콤보별로 갈리는지 (안 갈리면 콤보끼리 결과를 덮어쓴다)
  ④ 학습 run에 `feature_set` provenance가 남는지
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import yaml

from src.backtest.engine import BacktestEngine
from src.backtest.policy import run_policy
from src.backtest.runner import _run_id
from src.config_loader import (
    get_state_dim,
    resolve_paths_for_combo,
)
from src.data import schema

ASSETS = ["SPY", "EWY", "TLT", "GLD", "SHV"]
W = 30

M1_ASSET = ["MACD_Hist", "Rolling_Vol_20"]
M1_MARKET = ["Equity_Bond_Ratio"]


def _cfg(feature_store_dir: str = "data/feature_store") -> dict:
    """콤보 시스템이 켜진 최소 config."""
    return {
        "assets": list(ASSETS),
        "window": W,
        "transaction_cost": 0.001,
        "data": {
            "feature_store_dir": feature_store_dir,
            "meta_db": f"{feature_store_dir}/meta.sqlite",
        },
        "feature_combos": {
            "full": {
                "asset": list(schema.ASSET_FEATURES),
                "market": list(schema.MARKET_FEATURES),
            },
            "M1": {"asset": list(M1_ASSET), "market": list(M1_MARKET)},
        },
        "active_combo": "full",
        "features": {
            "asset": list(schema.ASSET_FEATURES),
            "market": list(schema.MARKET_FEATURES),
        },
    }


# ── ① resolve_paths_for_combo: features + 경로를 함께 ──

def test_resolve_paths_swaps_features_and_store_dir():
    cfg = _cfg()
    m1 = resolve_paths_for_combo(cfg, "M1")

    assert m1["features"]["asset"] == M1_ASSET
    assert m1["features"]["market"] == M1_MARKET
    # 경로도 함께 갈려야 한다 — 안 갈리면 M1 config로 full Feature Store를 읽는다.
    assert m1["data"]["feature_store_dir"].replace("\\", "/").endswith("feature_store/M1")
    assert m1["data"]["meta_db"].replace("\\", "/").endswith("feature_store/M1/meta.sqlite")
    # D = 155 + 5*K_asset + K_market = 155 + 10 + 1
    assert get_state_dim(m1) == 166


def test_resolve_paths_full_keeps_legacy_flat_path():
    """`full`은 기존 플랫 경로 그대로 — daily.yml·기존 policy.zip 하위호환의 핵심."""
    cfg = _cfg()
    full = resolve_paths_for_combo(cfg, "full")

    assert full["data"]["feature_store_dir"] == "data/feature_store"
    assert full["data"]["meta_db"] == "data/feature_store/meta.sqlite"
    assert get_state_dim(full) == 187


def test_resolve_paths_does_not_mutate_original():
    cfg = _cfg()
    resolve_paths_for_combo(cfg, "M1")

    assert cfg["features"]["asset"] == list(schema.ASSET_FEATURES)
    assert cfg["data"]["feature_store_dir"] == "data/feature_store"


# ── ② prev_weight 슬라이스가 콤보 차원을 따라가는지 ──

class _RecordingModel:
    """매 스텝 obs 전체를 기록하는 스텁 정책(관측 배치를 사후 검증하기 위함)."""

    def __init__(self):
        self.seen_obs: list[np.ndarray] = []

    def predict(self, obs, deterministic: bool = True):
        self.seen_obs.append(np.array(obs, dtype=np.float64))
        return np.zeros(len(ASSETS), dtype=np.float32), None


def _m1_state(n_rows: int) -> pd.DataFrame:
    """M1(D=166) Feature Store 산출물 모양의 State — prev_weight 칸은 build와 동일하게 0."""
    cols = schema.feature_names(W, M1_ASSET, M1_MARKET)
    idx = pd.date_range("2020-01-01", periods=n_rows, freq="B")
    return pd.DataFrame(np.zeros((n_rows, len(cols)), dtype=np.float32), index=idx, columns=cols)


def test_run_policy_injects_prev_weight_at_combo_slice(tmp_path):
    """M1(166차원)에서도 prev_weight가 **그 콤보의** 자리에 주입돼야 한다.

    prev_weight 블록 시작 위치는 K_asset·K_market에 달려 있다. 카논(6+2) 기본값으로
    슬라이스를 잡으면 166차원 obs에서 [182:187]을 가리켜 완전히 빗나간다.
    """
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.dump(_cfg()), encoding="utf-8")

    n = 4
    state_df = _m1_state(n)
    price_returns = pd.DataFrame(
        np.zeros((n, len(ASSETS))), index=state_df.index, columns=ASSETS
    )
    model = _RecordingModel()
    engine = BacktestEngine(initial_nav=1_000_000, config_path=str(config_path))

    run_policy(price_returns, state_df, model, engine, str(config_path), combo="M1")

    m1_slice = schema.prev_weight_slice(W, len(M1_ASSET), len(M1_MARKET))
    assert m1_slice == slice(161, 166)

    # 첫 스텝은 콜드스타트(SHV 100%) — env.reset과 같은 관례.
    first = model.seen_obs[0]
    assert len(first) == 166
    expected = np.zeros(len(ASSETS))
    expected[ASSETS.index("SHV")] = 1.0
    np.testing.assert_allclose(first[m1_slice], expected)

    # 카논 기본 슬라이스는 이 obs 범위 밖이라 아무것도 못 집는다 —
    # 콤보 개수를 안 넘기면 주입 자체가 깨진다는 것을 명시적으로 박제한다.
    assert first[schema.prev_weight_slice(W)].size == 0


def test_run_policy_rejects_state_df_from_other_combo(tmp_path):
    """full Feature Store를 M1이라고 넘기면 조용히 돌지 않고 막아야 한다."""
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.dump(_cfg()), encoding="utf-8")

    n = 3
    idx = pd.date_range("2020-01-01", periods=n, freq="B")
    full_state = pd.DataFrame(
        np.zeros((n, 187), dtype=np.float32), index=idx, columns=schema.feature_names(W)
    )
    price_returns = pd.DataFrame(np.zeros((n, len(ASSETS))), index=idx, columns=ASSETS)
    engine = BacktestEngine(initial_nav=1_000_000, config_path=str(config_path))

    with pytest.raises(ValueError, match="일치하지 않습니다"):
        run_policy(
            price_returns, full_state, _RecordingModel(), engine, str(config_path), combo="M1"
        )


# ── ③ S3 run_id가 콤보별로 갈리는지 ──

def test_run_id_keeps_legacy_format_for_full_and_none():
    """프론트·export.py가 기대하는 기존 키를 바꾸지 않는다."""
    assert _run_id(1, "policy") == "fold1_policy"
    assert _run_id(1, "policy", None) == "fold1_policy"
    assert _run_id(1, "policy", "full") == "fold1_policy"


def test_run_id_is_namespaced_per_combo():
    """콤보끼리 같은 fold 결과를 덮어쓰면 안 된다."""
    assert _run_id(1, "policy", "M1") == "M1_fold1_policy"
    assert _run_id(2, "1n", "M3") == "M3_fold2_1n"
    assert _run_id(1, "policy", "M1") != _run_id(1, "policy", "M2")


# ── ④ 학습 run provenance ──

def test_train_logs_feature_set_and_state_dim(tmp_path):
    """콤보로 학습한 run에는 feature_set·state_dim이 남아야 한다(실험 비교의 그룹 키)."""
    pytest.importorskip("stable_baselines3")
    pytest.importorskip("mlflow")

    from mlflow.tracking import MlflowClient

    from src.data import feature_store as fs
    from src.models.train import _to_tracking_uri, train

    # M1 콤보의 Feature Store를 그 콤보 경로에 심는다.
    base = tmp_path / "feature_store"
    combo_dir = base / "M1"
    cols = schema.feature_names(W, M1_ASSET, M1_MARKET)
    rng = np.random.default_rng(0)
    for split, n_rows in (("train", 40), ("valid", 10)):
        idx = pd.date_range("2020-01-01", periods=n_rows, freq="B")
        part = pd.DataFrame(
            rng.normal(size=(n_rows, len(cols))).astype(np.float32), index=idx, columns=cols
        )
        tgt = pd.DataFrame(
            rng.normal(scale=0.01, size=(n_rows, len(ASSETS))),
            index=idx,
            columns=[f"fwd_ret_{a}" for a in ASSETS],
        )
        fs.write_features(part, str(combo_dir), 0, split, expected_columns=cols)
        fs.write_targets(tgt, str(combo_dir), 0, split)

    meta_db = str(combo_dir / "meta.sqlite")
    fs.init_meta_db(meta_db)
    fs.write_run(meta_db, "build-m1", "2026-08-09T00:00:00", "hash-m1", W, 166, ASSETS, combo="M1")
    fs.write_fold(meta_db, "build-m1", {"fold_id": 0, "mode": "expanding"})

    cfg = _cfg(str(base))
    cfg["model"] = {
        "algorithm": "PPO",
        "fold_id": 0,
        "total_timesteps": 8,
        "n_steps": 8,
        "seed": 42,
        "mlflow_tracking_uri": str(tmp_path / "mlruns"),
        "mlflow_experiment": "test-combo-wiring",
        "model_dir": str(tmp_path / "models"),
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.dump(cfg), encoding="utf-8")

    result = train(str(config_path), combo="M1")

    assert result["feature_set"] == "M1"
    assert result["state_dim"] == 166
    # 그 콤보의 meta.sqlite에서 build run_id를 찾아야 한다(full 것을 집으면 안 됨).
    assert result["feature_store_run_id"] == "build-m1"
    assert "M1" in result["model_path"]

    params = MlflowClient(
        tracking_uri=_to_tracking_uri(str(tmp_path / "mlruns"))
    ).get_run(result["run_id"]).data.params
    assert params["feature_set"] == "M1"
    assert params["state_dim"] == "166"
    assert params["feature_store_run_id"] == "build-m1"


def test_train_lambda_kappa_args_override_config(tmp_path):
    """λ·κ를 인자로 넘기면 config 값을 이겨야 한다.

    실험 러너는 config 파일을 건드리지 않고 모든 콤보에 같은 λ·κ를 강제해야 한다.
    로드된 cfg dict를 고치는 방식은 train이 파일을 다시 읽어서 조용히 무시된다 —
    그래서 인자 경로가 실제로 MLflow까지 도달하는지 박제한다.
    """
    pytest.importorskip("stable_baselines3")
    pytest.importorskip("mlflow")

    from mlflow.tracking import MlflowClient

    from src.data import feature_store as fs
    from src.models.train import _to_tracking_uri, train

    base = tmp_path / "feature_store"
    cols = schema.feature_names(W)
    rng = np.random.default_rng(1)
    for split, n_rows in (("train", 40), ("valid", 10)):
        idx = pd.date_range("2020-01-01", periods=n_rows, freq="B")
        part = pd.DataFrame(
            rng.normal(size=(n_rows, len(cols))).astype(np.float32), index=idx, columns=cols
        )
        tgt = pd.DataFrame(
            rng.normal(scale=0.01, size=(n_rows, len(ASSETS))),
            index=idx,
            columns=[f"fwd_ret_{a}" for a in ASSETS],
        )
        fs.write_features(part, str(base), 0, split, expected_columns=cols)
        fs.write_targets(tgt, str(base), 0, split)

    cfg = _cfg(str(base))
    cfg["model"] = {
        "algorithm": "PPO",
        "fold_id": 0,
        "total_timesteps": 8,
        "n_steps": 8,
        "seed": 42,
        "train_cost_multiplier": 1.0,   # config 값 — 인자가 이겨야 한다
        "vol_penalty_coef": 0.0,
        "mlflow_tracking_uri": str(tmp_path / "mlruns"),
        "mlflow_experiment": "test-lambda-override",
        "model_dir": str(tmp_path / "models"),
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.dump(cfg), encoding="utf-8")

    result = train(str(config_path), cost_multiplier=10.0, vol_penalty_coef=0.5)

    params = MlflowClient(
        tracking_uri=_to_tracking_uri(str(tmp_path / "mlruns"))
    ).get_run(result["run_id"]).data.params
    assert float(params["train_cost_multiplier"]) == pytest.approx(10.0)
    assert float(params["vol_penalty_coef"]) == pytest.approx(0.5)
