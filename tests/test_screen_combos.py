"""screen_combos 테스트 — 콤보 전용 Feature Store 스크리닝 + MLflow 기록.

(pandas·pyarrow·scikit-learn·mlflow 필요) 합성 Feature Store에 알려진 선형 신호를 심어
mean_abs_ic가 이를 잡는지, screen_combo가 build 메타(run_id)와 올바르게 연결되는지,
log_to_mlflow가 로컬 파일스토어(tmp_path)에 params/metrics/artifacts를 남기는지 확인한다.
"""

from datetime import datetime, timezone

import numpy as np
import pytest

pd = pytest.importorskip("pandas")
pytest.importorskip("pyarrow")
pytest.importorskip("sklearn")
pytest.importorskip("mlflow")

from src.data import feature_store as fs
from src.data import schema
from src.data import screen_combos as sc

ASSETS = list(schema.ASSETS)
W = 30
M0_ASSET, M0_MARKET = [], ["Equity_Bond_Ratio"]
M1_ASSET, M1_MARKET = ["MACD_Hist", "Rolling_Vol_20"], ["Equity_Bond_Ratio"]


def _write_fold(out_dir, fold, asset_feats, market_feats, plant_signal=False, n_train=80, n_test=40, seed=0):
    cols = schema.feature_names(W, asset_features=asset_feats, market_features=market_feats)
    rng = np.random.default_rng(seed + fold)
    for split, n in (("train", n_train), ("test", n_test)):
        idx = pd.bdate_range("2015-01-01", periods=n) if split == "train" \
            else pd.bdate_range("2019-01-01", periods=n)
        feats = pd.DataFrame(rng.normal(size=(n, len(cols))), index=idx, columns=cols)
        tgt_idx = idx[:-1]
        tgts = pd.DataFrame(
            {f"fwd_ret_{a}": rng.normal(0, 0.01, len(tgt_idx)) for a in ASSETS}, index=tgt_idx
        )
        if plant_signal and market_feats:
            feats.loc[tgt_idx, f"mkt_{market_feats[0]}"] = tgts["fwd_ret_SPY"].to_numpy() * 3.0
        fs.write_features(feats, str(out_dir), fold, split)
        fs.write_targets(tgts, str(out_dir), fold, split)


def _cfg(tmp_path, active="M0"):
    return {
        "assets": ASSETS,
        "window": W,
        "transaction_cost": 0.001,
        "data": {
            "feature_store_dir": str(tmp_path / "fs"),
            "meta_db": str(tmp_path / "fs" / "meta.sqlite"),
        },
        "feature_combos": {
            "M0": {"asset": M0_ASSET, "market": M0_MARKET},
            "M1": {"asset": M1_ASSET, "market": M1_MARKET},
        },
        "active_combo": active,
        "features": {
            "params": {
                "sma_fast": 5, "sma_slow": 20, "rsi_length": 14, "rsi_28_length": 28,
                "macd": [12, 26, 9], "vol_window": 20, "bbands": [20, 2], "roc_length": 10,
                "annualize_vol": False, "ebr_ma_window": 20, "gvr_long_window": 60,
                "drawdown_lookback": 60,
            }
        },
        "normalize": {"returns_scope": "per_asset", "feature_scope": "per_column", "eps": 1e-8},
        "split": {"anchor_start": "2010-01-01", "valid_days": 252, "embargo_days": 34},
        "model": {
            "mlflow_tracking_uri": str(tmp_path / "mlruns"),
            "mlflow_screening_experiment": "test-screening",
        },
    }


def _seed_build_meta(cfg, combo):
    """screen_combo이 참조하는 build 메타(runs 테이블)를 미리 심어둔다(build() 없이 스크리닝만 테스트)."""
    from src.config_loader import get_feature_store_dir, get_meta_db, get_state_dim, resolve_combo

    resolved = resolve_combo(cfg, combo)
    db = get_meta_db(cfg, combo)
    fs.init_meta_db(db)
    run_id = f"{fs.config_hash(resolved)}-test"
    fs.write_run(
        db, run_id, datetime.now(timezone.utc).isoformat(), fs.config_hash(resolved),
        W, get_state_dim(resolved), ASSETS, combo=combo,
    )
    return get_feature_store_dir(cfg, combo), run_id


def test_mean_abs_ic_recovers_planted_signal(tmp_path):
    out_dir, _ = _seed_build_meta(_cfg(tmp_path), "M0")
    for f in (1, 2, 3):
        _write_fold(out_dir, f, M0_ASSET, M0_MARKET, plant_signal=True)
    mean_abs_ic, sign_stable_frac, detail = sc._mean_abs_ic(out_dir, M0_ASSET, M0_MARKET)
    assert mean_abs_ic > 0.8  # 3배 선형 신호를 심었으니 강하게 잡혀야 함
    assert sign_stable_frac == 1.0  # 항상 같은 부호로 심었으니 안정적이어야 함
    assert list(detail.index) == ["Equity_Bond_Ratio"]
    assert detail.loc["Equity_Bond_Ratio", "sign_stable"] == True  # noqa: E712


def test_mean_abs_ic_near_zero_on_noise(tmp_path):
    out_dir, _ = _seed_build_meta(_cfg(tmp_path), "M1")
    for f in (1, 2, 3):
        _write_fold(out_dir, f, M1_ASSET, M1_MARKET, plant_signal=False)
    mean_abs_ic, _, _ = sc._mean_abs_ic(out_dir, M1_ASSET, M1_MARKET)
    assert mean_abs_ic < 0.3


def test_screen_combo_raises_without_build(tmp_path):
    """build() 없이(meta 미기록) screen_combo를 부르면 명확한 에러가 나야 한다."""
    cfg = _cfg(tmp_path)
    out_dir = cfg["data"]["feature_store_dir"] + "/M0"
    for f in (1, 2, 3):
        _write_fold(out_dir, f, M0_ASSET, M0_MARKET)
    with pytest.raises(RuntimeError, match="build"):
        sc.screen_combo(cfg, "M0")


def test_screen_combo_end_to_end_shape(tmp_path):
    cfg = _cfg(tmp_path)
    out_dir, run_id = _seed_build_meta(cfg, "M0")
    for f in (1, 2, 3):
        _write_fold(out_dir, f, M0_ASSET, M0_MARKET, plant_signal=True)

    result = sc.screen_combo(cfg, "M0")
    assert result["combo"] == "M0"
    assert result["params"]["feature_store_run_id"] == run_id
    assert result["params"]["state_dim"] == 156
    assert result["params"]["asset_features"] == "(none)"
    assert result["params"]["market_features"] == "Equity_Bond_Ratio"
    assert result["params"]["verdict"] == "BASELINE"  # baseline_metrics 미지정 시
    assert set(result["metrics"]) == {
        "mean_abs_ic", "sign_stable_frac", "n_unstable_features", "ridge_rank_ic", "ridge_r2",
        "gbm_rank_ic", "gbm_r2", "max_abs_z", "n_distribution_issues", "passed",
    }
    assert result["metrics"]["passed"] == 1.0
    assert len(result["artifacts"]["feature_manifest"]) == 156
    assert "sign_stable" in result["artifacts"]["ic_detail"].columns
    assert "worst_extreme_col" in result["artifacts"]["distribution_report"].columns


def test_screen_combo_verdict_against_baseline(tmp_path):
    """M0에 강한 신호를 심으면 약한 baseline(ridge/gbm rank-IC)보다 높아 PASS해야 한다.

    verdict는 ridge_rank_ic 또는 gbm_rank_ic 중 하나라도 baseline 이상이면 PASS
    (PR #69 리뷰 반영 — sign_stable_frac은 콤보마다 분모가 달라 게이트에서 제외).
    """
    cfg = _cfg(tmp_path)
    out_dir, _ = _seed_build_meta(cfg, "M0")
    for f in (1, 2, 3):
        _write_fold(out_dir, f, M0_ASSET, M0_MARKET, plant_signal=True)

    result_pass = sc.screen_combo(
        cfg, "M0", baseline_metrics={"ridge_rank_ic": 0.01, "gbm_rank_ic": 0.01}
    )
    assert result_pass["params"]["verdict"] == "PASS"
    assert result_pass["metrics"]["passed"] == 1.0
    assert result_pass["metrics"]["n_unstable_features"] == 0.0

    result_fail = sc.screen_combo(
        cfg, "M0", baseline_metrics={"ridge_rank_ic": 999.0, "gbm_rank_ic": 999.0}
    )
    assert result_fail["params"]["verdict"] == "FAIL"
    assert result_fail["metrics"]["passed"] == 0.0


def test_log_to_mlflow_writes_params_metrics_artifacts(tmp_path):
    cfg = _cfg(tmp_path)
    out_dir, _ = _seed_build_meta(cfg, "M0")
    for f in (1, 2, 3):
        _write_fold(out_dir, f, M0_ASSET, M0_MARKET, plant_signal=True)
    result = sc.screen_combo(cfg, "M0")

    run_id = sc.log_to_mlflow(cfg, result)

    import mlflow

    client = mlflow.tracking.MlflowClient(tracking_uri=str(tmp_path / "mlruns"))
    run = client.get_run(run_id)
    assert run.data.params["feature_set"] == "M0"
    assert run.data.params["run_type"] == "screening"
    assert pytest.approx(run.data.metrics["ridge_rank_ic"]) == result["metrics"]["ridge_rank_ic"]
    artifact_names = {a.path for a in client.list_artifacts(run_id)}
    assert artifact_names == {
        "feature_manifest.json", "resolved_config.yaml",
        "screening_result.csv", "distribution_report.csv",
    }


def test_log_to_mlflow_uses_separate_experiment_from_ppo(tmp_path):
    """스크리닝 run은 robustam-ppo(RL 학습)와 분리된 별도 experiment에 쌓여야 한다(PR #69 리뷰)."""
    cfg = _cfg(tmp_path)
    cfg["model"]["mlflow_screening_experiment"] = "my-screening-experiment"
    out_dir, _ = _seed_build_meta(cfg, "M0")
    for f in (1, 2, 3):
        _write_fold(out_dir, f, M0_ASSET, M0_MARKET, plant_signal=True)
    result = sc.screen_combo(cfg, "M0")
    run_id = sc.log_to_mlflow(cfg, result)

    import mlflow

    client = mlflow.tracking.MlflowClient(tracking_uri=str(tmp_path / "mlruns"))
    run = client.get_run(run_id)
    experiment = client.get_experiment(run.info.experiment_id)
    assert experiment.name == "my-screening-experiment"
    assert experiment.name != cfg["model"].get("mlflow_experiment", "robustam-ppo")
