"""피처 콤보 실험 러너 검증 — 이슈 #34 Phase 2 (도현).

두 층으로 나눠 본다.
  ① 판정 로직: `judge_combo`·`weight_dispersion`·`fold_metrics_from_run`은 순수 함수라
     학습 없이 검증한다. 관문(비중편차·회전율) → 성과(vs 1/N 샤프) 2단계 판정과 기준값
     override가 계약대로 도는지 박제한다.
  ② 오케스트레이션: `run_experiment`가 콤보별 학습→백테스트→판정을 실제로 이어 붙이는지,
     provenance가 콤보별로 갈리는지, build 없는 콤보에서 학습 전에 멈추는지.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
import yaml

from src.data import schema
from src.models.experiment import (
    Criteria,
    FoldMetrics,
    fold_metrics_from_run,
    judge_combo,
    weight_dispersion,
)

ASSETS = ["SPY", "EWY", "TLT", "GLD", "SHV"]


# ── weight_dispersion ─────────────────────────────────────────────────

def _policy_nav(mean_weights: dict[str, float], n: int = 10) -> pd.DataFrame:
    """자산별 (상수) 평균비중을 갖는 policy NAV DataFrame 스텁."""
    data = {"nav": np.linspace(1e6, 1.1e6, n), "turnover": np.zeros(n)}
    for a in ASSETS:
        data[a] = np.full(n, mean_weights[a])
    return pd.DataFrame(data)


def test_weight_dispersion_is_max_minus_min_of_asset_means():
    df = _policy_nav({"SPY": 0.30, "EWY": 0.20, "TLT": 0.20, "GLD": 0.15, "SHV": 0.15})
    assert weight_dispersion(df, ASSETS) == pytest.approx(0.30 - 0.15, abs=1e-12)


def test_weight_dispersion_zero_for_equal_weight():
    df = _policy_nav({a: 0.2 for a in ASSETS})  # 1/N 흉내
    assert weight_dispersion(df, ASSETS) == pytest.approx(0.0, abs=1e-12)


def test_weight_dispersion_missing_column_raises():
    df = _policy_nav({a: 0.2 for a in ASSETS}).drop(columns=["SHV"])
    with pytest.raises(ValueError, match="비중 컬럼"):
        weight_dispersion(df, ASSETS)


# ── judge_combo: 관문(gate) ────────────────────────────────────────────

def _folds(dispersion, turnover, beats, gap) -> list[FoldMetrics]:
    """스칼라 또는 리스트를 받아 3 fold FoldMetrics를 만든다."""
    def as3(x):
        return x if isinstance(x, (list, tuple)) else [x, x, x]
    d, t, b, g = as3(dispersion), as3(turnover), as3(beats), as3(gap)
    return [FoldMetrics(i + 1, d[i], t[i], b[i], g[i]) for i in range(3)]


def test_adopted_when_gate_and_performance_pass():
    out = judge_combo(_folds(0.06, 0.25, True, 0.05))
    assert out["adopted"] is True
    assert out["gate_passed"] and out["performance_passed"]


def test_gate_fails_when_dispersion_mean_too_low():
    """편차 평균이 낮으면(1/N 흉내) 성과와 무관하게 기각."""
    out = judge_combo(_folds(0.04, 0.25, True, 0.05))
    assert out["gate_passed"] is False
    assert out["adopted"] is False
    assert any("비중편차" in r for r in out["reasons"])


def test_gate_fails_when_any_fold_dispersion_collapses():
    """평균은 통과해도 한 fold라도 붕괴선 아래면 기각."""
    out = judge_combo(_folds([0.08, 0.08, 0.02], 0.25, True, 0.05))
    assert out["gate"]["dispersion_min"] == pytest.approx(0.02)
    assert out["gate_passed"] is False


def test_gate_fails_when_turnover_too_high():
    out = judge_combo(_folds(0.06, 0.40, True, 0.05))
    assert out["gate_passed"] is False
    assert any("회전율" in r for r in out["reasons"])


# ── judge_combo: 성과(performance) ─────────────────────────────────────

def test_performance_fails_when_too_few_folds_beat_1n():
    """관문은 통과해도 vs 1/N 달성 fold가 부족하면 기각."""
    out = judge_combo(_folds(0.06, 0.25, [True, False, False], 0.05))
    assert out["gate_passed"] is True
    assert out["performance_passed"] is False
    assert out["performance"]["folds_beating_1n"] == 1


def test_performance_fails_when_worst_fold_gap_too_negative():
    """최악 fold의 샤프 격차가 재앙적이면 기각(단일 fold 요행 방지)."""
    out = judge_combo(_folds(0.06, 0.25, [True, True, False], [0.1, 0.1, -0.3]))
    assert out["gate_passed"] is True
    assert out["performance_passed"] is False
    assert out["performance"]["worst_sharpe_gap_vs_1n"] == pytest.approx(-0.3)


def test_two_of_three_folds_beating_is_enough():
    out = judge_combo(_folds(0.06, 0.25, [True, True, False], [0.05, 0.05, -0.05]))
    assert out["adopted"] is True


# ── 기준값 override (회의 확정 후 조정 대비) ──────────────────────────

def test_criteria_override_changes_verdict():
    folds = _folds(0.04, 0.25, True, 0.05)  # 기본 기준(0.05)엔 편차 미달
    assert judge_combo(folds)["adopted"] is False
    # 관문 편차 기준을 0.03으로 낮추면 통과
    loose = Criteria(min_dispersion_mean=0.03, min_dispersion_any_fold=0.02)
    assert judge_combo(folds, loose)["adopted"] is True


def test_empty_folds_raises():
    with pytest.raises(ValueError, match="folds"):
        judge_combo([])


# ── fold_metrics_from_run: runner.run_fold 출력 → FoldMetrics ──────────

def test_fold_metrics_from_run_extracts_normalized_fields():
    run_result = {
        "fold_id": 2,
        "strategies": {
            "RL policy": {"sharpe": 0.80, "avg_turnover": 0.27},
            "1/N": {"sharpe": 0.90},
        },
        "nav_by_strategy": {
            "RL policy": _policy_nav(
                {"SPY": 0.28, "EWY": 0.20, "TLT": 0.20, "GLD": 0.17, "SHV": 0.15}
            )
        },
        "comparison": {"1/N": {"beats_target": False}},
    }
    fm = fold_metrics_from_run(run_result, ASSETS)
    assert fm.fold_id == 2
    assert fm.avg_turnover == pytest.approx(0.27)
    assert fm.beats_1n is False
    assert fm.sharpe_gap_vs_1n == pytest.approx(0.80 - 0.90, abs=1e-12)
    assert fm.weight_dispersion == pytest.approx(0.28 - 0.15, abs=1e-12)


# ── run_experiment 오케스트레이션 (end-to-end) ─────────────────────────
#
# 판정 로직과 달리 여기는 train→backtest→judge 배선이 실제로 이어지는지를 본다.
# 콤보마다 State 차원이 다르므로(full=187, M1=166) 배선이 한 군데라도 어긋나면 깨진다.

W = 30
M1_ASSET = ["MACD_Hist", "Rolling_Vol_20"]
M1_MARKET = ["Equity_Bond_Ratio"]


def _e2e_cfg(feature_store_dir: str, tmp_path) -> dict:
    """실험 러너를 한 fold 굴리는 데 필요한 최소 config."""
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
        "split": {
            "anchor_start": "2010-01-01",
            "test_blocks": [["2020-01-01", "2021-12-31"]],
            "embargo_days": 1,
            "valid_days": 5,
        },
        "model": {
            "algorithm": "PPO",
            "total_timesteps": 8,
            "n_steps": 8,
            "mlflow_tracking_uri": str(tmp_path / "mlruns"),
            "mlflow_experiment": "test-experiment-e2e",
            "model_dir": str(tmp_path / "models"),
        },
    }


def _seed_combo_store(base, combo: str, asset_feats, market_feats, fold_id: int, seed: int) -> str:
    """콤보 하나의 Feature Store(train/valid/test + meta)를 통째로 심는다."""
    from src.data import feature_store as fs

    store = base if combo == "full" else base / combo
    cols = schema.feature_names(W, asset_feats, market_feats)
    rng = np.random.default_rng(seed)
    for split, n_rows in (("train", 40), ("valid", 12), ("test", 12)):
        idx = pd.date_range("2020-01-01", periods=n_rows, freq="B")
        part = pd.DataFrame(
            rng.normal(size=(n_rows, len(cols))).astype(np.float32), index=idx, columns=cols
        )
        tgt = pd.DataFrame(
            rng.normal(scale=0.01, size=(n_rows, len(ASSETS))),
            index=idx,
            columns=[f"fwd_ret_{a}" for a in ASSETS],
        )
        fs.write_features(part, str(store), fold_id, split, expected_columns=cols)
        fs.write_targets(tgt, str(store), fold_id, split)

    meta_db = str(store / "meta.sqlite")
    fs.init_meta_db(meta_db)
    run_id = f"build-{combo}"
    fs.write_run(
        meta_db, run_id, "2026-08-09T00:00:00", f"hash-{combo}",
        W, len(cols), ASSETS, combo=combo,
    )
    fs.write_fold(meta_db, run_id, {"fold_id": fold_id, "mode": "expanding"})
    return run_id


def test_run_experiment_end_to_end_two_combos(tmp_path):
    """full·M1 두 콤보를 실제로 학습→백테스트→판정까지 굴린다.

    성능은 보지 않는다 — 오케스트레이션이 끝까지 도는지와 provenance가 콤보별로
    갈리는지만 확인한다.
    """
    pytest.importorskip("stable_baselines3")
    pytest.importorskip("mlflow")

    from src.models.experiment import ExperimentSettings, run_experiment

    base = tmp_path / "feature_store"
    fold_id = 1
    _seed_combo_store(base, "full", schema.ASSET_FEATURES, schema.MARKET_FEATURES, fold_id, 1)
    _seed_combo_store(base, "M1", M1_ASSET, M1_MARKET, fold_id, 2)

    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.dump(_e2e_cfg(str(base), tmp_path)), encoding="utf-8")

    out_path = tmp_path / "verdict.json"
    results = run_experiment(
        combos=["full", "M1"],
        settings=ExperimentSettings(fold_ids=[fold_id], seed=7, total_timesteps=8, lam=10.0),
        config_path=str(config_path),
        out_path=str(out_path),
    )

    assert set(results["combos"]) == {"full", "M1"}
    # provenance가 콤보별로 갈려야 한다 — 한쪽 Feature Store를 공유하면 안 된다.
    assert results["combos"]["full"]["provenance"]["build_run_id"] == "build-full"
    assert results["combos"]["M1"]["provenance"]["build_run_id"] == "build-M1"
    assert results["combos"]["full"]["provenance"]["state_dim"] == 187
    assert results["combos"]["M1"]["provenance"]["state_dim"] == 166
    # 동일 조건이 그대로 기록돼야 한다(비교 실험의 전제).
    assert results["settings"]["seed"] == 7
    assert results["settings"]["lam"] == 10.0
    for combo in ("full", "M1"):
        verdict = results["combos"][combo]["verdict"]
        assert isinstance(verdict["adopted"], bool)
        assert verdict["per_fold"][0]["fold_id"] == fold_id
        assert verdict["reasons"]  # 사람이 읽을 사유가 항상 하나는 있어야 한다

    saved = json.loads(out_path.read_text(encoding="utf-8"))
    assert saved["combos"].keys() == results["combos"].keys()


def test_run_experiment_logs_test_metrics_to_mlflow(tmp_path):
    """백테스트(test) 지표가 학습 run에 되붙어야 한다.

    train.py는 valid split만 기록한다. 팀 주간계획 기록 항목과 형우 시각화가 요구하는
    test 샤프·MDD·회전율·비중편차는 백테스트에서 나오므로, JSON에만 남기면 팀 공용
    MLflow만 보고는 콤보를 비교할 수 없다.
    """
    pytest.importorskip("stable_baselines3")
    pytest.importorskip("mlflow")

    from mlflow.tracking import MlflowClient

    from src.models.experiment import ExperimentSettings, run_experiment
    from src.models.train import _to_tracking_uri

    base = tmp_path / "feature_store"
    _seed_combo_store(base, "M1", M1_ASSET, M1_MARKET, 1, 3)

    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.dump(_e2e_cfg(str(base), tmp_path)), encoding="utf-8")

    results = run_experiment(
        combos=["M1"],
        settings=ExperimentSettings(fold_ids=[1], seed=7, total_timesteps=8),
        config_path=str(config_path),
        out_path=str(tmp_path / "verdict.json"),
    )

    mlflow_run_id = results["combos"]["M1"]["runs"][0]["mlflow_run_id"]
    run = MlflowClient(
        tracking_uri=_to_tracking_uri(str(tmp_path / "mlruns"))
    ).get_run(mlflow_run_id)

    # 팀 주간계획 기록 항목 — valid만이 아니라 test 쪽도 전부 있어야 한다.
    for key in (
        "test_sharpe",
        "test_mdd",
        "test_avg_turnover",
        "test_weight_dispersion",
        "test_total_cost",
        "test_sharpe_gap_vs_1n",
        "test_bench_1n_sharpe",
        "verdict_adopted",
    ):
        assert key in run.data.metrics, f"MLflow에 {key}가 없습니다"

    # 학습 시점 params도 그대로 남아 있어야 한다(되붙이기가 덮어쓰지 않는지).
    assert run.data.params["feature_set"] == "M1"
    assert run.data.params["state_dim"] == "166"
    assert "valid_sharpe" in run.data.metrics

    # JSON과 MLflow의 값이 일치해야 한다(두 기록이 갈라지면 비교가 깨진다).
    verdict = results["combos"]["M1"]["verdict"]
    assert run.data.metrics["verdict_adopted"] == float(verdict["adopted"])
    assert run.data.metrics["test_weight_dispersion"] == pytest.approx(
        verdict["per_fold"][0]["weight_dispersion"]
    )
    assert run.data.tags["verdict"] in ("ADOPTED", "REJECTED")


def test_run_experiment_fails_fast_on_missing_build(tmp_path):
    """build 안 된 콤보가 섞이면 학습을 시작하기 전에 멈춰야 한다.

    세 번째 콤보에서 없다는 걸 알게 돼 앞의 학습 시간을 통째로 버리는 일을 막는 계약이다.
    """
    from src.models.experiment import ExperimentSettings, run_experiment

    base = tmp_path / "feature_store"
    _seed_combo_store(base, "full", schema.ASSET_FEATURES, schema.MARKET_FEATURES, 1, 1)

    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.dump(_e2e_cfg(str(base), tmp_path)), encoding="utf-8")

    # M1은 심지 않았다 → 학습 전에 RuntimeError로 막혀야 한다.
    with pytest.raises(RuntimeError, match="Feature Store가 없습니다"):
        run_experiment(
            combos=["full", "M1"],
            settings=ExperimentSettings(fold_ids=[1], total_timesteps=8),
            config_path=str(config_path),
            out_path=str(tmp_path / "verdict.json"),
        )
