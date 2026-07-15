"""src.models.select 테스트 — MLflow 기록에서 배포 policy 선택.

mlflow 파일스토어에 여러 run을 실제로 심어 valid 지표 최고 run이 뽑히는지,
fold 필터·빈 결과 예외·model_path 재구성을 검증한다. mlflow는 requirements에만
있으므로 importorskip으로 가드한다(test_train과 동일 패턴).
"""

import os

import pytest

mlflow = pytest.importorskip("mlflow")

from src.models.select import select_best_run
from src.models.train import _to_tracking_uri


def _cfg(tmp_path, experiment="test-select"):
    return {
        "model": {
            "mlflow_tracking_uri": str(tmp_path / "mlruns"),
            "mlflow_experiment": experiment,
            "model_dir": str(tmp_path / "models"),
        }
    }


def _log_run(cfg, *, fold_id, fs_run_id, metrics):
    """지정 파라미터·지표로 FINISHED MLflow run 하나를 기록하고 run_id를 반환한다."""
    os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
    model_cfg = cfg["model"]
    mlflow.set_tracking_uri(_to_tracking_uri(model_cfg["mlflow_tracking_uri"]))
    mlflow.set_experiment(model_cfg["mlflow_experiment"])
    with mlflow.start_run() as run:
        mlflow.log_params({"fold_id": fold_id, "feature_store_run_id": fs_run_id})
        mlflow.log_metrics(metrics)
        return run.info.run_id


def test_selects_highest_sharpe(tmp_path):
    cfg = _cfg(tmp_path)
    _log_run(cfg, fold_id=1, fs_run_id="buildA", metrics={"valid_sharpe": 0.5})
    win = _log_run(cfg, fold_id=1, fs_run_id="buildB", metrics={"valid_sharpe": 1.7})
    _log_run(cfg, fold_id=1, fs_run_id="buildC", metrics={"valid_sharpe": 1.1})

    best = select_best_run(cfg)  # 기본 metric=valid_sharpe, higher_is_better

    assert best["mlflow_run_id"] == win
    assert best["feature_store_run_id"] == "buildB"
    assert best["fold_id"] == 1
    assert best["metric"] == "valid_sharpe"
    assert best["metric_value"] == pytest.approx(1.7)
    # model_path는 train.py 파일명 규약을 그대로 재구성해야 한다.
    assert best["model_path"].endswith(f"ppo_fold1_buildB_{win}.zip")
    assert best["model_version"] == f"ppo-fold1-{win[:8]}"


def test_metric_and_direction_override(tmp_path):
    cfg = _cfg(tmp_path)
    lo = _log_run(cfg, fold_id=1, fs_run_id="bLo", metrics={"valid_total_txn_cost": 0.01})
    _log_run(cfg, fold_id=1, fs_run_id="bHi", metrics={"valid_total_txn_cost": 0.09})

    # 비용은 작을수록 좋음 → lower_is_better
    best = select_best_run(
        cfg, metric="valid_total_txn_cost", higher_is_better=False
    )
    assert best["mlflow_run_id"] == lo


def test_fold_filter(tmp_path):
    cfg = _cfg(tmp_path)
    # fold=2가 지표는 더 높지만, fold_id=1로 필터하면 fold=1 run이 뽑혀야 한다.
    f1 = _log_run(cfg, fold_id=1, fs_run_id="b1", metrics={"valid_sharpe": 0.4})
    _log_run(cfg, fold_id=2, fs_run_id="b2", metrics={"valid_sharpe": 2.0})

    best = select_best_run(cfg, fold_id=1)
    assert best["mlflow_run_id"] == f1
    assert best["fold_id"] == 1


def test_missing_experiment_raises(tmp_path):
    cfg = _cfg(tmp_path, experiment="does-not-exist")
    # tracking_uri 자체는 유효하지만 실험이 없음 → ValueError.
    _log_run(_cfg(tmp_path, experiment="other"), fold_id=1, fs_run_id="b", metrics={"valid_sharpe": 1.0})
    with pytest.raises(ValueError, match="실험을 찾을 수 없습니다"):
        select_best_run(cfg)


def test_no_run_with_metric_raises(tmp_path):
    cfg = _cfg(tmp_path)
    _log_run(cfg, fold_id=1, fs_run_id="b", metrics={"valid_total_log_return": 0.3})
    # valid_sharpe를 기록한 run이 없음 → ValueError.
    with pytest.raises(ValueError, match="지표를 가진 FINISHED run이 없습니다"):
        select_best_run(cfg, metric="valid_sharpe")
