"""src.models.train 테스트.

load_fold_env·evaluate는 순수 로직(Feature Store I/O + PortfolioEnv)이라 항상 실행된다.
train() 자체는 stable_baselines3·mlflow가 필요해 importorskip으로 가드한다
(test_s3_sync의 moto 가드와 동일 패턴 — requirements-dev에만 있음).
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from src.data import feature_store as fs
from src.data import schema
from src.models.train import _annualized_sharpe, evaluate, load_fold_env

ASSETS = ["SPY", "EWY", "TLT", "GLD", "SHV"]
FWD_RET_COLS = [f"fwd_ret_{a}" for a in ASSETS]


def _fake_cfg(feature_store_dir: str, window: int = 30, c: float = 0.001) -> dict:
    return {
        "assets": ASSETS,
        "window": window,
        "transaction_cost": c,
        "data": {"feature_store_dir": feature_store_dir},
    }


def _write_fake_fold(out_dir: str, fold_id: int, split: str, window: int, n_rows: int, seed: int = 0):
    cols = schema.feature_names(window)
    dates = pd.date_range("2020-01-01", periods=n_rows, freq="B")
    rng = np.random.default_rng(seed)

    state_df = pd.DataFrame(
        rng.normal(0.0, 1.0, size=(n_rows, len(cols))).astype(np.float32),
        index=dates,
        columns=cols,
    )
    targets_df = pd.DataFrame(
        rng.normal(0.0, 0.01, size=(n_rows, len(ASSETS))).astype(np.float32),
        index=dates,
        columns=FWD_RET_COLS,
    )
    fs.write_features(state_df, out_dir, fold_id, split)
    fs.write_targets(targets_df, out_dir, fold_id, split)


class _FakeModel:
    """predict()만 있으면 되는 evaluate() 테스트용 더미 정책 — sb3 없이도 검증 가능."""

    def predict(self, obs, deterministic: bool = True):
        return np.zeros(len(ASSETS), dtype=np.float32), None


def test_annualized_sharpe_edge_cases():
    # 스텝 2개 미만 → 위험조정 정의 불가 → 0.0
    assert _annualized_sharpe([]) == 0.0
    assert _annualized_sharpe([0.01]) == 0.0
    # 표준편차 0(상수 보상) → 0.0 (div-by-zero 가드)
    assert _annualized_sharpe([0.01, 0.01, 0.01]) == 0.0
    # 정상: 양의 평균·양의 분산 → 양수 Sharpe, √252 연율화 반영
    s = _annualized_sharpe([0.01, 0.02, 0.03])
    assert s > 0
    arr = np.array([0.01, 0.02, 0.03])
    expected = arr.mean() / arr.std(ddof=1) * np.sqrt(252)
    assert s == pytest.approx(expected)


def test_load_fold_env_reads_feature_store(tmp_path):
    out_dir = str(tmp_path / "feature_store")
    _write_fake_fold(out_dir, fold_id=0, split="train", window=30, n_rows=10)
    cfg = _fake_cfg(out_dir)

    env = load_fold_env(cfg, fold_id=0, split="train")

    assert env.observation_space.shape == (187,)
    assert env.action_space.shape == (5,)


def test_load_fold_env_missing_fold_raises(tmp_path):
    out_dir = str(tmp_path / "feature_store")
    cfg = _fake_cfg(out_dir)

    with pytest.raises(FileNotFoundError):
        load_fold_env(cfg, fold_id=0, split="train")


def test_evaluate_runs_full_episode_with_stub_model(tmp_path):
    out_dir = str(tmp_path / "feature_store")
    n_rows = 8
    _write_fake_fold(out_dir, fold_id=0, split="valid", window=30, n_rows=n_rows)
    cfg = _fake_cfg(out_dir)

    metrics = evaluate(_FakeModel(), cfg, fold_id=0, split="valid")

    # 에피소드는 len(state_df)-1 스텝에서 종료된다 (portfolio_env.step 종료 조건).
    assert metrics["valid_n_steps"] == n_rows - 1
    assert "valid_total_log_return" in metrics
    assert "valid_total_txn_cost" in metrics
    # valid_sharpe(위험조정수익 proxy)는 항상 유한해야 한다 (선택 로직 기본 지표).
    assert "valid_sharpe" in metrics
    assert np.isfinite(metrics["valid_sharpe"])
    # 균등 로짓(0벡터) → softmax는 균등비중. 초기 SHV 100%에서 균등비중으로 첫 스텝에 회전 발생.
    assert metrics["valid_avg_turnover"] > 0


def test_train_end_to_end_with_stable_baselines3(tmp_path):
    """실제 PPO 1스텝 학습 + MLflow 기록까지 도는 통합 테스트."""
    pytest.importorskip("stable_baselines3")
    pytest.importorskip("mlflow")

    out_dir = str(tmp_path / "feature_store")
    _write_fake_fold(out_dir, fold_id=0, split="train", window=30, n_rows=40, seed=1)
    _write_fake_fold(out_dir, fold_id=0, split="valid", window=30, n_rows=10, seed=2)

    # 이 fold를 만든 build run_id를 메타DB에 심어, train이 provenance로 집어오는지 본다.
    meta_db = str(tmp_path / "feature_store" / "meta.sqlite")
    fs.init_meta_db(meta_db)
    fs.write_run(meta_db, "build-xyz", "2026-06-28T00:00:00", "abc123", 30, 187, ASSETS)
    fs.write_fold(meta_db, "build-xyz", {"fold_id": 0, "mode": "expanding"})

    cfg = {
        "assets": ASSETS,
        "window": 30,
        "transaction_cost": 0.001,
        "data": {"feature_store_dir": out_dir, "meta_db": meta_db},
        "model": {
            "algorithm": "PPO",
            "policy": "MlpPolicy",
            "fold_id": 0,
            "total_timesteps": 64,
            "n_steps": 32,  # SB3 기본 2048 대신 테스트용으로 축소
            "seed": 42,
            "mlflow_tracking_uri": str(tmp_path / "mlruns"),
            "mlflow_experiment": "test-robustam-ppo",
            "model_dir": str(tmp_path / "models"),
        },
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.dump(cfg), encoding="utf-8")

    from src.models.train import _to_tracking_uri, train

    result = train(str(config_path), seed=7)  # config의 seed=42를 인자로 오버라이드

    assert "run_id" in result
    assert result["fold_id"] == 0
    assert Path(result["model_path"]).is_file()
    # provenance(이슈 #27): build run_id가 결과·모델 파일명에 반영돼야 한다.
    assert result["feature_store_run_id"] == "build-xyz"
    assert "build-xyz" in Path(result["model_path"]).name
    # --seed 인자가 config model.seed(42)를 이기고 MLflow에도 그 값이 남아야 한다
    # (같은 fold를 seed만 바꿔 배포 후보를 늘리는 용도).
    from mlflow.tracking import MlflowClient

    assert result["seed"] == 7
    client = MlflowClient(tracking_uri=_to_tracking_uri(str(tmp_path / "mlruns")))
    assert client.get_run(result["run_id"]).data.params["seed"] == "7"
