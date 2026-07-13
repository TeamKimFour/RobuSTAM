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
from src.models.train import evaluate, load_fold_env

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
    # 균등 로짓(0벡터) → softmax는 균등비중. 초기 SHV 100%에서 균등비중으로 첫 스텝에 회전 발생.
    assert metrics["valid_avg_turnover"] > 0


def test_train_end_to_end_with_stable_baselines3(tmp_path):
    """실제 PPO 1스텝 학습 + MLflow 기록까지 도는 통합 테스트."""
    pytest.importorskip("stable_baselines3")
    pytest.importorskip("mlflow")

    out_dir = str(tmp_path / "feature_store")
    _write_fake_fold(out_dir, fold_id=0, split="train", window=30, n_rows=40, seed=1)
    _write_fake_fold(out_dir, fold_id=0, split="valid", window=30, n_rows=10, seed=2)

    cfg = {
        "assets": ASSETS,
        "window": 30,
        "transaction_cost": 0.001,
        "data": {"feature_store_dir": out_dir},
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

    from src.models.train import train

    result = train(str(config_path))

    assert "run_id" in result
    assert result["fold_id"] == 0
    assert Path(result["model_path"]).is_file()
