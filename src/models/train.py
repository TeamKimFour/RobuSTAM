"""RobuSTAM PPO 학습 스크립트 (RunPod GPU 실행 대상, docs/model_training.md 참고).

Feature Store의 특정 fold train split을 PortfolioEnv에 태워 Stable-Baselines3 PPO를
학습하고, MLflow에 파라미터·검증지표·policy 아티팩트를 기록한다.

실행 전제: `python -m src.data.build`로 Feature Store(data/feature_store/)가 이미
채워져 있어야 한다(민지 파이프라인 산출물, gitignore됨). RunPod 등 새 인스턴스에서는
`docker/Dockerfile.train`이 collect→build→train을 이어서 실행한다.

무거운 의존성(stable_baselines3·mlflow·torch)은 requirements.txt에 있지만 학습에만
쓰이므로, import는 실제로 학습을 실행하는 `train()` 안에서만 한다 — 이 모듈을
가볍게 import(예: 다른 스크립트가 `load_fold_env`만 재사용)해도 무거운 초기화가
따라오지 않게 하기 위함.

실행:
    python -m src.models.train                              # config 기본 fold·timesteps
    python -m src.models.train --fold-id 1 --total-timesteps 500000
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from src.config_loader import load_config
from src.data import feature_store as fs
from src.env.portfolio_env import PortfolioEnv


class _BoundedActionWrapper(gym.ActionWrapper):
    """PortfolioEnv.action_space(-inf~+inf 로짓 Box)에 유한 bound를 씌운다.

    SB3 PPO는 연속 행동공간에 유한한 상하한을 요구하지만, env 쪽 무한 bound는
    "정책망 로짓을 그대로 받고 softmax는 step() 내부에서 처리한다"는 팀 계약
    (docs/state_spec.md·portfolio_env.py 문서화)이라 env는 건드리지 않는다.
    이 래퍼는 SB3에 보여줄 bound만 좁히고 값은 그대로 통과시킨다 — softmax는
    스케일에 민감하므로 ±bound(기본 10)면 사실상 원핫에 가까운 선호까지 표현 가능하다.
    """

    def __init__(self, env: gym.Env, bound: float = 10.0) -> None:
        super().__init__(env)
        self.action_space = spaces.Box(
            low=-bound, high=bound, shape=env.action_space.shape, dtype=np.float32
        )

    def action(self, action):
        return action


def _to_tracking_uri(value: str) -> str:
    """로컬 경로를 `file://` URI로 정규화한다.

    Windows에서 `C:\\...` 같은 경로를 그대로 MLflow에 넘기면 드라이브 문자 `C:`가
    URI scheme으로 오인돼(`UnsupportedModelRegistryStoreURIException`) 깨진다.
    이미 scheme이 있는 값(http://·databricks 등)은 그대로 둔다.
    """
    if "://" in value:
        return value
    return Path(value).resolve().as_uri()


def load_fold_env(cfg: dict, fold_id: int, split: str = "train") -> gym.Env:
    """지정 fold/split의 Feature Store 데이터를 읽어 PortfolioEnv를 만든다.

    SB3와 바로 호환되도록 `_BoundedActionWrapper`로 감싸 반환한다(action_bound는
    config `model.action_bound`, 기본 10).
    """
    out_dir = cfg["data"]["feature_store_dir"]
    state_df = fs.load_features(out_dir, fold_id, split)
    targets_df = fs.load_targets(out_dir, fold_id, split)
    env = PortfolioEnv(state_df, targets_df, cfg)
    bound = float(cfg.get("model", {}).get("action_bound", 10.0))
    return _BoundedActionWrapper(env, bound=bound)


def evaluate(model, cfg: dict, fold_id: int, split: str = "valid") -> dict:
    """정책을 결정적으로 1 에피소드 굴려 요약 지표를 낸다.

    학습 중 조기 확인용이며, 정식 성과 검증은 찬휘 백테스트 엔진(docs/backtest_engine.md)이
    맡는다 — 여기서는 같은 reward 계산을 재사용해 방향성만 빠르게 본다.
    """
    env = load_fold_env(cfg, fold_id, split)
    obs, _ = env.reset()
    total_reward = 0.0
    total_cost = 0.0
    total_turnover = 0.0
    n_steps = 0
    terminated = truncated = False
    while not (terminated or truncated):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += float(reward)
        total_cost += float(info["cost"])
        total_turnover += float(info["turnover"])
        n_steps += 1
    return {
        f"{split}_total_log_return": total_reward,
        f"{split}_total_txn_cost": total_cost,
        f"{split}_avg_turnover": total_turnover / max(n_steps, 1),
        f"{split}_n_steps": float(n_steps),
    }


def train(
    config_path: str = "config/config.yaml",
    fold_id: int | None = None,
    total_timesteps: int | None = None,
) -> dict:
    """PPO를 학습하고 MLflow에 기록한다. {run_id, model_path, ...평가지표} 반환."""
    # mlflow-skinny 3.14+에서 순수 파일 트래킹 스토어("mlruns/")가 기본 비활성(유지보수
    # 모드)이라 명시적으로 허용한다. DB 백엔드(sqlite:///...)로 옮기기 전까지의 임시 조치.
    os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")

    import mlflow
    from stable_baselines3 import PPO

    cfg = load_config(config_path)
    model_cfg = cfg.get("model", {})

    algorithm = model_cfg.get("algorithm", "PPO")
    if algorithm != "PPO":
        # 1차 알고리즘은 PPO로 확정(CLAUDE.md §3). SAC 비교는 별도 스크립트로 추가할 것.
        raise NotImplementedError(f"현재는 PPO만 지원합니다: {algorithm!r}")

    fold_id = fold_id if fold_id is not None else int(model_cfg.get("fold_id", 0))
    total_timesteps = (
        total_timesteps if total_timesteps is not None else int(model_cfg.get("total_timesteps", 200_000))
    )
    seed = int(model_cfg.get("seed", 42))
    policy = model_cfg.get("policy", "MlpPolicy")

    env = load_fold_env(cfg, fold_id, "train")

    mlflow.set_tracking_uri(_to_tracking_uri(model_cfg.get("mlflow_tracking_uri", "mlruns")))
    mlflow.set_experiment(model_cfg.get("mlflow_experiment", "robustam-ppo"))

    model_dir = Path(model_cfg.get("model_dir", "mlruns/models"))
    model_dir.mkdir(parents=True, exist_ok=True)

    with mlflow.start_run() as run:
        mlflow.log_params(
            {
                "algorithm": algorithm,
                "policy": policy,
                "fold_id": fold_id,
                "total_timesteps": total_timesteps,
                "window": cfg["window"],
                "transaction_cost": cfg["transaction_cost"],
                "seed": seed,
            }
        )

        ppo_kwargs = {"seed": seed, "verbose": 1}
        if "n_steps" in model_cfg:
            # SB3 기본값(2048) 오버라이드 — 짧은 fold·테스트에서 rollout을 줄이는 용도.
            ppo_kwargs["n_steps"] = int(model_cfg["n_steps"])
        model = PPO(policy, env, **ppo_kwargs)
        model.learn(total_timesteps=total_timesteps)

        model_path = model_dir / f"ppo_fold{fold_id}_{run.info.run_id}.zip"
        model.save(str(model_path))
        mlflow.log_artifact(str(model_path))

        metrics = evaluate(model, cfg, fold_id, split="valid")
        mlflow.log_metrics(metrics)

        result = {"run_id": run.info.run_id, "model_path": str(model_path), "fold_id": fold_id}
        result.update(metrics)
        return result


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RobuSTAM PPO 학습")
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--fold-id", type=int, default=None)
    parser.add_argument("--total-timesteps", type=int, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    result = train(args.config, fold_id=args.fold_id, total_timesteps=args.total_timesteps)
    print(result)
