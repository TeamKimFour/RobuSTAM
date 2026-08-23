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
    python -m src.models.train                              # config 기본 fold·timesteps·seed
    python -m src.models.train --fold-id 1 --total-timesteps 500000
    python -m src.models.train --fold-id 1 --seed 7         # 같은 fold의 배포 후보 추가
    python -m src.models.train --combo M1 --fold-id 1       # 피처 콤보 M1으로 학습
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from src.config_loader import (
    force_utf8_stdout,
    get_action_delta,
    get_active_combo,
    get_state_dim,
    load_config_for_combo,
)
from src.data import feature_store as fs
from src.env.portfolio_env import PortfolioEnv
from src.models.loader import get_algorithm, is_discrete


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


def load_fold_env(
    cfg: dict,
    fold_id: int,
    split: str = "train",
    cost_multiplier: float = 1.0,
    vol_penalty_coef: float = 0.0,
) -> gym.Env:
    """지정 fold/split의 Feature Store 데이터를 읽어 PortfolioEnv를 만든다.

    SB3와 바로 호환되도록 `_BoundedActionWrapper`로 감싸 반환한다(action_bound는
    config `model.action_bound`, 기본 10).

    `cost_multiplier`(λ, 개선안 A)와 `vol_penalty_coef`(κ, 개선안 D)는 모두 학습 보상에만
    거는 셰이핑 장치라 **train split에만** 넘긴다. 기본값(1.0·0.0)이라 평가·백테스트
    호출부는 실비용 그대로·페널티 없이 벤치마크와 같은 자로 측정된다.

    `model.algorithm`이 DQN이면(개선안 B, 행동 연속성) `_BoundedActionWrapper` 대신
    `DiscretePortfolioEnv`로 감싼다 — DQN은 이산 행동만 다루고, 하루 이전폭이 Δ로 묶여
    turnover가 구조적으로 2Δ 이하로 제한된다. 어느 래퍼를 쓸지는 cfg만 보고 정해지므로
    `evaluate()`처럼 알고리즘을 모르는 호출부도 학습과 같은 env 계약을 얻는다.
    """
    out_dir = cfg["data"]["feature_store_dir"]
    state_df = fs.load_features(out_dir, fold_id, split)
    targets_df = fs.load_targets(out_dir, fold_id, split)
    env = PortfolioEnv(
        state_df,
        targets_df,
        cfg,
        cost_multiplier=cost_multiplier,
        vol_penalty_coef=vol_penalty_coef,
    )
    if is_discrete(cfg):
        from src.env.discrete_env import DiscretePortfolioEnv

        return DiscretePortfolioEnv(env, delta=get_action_delta(cfg))
    bound = float(cfg.get("model", {}).get("action_bound", 10.0))
    return _BoundedActionWrapper(env, bound=bound)


TRADING_DAYS_PER_YEAR = 252  # 일봉 → 연율화 상수 (docs/state_spec.md·CLAUDE.md 데일리 리밸런싱)


def _annualized_sharpe(rewards: list[float]) -> float:
    """스텝별 net 로그보상 시계열의 연율화 Sharpe proxy = mean/std × √252.

    무위험수익률 0 가정(보상 자체가 비용 차감 로그수익). 스텝이 2개 미만이거나
    표준편차가 0이면 위험조정을 정의할 수 없어 0.0을 반환한다. 배포 policy 선택
    (src/models/select.py)의 기본 지표로, 단순 누적수익 대신 위험조정수익을 본다는
    프로젝트 철학(CLAUDE.md §1 샤프 극대화)을 따른다. 정식 성과검증은 여전히 찬휘
    백테스트 엔진 몫이며, 여기 값은 후보 policy 간 방향성 선별용 근사다.
    """
    if len(rewards) < 2:
        return 0.0
    arr = np.asarray(rewards, dtype=np.float64)
    std = float(arr.std(ddof=1))
    if std == 0.0:
        return 0.0
    return float(arr.mean() / std * np.sqrt(TRADING_DAYS_PER_YEAR))


def evaluate(model, cfg: dict, fold_id: int, split: str = "valid") -> dict:
    """정책을 결정적으로 1 에피소드 굴려 요약 지표를 낸다.

    학습 중 조기 확인용이며, 정식 성과 검증은 찬휘 백테스트 엔진(docs/backtest_engine.md)이
    맡는다 — 여기서는 같은 reward 계산을 재사용해 방향성만 빠르게 본다. `{split}_sharpe`는
    배포 policy 선택(select.py)의 기본 지표다.
    """
    env = load_fold_env(cfg, fold_id, split)
    obs, _ = env.reset()
    rewards: list[float] = []  # 스텝별 net 로그보상 R_t (Sharpe 산출용 시계열)
    total_cost = 0.0
    total_turnover = 0.0
    terminated = truncated = False
    while not (terminated or truncated):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        rewards.append(float(reward))
        total_cost += float(info["cost"])
        total_turnover += float(info["turnover"])
    n_steps = len(rewards)
    return {
        f"{split}_total_log_return": float(sum(rewards)),
        f"{split}_sharpe": _annualized_sharpe(rewards),
        f"{split}_total_txn_cost": total_cost,
        f"{split}_avg_turnover": total_turnover / max(n_steps, 1),
        f"{split}_n_steps": float(n_steps),
    }


def train(
    config_path: str = "config/config.yaml",
    fold_id: int | None = None,
    total_timesteps: int | None = None,
    seed: int | None = None,
    combo: str | None = None,
    cost_multiplier: float | None = None,
    vol_penalty_coef: float | None = None,
) -> dict:
    """PPO를 학습하고 MLflow에 기록한다. {run_id, model_path, ...평가지표} 반환.

    combo: 피처 콤보 이름(M0~M3 등). 생략 시 config의 `active_combo`(기본 `full`, 187차원)라
    기존 동작과 동일하다. 지정하면 그 콤보의 Feature Store(`data/feature_store/<combo>/`)를
    읽고 State 차원도 그 콤보 것으로 바뀐다(docs/data_pipeline.md §3-2).

    cost_multiplier(λ)·vol_penalty_coef(κ): 생략 시 config `model` 값. 여러 콤보를 같은
    조건으로 돌리는 실험 러너(experiment.py)가 config 파일을 건드리지 않고 동일 값을
    주입할 수 있도록 인자로도 연다 — seed·total_timesteps와 같은 오버라이드 규약이다.
    """
    # mlflow-skinny 3.14+에서 순수 파일 트래킹 스토어("mlruns/")가 기본 비활성(유지보수
    # 모드)이라 명시적으로 허용한다. DB 백엔드(sqlite:///...)로 옮기기 전까지의 임시 조치.
    os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")

    import mlflow

    # 콤보를 반영한 cfg — features(→state_dim)와 data 경로가 모두 그 콤보 것으로 바뀐다.
    # 이후 load_fold_env·evaluate·PortfolioEnv는 콤보를 몰라도 올바른 Feature Store를 읽는다.
    cfg = load_config_for_combo(config_path, combo)
    combo_name = get_active_combo(cfg, combo)
    model_cfg = cfg.get("model", {})

    # 1차 알고리즘은 PPO(CLAUDE.md §3). DQN은 이슈 #34 개선안 B(행동 연속성) 실험용으로,
    # 이산 어댑터가 하루 이전폭을 Δ로 묶어 turnover를 구조적으로 제한한다. SAC는 미지원.
    algorithm = get_algorithm(cfg)
    if algorithm not in ("PPO", "DQN"):
        raise NotImplementedError(f"현재는 PPO·DQN만 지원합니다: {algorithm!r}")

    # fold_id는 1부터 시작한다 (src/data/splits.py::make_folds가 1-indexed로 생성).
    fold_id = fold_id if fold_id is not None else int(model_cfg.get("fold_id", 1))
    total_timesteps = (
        total_timesteps if total_timesteps is not None else int(model_cfg.get("total_timesteps", 200_000))
    )
    # seed는 fold와 함께 배포 후보를 늘리는 축이다 — 같은 fold를 seed만 바꿔 여러 번 학습해
    # select.py가 valid 지표로 고를 후보 풀을 만든다(docs/model_training.md §6).
    seed = seed if seed is not None else int(model_cfg.get("seed", 42))
    policy = model_cfg.get("policy", "MlpPolicy")

    # 개선안 A(보상 셰이핑, 이슈 #34): 학습 보상의 거래비용만 λ배로 키워 과매매를 벌한다.
    # 평가(evaluate)·백테스트는 λ를 받지 않으므로 실비용 0.1% 축을 그대로 유지한다.
    cost_multiplier = float(
        cost_multiplier if cost_multiplier is not None
        else model_cfg.get("train_cost_multiplier", 1.0)
    )
    # 개선안 D(위험조정 보상, 이슈 #34): 최근 변동성 σ_recent에 κ를 걸어 흔들리는 경로를 벌한다.
    # λ와 마찬가지로 학습 전용 — 평가·백테스트는 κ=0으로 성과 측정 축을 유지한다.
    vol_penalty_coef = float(
        vol_penalty_coef if vol_penalty_coef is not None
        else model_cfg.get("vol_penalty_coef", 0.0)
    )

    env = load_fold_env(
        cfg,
        fold_id,
        "train",
        cost_multiplier=cost_multiplier,
        vol_penalty_coef=vol_penalty_coef,
    )

    # 학습에 쓰는 Feature Store fold를 만든 build run_id를 provenance로 확보한다(이슈 #27).
    # 이 값이 있어야 추론(precompute)이 config.inference.scaler_run_id로 동일 정규화 통계를
    # 재현할 수 있다. 미상(빌드 전 등)이면 "nofs"로 남겨 나중에 추적 불가함을 드러낸다.
    # meta_db도 콤보별로 갈린다(load_config_for_combo가 이미 반영) — 그 콤보의 build만 본다.
    meta_db = cfg.get("data", {}).get("meta_db")
    fs_run_id = fs.latest_run_id_for_fold(meta_db, fold_id) if meta_db else None
    fs_tag = fs_run_id or "nofs"

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
                "feature_store_run_id": fs_tag,  # 정규화 재현 출처(config.inference.scaler_run_id에 복사)
                # 어느 피처 콤보로 학습됐는지 — 조합 비교 실험(experiment.py)의 그룹 키다.
                # 스크리닝 run(screen_combos.py)의 feature_set과 같은 이름 규약을 쓴다.
                "feature_set": combo_name or "full",
                "state_dim": get_state_dim(cfg),
                "total_timesteps": total_timesteps,
                "window": cfg["window"],
                "transaction_cost": cfg["transaction_cost"],
                "train_cost_multiplier": cost_multiplier,  # λ (이슈 #34 개선안 A, 학습 전용)
                "vol_penalty_coef": vol_penalty_coef,  # κ (이슈 #34 개선안 D, 학습 전용)
                "seed": seed,
            }
        )
        if is_discrete(cfg):
            # Δ (이슈 #34 개선안 B) — 이 값이 turnover 상한 2Δ를 정하므로 run마다 남긴다.
            mlflow.log_param("action_delta", get_action_delta(cfg))

        if algorithm == "DQN":
            from stable_baselines3 import DQN

            # DQN 하이퍼파라미터도 PPO와 같은 규약 — config에 있을 때만 넘겨 SB3 기본값 보존.
            algo_kwargs: dict = {"seed": seed, "verbose": 1}
            for key, cast in (
                ("learning_rate", float),
                ("buffer_size", int),
                ("learning_starts", int),
                ("batch_size", int),
                ("gamma", float),
                ("train_freq", int),
                ("target_update_interval", int),
                ("exploration_fraction", float),
                ("exploration_final_eps", float),
            ):
                value = model_cfg.get("dqn", {}).get(key, model_cfg.get(key))
                if value is not None:
                    algo_kwargs[key] = cast(value)
            mlflow.log_params({f"dqn_{k}": v for k, v in algo_kwargs.items() if k != "verbose"})
            model = DQN(policy, env, **algo_kwargs)
        else:
            from stable_baselines3 import PPO

            ppo_kwargs = {"seed": seed, "verbose": 1}
            if "n_steps" in model_cfg:
                # SB3 기본값(2048) 오버라이드 — 짧은 fold·테스트에서 rollout을 줄이는 용도.
                ppo_kwargs["n_steps"] = int(model_cfg["n_steps"])
            # PPO 안정화 하이퍼파라미터 — config에 있을 때만 넘겨 SB3 기본값을 보존한다.
            # 이슈 #34에서 `clip_fraction 0.33`·`approx_kl 0.038`로 업데이트가 과격한 것이
            # 관측돼(과매매 원인 후보) 튜닝 통로를 연다. §2(보상·행동공간)와 무관한 학습 설정이다.
            for key, cast in (("learning_rate", float), ("target_kl", float), ("ent_coef", float)):
                if model_cfg.get(key) is not None:
                    ppo_kwargs[key] = cast(model_cfg[key])
            mlflow.log_params({k: ppo_kwargs[k] for k in ("learning_rate", "target_kl", "ent_coef")
                               if k in ppo_kwargs})
            model = PPO(policy, env, **ppo_kwargs)
        model.learn(total_timesteps=total_timesteps)

        # 파일명에 알고리즘·콤보·build run_id를 박아, 무엇으로 어떻게 학습됐는지 파일만 봐도
        # 알 수 있게 한다(PPO·DQN policy를 한 디렉토리에 섞어둬도 구분되고, 로드할 클래스도
        # 파일명으로 드러난다 — 실제 로드 판단은 config의 model.algorithm이 단일 출처다).
        model_path = (
            model_dir
            / f"{algorithm.lower()}_{combo_name or 'full'}_fold{fold_id}_{fs_tag}_{run.info.run_id}.zip"
        )
        model.save(str(model_path))
        mlflow.log_artifact(str(model_path))

        metrics = evaluate(model, cfg, fold_id, split="valid")
        mlflow.log_metrics(metrics)

        result = {
            "run_id": run.info.run_id,
            "model_path": str(model_path),
            # 소비처(experiment.py·runner.py)가 어느 SB3 클래스로 읽어야 하는지 알 수 있게 한다.
            "algorithm": algorithm,
            "fold_id": fold_id,
            "seed": seed,
            "feature_set": combo_name or "full",
            "state_dim": get_state_dim(cfg),
            "feature_store_run_id": fs_run_id,  # 배포 시 config.inference.scaler_run_id에 넣을 값
        }
        result.update(metrics)
        return result


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RobuSTAM PPO 학습")
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--fold-id", type=int, default=None)
    parser.add_argument("--total-timesteps", type=int, default=None)
    parser.add_argument(
        "--seed", type=int, default=None, help="config model.seed 오버라이드(배포 후보 늘리기용)"
    )
    parser.add_argument(
        "--combo", default=None, help="피처 콤보(M0~M3). 생략 시 config의 active_combo"
    )
    return parser.parse_args()


if __name__ == "__main__":
    force_utf8_stdout()  # MLflow http 트래킹의 이모지 출력이 cp949에서 터지는 것 방지
    args = _parse_args()
    result = train(
        args.config,
        fold_id=args.fold_id,
        total_timesteps=args.total_timesteps,
        seed=args.seed,
        combo=args.combo,
    )
    print(result)
    # 배포용 안내: 이 policy를 서빙하려면 config.inference를 아래 값으로 채운다(이슈 #27).
    print(
        "\n[배포] config.inference에 복사:\n"
        f"  model_path: {result['model_path']!r}\n"
        f"  scaler_run_id: {result['feature_store_run_id']!r}\n"
        f"  scaler_fold_id: {result['fold_id']}"
    )
