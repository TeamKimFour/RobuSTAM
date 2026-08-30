"""RobuSTAM FastAPI 추론 서버.

목표(CLAUDE.md §1 정량목표): 시장 상태 → 익일 최적 비중 반환, 평균 지연 200ms 이하.
1차 구현은 온라인 추론이 아니라 **precompute 패턴**이다 — 모델은 매일 배치로
`config.yaml`의 `data.precompute_path`(기본 data/precompute/latest.json)에 그날의
추천 비중을 미리 계산해 쓰고, 이 서버는 요청마다 그 파일을 읽어 반환하기만 한다.
파일 I/O 하나뿐이라 지연 목표를 구조적으로 만족한다.

precompute 자체(민지 담당, docs/data_pipeline.md §8)는 아직 미구현이라, 파일이 없으면
503으로 명시적으로 알린다 — 이는 버그가 아니라 파이프라인이 아직 안 돌았다는 정상 상태다.

/models/runs·/models/deployed는 frontend/app/models/page.tsx(형우)의 MLFLOW_RUNS mock을
대체할 프록시다. `MLFLOW_TRACKING_URI`(로컬은 Docker MLflow 서버 HTTP, 배포는 Supabase
Postgres 직접 연결)로 MLflow experiment("robustam-ppo", config.model.mlflow_experiment)를
조회한다. env var 미설정·MLflow 연결 실패는 500이 아니라 503으로 "아직 준비 안 됨"을
명확히 구분한다(위 /inference/latest 503 패턴과 동일).
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

import mlflow
from fastapi import Depends, FastAPI, HTTPException
from mlflow.tracking import MlflowClient

from src.api.schemas import DeployedModel, LatestInference, ModelRun
from src.config_loader import DEFAULT_CONFIG_PATH, get_assets, get_precompute_path, load_config

app = FastAPI(title="RobuSTAM API")

# 앱 시작 시 1회만 읽는다 — env var가 없으면 /models/* 두 엔드포인트가 503으로 알린다.
_MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI")
if _MLFLOW_TRACKING_URI:
    mlflow.set_tracking_uri(_MLFLOW_TRACKING_URI)

# train.py(273-275행) 현행 규약은 ppo_{combo}_fold{id}_{fs_tag}_{run_id}.zip이지만,
# 콤보 기능(PR #69) 이전에 배포된 config.yaml의 기존 model_path는 콤보 세그먼트가 없는
# ppo_fold{id}_{fs_tag}_{run_id}.zip 형태다(직접 확인함). run_id(MLflow 32자리 hex)는
# 두 규약 모두 `.zip` 바로 앞 세그먼트로 고정이라, 앞쪽 세그먼트 개수에 기대지 않고
# 끝에서부터 파싱해 두 규약을 동시에 지원한다.
_RUN_ID_FROM_MODEL_PATH = re.compile(r"_([0-9a-f]{32})\.zip$")

# DeployedModel.hyperparams로 노출하는 6개 키 — train.py가 MLflow에 항상 로깅하는 값만
# 고른다(learning_rate 등은 config에 있을 때만 조건부 로깅이라 제외). action_bound는
# MLflow run param이 아니라(래퍼 설정값, 로깅 안 됨) config.model에서 별도로 채운다.
_HYPERPARAM_MLFLOW_KEYS = ("algorithm", "policy", "total_timesteps", "seed", "train_cost_multiplier")


def get_config() -> dict:
    """요청마다 config.yaml을 읽는다. 테스트에서는 dependency_overrides로 대체한다."""
    return load_config(DEFAULT_CONFIG_PATH)


def _require_mlflow_configured() -> None:
    if not _MLFLOW_TRACKING_URI:
        raise HTTPException(
            status_code=503,
            detail="MLFLOW_TRACKING_URI가 설정되지 않았습니다 — MLflow 조회가 아직 준비되지 않았습니다.",
        )


def _mlflow_experiment_name(cfg: dict) -> str:
    """config.model.mlflow_experiment(기본 robustam-ppo, train.py와 동일 기본값)."""
    return cfg.get("model", {}).get("mlflow_experiment", "robustam-ppo")


def _deployed_run_id(cfg: dict) -> str | None:
    """config.inference.model_path 파일명에서 배포된 policy의 MLflow run_id를 뽑는다."""
    model_path = cfg.get("inference", {}).get("model_path", "")
    m = _RUN_ID_FROM_MODEL_PATH.search(Path(model_path).name)
    return m.group(1) if m else None


def _to_model_run(run, deployed_run_id: str | None) -> ModelRun:
    params = run.data.params
    return ModelRun(
        run_id=run.info.run_id,
        fold_id=int(params["fold_id"]),
        seed=int(params["seed"]),
        total_timesteps=int(params["total_timesteps"]),
        valid_sharpe=run.data.metrics.get("valid_sharpe"),
        status=run.info.status,
        deployed=(run.info.run_id == deployed_run_id),
    )


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/")
def root() -> dict:
    return {"service": "RobuSTAM API", "status": "ok"}


@app.get("/inference/latest", response_model=LatestInference)
def get_latest_inference(cfg: dict = Depends(get_config)) -> LatestInference:
    """precompute가 써둔 익일 추천 비중을 읽어 반환한다."""
    path = Path(get_precompute_path(cfg))
    if not path.is_file():
        raise HTTPException(
            status_code=503,
            detail=f"precompute 결과가 아직 없습니다: {path} (매일 배치 실행 전이거나 미구현)",
        )

    with path.open("r", encoding="utf-8") as f:
        raw = json.load(f)

    result = LatestInference.model_validate(raw)

    expected_assets = set(get_assets(cfg))
    actual_assets = set(result.weights)
    if actual_assets != expected_assets:
        raise HTTPException(
            status_code=500,
            detail=(
                f"{path}의 자산 구성이 config.yaml과 다릅니다. "
                f"기대: {expected_assets}, 실제: {actual_assets}"
            ),
        )

    total = sum(result.weights.values())
    if abs(total - 1.0) > 1e-3:
        raise HTTPException(
            status_code=500,
            detail=f"{path}의 비중 합이 1이 아닙니다 (현재 합: {total:.6f})",
        )

    return result


@app.get("/models/runs", response_model=list[ModelRun])
def get_model_runs(cfg: dict = Depends(get_config)) -> list[ModelRun]:
    """robustam-ppo experiment의 전체 학습 run을 MLflow에서 조회한다."""
    _require_mlflow_configured()
    experiment_name = _mlflow_experiment_name(cfg)

    try:
        client = MlflowClient()
        experiment = client.get_experiment_by_name(experiment_name)
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"MLflow 연결 실패: {e}") from e

    if experiment is None:
        raise HTTPException(
            status_code=503,
            detail=(
                f"MLflow experiment {experiment_name!r}가 아직 없습니다 — 학습이 아직 안 "
                "돌았거나 MLFLOW_TRACKING_URI가 다른 서버를 가리킵니다."
            ),
        )

    try:
        runs = client.search_runs(experiment_ids=[experiment.experiment_id])
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"MLflow 연결 실패: {e}") from e

    deployed_run_id = _deployed_run_id(cfg)
    return [_to_model_run(r, deployed_run_id) for r in runs]


@app.get("/models/deployed", response_model=DeployedModel)
def get_deployed_model(cfg: dict = Depends(get_config)) -> DeployedModel:
    """config.inference 섹션과 그 run의 MLflow 데이터를 합쳐 배포 모델 상세를 반환한다."""
    _require_mlflow_configured()
    inference = cfg.get("inference", {})
    run_id = _deployed_run_id(cfg)
    if run_id is None:
        raise HTTPException(
            status_code=503,
            detail=(
                "config.inference.model_path에서 run_id를 파싱할 수 없습니다: "
                f"{inference.get('model_path')!r}"
            ),
        )

    try:
        run = MlflowClient().get_run(run_id)
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"MLflow 연결 실패: {e}") from e

    params = run.data.params
    hyperparams = {k: params[k] for k in _HYPERPARAM_MLFLOW_KEYS if k in params}
    hyperparams["action_bound"] = str(cfg.get("model", {}).get("action_bound", 10.0))

    return DeployedModel(
        run_id=run_id,
        model_version=inference.get("model_version", ""),
        model_path=inference.get("model_path", ""),
        scaler_fold_id=int(inference.get("scaler_fold_id", 0)),
        valid_sharpe=run.data.metrics.get("valid_sharpe"),
        hyperparams=hyperparams,
    )
