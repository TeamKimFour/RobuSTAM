"""추론 API 응답 스키마.

latest.json 계약(docs/data_pipeline.md §8)의 코드측 정의. 민지 precompute가 이 스키마로
파일을 쓰고, 이 모듈이 그 파일을 읽어 검증한다.
"""

from __future__ import annotations

from pydantic import BaseModel


class LatestInference(BaseModel):
    """precompute가 생성하는 익일 추천 비중 1건.

    weights의 키 집합은 config.yaml의 자산 목록과 정확히 일치해야 하며(순서는
    JSON object라 보장되지 않으므로 집합으로만 검증), 값의 합은 1에 근접해야 한다.
    두 불변식은 파일 파싱 후 src/api/main.py가 검증한다(이 클래스는 형태만 검증).
    """

    date: str
    generated_at: str
    model_version: str
    weights: dict[str, float]


class ModelRun(BaseModel):
    """MLflow experiment(robustam-ppo)의 학습 run 1건 — GET /models/runs 응답 항목.

    validMdd·testSharpe(frontend/app/models/mock.ts)는 train.py가 MLflow에 로깅하지
    않는 값이라 이 스키마엔 없다 — 프론트가 추측·계산해서 채우지 않도록 응답에서
    아예 뺐다(회의 안건으로 별도 처리 중).
    """

    run_id: str
    fold_id: int
    seed: int
    total_timesteps: int
    valid_sharpe: float | None
    status: str  # MLflow RunStatus (FINISHED/FAILED/RUNNING)
    deployed: bool


class DeployedModel(BaseModel):
    """현재 config.inference가 가리키는 배포 policy 상세 — GET /models/deployed 응답.

    hyperparams는 train.py가 MLflow에 실제로 로깅하는 값 중 팀이 노출하기로 정한
    6개 키만 담는다. learning_rate·batch_size·gamma·gae_lambda·clip_range·n_steps는
    MLflow에 없거나(action_bound는 config에서 보완) 노출 대상이 아니다.
    """

    run_id: str
    model_version: str
    model_path: str
    scaler_fold_id: int
    valid_sharpe: float | None
    hyperparams: dict[str, str]
