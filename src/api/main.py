"""RobuSTAM FastAPI 추론 서버.

목표(CLAUDE.md §1 정량목표): 시장 상태 → 익일 최적 비중 반환, 평균 지연 200ms 이하.
1차 구현은 온라인 추론이 아니라 **precompute 패턴**이다 — 모델은 매일 배치로
`config.yaml`의 `data.precompute_path`(기본 data/precompute/latest.json)에 그날의
추천 비중을 미리 계산해 쓰고, 이 서버는 요청마다 그 파일을 읽어 반환하기만 한다.
파일 I/O 하나뿐이라 지연 목표를 구조적으로 만족한다.

precompute 자체(민지 담당, docs/data_pipeline.md §8)는 아직 미구현이라, 파일이 없으면
503으로 명시적으로 알린다 — 이는 버그가 아니라 파이프라인이 아직 안 돌았다는 정상 상태다.
"""

from __future__ import annotations

import json
from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException

from src.api.schemas import LatestInference
from src.config_loader import DEFAULT_CONFIG_PATH, get_assets, get_precompute_path, load_config

app = FastAPI(title="RobuSTAM API")


def get_config() -> dict:
    """요청마다 config.yaml을 읽는다. 테스트에서는 dependency_overrides로 대체한다."""
    return load_config(DEFAULT_CONFIG_PATH)


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
