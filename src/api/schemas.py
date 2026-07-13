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
