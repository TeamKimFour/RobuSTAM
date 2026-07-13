"""RobuSTAM FastAPI 추론 서버.

TODO(도현): 실제 추론 엔드포인트(시장 상태 → 익일 최적 비중, CLAUDE.md §1 정량목표
평균 지연 200ms 이하)로 교체 예정. 지금은 Docker Compose + nginx self-signed HTTPS
구성을 검증하기 위한 헬스체크 placeholder만 제공한다.
"""

from fastapi import FastAPI

app = FastAPI(title="RobuSTAM API (placeholder)")


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/")
def root() -> dict:
    return {"service": "RobuSTAM API", "status": "placeholder"}
