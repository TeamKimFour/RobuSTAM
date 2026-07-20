#!/bin/sh
# RobuSTAM API 컨테이너 진입점.
# PRECOMPUTE_S3_SYNC=true일 때만 S3 동기화 백그라운드 루프를 띄운다(기본 off — 로컬
# docker-compose.yml에서는 불필요한 S3 접근을 만들지 않도록 명시적 옵트인으로 뒀다).
set -e

if [ "${PRECOMPUTE_S3_SYNC:-false}" = "true" ]; then
    python -m src.inference.s3_fetch --loop &
fi

exec uvicorn src.api.main:app --host 0.0.0.0 --port 8000
