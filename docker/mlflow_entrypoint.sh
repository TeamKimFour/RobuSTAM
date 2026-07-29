#!/bin/sh
# RobuSTAM MLflow 서버 컨테이너 진입점.
# Render Postgres는 연결 문자열을 postgres:// 스킴으로 주는데, SQLAlchemy 1.4+
# (MLflow 3.x가 내부적으로 사용)는 이 스킴을 인식하지 못해 postgresql://로 바꿔줘야 한다.
# (PR #57 리뷰, 도현 지적사항 반영)
#
# 예시: postgres://user:pw@host/db  →  postgresql://user:pw@host/db
set -e

FIXED_DATABASE_URL=$(echo "$DATABASE_URL" | sed 's#^postgres://#postgresql://#')

exec mlflow server \
    --host 0.0.0.0 \
    --port "${PORT:-5000}" \
    --backend-store-uri "$FIXED_DATABASE_URL" \
    --artifacts-destination "s3://${S3_BUCKET}/mlflow-artifacts"
