#!/bin/sh
# RobuSTAM MLflow 서버 컨테이너 진입점.
# Render Postgres는 연결 문자열을 postgres:// 스킴으로 주는데, SQLAlchemy 1.4+
# (MLflow 3.x가 내부적으로 사용)는 이 스킴을 인식하지 못해 postgresql://로 바꿔줘야 한다.
# (PR #57 리뷰, 도현 지적사항 반영)
#
# 예시: postgres://user:pw@host/db  →  postgresql://user:pw@host/db
set -e

FIXED_DATABASE_URL=$(echo "$DATABASE_URL" | sed 's#^postgres://#postgresql://#')

# --allowed-hosts "*"·--cors-allowed-origins "*"는 Render 지원팀이 진단용으로 제안한
# 임시값이다(전체 허용, 배포 전 필수 아님). 문제가 실제로 이걸로 해결되는지 확인되면
# 다음 커밋에서 "*.onrender.com" 등으로 좁힐 예정 — 이 값 그대로 프로덕션에 두지 말 것.
exec mlflow server \
    --host 0.0.0.0 \
    --port "${PORT:-5000}" \
    --backend-store-uri "$FIXED_DATABASE_URL" \
    --artifacts-destination "s3://${S3_BUCKET}/mlflow-artifacts" \
    --allowed-hosts "*" \
    --cors-allowed-origins "*"
