#!/bin/sh
# RobuSTAM 학습 컨테이너 진입점 (docker/Dockerfile.train).
# data/feature_store가 비어 있으면(볼륨 미마운트) collect→build로 먼저 채운 뒤 학습한다.
# 인자는 그대로 src.models.train에 전달된다: ./train_entrypoint.sh --fold-id 1

set -e

if [ ! -d "data/feature_store" ] || [ -z "$(ls -A data/feature_store 2>/dev/null)" ]; then
  echo "[train_entrypoint] data/feature_store 비어 있음 — collect→build 먼저 실행"
  python -m src.data.collect
  python -m src.data.build
fi

exec python -m src.models.train "$@"
