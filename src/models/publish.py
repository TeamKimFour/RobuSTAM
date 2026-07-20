"""배포 policy 반출 — 학습 머신과 서빙 환경을 S3로 잇는다 (이슈 #33).

학습 산출물 중 **`policy.zip`만** 반출이 필요하다. Feature Store·정규화 통계는 서빙 쪽이
스스로 `build`해서 만들 수 있지만(PR #36의 `scaler_run_id` 자동 해석), **모델은 학습 없이는
만들 수 없기** 때문이다. 이 비대칭이 이 모듈의 존재 이유다.

    학습 머신                         S3                       서빙(CI·EC2)
    mlruns/models/<name>.zip  ──upload──>  <prefix>/models/<name>.zip  ──download──>  mlruns/models/<name>.zip
                                                                              │
                                                              precompute가 config.inference.model_path로 로드

**키 규칙 — 파일명을 그대로 보존한다.** `train.py`가 만드는 이름
`ppo_fold<id>_<build_run_id>_<mlflow_run_id>.zip`이 이미 fold·build run·학습 run을 인코딩해
자기설명적이고 충돌이 없다. 별도 키 체계를 만들면 규칙이 하나 더 늘 뿐이다.

**`config.inference.model_path`는 그대로 로컬 경로를 쓴다.** S3 키는 그 basename에서
결정적으로 유도되므로(`model_s3_key`), config 스키마를 바꾸지 않고도 양쪽이 같은 파일을
가리킨다. 서빙 환경은 `download` 후 precompute를 돌리면 된다.

실행:
    python -m src.models.publish upload            # 학습 머신에서
    python -m src.models.publish download          # 서빙 환경에서
"""

from __future__ import annotations

import os
from pathlib import Path

from src.config_loader import get_inference, load_config

MODELS_SUBDIR = "models"


def model_s3_key(model_path: str | Path, prefix: str = "") -> str:
    """policy 파일의 S3 키를 basename에서 유도한다.

    파일명이 이미 유일하고 자기설명적이라(fold·build run·mlflow run 인코딩) 그대로 보존한다.
    """
    name = Path(model_path).name
    base = (prefix or "").strip("/")
    return f"{base}/{MODELS_SUBDIR}/{name}" if base else f"{MODELS_SUBDIR}/{name}"


def upload_model(
    model_path: str | Path,
    bucket: str,
    prefix: str = "",
    *,
    client=None,
) -> str:
    """학습된 policy.zip을 S3에 올리고 업로드한 키를 반환한다."""
    import boto3

    path = Path(model_path)
    if not path.is_file():
        raise FileNotFoundError(f"업로드할 policy 파일이 없습니다: {path}")

    key = model_s3_key(path, prefix)
    s3 = client or boto3.client("s3")
    s3.upload_file(str(path), bucket, key)
    return key


def download_model(
    model_path: str | Path,
    bucket: str,
    prefix: str = "",
    *,
    client=None,
    dest: str | Path | None = None,
) -> Path:
    """S3에서 policy.zip을 내려받아 로컬 `model_path` 위치에 놓는다.

    `model_path`는 `config.inference.model_path`를 그대로 넘기면 된다 — 키는 basename에서
    유도되고, 파일도 그 경로에 놓여 precompute가 바로 로드할 수 있다.
    """
    import boto3

    key = model_s3_key(model_path, prefix)
    target = Path(dest) if dest is not None else Path(model_path)
    target.parent.mkdir(parents=True, exist_ok=True)

    s3 = client or boto3.client("s3")
    s3.download_file(bucket, key, str(target))
    return target


def _resolve(cfg: dict, model_path: str | None) -> tuple[str, str, str]:
    """(model_path, bucket, prefix)를 config·환경변수에서 해석한다.

    버킷·프리픽스는 `s3_sync`와 동일하게 환경변수(`S3_BUCKET`/`S3_PREFIX`)로 override
    가능하다(CI에서 GitHub Variables 주입).
    """
    d = cfg["data"]
    path = model_path or get_inference(cfg).get("model_path")
    if not path:
        raise SystemExit(
            "model_path가 없습니다 — --model-path 또는 config.inference.model_path를 채우세요."
        )
    bucket = os.environ.get("S3_BUCKET") or d.get("s3_bucket")
    if not bucket:
        raise SystemExit("S3 버킷 미설정(config data.s3_bucket·환경변수 S3_BUCKET 모두 없음).")
    prefix = (os.environ.get("S3_PREFIX") or d.get("s3_prefix") or "").strip("/")
    return path, bucket, prefix


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="배포 policy S3 반출 (이슈 #33)")
    parser.add_argument("action", choices=["upload", "download"])
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--model-path", default=None,
                        help="기본값은 config.inference.model_path")
    args = parser.parse_args()

    cfg = load_config(args.config)
    path, bucket, prefix = _resolve(cfg, args.model_path)

    if args.action == "upload":
        key = upload_model(path, bucket, prefix)
        print(f"업로드 완료: {path} → s3://{bucket}/{key}")
    else:
        target = download_model(path, bucket, prefix)
        print(f"다운로드 완료: s3://{bucket}/{model_s3_key(path, prefix)} → {target}")


if __name__ == "__main__":
    main()
