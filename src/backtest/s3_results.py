"""백테스트 실행 결과를 S3에 저장 — NAV·성과지표·config 스냅샷 보존.

engine.py의 calc_nav() 로직과는 무관한 독립 저장 단계다. 백테스트 루프가 모두
끝난 뒤 호출한다(BacktestEngine.save_results()가 이 모듈에 위임). src.data.s3_sync와
동일하게 자격증명은 boto3 기본 자격증명 체인(환경변수 AWS_*·~/.aws/credentials·
IAM Role)에서 읽고 코드에 하드코딩하지 않는다. 로컬 `.env` 파일이 있으면
python-dotenv로 먼저 os.environ에 로드한다(.env.example 참고).

저장 구조:
    s3://{bucket}/backtests/{실행날짜}_{실행ID}/
        nav.parquet    - NAV 시계열
        metrics.json   - Sharpe·MDD 등 성과 지표 (계산은 호출자 책임, 이 함수는 저장만 한다)
        config.yaml    - 이 실행에 쓰인 config 파일 원본 그대로 복사

버킷은 config의 `data.s3_bucket`을 쓰되, 환경변수 `S3_BUCKET`이 있으면 그것을
우선한다(src.data.s3_sync와 동일 관례). 버킷이 없으면(아직 미생성) 저장을 skip한다.
"""

from __future__ import annotations

import io
import json
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from src.config_loader import DEFAULT_CONFIG_PATH, load_config


def _run_prefix(run_id: str, run_date: str) -> str:
    return f"backtests/{run_date}_{run_id}"


def save_results_to_s3(
    nav: pd.DataFrame,
    metrics: dict,
    run_id: str | None = None,
    config_path: str = DEFAULT_CONFIG_PATH,
    *,
    bucket: str | None = None,
    run_date: str | None = None,
    client=None,
) -> str | None:
    """백테스트 실행 결과(NAV·지표·config 스냅샷)를 S3에 저장한다.

    Parameters
    ----------
    nav      : NAV 시계열 (run_equal_weight 등이 반환하는 DataFrame).
    metrics  : Sharpe·MDD 등 성과 지표 dict. 계산은 호출자 책임 — 이 함수는 저장만 한다.
    run_id   : 실행 식별자. 생략 시 짧은 UUID를 생성한다(MLflow run id를 넘겨도 된다).
    config_path : 이 실행에 쓰인 config.yaml 경로. 원본 그대로 S3에 복사된다.
    bucket   : 생략 시 config의 `data.s3_bucket`(환경변수 `S3_BUCKET`이 있으면 그것을 우선) 사용.
    client   : boto3 S3 client. 생략 시 실제 `boto3.client('s3')`를 생성한다(테스트는 mock 주입).

    Returns
    -------
    저장한 S3 prefix(``backtests/{날짜}_{run_id}``). 버킷 미설정으로 skip했으면 None.
    """
    cfg = load_config(config_path)
    bucket = bucket or os.environ.get("S3_BUCKET") or cfg.get("data", {}).get("s3_bucket")
    if not bucket:
        print("S3 버킷 미설정(config·환경변수·인자 모두 없음) → 저장 skip")
        return None

    run_id = run_id or uuid.uuid4().hex[:8]
    run_date = run_date or datetime.now(timezone.utc).strftime("%Y%m%d")
    prefix = _run_prefix(run_id, run_date)

    if client is None:
        from dotenv import load_dotenv

        load_dotenv()
        import boto3

        client = boto3.client("s3")

    nav_buf = io.BytesIO()
    nav.to_parquet(nav_buf)
    client.put_object(Bucket=bucket, Key=f"{prefix}/nav.parquet", Body=nav_buf.getvalue())

    client.put_object(
        Bucket=bucket,
        Key=f"{prefix}/metrics.json",
        Body=json.dumps(metrics, indent=2, ensure_ascii=False).encode("utf-8"),
    )

    client.put_object(
        Bucket=bucket,
        Key=f"{prefix}/config.yaml",
        Body=Path(config_path).read_bytes(),
    )

    print(f"백테스트 결과 저장 완료 → s3://{bucket}/{prefix}/")
    return prefix
