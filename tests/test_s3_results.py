"""s3_results 테스트 — moto(가짜 S3)로 실제 AWS 없이 백테스트 결과 저장을 검증한다.

moto·boto3는 requirements-dev에만 있으므로 importorskip으로 가드(미설치 CI에선 skip).
"""

import json

import pandas as pd
import pytest
import yaml

pytest.importorskip("boto3")
pytest.importorskip("moto")

import boto3
from moto import mock_aws

from src.backtest.s3_results import save_results_to_s3

BUCKET = "robustam-test"


def _write_config(tmp_path, s3_bucket: str = ""):
    """config_loader가 요구하는 필수 키를 채운 임시 config.yaml을 만든다."""
    cfg = {
        "assets": ["SPY", "EWY", "TLT", "GLD", "SHV"],
        "window": 30,
        "transaction_cost": 0.001,
        "data": {"s3_bucket": s3_bucket},
    }
    config_file = tmp_path / "config.yaml"
    config_file.write_text(yaml.dump(cfg, allow_unicode=True), encoding="utf-8")
    return config_file


def _nav_df():
    return pd.DataFrame({"nav": [1_000_000.0, 1_005_000.0, 1_010_000.0]})


@mock_aws
def test_saves_nav_metrics_config_under_expected_keys(tmp_path):
    config_file = _write_config(tmp_path)
    s3 = boto3.client("s3", region_name="us-east-1")
    s3.create_bucket(Bucket=BUCKET)

    prefix = save_results_to_s3(
        _nav_df(),
        {"sharpe": 1.2, "mdd": -0.15},
        run_id="abc123",
        config_path=str(config_file),
        bucket=BUCKET,
        run_date="20260713",
        client=s3,
    )

    assert prefix == "backtests/20260713_abc123"
    got = {o["Key"] for o in s3.list_objects_v2(Bucket=BUCKET)["Contents"]}
    assert got == {
        "backtests/20260713_abc123/nav.parquet",
        "backtests/20260713_abc123/metrics.json",
        "backtests/20260713_abc123/config.yaml",
    }


@mock_aws
def test_metrics_json_content_matches(tmp_path):
    config_file = _write_config(tmp_path)
    s3 = boto3.client("s3", region_name="us-east-1")
    s3.create_bucket(Bucket=BUCKET)

    save_results_to_s3(
        _nav_df(),
        {"sharpe": 1.2},
        run_id="r1",
        config_path=str(config_file),
        bucket=BUCKET,
        run_date="20260713",
        client=s3,
    )

    body = s3.get_object(Bucket=BUCKET, Key="backtests/20260713_r1/metrics.json")["Body"].read()
    assert json.loads(body) == {"sharpe": 1.2}


@mock_aws
def test_config_yaml_copied_verbatim(tmp_path):
    config_file = _write_config(tmp_path)
    original = config_file.read_bytes()  # 개행 변환 없는 원본 바이트와 비교(Windows CRLF 안전)
    s3 = boto3.client("s3", region_name="us-east-1")
    s3.create_bucket(Bucket=BUCKET)

    save_results_to_s3(
        _nav_df(),
        {},
        run_id="r1",
        config_path=str(config_file),
        bucket=BUCKET,
        run_date="20260713",
        client=s3,
    )

    body = s3.get_object(Bucket=BUCKET, Key="backtests/20260713_r1/config.yaml")["Body"].read()
    assert body == original


@mock_aws
def test_bucket_arg_overrides_config(tmp_path):
    """bucket 인자를 명시하면 config의 s3_bucket(비어 있음)보다 우선한다."""
    config_file = _write_config(tmp_path, s3_bucket="")
    s3 = boto3.client("s3", region_name="us-east-1")
    s3.create_bucket(Bucket=BUCKET)

    prefix = save_results_to_s3(
        _nav_df(), {}, run_id="r1", config_path=str(config_file), bucket=BUCKET, client=s3
    )
    assert prefix is not None


def test_skips_when_bucket_not_configured(tmp_path, monkeypatch):
    monkeypatch.delenv("S3_BUCKET", raising=False)
    config_file = _write_config(tmp_path, s3_bucket="")

    result = save_results_to_s3(
        _nav_df(), {}, run_id="r1", config_path=str(config_file), client=object()
    )
    assert result is None
