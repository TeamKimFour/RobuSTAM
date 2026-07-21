"""s3_fetch 테스트 — moto(가짜 S3)로 실제 AWS 없이 latest.json 수신 검증.

moto·boto3는 requirements-dev에만 있으므로 importorskip으로 가드(미설치 CI에선 skip).
"""

import json

import pytest

pytest.importorskip("boto3")
pytest.importorskip("moto")

import boto3
from moto import mock_aws

from src.inference.s3_fetch import fetch_once

BUCKET = "robustam-test"


def _cfg(tmp_path, bucket=BUCKET, prefix="robustam"):
    return {
        "data": {
            "s3_bucket": bucket,
            "s3_prefix": prefix,
            "precompute_path": str(tmp_path / "precompute" / "latest.json"),
        }
    }


@mock_aws
def test_fetch_downloads_existing_key(tmp_path):
    s3 = boto3.client("s3", region_name="us-east-1")
    s3.create_bucket(Bucket=BUCKET)
    payload = {"date": "2026-07-20", "weights": {"SPY": 1.0}}
    s3.put_object(
        Bucket=BUCKET,
        Key="robustam/precompute/latest.json",
        Body=json.dumps(payload).encode("utf-8"),
    )

    cfg = _cfg(tmp_path)
    assert fetch_once(cfg) is True

    dest = tmp_path / "precompute" / "latest.json"
    assert json.loads(dest.read_text(encoding="utf-8")) == payload


def test_fetch_skips_when_bucket_unset(tmp_path):
    cfg = _cfg(tmp_path, bucket="")
    assert fetch_once(cfg) is False
    assert not (tmp_path / "precompute" / "latest.json").exists()


@mock_aws
def test_fetch_returns_false_when_key_missing(tmp_path):
    s3 = boto3.client("s3", region_name="us-east-1")
    s3.create_bucket(Bucket=BUCKET)

    cfg = _cfg(tmp_path)
    assert fetch_once(cfg) is False
    assert not (tmp_path / "precompute" / "latest.json").exists()
