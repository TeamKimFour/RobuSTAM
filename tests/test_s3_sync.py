"""s3_sync 테스트 — moto(가짜 S3)로 실제 AWS 없이 업로드 검증.

moto·boto3는 requirements-dev에만 있으므로 importorskip으로 가드(미설치 CI에선 skip).
"""

import pytest

pytest.importorskip("boto3")
pytest.importorskip("moto")

import boto3
from moto import mock_aws

from src.data.s3_sync import sync_dir_to_s3

BUCKET = "robustam-test"


def _make_local(tmp_path):
    """feature_store 유사 구조(파티션 Parquet + meta.sqlite) 생성."""
    root = tmp_path / "feature_store"
    (root / "fold=1" / "split=train").mkdir(parents=True)
    (root / "fold=1" / "split=train" / "part.parquet").write_bytes(b"parquet")
    (root / "fold=1" / "split=train" / "targets.parquet").write_bytes(b"targets")
    (root / "meta.sqlite").write_bytes(b"sqlite")  # s3 직접쓰기 불가 → 파일 업로드로 처리
    return root


@mock_aws
def test_sync_uploads_all_files_with_prefix(tmp_path):
    root = _make_local(tmp_path)
    s3 = boto3.client("s3", region_name="us-east-1")
    s3.create_bucket(Bucket=BUCKET)

    keys = sync_dir_to_s3(str(root), BUCKET, "robustam/feature_store")

    got = {o["Key"] for o in s3.list_objects_v2(Bucket=BUCKET)["Contents"]}
    assert "robustam/feature_store/fold=1/split=train/part.parquet" in got
    assert "robustam/feature_store/fold=1/split=train/targets.parquet" in got
    assert "robustam/feature_store/meta.sqlite" in got  # SQLite도 파일로 업로드됨
    assert set(keys) == got
    assert len(keys) == 3


@mock_aws
def test_sync_empty_prefix(tmp_path):
    root = _make_local(tmp_path)
    s3 = boto3.client("s3", region_name="us-east-1")
    s3.create_bucket(Bucket=BUCKET)

    keys = sync_dir_to_s3(str(root), BUCKET, "")
    assert "meta.sqlite" in keys  # prefix 없으면 상대경로 그대로


def test_missing_dir_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        sync_dir_to_s3(str(tmp_path / "nope"), BUCKET, "x")
