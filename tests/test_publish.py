"""배포 policy S3 반출 테스트 (이슈 #33) — moto(가짜 S3)로 실제 AWS 없이 검증.

핵심은 **업로드→다운로드 왕복이 같은 파일을 같은 자리로 되돌리는가**다. 학습 머신과 서빙
환경이 config 하나(`inference.model_path`)만 공유해도 동작해야 한다.
"""

import pytest

pytest.importorskip("boto3")
pytest.importorskip("moto")

import boto3
from moto import mock_aws

from src.models.publish import download_model, model_s3_key, upload_model

BUCKET = "robustam-test"
MODEL_NAME = "ppo_fold1_b461a86027e2-20260718102003_50db7bc4c4a541bb843c6d58d190bee5.zip"


def _make_model(tmp_path, content=b"fake-policy-zip"):
    path = tmp_path / "mlruns" / "models" / MODEL_NAME
    path.parent.mkdir(parents=True)
    path.write_bytes(content)
    return path


# ── 키 규칙 ──
def test_key_preserves_filename():
    """파일명이 fold·build run·mlflow run을 인코딩하므로 그대로 보존한다."""
    key = model_s3_key(f"mlruns/models/{MODEL_NAME}", "robustam")
    assert key == f"robustam/models/{MODEL_NAME}"


def test_key_without_prefix():
    assert model_s3_key(f"any/dir/{MODEL_NAME}") == f"models/{MODEL_NAME}"


def test_key_ignores_local_directory():
    """로컬 경로가 달라도 같은 키 — 학습 머신과 서빙의 디렉토리 구조가 달라도 된다."""
    a = model_s3_key(f"/pod/RobuSTAM/mlruns/models/{MODEL_NAME}", "robustam")
    b = model_s3_key(f"C:/other/place/{MODEL_NAME}", "robustam")
    assert a == b


# ── 업로드·다운로드 ──
@mock_aws
def test_upload_then_download_roundtrip(tmp_path):
    """학습 머신에서 올린 파일이 서빙 환경의 같은 상대경로로 복원돼야 한다."""
    src = _make_model(tmp_path, b"policy-bytes-123")
    s3 = boto3.client("s3", region_name="us-east-1")
    s3.create_bucket(Bucket=BUCKET)

    key = upload_model(src, BUCKET, "robustam", client=s3)
    assert key == f"robustam/models/{MODEL_NAME}"
    assert s3.list_objects_v2(Bucket=BUCKET)["KeyCount"] == 1

    # 서빙 환경(다른 디렉토리)에서 config의 model_path로 내려받기
    serving = tmp_path / "serving" / "mlruns" / "models" / MODEL_NAME
    got = download_model(serving, BUCKET, "robustam", client=s3)

    assert got == serving
    assert serving.read_bytes() == b"policy-bytes-123"  # 바이트 동일


@mock_aws
def test_download_creates_parent_dirs(tmp_path):
    """서빙 환경에 mlruns/models가 없어도 알아서 만든다(첫 배포 시나리오)."""
    src = _make_model(tmp_path)
    s3 = boto3.client("s3", region_name="us-east-1")
    s3.create_bucket(Bucket=BUCKET)
    upload_model(src, BUCKET, "robustam", client=s3)

    fresh = tmp_path / "brand-new" / "mlruns" / "models" / MODEL_NAME
    assert not fresh.parent.exists()

    download_model(fresh, BUCKET, "robustam", client=s3)
    assert fresh.is_file()


@mock_aws
def test_upload_missing_file_raises(tmp_path):
    s3 = boto3.client("s3", region_name="us-east-1")
    s3.create_bucket(Bucket=BUCKET)
    with pytest.raises(FileNotFoundError, match="policy 파일이 없습니다"):
        upload_model(tmp_path / "nope.zip", BUCKET, "robustam", client=s3)


@mock_aws
def test_download_to_explicit_dest(tmp_path):
    """dest를 주면 그 경로로 받는다(임시 검증 등)."""
    src = _make_model(tmp_path)
    s3 = boto3.client("s3", region_name="us-east-1")
    s3.create_bucket(Bucket=BUCKET)
    upload_model(src, BUCKET, "robustam", client=s3)

    dest = tmp_path / "elsewhere.zip"
    got = download_model(src, BUCKET, "robustam", client=s3, dest=dest)
    assert got == dest and dest.is_file()
