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


# ── main(): precompute(latest.json) 업로드 — 서빙측 s3_fetch 키 계약 ──
def _cfg_with_dirs(tmp_path, with_precompute: bool):
    raw = tmp_path / "raw"; raw.mkdir()
    (raw / "prices_raw.parquet").write_bytes(b"x")
    fsd = tmp_path / "fs"; (fsd / "fold=1" / "split=train").mkdir(parents=True)
    (fsd / "meta.sqlite").write_bytes(b"s")
    pc = tmp_path / "precompute"
    if with_precompute:
        pc.mkdir()
        (pc / "latest.json").write_text('{"date": "2025-12-30"}', encoding="utf-8")
    return {
        "data": {
            "raw_dir": str(raw), "feature_store_dir": str(fsd),
            "precompute_path": str(pc / "latest.json"),
            "s3_bucket": BUCKET, "s3_prefix": "robustam",
        }
    }


@mock_aws
def test_main_uploads_precompute_latest_json(tmp_path, monkeypatch):
    """latest.json이 있으면 {prefix}/precompute/latest.json로 올라간다(찬휘 s3_fetch 키 계약)."""
    monkeypatch.setattr("src.config_loader.load_config",
                        lambda *a, **k: _cfg_with_dirs(tmp_path, with_precompute=True))
    monkeypatch.delenv("S3_BUCKET", raising=False)
    monkeypatch.delenv("S3_PREFIX", raising=False)
    s3 = boto3.client("s3", region_name="us-east-1")
    s3.create_bucket(Bucket=BUCKET)

    from src.data import s3_sync
    s3_sync.main()

    got = {o["Key"] for o in s3.list_objects_v2(Bucket=BUCKET)["Contents"]}
    assert "robustam/precompute/latest.json" in got  # ← 서빙측 다운로드 키와 일치


@mock_aws
def test_main_skips_precompute_when_absent(tmp_path, monkeypatch):
    """precompute 미실행(디렉토리 없음)이어도 실패하지 않고 raw·feature_store만 올린다."""
    monkeypatch.setattr("src.config_loader.load_config",
                        lambda *a, **k: _cfg_with_dirs(tmp_path, with_precompute=False))
    monkeypatch.delenv("S3_BUCKET", raising=False)
    monkeypatch.delenv("S3_PREFIX", raising=False)
    s3 = boto3.client("s3", region_name="us-east-1")
    s3.create_bucket(Bucket=BUCKET)

    from src.data import s3_sync
    s3_sync.main()  # 예외 없이 종료해야 함

    got = {o["Key"] for o in s3.list_objects_v2(Bucket=BUCKET)["Contents"]}
    assert not any("precompute" in k for k in got)
    assert "robustam/feature_store/meta.sqlite" in got
