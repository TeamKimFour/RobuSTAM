"""s3_sync 테스트 — moto(가짜 S3)로 실제 AWS 없이 업로드 검증.

moto·boto3는 requirements-dev에만 있으므로 importorskip으로 가드(미설치 CI에선 skip).
"""

import pytest

pytest.importorskip("boto3")
pytest.importorskip("moto")

import boto3
from moto import mock_aws

from src.data.s3_sync import main, sync_dir_to_s3

BUCKET = "robustam-test"


def _fake_cfg(tmp_path, bucket: str = "") -> dict:
    return {
        "data": {
            "raw_dir": str(tmp_path / "raw"),
            "feature_store_dir": str(tmp_path / "feature_store"),
            "precompute_path": str(tmp_path / "precompute" / "latest.json"),
            "s3_bucket": bucket,
            "s3_prefix": "robustam",
        }
    }


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


@mock_aws
def test_sync_excludes_tmp_files(tmp_path):
    """원자적 쓰기(tmp→rename) 중 죽으면 남는 *.tmp 잔여물은 업로드 대상에서 제외한다(도현 리뷰)."""
    root = tmp_path / "precompute"
    root.mkdir()
    (root / "latest.json").write_bytes(b'{"date": "2026-07-24"}')
    (root / "abc123.tmp").write_bytes(b"partial-write")  # _atomic_write_json의 잔여물 시뮬레이션

    s3 = boto3.client("s3", region_name="us-east-1")
    s3.create_bucket(Bucket=BUCKET)

    keys = sync_dir_to_s3(str(root), BUCKET, "robustam/precompute")

    assert keys == ["robustam/precompute/latest.json"]
    got = {o["Key"] for o in s3.list_objects_v2(Bucket=BUCKET)["Contents"]}
    assert got == {"robustam/precompute/latest.json"}


# ── main() — precompute 업로드 대상 추가 (도현, PR #47 코멘트) ──
@mock_aws
def test_main_skips_missing_precompute_dir(tmp_path, monkeypatch, capsys):
    """precompute가 한 번도 안 돈 환경(로컬 등) — 나머지는 올라가고 precompute만 skip, 예외 없음."""
    (tmp_path / "raw").mkdir()
    (tmp_path / "raw" / "prices.parquet").write_bytes(b"x")
    (tmp_path / "feature_store").mkdir()
    (tmp_path / "feature_store" / "meta.sqlite").write_bytes(b"x")
    # precompute 디렉토리는 만들지 않는다 — sync_dir_to_s3라면 FileNotFoundError.

    s3 = boto3.client("s3", region_name="us-east-1")
    s3.create_bucket(Bucket=BUCKET)
    monkeypatch.setenv("S3_BUCKET", BUCKET)
    monkeypatch.setattr("src.config_loader.load_config", lambda: _fake_cfg(tmp_path))

    main()  # 예외 없이 끝나야 한다

    got = {o["Key"] for o in s3.list_objects_v2(Bucket=BUCKET)["Contents"]}
    assert "robustam/raw/prices.parquet" in got
    assert "robustam/feature_store/meta.sqlite" in got
    assert not any("precompute" in k for k in got)
    assert "precompute" in capsys.readouterr().out


@mock_aws
def test_main_uploads_precompute_when_present(tmp_path, monkeypatch):
    """precompute가 이미 돈 환경 — latest.json도 s3_fetch.py가 기대하는 키로 업로드된다(§8-2)."""
    (tmp_path / "raw").mkdir()
    (tmp_path / "feature_store").mkdir()
    (tmp_path / "precompute").mkdir()
    (tmp_path / "precompute" / "latest.json").write_bytes(b'{"date": "2026-07-21"}')

    s3 = boto3.client("s3", region_name="us-east-1")
    s3.create_bucket(Bucket=BUCKET)
    monkeypatch.setenv("S3_BUCKET", BUCKET)
    monkeypatch.setattr("src.config_loader.load_config", lambda: _fake_cfg(tmp_path))

    main()

    got = {o["Key"] for o in s3.list_objects_v2(Bucket=BUCKET)["Contents"]}
    # src/inference/s3_fetch.py의 계약 키(§8-2)와 정확히 일치해야 수신 쪽이 찾을 수 있다.
    assert "robustam/precompute/latest.json" in got
