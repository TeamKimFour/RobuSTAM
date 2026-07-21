"""로컬 data 디렉토리를 S3에 업로드 — Feature Store 팀 공유·서비스 배포용.

`build.py`는 산출물을 로컬에 쓴다(특히 `meta.sqlite`는 s3://에 직접 못 씀). 이 모듈이 그 로컬
디렉토리를 파일 단위로 S3에 올린다. 자격증명은 환경(AWS_*)·IAM Role에서 boto3가 자동 로드하며,
config `data.s3_bucket`이 비어 있으면 업로드를 건너뛴다(로컬 개발 시 무해).

실행: `python -m src.data.s3_sync`
"""

from pathlib import Path


def sync_dir_to_s3(
    local_dir: str,
    bucket: str,
    prefix: str = "",
    *,
    client=None,
) -> list[str]:
    """local_dir 아래 모든 파일을 s3://bucket/prefix/<상대경로>로 업로드한다.

    반환: 업로드한 S3 키 목록. client 미지정 시 `boto3.client('s3')` 사용(테스트는 mock 주입).
    """
    import boto3

    local = Path(local_dir)
    if not local.is_dir():
        raise FileNotFoundError(f"업로드할 로컬 디렉토리가 없습니다: {local}")

    s3 = client or boto3.client("s3")
    prefix = prefix.strip("/")
    uploaded: list[str] = []
    for path in sorted(local.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(local).as_posix()
        key = f"{prefix}/{rel}" if prefix else rel
        s3.upload_file(str(path), bucket, key)
        uploaded.append(key)
    return uploaded


def main() -> None:
    """config 기반 업로드 — raw·feature_store·precompute 디렉토리를 S3에 올린다.

    버킷·프리픽스는 환경변수(`S3_BUCKET`/`S3_PREFIX`)로 override 가능(CI에서 GH vars 주입).

    precompute(`data.precompute_path`의 부모 디렉토리)는 아직 한 번도 안 돌았을 수 있다
    (로컬, 또는 daily.yml에서 이 스텝보다 먼저 도는 게 아직 없던 환경). `sync_dir_to_s3`는
    없는 디렉토리에 예외를 던지므로, 여기서 먼저 존재를 확인해 skip한다 — 그렇지 않으면
    raw·feature_store가 이미 올라간 뒤 precompute에서 터져 부분 실패로 헷갈리게 된다
    (도현, PR #47 코멘트).
    """
    import os

    from src.config_loader import load_config

    cfg = load_config()
    d = cfg["data"]
    bucket = os.environ.get("S3_BUCKET") or d.get("s3_bucket")
    if not bucket:
        print("S3 버킷 미설정(config·환경변수 모두 없음) → 업로드 skip")
        return

    base = (os.environ.get("S3_PREFIX") or d.get("s3_prefix") or "").strip("/")
    targets = [
        ("raw", d["raw_dir"]),
        ("feature_store", d["feature_store_dir"]),
        ("precompute", str(Path(d["precompute_path"]).parent)),  # 익일 추천 비중 — 서빙이 소비(§8-2)
    ]
    total = 0
    for name, local_dir in targets:
        if not Path(local_dir).is_dir():
            print(f"  {name}: 디렉토리 없음({local_dir}) → skip")
            continue
        prefix = f"{base}/{name}" if base else name
        keys = sync_dir_to_s3(local_dir, bucket, prefix)
        print(f"  {name}: {len(keys)}개 → s3://{bucket}/{prefix}/")
        total += len(keys)
    print(f"업로드 완료: 총 {total}개 파일")


if __name__ == "__main__":
    main()
