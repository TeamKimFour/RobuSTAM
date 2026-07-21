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
    """config 기반 업로드 — raw + feature_store 디렉토리를 S3에 올린다.

    버킷·프리픽스는 환경변수(`S3_BUCKET`/`S3_PREFIX`)로 override 가능(CI에서 GH vars 주입).
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

    # precompute latest.json 상위 디렉토리 — precompute가 아직 안 돌면(파일 없음) 건너뛴다.
    # 키 계약: {prefix}/precompute/latest.json (서빙측 src/inference/s3_fetch.py가 이 키를 받음).
    precompute_dir = str(Path(d.get("precompute_path", "data/precompute/latest.json")).parent)

    targets = [("raw", d["raw_dir"]), ("feature_store", d["feature_store_dir"])]
    if Path(precompute_dir).is_dir():
        targets.append(("precompute", precompute_dir))

    total = 0
    for name, local_dir in targets:
        prefix = f"{base}/{name}" if base else name
        keys = sync_dir_to_s3(local_dir, bucket, prefix)
        print(f"  {name}: {len(keys)}개 → s3://{bucket}/{prefix}/")
        total += len(keys)
    if not Path(precompute_dir).is_dir():
        print("  precompute: 디렉토리 없음 → skip (precompute 미실행 상태에서도 정상)")
    print(f"업로드 완료: 총 {total}개 파일")


if __name__ == "__main__":
    main()
