"""S3 → 로컬 latest.json 동기화 — 서빙 컨테이너가 주기적으로 최신 추천 비중을 받아온다.

daily.yml 배치(민지 담당, PR #42 코멘트 논의)가 s3://{bucket}/{prefix}/precompute/latest.json에
업로드하면, 이 모듈이 그것을 로컬 `config.data.precompute_path`로 내려받는다. 프로덕션
서빙 컨테이너는 호스트와 데이터 볼륨을 공유하지 않는 stateless 구조(docker-compose.prod.yml)라,
기동 시 1회 + 주기적 재동기화로 스스로 데이터를 채운다.

S3 키 계약: `{s3_prefix}/precompute/latest.json` — `src/data/s3_sync.py`가 쓰는 raw/feature_store와
동일한 prefix 규칙을 따른다.

버킷 미설정이거나 키가 아직 없으면(초기 배포 등) 조용히 skip한다 — `src/api/main.py`가 이미
파일 없음을 503으로 정상 처리하므로 여기서 예외를 던지지 않는다.

실행: python -m src.inference.s3_fetch          # 1회
      python -m src.inference.s3_fetch --loop    # 기동 시 + 주기 반복(기본 300초)
"""

from __future__ import annotations

import os
import tempfile
import time
from pathlib import Path

from src.config_loader import get_precompute_path, load_config


def _s3_key(prefix: str) -> str:
    prefix = prefix.strip("/")
    return f"{prefix}/precompute/latest.json" if prefix else "precompute/latest.json"


def fetch_once(cfg: dict, *, client=None) -> bool:
    """S3에서 latest.json을 내려받아 원자적으로 덮어쓴다.

    반환: 성공 시 True, 버킷 미설정·키 없음·네트워크 오류 등으로 skip/실패 시 False
    (호출자가 서버 기동을 막지 않도록 예외를 올리지 않는다).
    """
    d = cfg["data"]
    bucket = os.environ.get("S3_BUCKET") or d.get("s3_bucket")
    if not bucket:
        print("[s3_fetch] S3 버킷 미설정 → skip")
        return False

    prefix = os.environ.get("S3_PREFIX") or d.get("s3_prefix") or ""
    key = _s3_key(prefix)
    dest = Path(get_precompute_path(cfg))
    dest.parent.mkdir(parents=True, exist_ok=True)

    import boto3

    s3 = client or boto3.client("s3")
    fd, tmp = tempfile.mkstemp(dir=str(dest.parent), suffix=".tmp")
    os.close(fd)
    try:
        s3.download_file(bucket, key, tmp)
        os.replace(tmp, dest)
        print(f"[s3_fetch] 동기화 완료: s3://{bucket}/{key} -> {dest}")
        return True
    except Exception as e:  # noqa: BLE001 — 외부 S3 호출 경계, 서버 기동을 막지 않기 위해 폭넓게 흡수
        print(f"[s3_fetch] 다운로드 실패(skip): s3://{bucket}/{key} — {e}")
        return False
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--loop", action="store_true", help="기동 시 1회 후 주기 반복")
    parser.add_argument(
        "--interval",
        type=int,
        default=int(os.environ.get("PRECOMPUTE_SYNC_INTERVAL_SEC", "300")),
        help="반복 간격(초, --loop 시에만 사용, 기본 300)",
    )
    args = parser.parse_args()

    cfg = load_config()
    fetch_once(cfg)
    while args.loop:
        time.sleep(args.interval)
        fetch_once(cfg)


if __name__ == "__main__":
    main()
