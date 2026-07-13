#!/usr/bin/env bash
# 로컬 개발용 self-signed 인증서 생성 (git에 커밋하지 않음 — .gitignore 처리됨)
# 사용: bash docker/nginx/generate_cert.sh
set -e

CERT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/certs"
mkdir -p "$CERT_DIR"

# 맨 앞 // : Git Bash(Windows)가 "/CN=localhost"를 파일 경로로 오인해 변환하는 것을
# 막는 MSYS 관례(이중 슬래시는 경로 변환에서 제외됨). 다른 OS의 openssl도 "//CN=..."을
# 빈 첫 세그먼트로 무시하고 동일하게 파싱하므로 안전하다.
openssl req -x509 -nodes -days 365 \
  -newkey rsa:2048 \
  -keyout "$CERT_DIR/selfsigned.key" \
  -out "$CERT_DIR/selfsigned.crt" \
  -subj "//CN=localhost"

echo "생성 완료: $CERT_DIR/selfsigned.crt, $CERT_DIR/selfsigned.key"
