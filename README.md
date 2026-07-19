# RobuSTAM

여러 자산(주식·ETF·현금)의 비중을 강화학습 에이전트가 매 거래일 동적으로 조정해
위험조정수익(샤프지수)을 극대화하되, 거래비용·슬리피지 등 **현실적 제약 하에서 강건성을
정직하게 검증**하는 퀀트 MLOps 백테스팅·서비스 플랫폼.

- 자산 5종(고정): `SPY · EWY · TLT · GLD · SHV`
- 일봉 데일리 리밸런싱, 편도 거래비용 0.1%
- 자세한 규칙: [CLAUDE.md](CLAUDE.md) · State 명세: [docs/state_spec.md](docs/state_spec.md)
- 데이터 파이프라인: [docs/data_pipeline.md](docs/data_pipeline.md)

## 개발 환경 셋업

Python **3.12** 기준(CI와 동일 버전).

```bash
# 가상환경 생성·활성화
python3.12 -m venv .venv
source .venv/bin/activate

# 의존성 설치
pip install -r requirements.txt
```

## 데이터 수집

config(`config/config.yaml`)에 정의된 기간·자산으로 원시 일봉을 수집한다.

```bash
python -m src.data.collect
# → data/raw/prices_raw.parquet 생성
```

## Feature Store 빌드

원시 가격 → 187차원 정규화 State + 익일수익률(targets)을 fold/split 파티션으로 적재한다.

```bash
python -m src.data.build
# → data/feature_store/fold=*/split=*/{part,targets}.parquet + meta.sqlite
```

## Docker 실행 (파이프라인)

로컬·CI와 동일한 환경으로 collect·build를 컨테이너에서 실행한다. 데이터는 볼륨으로 마운트.

```bash
docker build -t robustam-pipeline -f docker/Dockerfile.pipeline .
docker run --rm -v "$PWD/data:/app/data" robustam-pipeline src.data.collect
docker run --rm -v "$PWD/data:/app/data" robustam-pipeline src.data.build
```

## Docker Compose 실행 (API + nginx HTTPS)

FastAPI 추론 서버(현재는 헬스체크 placeholder, `src/api/` — 도현 실제 코드로 교체 예정)를
nginx 리버스 프록시 뒤에서 self-signed HTTPS로 띄운다.

```bash
# 1) self-signed 인증서 생성 (최초 1회, git에 커밋되지 않음)
bash docker/nginx/generate_cert.sh

# 2) 빌드 후 실행
docker compose up --build
```

- FastAPI: `http://localhost:8000` (컨테이너 간 통신·직접 디버깅용)
- nginx(HTTPS): `https://localhost` → `/health` 호출 시 `{"status": "ok"}` 응답 확인

```bash
curl -k https://localhost/health
```

`-k`(또는 브라우저에서 "안전하지 않음" 경고 무시)가 필요한 이유: self-signed 인증서라
공인 CA가 서명하지 않았기 때문이다. 로컬 개발 환경에서는 정상이며, 실제 배포 시에는
공인 인증서(Let's Encrypt 등)로 교체해야 한다.

## Docker Compose 실행 (프로덕션 — API + Cloudflare Tunnel)

`docker-compose.prod.yml`은 로컬용 nginx(self-signed HTTPS) 대신 Cloudflare Tunnel이
공인 도메인의 TLS 종단을 전담하는 구조다. 호스트에 포트를 열지 않고 `cloudflared`가
아웃바운드로만 Cloudflare 엣지에 연결한다.

```bash
# 1) .env.prod 준비 (최초 1회, git에 커밋되지 않음)
cp .env.prod.example .env.prod
# CLOUDFLARE_TUNNEL_TOKEN 값을 Cloudflare Zero Trust 대시보드에서 발급받아 채운다.

# 2) 빌드 후 실행
docker compose -f docker-compose.prod.yml --env-file .env.prod up -d --build
```

> **TODO:** 실제 도메인이 아직 확정되지 않았다. 도메인 확정 후 Cloudflare 대시보드에서
> Public Hostname → `fastapi:8000` 라우팅을 설정해야 외부에서 접근 가능하다.

## S3 업로드 (선택 — 클라우드 공유)

Feature Store를 S3에 올려 팀·서비스가 공유한다. `config.data.s3_bucket`(또는 환경변수 `S3_BUCKET`)이
비어 있으면 자동 skip. `meta.sqlite`도 파일로 업로드한다(s3:// 직접쓰기 불가 회피).

```bash
S3_BUCKET=my-bucket S3_PREFIX=robustam python -m src.data.s3_sync
```

> 매일 자동 실행: `.github/workflows/daily.yml`(KST 07:00 cron) → collect→build→S3 업로드.
> **S3 연결 활성 완료**(2026-07-19 실연결 검증) — 버킷 `robustam-teamkimfour`, 인증은 GitHub
> OIDC(`vars.AWS_ROLE_ARN`)라 액세스 키가 필요 없다. 상세는 `docs/data_pipeline.md` §7.

## 백테스트 결과 S3 저장

`BacktestEngine.save_results(nav, metrics, run_id)`를 호출하면 NAV·성과지표·config
스냅샷을 S3에 저장한다. Feature Store 업로드(위 섹션)와 동일하게 `config.data.s3_bucket`
(또는 환경변수 `S3_BUCKET`)을 쓰고, 버킷이 비어 있으면 자동 skip한다.

```
s3://{버킷}/backtests/{실행날짜}_{실행ID}/
    nav.parquet    # NAV 시계열
    metrics.json   # Sharpe·MDD 등 성과 지표
    config.yaml    # 이 실행에 쓰인 config 원본 그대로
```

### 환경변수 설정 (AWS 자격증명)

AWS 자격증명은 코드에 하드코딩하지 않고 `.env` 파일(`python-dotenv`가 자동 로드)과
boto3 기본 자격증명 체인으로 처리한다.

```bash
cp .env.example .env
# .env를 열어 AWS_ACCESS_KEY_ID·AWS_SECRET_ACCESS_KEY·AWS_DEFAULT_REGION 채우기
```

`.env`는 `.gitignore` 처리되어 있어 커밋되지 않는다. 배포 환경(EC2/ECS 등)에서는 `.env` 없이
IAM Role을 쓰는 게 더 안전하다 — 그 경우 `.env` 값은 무시되고 Role 자격증명이 자동 적용된다.

## 테스트

```bash
pytest tests/ -v
```

## 데이터 품질 검증 · EDA

```bash
# 자동 품질 검사 + EDA 요약(통계·상관·이상치)
python -m src.data.validate

# 표·차트로 직접 탐색하는 노트북 (개발 의존성 필요)
pip install -r requirements-dev.txt
jupyter lab notebooks/eda_raw_prices.ipynb
```

## 프로젝트 구조 (구현된 부분)

```
config/config.yaml      # W·자산·거래비용·기간·분할 단일 출처(SSOT)
src/
  config_loader.py      # config 로드 + state_dim 산출
  data/
    schema.py           # 187차원 State 인덱스맵(코드측 SSOT)
    collect.py          # yfinance 수집
    returns.py          # 로그수익률·윈도우
    feature_store.py    # Feature Store 입출력 (Parquet 파티션 + SQLite 메타)
tests/                  # 차원·인덱스맵·룩어헤드 회귀 테스트
docs/                   # state_spec.md(State SSOT), data_pipeline.md
```
