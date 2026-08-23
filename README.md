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

## Makefile (자주 쓰는 명령 단축)

긴 docker·실험 명령을 팀 전체가 같은 방식으로 실행하도록 `Makefile`에 모아뒀다.
아래 문서의 원본 명령은 그대로 유효하며, Makefile은 그 위에 얹은 얇은 래퍼다.

```bash
make help
```

| 명령 | 하는 일 |
| --- | --- |
| `make mlflow-build` / `mlflow-up` / `mlflow-down` | MLflow 서버 이미지 빌드 · 기동 · 중지 |
| `make mlflow-logs` / `mlflow-health` / `mlflow-smoke` | 로그 추적 · 헬스체크 · 더미 run 1건 기록 |
| `make build-features [FEATURE_SET=M1]` | Feature Store 빌드 |
| `make screen` | 콤보 스크리닝 + MLflow 기록 |
| `make train [FEATURE_SET=M1] [FOLD=1] [SEED=42] [TIMESTEPS=200000]` | RL 학습 (기본 PPO. `config.model.algorithm: DQN`이면 이산 행동 학습 — docs/model_training.md §7-5) |
| `make backtest [FEATURE_SET=M1] [FOLD=1] [SPLIT=test]` | policy vs 벤치마크 백테스트 |
| `make experiment [COMBOS="full M0 M1"] [TIMESTEPS=200000]` | 콤보 실험 러너 (학습→백테스트→3지표 판정) |
| `make test` / `test-data` / `test-model` / `test-backtest` | 테스트 (`make test`는 CI와 동일) |

변수는 `make <target> VAR=값`으로 덮어쓴다. 실행 파일이 `python3`면 `make PYTHON=python3 ...`.

**전제**
- `mlflow-*` 계열은 프로젝트 루트에 `.env`가 있어야 한다(`.env.example` 복사 후 팀 vault 값으로 채움).
  Makefile은 존재 여부만 확인하고 **값을 읽거나 출력하지 않는다** — `docker compose`가 `env_file`로 직접 읽는다.
- `FEATURE_SET`(M0~M3)은 `build-features`·`train`·`backtest`에 모두 연결돼 있다. 생략하면
  config의 `active_combo`(기본 `full`)를 따른다. 여러 콤보를 한 번에 비교하려면 `make experiment
  COMBOS="full M0 M1 M2 M3"`을 쓴다.
- **`train`·`backtest`·`experiment`는 MLflow 서버가 떠 있어야 한다** (`make mlflow-up`).
  `config.model.mlflow_tracking_uri`가 팀 공용 서버(`http://localhost:5000`)를 가리키기 때문이다.
- **Windows에는 `make`가 기본 제공되지 않는다.** WSL 또는 Git Bash + `choco install make`로 실행할 것
  (레시피는 POSIX sh 기준).

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

## Docker Compose 실행 (로컬 개발)

### FastAPI 추론 서버 (`fastapi`)

FastAPI 추론 서버(현재는 헬스체크 placeholder, `src/api/` — 도현 실제 코드로 교체 예정)를
로컬에서 띄운다.

```bash
docker compose up --build fastapi
```

- FastAPI: `http://localhost:8000/health` → `{"status": "ok"}` 응답 확인

프로덕션 배포는 Render Blueprint(`render.yaml`)로 대체됨 — nginx self-signed HTTPS,
Cloudflare Tunnel 방식은 더 이상 쓰지 않음 (Render가 HTTPS/도메인 자동 처리).

### MLflow 트래킹 서버 (`mlflow`)

팀 공용 Render Postgres에 붙는 로컬 MLflow 서버. 팀원 각자 로컬에서 이 서비스를 띄우면
동일한 DB를 공유하므로 서로 experiment·run을 볼 수 있다.

**사전 준비**: `.env.example`을 `.env`로 복사한 뒤 팀 vault에서 받은 `DATABASE_URL`·
`AWS_*`·`S3_BUCKET` 값을 채운다. `DATABASE_URL`을 비우면 컨테이너가 SQLite로
조용히 폴백되어 팀 공용 DB에 안 붙으니 주의.

```bash
docker compose up --build -d mlflow
```

- MLflow UI: `http://localhost:5000`
- 학습 스크립트에서: `MLFLOW_TRACKING_URI=http://localhost:5000`

Render 위에 상시 배포된 MLflow 웹 서비스는 Free 512MB OOM으로 접었고(관련 논의는
PR #66 참조), 팀 공용 저장 계층(Postgres·S3)만 재활용한다.

## S3 업로드 (선택 — 클라우드 공유)

Feature Store를 S3에 올려 팀·서비스가 공유한다. `config.data.s3_bucket`(또는 환경변수 `S3_BUCKET`)이
비어 있으면 자동 skip. `meta.sqlite`도 파일로 업로드한다(s3:// 직접쓰기 불가 회피).

```bash
S3_BUCKET=my-bucket S3_PREFIX=robustam python -m src.data.s3_sync
```

> 매일 자동 실행: `.github/workflows/daily.yml`(KST 07:00 cron) →
> collect→build→policy.zip 수신→precompute→백테스트 리포트→S3 업로드.
> **S3 연결 활성 완료**(2026-07-19 실연결 검증) — 버킷 `robustam-teamkimfour`, 인증은 GitHub
> OIDC(`vars.AWS_ROLE_ARN`)라 액세스 키가 필요 없다. 상세는 `docs/data_pipeline.md` §7·§8-2.

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

## 프론트엔드 (Next.js 대시보드)

라우트 구조: [docs/frontend.md](docs/frontend.md)

```bash
cd frontend
npm install
cp .env.example .env.local     # NEXT_PUBLIC_API_BASE_URL 확인 (기본 http://localhost:8000)
npm run dev                    # http://localhost:3000
```

- `/` — 랜딩 (Backtest / Inference / Docs)
- `/inference` — FastAPI `/inference/latest` 실시간 조회 (배치 미준비 시 503 안내 표시)
- `/backtest` — 성과곡선·Drawdown·벤치마크 비교. `frontend/public/backtest.json`이 있으면 실데이터, 없으면 mock으로 자동 fallback(뱃지로 상태 표시). 생성: `python -m src.backtest.export` ([docs/backtest_engine.md §4-4](docs/backtest_engine.md))
  - **Vercel 배포**에서는 `npm run prebuild`가 `BACKTEST_JSON_URL`에서 다운로드해 `public/`에 배치한다. 미설정 시 자동 skip → mock으로 배포된다(빌드 실패 없음). 백엔드가 S3에 올리려면 `python -m src.backtest.export --upload-s3`.

FastAPI를 함께 띄우려면:

```bash
uvicorn src.api.main:app --reload
```

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
