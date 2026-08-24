# RobuSTAM — 자주 쓰는 명령 단축 (6주차 도현 1순위)
#
# 긴 docker·실험 명령을 팀 전체가 동일한 방식으로 실행하기 위한 얇은 래퍼다.
# 로직은 넣지 않는다 — 실제 동작은 전부 src/ 모듈과 docker-compose.yml에 있다.
#
#   make help                            사용 가능한 명령 목록
#   make mlflow-up                       MLflow 트래킹 서버 기동 (팀 공용 DB 연결)
#   make train FOLD=1 SEED=42            PPO 학습
#
# 보안 규칙 (팀 주간계획 "주의사항"):
#   - .env 값(DATABASE_URL·AWS 키)을 화면·로그에 절대 출력하지 않는다.
#     docker compose가 env_file로 직접 읽으므로 Makefile은 값을 만지지 않는다.
#   - 실패한 명령의 exit code를 그대로 전파한다 (|| true 금지).
#   - 이미 떠 있는 컨테이너를 중복 생성하지 않는다 (compose up -d가 재사용).
#
# Windows에서는 make가 기본 제공되지 않는다. Git Bash + choco install make 또는
# WSL에서 실행할 것 (레시피는 POSIX sh 기준으로 작성돼 있다).

SHELL := /bin/sh
.DEFAULT_GOAL := help

# ── 오버라이드 가능한 변수 ──
# 주의: `VAR ?= 값  # 주석` 형태로 쓰면 주석 앞 공백이 값에 그대로 붙는다(make 규칙).
#       ENV_FILE 같은 경로 변수는 그 공백 때문에 test -f가 깨지므로 주석은 위에 따로 둔다.

# macOS/Linux에서 실행 파일이 python3면: make PYTHON=python3 ...
PYTHON      ?= python
COMPOSE     ?= docker compose
# docker-compose.yml의 env_file과 같은 파일을 가리켜야 한다.
ENV_FILE    ?= .env
CONFIG      ?= config/config.yaml
MLFLOW_PORT ?= 5000

# M0|M1|M2|M3 — 생략 시 config의 active_combo를 따른다(단일 콤보용).
FEATURE_SET ?=
# experiment 전용 — 여러 콤보를 공백으로: COMBOS="full M0 M1". 생략 시 러너 기본값.
COMBOS      ?=
# 생략 시 config.split의 전체 fold 순회.
FOLD        ?=
SEED        ?= 42
# 생략 시 config.model의 기본 학습량.
TIMESTEPS   ?=
SPLIT       ?= test

PYTEST      ?= $(PYTHON) -m pytest

# 테스트 그룹 — 파일이 추가되면 여기에 함께 등록한다.
DATA_TESTS  := tests/test_assemble.py tests/test_build.py tests/test_collect.py \
               tests/test_feature_store.py tests/test_features.py tests/test_normalize.py \
               tests/test_returns.py tests/test_schema.py tests/test_splits.py \
               tests/test_validate.py tests/test_analyze_features.py \
               tests/test_screen_candidates.py tests/test_screen_features.py \
               tests/test_s3_sync.py tests/test_config.py
MODEL_TESTS := tests/test_train.py tests/test_select.py tests/test_publish.py \
               tests/test_env.py tests/test_discrete_env.py tests/test_env_robustness.py \
               tests/test_reward.py tests/test_cost_shaping.py tests/test_vol_penalty.py \
               tests/test_variable_state_dim.py tests/test_api.py \
               tests/test_precompute.py tests/test_s3_fetch.py
BT_TESTS    := tests/test_engine.py tests/test_benchmark.py tests/test_runner.py \
               tests/test_backtest_policy.py tests/test_backtest_export.py \
               tests/test_s3_results.py

.PHONY: help mlflow-build mlflow-up mlflow-down mlflow-logs mlflow-health mlflow-smoke \
        build-features screen train backtest experiment test test-data test-model test-backtest \
        require-env require-combo require-screen-combos

# ─────────────────────────── help ───────────────────────────

help: ## 이 도움말 출력
	@echo "RobuSTAM Makefile — 사용법: make <target> [VAR=값]"
	@echo ""
	@grep -E '^[a-zA-Z0-9_-]+:.*## ' $(MAKEFILE_LIST) \
	  | sort \
	  | awk 'BEGIN {FS = ":.*## "}; {printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}'
	@echo ""
	@echo "변수: FEATURE_SET(M0~M3) COMBOS FOLD SEED TIMESTEPS SPLIT CONFIG PYTHON ENV_FILE MLFLOW_PORT"
	@echo "예시: make build-features FEATURE_SET=M1"
	@echo "      make train FEATURE_SET=M1 FOLD=1 SEED=42 TIMESTEPS=200000"
	@echo "      make backtest FEATURE_SET=M1 FOLD=1 SPLIT=test"
	@echo "      make experiment COMBOS=\"full M0 M1 M2 M3\" TIMESTEPS=200000"
	@echo ""
	@echo "주의: train·backtest·experiment는 MLflow 서버가 떠 있어야 한다(make mlflow-up)."
	@echo "      config.model.mlflow_tracking_uri가 팀 공용 서버를 가리키기 때문이다."

# ─────────────────────────── 가드 ───────────────────────────

# .env 존재만 확인한다 — 내용은 읽지도 출력하지도 않는다.
require-env:
	@test -f "$(ENV_FILE)" || { \
	  echo "ERROR: $(ENV_FILE) 가 없습니다. .env.example을 복사한 뒤 팀 vault 값으로 채우세요."; \
	  echo "       (DATABASE_URL·S3_BUCKET·AWS 키. 값은 커밋·공유 금지)"; \
	  exit 1; }

# PR #69(민지, 피처 콤보) 머지 전에는 build(combo=...)가 없다 — 조용히 full을 빌드하지 않도록 막는다.
require-combo:
	@$(PYTHON) -c "import inspect, sys; from src.data.build import build; \
sys.exit(0 if 'combo' in inspect.signature(build).parameters else 1)" || { \
	  echo "ERROR: src.data.build가 아직 combo 인자를 받지 않습니다."; \
	  echo "       피처 콤보(M0~M3)는 PR #69 머지 후 사용 가능합니다."; \
	  exit 1; }

require-screen-combos:
	@$(PYTHON) -c "import importlib.util, sys; \
sys.exit(0 if importlib.util.find_spec('src.data.screen_combos') else 1)" || { \
	  echo "ERROR: src.data.screen_combos 모듈이 없습니다 (PR #69 머지 후 사용 가능)."; \
	  exit 1; }

# train.py·runner.py는 아직 콤보를 소비하지 않는다(6주차 도현 2순위).
# FEATURE_SET을 조용히 무시하고 full로 학습하면 실험이 오염되므로 명시적으로 실패시킨다.
# train·backtest가 콤보를 소비하게 된 뒤로는(6주차 2순위 완료) FEATURE_SET을 그대로
# --combo로 넘긴다. 생략하면 config의 active_combo를 따른다.
COMBO_ARG = $(if $(strip $(FEATURE_SET)),--combo $(FEATURE_SET),)

# ────────────────────────── MLflow ──────────────────────────

mlflow-build: ## MLflow 서버 이미지 빌드
	$(COMPOSE) build mlflow

mlflow-up: require-env ## MLflow 서버 기동 (기본 localhost:5000, 이미 떠 있으면 재사용)
	$(COMPOSE) up -d mlflow
	@echo "MLflow UI: http://localhost:$(MLFLOW_PORT)"

mlflow-down: ## MLflow 서버 중지·컨테이너 제거 (fastapi는 건드리지 않음)
	$(COMPOSE) rm -sf mlflow

mlflow-logs: ## MLflow 서버 로그 추적 (Ctrl-C로 종료)
	$(COMPOSE) logs -f mlflow

mlflow-health: ## MLflow 서버 헬스체크
	@curl -fsS "http://localhost:$(MLFLOW_PORT)/health" \
	  && echo "  <- MLflow OK (port $(MLFLOW_PORT))" \
	  || { echo "MLflow 응답 없음 — make mlflow-up 후 make mlflow-logs로 확인하세요."; exit 1; }

mlflow-smoke: ## 공용 DB 연결 확인용 더미 run 1건 기록
	@$(PYTHON) -c "import mlflow; \
mlflow.set_tracking_uri('http://localhost:$(MLFLOW_PORT)'); \
mlflow.set_experiment('robustam-smoke'); \
run = mlflow.start_run(run_name='smoke'); \
mlflow.log_param('source', 'makefile'); \
mlflow.log_metric('ok', 1); \
mlflow.end_run(); \
print('smoke run 기록 완료: run_id=' + run.info.run_id)"

# ──────────────────── 데이터 · 스크리닝 ────────────────────

build-features: ## Feature Store 빌드 (FEATURE_SET 지정 시 해당 콤보만)
ifeq ($(strip $(FEATURE_SET)),)
	$(PYTHON) -m src.data.build
else
	@$(MAKE) --no-print-directory require-combo
	$(PYTHON) -c "from src.data.build import build; build(combo='$(FEATURE_SET)')"
endif

screen: require-screen-combos ## 콤보 스크리닝 + MLflow 기록 (Baseline·M0~M3 일괄)
	$(PYTHON) -m src.data.screen_combos

# ──────────────────── 학습 · 백테스트 ────────────────────

train: ## PPO 학습 (FEATURE_SET·FOLD·SEED·TIMESTEPS)
	$(PYTHON) -m src.models.train --config $(CONFIG) --seed $(SEED) $(COMBO_ARG) \
	  $(if $(strip $(FOLD)),--fold-id $(FOLD),) \
	  $(if $(strip $(TIMESTEPS)),--total-timesteps $(TIMESTEPS),)

backtest: ## 백테스트 (policy vs 1/N·60:40·B&H, FEATURE_SET·FOLD·SPLIT)
	$(PYTHON) -m src.backtest.runner --config $(CONFIG) --split $(SPLIT) $(COMBO_ARG) \
	  $(if $(strip $(FOLD)),--fold-id $(FOLD),)

experiment: ## 콤보 실험 러너 — 학습→백테스트→3지표 판정 (COMBOS·FOLD·SEED·TIMESTEPS)
	$(PYTHON) -m src.models.experiment --config $(CONFIG) --seed $(SEED) \
	  $(if $(strip $(COMBOS)),--combos $(COMBOS),) \
	  $(if $(strip $(FOLD)),--folds $(FOLD),) \
	  $(if $(strip $(TIMESTEPS)),--timesteps $(TIMESTEPS),)

# ────────────────────────── 테스트 ──────────────────────────

test: ## 전체 테스트 (CI와 동일)
	$(PYTEST) tests/ -v

test-data: ## 데이터·피처 파이프라인 테스트만
	$(PYTEST) $(DATA_TESTS) -v

test-model: ## 환경·보상·학습·서빙 테스트만
	$(PYTEST) $(MODEL_TESTS) -v

test-backtest: ## 백테스트 엔진·벤치마크 테스트만
	$(PYTEST) $(BT_TESTS) -v
