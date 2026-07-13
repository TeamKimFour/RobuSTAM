# RobuSTAM — PPO 학습 파이프라인 설계서 v0.1

> RL 학습 레이어(`src/models/`)의 구조·실행 방법·RunPod 가이드.
> Feature Store 계약은 [docs/data_pipeline.md](data_pipeline.md), 환경 계약은
> [docs/state_spec.md](state_spec.md)가 SSOT이며, 본 문서는 그 위에서 도는 학습
> 스크립트를 설명한다.

---

## 1. 범위 (현재 구현 상태)

| 단계 | 모듈 | 상태 |
|---|---|---|
| fold 환경 로더 | `src/models/train.py::load_fold_env` | ✅ 구현 |
| PPO 학습 + MLflow 기록 | `src/models/train.py::train` | ✅ 구현 |
| 간이 검증(valid split 1에피소드) | `src/models/train.py::evaluate` | ✅ 구현 |
| 학습 컨테이너 | `docker/Dockerfile.train` | ✅ 구현 |
| RunPod 실행 | — | ⏳ 실제 GPU 파드 실행은 팀원이 수행(계정·과금 필요) |
| SAC 비교 | — | ⏳ 시간 여유 시(CLAUDE.md §3) |

---

## 2. 데이터 흐름

```
data/feature_store/fold=<id>/split=train  ──load_fold_env──>  PortfolioEnv
                                                                     │
                                                        _BoundedActionWrapper
                                                                     │
                                                        stable_baselines3.PPO.learn()
                                                                     │
                                              ┌──────────────────────┴───────────────────────┐
                                        mlruns/models/ppo_fold<id>_<run_id>.zip          MLflow run
                                        (policy 아티팩트)                        (params·valid 지표·아티팩트)
```

---

## 3. 모듈별 책임

- **`load_fold_env(cfg, fold_id, split)`** — `feature_store.load_features`/`load_targets`로
  지정 fold/split을 읽어 `PortfolioEnv`를 만들고, `_BoundedActionWrapper`로 감싸 반환한다.
- **`_BoundedActionWrapper`** — SB3 PPO는 연속 행동공간에 유한 상하한을 요구하지만,
  `PortfolioEnv.action_space`는 정책망 로짓을 그대로 받는 `-inf~+inf` Box다(softmax는
  env `step()` 내부 처리, 팀 계약). env 자체는 바꾸지 않고, SB3에 보여줄 bound만
  `config.model.action_bound`(기본 10)로 좁힌다 — 값은 그대로 통과(identity transform).
- **`train(config_path, fold_id, total_timesteps)`** — config `model` 섹션에서 하이퍼파라미터를
  읽어 PPO를 학습하고 MLflow run 안에서 파라미터·valid 지표·policy 아티팩트(`.zip`)를 기록한다.
  1차 알고리즘은 PPO만 지원(CLAUDE.md §3, SAC 시도 시 `algorithm` 분기 추가 필요).
- **`evaluate(model, cfg, fold_id, split)`** — 정책을 결정적으로 1 에피소드 굴려
  `{split}_total_log_return` 등을 요약한다. 학습 중 조기 확인용이며, 정식 성과 검증은
  찬휘 백테스트 엔진([docs/backtest_engine.md](backtest_engine.md))이 맡는다.

---

## 4. 설계 결정 및 발견된 이슈

### 4-1. mlflow-skinny 채택 (mlflow 풀 패키지 아님)
`mlflow==3.14.0`은 `pandas<3`을 요구해 프로젝트 고정값 `pandas==3.0.3`(§2)과 직접
충돌한다(`pip install`로 실제 검증함 — `ResolutionImpossible`). `mlflow-skinny`는 동일한
`import mlflow` 네임스페이스를 쓰는 경량 클라이언트로, `log_params`/`log_metrics`/
`log_artifact`/`start_run` 등 로컬 파일 트래킹에 필요한 API를 그대로 제공하면서 충돌이 없다.
Flask 기반 트래킹 서버 UI·matplotlib 등 무거운 부속만 빠진다(로컬 파일 스토어 사용에는 불필요).

### 4-2. MLFLOW_ALLOW_FILE_STORE 필요
mlflow-skinny 3.14+에서 순수 파일 트래킹 백엔드(`mlruns/`)가 "유지보수 모드"로 바뀌어
기본적으로 막혀 있다(`MlflowException`). `train()`이 `MLFLOW_ALLOW_FILE_STORE=true`를
자동으로 설정해 우회한다. 추후 팀 규모가 커지면 `sqlite:///mlruns/mlflow.db` 같은 DB
백엔드로 옮기는 게 mlflow 공식 권장 방향이다.

### 4-3. Windows 로컬 경로 → file:// URI 정규화
Windows에서 `C:\Users\...\mlruns`를 트래킹 URI로 그대로 넘기면 드라이브 문자 `C:`가
URI scheme으로 오인돼 깨진다(`UnsupportedModelRegistryStoreURIException`, 직접 재현·확인함).
`_to_tracking_uri()`가 로컬 경로를 `Path.resolve().as_uri()`로 정규화해 방지한다.
RunPod(Linux) 환경에서는 원래도 문제 없는 경로지만, 로컬 Windows 개발 환경 호환을 위해 유지한다.

### 4-4. action_bound=10 선택 근거
softmax는 스케일에 민감하다. 로짓 범위가 좁으면(예: ±1) 정책이 강한 선호를 표현하기
어렵고, 너무 넓으면(예: ±1000) SB3 기본 정책망 초기화·그래디언트 스케일과 안 맞을 수
있다. ±10이면 `softmax([10,-10,-10,-10,-10]) ≈ [0.9999..., ...]`로 사실상 원핫에 가까운
선호까지 표현 가능해 실용적 상한으로 충분하다. 필요 시 `config.model.action_bound`만
바꾸면 된다(코드 변경 불필요).

---

## 5. 실행

### 로컬 (CPU, 빠른 스모크 테스트)
```bash
# 0) Feature Store가 비어 있다면 먼저 채운다 (docs/data_pipeline.md §6)
python -m src.data.collect
python -m src.data.build

# 1) 기본 config로 학습 (fold_id=0, total_timesteps=200000 — config.model 기본값)
python -m src.models.train

# 2) fold·timesteps 오버라이드
python -m src.models.train --fold-id 1 --total-timesteps 500000
```
학습 결과는 `config.model.model_dir`(기본 `mlruns/models/`)에 `.zip`으로, 실험 기록은
`config.model.mlflow_tracking_uri`(기본 `mlruns/`)에 남는다. `mlflow ui --backend-store-uri mlruns`
로 로컬에서 대시보드 확인 가능(mlflow 풀 패키지 별도 설치 필요 — mlflow-skinny엔 UI 서버 없음).

### RunPod (GPU)
```bash
# 1) 이미지 빌드 (로컬 또는 RunPod 빌드 환경)
docker build -t robustam-train -f docker/Dockerfile.train .

# 2) GPU 파드에서 실행 — data/feature_store가 비어 있으면 collect→build를 자동으로 먼저 돈다
docker run --gpus all \
  -v $PWD/data:/app/data \
  -v $PWD/mlruns:/app/mlruns \
  robustam-train --fold-id 0 --total-timesteps 1000000
```
PyPI `torch` 휠이 CUDA 런타임을 포함하므로 별도 CUDA 베이스 이미지 없이도, RunPod가
컨테이너에 GPU를 노출해주면(`--gpus all`, RunPod 템플릿에서 기본 지원) 그대로 GPU
학습이 된다. RunPod 계정 생성·파드 기동·과금은 팀원이 직접 수행한다(자동화 범위 밖).

---

## 6. 아직 정해지지 않은 것

- **모델 선택 기준**: 여러 fold·seed로 학습한 policy 중 어떤 걸 precompute에 쓸지
  (valid 지표 기준 선택 로직 미구현 — 현재는 매 학습이 독립 MLflow run으로만 쌓임).
- **precompute 연동**: 학습된 `.zip`을 로드해 `docs/data_pipeline.md §8`의 latest.json을
  생성하는 스크립트(민지 담당 영역과 접점, 모델 준비 후 착수).
- **SAC 비교**: `train()`의 `algorithm` 분기가 현재 PPO만 지원 — SAC 추가 시 액션 래퍼는
  그대로 재사용 가능(SB3 SAC도 유한 bound 요구는 동일).

---

## 변경 이력
| 버전 | 날짜 | 내용 |
|---|---|---|
| v0.1 | 2026-07-13 | 최초 작성. PPO 학습 스크립트·RunPod Dockerfile·mlflow-skinny 전환 근거 반영. |
