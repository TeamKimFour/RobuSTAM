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
| 간이 검증(valid split 1에피소드, `valid_sharpe` 포함) | `src/models/train.py::evaluate` | ✅ 구현 |
| 배포 policy 선택(valid 지표 최고 run) | `src/models/select.py::select_best_run` | ✅ 구현 |
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
                          mlruns/models/ppo_fold<id>_<build_run_id>_<mlflow_run_id>.zip     MLflow run
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
  학습 fold를 만든 build `run_id`를 `feature_store.latest_run_id_for_fold`로 조회해 MLflow
  파라미터(`feature_store_run_id`)·모델 파일명·반환 dict에 provenance로 남긴다(이슈 #27) — 배포 시
  `config.inference.scaler_run_id`에 그대로 넣어 추론 정규화를 학습과 일치시키기 위함. CLI 실행
  끝에 `config.inference`에 복사할 값(`model_path`/`scaler_run_id`/`scaler_fold_id`)을 출력한다.
- **`evaluate(model, cfg, fold_id, split)`** — 정책을 결정적으로 1 에피소드 굴려
  `{split}_total_log_return`·`{split}_sharpe`(연율화 위험조정수익 proxy = 스텝별 net
  로그보상의 mean/std × √252)·비용·회전율을 요약한다. 학습 중 조기 확인용이며, 정식 성과
  검증은 찬휘 백테스트 엔진([docs/backtest_engine.md](backtest_engine.md))이 맡는다.
- **`select.select_best_run(cfg, metric, fold_id, higher_is_better)`** — MLflow 실험에서
  `metric`(기본 `valid_sharpe`)이 가장 좋은 FINISHED run을 골라 배포 정보를 반환한다.
  train.py 파일명 규약으로 `model_path`를 재구성하고, provenance 파라미터
  (`feature_store_run_id`)를 그대로 실어 `config.inference`에 넣을 값을 만든다. §6 참고.

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

### 4-5. fold_id는 1부터 시작 (0 아님)
`src/data/splits.py::make_folds`가 `enumerate(test_blocks, start=1)`로 fold를 생성해
`fold=1`, `fold=2`, `fold=3` 파티션이 만들어진다. 초기 구현 때 `config.model.fold_id`
기본값을 `0`으로 잘못 넣어 `FileNotFoundError: fold=0/split=train/part.parquet`가
났다(RunPod 실전 실행 중 발견). 기본값을 `1`로 수정했다 — fold_id를 지정할 때는
항상 1부터 시작한다는 점에 주의.

---

## 5. 실행

### 로컬 (CPU, 빠른 스모크 테스트)
```bash
# 0) Feature Store가 비어 있다면 먼저 채운다 (docs/data_pipeline.md §6)
python -m src.data.collect
python -m src.data.build

# 1) 기본 config로 학습 (fold_id=1, total_timesteps=200000 — config.model 기본값)
python -m src.models.train

# 2) fold·timesteps 오버라이드
python -m src.models.train --fold-id 1 --total-timesteps 500000
```
학습 결과는 `config.model.model_dir`(기본 `mlruns/models/`)에 `.zip`으로, 실험 기록은
`config.model.mlflow_tracking_uri`(기본 `mlruns/`)에 남는다. `mlflow ui --backend-store-uri mlruns`
로 로컬에서 대시보드 확인 가능(mlflow 풀 패키지 별도 설치 필요 — mlflow-skinny엔 UI 서버 없음).

### RunPod

RunPod 파드 **자체가 이미 컨테이너**라 대부분의 템플릿에는 Docker 데몬이 없다 — 파드 안에서
`docker build`는 실패한다. 실제로는 두 경로 중 하나를 쓴다.

**방법 A — 파드에서 소스 직접 실행 (간단, 레지스트리 불필요)**
```bash
git clone https://github.com/TeamKimFour/RobuSTAM.git && cd RobuSTAM
pip install -r requirements.txt
python -m src.data.collect
python -m src.data.build
python -m src.models.train --fold-id 1 --total-timesteps 200000
```

**방법 B — 커스텀 이미지 (로컬에 Docker가 있을 때)**
```bash
# 1) 로컬에서 빌드 후 레지스트리에 push
docker build -t <user>/robustam-train -f docker/Dockerfile.train .
docker push <user>/robustam-train
# 2) RunPod에서 이 이미지로 파드를 생성 (파드 안에서 build하지 않는다)

# 로컬에 GPU가 있다면 직접 실행도 가능
docker run --gpus all \
  -v $PWD/data:/app/data \
  -v $PWD/mlruns:/app/mlruns \
  <user>/robustam-train --fold-id 1 --total-timesteps 1000000
```

> **fold_id는 1부터다.** `--fold-id 0`은 `FileNotFoundError: fold=0/split=train/part.parquet`로
> 실패한다(§4-5). 예전 문서 예시가 `0`으로 적혀 있어 실제로 이 오류를 한 번 겪었다.

**GPU가 필요한가 — 대개 아니다.** 정책망이 `MlpPolicy`(187차원 입력의 작은 MLP)라 SB3 공식
권장대로 **CPU가 오히려 빠르다**(GPU는 커널 실행·전송 오버헤드가 이득을 넘어선다). 또한
`requirements.txt`가 `torch==2.13.0`으로 핀돼 있어 PyTorch 템플릿의 사전설치 torch는 어차피
재설치된다. 비용을 아끼려면 **CPU 파드**를 권한다. GPU 파드를 쓰더라도 PyPI `torch` 휠이 CUDA
런타임을 포함하므로 별도 CUDA 베이스 이미지는 불필요하다.

RunPod 계정 생성·파드 기동·과금은 팀원이 직접 수행한다(자동화 범위 밖). **작업이 끝나면 파드를
Terminate**한다 — Stop 상태에서도 디스크 요금이 계속 나간다.

### 배포 후보 여러 개 만들기
`select.py`(§6)가 고를 후보를 만들려면 fold·seed를 바꿔 여러 번 학습한다.
```bash
for f in 1 2 3; do
  for s in 42 7; do
    python -m src.models.train --fold-id $f --seed $s --total-timesteps 200000
  done
done
```

---

## 6. 배포 policy 선택 → config.inference (src/models/select.py)

여러 fold·seed로 학습해 MLflow에 쌓인 run 중 **어떤 policy를 서빙할지**를 valid 지표로 고른다.
기본 지표는 `valid_sharpe`(위험조정수익) — 단순 누적수익 대신 위험을 반영해 고른다는 프로젝트
철학(CLAUDE.md §1)을 따른다. 여기 Sharpe는 후보 선별용 근사이고, 정식 성과검증은 찬휘 백테스트
엔진 몫이다.

```
# 학습을 여러 번 쌓은 뒤, valid_sharpe 최고 run을 골라 배포값 출력
python -m src.models.select                       # 전체 fold 대상
python -m src.models.select --fold-id 1           # 특정 fold만
python -m src.models.select --metric valid_total_log_return   # 지표 교체
```

출력된 `model_path`·`scaler_run_id`·`scaler_fold_id`·`model_version`을 `config.yaml`의
`inference:` 섹션에 복사하면 precompute가 그 policy를 로드하고, provenance(`scaler_run_id`)로
학습과 동일한 정규화를 재현한다(§4-5·이슈 #27). 선택된 run에 provenance가 없으면
(`feature_store_run_id`=`nofs`) 경고를 내보내며, 재학습을 권장한다.

> `select_best_run`은 MLflow 파라미터·지표만 읽고 아티팩트를 내려받지 않는다. `model_path`는
> train.py 파일명 규약(`ppo_fold<id>_<build_run_id>_<mlflow_run_id>.zip`)으로 결정적으로
> 재구성하므로, 공유 MLflow 서버(찬휘)로 옮겨도 파일명 규약만 유지되면 동작한다.

---

## 7. DQN MVP용 이산 환경 어댑터 (3주차, `src/env/discrete_env.py`)

3주차 마일스톤 "DQN MVP E2E 연결"을 위해 `PortfolioEnv`(연속 로짓 계약)를 이산 행동으로
감싸는 어댑터. DQN은 유한한 행동 집합을 요구하므로, 원본 env는 손대지 않고 `ActionWrapper`로
정수 action → 로짓 벡터를 변환한다. 5주차 PPO 전환 시 이 어댑터는 그대로 두고
학습 스크립트가 원본 env(연속 로짓)를 다시 쓴다.

### 7-1. 행동 설계 — 옵션 B "SHV와 교환하거나 유지"
| action | 의미 |
|---|---|
| 0..3 | 자산 i(SPY/EWY/TLT/GLD)에 +Δ (SHV에서 이전) |
| 4..7 | 자산 i에서 -Δ (SHV로 반환) |
| 8    | 유지 (현재 비중 그대로) |

총 `2·A_tr + 1 = 9`개 (A_tr = SHV 제외 자산 수). SHV(현금성)를 상대 계정으로 삼으면
이전량이 명확하고 ∑w=1이 자동 보존된다. 설계 A(자산별 3택 = 3⁵=243)나 프리셋 방식 대비
Q(s,a) 테이블이 좁아 DQN 학습이 안정적이다.

**경계 처리:** 실제 이전량 = `min(Δ, 여유분)`으로 clip → 음수 비중 방지. SHV=0에서 +Δ,
자산=0에서 -Δ는 no-op으로 동작하지만 유지·반대방향 action은 항상 유효.

### 7-2. 로짓 변환 — softmax 계약 유지
원본 env는 내부에서 `_softmax()`로 정규화한다. 계산한 `w_new`(합=1)를
`log(clip(w_new, EPS, 1))`로 넘기면 softmax가 `w_new`를 EPS 근사오차 내에서 복원한다.

```
softmax(log(w)) = exp(log(w)) / Σ exp(log(w)) = w / Σw = w   (Σw=1일 때)
```

EPS(=1e-8)로 잘린 0-비중 자산은 softmax 후 약 1e-8 크기로 남아 원 의도와 사실상 일치
(테스트 `test_weights_sum_to_1_after_arbitrary_sequence`로 검증).

### 7-3. 학습에서 사용하는 법 (도현 인터페이스)
```python
from src.env.portfolio_env import PortfolioEnv
from src.env.discrete_env import DiscretePortfolioEnv
from stable_baselines3 import DQN

env = DiscretePortfolioEnv(
    PortfolioEnv(state_df, returns_df, cfg=cfg),
    delta=0.1,   # 스텝당 SHV↔자산 이전 폭 (0, 1]
)
model = DQN("MlpPolicy", env, ...).learn(total_timesteps=...)
```

관측 shape·보상 계약은 원본 env를 그대로 통과하므로 `MlpPolicy` 입력 차원은 187 그대로.
`tests/test_discrete_env.py::test_sb3_dqn_learns_a_few_steps`가 SB3 DQN이 이 어댑터에
붙어 예외 없이 학습되는지(파이프라인 결선)를 검증한다 — 3주차 완료 기준
"환경-에이전트 연결 확인"에 해당.

### 7-4. 4주차 이후 유지 방침
DQN MVP는 파이프라인 결선용이며, 성능 검증은 5주차 PPO(연속 로짓) 전환 후 정식으로 한다.
어댑터는 5주차 이후에도 대안 실험(이산 vs 연속)이 필요할 수 있어 유지한다.

---

## 8. 아직 정해지지 않은 것

- **precompute 연동**: 학습된 `.zip`을 로드해 `docs/data_pipeline.md §8`의 latest.json을
  생성하는 스크립트 — `src/inference/precompute.py`로 구현 완료, 남은 건 §6 선택값을
  `config.inference`에 실제로 채우는 배포 단계.
- **SAC 비교**: `train()`의 `algorithm` 분기가 현재 PPO만 지원 — SAC 추가 시 액션 래퍼는
  그대로 재사용 가능(SB3 SAC도 유한 bound 요구는 동일).

---

## 변경 이력
| 버전 | 날짜 | 내용 |
|---|---|---|
| v0.1 | 2026-07-13 | 최초 작성. PPO 학습 스크립트·RunPod Dockerfile·mlflow-skinny 전환 근거 반영. |
| v0.2 | 2026-07-15 | 이슈 #27: build `run_id` provenance 기록(MLflow `feature_store_run_id` 파라미터·모델 파일명 `ppo_fold<id>_<build_run_id>_<mlflow_run_id>.zip`). 배포 시 `config.inference.scaler_run_id`로 복사. |
| v0.3 | 2026-07-18 | §7 추가: DQN MVP용 `DiscretePortfolioEnv` 어댑터(옵션 B, action_space=Discrete(9))·로짓 변환 계약·환경-에이전트 연결 스모크 테스트. |
| v0.4 | 2026-07-18 | §6 추가: 배포 policy 선택 로직(`src/models/select.py`) — MLflow에서 `valid_sharpe` 최고 run 선택 → `config.inference` 값 출력. `evaluate`에 `valid_sharpe`(연율화 위험조정수익 proxy) 기록. `train.py --seed` 인자 추가(배포 후보 확장용). §5 RunPod 실행 절차 정정(`--fold-id 0`→`1`, 파드 내 `docker build` 불가·CPU 권장 명시). 미정 항목은 §8로 이동. |
