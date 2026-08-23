# RobuSTAM — 환경(Gymnasium) 파라미터 명세서 v1.0

> RL 환경(`src/env/`)의 하이퍼파라미터·인터페이스 계약·동작 규칙을 정리한 SSOT.
> **State(관측공간) 자체의 차원/인덱스 맵은 [docs/state_spec.md](state_spec.md)가 SSOT**이며 여기서 반복하지 않는다.
> 백테스트 엔진과 공유하는 회계 규약은 [docs/backtest_engine.md §4](backtest_engine.md)를 참고.
> 값의 출처는 언제나 [config/config.yaml](../config/config.yaml) — 코드에 절대 하드코딩하지 않는다.

---

## 1. 범위

이 문서가 다루는 것 / 다루지 않는 것.

| | 다룸 | 다루지 않음 (다른 SSOT) |
|---|---|---|
| **관측공간(State)** | env가 소비·검증하는 계약 | 차원 수식·인덱스 맵 → `state_spec.md` |
| **행동공간(Action)** | shape·bound·정규화 규칙 | RL 알고리즘·정책망 → `model_training.md` |
| **보상함수** | env가 위임하는 계약·입력 요구사항 | 수식·λ 튜닝 실험 → `docs/issue34_proposal.md` |
| **거래비용·리밸런싱 규약** | env의 소비 방식 | 백테스트 엔진 회계 → `backtest_engine.md` §4 |
| **하이퍼파라미터** | `config.yaml` 키 → env 소비처 매핑 | config 파일 자체 편집 → 이 문서 §5 참고 |

---

## 2. 모듈 구성

| 파일 | 클래스 | 역할 |
|---|---|---|
| [src/env/portfolio_env.py](../src/env/portfolio_env.py) | `PortfolioEnv` | 5자산 연속 행동 환경(PPO·SAC 대상). Softmax 정규화·거래비용·룩어헤드 방어. |
| [src/env/discrete_env.py](../src/env/discrete_env.py) | `DiscretePortfolioEnv` | DQN용 이산 행동 어댑터(9개 action). PortfolioEnv를 감쌈, 관측·보상 계약 보존. |
| [src/reward.py](../src/reward.py) | `calculate_reward_verbose`·`calculate_reward` | 순수 함수 보상 계산. env는 이 함수에 위임만 함(env↔reward 분리). |

`from src.env import PortfolioEnv` (패키지 진입점, [src/env/__init__.py](../src/env/__init__.py)).

---

## 3. `PortfolioEnv` — 연속 행동 환경

### 3-1. 생성자 계약

```python
env = PortfolioEnv(state_df, returns_df, cfg=None, cost_multiplier=1.0, vol_penalty_coef=0.0)
```

| 인자 | 타입 | 계약 |
|---|---|---|
| `state_df` | `pd.DataFrame` | 컬럼 = `schema.feature_names(W)`. 민지 [`src.data.assemble.assemble_state_matrix`](../src/data/assemble.py) 출력 그대로. |
| `returns_df` | `pd.DataFrame` | 컬럼 = `[fwd_ret_SPY, fwd_ret_EWY, fwd_ret_TLT, fwd_ret_GLD, fwd_ret_SHV]`. 민지 [`feature_store.load_targets`](../src/data/feature_store.py) 출력. **로그수익률**. |
| `cfg` | `dict \| None` | 생략 시 `config/config.yaml` 자동 로드. `assets`·`window`·`transaction_cost` 세 키 필수. |
| `cost_multiplier` | `float` | 학습 보상에만 거는 거래비용 가중치 λ (이슈 #34 개선안 A). 기본 `1.0`(셰이핑 없음). 보상용 `c = c_real·λ`로만 쓰고 실비용 `c_real`은 보존한다. **train split env에만** 넘긴다 — 평가·백테스트는 1.0(성과 측정 축 보존). 0 이하면 `ValueError`. |
| `vol_penalty_coef` | `float` | 위험조정 페널티 계수 κ (이슈 #34 개선안 D). 기본 `0.0`(페널티 없음). `R`에서 `κ·σ_recent`를 뺀다. σ_recent 윈도우는 `model.vol_penalty_window`(기본 20). **train split env에만** 넘긴다. 음수면 `ValueError`. |

**입력 검증 (모두 `ValueError`)**
1. `state_df` 컬럼 순서가 `schema.feature_names(W)`와 정확히 일치하지 않음
2. `returns_df` 컬럼이 `fwd_ret_<asset>` 순서와 다름
3. `state_df.index`와 `returns_df.index`가 다름
4. `len(state_df) < 2` (reset + 최소 1 step)

이 네 개는 **학습 초반에 조기 실패**시켜 조용한 misalignment 사고를 막는다.

### 3-2. 공간(space) 정의

| 요소 | 정의 | 근거 |
|---|---|---|
| **관측공간** | `Box(low=-inf, high=+inf, shape=(state_dim,), dtype=float32)` | `state_dim`은 `config_loader.get_state_dim(cfg)`로 산출(W=30 → 187). 하드코딩 금지. |
| **행동공간** | `Box(low=-inf, high=+inf, shape=(n_assets,), dtype=float32)` | env는 **로짓**을 받는다. 실제 비중은 내부 Softmax(§3-4). SB3용 유한 bound는 `train.py`가 `action_bound=10.0`으로 래핑(§5). |

**왜 shape이 `(A,)` 이고 `(A-1,)`이 아닌가?** Softmax가 자유도 1을 자동으로 처리하므로 학습 안정성만 놓고 정책이 대칭 표현을 자유롭게 쓸 수 있게 A차원 로짓을 그대로 받는다.

### 3-3. `reset()` — 초기 상태

- `self._t = 0`
- **초기 비중 = SHV 100%** — 팀 회의 확정 옵션 a. 무위험 자산에서 출발해 어떤 정책이 진입 비용을 어떻게 부담하는지 정직하게 반영. 초기값을 균등분배(1/N)로 하면 무료 진입이 되어 정책이 첫날 회전율을 인식 못 하는 사고 방지.
- 반환: `(_obs(), {})` — 표준 Gymnasium 시그니처.

### 3-4. `step(action)` — 5단계 흐름

```
action(로짓)                       returns_df
    │                                   │
    ▼                                   ▼
① Softmax → w_new(∑=1)          ② r_log = returns_df.iloc[t]
                                       │
                                       ▼
                              ③ r_arith = exp(r_log) − 1   (선택지 α)
    │                                   │
    └──────────┬────────────────────────┘
               ▼
④ src.reward.calculate_reward_verbose(w_prev, w_new, r_arith,
                                      c=c_real·λ, κ, σ_recent)   # λ·κ는 학습 전용
               │
               ▼
⑤ _current_weight = w_new;  _t += 1;  terminated = (_t >= len(state_df) − 1)
```

각 단계 세부:

- **① Softmax**: `_softmax(logits) = exp(z − max(z)) / Σ exp(z − max(z))`. max 감산으로 수치안정. ∑w=1은 이 함수로 항상 성립하므로 정책이 어떤 로짓을 뱉든 계약이 깨지지 않는다.
- **② `r_log`**: **`returns_df.iloc[self._t]`를 그대로** 조회. `fwd_ret[t] = log_ret[t+1]` 규약(민지 targets 저장 단계에서 이미 shift)이라 인덱스를 앞으로 밀지 **않는다** — 밀면 오히려 룩어헤드가 발생(과거 시점에서 미래 값을 이미 알고 있는 셈). 회귀 테스트 `test_env_robustness::test_each_step_uses_fwd_ret_at_current_index_only`가 이 규약을 박제.
- **③ 로그→산술 변환 (선택지 α, 팀 회의 확정)**: 민지 targets는 로그수익률로 저장, 도현 `calculate_reward`는 산술수익률을 기대. 이 어댑터 책임을 env가 진다: `r_arith = np.expm1(r_log)`. 백테스트 엔진(`src/backtest/policy.py`)도 동일한 `expm1` 변환을 하므로 두 경로가 정확히 같은 비중·수익을 계산한다.
- **④ 보상**: `src.reward.calculate_reward_verbose` 순수함수에 위임(`transaction_cost_rate=c=c_real·λ`). env는 값 검증(shape·∑w≈1)을 이 함수에 위임하고 반환 dict(`portfolio_return`·`log_return`·`turnover`·`transaction_cost`·`vol_penalty`·`reward`)를 info로 흘려준다. **단 `info["cost"]`는 그대로가 아니라 실비용으로 재계산**한다: `cost = turnover × c_real`(λ 무관, 진단·집계가 실제 지출을 보게 함). 보상에 반영된 셰이핑 비용은 `shaped_cost`(=`transaction_cost`, λ 반영)로, 최근 변동성은 `recent_vol`(σ_recent)로 별도 노출한다. λ=1·κ=0이면 기존 동작과 동일.
- **④-1 위험조정 셰이핑 (κ, 개선안 D)**: `R = log_ret − λ·c·turnover − κ·σ_recent`의 κ 항. **κ**(`vol_penalty_coef`)는 최근 실현 변동성 σ_recent에 페널티를 건다. σ_recent는 env가 롤링 윈도우(`model.vol_penalty_window`)로 **직전 스텝까지의** 포트폴리오 로그수익률만 모아 계산한다(당일 수익률 포함 시 룩어헤드). κ=0이면 페널티 없음(기존 동작). λ와 마찬가지로 학습 전용이라 평가·백테스트는 받지 않아 측정 축이 유지된다. 상세: [`docs/issue34_proposal.md`](issue34_proposal.md) 개선안 A·D.
- **⑤ 상태 업데이트**: 완전 리밸런싱 가정(기간 내 드리프트 무시, 백테스트 엔진과 동일 규약, `backtest_engine.md` §4).

### 3-5. `terminated` 판정

```python
terminated = self._t >= len(self._state) - 1
```

마지막 행에서 `step()` 호출 후 종료. 마지막 행의 `fwd_ret`는 그 다음 날 수익률(없음)이 아니라 마지막 관측일의 익일 수익률(존재)이므로 마지막 스텝도 유효한 학습 신호. 총 스텝 수 = `len(state_df) - 1`.

**`truncated`는 항상 `False`** — 시간초과 등 외부 종료 조건은 학습 루프(SB3 `TimeLimit` 래퍼 등) 몫으로 넘긴다.

### 3-6. `info` dict 키 (`step()` 반환값)

| 키 | 타입 | 의미 |
|---|---|---|
| `portfolio_return` | `float` | 산술 수익률 `Σ w_new · r_arith` (거래비용 차감 전) |
| `log_return` | `float` | `ln(1 + portfolio_return)` (거래비용 차감 전) |
| `turnover` | `float` | `Σ|w_new − w_prev|` (L1 회전율) |
| `cost` | `float` | **실비용** `c_real × turnover` (λ 무관 — 진단·집계가 실제 지출을 보게 함) |
| `shaped_cost` | `float` | 보상에 반영된 셰이핑 비용 `c_real·λ × turnover` (λ=1이면 `cost`와 같음) |
| `recent_vol` | `float` | σ_recent — 직전 스텝까지 실현 로그수익률의 표준편차(개선안 D). 표본<2면 0.0 |
| `vol_penalty` | `float` | `κ × recent_vol` (κ=0이면 0) |
| `weights` | `np.ndarray(A,)` | 오늘 확정된 목표 비중(Softmax 결과, ∑=1). float32 사본. |

MLflow 스텝 로깅·과매매 진단이 이 dict에서 읽는다.

### 3-7. `_obs()` — 관측 조립

State DataFrame의 현재 행을 float32로 복사한 뒤, `state_spec.md §3`의 **`[182:187]` prev_weight 블록**을 **런타임에 `_current_weight`로 덮어쓴다**. 이유:

- Feature Store가 적재하는 State에서 prev_weight 5칸은 **0**으로 고정(민지 `assemble`이 채움).
- env가 매 스텝 실제 직전 비중을 이 자리에 주입해야 정책이 "직전 비중을 아는" 상태로 판단.
- 백테스트 어댑터(`src/backtest/policy.py` §③)도 동일 규약을 지켜야 학습 관측과 배포 관측이 동일해진다.

`schema.prev_weight_slice(W)`가 이 슬라이스 위치를 반환.

---

## 4. `DiscretePortfolioEnv` — DQN 어댑터

`gymnasium.ActionWrapper` 패턴. 원본 env는 손대지 않고 action만 이산 정수 → 로짓으로 변환.

### 4-1. 행동 설계 (옵션 B, 3주차 확정)

| action 인덱스 | 의미 |
|---|---|
| `0..A_tr-1` (0~3) | 자산 `i`에 **+Δ** (SHV에서 뺌) |
| `A_tr..2·A_tr-1` (4~7) | 자산 `i`에서 **−Δ** (SHV로 반환) |
| `2·A_tr` (8) | **유지** (현재 비중 그대로) |

- 조정 대상 자산 = SHV를 제외한 `[SPY, EWY, TLT, GLD]` (A_tr = 4)
- 총 action 수 = `2·A_tr + 1 = 9`
- **SHV(현금성)를 상대 계정으로 삼는다** → ∑w=1이 자동 보존, 이전량 정의가 명확.
- 왜 옵션 A(3^5=243)를 안 썼나: 학습 안정성. Q(s,a) 테이블이 좁을수록 DQN이 잘 수렴하고, 데모 목적엔 9개면 충분.

### 4-2. 경계 처리 (clipping)

```
실제 이전량 = min(Δ, 여유분)
```

- `w_prev[SHV]=0.05`일 때 +0.1 action → 실제 +0.05만 이동. 음수 비중 방지.
- `w_prev[SHV]=0` 또는 `w_prev[asset]=0`에서의 해당 방향 action은 no-op. 유지·반대방향 action은 항상 살아 있어 에피소드가 막히지 않음.

### 4-3. 로짓 변환 (Softmax 계약 유지)

```
softmax(log(w_new)) = w_new     (∑w_new=1일 때 정확)
```

`log(clip(w_new, EPS=1e-8, 1))`로 변환해 원본 env에 넘긴다. 0-비중 자산은 Softmax 후 ~1e-8 크기로 남아 실질적으로 0. `test_env_robustness::test_discrete_*`가 이 근사 오차가 1e-6 이내임을 박제.

### 4-4. 파라미터

| 이름 | 기본값 | 유효 범위 | 영향 |
|---|---|---|---|
| `delta` | `0.1` | `(0.0, 1.0]` | 스텝당 최대 이전량. 크면 학습 초기 탐험 폭이 크지만 과매매 유도. 작으면 벤치마크 근처에 오래 머무름. DQN MVP엔 0.1 사용. |

**Δ의 단일 출처는 `config.model.action_delta`다** (`config_loader.get_action_delta`, 기본 0.1).
학습(`train.load_fold_env`)·백테스트(`backtest.policy.run_policy`)·추론(`inference.precompute`)이
모두 이 값을 읽는다 — 세 경로가 다른 Δ를 쓰면 백테스트가 학습과 어긋난 비중 궤적을 낸다.

**turnover 상한**: 한 스텝에 SHV↔자산 사이로 최대 Δ만 옮기므로 회전율(L1)은 **2Δ 이하**로
구조적으로 묶인다(콜드스타트 첫 배분 제외). 이것이 이슈 #34 개선안 B의 근거이며
`tests/test_discrete_training.py::test_backtest_turnover_is_structurally_capped_by_delta`가 박제한다.

### 4-5. 세 경로가 공유하는 이전 규칙

Δ 이전 계산은 순수 함수 `discrete_action_to_logits(w_prev, action, *, shv_idx, tradable_idx, delta)`
하나로 모았다(`src/env/discrete_env.py`). 어댑터·백테스트·추론이 각자 구현하면 조용히 갈라지므로,
어댑터의 `action()`도 이 함수를 호출한다. 두 구현이 갈라지면
`tests/test_discrete_training.py::test_pure_function_matches_wrapper_action`에서 잡힌다.

---

## 5. 파라미터 표 — `config.yaml` → env 소비처

env가 실제로 읽는 config 키만 정리(전체 스키마는 [config/config.yaml](../config/config.yaml)이 원본).

| config 키 | 소비 함수 | env 소비처 | 기본값 | 변경 시 영향 |
|---|---|---|---|---|
| `assets` | `get_assets` | 자산 순서·이름·개수 A. `DiscretePortfolioEnv`는 `"SHV"` 존재 필수. | `[SPY, EWY, TLT, GLD, SHV]` | **변경 금지** (CLAUDE.md §2). SHV 제거 시 discrete env 생성 실패. |
| `window` | `get_window` | 관측 shape 산출용 W (Softmax·step 로직엔 W가 직접 안 나타남; state_df 컬럼 검증에만 사용) | `30` | state_dim 재계산 필요(Feature Store 재빌드). W=20→137, 60→337. |
| `transaction_cost` | `get_transaction_cost` | reward의 `c`. `cost = c × turnover`. | `0.001` (편도 0.1%) | 정책이 회전율에 얼마나 벌 받는지 결정. CLAUDE.md §2 확정. 민감도는 6주 실험 대상. |
| `model.algorithm` | `models.loader.get_algorithm` | `PPO`면 `_BoundedActionWrapper`, `DQN`이면 `DiscretePortfolioEnv`로 감싼다(§4). 백테스트·추론의 행동 해석도 같은 키로 갈린다. | `"PPO"` | DQN 전환 시 행동공간이 이산 9개가 되고 회전율이 2Δ로 제한된다(이슈 #34 개선안 B). |
| `model.action_delta` | `config_loader.get_action_delta` | `DiscretePortfolioEnv(delta=…)` — 스텝당 SHV↔자산 이전폭 Δ(§4-4). PPO 경로는 읽지 않는다. | `0.1` | 회전율 상한 2Δ가 바뀐다. 범위 `(0, 1]` 밖이면 로드 시점에 에러. |
| `model.action_bound` | `train.py`가 소비 | env 자체엔 무해(action bound는 SB3 래핑에서만 사용) | `10.0` | 정책 로짓 범위. 너무 작으면 Softmax가 균등에 가까워져 학습 신호 약화, 너무 크면 극단 비중 진입해 회전율 폭발. |
| `model.train_cost_multiplier` | `train.py`가 소비 | env 생성자 `cost_multiplier`(λ) 인자로 전달. 보상용 `c = c_real·λ`. **train split env에만** 넘긴다. | `1.0` | λ (개선안 A). 회전율 페널티 강도. 평가·백테스트는 항상 1.0(성과 측정 축 보존). 0 이하면 `ValueError`. |
| `model.vol_penalty_coef` | `train.py`가 소비 | env 생성자 `vol_penalty_coef`(κ) 인자로 전달. **train split env에만** 넘긴다. 음수 거부. | `0.0` | κ (개선안 D). σ_recent 페널티 강도. 0=페널티 없음. 과하면 λ와 같은 붕괴(1/N 흉내) — 비중편차 감시. |
| `model.vol_penalty_window` | env가 직접 소비 | σ_recent 롤링 윈도우 길이(스텝). `deque(maxlen=W_vol)`. 2 미만이면 `ValueError`. | `20` | κ 페널티의 변동성 측정 구간. 짧으면 노이즈에 민감, 길면 반응 느림. κ=0이면 무의미. |
| `model.seed` | `train.py`가 소비 | env는 `super().reset(seed=seed)`로 numpy RNG 시드 설정. 현재 env 자체는 randomness 없음(deterministic). | `42` | 재현성. env 결과에 영향 없음. |

env가 소비하지 **않는** 키(주의): `data.*` · `features.*` · `split.*` · `normalize.*`는 **Feature Store 빌드 시점**에만 소비되고 env는 이미 정규화·조립된 결과(`state_df`)만 받는다. env에서 이 값을 다시 읽지 말 것.

---

## 6. 순수 보상 함수 (`src.reward`)

env가 위임하는 계약. 이 절은 env가 지켜야 하는 입력 규약을 정리한다(수식 자체는 [`src/reward.py`](../src/reward.py) 상단 docstring).

### 6-1. 입력 계약

| 인자 | shape | 단위 | 검증 |
|---|---|---|---|
| `prev_weights` | `(A,)` | 비율(합=1) | `np.isclose(sum, 1.0, atol=1e-3)` — 어긋나면 `ValueError` |
| `new_weights` | `(A,)` | 비율(합=1) | 동일 |
| `asset_returns` | `(A,)` | **산술수익률** (예: 1% = 0.01) | shape만 일치 확인 |
| `transaction_cost_rate` | scalar | 소수(0.001 = 0.1%) | 기본값 `DEFAULT_TRANSACTION_COST_RATE=0.001` |
| `vol_penalty_coef` | scalar | κ (개선안 D) | 기본 `0.0`. 음수면 `ValueError` |
| `recent_vol` | scalar | σ_recent (env가 롤링 계산해 전달) | 기본 `0.0`. per-step 함수라 스칼라로 받아 κ만 곱함 |

세 배열의 shape이 서로 다르면 `ValueError`. 세 함수 사용처가 정확히 같은 A를 쓴다는 계약(env·백테스트 엔진 모두 `config.assets` 길이). `vol_penalty_coef`·`recent_vol`은 스칼라 셰이핑 인자로, 기본값이면 보상식이 기존과 동일하다.

### 6-2. verbose vs. scalar 래퍼

- `calculate_reward_verbose(...)` — dict 반환. env가 사용(중간값을 info로 노출).
- `calculate_reward(...)` — verbose를 감싸 `reward` 스칼라만 반환. 순수 스칼라 필요한 호출자용 얇은 래퍼.

### 6-3. 안정성 가드

- `_MIN_PORTFOLIO_RETURN = -0.999999` — `log(1 + r)` 정의역 보호. 이론상 `r ≤ -1`인 스텝(예: 극단 가상 시나리오)에서 log가 폭발하지 않도록 클립.
- 실 데이터로는 도달할 수 없는 방어선(하루 -100% 이하 수익률은 실제 시장에 없음). 방어 코드지 결과에 영향은 없음.

---

## 7. 계약 요약표

env를 사용하는 모든 코드가 지켜야 하는 5개 불변식.

| # | 불변식 | 어긋나면 |
|---|---|---|
| 1 | 자산 순서 = `[SPY, EWY, TLT, GLD, SHV]` (CLAUDE.md §2) | `_shv_idx` 오탐지, discrete env 실패 |
| 2 | `state_df.columns == schema.feature_names(W)` | 생성자 `ValueError` |
| 3 | `returns_df.columns == [fwd_ret_<asset>]` | 생성자 `ValueError` |
| 4 | `returns_df.iloc[t]` = 로그수익률, `= log_ret[t+1]` (fwd 규약) | 룩어헤드 발생 또는 한 칸 밀림 |
| 5 | ∑(Softmax(action)) == 1 (자동 보장) | 정책이 어떤 로짓을 뱉든 항상 성립 |

**회귀 테스트**: 총 62 케이스, CI 필수.
- [tests/test_env.py](../tests/test_env.py) — `PortfolioEnv` 계약 준수 21케이스
- [tests/test_env_robustness.py](../tests/test_env_robustness.py) — 과매매·룩어헤드 견고성 12케이스
- [tests/test_discrete_env.py](../tests/test_discrete_env.py) — 이산 어댑터 13케이스
- [tests/test_reward.py](../tests/test_reward.py) — 보상 수식 16케이스

---

## 8. 잘못 쓰기 쉬운 함정

과거에 실제로 발견되어 회귀 테스트로 박제된 사고 목록.

- **"한 칸 밀림"** — `returns_df.iloc[t+1]`을 조회하면 룩어헤드 발생. 규약상 `iloc[t]` 자체가 이미 next-day 값. (fix 커밋 `090b573`, `test_each_step_uses_fwd_ret_at_current_index_only`가 재발 방지)
- **로그↔산술 혼용** — reward에 로그수익률을 그대로 넘기면 계산이 이중 로그 처리로 왜곡. env가 `expm1` 하는 이유. 백테스트 엔진(`src/backtest/policy.py`)도 동일 변환 필수.
- **prev_weight 안 채움** — Feature Store의 prev_weight 5칸은 0으로 저장돼 있다. env가 매 스텝 `_current_weight`로 덮어쓰지 않으면 정책이 항상 "직전 비중=0"이라고 오해. 이 규약은 `backtest/policy.py`에도 그대로 적용됨.
- **action bound 없이 SB3에 넘기기** — env는 `low=-inf, high=+inf`지만 SB3는 유한 bound가 필요. `train.py`가 `action_bound=10.0`으로 감싸 사용. 이 값 없이 학습 시작하면 크래시.
- **Softmax를 밖에서 또 함** — env가 이미 Softmax를 하므로 정책망이 Softmax 출력층을 두면 이중 Softmax가 되어 분포가 과도하게 평탄해짐. **정책망은 로짓 그대로 뱉을 것**.

---

## 9. 연관 문서·모듈

| 항목 | 위치 |
|---|---|
| State 인덱스 맵·차원 수식 | [docs/state_spec.md](state_spec.md) |
| 데이터 파이프라인·Feature Store 계약 | [docs/data_pipeline.md](data_pipeline.md) |
| 백테스트 엔진 회계 규약(리밸런싱·거래비용 공유) | [docs/backtest_engine.md](backtest_engine.md) §4 |
| RL 학습·PPO 튜닝 | [docs/model_training.md](model_training.md) |
| 보상 λ 실험·과매매 진단 | `docs/issue34_proposal.md`, `docs/model_diagnosis.md` |
| 핵심 규칙 SSOT | [CLAUDE.md](../CLAUDE.md) §2 |
| config 파일 | [config/config.yaml](../config/config.yaml) |

---

## 변경 이력

| 버전 | 날짜 | 내용 |
|---|---|---|
| v1.0 | 2026-07-27 | 최초 작성. 5주 정식 항목("환경 파라미터 정리"). PortfolioEnv·DiscretePortfolioEnv·reward 3개 모듈의 계약·하이퍼파라미터·불변식·함정을 SSOT로 정리. state_spec.md(State 차원)와 backtest_engine.md(회계 규약)를 참조하고 중복 서술은 피했다. |
| v1.1 | 2026-07-29 | 학습 전용 거래비용 셰이핑 λ(이슈 #34 개선안 A) 반영: §3-1 생성자 표에 `cost_multiplier` 행·시그니처, §4 "④ 보상"에 `info["cost"]` 실비용 재계산·`shaped_cost` 추가, §5 표에 `model.train_cost_multiplier` 행. 보상식 `R = log_ret − λ·c·turnover`. λ=1에서 기존 동작 불변. |
| v1.2 | 2026-07-30 | 위험조정 보상 κ(이슈 #34 개선안 D) 반영: §3-1에 `vol_penalty_coef` 행·시그니처, §4에 `vol_penalty`·`recent_vol`·④-1(κ 셰이핑) 추가, §5 표에 `model.vol_penalty_coef`·`vol_penalty_window` 행, §6 입력 계약에 `vol_penalty_coef`·`recent_vol`. 보상식 `R = log_ret − λ·c·turnover − κ·σ_recent`. κ=0에서 기존 동작 불변. |
| v1.3 | 2026-08-22 | 이산 행동 학습 경로(이슈 #34 개선안 B) 반영: §4-4에 Δ 단일 출처(`model.action_delta`)·turnover 2Δ 상한, §4-5 공유 순수함수 `discrete_action_to_logits` 신설, §5 표에 `model.algorithm`·`model.action_delta` 행. `algorithm: PPO` 기본값에서 기존 동작 불변. |
