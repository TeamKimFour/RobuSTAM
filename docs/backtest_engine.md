# RobuSTAM — 백테스트 엔진 설계서 v0.1

> 백테스트 엔진 레이어(`src/backtest/`)의 구조·흐름·설계 결정 기록.
> 거래비용·자산 순서 등 핵심 규칙은 [CLAUDE.md §2](../CLAUDE.md)가 SSOT이며,
> 설정값은 [config/config.yaml](../config/config.yaml)에서만 읽는다.

---

## 1. 범위 (현재 구현 상태)

| 단계 | 모듈 | 상태 |
|---|---|---|
| NAV 갱신 및 수수료 차감 | `src/backtest/engine.py` | ✅ 구현 |
| 멀티 벤치마크 (1/N · 60:40 · B&H) | `src/backtest/benchmark.py` | ✅ 구현 |
| walk-forward 백테스트 루프 | `src/backtest/runner.py` | ⏳ 3주차 |
| 결과 저장 (S3) | `src/backtest/s3_results.py` | ✅ 구현 |
| 결과 저장 (DB 연동) | `src/backtest/engine.py` | ⏳ 도현과 협의 후 |

---

## 2. 데이터 흐름

```
AI 모델 (도현) → action (자산 비중 벡터, ∑wᵢ=1)
                          │
                     engine.py
              ┌───────────────────────┐
              │  1. 리밸런싱·수수료   │
              │     차감(start-of-day)│
              │  2. 새 비중으로 오늘   │
              │     주가 변동 반영     │
              │  3. 새 NAV 반환        │
              └───────────────────────┘
                          │
                   NAV, cost 반환
                          │
                  (추후) DB 저장 → execution_log
```

---

## 3. 모듈별 책임

- **`engine.py`** — `BacktestEngine` 클래스. `calc_nav()`로 하루치 NAV를 갱신하고 수수료를
  차감한다. 거래비용률은 `config.yaml`에서 읽는다(하드코딩 금지).
- **`benchmark.py`** — 1/N·60:40·B&H 세 벤치마크를 거래일 단위로 `calc_nav()`를 반복
  호출해 NAV 곡선(pd.DataFrame)으로 만든다. 자산 목록은 `config.yaml`에서 읽고,
  `price_returns`의 컬럼 순서가 이와 다르면 예외를 던진다(CLAUDE.md §2 자산 순서 고정).
- **`s3_results.py`** — 백테스트 실행 결과(NAV·성과지표·config 스냅샷)를 S3에 저장한다.
  `calc_nav()` 로직과 무관한 독립 단계이며, `BacktestEngine.save_results()`가 위임한다.
  `src/data/s3_sync.py`와 동일하게 자격증명은 boto3 기본 자격증명 체인에서 읽는다.

---

## 4. 핵심 수식

**리밸런싱 시점: start-of-day.** 매 스텝 시작에 전날 비중에서 새 비중으로 리밸런싱하고,
그 새 비중으로 당일 수익을 실현한다. (형우 Gym `env`와 관점 통일 — CLAUDE.md §7-2)

**완전 리밸런싱 가정(팀 회의 확정, env와 동일):** 기간 내 드리프트는 무시한다.
여러 날을 순회하는 호출자(walk-forward 루프 등)는 다음 스텝의 `prev_weights`로
이번 스텝의 `new_weights`를 드리프트 조정 없이 그대로 넘겨야 한다. 실제 장중
가격 변동으로 인한 비중 드리프트는 시뮬레이션하지 않는다.

### 수수료 차감 (리밸런싱 전 NAV 기준)
```
turnover      = Σ|new_wᵢ - prev_wᵢ|
cost          = prev_nav × turnover × c
nav_after_cost = prev_nav - cost
```
- `c` : 편도 거래비용률 (`config.yaml`의 `transaction_cost`, 기본 0.1%)
- `turnover` : 비중 변화량의 합 → 많이 바꿀수록 수수료 증가 (과매매 억제)

### NAV 갱신 (새 비중으로 당일 수익 실현)
```
new_nav = nav_after_cost × (1 + Σ(new_wᵢ × rᵢ))
```
- `new_wᵢ` : 리밸런싱 후(오늘) 자산 비중
- `rᵢ` : 오늘 자산 수익률

---

## 4-1. 벤치마크 전략 (`src/backtest/benchmark.py`)

세 전략 모두 `calc_nav()`를 거래일 단위로 반복 호출해 NAV 곡선을 만든다. 엔진 인터페이스
자체는 변경하지 않고, 전략별로 `prev_weights`/`new_weights`에 무엇을 넘길지만 다르게
구성한다. 초기 보유는 첫날 진입 전 **SHV(현금성) 100%**에서 시작한다(`PortfolioEnv.reset()`과
동일 관례).

### 1/N (동일비중) · 60:40 — 목표비중 고정
매일 `prev_weights = new_weights = 고정 목표비중`으로 호출한다. engine.py §4의 "여러 날을
순회하는 호출자는 다음 스텝의 prev_weights로 이번 스텝의 new_weights를 드리프트 조정 없이
그대로 넘겨야 한다"는 규약을 그대로 따른 것 — 목표비중 자체가 매일 동일하므로 첫날(초기
보유 SHV 100% → 목표비중 진입)에만 비용이 발생하고, 이후로는 `turnover=0`이다.

- **1/N**: 5개 자산에 각 1/5.
- **60:40**: 주식군(`SPY·EWY`) 60% : 안전자산군(`TLT·GLD·SHV`) 40%, **그룹 내부는 균등분배**
  (SPY 30%/EWY 30%, TLT·GLD·SHV 각 13.33%). 별도 근거(시가총액 등) 없이 임의 가중치를
  주는 것보다 "정직한 검증"(CLAUDE.md §1) 취지에 맞아 균등분배로 확정했다(회의 전 임시
  확정, 이견 있으면 재논의).

### B&H (매수후보유) — 리밸런싱 없음, 실제 드리프트 계산
"오늘 목표비중을 정하지 않는다"는 것 자체가 이 전략의 정의이므로, 매일
`new_weights = 오늘 실제 보유 비중`으로 두어 `prev_weights`와 동일하게 만들어
`turnover=0`/`cost=0`을 보장한다. 그 "실제 보유 비중"은 이 모듈이

```
w_i,t+1 = w_i,t · (1+r_i,t) / (1 + Σ w_i,t·r_i,t)
```

공식으로 직접 드리프트시켜 다음 날의 `new_weights`로 사용한다. engine.py에 넘기는
prev/new_weights를 "그대로 이어받는" 규약은 그대로 지키면서, 그 이어받는 값 자체를
B&H 정의에 맞게 매일 드리프트시키는 것이라 규약 위반이 아니다.

engine.py가 start-of-day 리밸런싱(새 비중이 그날 수익을 바로 실현)이므로, 첫날 진입한
초기 목표비중도 **진입 당일 수익률부터** 드리프트를 시작한다(둘째 날부터가 아니다).

### 자산 순서 검증
`price_returns`(pd.DataFrame)의 컬럼은 `config.assets` 순서와 정확히 일치해야 하며,
다르면 `ValueError`를 던진다(`PortfolioEnv`의 컬럼 검증과 동일한 관례).

---

## 4-2. 백테스트 결과 S3 저장 (`src/backtest/s3_results.py`)

`BacktestEngine.save_results(nav, metrics, run_id, ...)`를 호출하면 내부적으로
`save_results_to_s3()`에 위임한다. `calc_nav()`는 전혀 건드리지 않는, 완전히 독립된
저장 단계다.

### 저장 구조
```
s3://{버킷}/backtests/{실행날짜}_{실행ID}/
    nav.parquet    - NAV 시계열
    metrics.json   - Sharpe·MDD 등 성과 지표 (계산은 호출자 책임 — 이 모듈은 저장만 한다)
    config.yaml    - 이 실행에 쓰인 config 파일 원본 그대로 복사(바이트 단위 그대로)
```
`실행날짜`는 UTC 기준 `YYYYMMDD`(override 가능), `실행ID`는 생략 시 짧은 UUID를
자동 생성한다(MLflow run id 등 호출자가 원하는 값을 넘겨도 된다).

### 버킷 설정
`src/data/s3_sync.py`와 동일한 관례: **버킷을 새로 추가하지 않고 `config.data.s3_bucket`을
재사용**한다(환경변수 `S3_BUCKET`이 있으면 그것을 우선). 버킷이 비어 있으면(아직 미생성)
저장을 조용히 skip한다 — 로컬 개발 중엔 무해하다.

### 자격증명
AWS 자격증명은 코드에 절대 하드코딩하지 않는다. `.env.example`을 복사해 `.env`를 만들고
`AWS_ACCESS_KEY_ID`/`AWS_SECRET_ACCESS_KEY`/`AWS_DEFAULT_REGION`을 채우면
`python-dotenv`가 로드해 boto3 기본 자격증명 체인에 흘려보낸다. 배포 환경에서는 `.env` 없이
IAM Role을 쓰는 게 더 안전하다(이 경우 `.env` 값은 무시되고 Role 자격증명이 자동 적용).

---

## 5. 설계 결정

### 5-1. 거래비용률 — config.yaml 단일 출처
`transaction_cost`는 `config/config.yaml`에서만 읽는다. 코드에 하드코딩 금지.
현재 확정값: **편도 0.1%** (CLAUDE.md §2).

### 5-2. 슬리피지
슬리피지는 형우의 Gymnasium 환경(`src/env/`)에서 처리한다.
백테스트 엔진은 거래비용만 담당한다.

### 5-3. DB 연동
`execution_log` 테이블 스키마는 도현과 협의 후 확정 예정.
확정 전까지 엔진은 NAV·cost만 반환하고 저장은 하지 않는다.

---

## 6. 연관 모듈 인터페이스

| 모듈 | 담당 | 연관 내용 |
|---|---|---|
| `src/env/` | 형우 | Softmax 정규화, 슬리피지 반영 |
| `src/models/` | 도현 | action (자산 비중 벡터) 생성 |
| `src/api/` | 도현 | `execution_log` DB 스키마 협의 필요 |
| `config/config.yaml` | 공용 | `transaction_cost`, `assets`, `data.s3_bucket` |
| `src/data/s3_sync.py` | 민지 | 동일한 S3 버킷·자격증명 관례 공유(Feature Store 업로드) |

---

## 변경 이력

| 버전 | 날짜 | 내용 |
|---|---|---|
| v0.1 | 2026-07-01 | 최초 작성. NAV 갱신·수수료 차감 로직 구현 반영. |
| v0.2 | 2026-07-08 | 리밸런싱 시점을 end-of-day → start-of-day로 변경, env와 관점 통일. 완전 리밸런싱(드리프트 무시) 가정 명시. |
| v0.3 | 2026-07-11 | `benchmark.py`(1/N·60:40·B&H) 구현 반영. 60:40 그룹 내부 균등분배 확정, B&H 실제 드리프트 계산 로직 명시. |
| v0.4 | 2026-07-13 | `s3_results.py`(백테스트 결과 S3 저장) 구현 반영. `BacktestEngine.save_results()` 추가, 버킷은 `data.s3_bucket` 재사용. |
