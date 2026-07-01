# RobuSTAM — 백테스트 엔진 설계서 v0.1

> 백테스트 엔진 레이어(`src/backtest/`)의 구조·흐름·설계 결정 기록.
> 거래비용·자산 순서 등 핵심 규칙은 [CLAUDE.md §2](../CLAUDE.md)가 SSOT이며,
> 설정값은 [config/config.yaml](../config/config.yaml)에서만 읽는다.

---

## 1. 범위 (현재 구현 상태)

| 단계 | 모듈 | 상태 |
|---|---|---|
| NAV 갱신 및 수수료 차감 | `src/backtest/engine.py` | ✅ 구현 |
| 멀티 벤치마크 (1/N · 60:40 · B&H) | `src/backtest/benchmark.py` | ⏳ 2주차 |
| walk-forward 백테스트 루프 | `src/backtest/runner.py` | ⏳ 3주차 |
| 결과 저장 (DB 연동) | `src/backtest/engine.py` | ⏳ 도현과 협의 후 |

---

## 2. 데이터 흐름

```
AI 모델 (도현) → action (자산 비중 벡터, ∑wᵢ=1)
                          │
                     engine.py
              ┌───────────────────────┐
              │  1. 주가 변동 반영     │
              │  2. 수수료 차감        │
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

---

## 4. 핵심 수식

### NAV 갱신
```
nav_after_return = prev_nav × (1 + Σ(wᵢ × rᵢ))
```
- `wᵢ` : 전날 자산 비중
- `rᵢ` : 오늘 자산 수익률

### 수수료 차감
```
turnover = Σ|new_wᵢ - prev_wᵢ|
cost     = nav_after_return × turnover × c
new_nav  = nav_after_return - cost
```
- `c` : 편도 거래비용률 (`config.yaml`의 `transaction_cost`, 기본 0.1%)
- `turnover` : 비중 변화량의 합 → 많이 바꿀수록 수수료 증가 (과매매 억제)

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
| `config/config.yaml` | 공용 | `transaction_cost`, `assets` |

---

## 변경 이력

| 버전 | 날짜 | 내용 |
|---|---|---|
| v0.1 | 2026-07-01 | 최초 작성. NAV 갱신·수수료 차감 로직 구현 반영. |
