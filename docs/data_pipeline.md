# RobuSTAM — 데이터 파이프라인 설계서 v0.1

> 데이터 엔지니어링 레이어(`src/data/`)의 구조·흐름·설계 결정 기록.
> State 명세는 [docs/state_spec.md](state_spec.md)가 SSOT이며, 본 문서는 그 명세를 산출하는
> 파이프라인을 설명한다. 두 문서가 충돌하면 state_spec.md가 우선한다.

---

## 1. 범위 (현재 구현 상태)

| 단계 | 모듈 | 상태 |
|---|---|---|
| 설정 단일 출처 | `config/config.yaml`, `src/config_loader.py` | ✅ 구현 |
| State 인덱스맵 | `src/data/schema.py` | ✅ 구현 |
| 원시 수집 | `src/data/collect.py` | ✅ 구현 |
| 로그수익률·윈도우 | `src/data/returns.py` | ✅ 구현 |
| 기술적 지표 6+2 | `src/data/features.py` | ⏳ 2주차 |
| walk-forward 분할 | `src/data/splits.py` | ⏳ 2주차 |
| z-score 정규화 | `src/data/normalize.py` | ⏳ 2주차 |
| 187차원 조립 | `src/data/assemble.py` | ⏳ 2주차 |
| Feature Store 출력 | `src/data/feature_store.py` | ⏳ 2주차 |

---

## 2. 데이터 흐름

```
yfinance ──collect.py──> data/raw/prices_raw.parquet   (조정종가, 공통 거래일 교집합)
                              │
                         returns.py
                      로그수익률 + W일 윈도우
                              │
              ┌───────────────┴─── (2주차) ───────────────┐
        features.py                                   splits.py
     지표 6+2 (최신값)                          walk-forward folds
              └───────────────┬──────────────────────────┘
                       normalize.py  (fold train에서만 fit)
                              │
                         assemble.py  (schema 인덱스맵으로 187 조립)
                              │
                      feature_store.py → data/feature_store/ (Parquet + SQLite)
```

---

## 3. 모듈별 책임

- **`config_loader.py`** — config.yaml을 읽는 단일 통로. `get_state_dim()`이 W에서 차원을
  산출(187 하드코딩 금지). 서드파티는 PyYAML 하나(함수 내부 import).
- **`schema.py`** — State 인덱스맵의 코드측 SSOT. `returns_slice / asset_feature_slice /
  market_feature_slice / prev_weight_slice / feature_names / index_map`. 순수 파이썬.
- **`collect.py`** — yfinance 조정종가 수집, 공통 거래일 교집합 정렬(`_align`), Parquet 캐시.
  `python -m src.data.collect`로 실행.
- **`returns.py`** — `log_returns`(첫 행 drop), `return_window(t, W)`(t 포함, 미래 미포함).

---

## 4. 설계 결정

### 4-1. walk-forward 분할 — Expanding 방식 (config에 기록, 구현 2주차)
test 블록은 국면별 OOS 검증(CLAUDE.md 목표)에 맞춰 2년 단위:

| Fold | train | test | 대표 국면 |
|---|---|---|---|
| 1 | 2010–2019 | 2020–2021 | COVID 폭락·회복 |
| 2 | 2010–2021 | 2022–2023 | 고금리·채권 급락 |
| 3 | 2010–2023 | 2024–2025 | 최근 |

- **Expanding(2010 고정, train 누적)** 채택: 10년 기반을 유지하면서 fold마다 최근 데이터를 반영하고
  정규화 통계를 재fit해 분포 드리프트를 막는다.
- 각 fold train 마지막 ~1년(`valid_days: 252`)을 validation으로 분리.
- train↔test 사이 `embargo_days: 34`(≥ 최장 지표 lookback) 갭으로 경계 누수 차단.

### 4-2. 데이터 기간 — 2009-10 ~ 2025-12
목표 시작(2010-01)보다 약 60거래일(`warmup_buffer_days`) 앞당겨 받는다. 2주차 지표(MACD ~34일
warm-up 등)가 NaN으로 잘려도 2010-01 목표 구간이 보존되도록.

---

## 5. 룩어헤드(Look-ahead Bias) 방지 — 3중 방어선

| 층 | 위치 | 방법 |
|---|---|---|
| ① 수집 | `collect._align` | ffill/bfill 금지, 공통 거래일 교집합만, 시작일 버퍼 선행 |
| ② 윈도우 | `returns.return_window` | 윈도우는 t 포함·**t+1 이후 미포함** (회귀 테스트로 고정) |
| ③ 정규화 | `normalize`(2주차) | z-score μ/σ를 fold train에서만 fit, valid/test는 적용만. embargo gap |

→ ②는 `tests/test_returns.py::test_return_window_excludes_future`로,
①은 `tests/test_collect.py`로 박제됨.

---

## 6. 실행·검증

```bash
# 1) 가상환경 (Python 3.12 — CI와 동일)
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2) 원시 데이터 수집 → data/raw/prices_raw.parquet
python -m src.data.collect

# 3) 테스트 (차원·인덱스맵·룩어헤드 회귀)
pytest tests/ -v
```

현재 수집 결과: **4087 거래일 × 5자산, 결측 0** (2009-10-01 ~ 2025-12-30).

---

## 변경 이력
| 버전 | 날짜 | 내용 |
|---|---|---|
| v0.1 | 2026-06-28 | 최초 작성. 기반(config·schema)·수집(collect·returns) 구현 반영. |
