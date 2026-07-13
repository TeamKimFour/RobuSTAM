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
| 기술적 지표 6+2 | `src/data/features.py` | ✅ 구현 |
| 187차원 조립 | `src/data/assemble.py` | ✅ 구현 |
| walk-forward 분할 | `src/data/splits.py` | ✅ 구현 |
| z-score 정규화 | `src/data/normalize.py` | ✅ 구현 |
| Feature Store 입출력 | `src/data/feature_store.py` | ✅ 구현 (scaler_stats·targets 포함) |
| 빌드 오케스트레이터 | `src/data/build.py` | ✅ 구현 (실데이터 적재) |
| Feature Store S3 업로드 | `src/data/s3_sync.py` | ✅ 구현 (⏳ AWS role·버킷 세팅 후 활성) |
| 파이프라인 컨테이너 | `docker/Dockerfile.pipeline` | ✅ 구현 |
| 매일 자동 빌드 (CI) | `.github/workflows/daily.yml` | ✅ 구현 |

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
- **`features.py`** — 자산 6지표 + 시장 2지표 계산(pandas-ta). 산식은 state_spec §3-1·config `features.params`.
  warm-up NaN은 통합 drop(메우기 금지). 컬럼명 = `feat_{asset}_{name}`/`mkt_{name}`.
- **`assemble.py`** — 수익률 윈도우 + 지표 + prev_weight(0)를 schema 슬라이스로 187 wide 조립.
  컬럼 = `feature_names(W)`, 시장지표는 자산별 복제 없이 단일 배치.
- **`splits.py`** — Expanding walk-forward `Fold` 생성(`make_folds`). anchor 고정·train 누적,
  test 블록 전진, train↔test 사이 embargo 거래일 갭.
- **`normalize.py`** — `ZScoreScaler`(fit/transform 분리). μ/σ는 **fold train에서만 fit**,
  valid/test는 적용만. 수익률 per_asset·지표 per_column·prev_weight 제외·std=0 가드. 통계 직렬화(`scaler_stats`).
- **`build.py`** — 오케스트레이터. 수집→지표→조립→(fold별 train fit→transform)→적재 + `targets`.
  실행 `python -m src.data.build`.
- **`s3_sync.py`** — 로컬 data(raw·feature_store)를 S3에 업로드(boto3). `meta.sqlite`도 파일로 업로드
  (s3:// 직접쓰기 불가 회피). 버킷 미설정 시 skip. 실행 `python -m src.data.s3_sync`.
- **`docker/Dockerfile.pipeline`** — collect·build 컨테이너 이미지(Python 3.12·핀 의존성).
- **`.github/workflows/daily.yml`** — 매일 KST 07:00 cron: collect→build→(AWS role 설정 시)S3 업로드.
  precompute(오늘의 비중, §8)는 모델 준비 후 추가. 실제 S3 연결은 찬휘 AWS 세팅(OIDC role·버킷) 후 활성.
- **`feature_store.py`** — 가공된 187차원 피처의 저장·조회. Parquet 파티션 입출력
  (`write_features / load_features / partition_path`) + SQLite 메타
  (`init_meta_db / write_run / write_feature_columns / write_fold / read_*`). I/O 골격 구현 완료,
  실데이터는 2주차 지표·정규화 후 채움.

---

## 3-1. 저장소(Feature Store) 설계

가공된 187차원 피처를 모델 학습·추론에 공급하기 위한 저장 계층. 두 부분으로 구성한다.

### Parquet — 피처 행렬 (wide 포맷)
```
data/feature_store/fold=<id>/split=<train|valid|test>/part.parquet
```
- **wide 채택**: State는 고정 187차원 밀집 행렬이라 long(녹는) 포맷은 행수 187배 + 조인 비용.
  wide(date 인덱스 + 187개 피처 컬럼)가 학습 로더에 직접 매핑되어 유리.
- 컬럼명은 `schema.feature_names(W)` 그대로 사용(`ret_SPY_lag0` … `prevw_SHV`) → 자기설명·차원 검증 용이.
- **fold/split 파티션**: walk-forward 학습의 자연 접근 단위("Fold1의 train만 로드").

### targets.parquet — 익일 수익률 (보상·백테스트용)
State 파티션과 같은 폴더에 `targets.parquet` 동반 저장. 컬럼 `fwd_ret_<asset>`, `fwd_ret[t]=logret[t+1]`,
State와 index 정렬(익일 없는 마지막 행 drop). **정규화 안 함**(실제 수익률 스케일). `load_targets`로 로드.
정규화된 State(관측)로는 보상을 계산할 수 없으므로, 형우 환경·찬휘 백테스트가 이 파일을 join해 사용한다.

### SQLite — 메타DB (`data/feature_store/meta.sqlite`)
데이터 산출물은 `.gitignore`(`/data/`)되므로, "어떤 config·통계로 만들어졌나"의 **감사 기록**이자
추론(도현) 재사용 근거. 테이블:

| 테이블 | 내용 | 채우는 시점 |
|---|---|---|
| `runs` | config 스냅샷(W·state_dim·자산순서·config_hash) | 1주차 |
| `feature_columns` | 187개 컬럼명·인덱스 (차원 감사) | 1주차 |
| `folds` | 각 fold train/valid/test 날짜 경계·embargo | ✅ (splits) |
| `scaler_stats` | 정규화 평균·표준편차 (추론 재사용) | ✅ (normalize) |

> 데이터 소스는 **yfinance 단일**이다. 초기 기획서의 KRX·FinRL은 저장소 문서(CLAUDE.md 스택·state_spec)에
> 없고, 고정 US ETF 유니버스 5종이 모두 yfinance로 커버되므로 채택하지 않는다.

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

# 3) Feature Store 빌드 → data/feature_store/fold=*/split=* (187 State + targets + 메타)
python -m src.data.build

# 4) 테스트 (차원·인덱스맵·룩어헤드 회귀)
pytest tests/ -v

# 4) 데이터 품질 검증 (EDA 요약 + 자동 검사)
python -m src.data.validate

# 5) (선택) EDA 노트북 — 표·차트로 직접 탐색
pip install -r requirements-dev.txt
jupyter lab notebooks/eda_raw_prices.ipynb
```

현재 수집 결과: **4087 거래일 × 5자산, 결측 0** (2009-10-01 ~ 2025-12-30).

---

## 7. 데이터 품질 검증 · EDA

`src/data/validate.py` — 두 가지를 제공한다.
- **`summarize_prices(df)`** — EDA 요약(통계·분포·상관·이상치). `notebooks/eda_raw_prices.ipynb`에서
  Plotly 차트(가격 추이·수익률 분포·상관 히트맵·롤링 변동성)와 함께 시각적으로 탐색.
- **`validate_prices(df, assets)`** — 자동 품질 게이트: 자산 컬럼·순서, 인덱스 정렬·중복, NaN,
  가격>0, 극단 일간변동, 최소 행수. 위반 목록 반환. `tests/test_validate.py`로 박제.

### 실데이터 관측 요약 (2009-10-01 ~ 2025-12-30, 4087일)
- **무결성**: 결측 0, 0이하 가격 0, 극단 이동(|logret|>0.25) 0, 분할 미조정 의심 0 → **모든 검증 통과**.
  (영업일 대비 갭 152는 16년치 미국 증시 공휴일로 정상.)
- **연율 변동성**: SPY 17%·EWY 25%·TLT 15%·GLD 16%·**SHV 0.3%**(현금성 확인).
- **분포**: 주식(SPY/EWY/GLD) 음의 왜도·높은 첨도(SPY 첨도 12) — 폭락 꼬리 위험, 상식과 일치.
- **상관**: **SPY↔TLT −0.30**(주식-채권 헤지), SPY↔EWY 0.72(주식군), **SHV 무상관**(현금) — 자산 구성 타당.

---

## 8. precompute — latest.json 계약 (민지 생성 → 도현 API 소비)

매일 배치로 PPO 정책이 계산한 **익일 추천 비중 1건**을 `config.yaml`의 `data.precompute_path`
(기본 `data/precompute/latest.json`)에 덮어쓴다. `src/api/main.py`의 `GET /inference/latest`가
매 요청마다 이 파일을 읽기만 하므로, 온라인 추론 없이도 CLAUDE.md §1의 평균 지연 200ms 이하를
구조적으로 만족한다.

### 스키마
```json
{
  "date": "2026-07-14",
  "generated_at": "2026-07-13T09:15:00+09:00",
  "model_version": "ppo_v1",
  "weights": {"SPY": 0.40, "EWY": 0.15, "TLT": 0.20, "GLD": 0.15, "SHV": 0.10}
}
```

| 필드 | 의미 |
|---|---|
| `date` | 이 비중이 적용되는 거래일(익일) |
| `generated_at` | precompute 배치 실행 시각 (ISO 8601) |
| `model_version` | 사용된 모델 버전 (`docs/db_schema.md`의 `model_versions.model_name`과 매칭 가능) |
| `weights` | 자산 티커 → 비중. **키 집합은 `config.yaml`의 `assets`와 정확히 일치해야 하고 값의 합은 1**(API가 요청마다 검증) |

### 계약 검증 (`src/api/main.py::get_latest_inference`)
- 파일이 없으면 **503**을 반환한다 — precompute가 아직 안 돌았다는 정상 상태이지 버그가 아니다(1주차 현재
  precompute는 미구현 상태).
- `weights`의 자산 키가 `config.yaml`과 다르거나 합이 1에서 벗어나면 **500**을 반환해 파이프라인 오류를
  조기에 드러낸다.

### 아직 정해지지 않은 것 (민지 precompute 구현 시 확정 필요)
- daily.yml에 precompute 스텝 추가 시점(모델 학습 파이프라인·`src/models/` 완성 후)
- 배치 실패 시 이전 `latest.json`을 유지할지, 파일을 지울지(현재 API는 파일이 있으면 무조건 최신으로 신뢰)

---

## 변경 이력
| 버전 | 날짜 | 내용 |
|---|---|---|
| v0.1 | 2026-06-28 | 최초 작성. 기반(config·schema)·수집(collect·returns) 구현 반영. |
| v0.2 | 2026-06-28 | Feature Store 저장소(§3-1), 데이터 품질 검증·EDA(§7) 추가. |
| v0.3 | 2026-07-02 | 지표 계산(features.py)·187 조립(assemble.py) 구현. state_spec §3-1 산식 반영. |
| v0.4 | 2026-07-03 | walk-forward 분할·z-score 정규화·build 오케스트레이터 구현. Feature Store 실데이터 적재 + targets. |
| v0.5 | 2026-07-09 | 배포 준비 반영: Docker 이미지·S3 업로드(s3_sync)·일일 워크플로(daily.yml). 실제 클라우드 연결 대기. |
| v0.6 | 2026-07-13 | precompute latest.json 계약(§8) 추가 — src/api/main.py 추론 엔드포인트 구현과 함께. |
