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
| 원시 수집 | `src/data/collect.py` | ✅ 구현 (재시도·품질 게이트 §7-1) |
| 로그수익률·윈도우 | `src/data/returns.py` | ✅ 구현 |
| 기술적 지표 6+2 | `src/data/features.py` | ✅ 구현 |
| 187차원 조립 | `src/data/assemble.py` | ✅ 구현 |
| walk-forward 분할 | `src/data/splits.py` | ✅ 구현 |
| z-score 정규화 | `src/data/normalize.py` | ✅ 구현 |
| Feature Store 입출력 | `src/data/feature_store.py` | ✅ 구현 (scaler_stats·targets 포함) |
| 빌드 오케스트레이터 | `src/data/build.py` | ✅ 구현 (실데이터 적재) |
| Feature Store S3 업로드 | `src/data/s3_sync.py` | ✅ **활성** (2026-07-19 실연결 검증, §7-2) |
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
- **`.github/workflows/daily.yml`** — 매일 KST 07:00 cron: collect→build→**S3 업로드(활성)**.
  precompute(오늘의 비중, §8)는 모델 준비 후 추가. S3 연결은 2026-07-19 실연결 검증 완료(§7-2).
- **`feature_store.py`** — 가공된 187차원 피처의 저장·조회. Parquet 파티션 입출력
  (`write_features / load_features / partition_path`) + SQLite 메타
  (`init_meta_db / write_run / write_feature_columns / write_fold / read_*`). I/O 골격 구현 완료,
  실데이터는 2주차 지표·정규화 후 채움.
- **`inference/precompute.py`** — 오늘 State→익일 비중(`latest.json`) 생산자(§8). `build_today_obs`
  (민지: 오늘 정규화 187 obs) + `generate_latest`(도현: predict→softmax→원자적 기록). `config.inference`가
  정규화 통계 출처를 지정. 실행 `python -m src.inference.precompute`.

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

### 7-1. 수집 신뢰성 — 재시도 · 품질 게이트 · 캐시 보호

**배경(2026-07-17·18 daily.yml 장애):** Yahoo가 CI 러너 IP를 간헐적으로 레이트리밋해
`yfinance`가 **0행**을 반환했다(`수집 완료: 0일 × 5자산`, 기간 `NaT ~ NaT`). 당시 `collect`는
이를 **"성공"으로 처리**했고, 빈 데이터가 흘러가 지표 계산에서 원인과 동떨어진
`TypeError: NoneType - NoneType`(pandas-ta `sma`가 None 반환)으로 터졌다. 더 위험하게는,
build가 통과했다면 `s3_sync`가 **빈 데이터로 팀 공용 S3 스냅샷을 덮어썼을** 것이다.

**대응 — `collect.fetch_prices`가 두 겹으로 막는다:**

| 층 | 동작 |
|---|---|
| ① 재시도 | `yf.download`를 지수 백오프로 재시도(기본 3회, 2→4→8초). 빈 응답·예외 모두 재시도 대상 — transient 레이트리밋을 흡수 |
| ② 품질 게이트 | `validate_prices(prices, assets, min_rows)` 호출. 위반 시 **즉시 `ValueError`** (조용한 성공 금지) |
| ③ 캐시 보호 | **게이트를 통과했을 때만** Parquet 기록 → 나쁜 데이터가 **좋은 캐시를 덮어쓰지 못한다** |

- 기준 행수는 config `data.min_rows`(기본 1000). 전체 기간 기대치는 약 4,087행.
- 2차 방어선으로 `features._asset_feature`가 pandas-ta의 `None` 반환을 감지해
  **자산·지표·입력 행수를 담은 `ValueError`** 로 바꾼다(원인 즉시 파악).
- `tests/test_collect.py`가 재시도 회복·빈 응답 실패·행수 미달·**캐시 미덮어씀**을 박제한다.

### 7-2. S3 실연결 검증 기록 (2026-07-19)

`daily.yml`을 `dev`에서 수동 트리거해 **collect → build → OIDC 인증 → S3 업로드 전 스텝 통과**를
확인했다. 이전까지는 빌드 단계에서 끊겨 AWS 스텝에 도달한 적이 없어 미검증 상태였다.

| 스텝 | 결과 |
|---|---|
| 원시 수집 | 4087일 × 5자산 (2009-10-01 ~ 2025-12-30) |
| Feature Store 빌드 | `state_all (4027, 187)`, folds 3 |
| AWS 자격증명(OIDC) | ✅ `vars.AWS_ROLE_ARN`으로 AssumeRole 성공 |
| S3 업로드 | ✅ 총 20개 파일 |

업로드 결과(버킷 직접 조회로 실물 확인):
```
s3://robustam-teamkimfour/robustam/raw/prices_raw.parquet            160KB
s3://robustam-teamkimfour/robustam/feature_store/fold={1,2,3}/split={train,valid,test}/
    part.parquet + targets.parquet                                    18개
s3://robustam-teamkimfour/robustam/feature_store/meta.sqlite         131KB
```
fold별 train이 3.6 → 4.5 → 5.2MB로 증가 — expanding walk-forward가 의도대로 누적된 것을 확인.

**인증 방식**: GitHub OIDC(액세스 키 없음). 저장소 Variables `AWS_ROLE_ARN`·`S3_BUCKET`·
`S3_PREFIX`로 스텝이 조건부 활성된다(`if: vars.AWS_ROLE_ARN != ''`). 스케줄 실행은 **기본
브랜치(`dev`)** 에서 돌며, 해당 브랜치로 AssumeRole이 정상 동작함을 확인했다.

> **주의**: 이 검증은 "연결이 된다"를 확인한 것이고, **매일 안정적으로 돈다는 보장은 아니다.**
> 7/17·7/18은 CI 레이트리밋으로 수집이 0행이 되어 연속 실패했다(§7-1의 재시도·품질 게이트로 대응). 수집 신뢰성 보강이
> 함께 있어야 daily가 실제로 유지된다.


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

### 생산자 구현 — `src/inference/precompute.py` (민지·도현 접점)
계약은 민지가 확정한다. 두 함수로 나뉜다:

- **`build_today_obs(cfg, prev_weights=None) -> (obs, date)`** (민지 책임 — 데이터 글루)
  - `build.py`와 동일 체인(수집 캐시→로그수익률→지표→187 조립)으로 **마지막 행**을 취한다.
    전체 원시로 계산해 마지막 행이 build 산출물과 **정확히 일치**한다(정규화 재현 대조 가능).
  - 정규화 재현: `config inference.scaler_run_id/fold_id`가 가리키는 fold의 `scaler_stats`를
    `read_scaler_stats`→`ZScoreScaler.from_stats_rows`로 복원해 학습과 동일한 z-score를 적용
    (룩어헤드 없음). prev_weight 5칸은 `prev_weights`(정규화 안 함), `None`이면 SHV 100% 콜드스타트.
  - 반환 `obs`: `float32`, shape `(state_dim,)`, 컬럼순서 `schema.feature_names(W)`.
- **`generate_latest(cfg, model=None) -> dict`** (도현 책임 — 모델 파트)
  - 직전 `latest.json`에서 prev 비중을 읽어 `build_today_obs` 호출 → `PPO.load(inference.model_path)`
    → `predict` → `portfolio_env._softmax`(학습 step과 동일) → 아래 스키마로 **원자적 기록**(tmp→rename).
  - `model` 미지정 시 config에서 로드(테스트는 목 모델 주입). 실행 `python -m src.inference.precompute`.

### `inference` 설정 (`config.yaml`)
어느 policy를 어느 build 통계로 정규화 재현할지 지정한다 — **학습 provenance와 값이 일치해야 한다**:
| 키 | 의미 |
|---|---|
| `model_path` | 배포 policy.zip 경로 (도현이 배포 확정 후 채움) |
| `scaler_run_id` | 그 policy가 학습된 build `run_id` (meta.sqlite `scaler_stats` 조회 키) |
| `scaler_fold_id` | 그 policy가 학습된 `fold_id` |
| `model_version` | `latest.json`에 기록되는 버전 태그 |

> **build run_id provenance (이슈 #27, 구현 완료)**: `train.py`는 학습 시작 시 해당 fold를 가장
> 최근에 빌드한 build `run_id`를 `feature_store.latest_run_id_for_fold(meta_db, fold_id)`로
> 조회한다(파티션은 최신 build가 덮어쓰므로 `runs.created_at` 최신 run이 디스크의 주인). 그 값을
> MLflow `feature_store_run_id` 파라미터와 모델 파일명(`ppo_fold{fold}_{build_run_id}_{mlflow_run_id}.zip`)에
> 기록한다. 학습 종료 시 CLI가 `config.inference`에 넣을 `model_path`/`scaler_run_id`/`scaler_fold_id`를
> 그대로 출력하므로, 배포할 policy를 정한 뒤 그 값을 `config.yaml`에 복사하면 추론 정규화가 학습과
> 일치한다. (provenance 미상이면 `scaler_run_id` 자리에 `nofs`가 박혀 추적 불가함이 드러난다.)

### 계약 검증 (`src/api/main.py::get_latest_inference`)
- 파일이 없으면 **503**을 반환한다 — precompute가 아직 안 돌았다는 정상 상태이지 버그가 아니다.
- `weights`의 자산 키가 `config.yaml`과 다르거나 합이 1에서 벗어나면 **500**을 반환해 파이프라인 오류를
  조기에 드러낸다. (생산자는 원자적 기록으로 부분쓰기 상태의 파일 노출을 방지한다.)

### 8-1. 배포 산출물 반출 — policy.zip만 옮기면 된다

학습 머신(RunPod 파드 등)에서 서빙 환경으로 넘겨야 하는 것은 **policy.zip 하나뿐**이다.

| 파일 | 반출 필요? | 이유 |
|---|---|---|
| `ppo_fold<id>_<build_run_id>_<mlflow_run_id>.zip` | **필요** | 서빙 환경은 모델을 스스로 만들 수 없다 (gitignore) |
| `meta.sqlite` | **불필요** | 각자 `build`해서 자기 run의 통계를 쓰면 된다 (아래) |

#### build run_id는 머신 간 재현되지 않는다
```python
run_id = f"{config_hash(cfg)}-{datetime.now().strftime('%Y%m%d%H%M%S')}"
```
타임스탬프가 들어가므로 다른 머신에서 `build`를 다시 돌려도 같은 run_id가 나오지 않는다.
`scaler_stats`는 `(run_id, fold_id, feature_name)`이 기본키라, 학습 머신의 run_id를
`inference.scaler_run_id`에 그대로 박아두면 서빙 환경에서 조회에 실패한다.

#### 해결: `scaler_run_id`를 비우면 자동 해석
`_restore_scaler`는 `scaler_run_id`가 비어 있으면 **현재 config의 `config_hash`로 만들어진
최신 build run**을 자동 선택한다(`feature_store.latest_run_id_for_config`). 따라서 CI·서버가
각자 `build`를 돌린 뒤 자기 run을 쓰면 되고, 통계 파일을 옮길 필요가 없다.

**근거 — 재빌드해도 통계가 사실상 동일하다(실측).** 같은 config로 하루 간격 재빌드해 fold1
통계 182개를 비교한 결과:

| 항목 | 결과 |
|---|---|
| 키 집합·행수 | 완전 일치 (182개) |
| 완전 일치 | 37/182 |
| **최대 상대오차** | **7.4e-4 (0.07%)** |

fold train 구간이 `anchor ~ train_end`로 **거래일 위치 기준** 고정되고(`splits.make_folds`가
`idx.get_loc`으로 위치 계산) 새 데이터는 뒤에만 붙으므로 train 구간 자체가 변하지 않는다.
남는 미세 오차는 `auto_adjust=True`(배당 소급 조정)의 **반올림 노이즈**다 — 로그수익률은
균일 스케일에 불변이지만 조정가가 유한 자릿수로 저장되기 때문. 오차가 큰 항목이
`feat_TLT_MACD_Hist`(가격 차분이라 스케일 불변 아님)와 `SHV` 계열(값이 1e-4 수준이라 상대오차가
커 보임)인 것도 이 설명과 일치한다. z-score 기준 0.0007σ 수준이라 정책 입력에 무의미하다.

> **`config_hash`로 좁히는 이유**: 그냥 "최신 run"을 쓰면 `window`·자산구성이 다른 빌드의
> 통계를 조용히 집어 차원·의미가 어긋난다. 해시가 같아야 최소한 State 규격이 같다.

##### `config_hash` 범위 — 자동 해석의 유일한 안전장치

자동 해석이 도입되면서 `config_hash`는 "이 build를 이 정책에 써도 되는가"를 판정하는 **유일한
안전장치**가 됐다. 그래서 **통계 값을 바꾸는 설정을 모두** 해싱한다:

| 포함 | 왜 |
|---|---|
| `assets`·`window` | State 규격(차원·자산 순서) |
| `features` | 지표 파라미터(RSI 길이·MACD·bbands)가 바뀌면 지표 값이 달라짐 |
| `normalize` | scope·eps가 바뀌면 μ/σ 정의가 달라짐 |
| `split` | anchor·test_blocks·embargo가 바뀌면 **fold train 구간**이 달라져 통계가 달라짐 |

| 제외 | 왜 |
|---|---|
| `data.start`/`end` | fold 경계는 anchor·test_blocks가 정하고 신규 데이터는 뒤에만 붙으므로 train 구간·통계 불변. 포함하면 기간 연장마다 불필요한 재빌드를 부른다 |
| `transaction_cost` | 환경·보상의 값이지 데이터 산출물과 무관 |
| `inference`·`model` | 소비 측 설정이라 build 산출물에 영향 없음 |

따라서 지표 파라미터나 fold 경계를 바꾼 build는 **해시가 달라져 자동 선택에서 자연히 걸러지고**,
`build run이 없습니다`로 **조용히가 아니라 크게 실패**한다. 엄밀한 감사 추적이 필요하면 여전히
`scaler_run_id`를 명시해 고정한다.

> **호환성**: 해시 범위를 넓히면 **이전 run_id는 더 이상 자동 매칭되지 않는다**. 명시 고정한
> `scaler_run_id`는 그대로 동작하고, 자동 해석은 **재빌드 후** 새 run을 집는다. `daily.yml`이
> 매일 build하므로 CI는 다음 실행에서 자연히 회복되고, 로컬은 `python -m src.data.build` 한 번이면 된다.
> (통계 값 자체는 재빌드해도 동일하므로 — 위 실측 7.4e-4 — 정책 호환성에는 영향이 없다.)

#### policy.zip 반출 방법
- **현재(수동)**: RunPod은 `runpodctl send <파일>` → 로컬에서 `runpodctl receive <코드>`.
  컨테이너 디스크는 파드를 Stop/Terminate하면 삭제되므로 **끄기 전에** 받아야 한다.
- **지향점**: S3(`config.data.s3_bucket`) 업로드 자동화. 찬휘 AWS 세팅 완료 후 `s3_sync`와
  같은 방식으로 policy를 올리면 학습 머신과 서빙 머신이 완전히 분리된다(이슈 #33).

### 아직 정해지지 않은 것
- daily.yml precompute 스텝 활성 시점 — 도현 실제 policy.zip + `inference` 값 확정 후(§작업④).
- 배치 실패 시 이전 `latest.json` 유지 여부(현재는 원자적 덮어쓰기라 성공 시에만 교체).
- **policy.zip 반출 자동화**(§8-1) — 현재 수동 전송. S3 경로 확정 필요(이슈 #33).

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
| v0.7 | 2026-07-13 | precompute 생산자 구현(`src/inference/precompute.py`)·`config.inference` 섹션 반영. build_today_obs(민지 글루)·generate_latest(도현 모델 파트) 접점 계약 확정. |
| v0.8 | 2026-07-15 | 이슈 #27: `train.py`가 학습 fold의 build `run_id`를 provenance로 기록(MLflow 파라미터·모델 파일명). `feature_store.latest_run_id_for_fold` 추가. 배포 시 `config.inference.scaler_run_id`에 복사. |
| v0.9 | 2026-07-18 | §8-1 추가: 배포 산출물(policy.zip + meta.sqlite) 반출 절차. RunPod 실학습 중 발견 — build `run_id`가 타임스탬프 기반이라 재현 불가하므로 `meta.sqlite`를 함께 옮기지 않으면 정규화 재현이 실패한다. |
| v0.10 | 2026-07-19 | §8-1 개정: `meta.sqlite` 반출 불필요로 정정. `scaler_run_id`를 비우면 같은 `config_hash`의 최신 build run을 자동 선택하도록 `_restore_scaler` 개선(`feature_store.latest_run_id_for_config`). 재빌드 통계 동일성 실측(최대 상대오차 7.4e-4) 근거 첨부. 민지 제안(이슈 #33). |
| v0.11 | 2026-07-19 | `config_hash` 범위를 `features`·`normalize`·`split`까지 확대(PR #36 후속). 자동 run 해석의 유일한 안전장치이므로 통계 값을 바꾸는 설정을 모두 포함해 비호환 build를 조용히 선택하지 못하게 한다. `data.start/end`·`transaction_cost`는 통계 불변이라 의도적으로 제외. 기존 run_id는 재빌드 후 자동 회복. |
| v0.12 | 2026-07-19 | §7-2 추가: S3 실연결 검증 기록(daily.yml 수동 트리거로 OIDC 인증·업로드 20개 파일 확인). §1 표·§3·README의 "AWS 세팅 후 활성" 표기를 활성 완료로 정정. (§7-1 수집 신뢰성은 PR #37에서 추가됐으나 머지 중 이력 항목이 누락돼 여기 함께 기록한다.) |
