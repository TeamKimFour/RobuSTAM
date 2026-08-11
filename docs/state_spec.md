# RobuSTAM — State(관측공간) 명세서 v1.3

> **상태: 확정(권장안 채택)** · 부속결정 ①②③ 기본값으로 확정 완료(회의에서 통보)
> 본 문서는 형우(Gym 환경) · 도현(SB3 모델) · 민지(Feature Store)가 공유하는 **단일 인터페이스 계약**이다.
> 이 문서와 코드가 어긋나면 이 문서가 우선한다. 변경은 회의 합의 후 버전 올려 반영.

---

## 0. 한 줄 요약

```
관측공간(State) = Box(shape=(187,))   # W=30 기준, config의 W에서 산출
```

자산 순서 **고정**: `[SPY, EWY, TLT, GLD, SHV]` (모든 블록 이 순서 준수)

---

## 1. 설계 원칙

**"원시 수익률만 시계열, 가공지표는 최신값"** 하이브리드.

- 로그수익률은 가격 경로(sequence) 자체가 신호이므로 **30일 history** 부여.
- RSI·Vol·MACD·MA_Cross 등 가공지표는 **이미 과거를 요약한 롤링 값**이라 history 부여 시
  자기상관 99%로 정보는 안 늘고 차원만 폭증 → **최신 1일치만** 사용.
- 차원-샘플 비율 보호: 학습구간 약 1,500~2,000 샘플 대비 187차원이 건전(과적합·수렴난이도·200ms 추론 모두 유리).

> 폐기된 안: 1,355(시장지표 자산별 중복곱 오류) / 205(지표 50 산출 근거 불명확).
> full-history 확장안(중복 제거 후 정확값)은 **1,115**이며, 1차 모델 비권장·성능 한계 시 확장 카드로만 보관.

---

## 2. 차원 수식

```
D = (A × W) + (K_asset × A) + K_market + A
  = (5 × 30) + (6 × 5)      + 2        + 5
  = 150      + 30           + 2        + 5
  = 187
```

| 기호 | 의미 | 값 |
|---|---|---|
| `A` | 자산 수 | 5 |
| `W` | 수익률 history 윈도우 (하이퍼파라미터, 회의 확정 30) | 30 |
| `K_asset` | 자산별 최신 지표 수 | 6 |
| `K_market` | 시장 공통 최신 지표 수 | 2 |
| `+A` | 직전 보유 비중(Prev_Weight) | 5 |

**W 변경 시 자동 산출:** `W=20 → 137`, `W=30 → 187`, `W=60 → 337`

**K_asset·K_market 변형 지원 (v1.1):** 지표 개수가 바뀌어도 코드는 config만 보고 D를 산출한다.
예) `W=30`에서 `K_asset=2, K_market=1 → D=166`, `K_asset=3, K_market=1 → D=171`,
`K_asset=3, K_market=2 → D=172`. 검증은 `tests/test_variable_state_dim.py`가 파라메트라이즈로
env·모델 왕복 계약을 확인한다.

**피처 콤보 M0~M3 (v1.2):** 위 변형 메커니즘을 실제로 소비하는 이름 붙은 콤보 카탈로그가
`config.yaml`의 `feature_combos`(§ CLAUDE.md 원칙과 동일하게 config가 SSOT)로 확정됐다.
`active_combo`(기본 `full`)가 소비되는 콤보를 고르고, `config_loader.resolve_combo`가 그
콤보의 `asset`/`market` 리스트를 `config.features`에 반영한다. `full`은 기존 카논 6+2(=187)로
env·추론·기존 policy.zip과의 하위호환을 위해 기본값을 유지한다.

| 콤보 | asset_features | market_features | D (W=30) |
|---|---|---|---|
| `full`(기본) | 6종(카논) | `Equity_Bond_Ratio`, `Gold_Vol_Ratio` | 187 |
| `M0` | (없음) | `Equity_Bond_Ratio` | 156 |
| `M1` | `MACD_Hist`, `Rolling_Vol_20` | `Equity_Bond_Ratio` | 166 |
| `M2` | `MACD_Hist`, `Rolling_Vol_20`, `RSI_28` | `Equity_Bond_Ratio` | 171 |
| `M3` | `MACD_Hist`, `Rolling_Vol_20`, `RSI_28` | `Equity_Bond_Ratio`, `Drawdown` | 172 |

근거: `docs/feature_candidates.md`의 대리모델 스크리닝 — EBR(=M0의 유일 시장지표) 단독이
GBM R² 유일 양수를 기록해 M0을 추가했고, M1~M3는 RSI 길이 튜닝(14→28)·Drawdown 국면신호를
단계적으로 얹은 조합이다. **현재 범위**: `src/data/build.py`가 `build(combo="M0")` 등으로
콤보별 Feature Store를 병행 생성할 수 있다. `src/env`·`src/models/train.py`는 아직
`active_combo`(=`full`)만 소비한다 — M0~M3를 실제 RL 학습·백테스트에 연결하는 것은 별도
후속 작업이다(docs/data_pipeline.md §3-2).

---

## 3. 인덱스 맵 (구현 기준)

| 인덱스 | 블록 | 구성 |
|---|---|---|
| `0:150` | **수익률 윈도우** | 자산별 30일 로그수익률 (각 t-29 … t-0 순) |
| | | `[0:30]` SPY · `[30:60]` EWY · `[60:90]` TLT · `[90:120]` GLD · `[120:150]` SHV |
| `150:180` | **자산별 최신지표** | 자산당 6개: `[MA_Cross_5_20, RSI_14, MACD_Hist, Rolling_Vol_20, Bollinger_Band_Width, ROC_10]` |
| | | `[150:156]` SPY · `[156:162]` EWY · `[162:168]` TLT · `[168:174]` GLD · `[174:180]` SHV |
| `180:182` | **시장 공통지표** | `[180]` Equity_Bond_Ratio · `[181]` Gold_Vol_Ratio |
| `182:187` | **직전 비중** | `[182:187]` SPY, EWY, TLT, GLD, SHV |

> `Equity_Bond_Ratio`(SPY↔TLT), `Gold_Vol_Ratio`(GLD)는 **시장 전체 단일값**이다.
> 자산별로 곱하지 말 것 — 이것이 1,355안의 핵심 오류였다.

### 3-1. 지표 산식 (2주차 확정 · 파라미터는 config `features.params`)

모든 지표는 **인과적**(과거만 참조)이며 warm-up NaN은 메우지 않고 drop한다. 파라미터는 `config.yaml`에서 읽는다.

| 지표 | 산식 | 입력 |
|---|---|---|
| `MA_Cross_5_20` | `(SMA5 − SMA20) / SMA20` (정규화된 크로스 강도) | 조정종가 |
| `RSI_14` | `ta.rsi(close, 14)` (Wilder) | 조정종가 |
| `MACD_Hist` | `ta.macd(close,12,26,9)` 의 히스토그램(`MACDh`) | 조정종가 |
| `Rolling_Vol_20` | `logret.rolling(20).std()` (연율화 안 함, `annualize_vol=false`) | 로그수익률 |
| `Bollinger_Band_Width` | `ta.bbands(close,20,2)` 의 밴드폭(`BBB` = (상단−하단)/중앙) | 조정종가 |
| `ROC_10` | `ta.roc(close, 10)` = `close/close.shift(10) − 1` | 조정종가 |
| `Equity_Bond_Ratio` | `(SPY/TLT) / (SPY/TLT).rolling(20).mean() − 1` (주식·채권 상대강도의 추세 편차) | SPY·TLT 종가 |
| `Gold_Vol_Ratio` | `Rolling_Vol_20(GLD) / GLD_logret.rolling(60).std()` (금 단기/장기 변동성 비, 레짐 신호) | GLD 로그수익률 |
| `RSI_28`(M2·M3, v1.2) | `ta.rsi(close, 28)` (Wilder, `RSI_14`와 별도 길이 파라미터 `rsi_28_length`) | 조정종가 |
| `Drawdown`(M3, v1.2, **시장** 단일값) | `SPY_close / SPY_close.rolling(drawdown_lookback).max() − 1` (최근 고점 대비 낙폭, 국면 신호) | SPY 종가 |

> 세 지표(`MA_Cross_5_20`·`Equity_Bond_Ratio`·`Gold_Vol_Ratio`)는 v1.0 명세에 산식이 없었다.
> 위 산식은 **2주차 제안값**이며 PR 리뷰에서 팀 확정한다(변경 시 본 표를 우선 갱신).
> `Drawdown`의 `drawdown_lookback`은 **60일로 확정**(민지, 2026-08-09 — 252일 대안도
> 검토했으나 60일로 결정). 향후 재검토 시 `config.yaml`의 값만 바꾸고 재빌드하면 된다
> (코드 변경 불필요). `Equity_Bond_Ratio`처럼 자산별로 곱하지 않는다 — 시장 전체 단일값.

---

## 4. 구현 스니펫 (형우·도현 공용)

`187`을 상수로 박지 말 것. config의 `W`에서 산출한다.

```python
# config.yaml
#   window: 30
import numpy as np
from gymnasium import spaces

W            = config["window"]   # 30 (20/60 비교 시 여기만 변경)
N_ASSETS     = 5                  # SPY, EWY, TLT, GLD, SHV
N_ASSET_FEAT = 6                  # MA_Cross, RSI, MACD_Hist, Rolling_Vol, Bollinger_Width, ROC
N_MARKET_FEAT = 2                 # Equity_Bond_Ratio, Gold_Vol_Ratio

state_dim = (N_ASSETS * W) + (N_ASSET_FEAT * N_ASSETS) + N_MARKET_FEAT + N_ASSETS
# W=30 → 187

observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(state_dim,), dtype=np.float32)
```

> `low/high`는 부속결정 ①(정규화)에서 z-score 표준화를 채택하므로 `-inf/+inf`.
> 정규화를 안 한다면 피처별 범위를 별도 지정해야 한다.

---

## 5. 부속결정 — 확정 사항

> 셋 다 업계 표준 정답이 있어 기본값으로 **확정**한다. 회의에서는 변경이 아니라 통보 대상.
> 이견이 있을 경우에만 회의에서 재논의.

### ① 정규화(스케일링) — **확정** · 담당: 민지
- **방식:** 피처별 **z-score 표준화**(`(값-평균)/표준편차`)를 **Feature Store 출력 단계**에서 적용한다.
- **룩어헤드 방지:** 평균·표준편차 통계량은 **각 walk-forward 학습구간(in-sample)에서만 fit**,
  검증·테스트구간엔 그 값을 **적용만** 한다. (fold마다 재fit)
- **환경 영향:** 표준화로 값 범위가 열려 있으므로 `Box(low=-inf, high=+inf)`로 고정(§4).
- 이유: 로그수익률(±0.05) vs RSI(0~100) vs 비중(0~1)의 스케일 차가 학습을 불안정하게 만듦.

### ② SHV 지표 칸 처리 — **확정** · 담당: 형우·도현
- **결정:** SHV도 지표 **6칸 균일 유지**(빼지 않는다).
- 이유: SHV(현금성)는 지표값이 무의미하지만, 빼면 인덱스 맵이 어긋나 양쪽 코드가 깨짐.
  모델이 "SHV 지표는 항상 0 근처"로 학습하게 둔다.

### ③ W는 하이퍼파라미터 — **확정** · 담당: 형우·도현
- **결정:** `187`을 코드에 직접 박지 않고 `config`의 `W`에서 산출(§4 스니펫). **기본값 `W=30`.**
- 이유: 20/60일 비교 실험 시 차원이 자동 변경되어야 하며, Feature Store 출력 차원과
  환경 기대 차원이 항상 일치해야 학습이 시작됨.
- **v1.1 확장:** `K_asset`(config.features.asset 길이)·`K_market`(config.features.market 길이)도
  같은 원칙으로 config에서 온다. 민지 Feature Store가 지표 셋을 바꿔도 `schema.state_dim`·
  `schema.prev_weight_slice`·`schema.feature_names`가 그 개수를 인자로 받아 슬라이스를
  재계산하므로 env/추론 파이프라인이 자동으로 새 D를 소비한다.
- **v1.3 M0(K_asset=0) 소비자 검증 완료:** `tests/test_m0_env_consumer.py`가 자산 지표
  블록이 통째로 비는 M0(D=156)에서 env 소비자 계약을 다각도로 검증한다 — 빈 자산
  슬라이스, market 시작 위치, `feature_names`가 `feat_*`를 만들지 않음, `assemble` →
  `PortfolioEnv` 결합, 다중 스텝 진행, 룩어헤드 방어, turnover·cost 이론값 일치,
  DiscretePortfolioEnv 이산 액션(hold·+Δ·-Δ) 계약. PR #69 리뷰 후속(도현→형우, M0
  RunPod 학습 준비).

---

## 변경 이력

| 버전 | 날짜 | 내용 |
|---|---|---|
| v1.0 | 2026-06-28 | 최초 작성. 187 권장안 확정, 부속결정 ①②③ 명시. |
| v1.1 | 2026-08-05 | K_asset·K_market도 config에서 산출. schema 헬퍼가 인자를 받아 D 가변 지원(D=166/171/172 검증). |
| v1.2 | 2026-08-08 | 피처 콤보 M0~M3 확정(§2, `config.yaml` `feature_combos`/`active_combo`). `RSI_28`·`Drawdown`(시장) 신규 지표 추가(§3-1), `drawdown_lookback`은 2026-08-09 60일로 확정(252일 대안 검토 후). `full`(하위호환, D=187)이 기본 콤보로 유지됨 — env/train.py 실연결은 후속 작업. |
| v1.3 | 2026-08-11 | M0(K_asset=0, D=156) env 소비자 계약 심층 검증 추가(`tests/test_m0_env_consumer.py` 16개). PR #69 리뷰 후속으로 도현이 M0을 RunPod 학습 후보로 예고한 데 대응. schema 슬라이스·assemble·다중 스텝·룩어헤드·비용·이산 액션까지 M0 특유 상황을 명시적으로 커버. |
