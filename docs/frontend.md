# 프론트엔드 (Next.js 14 App Router)

Vercel 배포 대상 대시보드. Plotly.js로 자산 비중·성과곡선을 시각화하고, FastAPI
추론 서버(`src/api/main.py`)에서 익일 추천 비중을 가져와 표시한다.

## 라우트

| 경로 | 목적 | 데이터 소스 |
| --- | --- | --- |
| `/` | 랜딩 페이지 | 정적 |
| `/inference` | 오늘의 권장 자산 비중 + 시장 상태 지표 | **`GET /inference/latest`** (실 API) · 시장 지표는 mock |
| `/backtest` | 성과·롤링·리스크·자산분해·극단분석 통합 리포트 | 현재 시연용 합성 데이터 |
| `/models` | MLflow 실험 랭킹·학습 곡선·하이퍼파라미터·배포 모델 | mock (MLflow API 프록시 전) |
| `/dashboard` | 하위 호환용 리다이렉트 → `/inference` | — |

## 환경변수

`.env.local`(gitignore) 또는 배포 환경변수로 다음을 설정한다.

| 이름 | 기본값 | 설명 |
| --- | --- | --- |
| `NEXT_PUBLIC_API_BASE_URL` | `http://localhost:8000` | FastAPI 베이스 URL. 배포 시 Render API URL(robustam-api.onrender.com)로 교체 |

`.env.example` 참조. `NEXT_PUBLIC_` 접두어는 클라이언트 번들 노출 대상만.

## 페이지별 상태 (2026-07-19 기준)

### `/inference` — API 연동됨

- 오늘의 권장 자산 배분(도넛): API의 `weights`로 렌더 (자산 순서·색상은
  `app/lib/assets.ts`가 CLAUDE.md §2와 동기화된 SSOT).
- 배치 상태 카드: `model_version`, `generated_at`, 응답 지연(성능 목표 <200ms 판정).
- 3상태 표시: 로딩 스켈레톤 / 오류 카드(네트워크·503·기타 HTTP) / 정상.
- 아직 API로 나오지 않는 카드(전일 변동성·특징 벡터·이력)는 `DemoBadge`로 명시.

### `/backtest` — 실데이터 우선, 없으면 mock

섹션 구성:
- **핵심 지표**: KPI(CAGR/Sharpe/MDD/Vol) + 확장 리스크(Sortino, Calmar, VaR/CVaR 95%, Skew, Kurtosis, Beta, Hit Ratio, Best/Worst day).
- **성과 vs 벤치마크**: 누적 NAV(1/N·60:40·B&H 오버레이), Drawdown 곡선, 전략 비교표, 벤치마크 대비 초과수익(α) 시계열.
- **롤링·시즈널 분석**: Rolling Sharpe(252d)·Rolling Vol 이중축, 월별 수익률 히트맵.
- **거래·비용**: 일일 Turnover 바 + 누적 거래비용(초기 NAV 대비 %) 이중축.
- **자산 분해**: 자산별 누적 P&L 기여도 스택 영역, 자산 상관계수 히트맵, walk-forward Fold별 성과표.
- **극단·수중 분석**: Underwater 지속기간 히스토그램, Best·Worst 10 거래일.
- **피처 콤보 비교 (M0~M3 · full)** *(신규, 3순위)*: 콤보별 성과 표(Sharpe/CAGR/MDD/Vol/회전율/거래비용, Sharpe 최고 강조), 콤보 5종 누적 NAV 오버레이 차트, Fold × 콤보 매트릭스(Sharpe·MDD 2종), 자산 비중 스택 영역(시간축 국면 이동 시각화).
  - 실데이터 소스: `python -m src.backtest.export --combo full=runs/full.zip --combo M0=runs/m0.zip …` 로 생성된 `backtest.json`의 `combos` 필드. 미탑재(구 산출물 또는 도현 M0~M3 학습 전)이면 `mock.ts::COMBOS_MOCK`으로 조용히 폴백.
  - 콤보 파라미터의 SSOT는 백엔드 `src.backtest.export.COMBO_ORDER/COLOR/DISPLAY` — FE `mock.ts::COMBO_PARAMS`가 색·표시명을 그와 동기화한다.
- **데이터 소스 배지**: 상단 우측에 실데이터/mock 여부·생성 시각 표시.
- 시연용 mock은 `app/backtest/mock.ts`에서 결정론적으로 파생(seed 고정). `python -m src.backtest.export` 실행 시 `public/backtest.json`이 생성돼 자동으로 실데이터로 교체된다.

### `/models` — MLflow 대시 (mock, 스키마는 train.py와 1:1)

**원칙**: 이 페이지에 표시되는 모든 필드는 `src/models/train.py`가 실제로 MLflow에
로깅하는 항목과 1:1로 대응한다. 로깅되지 않는 값은 **표시하지 않고 배너로 명시적으로
안내**한다 (mock으로 위장하지 않음).

- **배포 모델 카드**: `run_id`·`model_version`·`model_path`·`valid_sharpe`·`feature_set`·
  `state_dim`·scaler fold·`feature_store_run_id`.
- **배포 파라미터 표** (`train.py::log_params` 그대로): `algorithm`·`policy`·`feature_set`·
  `state_dim`·`window`·`transaction_cost`·`total_timesteps`·`train_cost_multiplier`(λ)·
  `vol_penalty_coef`(κ)·`learning_rate`·`seed`.
- **실험 랭킹 표**: `valid_sharpe`·`valid_total_log_return`·`valid_avg_turnover`·
  `valid_total_txn_cost` (모두 `evaluate(split="valid")` 반환값).
- **Fold × Seed × Timesteps 히트맵**: `valid_sharpe` 기준.
- **경고 배너 2종**: ①이슈 #34 배포 정책 valid Sharpe 음수 · ②MLflow 로깅 미배선 안내
  (valid MDD, test Sharpe, 학습 곡선 3종 — 도현 담당 회의 안건).
- **학습 곡선 카드**: 현재 "미배선" 안내 카드 (SB3 `model.learn()`에 MLflow 콜백이
  붙으면 원래 3축 차트로 복원).


## API 계약

`app/lib/api.ts`가 정의하는 응답 타입은 `src/api/schemas.py::LatestInference`와
일치해야 한다.

```ts
type LatestInference = {
  date: string;
  generated_at: string;
  model_version: string;
  weights: Record<string, number>;
};
```

precompute가 아직 안 돌았을 때 백엔드는 503을 반환하고(`src/api/main.py`),
프론트는 이를 "오늘의 추천 비중이 아직 준비되지 않음"으로 명시적으로 표시한다.

## 로컬 실행

```bash
# 백엔드
uvicorn src.api.main:app --reload  # :8000

# 프론트
cd frontend
npm install
cp .env.example .env.local
npm run dev                         # :3000
```
