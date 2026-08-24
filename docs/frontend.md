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
- **국면 격자 스위처(상단)**: `전 구간 · COVID · 고금리 · 최근` 4단 세그먼트 컨트롤. URL 쿼리(`?regime=fold1` 등)로 상태를 관리해 SSR·공유 URL과 자연스럽게 맞물린다. fold ↔ 국면 매핑은 `docs/data_pipeline.md §4-1`(Fold 1=2020–2021 COVID / Fold 2=2022–2023 고금리 / Fold 3=2024–2025 최근)이 SSOT이며, `app/backtest/regime.ts`에 이관되어 있다. 민지 별도 국면 라벨 데이터셋이 붙기 전까지는 fold 단위가 사실상 국면 격자다.
- **핵심 지표**: KPI(CAGR/Sharpe/MDD/Vol) + 확장 리스크(Sortino, Calmar, VaR/CVaR 95%, Skew, Kurtosis, Beta, Hit Ratio, Best/Worst day). 국면 필터 적용 시 KPI 4종은 국면 구간에서 재산출되고, 확장 리스크는 전 구간 기준(스칼라 재산출 로직 미구현)이라 별도 안내 카드로 대체된다.
- **성과 vs 벤치마크**: 누적 NAV(1/N·60:40·B&H 오버레이), Drawdown 곡선, 전략 비교표, 벤치마크 대비 초과수익(α) 시계열. 모두 국면 구간으로 필터·재산출.
- **국면별 RL vs 벤치마크 판정** *(신규, 8주차 발표 하이라이트)*: fold(=국면)별로 RL이 1/N·60:40·B&H 각각에 대해 CLAUDE.md §1 기준(샤프 +15% 개선 또는 MDD 20% 방어)을 만족했는지 카드로 노출. 스위처가 `전 구간`이면 3장 그리드, 특정 국면이면 그 fold 한 장을 확대. 데이터 소스는 `backtest.json`의 `comparison[].vs_benchmark`(runner._compare_to_benchmarks 원본).
- **롤링·시즈널 분석**: Rolling Sharpe(252d)·Rolling Vol 이중축, 월별 수익률 히트맵. 국면 필터로 잘린 시계열로 재계산.
- **거래·비용**: 일일 Turnover 바 + 누적 거래비용(초기 NAV 대비 %) 이중축. 국면 필터 적용.
- **자산 분해**: 자산별 누적 P&L 기여도 스택 영역(국면 필터), 자산 상관계수 히트맵(전 구간), walk-forward Fold별 성과표(국면 매핑된 fold 한 줄로 필터).
- **극단·수중 분석**: Underwater 지속기간 히스토그램, Best·Worst 10 거래일. 국면 필터 적용.
- 모든 시계열은 `app/backtest/mock.ts`에서 결정론적으로 파생(seed 고정). `python -m src.backtest.export` 실행 시 `public/backtest.json`이 생성돼 자동으로 실데이터로 교체된다.

### `/models` — MLflow 대시(mock)

- 배포 모델 카드(run_id, model_version, config_hash, scaler fold).
- 배포 하이퍼파라미터(PPO, learning_rate, gamma, GAE 등).
- 학습 곡선(reward·entropy·KL).
- Fold × Seed × Timesteps 그리드 히트맵.
- 12개 run 랭킹표. 이슈 #34(과매매로 valid Sharpe 음수) 경고 배너 상단 노출.


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
