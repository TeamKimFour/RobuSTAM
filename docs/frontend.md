# 프론트엔드 (Next.js 14 App Router)

Vercel 배포 대상 대시보드. Plotly.js로 자산 비중·성과곡선을 시각화하고, FastAPI
추론 서버(`src/api/main.py`)에서 익일 추천 비중을 가져와 표시한다.

## 라우트

| 경로 | 목적 | 데이터 소스 |
| --- | --- | --- |
| `/` | 랜딩 페이지 | 정적 |
| `/inference` | 오늘의 권장 자산 비중 + 배치 상태 | **`GET /inference/latest`** (FastAPI) |
| `/backtest` | 성과곡선·Drawdown·벤치마크 비교·KPI | 현재 시연용 합성 데이터 (엔진 API 결정 후 교체) |
| `/dashboard` | 하위 호환용 리다이렉트 → `/inference` | — |

## 환경변수

`.env.local`(gitignore) 또는 배포 환경변수로 다음을 설정한다.

| 이름 | 기본값 | 설명 |
| --- | --- | --- |
| `NEXT_PUBLIC_API_BASE_URL` | `http://localhost:8000` | FastAPI 베이스 URL. 배포 시 nginx/CloudFront URL로 교체 |

`.env.example` 참조. `NEXT_PUBLIC_` 접두어는 클라이언트 번들 노출 대상만.

## 페이지별 상태 (2026-07-19 기준)

### `/inference` — API 연동됨

- 오늘의 권장 자산 배분(도넛): API의 `weights`로 렌더 (자산 순서·색상은
  `app/lib/assets.ts`가 CLAUDE.md §2와 동기화된 SSOT).
- 배치 상태 카드: `model_version`, `generated_at`, 응답 지연(성능 목표 <200ms 판정).
- 3상태 표시: 로딩 스켈레톤 / 오류 카드(네트워크·503·기타 HTTP) / 정상.
- 아직 API로 나오지 않는 카드(전일 변동성·특징 벡터·이력)는 `DemoBadge`로 명시.

### `/backtest` — API 미연결(시연용)

- KPI(CAGR / Sharpe / MDD / Vol), 누적 성과 곡선, Drawdown 곡선, 전략별 지표 테이블.
- 데이터는 결정론적 합성(`app/backtest/mock.ts`, seed 고정). 실 데이터가 아니라는
  경고 배지를 상단에 표기.
- 백엔드 백테스트 결과 조회 엔드포인트가 확정되면 `mock.ts` → API fetch로 교체.

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
