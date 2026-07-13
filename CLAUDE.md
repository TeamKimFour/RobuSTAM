# CLAUDE.md

이 파일은 Claude Code(및 협업자)가 본 저장소에서 작업할 때 따라야 하는 규칙과 프로젝트 맥락을 정의한다.

---

## 1. 프로젝트 개요

**RobuSTAM** — 여러 자산(주식·ETF·현금)의 비중을 RL 에이전트가 매 거래일 동적으로 조정해
위험조정수익(샤프지수)을 극대화하되, 매크로 국면별 유효성과 한계를 **현실적 제약(거래비용·슬리피지)**
하에서 정직하게 검증하는 완결형 퀀트 MLOps 백테스팅·서비스 플랫폼.

### 핵심 철학
- **결과론적 수익률 자랑이 아니라 강건성(Robustness) 검증이 목표.**
  거래비용·슬리피지·데이터 누수를 엄격히 통제한 상태에서 시장 국면별 유효성과 한계를 투명하게 규명한다.

### 정량 목표
- 백테스트 현실성: 편도 거래수수료 **0.1%**(확정) + 슬리피지 반영 커스텀 Gymnasium 환경.
- 성능: 역사적 폭락장·고금리 국면 Out-of-sample 구간에서 벤치마크(1/N, 60:40, B&H) 대비
  **샤프 15%+ 개선** 또는 **MDD 20%+ 방어**.
- API: 시장 상태 → 익일 최적 비중 반환 FastAPI 추론, **평균 지연 200ms 이하**.

### 범위
- **In:** 일봉(Daily) 데일리 리밸런싱, 고정 5종 ETF, 오프라인 백테스트 + 추론 API 서빙.
- **Out:** 실거래 연동, 고빈도(분/초/틱) 매매, 대규모 종목 다변화.

---

## 2. 자산 유니버스 & 핵심 규칙 (절대 변경 금지)

- **자산 5종, 순서 고정:** `[SPY, EWY, TLT, GLD, SHV]`
  (미국주식 · 한국주식 · 미국장기채 · 금 · 초단기채/현금성)
- **State(관측공간):** `Box(shape=(187,))` — **상세 명세는 [docs/state_spec.md](docs/state_spec.md)가 단일 진실원천(SSOT).**
  `187`을 코드에 하드코딩하지 말 것. `config`의 `W`에서 산출(`(5×W)+37`, 기본 `W=30`).
- **Action:** 자산별 목표 비중 벡터, `∑wᵢ=1`. **Softmax 정규화는 형우 Gym `step()`에서 처리.**
- **거래비용 c = 편도 0.1%**, 슬리피지 반영. 보상함수에 거래비용 항 필수(과매매 방지).
- **룩어헤드(Look-ahead Bias) 방지 최우선.** 정규화 통계량은 walk-forward 학습구간에서만 fit.

---

## 3. 기술 스택

| 레이어 | 도구 |
| --- | --- |
| Data Engineering | yfinance, pandas, numpy, pandas-ta, Apache Arrow(Parquet), SQLite |
| RL / Modeling | Gymnasium(커스텀 환경), Stable-Baselines3(PPO→SAC), PyTorch, MLflow |
| Backend / Serving | FastAPI, Pydantic, uvicorn, 자체 백테스트 엔진, 멀티 벤치마크 모듈 |
| Frontend | Next.js(Vercel 배포), React, Plotly.js, quantstats |
| DevOps | Docker / Docker Compose, GitHub Actions, Airflow/cron, RunPod(GPU) |
| 협업 | Git/GitHub, Notion(문서), Discord(알림) |

1차 알고리즘은 **PPO**(안정성), 시간 여유 시 SAC 비교.

---

## 4. 팀 R&R

| 팀원 | 역할 |
| --- | --- |
| 김민지 | MLOps, 데이터 엔지니어 |
| 김형우 | 강화학습 환경 설계, 데이터/지표 시각화(Frontend) |
| 김도현 | 백엔드, 모델링(RL 본체·추론 API) |
| 김찬휘 | 백엔드, 클라우드 & DevOps(백테스트 엔진·배포) |

---

## 5. 개발 워크플로우

### 5-1. 브랜치 전략
- **`main`** — 최종 배포용. **직접 push 절대 금지.**
- **`dev`** — 4명 코드 통합 브랜치. 기본(Default) 브랜치. **직접 push 금지(PR로만 병합).**
- **개인 작업 브랜치** — `dev`에서 분기. 명명: `<type>/<설명>` 예) `feat/email-auth`

### 5-2. 커밋 메시지 규칙

**타입(Type)**

| 타입 | 의미 | 예시 |
| --- | --- | --- |
| `feat` | 새로운 기능 추가 | `feat: 구글 로그인 기능 구현` |
| `fix` | 버그 수정 | `fix: API 응답 데이터 누락 수정` |
| `docs` | 문서 수정 | `docs: 설치 방법 섹션 업데이트` |
| `style` | 의미에 영향 없는 스타일(세미콜론·들여쓰기) | `style: lint 규칙 적용` |
| `refactor` | 기능 유지한 코드 구조 개선 | `refactor: 중복 DB 연결 로직 함수화` |
| `test` | 테스트 코드 추가/수정 | `test: 회원가입 API 유닛 테스트 추가` |
| `chore` | 빌드·패키지 설정 등 코드 외적인 것 | `chore: docker-compose 설정 변경` |

스코프 사용 가능: `feat(api): JWT 인증 미들웨어 추가`, `chore(deploy): 도커 빌드 단계 추가`

**커밋 7대 규칙**
1. 제목과 본문은 한 줄 띄운다.
2. 제목은 50자 이내.
3. 제목 끝에 마침표(.)를 찍지 않는다.
4. 제목은 명령문 사용(`fixed` ❌ → `fix` ✅).
5. 본문은 '어떻게'보다 **'무엇을', '왜'** 했는지 설명.
6. 한 커밋에는 최대한 하나의 작업만 담는다.
7. (본문이 길면) 각 줄은 적절히 줄바꿈해 가독성을 유지한다.

### 5-3. 금지 사항 ⛔
- `git push --force` 등 **강제 명령어 사용 금지.**
- `main` · `dev`에 **직접 push 금지** (반드시 개인 브랜치 → PR).

---

## 6. Claude Code 작업 시 필수 준수 사항

### 6-1. 커밋·PR에 Claude 기여 흔적을 남기지 않는다 (중요)
- 커밋 메시지에 **`Co-Authored-By: Claude ...` 라인을 절대 추가하지 않는다.**
- 커밋·PR 본문에 **"Generated with Claude Code", "🤖", Claude 관련 서명/푸터를 넣지 않는다.**
- PR 설명은 사람이 작성한 것처럼 프로젝트 내용만 담는다.
- 커밋 메시지는 위 §5-2 규칙(타입·한국어·50자·명령문)만 따른다.

### 6-2. 작업 규칙
- 명시적 요청이 있을 때만 커밋·push·PR을 수행한다.
- 절대 `main`/`dev`에서 직접 작업하지 않는다. `dev`에서 개인 브랜치를 분기한다.
- `--force` 및 히스토리 재작성 금지.
- 응답·문서·커밋은 **한국어**로 작성한다.
- State 차원, 자산 순서, 거래비용 등 §2 핵심 규칙은 임의 변경하지 않는다(변경 시 회의 합의 필요).
- **코드·구조·결정을 변경하면 관련 문서를 같은 PR에서 함께 갱신한다.** 인터페이스·차원·핵심 규칙이
  바뀌면 해당 SSOT 문서(`docs/state_spec.md`, `config/config.yaml`)를 **우선** 갱신하고 코드를 맞춘다.
  새 모듈·파이프라인은 `docs/`에 설계 기록을 남기고, 사용법 변화는 `README.md`에 반영한다.
  문서 갱신 없는 코드 변경 PR은 미완성으로 본다.
- **PR 생성 시 `.github/PULL_REQUEST_TEMPLATE.md` 형식을 따른다.** CLI(`gh pr create`)로 만들 때도
  템플릿 섹션(작업 개요·작업 영역·관련 이슈·변경 상세·리뷰어 체크리스트·기타)을 채워 본문을 작성한다.
  리뷰어 체크리스트는 정직하게 표시하고, 해당 없는 항목은 "해당 없음"으로 명시한다.

---

## 7. 디렉토리 구조 (계획안)

> 아직 구현 전이라 **확정이 아닌 제안**이다. 실제 구현하며 조정한다.

```
RobuSTAM/
├── CLAUDE.md
├── config/
│   └── config.yaml          # W, 자산목록, 거래비용 c 등 하이퍼파라미터 (단일 출처)
├── docs/
│   └── state_spec.md        # State 명세 SSOT
├── data/                    # (gitignore) 데이터 산출물
│   ├── raw/                 #   yfinance 원시 일봉
│   └── feature_store/       #   피처 Parquet + SQLite
├── src/
│   ├── data/                # [민지] 수집·피처 생성·정규화·walk-forward 분할
│   ├── env/                 # [형우] Gymnasium 커스텀 환경 (Softmax, 거래비용·슬리피지)
│   ├── models/              # [도현] RL 학습(PPO→SAC), 정책, MLflow 연동
│   ├── backtest/            # [찬휘] 백테스트 엔진, 멀티 벤치마크(1/N·60:40·B&H)
│   └── api/                 # [도현] FastAPI 추론 서버 (상태→익일 비중)
├── frontend/                # [형우] Next.js 대시보드 (Plotly.js, Vercel 배포)
├── notebooks/               # 실험·분석 노트북
├── tests/                   # 단위·통합 테스트
├── mlruns/                  # (gitignore) MLflow 실험 추적
└── docker/                  # Dockerfile, docker-compose.yml
```

대괄호는 주 담당자(§4). `config.yaml`은 `W`·자산목록·거래비용 등 §2 핵심값의 단일 출처로,
환경·모델·Feature Store가 모두 여기서 읽어 차원·비용이 항상 일치하도록 한다.

---

## 8. 참고 문서
- [docs/state_spec.md](docs/state_spec.md) — State(관측공간) 명세 SSOT (`Box(187)`, 인덱스 맵, 부속결정)
