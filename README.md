# RobuSTAM

여러 자산(주식·ETF·현금)의 비중을 강화학습 에이전트가 매 거래일 동적으로 조정해
위험조정수익(샤프지수)을 극대화하되, 거래비용·슬리피지 등 **현실적 제약 하에서 강건성을
정직하게 검증**하는 퀀트 MLOps 백테스팅·서비스 플랫폼.

- 자산 5종(고정): `SPY · EWY · TLT · GLD · SHV`
- 일봉 데일리 리밸런싱, 편도 거래비용 0.1%
- 자세한 규칙: [CLAUDE.md](CLAUDE.md) · State 명세: [docs/state_spec.md](docs/state_spec.md)
- 데이터 파이프라인: [docs/data_pipeline.md](docs/data_pipeline.md)

## 개발 환경 셋업

Python **3.12** 기준(CI와 동일 버전).

```bash
# 가상환경 생성·활성화
python3.12 -m venv .venv
source .venv/bin/activate

# 의존성 설치
pip install -r requirements.txt
```

## 데이터 수집

config(`config/config.yaml`)에 정의된 기간·자산으로 원시 일봉을 수집한다.

```bash
python -m src.data.collect
# → data/raw/prices_raw.parquet 생성
```

## Feature Store 빌드

원시 가격 → 187차원 정규화 State + 익일수익률(targets)을 fold/split 파티션으로 적재한다.

```bash
python -m src.data.build
# → data/feature_store/fold=*/split=*/{part,targets}.parquet + meta.sqlite
```

## 테스트

```bash
pytest tests/ -v
```

## 데이터 품질 검증 · EDA

```bash
# 자동 품질 검사 + EDA 요약(통계·상관·이상치)
python -m src.data.validate

# 표·차트로 직접 탐색하는 노트북 (개발 의존성 필요)
pip install -r requirements-dev.txt
jupyter lab notebooks/eda_raw_prices.ipynb
```

## 프로젝트 구조 (구현된 부분)

```
config/config.yaml      # W·자산·거래비용·기간·분할 단일 출처(SSOT)
src/
  config_loader.py      # config 로드 + state_dim 산출
  data/
    schema.py           # 187차원 State 인덱스맵(코드측 SSOT)
    collect.py          # yfinance 수집
    returns.py          # 로그수익률·윈도우
    feature_store.py    # Feature Store 입출력 (Parquet 파티션 + SQLite 메타)
tests/                  # 차원·인덱스맵·룩어헤드 회귀 테스트
docs/                   # state_spec.md(State SSOT), data_pipeline.md
```
