"""yfinance 원시 일봉 수집 모듈.

5종 자산의 조정종가(auto_adjust)를 받아 **공통 거래일 교집합**으로 정렬한다.
결측은 drop하며 **ffill로 미래 값을 끌어오지 않는다**(룩어헤드 1차 방어선).
산출물은 data/raw/prices_raw.parquet에 캐시한다(yfinance 비결정성·재다운로드 비용 회피).

수집 신뢰성 (2026-07 daily.yml 장애 대응):
    Yahoo는 CI 러너 IP를 간헐적으로 레이트리밋해 **빈 응답(0행)** 을 준다. 과거에는 이때
    수집이 "성공"으로 끝나 빈 캐시를 남겼고, 한참 뒤 지표 계산에서 정체불명의
    `TypeError: NoneType - NoneType`으로 터졌다(더 나아가 S3에 빈 데이터를 덮어쓸 뻔했다).
    그래서 여기서 두 겹으로 막는다.
      ① 지수 백오프 **재시도** — transient 레이트리밋을 흡수.
      ② `validate_prices` **품질 게이트** — 통과했을 때만 캐시에 쓴다.
         즉 **나쁜 데이터가 좋은 캐시를 덮어쓰지 못한다.**
"""

import time
from pathlib import Path

import pandas as pd

from src.data.validate import validate_prices

RAW_FILENAME = "prices_raw.parquet"

# 전체 기간(2009-10~2025-12) 기대 행수는 약 4,087. 이보다 크게 적으면 수집 사고로 본다.
DEFAULT_MIN_ROWS = 1000
DEFAULT_RETRIES = 3
DEFAULT_BACKOFF = 2.0  # 초 — 2 → 4 → 8


def _align(close: pd.DataFrame, assets: list[str]) -> pd.DataFrame:
    """가격 표를 자산 순서로 재정렬하고 공통 거래일 교집합만 남긴다.

    - 컬럼을 assets 순서로 고정.
    - 어느 자산이라도 값이 없는 날(row)은 drop → 5종 모두 거래된 날만.
    - **ffill/bfill 금지**: 결측을 채우면 미래·과거 정보가 새어 룩어헤드가 된다.
    """
    missing = [a for a in assets if a not in close.columns]
    if missing:
        raise ValueError(f"수집 결과에 누락된 자산이 있습니다: {missing}")
    aligned = close[assets].dropna(how="any")
    aligned.index.name = "date"
    return aligned


def _download_once(assets: list[str], start: str, end: str) -> pd.DataFrame:
    """yfinance 1회 호출 → 정렬된 가격 DataFrame. 빈 응답은 빈 DataFrame으로 반환한다."""
    import yfinance as yf

    raw = yf.download(
        tickers=assets,
        start=start,
        end=end,
        auto_adjust=True,   # 배당·분할 조정가 → 총수익 일관성
        progress=False,
        group_by="column",
    )

    # 레이트리밋 시 빈 응답이 온다 → 컬럼 접근 전에 걸러 재시도 대상으로 넘긴다.
    if raw is None or len(raw) == 0:
        return pd.DataFrame(columns=assets)

    # 다중 티커는 컬럼이 MultiIndex((가격종류, 티커)). 종가만 선택.
    close = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw[["Close"]]
    if isinstance(close, pd.Series):  # 단일 티커 방어
        close = close.to_frame()

    return _align(close, assets)


def _download_with_retry(
    assets: list[str],
    start: str,
    end: str,
    *,
    retries: int,
    backoff: float,
    sleep=time.sleep,
) -> pd.DataFrame:
    """지수 백오프로 재시도한다. 빈 응답·예외 모두 재시도 대상(transient 레이트리밋)."""
    last_err = "원인 불명"
    for attempt in range(1, retries + 1):
        try:
            prices = _download_once(assets, start, end)
            if len(prices):
                return prices
            last_err = "빈 응답(0행)"
        except Exception as exc:  # 네트워크·API 변경 등 — 재시도로 흡수 가능한 경우가 많다
            last_err = f"{type(exc).__name__}: {exc}"
        if attempt < retries:
            wait = backoff * (2 ** (attempt - 1))
            print(f"  수집 실패({last_err}) — {wait:.0f}초 후 재시도 {attempt}/{retries}")
            sleep(wait)

    raise ValueError(
        f"yfinance 수집 실패 — {retries}회 시도 모두 실패({last_err}). "
        "CI에서 Yahoo가 러너 IP를 레이트리밋하면 빈 응답이 올 수 있습니다. "
        "기존 캐시는 덮어쓰지 않았습니다."
    )


def fetch_prices(
    assets: list[str],
    start: str,
    end: str,
    raw_dir: str | None = None,
    *,
    min_rows: int = DEFAULT_MIN_ROWS,
    retries: int = DEFAULT_RETRIES,
    backoff: float = DEFAULT_BACKOFF,
    sleep=time.sleep,
) -> pd.DataFrame:
    """yfinance에서 자산별 조정종가를 받아 **검증된** 가격 DataFrame을 반환한다.

    반환: index=date, columns=assets(고정 순서), 값=조정종가.
    재시도로 transient 실패를 흡수하고, `validate_prices` 게이트를 통과한 경우에만
    raw_dir에 캐시한다(나쁜 데이터가 좋은 캐시를 덮어쓰지 못하게).

    Raises
    ------
    ValueError
        재시도 후에도 수집이 비었거나, 품질 검증(행수·결측·자산순서 등)에 실패한 경우.
    """
    prices = _download_with_retry(
        assets, start, end, retries=retries, backoff=backoff, sleep=sleep
    )

    issues = validate_prices(prices, assets, min_rows=min_rows)
    if issues:
        raise ValueError(
            f"수집 데이터 품질 검증 실패({len(prices)}행): "
            + "; ".join(issues)
            + " — 기존 캐시는 덮어쓰지 않았습니다."
        )

    if raw_dir is not None:
        out = Path(raw_dir)
        out.mkdir(parents=True, exist_ok=True)
        prices.to_parquet(out / RAW_FILENAME)

    return prices


def load_raw(raw_dir: str) -> pd.DataFrame:
    """캐시된 원시 가격 Parquet을 읽는다."""
    path = Path(raw_dir) / RAW_FILENAME
    if not path.is_file():
        raise FileNotFoundError(f"원시 데이터 캐시가 없습니다: {path} (먼저 fetch_prices 실행)")
    return pd.read_parquet(path)


def main() -> None:
    """config 기반 수집 실행 — `python -m src.data.collect`."""
    from src.config_loader import get_assets, load_config

    cfg = load_config()
    assets = get_assets(cfg)
    d = cfg["data"]
    prices = fetch_prices(
        assets, d["start"], d["end"], raw_dir=d["raw_dir"],
        min_rows=int(d.get("min_rows", DEFAULT_MIN_ROWS)),
    )
    print(f"수집 완료: {prices.shape[0]}일 × {prices.shape[1]}자산")
    print(f"  기간: {prices.index.min().date()} ~ {prices.index.max().date()}")
    print(f"  결측치: {int(prices.isna().sum().sum())}개")
    print(f"  저장: {Path(d['raw_dir']) / RAW_FILENAME}")


if __name__ == "__main__":
    main()
