"""yfinance 원시 일봉 수집 모듈.

5종 자산의 조정종가(auto_adjust)를 받아 **공통 거래일 교집합**으로 정렬한다.
결측은 drop하며 **ffill로 미래 값을 끌어오지 않는다**(룩어헤드 1차 방어선).
산출물은 data/raw/prices_raw.parquet에 캐시한다(yfinance 비결정성·재다운로드 비용 회피).
"""

from pathlib import Path

import pandas as pd

RAW_FILENAME = "prices_raw.parquet"


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


def fetch_prices(
    assets: list[str],
    start: str,
    end: str,
    raw_dir: str | None = None,
) -> pd.DataFrame:
    """yfinance에서 자산별 조정종가를 받아 정렬된 가격 DataFrame을 반환한다.

    반환: index=date, columns=assets(고정 순서), 값=조정종가.
    raw_dir이 주어지면 Parquet으로 캐시한다.
    """
    import yfinance as yf

    raw = yf.download(
        tickers=assets,
        start=start,
        end=end,
        auto_adjust=True,   # 배당·분할 조정가 → 총수익 일관성
        progress=False,
        group_by="column",
    )

    # 다중 티커는 컬럼이 MultiIndex((가격종류, 티커)). 종가만 선택.
    close = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw[["Close"]]
    if isinstance(close, pd.Series):  # 단일 티커 방어
        close = close.to_frame()

    prices = _align(close, assets)

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
    prices = fetch_prices(assets, d["start"], d["end"], raw_dir=d["raw_dir"])
    print(f"수집 완료: {prices.shape[0]}일 × {prices.shape[1]}자산")
    print(f"  기간: {prices.index.min().date()} ~ {prices.index.max().date()}")
    print(f"  결측치: {int(prices.isna().sum().sum())}개")
    print(f"  저장: {Path(d['raw_dir']) / RAW_FILENAME}")


if __name__ == "__main__":
    main()
