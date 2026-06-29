"""원시 가격 데이터 품질 검증 + EDA 요약.

데이터 엔지니어링의 두 축: ① 데이터를 눈으로 확인(EDA) ② 품질 불변식을 자동 검사로 박제.
- summarize_prices: 사람이 읽는 EDA 요약(통계·분포·상관·이상치). 노트북·콘솔에서 사용.
- validate_prices: 자동 품질 게이트(NaN·음수·미정렬·극단점프·자산순서). 위반 시 이슈 목록 반환.
- main: 실데이터(prices_raw.parquet)에 둘 다 실행.

(모듈명이 stdlib `inspect`와 겹치지 않도록 EDA 요약도 본 모듈에 둔다.)
"""

import numpy as np
import pandas as pd

TRADING_DAYS = 252
EXTREME_LOG_MOVE = 0.25   # 일간 로그수익률 |.|>0.25(≈±28%) → 이상치 후보(분할 미조정 등)
HARD_MAX_LOG_MOVE = 0.4   # |.|>0.4(≈±49%) → 하드 위반 임계


def summarize_prices(df: pd.DataFrame) -> dict:
    """가격 데이터의 EDA 요약을 dict로 반환한다(사람이 읽는 용도)."""
    rets = np.log(df / df.shift(1)).iloc[1:]
    extreme = (rets.abs() > EXTREME_LOG_MOVE).sum()
    # 인접일 가격비(분할 미조정 의심: >1.5 또는 <0.67)
    ratio = df / df.shift(1)
    split_suspect = ((ratio > 1.5) | (ratio < 0.67)).sum()
    gaps = _calendar_gaps(df.index)

    return {
        "shape": df.shape,
        "period": (str(df.index.min().date()), str(df.index.max().date())),
        "n_trading_days": len(df),
        "dtypes": {c: str(t) for c, t in df.dtypes.items()},
        "index_type": type(df.index).__name__,
        "index_sorted": bool(df.index.is_monotonic_increasing),
        "index_unique": bool(df.index.is_unique),
        "calendar_gaps": gaps,
        "price_stats": df.describe().loc[["min", "mean", "max"]].round(2).to_dict(),
        "last_price": df.iloc[-1].round(2).to_dict(),
        "nan_count": int(df.isna().sum().sum()),
        "nonpositive_count": int((df <= 0).sum().sum()),
        "ret_annual_mean": (rets.mean() * TRADING_DAYS).round(4).to_dict(),
        "ret_annual_vol": (rets.std() * np.sqrt(TRADING_DAYS)).round(4).to_dict(),
        "ret_skew": rets.skew().round(3).to_dict(),
        "ret_kurtosis": rets.kurtosis().round(3).to_dict(),
        "extreme_moves": {k: int(v) for k, v in extreme.items()},
        "split_suspect": {k: int(v) for k, v in split_suspect.items()},
        "correlation": rets.corr().round(3).to_dict(),
    }


def _calendar_gaps(index: pd.DatetimeIndex) -> int:
    """영업일(B) 기준 기대 거래일 대비 빠진 날 수(공휴일 포함하므로 참고용)."""
    expected = pd.bdate_range(index.min(), index.max())
    return int(len(expected) - len(index))


def validate_prices(
    df: pd.DataFrame,
    assets: list[str],
    *,
    max_log_move: float = HARD_MAX_LOG_MOVE,
    min_rows: int = 252,
) -> list[str]:
    """품질 불변식을 검사하고 위반 목록을 반환한다(빈 목록 = 통과).

    검사: 자산 컬럼·순서, 인덱스 정렬·중복, NaN, 가격>0, 극단 일간변동, 최소 행수.
    """
    issues: list[str] = []

    if list(df.columns) != list(assets):
        issues.append(f"자산 컬럼·순서 불일치: {list(df.columns)} != {list(assets)}")
    if not df.index.is_monotonic_increasing:
        issues.append("날짜 인덱스가 정렬되어 있지 않음")
    if not df.index.is_unique:
        issues.append("날짜 인덱스에 중복이 있음")

    nan = int(df.isna().sum().sum())
    if nan:
        issues.append(f"결측치 {nan}개 존재 (ffill 금지 — 정렬·교집합 점검 필요)")

    nonpos = int((df <= 0).sum().sum())
    if nonpos:
        issues.append(f"0 이하 가격 {nonpos}개 존재")

    if len(df) < min_rows:
        issues.append(f"행 수 부족: {len(df)} < {min_rows}")

    if nonpos == 0 and not df.isna().all().all():
        with np.errstate(invalid="ignore", divide="ignore"):
            rets = np.log(df / df.shift(1)).iloc[1:]
        extreme = int((rets.abs() > max_log_move).sum().sum())
        if extreme:
            issues.append(f"극단 일간 로그수익률(|.|>{max_log_move}) {extreme}건 — 분할 미조정·이상치 점검")

    return issues


def main() -> None:
    """실데이터 로드 → EDA 요약 출력 → 검증 리포트. `python -m src.data.validate`."""
    from src.config_loader import get_assets, load_config
    from src.data.collect import load_raw

    cfg = load_config()
    assets = get_assets(cfg)
    df = load_raw(cfg["data"]["raw_dir"])

    s = summarize_prices(df)
    print("=" * 60)
    print("원시 가격 EDA 요약")
    print("=" * 60)
    print(f"기간: {s['period'][0]} ~ {s['period'][1]} | 거래일 {s['n_trading_days']}일 | shape {s['shape']}")
    print(f"인덱스 정렬={s['index_sorted']} 고유={s['index_unique']} 영업일대비 갭(공휴일포함)={s['calendar_gaps']}")
    print(f"결측 {s['nan_count']} | 0이하가격 {s['nonpositive_count']}")
    print(f"마지막 가격: {s['last_price']}")
    print(f"연율 수익률: {s['ret_annual_mean']}")
    print(f"연율 변동성: {s['ret_annual_vol']}")
    print(f"왜도: {s['ret_skew']}")
    print(f"첨도: {s['ret_kurtosis']}")
    print(f"극단 이동(|logret|>{EXTREME_LOG_MOVE}) 건수: {s['extreme_moves']}")
    print(f"분할 미조정 의심: {s['split_suspect']}")
    print("상관행렬:")
    print(pd.DataFrame(s["correlation"]).round(2).to_string())

    print("\n" + "=" * 60)
    print("품질 검증")
    print("=" * 60)
    issues = validate_prices(df, assets)
    if issues:
        print(f"⚠️ 이슈 {len(issues)}건:")
        for i in issues:
            print(f"  - {i}")
    else:
        print("✅ 모든 품질 검사 통과")


if __name__ == "__main__":
    main()
