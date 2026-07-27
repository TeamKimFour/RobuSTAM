"""새 피처 후보 탐색 스크리닝 — 원시가격에서 후보를 만들어 fold별 test IC로 평가 (이슈 #34).

기존 스크리닝(`screen_features.py`)은 Feature Store의 187 컬럼에서 부분집합만 봤다. 이 모듈은
**아직 State에 없는 새 피처 후보**를 원시가격에서 직접 계산해 예측력(스피어만 IC)을 본다.
Feature Store를 건드리지 않는 **탐색용 샌드박스**다 — 통과한 후보만 회의 합의 후 §2에 정식 반영한다.

세 방향(회의 결정):
  ① 자산간 상대강도 — EBR(SPY/TLT)처럼 자산 쌍 비율의 이동평균 대비 편차. EBR이 유일 안정 신호였음.
  ② 국면(regime) — 변동성 레짐·추세·낙폭. 22σ 분포 드리프트 대응.
  ③ 기존 지표 파라미터 튜닝 — RSI/ROC lookback을 바꿔 IC가 오르는지.

룩어헤드: 모든 후보는 **인과적**(과거만 참조), 타깃은 익일수익률(fwd_ret[t]=logret[t+1]).
IC는 각 fold **test 구간**(OOS)에서 측정하고 fold간 부호 안정성을 함께 본다.

실행: `python -m src.data.screen_candidates`
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.config_loader import load_config
from src.data.collect import load_raw
from src.data.returns import log_returns


def _spearman(x: pd.Series, y: pd.Series) -> float:
    """스피어만 상관 = 순위의 피어슨 상관 (scipy 불필요)."""
    return float(x.rank().corr(y.rank()))


def _candidates(close: pd.DataFrame, logret: pd.DataFrame) -> dict[str, tuple[pd.Series, str]]:
    """후보 {이름: (시계열, 타깃자산)} — 전부 인과적(과거만). 타깃=예측 대상 자산의 익일수익률."""
    import pandas_ta as ta

    c = close
    out: dict[str, tuple[pd.Series, str]] = {}

    # ── ① 자산간 상대강도 (비율의 20일 이평 대비 편차) — EBR과 같은 산식 계열 ──
    def rs(a: str, b: str) -> pd.Series:
        r = c[a] / c[b]
        return r / r.rolling(20).mean() - 1.0

    out["RS_SPY_GLD"] = (rs("SPY", "GLD"), "SPY")   # 주식/금
    out["RS_SPY_TLT"] = (rs("SPY", "TLT"), "SPY")   # 주식/채권 (=EBR, 새니티)
    out["RS_GLD_TLT"] = (rs("GLD", "TLT"), "GLD")   # 금/채권
    out["RS_EWY_SPY"] = (rs("EWY", "SPY"), "EWY")   # 한국/미국

    # ── ② 국면(regime) 신호 ──
    vs = logret["SPY"].rolling(20).std() / logret["SPY"].rolling(60).std()
    out["Regime_Vol_SPY"] = (vs, "SPY")                                  # 단기/장기 변동성비
    out["Regime_Trend_SPY"] = (c["SPY"] / c["SPY"].rolling(50).mean() - 1.0, "SPY")  # 50일 추세
    out["Regime_Drawdown_SPY"] = (c["SPY"] / c["SPY"].rolling(60).max() - 1.0, "SPY")  # 최근고점 대비 낙폭

    # ── ③ 기존 지표 파라미터 튜닝 (SPY 기준, RSI_14·ROC_10과 비교) ──
    out["RSI_SPY_7"] = (ta.rsi(c["SPY"], length=7), "SPY")
    out["RSI_SPY_28"] = (ta.rsi(c["SPY"], length=28), "SPY")
    out["ROC_SPY_20"] = (ta.roc(c["SPY"], length=20), "SPY")
    out["ROC_SPY_60"] = (ta.roc(c["SPY"], length=60), "SPY")

    # ── 대조군 (기존 지표 재계산 — 새 후보와 같은 잣대로 비교) ──
    out["baseline_EBR"] = (rs("SPY", "TLT"), "SPY")  # RS_SPY_TLT와 동일(대조 표기용)
    out["baseline_RSI_14"] = (ta.rsi(c["SPY"], length=14), "SPY")
    return out


def screen(out_dir_cfg: dict | None = None) -> pd.DataFrame:
    """후보별 fold1~3 test IC와 부호 안정성을 표로 반환한다."""
    cfg = out_dir_cfg or load_config()
    prices = load_raw(cfg["data"]["raw_dir"])
    logret = log_returns(prices)
    close = prices.loc[logret.index]
    fwd = logret.shift(-1)  # fwd_ret[t] = logret[t+1]
    blocks = [(pd.Timestamp(s), pd.Timestamp(e)) for s, e in cfg["split"]["test_blocks"]]

    rows = {}
    for name, (series, asset) in _candidates(close, logret).items():
        y = fwd[asset]
        ics = []
        for s, e in blocks:
            mask = (series.index >= s) & (series.index <= e)
            f = series[mask]
            common = f.dropna().index.intersection(y.dropna().index)
            ics.append(_spearman(f.loc[common], y.loc[common]) if len(common) > 20 else np.nan)
        signs = {np.sign(v) for v in ics if not np.isnan(v)}
        rows[name] = {
            "mean_abs_IC": round(float(np.nanmean(np.abs(ics))), 4),
            "IC_f1": round(ics[0], 3), "IC_f2": round(ics[1], 3), "IC_f3": round(ics[2], 3),
            "sign_stable": len(signs) == 1,
        }
    df = pd.DataFrame(rows).T
    return df.sort_values("mean_abs_IC", ascending=False)


def main() -> None:
    pd.set_option("display.width", 120)
    print("=" * 78)
    print("새 피처 후보 스크리닝 — fold1~3 test IC (원시가격, State 미변경 탐색)")
    print("=" * 78)
    df = screen()
    print(df.to_string())
    print(
        "\n  · 기존 지표 IC 참고: RSI_14~0.06, EBR~-0.10 (모두 약함)"
        "\n  · 유망 = mean_abs_IC 높고 sign_stable=True (fold마다 같은 방향)"
        "\n  ⚠️ IC는 사전 필터 — 통과 후보만 §2 회의 합의 → 정식 반영 → 도현 RL 최종 판정"
    )


if __name__ == "__main__":
    main()
