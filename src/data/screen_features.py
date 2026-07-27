"""피처 스크리닝 — 대리모델로 피처셋 예측력을 빠르게 비교 (모델 왕복 제거, 이슈 #34).

RL 학습·백테스트는 오래 걸린다. 그 전에 "이 피처셋에 학습 가능한 신호가 있나"를 가벼운
대리모델(sklearn)로 초 단위로 걸러, 통과한 후보만 도현 RL로 넘긴다. 민지↔도현 왕복을
수십 번 → 몇 번으로 줄이는 MLOps 글루.

방법: 자산 피처 + 시장 피처 → 익일수익률을 train에서 대리모델(Ridge·GBM)로 학습하고, test에서
**rank-IC·R²**를 측정한다. 여러 피처셋(컬럼 부분집합)을 fold1~3에서 비교한다. Feature Store를
재빌드하지 않고 기존 187에서 컬럼만 골라 부분집합을 만든다.

⚠️ **대리모델 ≠ RL.** 이것은 "확실히 신호 없는" 피처셋을 빠르게 쳐내는 **사전 필터**(필요조건)다.
대리모델도 못 맞히면 RL은 더 못 한다. 하지만 대리모델이 맞힌다고 RL 성공이 보장되진 않는다 —
**최종 판정은 도현 RL + 백테스트 3지표(샤프·비중편차·회전율)** 다.

실행: `python -m src.data.screen_features`
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.data import schema
from src.data.analyze_features import FOLDS, _aligned, _spearman

# 사전 정의 피처셋 — (포함할 자산지표 목록, 포함할 시장지표 목록).
# no_noise: fold마다 부호가 뒤집히는 4종(RSI·ROC·MA_Cross·Bollinger)을 뺀 것.
FEATURESETS: dict[str, tuple[list[str], list[str]]] = {
    "full": (list(schema.ASSET_FEATURES), list(schema.MARKET_FEATURES)),
    "no_noise": (["MACD_Hist", "Rolling_Vol_20"], list(schema.MARKET_FEATURES)),
    "market_only": ([], list(schema.MARKET_FEATURES)),
    "ebr_only": ([], ["Equity_Bond_Ratio"]),
}


def _xy(out_dir, fold, split, asset_inds, mkt_inds):
    """피처셋 컬럼으로 풀링 (X, y)를 만든다 — (자산, 일) 샘플을 세로로 스택.

    자산 지표는 자산별 컬럼, 시장 지표는 전 자산 공통값(각 자산 샘플에 동일 부착).
    y는 그 자산의 익일수익률.
    """
    feats, tgts = _aligned(out_dir, fold, split)
    xs, ys = [], []
    for asset in schema.ASSETS:
        cols = [f"feat_{asset}_{ind}" for ind in asset_inds] + [f"mkt_{m}" for m in mkt_inds]
        if not cols:
            continue
        xs.append(feats[cols].to_numpy())
        ys.append(tgts[f"fwd_ret_{asset}"].to_numpy())
    return np.vstack(xs), np.concatenate(ys)


def surrogate_score(out_dir: str, asset_inds: list[str], mkt_inds: list[str]) -> dict:
    """fold1~3에서 train fit → test 예측 후 rank-IC·R² 평균(Ridge·GBM)을 낸다."""
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.linear_model import Ridge
    from sklearn.metrics import r2_score

    acc = {"ridge_ic": [], "ridge_r2": [], "gbm_ic": [], "gbm_r2": []}
    for f in FOLDS:
        x_tr, y_tr = _xy(out_dir, f, "train", asset_inds, mkt_inds)
        x_te, y_te = _xy(out_dir, f, "test", asset_inds, mkt_inds)
        models = (
            ("ridge", Ridge(alpha=1.0)),
            ("gbm", GradientBoostingRegressor(n_estimators=100, max_depth=3, random_state=0)),
        )
        for name, model in models:
            model.fit(x_tr, y_tr)
            pred = model.predict(x_te)
            acc[f"{name}_ic"].append(_spearman(pd.Series(pred), pd.Series(y_te)))
            acc[f"{name}_r2"].append(r2_score(y_te, pred))
    return {k: round(float(np.mean(v)), 4) for k, v in acc.items()}


def compare_featuresets(out_dir: str = "data/feature_store", defs: dict | None = None) -> pd.DataFrame:
    """여러 피처셋의 대리모델 예측력을 한 표로 비교한다(fold1~3 test 평균)."""
    defs = defs or FEATURESETS
    rows = {name: surrogate_score(out_dir, a, m) for name, (a, m) in defs.items()}
    return pd.DataFrame(rows).T


def main() -> None:
    pd.set_option("display.width", 120)
    print("=" * 72)
    print("피처 스크리닝 — 대리모델 예측력 (fold1~3 test 평균)")
    print("=" * 72)
    df = compare_featuresets()
    print(df.to_string())
    print(
        "\n  · rank-IC: 예측값이 실제 익일수익률을 순위로 얼마나 맞추나 (0≈무신호)"
        "\n  · R²: 분산 설명력 (음수 = 평균보다도 못 맞힘 = 신호 없음)"
        "\n  ⚠️ 대리모델은 사전 필터일 뿐 — 최종 판정은 도현 RL + 백테스트 3지표"
    )


if __name__ == "__main__":
    main()
