"""피처 예측력·분포 드리프트 분석 — 모델 일반화 실패의 데이터측 규명 (이슈 #34 회의 자료).

모델(PPO)이 fold1~3 OOS에서 전부 1/N에 진다(#43). "학습 부족이 아니라 과적합 + 일반화
실패"(도현 진단)를 **데이터로 뒷받침**한다. 모델·보상은 건드리지 않고, Feature Store만 분석한다.

두 축:
  ① 피처 예측력(IC) — 각 지표가 익일 수익률을 실제로 예측하는가.
     지표값과 fwd_ret의 스피어만 상관(순위 상관이라 z-score 정규화에 불변 → 정규화된 Feature
     Store를 그대로 써도 raw와 동일). fold마다 부호가 뒤집히는 지표 = 국면마다 반대로 작동 = 노이즈.
  ② 분포 드리프트 — train에서 fit한 정규화(μ=0,σ=1)를 test에 적용했을 때, test 통계가 (0,1)에서
     벗어난 정도가 곧 **train→test 분포 이동**이다. 크게 벗어난 지표 = 모델이 학습 때 못 본 분포를
     받는다 = 일반화 실패의 직접 증거.

실행: `python -m src.data.analyze_features`
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.data import feature_store as fs
from src.data import schema

FOLDS = (1, 2, 3)


def _spearman(x: pd.Series, y: pd.Series) -> float:
    """스피어만 상관 = 순위의 피어슨 상관. scipy 없이 계산(pandas 순위 + 피어슨)."""
    return float(x.rank().corr(y.rank()))  # method 기본=pearson, scipy 불필요


def _aligned(out_dir: str, fold: int, split: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(features, targets)를 공통 인덱스로 정렬해 반환. targets는 마지막 행이 없어 교집합을 쓴다."""
    feats = fs.load_features(out_dir, fold, split)
    tgts = fs.load_targets(out_dir, fold, split)
    common = feats.index.intersection(tgts.index)
    return feats.loc[common], tgts.loc[common]


# ── ① 피처 예측력 (IC = 스피어만 상관) ─────────────────────────────
def feature_ic(out_dir: str, fold: int, split: str) -> pd.DataFrame:
    """자산 지표 IC(자산 지표 vs 자기 자산 익일수익률) + 시장 지표 IC(vs SPY 익일수익률).

    반환: index=지표명, columns=자산(+'market'), 값=스피어만 IC.
    """
    feats, tgts = _aligned(out_dir, fold, split)
    rows: dict[str, dict[str, float]] = {ind: {} for ind in schema.ASSET_FEATURES}
    for asset in schema.ASSETS:
        y = tgts[f"fwd_ret_{asset}"]
        for ind in schema.ASSET_FEATURES:
            rows[ind][asset] = _spearman(feats[f"feat_{asset}_{ind}"], y)
    df = pd.DataFrame(rows).T  # index=지표, columns=자산

    # 시장 지표는 자산 비특이 → 대표로 SPY 익일수익률과 상관(주식 국면 프록시)
    y_spy = tgts["fwd_ret_SPY"]
    mkt = {m: _spearman(feats[f"mkt_{m}"], y_spy) for m in schema.MARKET_FEATURES}
    return df, pd.Series(mkt, name="IC_vs_SPY")


def ic_stability(out_dir: str) -> pd.DataFrame:
    """지표별 fold간 IC 안정성 — 평균 |IC|(예측력)와 부호 반전 여부(노이즈성)를 요약.

    자산 평균 IC를 fold1~3 test에서 구하고, 부호가 fold마다 뒤집히면 flagged.
    """
    per_fold = {}
    for f in FOLDS:
        df, _ = feature_ic(out_dir, f, "test")
        per_fold[f] = df.mean(axis=1)  # 지표별 자산 평균 IC
    mat = pd.DataFrame(per_fold)  # index=지표, columns=fold
    signs = np.sign(mat.to_numpy())
    out = pd.DataFrame(
        {
            "mean_abs_IC": mat.abs().mean(axis=1).round(4),
            "IC_fold1": mat[1].round(4),
            "IC_fold2": mat[2].round(4),
            "IC_fold3": mat[3].round(4),
            "sign_flip": [len(set(s[~np.isnan(s)])) > 1 for s in signs],
        },
        index=mat.index,
    )
    return out.sort_values("mean_abs_IC", ascending=False)


# ── ② 분포 드리프트 (test가 train 정규화 기준에서 벗어난 정도) ──────
def distribution_drift(out_dir: str, fold: int) -> pd.DataFrame:
    """지표·시장 컬럼의 train/test 정규화 통계 비교.

    정규화는 train에서 fit(μ=0,σ=1)이므로, test의 mean/std가 (0,1)에서 벗어난 정도가 곧
    train→test 분포 이동이다. |test_mean|·|test_std−1|이 클수록 학습 때 못 본 분포.
    """
    tr = fs.load_features(out_dir, fold, "train")
    te = fs.load_features(out_dir, fold, "test")
    cols = [c for c in tr.columns if c.startswith(("feat_", "mkt_"))]
    return pd.DataFrame(
        {
            "train_mean": tr[cols].mean().round(3),
            "train_std": tr[cols].std().round(3),
            "test_mean": te[cols].mean().round(3),
            "test_std": te[cols].std().round(3),
            "test_min": te[cols].min().round(2),
            "test_max": te[cols].max().round(2),
        }
    )


def drift_summary(out_dir: str) -> pd.DataFrame:
    """fold별 분포 드리프트 크기 요약 — |test_mean| 평균, |test_std−1| 평균, 최대 |z|."""
    rows = {}
    for f in FOLDS:
        d = distribution_drift(out_dir, f)
        rows[f] = {
            "mean_|test_mean|": d["test_mean"].abs().mean().round(3),
            "mean_|test_std-1|": (d["test_std"] - 1).abs().mean().round(3),
            "max_|z|": max(d["test_max"].abs().max(), d["test_min"].abs().max()).round(2),
        }
    return pd.DataFrame(rows).T


# ── 정규화 sanity ─────────────────────────────────────────────────
def normalization_sanity(out_dir: str) -> dict:
    """train 지표 μ≈0·σ≈1, NaN/inf 없음, 극단값 위치를 점검한다."""
    issues = []
    worst = {}
    for f in FOLDS:
        tr = fs.load_features(out_dir, f, "train")
        cols = [c for c in tr.columns if c.startswith(("feat_", "mkt_"))]
        if not np.isfinite(tr[cols].to_numpy()).all():
            issues.append(f"fold{f}: NaN/inf 존재")
        mu = tr[cols].mean().abs().max()
        if mu > 1e-6:
            issues.append(f"fold{f}: train 지표 평균이 0에서 벗어남(max |μ|={mu:.2e})")
        te = fs.load_features(out_dir, f, "test")
        flat = te[cols].abs().stack()
        worst[f] = (flat.idxmax(), round(float(flat.max()), 2))
    return {"issues": issues, "worst_test_extreme": worst}


def main() -> None:
    out_dir = "data/feature_store"
    pd.set_option("display.width", 120)

    print("=" * 70)
    print("① 피처 예측력 (IC = 스피어만 상관, fold1~3 test)")
    print("=" * 70)
    print(ic_stability(out_dir).to_string())
    print("\n  · mean_abs_IC 낮음 = 예측력 약함(노이즈)  · sign_flip=True = 국면마다 반대 작동")

    print("\n" + "=" * 70)
    print("② 분포 드리프트 (test가 train 정규화 기준에서 벗어난 정도)")
    print("=" * 70)
    print(drift_summary(out_dir).to_string())
    print("\n  · |test_mean|이 0, |test_std-1|이 0에서 멀수록 train→test 분포 이동 큼")
    print("  · max_|z|: test에서 학습 분포 기준 몇 σ까지 벗어났나")

    print("\n" + "=" * 70)
    print("정규화 sanity")
    print("=" * 70)
    s = normalization_sanity(out_dir)
    print("이슈:", s["issues"] or "없음 (train μ≈0·σ≈1, NaN/inf 없음)")
    for f, (col, val) in s["worst_test_extreme"].items():
        print(f"  fold{f} test 최대 이탈: {col} = {val}σ")


if __name__ == "__main__":
    main()
