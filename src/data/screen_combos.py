"""Baseline·M0~M3 콤보 스크리닝 — 콤보 전용 Feature Store로 예측력·분포를 측정하고 MLflow에 기록.

`screen_features.py`는 이미 빌드된 `full`(187) Feature Store에서 컬럼만 골라 대리모델을
돌린다. 하지만 `M2`/`M3`가 쓰는 `RSI_28`·`Drawdown`은 `full`에 아예 없는 신규 지표라(콤보
전용 빌드에만 존재) 그 방식으로는 스크리닝할 수 없다. 이 모듈은 `src.data.build`가 콤보별로
따로 만든 자기 Feature Store(`data/feature_store/<combo>/`)를 대상으로 삼는다.

측정 항목(팀 주간계획 "3순위: 정확한 조합 스크리닝"·"4순위: MLflow 스크리닝 기록 연결" 스펙):
  - Baseline(`full`, 기존 6+2)을 먼저 스크리닝해 M0~M3 판정의 기준값으로 삼는다.
  - Parameters: run_type, feature_set, state_dim, feature_store_run_id, config_hash,
    asset_features, market_features, drawdown_lookback, git_commit, verdict(PASS/FAIL/BASELINE)
  - Metrics: mean_abs_ic, sign_stable_frac, n_unstable_features, ridge_rank_ic, ridge_r2,
    gbm_rank_ic, gbm_r2, max_abs_z, n_distribution_issues, passed
  - Artifacts: feature_manifest.json, resolved_config.yaml,
    screening_result.csv(fold별 지표 IC + 부호 안정성), distribution_report.csv(fold별
    분포 드리프트 + 최악 극단값 컬럼)

통과 판정: **Ridge 또는 GBM rank-IC가 Baseline 이상**이면 PASS(PR #69 도현 리뷰 반영).
`sign_stable_frac`(부호 안정 지표 비율)은 콤보마다 지표 개수가 달라 분모가 다르다 —
지표 1개짜리 콤보(M0)는 0.0/1.0밖에 못 나오는 등 콤보 간 직접 비교가 안 되므로 게이트에서
뺐다. 참고 지표로만 남기고, 콤보 크기에 무관한 절대량인 `n_unstable_features`(부호가
불안정한 지표 개수)를 함께 기록한다. ridge_rank_ic/gbm_rank_ic는 같은 fold test 구간·같은
타깃에 대한 결합 예측력이라 콤보 간 직접 비교가 가능하다. 대리모델 값 자체가 작다는 한계는
여전하다(`docs/feature_candidates.md` §5-1).

MLflow tracking uri는 config `model.mlflow_tracking_uri`(기본 로컬 파일스토어 `mlruns`)를
그대로 쓴다 — Docker/팀 공용 DB 없이도 로컬에서 즉시 기록된다. RL 학습 run(`robustam-ppo`)과
섞이지 않도록 별도 experiment(`model.mlflow_screening_experiment`, 기본 `robustam-screening`)에
기록한다.

실행: `python -m src.data.screen_combos`
"""

from __future__ import annotations

import json
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from src.config_loader import (
    get_asset_features,
    get_feature_store_dir,
    get_market_features,
    get_meta_db,
    get_state_dim,
    get_window,
    load_config,
    resolve_combo,
)
from src.data import feature_store as fs
from src.data import schema
from src.data.analyze_features import FOLDS, _aligned, _spearman, drift_summary, normalization_sanity
from src.data.screen_features import surrogate_score

# full(기존 6+2)을 Baseline으로 먼저 스크리닝하고, 그 mean_abs_ic를 기준으로 M0~M3의
# 통과 여부를 판정한다(팀 주간계획 "3순위": Baseline·M1·M2·M3 비교).
BASELINE_COMBO = "full"
COMBOS = (BASELINE_COMBO, "M0", "M1", "M2", "M3")


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parents[2]
        ).decode().strip()
    except Exception:
        return "unknown"


def _mean_abs_ic(
    out_dir: str, asset_feats: list[str], market_feats: list[str]
) -> tuple[float, float, pd.DataFrame]:
    """개별 지표 스피어만 IC(fold1~3 test)·부호 안정성·mean|IC|를 계산한다.

    자산 지표는 자기 자산 익일수익률, 시장 지표는 SPY 익일수익률과 상관(analyze_features와 동일 관례).
    반환: (전체 mean|IC|, 부호 안정 지표 비율[0~1], fold별 IC·mean_abs_IC·sign_stable 세부 표).
    """
    fold_cols = [f"fold{f}" for f in FOLDS]
    rows: dict[str, list[float]] = {name: [] for name in (*asset_feats, *market_feats)}
    for f in FOLDS:
        feats, tgts = _aligned(out_dir, f, "test")
        for ind in asset_feats:
            ics = [_spearman(feats[f"feat_{a}_{ind}"], tgts[f"fwd_ret_{a}"]) for a in schema.ASSETS]
            rows[ind].append(float(np.nanmean(ics)))
        y_spy = tgts["fwd_ret_SPY"]
        for m in market_feats:
            rows[m].append(_spearman(feats[f"mkt_{m}"], y_spy))

    detail = pd.DataFrame(rows, index=fold_cols).T
    detail["mean_abs_IC"] = detail.abs().mean(axis=1)
    # 부호 안정성: fold마다 부호가 같으면(NaN 제외) 안정 — ic_stability(analyze_features)와 동일 관례.
    signs = np.sign(detail[fold_cols].to_numpy())
    detail["sign_stable"] = [len({s for s in row if not np.isnan(s)}) <= 1 for row in signs]

    mean_abs_ic = float(detail["mean_abs_IC"].mean()) if len(detail) else float("nan")
    sign_stable_frac = float(detail["sign_stable"].mean()) if len(detail) else float("nan")
    return mean_abs_ic, sign_stable_frac, detail


def screen_combo(cfg: dict, combo: str, baseline_metrics: dict | None = None) -> dict:
    """콤보 하나를 스크리닝해 MLflow 로깅에 필요한 모든 것을 dict로 반환한다.

    baseline_metrics(Baseline의 metrics dict)를 주면 그 ridge_rank_ic/gbm_rank_ic 대비
    통과 여부를 판정한다. 생략(Baseline 자기 자신을 스크리닝할 때)하면 판정하지 않고
    "BASELINE"으로 표시한다.
    """
    resolved = resolve_combo(cfg, combo)
    out_dir = get_feature_store_dir(cfg, combo)
    db = get_meta_db(cfg, combo)
    asset_feats = get_asset_features(resolved)
    market_feats = get_market_features(resolved)
    W = get_window(resolved)
    config_hash = fs.config_hash(resolved)
    run_id = fs.latest_run_id_for_config(db, config_hash)
    if run_id is None:
        raise RuntimeError(
            f"콤보 '{combo}'의 build가 없습니다 — 먼저 build(combo='{combo}')를 실행하세요."
        )

    surrogate = surrogate_score(out_dir, asset_feats, market_feats)
    mean_abs_ic, sign_stable_frac, ic_detail = _mean_abs_ic(out_dir, asset_feats, market_feats)
    drift = drift_summary(out_dir)
    max_abs_z = float(drift["max_|z|"].max()) if len(drift) else float("nan")

    # 결측치·극단값(초기 warm-up NaN 잔존 여부·test 최악 이탈치) — analyze_features 재사용.
    # worst_test_extreme[fold] = ((날짜, 컬럼명), z) — stack()된 (row,col) MultiIndex의 idxmax.
    sanity = normalization_sanity(out_dir)
    n_issues = len(sanity["issues"])
    for f, ((worst_date, worst_col), val) in sanity["worst_test_extreme"].items():
        drift.loc[f, "worst_extreme_col"] = worst_col
        drift.loc[f, "worst_extreme_date"] = str(worst_date)
        drift.loc[f, "worst_extreme_z"] = val

    n_unstable_features = int((~ic_detail["sign_stable"]).sum()) if len(ic_detail) else 0

    # 최종 통과 여부: Ridge 또는 GBM rank-IC가 Baseline(full) 이상이면 PASS.
    # sign_stable_frac은 콤보마다 분모(지표 개수)가 달라 게이트로 쓰지 않는다(PR #69 리뷰).
    # Baseline 본인은 비교 대상이 없으므로 판정하지 않는다.
    if baseline_metrics is None:
        verdict, passed = "BASELINE", 1.0
    else:
        passed_bool = (
            surrogate["ridge_ic"] >= baseline_metrics["ridge_rank_ic"]
            or surrogate["gbm_ic"] >= baseline_metrics["gbm_rank_ic"]
        )
        verdict, passed = ("PASS" if passed_bool else "FAIL"), float(passed_bool)

    return {
        "combo": combo,
        "params": {
            "run_type": "screening",
            "feature_set": combo,
            "state_dim": get_state_dim(resolved),
            "feature_store_run_id": run_id,
            "config_hash": config_hash,
            "asset_features": ",".join(asset_feats) or "(none)",
            "market_features": ",".join(market_feats),
            "drawdown_lookback": resolved["features"]["params"].get("drawdown_lookback"),
            "git_commit": _git_commit(),
            "verdict": verdict,
            "verdict_rule": "ridge_rank_ic>=baseline(full) OR gbm_rank_ic>=baseline(full)",
        },
        "metrics": {
            "mean_abs_ic": mean_abs_ic,
            "sign_stable_frac": sign_stable_frac,
            "n_unstable_features": float(n_unstable_features),
            "ridge_rank_ic": surrogate["ridge_ic"],
            "ridge_r2": surrogate["ridge_r2"],
            "gbm_rank_ic": surrogate["gbm_ic"],
            "gbm_r2": surrogate["gbm_r2"],
            "max_abs_z": max_abs_z,
            "n_distribution_issues": float(n_issues),
            "passed": passed,
        },
        "artifacts": {
            "feature_manifest": schema.feature_names(W, asset_feats, market_feats),
            "resolved_features": resolved["features"],
            "ic_detail": ic_detail,
            "distribution_report": drift,
        },
    }


def log_to_mlflow(cfg: dict, result: dict) -> str:
    """screen_combo() 결과 하나를 MLflow run으로 기록한다. run_id를 반환."""
    import os

    # mlflow-skinny 3.14+에서 순수 파일 트래킹 스토어("mlruns/")가 기본 비활성(유지보수
    # 모드)이라 명시적으로 허용한다. train.py/select.py와 동일한 임시 조치(DB 백엔드 이전까지).
    os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")

    import mlflow
    import yaml

    model_cfg = cfg.get("model", {})
    tracking_uri = model_cfg.get("mlflow_tracking_uri", "mlruns")
    # 상대경로면 file:// URI로 정규화(train.py._to_tracking_uri와 동일 규칙) — 어느 cwd에서
    # 실행해도 같은 mlruns/를 가리키고, scheme 없는 경로가 URI로 오인되는 것도 방지한다.
    if "://" not in tracking_uri:
        tracking_uri = Path(tracking_uri).resolve().as_uri()
    mlflow.set_tracking_uri(tracking_uri)
    # RL 학습 run(robustam-ppo)과 분리된 별도 experiment — select.py는 valid_sharpe 유무로
    # 걸러 섞여도 깨지진 않지만, UI 조회 편의를 위해 처음부터 분리한다(PR #69 리뷰).
    mlflow.set_experiment(model_cfg.get("mlflow_screening_experiment", "robustam-screening"))

    with mlflow.start_run(run_name=f"screening-{result['combo']}") as run:
        mlflow.log_params(result["params"])
        mlflow.log_metrics(result["metrics"])

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            (tmp_path / "feature_manifest.json").write_text(
                json.dumps(result["artifacts"]["feature_manifest"], ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            (tmp_path / "resolved_config.yaml").write_text(
                yaml.dump(result["artifacts"]["resolved_features"], allow_unicode=True),
                encoding="utf-8",
            )
            result["artifacts"]["ic_detail"].to_csv(tmp_path / "screening_result.csv")
            result["artifacts"]["distribution_report"].to_csv(tmp_path / "distribution_report.csv")
            mlflow.log_artifacts(str(tmp_path))

        return run.info.run_id


def screen_all(config_path: str = "config/config.yaml") -> pd.DataFrame:
    """Baseline(full)·M0~M3 전부 스크리닝하고 MLflow에 기록한 뒤 요약 표를 반환한다.

    Baseline을 먼저 돌려 그 ridge/gbm rank-IC를 기준값으로 나머지 콤보의 통과 여부를 판정한다.
    """
    cfg = load_config(config_path)

    baseline_result = screen_combo(cfg, BASELINE_COMBO)

    rows = {}
    for combo in COMBOS:
        result = (
            baseline_result if combo == BASELINE_COMBO
            else screen_combo(cfg, combo, baseline_metrics=baseline_result["metrics"])
        )
        mlflow_run_id = log_to_mlflow(cfg, result)
        rows[combo] = {
            **result["metrics"],
            "verdict": result["params"]["verdict"],
            "mlflow_run_id": mlflow_run_id,
        }
    return pd.DataFrame(rows).T


def main() -> None:
    pd.set_option("display.width", 140)
    print("=" * 100)
    print("Baseline(full)·M0~M3 콤보 스크리닝 — 콤보 전용 Feature Store, MLflow(run_type=screening) 기록")
    print("=" * 100)
    df = screen_all()
    print(df.to_string())
    print(
        "\n  · ridge_rank_ic/gbm_rank_ic: 대리모델 예측 순위와 실제 익일수익률 순위의 상관(콤보 간 직접 비교 가능)"
        "\n  · mean_abs_ic/sign_stable_frac: 개별 지표 |스피어만| 평균·부호 안정 비율(콤보마다 분모가 달라 참고용)"
        "\n  · n_unstable_features: fold1~3에서 부호가 뒤집힌 지표 개수(콤보 크기 무관, 절대량)"
        "\n  · max_abs_z: test가 train 정규화 기준(μ=0,σ=1)에서 벗어난 최대 표준편차"
        "\n  · verdict: PASS = ridge_rank_ic 또는 gbm_rank_ic가 Baseline(full) 이상"
        "\n  ⚠️ 대리모델·IC는 사전 필터일 뿐 — 최종 판정은 도현 RL + 백테스트 3지표"
    )


if __name__ == "__main__":
    main()
