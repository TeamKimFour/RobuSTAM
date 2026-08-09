"""M0~M3 콤보 스크리닝 — 콤보 전용 Feature Store로 예측력·분포를 측정하고 MLflow에 기록.

`screen_features.py`는 이미 빌드된 `full`(187) Feature Store에서 컬럼만 골라 대리모델을
돌린다. 하지만 `M2`/`M3`가 쓰는 `RSI_28`·`Drawdown`은 `full`에 아예 없는 신규 지표라(콤보
전용 빌드에만 존재) 그 방식으로는 스크리닝할 수 없다. 이 모듈은 `src.data.build`가 콤보별로
따로 만든 자기 Feature Store(`data/feature_store/<combo>/`)를 대상으로 삼는다.

측정 항목(팀 주간계획 "4순위: MLflow 스크리닝 기록 연결" 스펙):
  - Parameters: run_type, feature_set, state_dim, feature_store_run_id, config_hash,
    asset_features, market_features, drawdown_lookback, git_commit
  - Metrics: mean_abs_ic, ridge_rank_ic, ridge_r2, gbm_rank_ic, gbm_r2, max_abs_z
  - Artifacts: feature_manifest.json, resolved_config.yaml, screening_result.csv,
    distribution_report.csv

MLflow tracking uri는 config `model.mlflow_tracking_uri`(기본 로컬 파일스토어 `mlruns`)를
그대로 쓴다 — Docker/팀 공용 DB 없이도 로컬에서 즉시 기록된다.

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
from src.data.analyze_features import FOLDS, _aligned, _spearman, drift_summary
from src.data.screen_features import surrogate_score

COMBOS = ("M0", "M1", "M2", "M3")


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parents[2]
        ).decode().strip()
    except Exception:
        return "unknown"


def _mean_abs_ic(out_dir: str, asset_feats: list[str], market_feats: list[str]) -> tuple[float, pd.DataFrame]:
    """개별 지표 스피어만 IC(fold1~3 test 평균 |IC|)와 세부 표를 반환한다.

    자산 지표는 자기 자산 익일수익률, 시장 지표는 SPY 익일수익률과 상관(analyze_features와 동일 관례).
    """
    rows: dict[str, list[float]] = {name: [] for name in (*asset_feats, *market_feats)}
    for f in FOLDS:
        feats, tgts = _aligned(out_dir, f, "test")
        for ind in asset_feats:
            ics = [_spearman(feats[f"feat_{a}_{ind}"], tgts[f"fwd_ret_{a}"]) for a in schema.ASSETS]
            rows[ind].append(float(np.nanmean(ics)))
        y_spy = tgts["fwd_ret_SPY"]
        for m in market_feats:
            rows[m].append(_spearman(feats[f"mkt_{m}"], y_spy))

    detail = pd.DataFrame(rows, index=[f"fold{f}" for f in FOLDS]).T
    detail["mean_abs_IC"] = detail.abs().mean(axis=1)
    mean_abs_ic = float(detail["mean_abs_IC"].mean()) if len(detail) else float("nan")
    return mean_abs_ic, detail


def screen_combo(cfg: dict, combo: str) -> dict:
    """콤보 하나를 스크리닝해 MLflow 로깅에 필요한 모든 것을 dict로 반환한다."""
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
    mean_abs_ic, ic_detail = _mean_abs_ic(out_dir, asset_feats, market_feats)
    drift = drift_summary(out_dir)
    max_abs_z = float(drift["max_|z|"].max()) if len(drift) else float("nan")

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
        },
        "metrics": {
            "mean_abs_ic": mean_abs_ic,
            "ridge_rank_ic": surrogate["ridge_ic"],
            "ridge_r2": surrogate["ridge_r2"],
            "gbm_rank_ic": surrogate["gbm_ic"],
            "gbm_r2": surrogate["gbm_r2"],
            "max_abs_z": max_abs_z,
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
    mlflow.set_experiment(model_cfg.get("mlflow_experiment", "robustam-ppo"))

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
    """M0~M3 전부 스크리닝하고 MLflow에 기록한 뒤 요약 표를 반환한다."""
    cfg = load_config(config_path)
    rows = {}
    for combo in COMBOS:
        result = screen_combo(cfg, combo)
        mlflow_run_id = log_to_mlflow(cfg, result)
        rows[combo] = {**result["metrics"], "mlflow_run_id": mlflow_run_id}
    return pd.DataFrame(rows).T


def main() -> None:
    pd.set_option("display.width", 120)
    print("=" * 90)
    print("M0~M3 콤보 스크리닝 — 콤보 전용 Feature Store, MLflow(run_type=screening) 기록")
    print("=" * 90)
    df = screen_all()
    print(df.to_string())
    print(
        "\n  · ridge_rank_ic/gbm_rank_ic: 대리모델 예측 순위와 실제 익일수익률 순위의 상관"
        "\n  · mean_abs_ic: 개별 지표(자산 자기IC·시장 SPY IC) |스피어만| 평균"
        "\n  · max_abs_z: test가 train 정규화 기준(μ=0,σ=1)에서 벗어난 최대 표준편차"
        "\n  ⚠️ 대리모델·IC는 사전 필터일 뿐 — 최종 판정은 도현 RL + 백테스트 3지표"
    )


if __name__ == "__main__":
    main()
