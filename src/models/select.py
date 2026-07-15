"""배포 policy 선택 — MLflow 기록에서 valid 지표가 가장 좋은 학습 run을 고른다.

`train.py`가 매 학습을 독립 MLflow run으로 쌓아두면(파라미터·valid 지표·provenance),
이 모듈이 그중 배포할 policy 하나를 valid 지표로 골라 `config.inference`에 넣을 값
(model_path·scaler_run_id·fold_id·model_version)을 뽑아준다.

기본 선택 지표는 `valid_sharpe`(위험조정수익) — 단순 누적수익 대신 위험을 반영해 고른다는
프로젝트 철학(CLAUDE.md §1). 정식 성과검증은 찬휘 백테스트 엔진 몫이고, 여기 Sharpe는
후보 policy 간 방향성 선별용 근사다(train.evaluate가 기록).

모델 파일 경로는 train.py의 파일명 규약
    ppo_fold{fold_id}_{feature_store_run_id}_{mlflow_run_id}.zip
을 그대로 재구성한다(아티팩트 다운로드 없이 결정적으로 산출).

무거운 의존(mlflow)은 실제 조회를 하는 함수 안에서만 import한다(train.py와 동일 방침).

실행:
    python -m src.models.select                       # 기본: valid_sharpe 최고 run
    python -m src.models.select --metric valid_total_log_return --fold-id 1
"""

from __future__ import annotations

import argparse
from pathlib import Path

from src.config_loader import load_config
from src.models.train import _to_tracking_uri

# 배포 후보를 고를 기본 지표 — evaluate가 기록하는 valid 키 중 위험조정수익.
DEFAULT_METRIC = "valid_sharpe"


def select_best_run(
    cfg: dict,
    *,
    metric: str = DEFAULT_METRIC,
    fold_id: int | None = None,
    higher_is_better: bool = True,
    experiment: str | None = None,
) -> dict:
    """MLflow 실험에서 `metric`이 가장 좋은 FINISHED run의 배포 정보를 반환한다.

    Parameters
    ----------
    cfg : config.yaml 사전(model 섹션에서 tracking_uri·experiment·model_dir을 읽는다).
    metric : 선택 기준 valid 지표 이름(기본 `valid_sharpe`).
    fold_id : 지정 시 해당 fold로 학습된 run만 후보로 본다(None이면 전체).
    higher_is_better : 지표가 클수록 좋은지(Sharpe·수익=True, 비용류=False).
    experiment : MLflow 실험명(None이면 config `model.mlflow_experiment`).

    Returns
    -------
    dict — mlflow_run_id·feature_store_run_id·fold_id·model_path·model_version·
           metric·metric_value. 이 값을 `config.inference`에 넣으면 배포된다.

    Raises
    ------
    ValueError — 실험이 없거나, 해당 지표를 가진 FINISHED run이 하나도 없을 때.
    """
    from mlflow.tracking import MlflowClient

    model_cfg = cfg.get("model", {})
    tracking_uri = _to_tracking_uri(model_cfg.get("mlflow_tracking_uri", "mlruns"))
    experiment = experiment or model_cfg.get("mlflow_experiment", "robustam-ppo")

    client = MlflowClient(tracking_uri=tracking_uri)
    exp = client.get_experiment_by_name(experiment)
    if exp is None:
        raise ValueError(
            f"MLflow 실험을 찾을 수 없습니다: {experiment!r} (tracking_uri={tracking_uri}). "
            "먼저 python -m src.models.train 으로 학습 run을 쌓아야 합니다."
        )

    filters = ["attributes.status = 'FINISHED'"]
    if fold_id is not None:
        filters.append(f"params.fold_id = '{fold_id}'")
    runs = client.search_runs([exp.experiment_id], filter_string=" and ".join(filters))

    # order_by는 지표 없는 run을 NaN으로 섞을 수 있어 Python에서 명시적으로 필터·비교한다.
    candidates = [r for r in runs if metric in r.data.metrics]
    if not candidates:
        raise ValueError(
            f"'{metric}' 지표를 가진 FINISHED run이 없습니다 "
            f"(experiment={experiment!r}, fold_id={fold_id}). "
            "train.py가 해당 지표를 기록했는지 확인하세요."
        )
    pick = max if higher_is_better else min
    best = pick(candidates, key=lambda r: r.data.metrics[metric])

    fs_run_id = best.data.params.get("feature_store_run_id", "nofs")
    best_fold = int(best.data.params["fold_id"])
    mlflow_run_id = best.info.run_id
    model_dir = model_cfg.get("model_dir", "mlruns/models")
    model_path = str(
        Path(model_dir) / f"ppo_fold{best_fold}_{fs_run_id}_{mlflow_run_id}.zip"
    )
    return {
        "mlflow_run_id": mlflow_run_id,
        "feature_store_run_id": fs_run_id,
        "fold_id": best_fold,
        "model_path": model_path,
        "model_version": f"ppo-fold{best_fold}-{mlflow_run_id[:8]}",
        "metric": metric,
        "metric_value": float(best.data.metrics[metric]),
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RobuSTAM 배포 policy 선택")
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--metric", default=DEFAULT_METRIC, help="선택 기준 valid 지표")
    parser.add_argument("--fold-id", type=int, default=None, help="특정 fold만 후보로")
    parser.add_argument("--experiment", default=None, help="MLflow 실험명 오버라이드")
    parser.add_argument(
        "--lower-is-better",
        action="store_true",
        help="지표가 작을수록 좋은 경우(비용류) 지정",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    cfg = load_config(args.config)
    best = select_best_run(
        cfg,
        metric=args.metric,
        fold_id=args.fold_id,
        higher_is_better=not args.lower_is_better,
    )
    print(best)
    if best["feature_store_run_id"] == "nofs":
        print(
            "\n[경고] 선택된 run에 build run_id(provenance)가 없습니다 — 정규화 재현이 "
            "불가하니 provenance가 기록된 run으로 재학습을 권장합니다(이슈 #27)."
        )
    print(
        f"\n[배포] {best['metric']}={best['metric_value']:.4f} 최고 run 선택. "
        "config.inference에 복사:\n"
        f"  model_path: {best['model_path']!r}\n"
        f"  scaler_run_id: {best['feature_store_run_id']!r}\n"
        f"  scaler_fold_id: {best['fold_id']}\n"
        f"  model_version: {best['model_version']!r}"
    )


if __name__ == "__main__":
    main()
