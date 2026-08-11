"""피처 콤보 실험 러너 — 콤보별 학습→백테스트→3지표 자동 판정 (6주차 도현 2·3순위).

민지 스크리닝(`src/data/screen_combos.py`)이 만든 콤보별 Feature Store를 받아, 각 콤보를
**동일 조건**(같은 fold·seed·학습량·λ·κ)으로 RL 학습하고 백테스트한 뒤, 회의에서 확정할
**3지표 판정 기준**(비중편차·회전율 관문 + 샤프 성과)을 씌워 채택/기각을 자동으로 낸다.

설계 원칙
    - 판정 로직(`judge_combo`)은 **순수 함수**라 학습 없이 단독 검증 가능(tests/test_experiment.py).
    - 기준값은 `Criteria` dataclass로 분리 — 회의 결정에 따라 한 곳만 바꾸면 된다. 기본값은
      도현 제안 초안(λ 스윕 실측 근거, `docs/issue34_proposal.md` 개선안 A·D)이다.
    - 콤보 선택은 `config.yaml`의 `feature_combos`가 SSOT다. 이 모듈은 이름만 넘기고
      `config_loader.resolve_paths_for_combo`가 피처 리스트·Feature Store 경로를 함께 갈아끼운다.
    - **동일 조건 보장**: 콤보마다 fold·seed·total_timesteps·λ·κ를 같은 값으로 강제한다.
      콤보 간 차이가 피처셋에서만 오도록 하기 위함이다(비교 실험의 전제).

판정 구조 (관문 → 성과, AND)
    관문(Gate): 세 지표가 상충하므로 먼저 통과해야 성과 판정 대상이 된다.
        - 비중편차 평균 ≥ min_dispersion_mean, 그리고 어느 fold도 ≥ min_dispersion_any_fold
          (1/N 복제 배제 — 편차가 낮으면 샤프가 좋아도 알파가 아니라 균등비중 흉내다)
        - 회전율 평균 ≤ max_turnover_mean (과매매 배제)
    성과(Performance): 관문 통과분만 판정. 주 기준은 vs 1/N(가장 높은 벤치마크).
        - 3 fold 중 min_folds_beating_1n개 이상에서 CLAUDE.md §1(샤프 +15% 또는 MDD 20% 방어) 달성
        - 최악 fold의 vs 1/N 샤프 격차 ≥ min_worst_sharpe_gap (단일 fold 요행·재앙 배제)

실행:
    python -m src.models.experiment                       # M0~M3 전부
    python -m src.models.experiment --combos M1 M2        # 일부만
    python -m src.models.experiment --timesteps 200000
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from statistics import mean
from typing import Any

from src.config_loader import (
    DEFAULT_CONFIG_PATH,
    get_assets,
    get_meta_db,
    get_state_dim,
    load_config,
    resolve_paths_for_combo,
)


@dataclass(frozen=True)
class Criteria:
    """3지표 통과 기준 (도현 회의 제안 초안 = 기본값, 회의 확정 후 조정).

    근거: λ 스윕 실측(fold1). 비중편차 0.05는 신호를 잡던 기본 lr(0.0667)과 붕괴선
    (λ=10/seed42 0.0244) 사이 안전선. 회전율 0.30은 기존 0.859 대비 65%+ 감소선.
    """

    min_dispersion_mean: float = 0.05      # 비중편차(max-min) 3 fold 평균 하한
    min_dispersion_any_fold: float = 0.03  # 어느 fold도 이 아래면 붕괴로 기각
    max_turnover_mean: float = 0.30        # 회전율 3 fold 평균 상한
    min_folds_beating_1n: int = 2          # vs 1/N beats_target을 만족해야 하는 최소 fold 수
    min_worst_sharpe_gap: float = -0.10    # 최악 fold의 (policy − 1/N) 샤프 격차 하한


@dataclass
class FoldMetrics:
    """judge_combo가 소비하는 fold별 정규화 지표 (run_fold 결과에서 추출)."""

    fold_id: int
    weight_dispersion: float   # policy 자산별 평균비중의 max−min
    avg_turnover: float        # policy 일평균 L1 회전율
    beats_1n: bool             # vs 1/N CLAUDE.md §1 달성 여부
    sharpe_gap_vs_1n: float    # policy 샤프 − 1/N 샤프 (절대 격차)


def weight_dispersion(policy_nav_df, assets: list[str]) -> float:
    """policy NAV DataFrame의 자산별 비중 컬럼에서 비중편차(자산별 평균비중의 max−min)를 낸다.

    `src.backtest.policy._run_policy`가 매 행에 자산별 목표비중을 컬럼으로 실어주므로
    (예: SPY·EWY·TLT·GLD·SHV), 각 자산 평균을 낸 뒤 최대−최소를 반환한다. 이 값이 0에
    가까우면 정책이 1/N(균등비중)을 흉내낸 것이다(issue34_proposal §2-2).
    """
    missing = [a for a in assets if a not in policy_nav_df.columns]
    if missing:
        raise ValueError(
            f"policy NAV에 자산 비중 컬럼이 없습니다: {missing}. "
            "src.backtest.policy.run_policy_on_fold 출력을 넘기세요."
        )
    means = [float(policy_nav_df[a].mean()) for a in assets]
    return max(means) - min(means)


def fold_metrics_from_run(run_result: dict, assets: list[str]) -> FoldMetrics:
    """`src.backtest.runner.run_fold()` 반환 dict → judge_combo용 FoldMetrics로 정규화한다."""
    policy = run_result["strategies"]["RL policy"]
    one_over_n = run_result["strategies"]["1/N"]
    return FoldMetrics(
        fold_id=run_result["fold_id"],
        weight_dispersion=weight_dispersion(run_result["nav_by_strategy"]["RL policy"], assets),
        avg_turnover=float(policy["avg_turnover"]),
        beats_1n=bool(run_result["comparison"]["1/N"]["beats_target"]),
        sharpe_gap_vs_1n=float(policy["sharpe"]) - float(one_over_n["sharpe"]),
    )


def judge_combo(folds: list[FoldMetrics], criteria: Criteria | None = None) -> dict[str, Any]:
    """조합의 fold별 지표에 3지표 기준을 씌워 관문·성과·채택 여부를 낸다 (순수 함수).

    Returns dict — gate/performance 세부와 최종 `adopted`, 사람이 읽을 `reasons`.
    """
    if not folds:
        raise ValueError("folds가 비어 있습니다 — 최소 1개 fold 결과가 필요합니다.")
    c = criteria or Criteria()

    dispersions = [f.weight_dispersion for f in folds]
    turnovers = [f.avg_turnover for f in folds]
    dispersion_mean = mean(dispersions)
    turnover_mean = mean(turnovers)
    min_dispersion = min(dispersions)

    gate_dispersion = dispersion_mean >= c.min_dispersion_mean and min_dispersion >= c.min_dispersion_any_fold
    gate_turnover = turnover_mean <= c.max_turnover_mean
    gate_passed = gate_dispersion and gate_turnover

    folds_beating = sum(1 for f in folds if f.beats_1n)
    worst_sharpe_gap = min(f.sharpe_gap_vs_1n for f in folds)
    perf_folds = folds_beating >= c.min_folds_beating_1n
    perf_worst = worst_sharpe_gap >= c.min_worst_sharpe_gap
    performance_passed = perf_folds and perf_worst

    adopted = gate_passed and performance_passed

    reasons: list[str] = []
    if not gate_dispersion:
        reasons.append(
            f"관문 실패(비중편차): 평균 {dispersion_mean:.4f}(≥{c.min_dispersion_mean}) · "
            f"최소 {min_dispersion:.4f}(≥{c.min_dispersion_any_fold}) — 1/N 흉내 의심"
        )
    if not gate_turnover:
        reasons.append(f"관문 실패(회전율): 평균 {turnover_mean:.4f}(≤{c.max_turnover_mean}) — 과매매")
    if gate_passed and not perf_folds:
        reasons.append(
            f"성과 실패: vs 1/N 달성 {folds_beating}/{len(folds)} fold(≥{c.min_folds_beating_1n} 필요)"
        )
    if gate_passed and not perf_worst:
        reasons.append(
            f"성과 실패(최악 fold): 샤프격차 {worst_sharpe_gap:+.3f}(≥{c.min_worst_sharpe_gap} 필요)"
        )
    if adopted:
        reasons.append("채택: 관문·성과 모두 통과")

    return {
        "adopted": adopted,
        "gate_passed": gate_passed,
        "performance_passed": performance_passed,
        "gate": {
            "dispersion_mean": dispersion_mean,
            "dispersion_min": min_dispersion,
            "turnover_mean": turnover_mean,
            "dispersion_ok": gate_dispersion,
            "turnover_ok": gate_turnover,
        },
        "performance": {
            "folds_beating_1n": folds_beating,
            "folds_total": len(folds),
            "worst_sharpe_gap_vs_1n": worst_sharpe_gap,
        },
        "criteria": asdict(c),
        "reasons": reasons,
        "per_fold": [asdict(f) for f in folds],
    }


# config.yaml의 feature_combos 중 실험 대상. `full`(Baseline 187)은 비교 기준으로 함께 돌린다.
DEFAULT_COMBOS = ("full", "M0", "M1", "M2", "M3")


def combo_provenance(cfg: dict, combo: str, fold_id: int) -> dict[str, Any]:
    """콤보의 Feature Store build run_id·차원을 provenance로 확보한다.

    민지 쪽 build를 안 돌린 콤보를 조용히 건너뛰지 않도록, 여기서 먼저 확인하고
    없으면 그 자리에서 알려준다(학습을 몇 시간 돌린 뒤 깨지는 것보다 낫다).
    """
    from src.data import feature_store as fs

    resolved = resolve_paths_for_combo(cfg, combo)
    meta_db = get_meta_db(cfg, combo)
    build_run_id = fs.latest_run_id_for_fold(meta_db, fold_id)
    if build_run_id is None:
        raise RuntimeError(
            f"콤보 '{combo}'의 fold{fold_id} Feature Store가 없습니다 — "
            f"먼저 build(combo='{combo}')를 실행하세요(meta_db={meta_db})."
        )
    return {
        "combo": combo,
        "build_run_id": build_run_id,
        "state_dim": get_state_dim(resolved),
        "asset_features": list(resolved["features"]["asset"]),
        "market_features": list(resolved["features"]["market"]),
    }


def _mlflow_client(cfg: dict):
    """train.py와 같은 tracking uri 규칙으로 MlflowClient를 만든다."""
    import os

    # 파일 스토어가 기본 비활성이라 명시 허용 (train.py·screen_combos.py와 동일 임시 조치).
    os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
    from mlflow.tracking import MlflowClient

    from src.models.train import _to_tracking_uri

    uri = _to_tracking_uri(cfg.get("model", {}).get("mlflow_tracking_uri", "mlruns"))
    return MlflowClient(tracking_uri=uri)


def log_backtest_to_run(cfg: dict, mlflow_run_id: str, run_result: dict, fm: FoldMetrics) -> None:
    """백테스트(test split) 지표를 그 fold를 학습한 MLflow run에 되붙인다.

    train.py는 valid split만 기록한다(학습 중 조기 확인용). 그런데 팀 주간계획이 요구하는
    기록 항목과 형우 시각화(콤보별 fold 샤프·MDD·회전율·비중편차)는 **test 지표**다.
    그 값은 백테스트를 돌려야 나오므로 학습 run이 닫힌 뒤에 client로 추가 기록한다.

    JSON에만 남기면 팀 공용 DB를 쓰는 의미가 없어진다 — 형우가 MLflow만 보고 비교할 수
    있게 하는 것이 이 함수의 목적이다.
    """
    client = _mlflow_client(cfg)
    policy = run_result["strategies"]["RL policy"]
    metrics = {
        "test_sharpe": float(policy["sharpe"]),
        "test_total_return": float(policy["total_return"]),
        "test_mdd": float(policy["mdd"]),
        "test_avg_turnover": float(policy["avg_turnover"]),
        "test_total_cost": float(policy["total_cost"]),
        # 3지표 중 비중편차는 summarize()에 없어 러너가 보완해 계산한 값이다.
        "test_weight_dispersion": float(fm.weight_dispersion),
        "test_sharpe_gap_vs_1n": float(fm.sharpe_gap_vs_1n),
        "test_beats_1n": float(fm.beats_1n),
    }
    # 벤치마크 샤프도 같이 남긴다 — 형우가 비교 그래프를 그릴 때 재계산이 필요 없도록.
    for name, key in (("1/N", "1n"), ("60:40", "60_40"), ("B&H", "bh")):
        bench = run_result["strategies"].get(name)
        if bench is not None:
            metrics[f"test_bench_{key}_sharpe"] = float(bench["sharpe"])
            metrics[f"test_bench_{key}_mdd"] = float(bench["mdd"])

    for key, value in metrics.items():
        client.log_metric(mlflow_run_id, key, value)


def log_verdict_to_runs(cfg: dict, mlflow_run_ids: list[str], verdict: dict) -> None:
    """콤보 판정 결과를 그 콤보의 모든 fold run에 붙인다(판정은 fold 통합이라 run마다 같은 값)."""
    client = _mlflow_client(cfg)
    for run_id in mlflow_run_ids:
        client.log_metric(run_id, "verdict_adopted", float(verdict["adopted"]))
        client.log_metric(run_id, "verdict_gate_passed", float(verdict["gate_passed"]))
        client.log_metric(run_id, "verdict_performance_passed", float(verdict["performance_passed"]))
        client.set_tag(run_id, "verdict", "ADOPTED" if verdict["adopted"] else "REJECTED")
        client.set_tag(run_id, "verdict_reasons", " | ".join(verdict["reasons"]))


@dataclass
class ExperimentSettings:
    """모든 콤보에 **동일하게** 적용되는 학습·판정 조건 (비교 실험의 전제).

    콤보마다 이 값이 달라지면 성능 차이가 피처셋 때문인지 조건 때문인지 알 수 없다.
    """

    fold_ids: list[int] | None = None      # None이면 config.split 전체 fold
    seed: int = 42
    total_timesteps: int | None = None     # None이면 config.model 기본값
    lam: float = 10.0                      # λ (개선안 A) — train_cost_multiplier
    kappa: float = 0.0                     # κ (개선안 D) — vol_penalty_coef
    split: str = "test"
    criteria: Criteria = field(default_factory=Criteria)


def run_experiment(
    combos: list[str] | None = None,
    settings: ExperimentSettings | None = None,
    config_path: str = DEFAULT_CONFIG_PATH,
    out_path: str = "mlruns/experiment_verdict.json",
) -> dict[str, Any]:
    """콤보별로 fold 학습→백테스트→3지표 판정을 돌리고 결과를 JSON으로 기록한다.

    각 콤보는 자기 Feature Store(`data/feature_store/<combo>/`)를 쓰고, 그 외 조건
    (fold·seed·학습량·λ·κ)은 `settings`로 전부 동일하게 고정된다.
    """
    # 무거운 의존성은 실행 시점에만 import (train.py와 동일 규약).
    from stable_baselines3 import PPO

    from src.backtest.runner import _fold_ids, run_fold
    from src.models.train import train

    combos = list(combos or DEFAULT_COMBOS)
    st = settings or ExperimentSettings()
    cfg = load_config(config_path)
    assets = get_assets(cfg)
    fold_ids = st.fold_ids or _fold_ids(cfg)

    results: dict[str, Any] = {
        "settings": {**asdict(st), "fold_ids": fold_ids, "config_path": config_path},
        "combos": {},
    }

    # 학습 전에 모든 콤보의 Feature Store 존재를 먼저 확인한다 — 3번째 콤보에서
    # 깨지느라 앞의 학습 시간을 버리는 일이 없도록.
    provenance = {c: combo_provenance(cfg, c, fold_ids[0]) for c in combos}

    for combo in combos:
        fold_metrics: list[FoldMetrics] = []
        runs: list[dict[str, Any]] = []
        for fold_id in fold_ids:
            # λ·κ는 인자로 직접 넘긴다 — config 파일을 건드리지 않고 모든 콤보에 같은 값을
            # 강제하기 위함이다(로드된 cfg dict를 고쳐봐야 train이 파일을 다시 읽으므로 무시된다).
            trained = train(
                config_path=config_path,
                fold_id=fold_id,
                total_timesteps=st.total_timesteps,
                seed=st.seed,
                combo=combo,
                cost_multiplier=st.lam,
                vol_penalty_coef=st.kappa,
            )
            model = PPO.load(trained["model_path"])
            run_result = run_fold(fold_id, model, config_path, st.split, combo=combo)
            fm = fold_metrics_from_run(run_result, assets)
            fold_metrics.append(fm)
            # test 지표를 학습 run에 되붙인다 — 팀 공용 MLflow만 보고 콤보 비교가 되도록.
            log_backtest_to_run(cfg, trained["run_id"], run_result, fm)
            runs.append(
                {
                    "fold_id": fold_id,
                    "mlflow_run_id": trained["run_id"],
                    "model_path": trained["model_path"],
                    "feature_store_run_id": trained["feature_store_run_id"],
                    "valid_sharpe": trained.get("valid_sharpe"),
                }
            )

        verdict = judge_combo(fold_metrics, st.criteria)
        log_verdict_to_runs(cfg, [r["mlflow_run_id"] for r in runs], verdict)
        results["combos"][combo] = {
            "provenance": provenance[combo],
            "runs": runs,
            "verdict": verdict,
        }

    results["adopted"] = [n for n, r in results["combos"].items() if r["verdict"]["adopted"]]

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    return results


def main() -> None:
    """CLI — 콤보 실험 실행."""
    import argparse

    from src.config_loader import force_utf8_stdout

    # 콤보 x fold마다 MLflow가 실행 URL(이모지 포함)을 찍는다. cp949 콘솔에서 여기가
    # 터지면 run이 RUNNING으로 남아 select.py에서 빠지므로 진입점에서 먼저 고정한다.
    force_utf8_stdout()

    parser = argparse.ArgumentParser(description="피처 콤보 실험 + 3지표 자동 판정")
    parser.add_argument("--config", default=DEFAULT_CONFIG_PATH)
    parser.add_argument(
        "--combos", nargs="+", default=None,
        help=f"실험할 콤보 (기본: {' '.join(DEFAULT_COMBOS)})",
    )
    parser.add_argument("--folds", nargs="+", type=int, default=None, help="생략 시 전체 fold")
    parser.add_argument("--timesteps", type=int, default=None, help="생략 시 config 기본값")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lam", type=float, default=10.0, help="λ — 학습 전용 거래비용 배수")
    parser.add_argument("--kappa", type=float, default=0.0, help="κ — 학습 전용 변동성 페널티")
    parser.add_argument("--out", default="mlruns/experiment_verdict.json")
    args = parser.parse_args()

    settings = ExperimentSettings(
        fold_ids=args.folds,
        seed=args.seed,
        total_timesteps=args.timesteps,
        lam=args.lam,
        kappa=args.kappa,
    )
    results = run_experiment(
        combos=args.combos, settings=settings, config_path=args.config, out_path=args.out
    )

    print("\n=== 콤보별 3지표 판정 ===")
    print(f"(공통 조건: fold={results['settings']['fold_ids']} seed={args.seed} "
          f"λ={args.lam} κ={args.kappa})")
    for name, r in results["combos"].items():
        v = r["verdict"]
        p = r["provenance"]
        mark = "✅ 채택" if v["adopted"] else "❌ 기각"
        print(f"\n[{name}] D={p['state_dim']} build={p['build_run_id']} → {mark}")
        for reason in v["reasons"]:
            print(f"  - {reason}")
    print(f"\n채택 콤보: {results['adopted'] or '없음'}")
    print(f"기록: {args.out}")


if __name__ == "__main__":
    main()
