"""Frontend 대시보드용 백테스트 결과 JSON 익스포터.

`runner.run_fold()`를 fold마다 실행하고, 결과(전략별 NAV 시계열·지표)를 하나의 JSON
파일로 조립해 `frontend/public/backtest.json`에 저장한다. FE는 이 파일이 있으면
실데이터를 그리고, 없으면 `mock.ts`로 fallback한다(`frontend/app/backtest/loader.ts`).

배치 지향(daily.yml 또는 수동 실행) — 서빙 API는 건드리지 않는다.

계약 (loader.ts와 일치):
    {
        "generated_at": ISO-8601 UTC,
        "initial_nav": float,
        "periods": [{"fold_id", "start", "end"}, ...],
        "strategies": [
            {
                "name": "RL policy" | "1/N" | "60:40" | "B&H",
                "color": "#hex",
                "nav": [{"date": "YYYY-MM-DD", "value": float}, ...],
                    # fold를 이어붙인 정규화 시계열(초기=1). fold 경계는 값이 연속되도록
                    # 이전 fold의 마지막 값을 곱해 스케일링.
                "metrics": {"cagr", "sharpe", "mdd", "vol", "total_return",
                            "avg_turnover", "total_cost"}
            }, ...
        ],
        "fold_table": [{"fold_id", "period", "sharpe", "cagr", "mdd"}, ...]   # RL policy만
        "comparison": [{"fold_id", "vs_benchmark": {...}}, ...]               # runner._compare_to_benchmarks 원본
        "verdict_summary": {                                                 # 벤치마크별 fold 통과 집계
            "1/N": {"folds_passed": int, "folds_total": int, "pass_rate": float}, ...
        }   # 이슈 #46 판정 기준 미확정 — 숫자만 내고 합격/불합격 이분법은 넣지 않음.
            # loader.ts는 아직 이 키를 안 읽음(옵셔널 필드라 추가해도 FE는 그대로 동작).
        "combos": [                                                          # NEW · 옵셔널
            {
                "combo": "full" | "M0" | "M1" | "M2" | "M3",
                "display_name": "Full (D=187)" | ...,   # FE 표시명
                "color": "#hex",                        # 콤보 오버레이 색
                "nav": [{"date", "value"}, ...],        # fold 이어붙인 정규화 곡선
                "metrics": {                             # 전 구간 종합지표
                    "cagr", "sharpe", "mdd", "vol", "total_return",
                    "avg_turnover", "total_cost"
                },
                "fold_table": [                          # 콤보별 fold 상세 (도현·찬휘 3순위)
                    {"fold_id", "period", "sharpe", "cagr", "mdd",
                     "avg_turnover", "total_cost"}, ...
                ]
            }, ...
        ]   # 도현이 M0~M3 + full 각 policy를 학습해 넘길 때 populated. 없으면 FE는
            # 콤보 비교 섹션을 조용히 스킵한다(하위호환 — 기존 단일 policy 배포 유지).

실행:
    python -m src.backtest.export
    python -m src.backtest.export --out path/file.json --model-path runs/best.zip
    # 콤보 다중 policy (도현 M0~M3 학습 산출물 병합):
    python -m src.backtest.export --combo full=runs/full.zip --combo M0=runs/m0.zip \\
        --combo M1=runs/m1.zip --combo M2=runs/m2.zip --combo M3=runs/m3.zip
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.backtest.runner import _fold_ids, run_fold
from src.config_loader import DEFAULT_CONFIG_PATH, get_inference, load_config

# FE mock.ts와 이름·색이 동일해야 시각적 일관성이 유지된다(전략 순서도 mock과 동일).
STRATEGY_ORDER: tuple[str, ...] = ("RL policy", "60:40", "1/N", "B&H")
STRATEGY_COLOR: dict[str, str] = {
    "RL policy": "#8b5cf6",
    "60:40": "#10b981",
    "1/N": "#a5b4fc",
    "B&H": "#dc2626",
}

# 콤보 카탈로그 — docs/state_spec.md §2 v1.2. FE ComboComparisonTable·오버레이 차트가
# 이 순서/색을 그대로 소비한다(mock.ts와 일치해야 시각적 일관성 유지).
COMBO_ORDER: tuple[str, ...] = ("full", "M0", "M1", "M2", "M3")
COMBO_COLOR: dict[str, str] = {
    "full": "#8b5cf6",  # RL policy와 같은 보라 (기본 정책 = full 콤보)
    "M0": "#f59e0b",    # 앰버 — EBR 단독, 지표 최소
    "M1": "#22d3ee",    # 시안
    "M2": "#10b981",    # 에메랄드
    "M3": "#f472b6",    # 핑크 — 지표 최대(Drawdown 포함)
}
COMBO_DISPLAY: dict[str, str] = {
    "full": "Full (D=187, 6+2)",
    "M0": "M0 (D=156, 0+1)",
    "M1": "M1 (D=166, 2+1)",
    "M2": "M2 (D=171, 3+1)",
    "M3": "M3 (D=172, 3+2)",
}

DEFAULT_OUT_PATH = "frontend/public/backtest.json"
TRADING_DAYS_PER_YEAR = 252


def _nav_to_points(nav_df: pd.DataFrame, initial_nav: float) -> list[dict[str, Any]]:
    """NAV DataFrame → [{date, value(초기=1로 정규화)}] 리스트."""
    normalized = nav_df["nav"] / initial_nav
    return [
        {"date": str(pd.Timestamp(idx).date()), "value": float(v)}
        for idx, v in normalized.items()
    ]


def _concat_folds(points_per_fold: list[list[dict[str, Any]]]) -> list[dict[str, Any]]:
    """fold별 정규화 NAV points를 이어붙여 하나의 연속 곡선으로 만든다.

    각 fold는 자기 초기값 1에서 시작하므로, 다음 fold는 이전 fold의 마지막 값(=배수)을
    곱해 연속화한다. fold 사이 갭이 있어도 값은 이어진다(x축은 날짜라 자연스러운 갭 표현).
    """
    result: list[dict[str, Any]] = []
    multiplier = 1.0
    for pts in points_per_fold:
        if not pts:
            continue
        for p in pts:
            result.append({"date": p["date"], "value": p["value"] * multiplier})
        multiplier = result[-1]["value"]
    return result


def _metrics_from_points(points: list[dict[str, Any]]) -> dict[str, float]:
    """정규화 NAV points에서 CAGR·Sharpe·MDD·Vol·총수익률을 재계산한다.

    fold 이어붙인 곡선을 인자로 받으면 전 구간 종합지표, 단일 fold를 받으면 그 fold 지표.
    """
    if len(points) < 2:
        return {"cagr": 0.0, "sharpe": 0.0, "mdd": 0.0, "vol": 0.0, "total_return": 0.0}
    values = np.array([p["value"] for p in points], dtype=float)
    daily_ret = np.diff(values) / values[:-1]
    std = float(daily_ret.std(ddof=0))
    total_return = float(values[-1] / values[0] - 1)
    years = max(len(daily_ret) / TRADING_DAYS_PER_YEAR, 1e-9)
    cagr = float((values[-1] / values[0]) ** (1.0 / years) - 1.0) if values[0] > 0 else 0.0
    sharpe = float(daily_ret.mean() / std * np.sqrt(TRADING_DAYS_PER_YEAR)) if std > 0 else 0.0
    vol = float(std * np.sqrt(TRADING_DAYS_PER_YEAR))
    peak = np.maximum.accumulate(values)
    mdd = float((values / peak - 1.0).min())
    return {
        "cagr": cagr,
        "sharpe": sharpe,
        "mdd": mdd,
        "vol": vol,
        "total_return": total_return,
    }


def _summarize_verdict(fold_results: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """벤치마크별 fold 통과 현황을 집계한다.

    `fold_results`는 `run_fold()` 반환 dict들의 리스트 — 각 `comparison`은
    `{"1/N": {..., "beats_target": bool}, "60:40": {...}, "B&H": {...}}` (CLAUDE.md §1 판정).
    이슈 #46(비중편차·회전율 등 판정 기준)이 팀 미확정이라, "overall_pass" 같은 이분법
    합격/불합격 필드는 넣지 않고 fold 통과 개수·비율 숫자만 낸다 — 판정은 사람이 한다.
    """
    benchmarks = [name for name in STRATEGY_ORDER if name != "RL policy"]
    total = len(fold_results)
    summary: dict[str, dict[str, Any]] = {}
    for name in benchmarks:
        passed = sum(
            1 for r in fold_results if r["comparison"].get(name, {}).get("beats_target")
        )
        summary[name] = {
            "folds_passed": passed,
            "folds_total": total,
            "pass_rate": (passed / total) if total > 0 else 0.0,
        }
    return summary


def _build_combo_snapshot(
    combo: str,
    model,
    config_path: str,
    initial_nav: float,
    run_fold_fn,
) -> dict[str, Any]:
    """단일 콤보(=policy 하나)에 대해 fold 전체를 돌려 combo 항목을 조립한다.

    반환 스키마는 계약 §"combos" 항목과 일치. RL policy 성과만 남기고(콤보 비교의 관심사),
    벤치마크는 상위 `strategies`가 이미 담고 있어 중복 저장하지 않는다.
    """
    cfg = load_config(config_path)
    fold_ids = _fold_ids(cfg)

    per_fold_points: list[list[dict[str, Any]]] = []
    per_fold_metrics: list[dict[str, Any]] = []
    per_fold_period: list[tuple[str, str]] = []
    for fold_id in fold_ids:
        result = run_fold_fn(fold_id, model, config_path, "test", initial_nav=initial_nav)
        rl_nav_df = result["nav_by_strategy"]["RL policy"]
        per_fold_points.append(_nav_to_points(rl_nav_df, initial_nav))
        per_fold_metrics.append(result["strategies"]["RL policy"])
        per_fold_period.append(result["period"])

    concat_pts = _concat_folds(per_fold_points)
    overall = _metrics_from_points(concat_pts)
    overall["avg_turnover"] = (
        float(np.mean([m["avg_turnover"] for m in per_fold_metrics])) if per_fold_metrics else 0.0
    )
    overall["total_cost"] = (
        float(np.sum([m["total_cost"] for m in per_fold_metrics])) if per_fold_metrics else 0.0
    )

    fold_table = []
    for i, fold_id in enumerate(fold_ids):
        fm = _metrics_from_points(per_fold_points[i])
        start, end = per_fold_period[i]
        fold_table.append({
            "fold_id": fold_id,
            "period": f"{start} ~ {end}",
            "sharpe": fm["sharpe"],
            "cagr": fm["cagr"],
            "mdd": fm["mdd"],
            "avg_turnover": float(per_fold_metrics[i]["avg_turnover"]),
            "total_cost": float(per_fold_metrics[i]["total_cost"]),
        })

    return {
        "combo": combo,
        "display_name": COMBO_DISPLAY.get(combo, combo),
        "color": COMBO_COLOR.get(combo, "#8b5cf6"),
        "nav": concat_pts,
        "metrics": overall,
        "fold_table": fold_table,
    }


def build_export(
    model,
    config_path: str = DEFAULT_CONFIG_PATH,
    *,
    initial_nav: float = 1_000_000,
    run_fold_fn=run_fold,
    models_by_combo: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """모든 fold에서 run_fold를 실행하고 FE JSON 계약으로 조립한다.

    `run_fold_fn`은 테스트가 모델·Feature Store 없이 가짜 결과를 주입할 수 있도록 분리했다.

    `models_by_combo`가 주어지면 각 콤보별로 별도 policy를 굴려 `combos` 항목을
    함께 조립한다(도현이 M0~M3 + full 각 policy를 학습해 넘길 때). None(기본)이면
    `combos` 필드는 빠지고 기존 단일 policy 계약만 나간다(하위호환).
    """
    cfg = load_config(config_path)
    fold_ids = _fold_ids(cfg)

    fold_results: list[dict[str, Any]] = []
    per_strategy_points: dict[str, list[list[dict[str, Any]]]] = {n: [] for n in STRATEGY_ORDER}
    per_strategy_fold_metrics: dict[str, list[dict[str, Any]]] = {n: [] for n in STRATEGY_ORDER}

    for fold_id in fold_ids:
        result = run_fold_fn(fold_id, model, config_path, "test", initial_nav=initial_nav)
        fold_results.append(result)
        for name in STRATEGY_ORDER:
            nav_df = result["nav_by_strategy"][name]
            per_strategy_points[name].append(_nav_to_points(nav_df, initial_nav))
            per_strategy_fold_metrics[name].append(result["strategies"][name])

    strategies: list[dict[str, Any]] = []
    for name in STRATEGY_ORDER:
        concat_pts = _concat_folds(per_strategy_points[name])
        metrics = _metrics_from_points(concat_pts)
        # summarize()의 회전율·비용은 평균/합으로 부가 정보로 넘긴다(FE에서 참조).
        fold_metrics = per_strategy_fold_metrics[name]
        metrics["avg_turnover"] = (
            float(np.mean([m["avg_turnover"] for m in fold_metrics])) if fold_metrics else 0.0
        )
        metrics["total_cost"] = (
            float(np.sum([m["total_cost"] for m in fold_metrics])) if fold_metrics else 0.0
        )
        strategies.append(
            {"name": name, "color": STRATEGY_COLOR[name], "nav": concat_pts, "metrics": metrics}
        )

    periods = [
        {"fold_id": r["fold_id"], "start": r["period"][0], "end": r["period"][1]}
        for r in fold_results
    ]

    fold_table = [
        {
            "fold_id": r["fold_id"],
            "period": f"{r['period'][0]} ~ {r['period'][1]}",
            **{
                k: v
                for k, v in _metrics_from_points(per_strategy_points["RL policy"][i]).items()
                if k in {"sharpe", "cagr", "mdd"}
            },
        }
        for i, r in enumerate(fold_results)
    ]

    bundle: dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "initial_nav": float(initial_nav),
        "periods": periods,
        "strategies": strategies,
        "fold_table": fold_table,
        "comparison": [
            {"fold_id": r["fold_id"], "vs_benchmark": r["comparison"]} for r in fold_results
        ],
        "verdict_summary": _summarize_verdict(fold_results),
    }

    if models_by_combo:
        # COMBO_ORDER 순서 유지 — FE 표 정렬이 이 순서를 기대한다.
        # 사전에 없는 콤보는 넣지 않는다. 사전에 있는데 모델이 없으면 그 콤보만 스킵.
        combos_out = [
            _build_combo_snapshot(name, models_by_combo[name], config_path, initial_nav, run_fold_fn)
            for name in COMBO_ORDER
            if name in models_by_combo
        ]
        if combos_out:
            bundle["combos"] = combos_out
    return bundle


def write_export(bundle: dict[str, Any], out_path: str = DEFAULT_OUT_PATH) -> None:
    """JSON을 원자적으로 쓴다 (tmp → rename)."""
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(out.suffix + ".tmp")
    tmp.write_text(json.dumps(bundle, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(out)


# S3 키 계약: {prefix}/frontend/backtest.json — Vercel prebuild(scripts/fetch-backtest.mjs)가
# BACKTEST_JSON_URL로 이 객체를 내려받는다. precompute latest.json(§8-2)과 같은 prefix 규칙.
S3_KEY_SUFFIX = "frontend/backtest.json"


def _s3_key(prefix: str) -> str:
    prefix = prefix.strip("/")
    return f"{prefix}/{S3_KEY_SUFFIX}" if prefix else S3_KEY_SUFFIX


def upload_to_s3(
    local_path: str,
    config_path: str = DEFAULT_CONFIG_PATH,
    *,
    bucket: str | None = None,
    client=None,
) -> str | None:
    """생성된 JSON을 S3에 올린다.

    버킷은 환경변수(`S3_BUCKET`) → 인자 → config(`data.s3_bucket`) 순으로 결정한다
    (s3_sync·s3_results와 동일 관례). 버킷 미설정 시 조용히 skip해 로컬 개발을 막지 않는다.
    Content-Type은 `application/json`으로 명시(브라우저·CDN 캐시 힌트).

    Returns
    -------
    업로드된 S3 URL(가상 호스트 스타일) 또는 skip 시 None. Vercel prebuild 스크립트의
    `BACKTEST_JSON_URL`로 이 URL을 쓴다.
    """
    import os

    cfg = load_config(config_path)
    d = cfg.get("data", {})
    bucket = os.environ.get("S3_BUCKET") or bucket or d.get("s3_bucket")
    if not bucket:
        print("[export] S3 버킷 미설정(env·config 모두 없음) → 업로드 skip")
        return None

    prefix = (os.environ.get("S3_PREFIX") or d.get("s3_prefix") or "").strip("/")
    key = _s3_key(prefix)
    region = os.environ.get("AWS_DEFAULT_REGION") or "ap-northeast-2"

    if client is None:
        import boto3  # boto3는 requirements.txt에 포함됨(무거운 의존 아님).

        client = boto3.client("s3")

    with open(local_path, "rb") as f:
        client.put_object(Bucket=bucket, Key=key, Body=f.read(), ContentType="application/json")

    url = f"https://{bucket}.s3.{region}.amazonaws.com/{key}"
    print(f"[export] 업로드 완료 → {url}")
    return url


def _parse_combo_arg(pairs: list[str] | None) -> dict[str, str]:
    """`--combo full=path.zip --combo M0=path.zip` → {"full": "path.zip", "M0": "path.zip"}.

    잘못된 콤보명은 SystemExit로 명시적으로 거부한다 (오타로 조용히 스킵되면 산출물이 어긋남).
    """
    if not pairs:
        return {}
    out: dict[str, str] = {}
    for pair in pairs:
        if "=" not in pair:
            raise SystemExit(f"--combo는 name=path 형식이어야 합니다: {pair!r}")
        name, path = pair.split("=", 1)
        name = name.strip()
        path = path.strip()
        if name not in COMBO_ORDER:
            raise SystemExit(
                f"알 수 없는 콤보명: {name!r} (허용: {list(COMBO_ORDER)})"
            )
        out[name] = path
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="FE 대시보드용 백테스트 결과 JSON 익스포터")
    parser.add_argument("--config", default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--out", default=DEFAULT_OUT_PATH)
    parser.add_argument("--model-path", default=None, help="기본값은 config.inference.model_path")
    parser.add_argument("--initial-nav", type=float, default=1_000_000)
    parser.add_argument(
        "--combo",
        action="append",
        default=[],
        metavar="NAME=PATH",
        help=(
            "콤보명=policy.zip 경로. 반복 지정 시 여러 콤보(NAV·metrics)를 함께 익스포트한다. "
            "예: --combo full=runs/full.zip --combo M0=runs/m0.zip"
        ),
    )
    parser.add_argument(
        "--upload-s3",
        action="store_true",
        help="생성 후 S3(=Vercel prebuild 소스)에도 업로드. 버킷 미설정 시 조용히 skip.",
    )
    args = parser.parse_args()

    from stable_baselines3 import PPO

    cfg = load_config(args.config)
    model_path = args.model_path or get_inference(cfg).get("model_path")
    if not model_path:
        raise SystemExit("model_path가 없습니다 — --model-path 또는 config.inference.model_path")
    model = PPO.load(model_path)

    combo_paths = _parse_combo_arg(args.combo)
    models_by_combo = {name: PPO.load(p) for name, p in combo_paths.items()} or None

    bundle = build_export(
        model,
        args.config,
        initial_nav=args.initial_nav,
        models_by_combo=models_by_combo,
    )
    write_export(bundle, args.out)
    print(f"백테스트 export 완료 → {args.out}")
    print(f"  fold {len(bundle['periods'])}개 · 전략 {len(bundle['strategies'])}개 저장")
    if "combos" in bundle:
        print(f"  콤보 {len(bundle['combos'])}종 병행 저장: "
              f"{[c['combo'] for c in bundle['combos']]}")

    if args.upload_s3:
        url = upload_to_s3(args.out, args.config)
        if url:
            print(f"  Vercel 프리빌드용 URL: BACKTEST_JSON_URL={url}")


if __name__ == "__main__":
    main()
