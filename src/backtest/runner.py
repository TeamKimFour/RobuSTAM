"""Walk-forward 백테스트 러너 — fold별로 policy vs 벤치마크 3종을 비교하고 S3에 저장한다.

`config.split`(anchor·test_blocks·embargo_days)이 정의하는 fold마다:
    1) `policy.run_policy_on_fold()`로 RL policy의 NAV 곡선을 얻는다.
    2) `benchmark.run_equal_weight/sixty_forty/buy_and_hold()`로 같은 기간 벤치마크 3종을 얻는다.
    3) 넷 다 `policy.summarize()`로 지표(총수익·샤프·MDD·회전율·누적비용)를 뽑고,
       `s3_results.save_results_to_s3()`(수정 없음, 그대로)로 **각각 별도 run**으로 저장한다.
       run_id 규칙(팀 확정): `fold{N}_policy` / `fold{N}_1n` / `fold{N}_60_40` / `fold{N}_bh`.
    4) CLAUDE.md §1 판정 기준(벤치마크 대비 샤프 15%+ 개선 **또는** MDD 20%+ 방어)으로
       policy를 각 벤치마크와 비교해 콘솔에 요약 출력한다.

이슈 #46(비중편차·회전율 판정 기준)은 아직 팀 미확정이라 넣지 않았다. `strategies`/`comparison`
값은 `summarize()`가 반환하는 순수 dict를 그대로 담아 옮기기만 하므로, 나중에 `summarize()`
쪽에 키(예: 비중편차)만 추가하면 이 모듈은 코드를 손대지 않아도 그 값을 그대로 저장·출력한다.

실행:
    python -m src.backtest.runner                 # test split, config.split 전체 fold 순회
    python -m src.backtest.runner --fold-id 2      # fold2만
    python -m src.backtest.runner --split valid    # valid split으로
"""

from __future__ import annotations

import numpy as np

from src.backtest import benchmark as bm
from src.backtest.engine import BacktestEngine
from src.backtest.policy import run_policy_on_fold, summarize, targets_to_price_returns
from src.backtest.s3_results import save_results_to_s3
from src.config_loader import (
    DEFAULT_CONFIG_PATH,
    get_assets,
    get_inference,
    load_config,
    resolve_paths_for_combo,
)

# CLAUDE.md §1 정량 목표 — policy가 벤치마크 대비 둘 중 하나를 만족하면 "달성"으로 판정한다.
SHARPE_IMPROVEMENT_TARGET = 0.15
MDD_DEFENSE_TARGET = 0.20

# run_id에 쓰는 전략별 짧은 키(팀 확정: fold{N}_{key}).
_STRATEGY_KEYS = {"RL policy": "policy", "1/N": "1n", "60:40": "60_40", "B&H": "bh"}


def _fold_ids(cfg: dict) -> list[int]:
    """config.split.test_blocks 개수 = fold 개수 (1부터 시작, splits.py 관례)."""
    return list(range(1, len(cfg["split"]["test_blocks"]) + 1))


def _run_id(fold_id: int, strategy_key: str, combo: str | None = None) -> str:
    """S3 run_id. 콤보를 쓰면 접두어를 붙여 콤보끼리 결과를 덮어쓰지 않게 한다.

    `full`(기본 콤보)·None은 기존 `fold{N}_{key}` 규칙 그대로 — 프론트·export.py가
    기대하는 키를 바꾸지 않는다(get_feature_store_dir와 동일한 하위호환 규칙).
    """
    if combo in (None, "full"):
        return f"fold{fold_id}_{strategy_key}"
    return f"{combo}_fold{fold_id}_{strategy_key}"


def _compare_to_benchmarks(strategies: dict[str, dict]) -> dict[str, dict]:
    """policy가 각 벤치마크 대비 CLAUDE.md §1 기준(샤프 15%+ 개선 또는 MDD 20%+ 방어)을 만족하는가.

    `strategies`의 각 값은 `summarize()` dict를 그대로 받는다 — 나중에 이슈 #46 지표(비중편차·
    회전율)가 그 dict에 추가돼도 이 함수는 sharpe/mdd 키만 보므로 그대로 동작한다.
    """
    policy = strategies["RL policy"]
    out: dict[str, dict] = {}
    for name, m in strategies.items():
        if name == "RL policy":
            continue
        sharpe_improvement = (
            (policy["sharpe"] - m["sharpe"]) / abs(m["sharpe"]) if m["sharpe"] != 0 else float("nan")
        )
        # MDD는 음수(예: -0.15)라 절대값 기준 — policy의 낙폭이 벤치마크보다 얼마나 작은가.
        mdd_defense = (
            (abs(m["mdd"]) - abs(policy["mdd"])) / abs(m["mdd"]) if m["mdd"] != 0 else float("nan")
        )
        out[name] = {
            "sharpe_improvement_pct": sharpe_improvement,
            "mdd_defense_pct": mdd_defense,
            "beats_target": bool(
                (not np.isnan(sharpe_improvement) and sharpe_improvement >= SHARPE_IMPROVEMENT_TARGET)
                or (not np.isnan(mdd_defense) and mdd_defense >= MDD_DEFENSE_TARGET)
            ),
        }
    return out


def run_fold(
    fold_id: int,
    model,
    config_path: str = DEFAULT_CONFIG_PATH,
    split: str = "test",
    *,
    initial_nav: float = 1_000_000,
    combo: str | None = None,
) -> dict:
    """한 fold의 policy+벤치마크 3종을 계산하고, 각각 별도 run으로 S3에 저장한다.

    `combo`(M0~M3)를 주면 그 콤보의 Feature Store를 읽는다 — policy의 State 차원이
    학습 때와 같아야 하므로, 학습에 쓴 콤보와 반드시 같은 값을 넘겨야 한다.
    벤치마크 3종은 targets(익일수익률)만 쓰므로 콤보와 무관하게 동일하다.

    Returns
    -------
    dict — {
        "fold_id": int,
        "period": (시작일, 종료일) 문자열 튜플 (이 fold `split` 구간),
        "strategies": {"RL policy": {...}, "1/N": {...}, "60:40": {...}, "B&H": {...}},
                       # 각 값은 summarize()가 반환하는 지표 dict 그대로.
        "nav_by_strategy": {전략명: pd.DataFrame(NAV 시계열, index=거래일)},
                       # export.py 등 후처리에서 시계열이 필요할 때 소비 (S3 저장은 이미 수행됨).
        "s3_prefixes": {전략명: S3 prefix 또는 None(버킷 미설정 시 skip)},
        "comparison": {"1/N": {...}, "60:40": {...}, "B&H": {...}},
                       # _compare_to_benchmarks() 결과 — CLAUDE.md §1 판정.
    }
    """
    from src.data import feature_store as fs

    cfg = resolve_paths_for_combo(load_config(config_path), combo)
    assets = get_assets(cfg)
    mk = lambda: BacktestEngine(initial_nav=initial_nav, config_path=config_path)

    policy_nav = run_policy_on_fold(model, fold_id, split, config_path, mk(), combo=combo)

    targets = fs.load_targets(cfg["data"]["feature_store_dir"], fold_id, split)
    price_returns = targets_to_price_returns(targets, assets).loc[policy_nav.index]

    nav_by_strategy = {
        "RL policy": policy_nav,
        "1/N": bm.run_equal_weight(price_returns, mk(), config_path),
        "60:40": bm.run_sixty_forty(price_returns, mk(), config_path),
        "B&H": bm.run_buy_and_hold(price_returns, mk(), config_path),
    }

    strategies: dict[str, dict] = {}
    s3_prefixes: dict[str, str | None] = {}
    for name, nav_df in nav_by_strategy.items():
        metrics = summarize(nav_df, initial_nav)
        strategies[name] = metrics
        s3_prefixes[name] = save_results_to_s3(
            nav_df,
            metrics,
            run_id=_run_id(fold_id, _STRATEGY_KEYS[name], combo),
            config_path=config_path,
        )

    return {
        "fold_id": fold_id,
        "combo": combo or cfg.get("active_combo"),
        "period": (str(policy_nav.index[0].date()), str(policy_nav.index[-1].date())),
        "strategies": strategies,
        "nav_by_strategy": nav_by_strategy,
        "s3_prefixes": s3_prefixes,
        "comparison": _compare_to_benchmarks(strategies),
    }


def _print_fold_report(result: dict) -> None:
    fold_id = result["fold_id"]
    start, end = result["period"]
    combo = result.get("combo")
    combo_tag = f" [{combo}]" if combo else ""
    print(f"\n=== fold{fold_id}{combo_tag} test: {start} ~ {end} ===")
    print(f"{'전략':<10}{'총수익':>10}{'샤프':>9}{'MDD':>10}{'회전율':>10}{'누적비용':>14}")
    print("-" * 63)
    for name, m in result["strategies"].items():
        print(
            f"{name:<10}{m['total_return']:>9.2%}{m['sharpe']:>9.3f}{m['mdd']:>9.2%}"
            f"{m['avg_turnover']:>10.4f}{m['total_cost']:>14,.0f}"
        )

    print(f"\n{'vs 벤치마크':<10}{'샤프개선':>12}{'MDD방어':>12}{'CLAUDE §1':>12}")
    print("-" * 46)
    for name, c in result["comparison"].items():
        status = "달성" if c["beats_target"] else "미달"
        print(f"{name:<10}{c['sharpe_improvement_pct']:>11.1%}{c['mdd_defense_pct']:>12.1%}{status:>12}")


def main() -> None:
    """CLI — walk-forward: 전체(또는 지정) fold에서 policy vs 벤치마크 3종을 비교·저장한다.

        python -m src.backtest.runner                 # test split, config.split 전체 fold 순회
        python -m src.backtest.runner --fold-id 2      # fold2만
    """
    import argparse

    from stable_baselines3 import PPO

    parser = argparse.ArgumentParser(description="Walk-forward 백테스트: policy vs 벤치마크")
    parser.add_argument("--config", default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--fold-id", type=int, default=None, help="생략 시 config.split 전체 fold 순회")
    parser.add_argument("--split", default="test", choices=["train", "valid", "test"])
    parser.add_argument("--model-path", default=None, help="기본값은 config.inference.model_path")
    parser.add_argument("--initial-nav", type=float, default=1_000_000)
    parser.add_argument(
        "--combo", default=None, help="피처 콤보(M0~M3). 학습에 쓴 콤보와 같아야 한다"
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    model_path = args.model_path or get_inference(cfg).get("model_path")
    if not model_path:
        raise SystemExit("model_path가 없습니다 — --model-path 또는 config.inference.model_path")
    model = PPO.load(model_path)

    fold_ids = [args.fold_id] if args.fold_id is not None else _fold_ids(cfg)
    for fold_id in fold_ids:
        result = run_fold(
            fold_id, model, args.config, args.split,
            initial_nav=args.initial_nav, combo=args.combo,
        )
        _print_fold_report(result)


if __name__ == "__main__":
    main()
