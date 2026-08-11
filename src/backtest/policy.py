"""정책→백테스트 어댑터 — 학습된 RL policy를 BacktestEngine 위에서 굴린다.

`benchmark.py`가 고정 목표비중 전략(1/N·60:40·B&H)을 NAV 곡선으로 바꾸는 것과 같은 자리에,
**학습된 policy**를 넣는 모듈이다. 출력 포맷을 벤치마크와 동일하게 맞춰 같은 축에서 비교한다.

두 계약을 잇는 것이 이 모듈의 존재 이유다:

    민지 targets.parquet          도현 policy              찬휘 BacktestEngine
    fwd_ret_<asset> (로그수익률)   obs(187) → 로짓          price_returns (단순수익률)
              │                       │                          │
              └── expm1 + 컬럼 리네임 ─┴── softmax + prev_weight ─┘

  ① **로그수익률 → 단순수익률**: 민지 targets는 로그수익률(`fwd_ret[t] = log_ret[t+1]`)로
     저장되지만 엔진은 산술수익률을 기대한다(`nav * (1 + w·r)`). `expm1`로 변환한다.
     env가 `step()`에서 하는 변환(선택지 α)과 동일하다.
  ② **컬럼 리네임**: `fwd_ret_SPY` → `SPY`. 엔진·벤치마크는 자산명 컬럼을 기대한다.
  ③ **prev_weight 주입**: Feature Store에 적재된 State의 prev_weight 칸은 **0**이다
     (`assemble`이 0으로 채우고 `PortfolioEnv`가 런타임에 덮어쓴다). 이 어댑터도 매 스텝
     직전 비중을 주입해야 정책이 학습 때와 같은 관측을 본다 — 빠뜨리면 정책이 "직전 비중을
     모르는" 상태로 판단해 백테스트가 조용히 학습과 어긋난다.
  ④ **softmax**: 정책망 출력은 로짓이므로 `portfolio_env._softmax`를 그대로 재사용한다
     (학습 `step()`·precompute와 동일 함수 — 세 경로가 같은 비중을 내도록).

회계 규약은 `benchmark.py`와 동일하다(start-of-day 리밸런싱). 초기 보유는 SHV 100%
(`PortfolioEnv.reset()` 관례).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.backtest.benchmark import sixty_forty_target
from src.backtest.engine import BacktestEngine
from src.config_loader import DEFAULT_CONFIG_PATH, get_assets, get_window, load_config
from src.data import schema


def targets_to_price_returns(targets_df: pd.DataFrame, assets: list[str]) -> pd.DataFrame:
    """민지 targets(로그수익률) → 엔진용 단순수익률 DataFrame으로 변환한다.

    `fwd_ret_<asset>` 컬럼을 `<asset>`으로 리네임하고 `expm1`로 산술수익률화한다.
    엔진·벤치마크가 기대하는 입력 포맷(`price_returns`)과 정확히 같아진다.
    """
    expected = [f"fwd_ret_{a}" for a in assets]
    missing = [c for c in expected if c not in targets_df.columns]
    if missing:
        raise ValueError(
            f"targets에 필요한 컬럼이 없습니다: {missing} "
            "(src.data.feature_store.load_targets 출력을 사용하세요)."
        )
    out = np.expm1(targets_df[expected].to_numpy(dtype=np.float64))
    return pd.DataFrame(out, index=targets_df.index, columns=list(assets))


def run_policy(
    price_returns: pd.DataFrame,
    state_df: pd.DataFrame,
    model,
    engine: BacktestEngine,
    config_path: str = DEFAULT_CONFIG_PATH,
    *,
    deterministic: bool = True,
) -> pd.DataFrame:
    """학습된 policy의 NAV 곡선을 만든다 (벤치마크 함수들과 동일한 출력 포맷).

    Parameters
    ----------
    price_returns : 단순수익률 (`targets_to_price_returns` 출력). 컬럼 = 자산명.
    state_df : 정규화된 187차원 State (`feature_store.load_features` 출력).
        `price_returns`와 index가 정렬돼 있어야 한다(교집합만 사용).
    model : `predict(obs, deterministic=...) -> (action, _)` 인터페이스(SB3 정책).
    engine : 거래비용·초기 NAV를 쥔 백테스트 엔진.
    deterministic : 백테스트는 재현성을 위해 결정적 행동을 기본으로 한다.

    Returns
    -------
    DataFrame — index=date, 컬럼 `nav`·`cost`·`turnover` + 자산별 비중.
    """
    from src.env.portfolio_env import _softmax  # 학습 step()·precompute와 동일 softmax

    cfg = load_config(config_path)
    assets = get_assets(cfg)
    W = get_window(cfg)

    if list(price_returns.columns) != list(assets):
        raise ValueError(
            f"price_returns 컬럼이 자산 순서와 다릅니다. 기대: {assets}, "
            f"실제: {list(price_returns.columns)}"
        )
    expected_cols = schema.feature_names(W)
    if list(state_df.columns) != expected_cols:
        raise ValueError(
            "state_df 컬럼이 schema.feature_names(W)와 일치하지 않습니다 "
            "(feature_store.load_features 출력을 사용하세요)."
        )

    # targets는 익일 수익률이 없는 마지막 행이 빠져 있어 State보다 짧을 수 있다 → 교집합.
    idx = state_df.index.intersection(price_returns.index)
    if len(idx) == 0:
        raise ValueError("state_df와 price_returns의 공통 날짜가 없습니다.")
    state_df = state_df.loc[idx]
    price_returns = price_returns.loc[idx]

    prev_slice = schema.prev_weight_slice(W)
    weights = np.zeros(len(assets), dtype=np.float64)
    weights[assets.index("SHV")] = 1.0  # 콜드스타트 = 현금 100% (env.reset과 동일)
    nav = engine.initial_nav

    records = []
    for date, r in zip(idx, price_returns.to_numpy(dtype=np.float64)):
        obs = state_df.loc[date].to_numpy(dtype=np.float32).copy()
        # ③ 직전 비중 주입 — 적재된 State의 prev_weight 칸은 0이므로 반드시 덮어써야 한다.
        obs[prev_slice] = weights.astype(np.float32)

        action, _ = model.predict(obs, deterministic=deterministic)
        target = _softmax(np.asarray(action, dtype=np.float64))  # ④ 로짓 → 비중(합=1)

        new_nav, cost = engine.calc_nav(nav, weights, target, r)
        records.append(
            {
                "date": date,
                "nav": new_nav,
                "cost": cost,
                "turnover": float(np.abs(target - weights).sum()),
                **dict(zip(assets, target)),
            }
        )
        nav = new_nav
        weights = target  # 완전 리밸런싱 — 기간 내 드리프트 무시(engine 규약)

    return pd.DataFrame.from_records(records, index="date")


def run_policy_on_fold(
    model,
    fold_id: int,
    split: str = "test",
    config_path: str = DEFAULT_CONFIG_PATH,
    engine: BacktestEngine | None = None,
    *,
    deterministic: bool = True,
) -> pd.DataFrame:
    """Feature Store의 fold/split을 읽어 policy를 굴린다(편의 진입점).

    정식 성과검증은 **test split**에서 한다 — train/valid는 학습·모델선택에 이미 쓰였다.
    """
    from src.data import feature_store as fs

    cfg = load_config(config_path)
    assets = get_assets(cfg)
    out_dir = cfg["data"]["feature_store_dir"]

    state_df = fs.load_features(out_dir, fold_id, split)
    targets_df = fs.load_targets(out_dir, fold_id, split)
    price_returns = targets_to_price_returns(targets_df, assets)

    engine = engine if engine is not None else BacktestEngine(config_path=config_path)
    return run_policy(price_returns, state_df, model, engine, config_path,
                      deterministic=deterministic)


def summarize(nav_df: pd.DataFrame, initial_nav: float) -> dict:
    """NAV 곡선에서 비교용 요약지표를 뽑는다(총수익·샤프·MDD·회전율·누적비용·비중편차).

    비중편차(weight_deviation, 이슈 #46)는 60:40 벤치마크 대비 Active Share를 매일 계산해
    기간 평균한 것이다:

        weight_deviation = mean_over_days( 0.5 * Σ|policy_weight_i - benchmark_weight_i| )

    자산 컬럼은 `nav_df.columns`에서 `nav`/`cost`/`turnover`를 제외한 나머지로 추출한다
    (하드코딩 금지 — policy든 벤치마크든 이 함수 하나로 계산 가능하게).
    """
    nav = nav_df["nav"]
    r = nav.pct_change().dropna()
    std = float(r.std())

    assets = [c for c in nav_df.columns if c not in ("nav", "cost", "turnover")]
    benchmark_weights = sixty_forty_target(assets)
    daily_deviation = 0.5 * (nav_df[assets] - benchmark_weights).abs().sum(axis=1)

    return {
        "total_return": float(nav.iloc[-1] / initial_nav - 1),
        "sharpe": float(r.mean() / std * np.sqrt(252)) if std > 0 else 0.0,
        "mdd": float((nav / nav.cummax() - 1).min()),
        "avg_turnover": float(nav_df["turnover"].mean()),
        "total_cost": float(nav_df["cost"].sum()),
        "weight_deviation": float(daily_deviation.mean()),
    }


def main() -> None:
    """CLI — 배포 policy를 벤치마크와 같은 축에서 비교한다.

        python -m src.backtest.policy --fold-id 1 --split test
    """
    import argparse

    from stable_baselines3 import PPO

    from src.backtest import benchmark as bm
    from src.config_loader import get_inference
    from src.data import feature_store as fs

    parser = argparse.ArgumentParser(description="정책 vs 벤치마크 백테스트")
    parser.add_argument("--config", default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--fold-id", type=int, default=1)
    parser.add_argument("--split", default="test", choices=["train", "valid", "test"])
    parser.add_argument("--model-path", default=None, help="기본값은 config.inference.model_path")
    parser.add_argument("--initial-nav", type=float, default=1_000_000)
    args = parser.parse_args()

    cfg = load_config(args.config)
    assets = get_assets(cfg)
    model_path = args.model_path or get_inference(cfg).get("model_path")
    if not model_path:
        raise SystemExit("model_path가 없습니다 — --model-path 또는 config.inference.model_path")

    model = PPO.load(model_path)
    mk = lambda: BacktestEngine(initial_nav=args.initial_nav, config_path=args.config)

    policy_nav = run_policy_on_fold(model, args.fold_id, args.split, args.config, mk())
    targets = fs.load_targets(cfg["data"]["feature_store_dir"], args.fold_id, args.split)
    price_returns = targets_to_price_returns(targets, assets).loc[policy_nav.index]

    runs = {
        "RL policy": policy_nav,
        "1/N": bm.run_equal_weight(price_returns, mk(), args.config),
        "60:40": bm.run_sixty_forty(price_returns, mk(), args.config),
        "B&H": bm.run_buy_and_hold(price_returns, mk(), args.config),
    }

    print(f"fold{args.fold_id} {args.split}: "
          f"{policy_nav.index[0].date()} ~ {policy_nav.index[-1].date()} ({len(policy_nav)}일)\n")
    print(f"{'전략':<10}{'총수익':>10}{'샤프':>9}{'MDD':>10}{'회전율':>10}{'누적비용':>14}")
    print("-" * 63)
    for name, df in runs.items():
        m = summarize(df, args.initial_nav)
        print(f"{name:<10}{m['total_return']:>9.2%}{m['sharpe']:>9.3f}{m['mdd']:>9.2%}"
              f"{m['avg_turnover']:>10.4f}{m['total_cost']:>14,.0f}")


if __name__ == "__main__":
    main()
