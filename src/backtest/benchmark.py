"""벤치마크 전략 모듈 — 1/N(동일비중)·60:40·B&H(매수후보유).

세 전략 모두 `BacktestEngine.calc_nav()`를 거래일 단위로 반복 호출해 NAV 곡선을
만든다. 자산 순서·거래비용은 `config.yaml`에서만 읽는다(CLAUDE.md §2 — 하드코딩 금지).

회계 규약 (engine.py는 변경 없이 그대로 사용, PR #15 기준 start-of-day 리밸런싱):
    `calc_nav(prev_weights, new_weights, ...)`는 리밸런싱 비용을 prev_nav 기준으로
    먼저 매긴 뒤(turnover = |new_weights - prev_weights|), **새 비중(new_weights)**으로
    오늘 수익을 바로 실현한다(형우 env와 동일한 start-of-day 관점).
    즉 오늘 확정한 new_weights가 오늘 수익률을 실현하는 주체이므로, 그날 이후의 "실제
    보유 비중"은 이미 그날 수익률만큼 드리프트된 상태다. 이 두 인자에 무엇을 넣을지는
    호출자 책임이라 전략별로 아래처럼 구성한다.

    - 1/N·60:40(목표비중 고정): 매일 prev_weights=new_weights=고정 목표비중으로 호출.
      engine.py의 "여러 날을 순회하는 호출자는 다음 스텝의 prev_weights로 이번 스텝의
      new_weights를 드리프트 조정 없이 그대로 넘겨야 한다"는 규약을 그대로 따른 것 —
      목표비중 자체가 안 바뀌므로 이 값을 그대로 넘기면 첫날(초기 보유 SHV 100% → 목표
      비중 진입)에만 비용이 발생하고 이후로는 turnover=0.
    - B&H(리밸런싱 없음): "오늘 목표비중을 정하지 않는다"는 것 자체가 이 전략의 정의이므로,
      매일 new_weights = 오늘 실제 보유 비중(=어제 new_weights가 어제 수익률로 드리프트된
      값)으로 두어 prev_weights와 동일하게 만들어 turnover=0/cost=0을 보장한다. 그 드리프트
      값은 이 모듈이 `w_i·(1+r_i)/(1+포트폴리오수익률)` 공식으로 직접 계산해 다음 날의
      new_weights(=오늘 실제 보유 비중)로 사용한다 — engine.py에 넘기는 prev/new_weights의
      "그대로 이어받기" 규약은 그대로 지키면서, 그 이어받는 값 자체를 B&H 정의에 맞게
      매일 드리프트시키는 것이라 규약 위반이 아니다. start-of-day 모델이라 새 비중이
      진입 당일부터 바로 수익을 실현하므로, 드리프트는 진입일(day0)부터 계산한다.

    초기 보유는 첫날 진입 전 SHV(현금성) 100%에서 시작한다(PortfolioEnv.reset()과 동일 관례).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.backtest.engine import BacktestEngine
from src.config_loader import DEFAULT_CONFIG_PATH, get_assets, load_config

# 60:40 벤치마크의 자산 역할 구분 (자산 유니버스 고정 — CLAUDE.md §2)
STOCK_ASSETS = ("SPY", "EWY")
SAFE_ASSETS = ("TLT", "GLD", "SHV")
STOCK_ALLOC = 0.6
SAFE_ALLOC = 0.4

CASH_ASSET = "SHV"


def _validated_assets(price_returns: pd.DataFrame, config_path: str) -> list[str]:
    """price_returns 컬럼이 config 자산 순서와 일치하는지 검증하고 자산 목록을 반환한다."""
    assets = get_assets(load_config(config_path))
    if list(price_returns.columns) != assets:
        raise ValueError(
            f"price_returns 컬럼이 config 자산 순서와 다릅니다. 기대: {assets}, "
            f"실제: {list(price_returns.columns)} (CLAUDE.md §2 — 자산 순서 고정)"
        )
    return assets


def _initial_holding(assets: list[str]) -> np.ndarray:
    """첫날 진입 전 보유 비중 — SHV 100%에서 시작 (PortfolioEnv.reset()과 동일 관례)."""
    weights = np.zeros(len(assets))
    weights[assets.index(CASH_ASSET)] = 1.0
    return weights


def equal_weight_target(assets: list[str]) -> np.ndarray:
    """1/N(동일비중) 목표 비중."""
    n = len(assets)
    return np.full(n, 1.0 / n)


def sixty_forty_target(
    assets: list[str],
    stock_assets: tuple[str, ...] = STOCK_ASSETS,
    safe_assets: tuple[str, ...] = SAFE_ASSETS,
    stock_alloc: float = STOCK_ALLOC,
    safe_alloc: float = SAFE_ALLOC,
) -> np.ndarray:
    """60:40(주식군:안전자산군) 목표 비중 — 그룹 내 균등분배."""
    if not np.isclose(stock_alloc + safe_alloc, 1.0):
        raise ValueError("stock_alloc + safe_alloc는 1이어야 합니다.")
    if set(stock_assets) | set(safe_assets) != set(assets):
        raise ValueError("stock_assets·safe_assets를 합치면 assets 전체와 일치해야 합니다.")

    weights = np.zeros(len(assets))
    per_stock = stock_alloc / len(stock_assets)
    per_safe = safe_alloc / len(safe_assets)
    for a in stock_assets:
        weights[assets.index(a)] = per_stock
    for a in safe_assets:
        weights[assets.index(a)] = per_safe
    return weights


def _run_constant_target(
    price_returns: pd.DataFrame,
    engine: BacktestEngine,
    assets: list[str],
    target: np.ndarray,
) -> pd.DataFrame:
    """목표비중이 고정된 전략(1/N·60:40)의 공용 실행 루프."""
    weights = _initial_holding(assets)
    nav = engine.initial_nav

    records = []
    for date, row in price_returns.iterrows():
        r = row.to_numpy(dtype=np.float64)
        new_nav, cost = engine.calc_nav(nav, weights, target, r)
        turnover = float(np.abs(target - weights).sum())
        records.append(
            {"date": date, "nav": new_nav, "cost": cost, "turnover": turnover, **dict(zip(assets, target))}
        )
        nav = new_nav
        weights = target  # 매일 target으로 완전 리밸런싱 (기간 내 드리프트 무시)

    return pd.DataFrame.from_records(records, index="date")


def run_equal_weight(
    price_returns: pd.DataFrame,
    engine: BacktestEngine,
    config_path: str = DEFAULT_CONFIG_PATH,
) -> pd.DataFrame:
    """1/N(동일비중) 벤치마크 NAV 곡선. 매일 5개 자산에 균등(1/5씩) 리밸런싱한다."""
    assets = _validated_assets(price_returns, config_path)
    target = equal_weight_target(assets)
    return _run_constant_target(price_returns, engine, assets, target)


def run_sixty_forty(
    price_returns: pd.DataFrame,
    engine: BacktestEngine,
    config_path: str = DEFAULT_CONFIG_PATH,
    stock_assets: tuple[str, ...] = STOCK_ASSETS,
    safe_assets: tuple[str, ...] = SAFE_ASSETS,
    stock_alloc: float = STOCK_ALLOC,
    safe_alloc: float = SAFE_ALLOC,
) -> pd.DataFrame:
    """60:40(주식군:안전자산군) 벤치마크 NAV 곡선. 그룹 내부는 균등분배로 매일 리밸런싱한다."""
    assets = _validated_assets(price_returns, config_path)
    target = sixty_forty_target(assets, stock_assets, safe_assets, stock_alloc, safe_alloc)
    return _run_constant_target(price_returns, engine, assets, target)


def run_buy_and_hold(
    price_returns: pd.DataFrame,
    engine: BacktestEngine,
    config_path: str = DEFAULT_CONFIG_PATH,
    initial_target: np.ndarray | None = None,
) -> pd.DataFrame:
    """B&H(매수후보유) 벤치마크 NAV 곡선.

    첫날만 initial_target(기본 1/N)으로 진입하고, 이후로는 리밸런싱 없이 자산 가격
    변동에 따라 실제 보유 비중이 자연스럽게 드리프트한다. 리밸런싱이 없으므로 turnover와
    거래비용은 첫날 이후 항상 0이다.

    engine.py가 start-of-day 리밸런싱(새 비중이 그날 수익을 바로 실현)이므로, 첫날 진입한
    initial_target도 진입 당일 수익률만큼 바로 드리프트를 시작한다(둘째 날부터가 아니다).
    """
    assets = _validated_assets(price_returns, config_path)
    if initial_target is None:
        initial_target = equal_weight_target(assets)

    weights = _initial_holding(assets)  # 오늘 진입 전 실제 보유 비중
    nav = engine.initial_nav

    records = []
    for i, (date, row) in enumerate(price_returns.iterrows()):
        r = row.to_numpy(dtype=np.float64)
        # 첫날만 목표비중으로 진입(리밸런싱), 이후로는 "오늘 목표비중을 정하지 않음" = 어제
        # 드리프트된 실제 보유 비중을 그대로 오늘 비중으로 사용(turnover=0)
        target_today = initial_target if i == 0 else weights

        new_nav, cost = engine.calc_nav(nav, weights, target_today, r)
        turnover = float(np.abs(target_today - weights).sum())
        records.append(
            {"date": date, "nav": new_nav, "cost": cost, "turnover": turnover, **dict(zip(assets, target_today))}
        )

        # target_today가 오늘 수익률을 바로 실현했으므로(start-of-day), 다음 날 실제 보유
        # 비중은 오늘부터 드리프트된 값이다.
        portfolio_return = float(np.dot(target_today, r))
        weights = target_today * (1.0 + r) / (1.0 + portfolio_return)

        nav = new_nav

    return pd.DataFrame.from_records(records, index="date")
