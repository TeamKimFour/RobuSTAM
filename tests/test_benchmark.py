"""benchmark.py(1/N·60:40·B&H) 테스트.

목표 비중 산출 함수와, engine.calc_nav 연동 실행 루프(특히 B&H의 turnover/cost=0
불변식)를 검증한다. config 로드는 _fake_cfg를 통해 실제 파일 없이도 실행된다.
"""

import numpy as np
import pandas as pd
import pytest

from src.backtest.benchmark import (
    equal_weight_target,
    run_buy_and_hold,
    run_equal_weight,
    run_sixty_forty,
    sixty_forty_target,
)
from src.backtest.engine import BacktestEngine

ASSETS = ["SPY", "EWY", "TLT", "GLD", "SHV"]


@pytest.fixture
def config_path(tmp_path):
    """config 파일 없이 실행하기 위한 임시 config.yaml 픽스처."""
    import yaml

    cfg = {"assets": ASSETS, "window": 30, "transaction_cost": 0.001}
    config_file = tmp_path / "config.yaml"
    config_file.write_text(yaml.dump(cfg), encoding="utf-8")
    return str(config_file)


@pytest.fixture
def engine(config_path):
    return BacktestEngine(initial_nav=1_000_000, config_path=config_path)


@pytest.fixture
def price_returns():
    """3거래일치 자산별 수익률 (자산마다 다르게 움직여 드리프트를 검증할 수 있게 구성)."""
    return pd.DataFrame(
        [
            [0.02, 0.01, -0.01, 0.03, 0.0],
            [0.01, -0.02, 0.00, 0.01, 0.0],
            [-0.01, 0.00, 0.01, -0.02, 0.0],
        ],
        columns=ASSETS,
    )


# ── 목표 비중 산출 ──────────────────────────────────────────────


def test_equal_weight_target_sums_to_one():
    target = equal_weight_target(ASSETS)
    assert target == pytest.approx(np.full(5, 0.2))
    assert target.sum() == pytest.approx(1.0)


def test_sixty_forty_target_group_split():
    target = sixty_forty_target(ASSETS)
    # SPY, EWY = 주식군 60% 균등분배 → 각 0.3
    assert target[ASSETS.index("SPY")] == pytest.approx(0.3)
    assert target[ASSETS.index("EWY")] == pytest.approx(0.3)
    # TLT, GLD, SHV = 안전자산군 40% 균등분배 → 각 0.1333...
    for a in ("TLT", "GLD", "SHV"):
        assert target[ASSETS.index(a)] == pytest.approx(0.4 / 3)
    assert target.sum() == pytest.approx(1.0)


def test_sixty_forty_target_rejects_mismatched_groups():
    with pytest.raises(ValueError):
        sixty_forty_target(ASSETS, stock_assets=("SPY",), safe_assets=("TLT", "GLD", "SHV"))


# ── 1/N ──────────────────────────────────────────────────────────


def test_equal_weight_first_day_pays_entry_cost(price_returns, engine, config_path):
    result = run_equal_weight(price_returns, engine, config_path)
    assert result["cost"].iloc[0] > 0  # SHV 100% → 1/5 진입 비용


def test_equal_weight_zero_turnover_after_first_day(price_returns, engine, config_path):
    result = run_equal_weight(price_returns, engine, config_path)
    # 목표비중이 고정이므로 둘째 날부터는 리밸런싱이 필요 없다(turnover=0, cost=0)
    assert np.allclose(result["turnover"].iloc[1:], 0.0)
    assert np.allclose(result["cost"].iloc[1:], 0.0)


def test_equal_weight_weights_constant(price_returns, engine, config_path):
    result = run_equal_weight(price_returns, engine, config_path)
    for a in ASSETS:
        assert np.allclose(result[a], 0.2)


# ── 60:40 ────────────────────────────────────────────────────────


def test_sixty_forty_zero_turnover_after_first_day(price_returns, engine, config_path):
    result = run_sixty_forty(price_returns, engine, config_path)
    assert np.allclose(result["turnover"].iloc[1:], 0.0)
    assert np.allclose(result["cost"].iloc[1:], 0.0)


def test_sixty_forty_weights_match_group_split(price_returns, engine, config_path):
    result = run_sixty_forty(price_returns, engine, config_path)
    assert result["SPY"].iloc[0] == pytest.approx(0.3)
    assert result["SHV"].iloc[0] == pytest.approx(0.4 / 3)


# ── B&H ──────────────────────────────────────────────────────────


def test_buy_and_hold_zero_cost_after_first_day(price_returns, engine, config_path):
    result = run_buy_and_hold(price_returns, engine, config_path)
    assert result["cost"].iloc[0] > 0  # 첫날만 SHV 100% → 초기 목표비중 진입 비용 발생
    assert np.allclose(result["turnover"].iloc[1:], 0.0)
    assert np.allclose(result["cost"].iloc[1:], 0.0)


def test_buy_and_hold_weights_drift_with_prices(price_returns, engine, config_path):
    """리밸런싱이 없으므로 자산별 수익률 차이만큼 비중이 자연스럽게 흘러가야 한다.

    각 행의 비중은 "그 날 실제로 들고 들어가 그 날 수익률을 실현한 비중"을 기록한다.
    engine.py가 start-of-day 리밸런싱이라 진입일(day0)의 initial_target도 day0 수익률을
    바로 실현하므로, day0 자체는 아직 드리프트 전(방금 정한 target)이지만 day1부터는
    이미 day0 수익률만큼 드리프트된 값이 관찰된다.
    """
    result = run_buy_and_hold(price_returns, engine, config_path)

    # day0은 initial_target(1/N) 그대로 (드리프트는 day0 수익률을 실현한 뒤 발생)
    assert result["GLD"].iloc[0] == pytest.approx(0.2)

    # day1: day0 수익률(GLD +3%가 최고, TLT -1%가 최저)을 반영해
    # GLD 비중은 1/5보다 커지고 TLT 비중은 1/5보다 작아진다.
    assert result["GLD"].iloc[1] > 0.2
    assert result["TLT"].iloc[1] < 0.2

    # 매일 비중의 합은 항상 1이어야 한다(드리프트여도 정규화된 비중이므로).
    weight_cols = ASSETS
    row_sums = result[weight_cols].sum(axis=1)
    assert np.allclose(row_sums.to_numpy(), np.ones(len(result)))


def test_buy_and_hold_matches_manual_drift_formula(price_returns, engine, config_path):
    """이후 일차 비중이 w_i*(1+r_i)/(1+포트폴리오수익률) 수식과 일치하는지 직접 검증한다.

    start-of-day 모델이라 initial_target은 day0 수익률을 바로 실현하므로, day1 행은
    이미 day0 수익률로 드리프트된 값이어야 한다. day2 행은 day1 값이 day1 수익률로
    한 번 더 드리프트된(복리) 값이어야 한다.
    """
    result = run_buy_and_hold(price_returns, engine, config_path)

    target0 = np.full(5, 0.2)
    r0 = price_returns.iloc[0].to_numpy()
    portfolio_return_0 = np.dot(target0, r0)
    expected_day1 = target0 * (1 + r0) / (1 + portfolio_return_0)

    for i, a in enumerate(ASSETS):
        assert result[a].iloc[1] == pytest.approx(expected_day1[i])

    r1 = price_returns.iloc[1].to_numpy()
    portfolio_return_1 = np.dot(expected_day1, r1)
    expected_day2 = expected_day1 * (1 + r1) / (1 + portfolio_return_1)

    for i, a in enumerate(ASSETS):
        assert result[a].iloc[2] == pytest.approx(expected_day2[i])


# ── 자산 순서 검증 ─────────────────────────────────────────────────


def test_rejects_mismatched_asset_order(engine, config_path):
    bad_returns = pd.DataFrame(
        [[0.01, 0.01, 0.01, 0.01, 0.01]],
        columns=["EWY", "SPY", "TLT", "GLD", "SHV"],  # 순서 뒤바뀜
    )
    with pytest.raises(ValueError):
        run_equal_weight(bad_returns, engine, config_path)
