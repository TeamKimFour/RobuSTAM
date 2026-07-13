"""BacktestEngine 테스트.

calc_nav의 핵심 수식(NAV 갱신·수수료 차감)을 검증한다.
config 로드는 _fake_cfg를 통해 실제 파일 없이도 실행된다.
"""

import numpy as np
import pandas as pd
import pytest

from src.backtest.engine import BacktestEngine


@pytest.fixture
def engine(tmp_path, monkeypatch):
    """config 파일 없이 BacktestEngine을 생성하는 픽스처."""
    import yaml
    from src import config_loader as cl

    cfg = {
        "assets": ["SPY", "EWY", "TLT", "GLD", "SHV"],
        "window": 30,
        "transaction_cost": 0.001,
    }

    # config 파일 생성
    config_file = tmp_path / "config.yaml"
    config_file.write_text(yaml.dump(cfg), encoding="utf-8")

    return BacktestEngine(initial_nav=1_000_000, config_path=str(config_file))


def test_nav_increases_on_positive_return(engine):
    """수익률이 양수이고 비중 변경이 없으면 NAV가 증가해야 한다."""
    prev_weights = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
    price_returns = np.array([0.01, 0.01, 0.01, 0.01, 0.01])

    new_nav, cost = engine.calc_nav(
        prev_nav=1_000_000,
        prev_weights=prev_weights,
        new_weights=prev_weights,  # 비중 변경 없음
        price_returns=price_returns,
    )

    assert new_nav > 1_000_000
    assert cost == 0.0  # 비중 변경 없으면 수수료 없음


def test_cost_deducted_on_weight_change(engine):
    """비중 변경이 있으면 수수료가 차감되어야 한다."""
    prev_weights = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
    new_weights = np.array([0.4, 0.1, 0.2, 0.2, 0.1])  # 비중 변경
    price_returns = np.zeros(5)  # 수익률 0 (수수료 효과만 확인)

    new_nav, cost = engine.calc_nav(
        prev_nav=1_000_000,
        prev_weights=prev_weights,
        new_weights=new_weights,
        price_returns=price_returns,
    )

    assert cost > 0
    assert new_nav < 1_000_000


def test_transaction_cost_loaded_from_config(engine):
    """config.yaml에서 읽은 거래비용률이 올바르게 적용되어야 한다."""
    assert engine.transaction_cost == 0.001


def test_nav_calculation_is_correct(engine):
    """NAV 갱신 수식을 수동으로 검증한다."""
    prev_nav = 1_000_000
    prev_weights = np.array([0.4, 0.0, 0.3, 0.2, 0.1])
    new_weights = np.array([0.6, 0.0, 0.2, 0.1, 0.1])
    price_returns = np.array([0.01, 0.0, -0.005, 0.002, 0.0])

    new_nav, cost = engine.calc_nav(
        prev_nav=prev_nav,
        prev_weights=prev_weights,
        new_weights=new_weights,
        price_returns=price_returns,
    )

    # 수동 계산 (start-of-day: 리밸런싱 → 수수료 차감 → 새 비중으로 수익 실현)
    turnover = np.sum(np.abs(new_weights - prev_weights))
    expected_cost = prev_nav * turnover * 0.001
    nav_after_cost = prev_nav - expected_cost
    expected_nav = nav_after_cost * (1 + np.dot(new_weights, price_returns))

    assert new_nav == pytest.approx(expected_nav)
    assert cost == pytest.approx(expected_cost)


def test_save_results_delegates_to_s3_results(engine):
    """save_results()가 자신의 config_path를 그대로 넘겨 s3_results에 위임하는지 확인."""
    pytest.importorskip("boto3")
    pytest.importorskip("moto")

    import boto3
    from moto import mock_aws

    nav = pd.DataFrame({"nav": [1_000_000.0, 1_010_000.0]})

    with mock_aws():
        s3 = boto3.client("s3", region_name="us-east-1")
        s3.create_bucket(Bucket="robustam-test")

        prefix = engine.save_results(
            nav, {"sharpe": 1.0}, run_id="e2e1", bucket="robustam-test", client=s3
        )

        assert prefix is not None
        keys = {o["Key"] for o in s3.list_objects_v2(Bucket="robustam-test")["Contents"]}
        assert f"{prefix}/nav.parquet" in keys
        assert f"{prefix}/config.yaml" in keys