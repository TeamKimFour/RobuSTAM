"""src.backtest.export 테스트 — FE JSON 계약과 fold 연속화 로직을 검증한다.

실제 모델·Feature Store 없이 `run_fold_fn`을 가짜로 주입해, 익스포터가
NAV를 이어붙이고 지표를 재계산해서 계약대로 JSON을 조립하는지만 확인한다.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from src.backtest import export as ex


def _fake_nav(dates: pd.DatetimeIndex, initial_nav: float, growth: float) -> pd.DataFrame:
    """(nav, turnover, cost) 컬럼을 가진 fake NAV DataFrame. NAV는 매일 growth 배로 증가."""
    factors = np.array([(1.0 + growth) ** i for i in range(len(dates))])
    nav = initial_nav * factors
    return pd.DataFrame({"nav": nav, "turnover": 0.02, "cost": 100.0}, index=dates)


def _fake_run_fold_factory(initial_nav: float):
    """fold별 fake 결과를 반환하는 run_fold_fn 대체품을 만든다."""
    periods = {
        1: pd.bdate_range("2020-01-02", periods=5),
        2: pd.bdate_range("2022-01-03", periods=5),
        3: pd.bdate_range("2024-01-02", periods=5),
    }
    # RL policy만 다른 fold별 성장률로, 나머지는 동일.
    growth_map = {"RL policy": (0.01, 0.005, 0.002), "1/N": (0.003,) * 3,
                  "60:40": (0.004,) * 3, "B&H": (0.006,) * 3}

    def fake(fold_id, model, config_path, split, *, initial_nav=initial_nav):
        idx = periods[fold_id]
        nav_by_strategy = {
            name: _fake_nav(idx, initial_nav, growth_map[name][fold_id - 1])
            for name in ex.STRATEGY_ORDER
        }
        strategies = {
            name: {
                "total_return": 0.1, "sharpe": 1.0, "mdd": -0.05,
                "avg_turnover": 0.02, "total_cost": 500.0,
            }
            for name in ex.STRATEGY_ORDER
        }
        return {
            "fold_id": fold_id,
            "period": (str(idx[0].date()), str(idx[-1].date())),
            "strategies": strategies,
            "nav_by_strategy": nav_by_strategy,
            "s3_prefixes": {name: None for name in ex.STRATEGY_ORDER},
            "comparison": {name: {"beats_target": False, "sharpe_improvement_pct": 0.0,
                                  "mdd_defense_pct": 0.0} for name in ex.STRATEGY_ORDER
                           if name != "RL policy"},
        }

    return fake


@pytest.fixture()
def cfg_path(tmp_path):
    cfg = {
        "assets": ["SPY", "EWY", "TLT", "GLD", "SHV"],
        "window": 30,
        "transaction_cost": 0.001,
        "split": {"test_blocks": [
            ["2020-01-01", "2021-12-31"],
            ["2022-01-01", "2023-12-31"],
            ["2024-01-01", "2025-12-31"],
        ]},
    }
    p = tmp_path / "config.yaml"
    p.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    return str(p)


# ── ① 순수 함수: NAV 이어붙이기 ──
def test_concat_folds_multiplies_by_previous_end_value():
    """fold2의 첫 값 1.0은 fold1의 마지막 값(=1.2)만큼 스케일링돼 1.2가 돼야 한다."""
    fold1 = [{"date": "2020-01-01", "value": 1.0}, {"date": "2020-01-02", "value": 1.2}]
    fold2 = [{"date": "2022-01-01", "value": 1.0}, {"date": "2022-01-02", "value": 1.1}]
    out = ex._concat_folds([fold1, fold2])
    assert [p["value"] for p in out] == pytest.approx([1.0, 1.2, 1.2, 1.32])


def test_concat_folds_skips_empty_folds():
    out = ex._concat_folds([[], [{"date": "2020-01-01", "value": 1.0}]])
    assert len(out) == 1


# ── ② 순수 함수: 지표 재계산 ──
def test_metrics_from_points_recovers_analytic_values():
    """단조 증가 곡선의 total_return·MDD·CAGR을 이론값과 대조한다.

    Sharpe는 부동소수 잡음(1.01·1.01·... 누적)으로 σ가 0이 아니게 나오므로 값을 못 박지 않고
    다른 지표만 확인한다. std=0 경계는 아래 별도 테스트에서 다룬다.
    """
    pts = [{"date": f"2020-01-{i+1:02d}", "value": 1.01 ** i} for i in range(20)]
    m = ex._metrics_from_points(pts)
    assert m["total_return"] == pytest.approx(1.01 ** 19 - 1.0)
    assert m["mdd"] == pytest.approx(0.0, abs=1e-12)  # 단조 증가라 낙폭 0
    # CAGR: 19영업일 상승, 연 252영업일 기준
    expected_cagr = (1.01 ** 19) ** (252 / 19) - 1.0
    assert m["cagr"] == pytest.approx(expected_cagr, rel=1e-6)


def test_metrics_from_points_returns_zero_on_flat_curve():
    """모든 값이 동일하면 std=0 경계 → sharpe·vol 모두 0."""
    pts = [{"date": f"2020-01-{i+1:02d}", "value": 1.0} for i in range(10)]
    m = ex._metrics_from_points(pts)
    assert m["vol"] == 0.0
    assert m["sharpe"] == 0.0
    assert m["mdd"] == 0.0
    assert m["total_return"] == 0.0


def test_metrics_from_points_handles_short_series():
    assert ex._metrics_from_points([{"date": "2020-01-01", "value": 1.0}]) == {
        "cagr": 0.0, "sharpe": 0.0, "mdd": 0.0, "vol": 0.0, "total_return": 0.0,
    }


# ── ③ _summarize_verdict: 벤치마크별 fold 통과 집계 (합격/불합격 이분법 없음) ──
def test_summarize_verdict_counts_passed_folds():
    fold_results = [
        {"comparison": {"1/N": {"beats_target": True}, "60:40": {"beats_target": False},
                        "B&H": {"beats_target": True}}},
        {"comparison": {"1/N": {"beats_target": True}, "60:40": {"beats_target": False},
                        "B&H": {"beats_target": False}}},
        {"comparison": {"1/N": {"beats_target": False}, "60:40": {"beats_target": False},
                        "B&H": {"beats_target": True}}},
    ]
    out = ex._summarize_verdict(fold_results)

    assert out["1/N"] == {"folds_passed": 2, "folds_total": 3, "pass_rate": pytest.approx(2 / 3)}
    assert out["60:40"] == {"folds_passed": 0, "folds_total": 3, "pass_rate": 0.0}
    assert out["B&H"] == {"folds_passed": 2, "folds_total": 3, "pass_rate": pytest.approx(2 / 3)}
    # 이분법 합격/불합격 필드는 넣지 않는다 — beats_target을 fold 개수·비율로만 집계한다.
    for v in out.values():
        assert "overall_pass" not in v


def test_summarize_verdict_handles_empty_fold_results():
    """fold가 하나도 없으면 0으로 나누기 없이 folds_total=0·pass_rate=0.0이어야 한다."""
    out = ex._summarize_verdict([])
    for name in ("1/N", "60:40", "B&H"):
        assert out[name] == {"folds_passed": 0, "folds_total": 0, "pass_rate": 0.0}


# ── ④ build_export: 계약 준수 ──
def test_build_export_produces_valid_contract(cfg_path):
    bundle = ex.build_export(
        model=None, config_path=cfg_path, initial_nav=1_000_000,
        run_fold_fn=_fake_run_fold_factory(1_000_000),
    )

    # 최상위 키
    for key in (
        "generated_at", "initial_nav", "periods", "strategies",
        "fold_table", "comparison", "verdict_summary",
    ):
        assert key in bundle, f"missing key: {key}"

    # _fake_run_fold_factory는 3 fold 전부 beats_target=False로 고정 — 통과 0건이어야 한다.
    for name in ("1/N", "60:40", "B&H"):
        assert bundle["verdict_summary"][name] == {
            "folds_passed": 0, "folds_total": 3, "pass_rate": 0.0,
        }

    # 3 fold × 4 전략
    assert len(bundle["periods"]) == 3
    assert [p["fold_id"] for p in bundle["periods"]] == [1, 2, 3]
    assert len(bundle["strategies"]) == 4
    assert [s["name"] for s in bundle["strategies"]] == list(ex.STRATEGY_ORDER)

    # 각 전략은 nav·metrics 필드
    for s in bundle["strategies"]:
        assert s["color"].startswith("#")
        assert len(s["nav"]) == 15  # 5일 × 3 fold, 이어붙임
        assert s["nav"][0]["value"] == pytest.approx(1.0)  # 초기 정규화
        for k in ("cagr", "sharpe", "mdd", "vol", "total_return", "avg_turnover", "total_cost"):
            assert k in s["metrics"]

    # fold_table은 RL policy 지표
    assert len(bundle["fold_table"]) == 3
    for row in bundle["fold_table"]:
        assert set(row.keys()) == {"fold_id", "period", "sharpe", "cagr", "mdd"}


def test_build_export_nav_is_monotonic_when_all_folds_growing(cfg_path):
    """모든 fold가 상승 곡선이면 이어붙인 NAV도 단조 증가해야 한다."""
    bundle = ex.build_export(
        model=None, config_path=cfg_path, initial_nav=1_000_000,
        run_fold_fn=_fake_run_fold_factory(1_000_000),
    )
    for s in bundle["strategies"]:
        values = [p["value"] for p in s["nav"]]
        assert all(values[i + 1] >= values[i] for i in range(len(values) - 1)), s["name"]


# ── ⑤ write_export: 원자적 쓰기 ──
def test_write_export_creates_file_with_valid_json(tmp_path, cfg_path):
    bundle = ex.build_export(
        model=None, config_path=cfg_path, initial_nav=1_000_000,
        run_fold_fn=_fake_run_fold_factory(1_000_000),
    )
    out = tmp_path / "sub" / "backtest.json"
    ex.write_export(bundle, str(out))
    assert out.is_file()
    parsed = json.loads(out.read_text(encoding="utf-8"))
    assert parsed["initial_nav"] == 1_000_000
    assert not (tmp_path / "sub" / "backtest.json.tmp").exists()  # 임시파일 잔여 없음


# ── ⑥ upload_to_s3: S3 키 계약·skip 동작 ──
def test_s3_key_matches_frontend_contract():
    """FE prebuild(scripts/fetch-backtest.mjs)와 반드시 같은 키 규칙을 써야 한다."""
    assert ex._s3_key("robustam") == "robustam/frontend/backtest.json"
    assert ex._s3_key("") == "frontend/backtest.json"
    assert ex._s3_key("/robustam/") == "robustam/frontend/backtest.json"  # 슬래시 trim


@pytest.fixture()
def _cfg_with_bucket(tmp_path):
    cfg = {
        "assets": ["SPY", "EWY", "TLT", "GLD", "SHV"],
        "window": 30,
        "transaction_cost": 0.001,
        "split": {"test_blocks": [["2020-01-01", "2021-12-31"]]},
        "data": {"s3_bucket": "test-bucket", "s3_prefix": "robustam"},
    }
    p = tmp_path / "cfg.yaml"
    p.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    return str(p)


def test_upload_skips_when_bucket_missing(tmp_path, monkeypatch, capsys):
    """버킷이 config·환경변수 어디에도 없으면 조용히 skip한다."""
    monkeypatch.delenv("S3_BUCKET", raising=False)
    cfg = {"data": {}, "assets": ["SPY", "EWY", "TLT", "GLD", "SHV"],
           "window": 30, "transaction_cost": 0.001}
    p = tmp_path / "cfg.yaml"
    p.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    payload = tmp_path / "backtest.json"
    payload.write_text('{"generated_at": "x"}', encoding="utf-8")

    url = ex.upload_to_s3(str(payload), str(p))
    assert url is None
    assert "skip" in capsys.readouterr().out.lower()


def test_upload_puts_object_at_frontend_key(tmp_path, monkeypatch, _cfg_with_bucket):
    """계약 키(`{prefix}/frontend/backtest.json`)에 정확히 올라간다 (moto)."""
    moto = pytest.importorskip("moto")
    boto3 = pytest.importorskip("boto3")
    from moto import mock_aws

    monkeypatch.delenv("S3_BUCKET", raising=False)
    monkeypatch.delenv("S3_PREFIX", raising=False)
    payload = tmp_path / "backtest.json"
    payload.write_bytes(b'{"strategies": []}')

    with mock_aws():
        s3 = boto3.client("s3", region_name="us-east-1")
        s3.create_bucket(Bucket="test-bucket")
        url = ex.upload_to_s3(str(payload), _cfg_with_bucket, client=s3)

        assert url is not None
        assert url.endswith("robustam/frontend/backtest.json")
        obj = s3.get_object(Bucket="test-bucket", Key="robustam/frontend/backtest.json")
        assert obj["Body"].read() == b'{"strategies": []}'
        assert obj["ContentType"] == "application/json"


def test_upload_env_bucket_overrides_config(tmp_path, monkeypatch, _cfg_with_bucket):
    """S3_BUCKET 환경변수가 config의 버킷을 override한다(s3_sync와 같은 관례)."""
    moto = pytest.importorskip("moto")
    boto3 = pytest.importorskip("boto3")
    from moto import mock_aws

    monkeypatch.setenv("S3_BUCKET", "override-bucket")
    monkeypatch.delenv("S3_PREFIX", raising=False)
    payload = tmp_path / "backtest.json"
    payload.write_bytes(b'{}')

    with mock_aws():
        s3 = boto3.client("s3", region_name="us-east-1")
        s3.create_bucket(Bucket="override-bucket")
        url = ex.upload_to_s3(str(payload), _cfg_with_bucket, client=s3)

    assert url is not None
    assert "override-bucket" in url
