"""src.backtest.runner 테스트.

핵심은 네 가지다:
  ① fold 개수·run_id 네이밍 규칙(fold{N}_policy/1n/60_40/bh, 팀 확정)이 정확한가
  ② 하이브리드 판정(성과: 샤프 15%+ 개선 또는 MDD 20%+ 방어 / 정상성: 비중편차·회전율,
     이슈 #46 팀 확정)이 올바르게 계산되는가
  ③ run_fold()가 policy+벤치마크 3종을 각각 별도 run으로 S3에 저장하는가(moto)
  ④ weight_deviation(policy.summarize()) 계산이 손계산과 일치하는가
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import yaml

from src.backtest.policy import summarize
from src.backtest.runner import (
    TURNOVER_TARGET,
    WEIGHT_DEVIATION_TARGET,
    _compare_to_benchmarks,
    _fold_ids,
    _run_id,
    run_fold,
)
from src.data import feature_store as fs
from src.data import schema

# 성과 조건(샤프/MDD)과 무관하게 정상성 게이트를 항상 통과시키는 값(테스트 헬퍼용).
_SANE_POLICY_EXTRAS = {"weight_deviation": 0.10, "avg_turnover": 0.10}

ASSETS = ["SPY", "EWY", "TLT", "GLD", "SHV"]
FWD_RET_COLS = [f"fwd_ret_{a}" for a in ASSETS]
W = 30


# ── ① fold 개수·run_id 네이밍 ──
def test_fold_ids_from_split_test_blocks():
    cfg = {"split": {"test_blocks": [["2020-01-01", "2021-12-31"], ["2022-01-01", "2023-12-31"]]}}
    assert _fold_ids(cfg) == [1, 2]


def test_run_id_naming_matches_team_convention():
    assert _run_id(1, "policy") == "fold1_policy"
    assert _run_id(2, "1n") == "fold2_1n"
    assert _run_id(3, "60_40") == "fold3_60_40"
    assert _run_id(1, "bh") == "fold1_bh"


# ── ② 하이브리드 판정(성과 AND 정상성, 이슈 #46) ──
def test_compare_beats_target_via_sharpe_improvement():
    strategies = {
        "RL policy": {"sharpe": 1.20, "mdd": -0.10, **_SANE_POLICY_EXTRAS},
        "1/N": {"sharpe": 1.00, "mdd": -0.10},  # 샤프 개선 20% ≥ 15% → 성과 조건 달성
    }
    out = _compare_to_benchmarks(strategies)
    assert out["1/N"]["sharpe_improvement_pct"] == pytest.approx(0.20)
    assert out["1/N"]["mdd_defense_pct"] == pytest.approx(0.0)
    assert out["1/N"]["weight_deviation_ok"] is True
    assert out["1/N"]["turnover_ok"] is True
    assert out["1/N"]["beats_target"] is True


def test_compare_beats_target_via_mdd_defense():
    strategies = {
        "RL policy": {"sharpe": 0.50, "mdd": -0.08, **_SANE_POLICY_EXTRAS},
        "60:40": {"sharpe": 0.55, "mdd": -0.10},  # MDD 방어 20% ≥ 20% → 달성(샤프는 오히려 후퇴)
    }
    out = _compare_to_benchmarks(strategies)
    assert out["60:40"]["mdd_defense_pct"] == pytest.approx(0.20)
    assert out["60:40"]["sharpe_improvement_pct"] < 0
    assert out["60:40"]["beats_target"] is True


def test_compare_misses_target_when_both_fall_short():
    strategies = {
        "RL policy": {"sharpe": 1.00, "mdd": -0.10, **_SANE_POLICY_EXTRAS},
        "B&H": {"sharpe": 0.95, "mdd": -0.11},  # 둘 다 미미한 개선 — 성과 조건 미달
    }
    out = _compare_to_benchmarks(strategies)
    assert out["B&H"]["beats_target"] is False


def test_compare_guards_zero_division():
    strategies = {
        "RL policy": {"sharpe": 0.5, "mdd": -0.05, **_SANE_POLICY_EXTRAS},
        "1/N": {"sharpe": 0.0, "mdd": 0.0},  # 0으로 나누기 방지 — nan, beats_target은 False
    }
    out = _compare_to_benchmarks(strategies)
    assert np.isnan(out["1/N"]["sharpe_improvement_pct"])
    assert np.isnan(out["1/N"]["mdd_defense_pct"])
    assert out["1/N"]["beats_target"] is False


def test_compare_fails_when_weight_deviation_too_low_despite_good_performance():
    """성과 조건은 통과해도 비중편차가 60:40을 그대로 베낀 수준이면 정상성 게이트에서 떨어진다."""
    strategies = {
        "RL policy": {
            "sharpe": 1.20, "mdd": -0.10,
            "weight_deviation": WEIGHT_DEVIATION_TARGET - 0.01,  # 하한 미달(패시브 흉내)
            "avg_turnover": 0.10,
        },
        "1/N": {"sharpe": 1.00, "mdd": -0.10},  # 샤프 개선 20% — 성과만 보면 달성
    }
    out = _compare_to_benchmarks(strategies)
    assert out["1/N"]["weight_deviation_ok"] is False
    assert out["1/N"]["turnover_ok"] is True
    assert out["1/N"]["beats_target"] is False


def test_compare_fails_when_turnover_too_high_despite_good_performance():
    """성과 조건은 통과해도 회전율이 과매매 상한을 넘으면 정상성 게이트에서 떨어진다."""
    strategies = {
        "RL policy": {
            "sharpe": 1.20, "mdd": -0.10,
            "weight_deviation": 0.10,
            "avg_turnover": TURNOVER_TARGET + 0.1,  # 과매매(상한 초과)
        },
        "1/N": {"sharpe": 1.00, "mdd": -0.10},
    }
    out = _compare_to_benchmarks(strategies)
    assert out["1/N"]["weight_deviation_ok"] is True
    assert out["1/N"]["turnover_ok"] is False
    assert out["1/N"]["beats_target"] is False


def test_compare_weight_deviation_and_turnover_ok_are_shared_across_benchmarks():
    """정상성 플래그는 policy 자체의 속성이라 벤치마크가 몇 개든 동일한 값이어야 한다."""
    strategies = {
        "RL policy": {"sharpe": 1.20, "mdd": -0.10, **_SANE_POLICY_EXTRAS},
        "1/N": {"sharpe": 1.00, "mdd": -0.10},
        "60:40": {"sharpe": 1.00, "mdd": -0.10},
        "B&H": {"sharpe": 1.00, "mdd": -0.10},
    }
    out = _compare_to_benchmarks(strategies)
    assert {out[name]["weight_deviation_ok"] for name in ("1/N", "60:40", "B&H")} == {True}
    assert {out[name]["turnover_ok"] for name in ("1/N", "60:40", "B&H")} == {True}


# ── ④ weight_deviation 계산 (policy.summarize) ──
def test_summarize_weight_deviation_matches_hand_calculation():
    """policy 비중(SPY=EWY=0.5, 나머지 0)의 60:40 대비 편차를 손계산과 대조한다.

    60:40 목표(sixty_forty_target)는 SPY=EWY=0.3, TLT=GLD=SHV=0.4/3(=2/15)다.
    Active Share = 0.5 * (|0.5-0.3|*2 + |0-2/15|*3) = 0.5 * (0.4+0.4) = 0.4.
    """
    dates = pd.bdate_range("2020-01-01", periods=2)
    nav_df = pd.DataFrame(
        {
            "nav": [1000.0, 1010.0],
            "cost": [0.0, 0.0],
            "turnover": [0.0, 0.0],
            "SPY": [0.5, 0.5],
            "EWY": [0.5, 0.5],
            "TLT": [0.0, 0.0],
            "GLD": [0.0, 0.0],
            "SHV": [0.0, 0.0],
        },
        index=dates,
    )
    m = summarize(nav_df, initial_nav=1000.0)
    assert m["weight_deviation"] == pytest.approx(0.4)


def test_summarize_weight_deviation_zero_when_matching_benchmark():
    """policy 비중이 60:40 그 자체면 편차는 0이어야 한다."""
    dates = pd.bdate_range("2020-01-01", periods=1)
    nav_df = pd.DataFrame(
        {
            "nav": [1000.0], "cost": [0.0], "turnover": [0.0],
            "SPY": [0.3], "EWY": [0.3], "TLT": [2 / 15], "GLD": [2 / 15], "SHV": [2 / 15],
        },
        index=dates,
    )
    m = summarize(nav_df, initial_nav=1000.0)
    assert m["weight_deviation"] == pytest.approx(0.0, abs=1e-9)


# ── ③ run_fold — Feature Store fixture + moto S3 ──
def _write_fake_fold(out_dir: str, fold_id: int, split: str, n_rows: int = 8, seed: int = 0):
    cols = schema.feature_names(W)
    dates = pd.date_range("2020-01-01", periods=n_rows, freq="B")
    rng = np.random.default_rng(seed)

    state_df = pd.DataFrame(
        rng.normal(0.0, 1.0, size=(n_rows, len(cols))).astype(np.float32), index=dates, columns=cols
    )
    targets_df = pd.DataFrame(
        rng.normal(0.0, 0.01, size=(n_rows, len(ASSETS))).astype(np.float32),
        index=dates,
        columns=FWD_RET_COLS,
    )
    fs.write_features(state_df, out_dir, fold_id, split)
    fs.write_targets(targets_df, out_dir, fold_id, split)


def _write_config(tmp_path, feature_store_dir: str, s3_bucket: str = "") -> str:
    cfg = {
        "assets": ASSETS,
        "window": W,
        "transaction_cost": 0.001,
        "data": {"feature_store_dir": feature_store_dir, "s3_bucket": s3_bucket},
    }
    path = tmp_path / "config.yaml"
    path.write_text(yaml.dump(cfg, allow_unicode=True), encoding="utf-8")
    return str(path)


class _ConstLogitModel:
    def __init__(self, logits):
        self.logits = np.asarray(logits, dtype=np.float32)

    def predict(self, obs, deterministic: bool = True):
        return self.logits, None


def test_run_fold_returns_four_strategies_and_comparison(tmp_path):
    out_dir = str(tmp_path / "feature_store")
    _write_fake_fold(out_dir, fold_id=1, split="test")
    config_path = _write_config(tmp_path, out_dir)
    model = _ConstLogitModel(np.zeros(5))

    result = run_fold(1, model, config_path, "test", initial_nav=1000.0)

    assert result["fold_id"] == 1
    assert set(result["strategies"]) == {"RL policy", "1/N", "60:40", "B&H"}
    for m in result["strategies"].values():
        assert set(m) == {
            "total_return", "sharpe", "mdd", "avg_turnover", "total_cost", "weight_deviation",
        }
    assert set(result["comparison"]) == {"1/N", "60:40", "B&H"}
    for c in result["comparison"].values():
        assert {"weight_deviation_ok", "turnover_ok"} <= set(c)
    # S3 버킷 미설정 → 전부 skip(None)
    assert all(v is None for v in result["s3_prefixes"].values())


@pytest.mark.parametrize("has_moto", [True])
def test_run_fold_saves_four_separate_s3_runs(tmp_path, has_moto):
    pytest.importorskip("boto3")
    pytest.importorskip("moto")
    import boto3
    from moto import mock_aws

    with mock_aws():
        bucket = "robustam-test"
        s3 = boto3.client("s3", region_name="us-east-1")
        s3.create_bucket(Bucket=bucket)

        out_dir = str(tmp_path / "feature_store")
        _write_fake_fold(out_dir, fold_id=1, split="test")
        config_path = _write_config(tmp_path, out_dir, s3_bucket=bucket)
        model = _ConstLogitModel(np.zeros(5))

        result = run_fold(1, model, config_path, "test", initial_nav=1000.0)

        assert result["s3_prefixes"]["RL policy"] is not None
        prefixes = set(result["s3_prefixes"].values())
        assert len(prefixes) == 4  # 4개 run이 서로 다른 prefix(같은 run_date라도 run_id가 다름)

        got_keys = {o["Key"] for o in s3.list_objects_v2(Bucket=bucket, Prefix="backtests/")["Contents"]}
        for suffix in ("fold1_policy", "fold1_1n", "fold1_60_40", "fold1_bh"):
            assert any(suffix in k for k in got_keys), f"{suffix} 관련 키가 S3에 없음: {got_keys}"
