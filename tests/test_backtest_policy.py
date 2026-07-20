"""정책→백테스트 어댑터 테스트 (src/backtest/policy.py).

두 계약 사이의 변환이 핵심이라 그 지점들을 집중적으로 박제한다.
  ① targets(로그수익률) → 엔진용 단순수익률(expm1) + 컬럼 리네임
  ② 적재 State의 prev_weight(=0)를 매 스텝 직전 비중으로 덮어쓰는지
  ③ 로짓 → softmax 비중, 출력 포맷이 벤치마크와 동일한지
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.backtest.engine import BacktestEngine
from src.backtest.policy import run_policy, targets_to_price_returns
from src.data import schema

ASSETS = ["SPY", "EWY", "TLT", "GLD", "SHV"]
W = 30


def _state(n_rows: int, start: str = "2020-01-01") -> pd.DataFrame:
    cols = schema.feature_names(W)
    idx = pd.date_range(start, periods=n_rows, freq="B")
    # 적재 State의 prev_weight 칸은 build 산출물과 동일하게 0으로 둔다(핵심 전제).
    data = np.zeros((n_rows, len(cols)), dtype=np.float32)
    return pd.DataFrame(data, index=idx, columns=cols)


def _targets(index, log_rets: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame(log_rets, index=index, columns=[f"fwd_ret_{a}" for a in ASSETS])


class _ConstLogitModel:
    """항상 같은 로짓을 내는 정책 — softmax 결과를 예측 가능하게 만든다."""

    def __init__(self, logits):
        self.logits = np.asarray(logits, dtype=np.float32)
        self.seen_prev_weights = []

    def predict(self, obs, deterministic: bool = True):
        # 어댑터가 주입한 prev_weight를 기록해 ②를 검증할 수 있게 한다.
        self.seen_prev_weights.append(np.array(obs[schema.prev_weight_slice(W)]))
        return self.logits, None


# ── ① 로그 → 단순수익률 변환 ──
def test_targets_to_price_returns_converts_and_renames():
    idx = pd.date_range("2020-01-01", periods=3, freq="B")
    log_rets = np.array([[0.01, -0.02, 0.0, 0.005, 0.0001]] * 3)
    tgt = _targets(idx, log_rets)

    out = targets_to_price_returns(tgt, ASSETS)

    assert list(out.columns) == ASSETS  # 컬럼 리네임
    np.testing.assert_allclose(out.to_numpy(), np.expm1(log_rets))  # expm1 변환
    # 로그수익률과 단순수익률은 다른 값이어야 한다(변환이 실제로 일어났는지)
    assert not np.allclose(out.to_numpy(), log_rets)


def test_targets_missing_column_raises():
    idx = pd.date_range("2020-01-01", periods=2, freq="B")
    tgt = _targets(idx, np.zeros((2, 5))).drop(columns=["fwd_ret_TLT"])
    with pytest.raises(ValueError, match="필요한 컬럼이 없습니다"):
        targets_to_price_returns(tgt, ASSETS)


# ── ② prev_weight 주입 (빠뜨리기 쉬운 핵심 계약) ──
def test_prev_weight_is_injected_into_obs():
    """정책이 보는 obs의 prev_weight가 직전 스텝 비중과 일치해야 한다.

    Feature Store의 State는 prev_weight가 0이라, 주입을 빠뜨리면 정책이 계속 0을 본다.
    백테스트가 조용히 학습과 어긋나는 지점이라 명시적으로 박제한다.
    """
    n = 4
    state = _state(n)
    pr = pd.DataFrame(np.zeros((n, 5)), index=state.index, columns=ASSETS)
    # 균등 로짓 → softmax는 균등비중(0.2씩)
    model = _ConstLogitModel(np.zeros(5))

    run_policy(pr, state, model, BacktestEngine(initial_nav=1.0))

    seen = model.seen_prev_weights
    assert len(seen) == n
    # 첫 스텝은 콜드스타트 SHV 100%
    np.testing.assert_allclose(seen[0], [0, 0, 0, 0, 1], atol=1e-6)
    # 이후 스텝은 직전 스텝의 목표비중(균등)을 봐야 한다 — 0이면 주입 누락
    for prev in seen[1:]:
        np.testing.assert_allclose(prev, [0.2] * 5, atol=1e-6)
        assert prev.sum() > 0


# ── ③ 출력 포맷·회계 ──
def test_output_schema_matches_benchmarks():
    n = 3
    state = _state(n)
    pr = pd.DataFrame(np.zeros((n, 5)), index=state.index, columns=ASSETS)
    model = _ConstLogitModel(np.zeros(5))

    nav_df = run_policy(pr, state, model, BacktestEngine(initial_nav=1000.0))

    assert list(nav_df.columns) == ["nav", "cost", "turnover", *ASSETS]
    assert nav_df.index.name == "date"
    assert len(nav_df) == n
    # 비중 합 = 1
    np.testing.assert_allclose(nav_df[ASSETS].sum(axis=1).to_numpy(), 1.0, atol=1e-6)


def test_first_step_turnover_and_cost_from_cold_start():
    """SHV 100% → 균등비중 진입이면 첫날 turnover=1.6, 비용은 그만큼 발생한다."""
    state = _state(1)
    pr = pd.DataFrame(np.zeros((1, 5)), index=state.index, columns=ASSETS)
    model = _ConstLogitModel(np.zeros(5))
    engine = BacktestEngine(initial_nav=1000.0)

    nav_df = run_policy(pr, state, model, engine)

    # |0.2-0| x4 + |0.2-1| = 0.8 + 0.8 = 1.6
    assert nav_df["turnover"].iloc[0] == pytest.approx(1.6, abs=1e-6)
    assert nav_df["cost"].iloc[0] == pytest.approx(1000.0 * 1.6 * engine.transaction_cost)


def test_zero_returns_only_lose_transaction_cost():
    """수익률 0이면 NAV 감소분은 정확히 거래비용이어야 한다(회계 검증)."""
    n = 2
    state = _state(n)
    pr = pd.DataFrame(np.zeros((n, 5)), index=state.index, columns=ASSETS)
    model = _ConstLogitModel(np.zeros(5))
    engine = BacktestEngine(initial_nav=1000.0)

    nav_df = run_policy(pr, state, model, engine)

    # 2일차는 비중이 그대로라 turnover=0 → 비용 0 → NAV 불변
    assert nav_df["turnover"].iloc[1] == pytest.approx(0.0, abs=1e-9)
    assert nav_df["nav"].iloc[1] == pytest.approx(nav_df["nav"].iloc[0])


def test_index_intersection_when_targets_shorter():
    """targets는 익일 수익률 없는 마지막 행이 빠져 State보다 짧다 → 교집합으로 정렬."""
    state = _state(5)
    pr = pd.DataFrame(np.zeros((3, 5)), index=state.index[:3], columns=ASSETS)
    model = _ConstLogitModel(np.zeros(5))

    nav_df = run_policy(pr, state, model, BacktestEngine(initial_nav=1.0))

    assert len(nav_df) == 3
    assert list(nav_df.index) == list(state.index[:3])


def test_column_order_mismatch_raises():
    state = _state(2)
    pr = pd.DataFrame(np.zeros((2, 5)), index=state.index, columns=["EWY", "SPY", "TLT", "GLD", "SHV"])
    with pytest.raises(ValueError, match="자산 순서와 다릅니다"):
        run_policy(pr, state, _ConstLogitModel(np.zeros(5)), BacktestEngine())
