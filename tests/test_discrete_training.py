"""이산 행동(DQN) 학습·백테스트 배선 테스트 — 이슈 #34 개선안 B(행동 연속성).

`DiscretePortfolioEnv` 자체의 계약은 tests/test_discrete_env.py가 이미 박제한다. 여기서는
**세 경로(학습 env · 백테스트 · 추론)가 같은 Δ 이전 규칙을 쓰는지**를 검증한다 —
경로별로 규칙을 따로 구현하면 백테스트가 학습과 조용히 어긋나기 때문이다
(src/backtest/policy.py 모듈 독스트링 ③이 경고하는 사고 유형).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import yaml

from src.backtest.engine import BacktestEngine
from src.backtest.policy import run_policy
from src.config_loader import get_action_delta, load_config
from src.data import schema
from src.models.loader import get_algorithm, is_discrete, load_policy

pytest.importorskip("gymnasium")

from src.env.discrete_env import DiscretePortfolioEnv, discrete_action_to_logits  # noqa: E402
from src.env.portfolio_env import PortfolioEnv, _softmax  # noqa: E402

ASSETS = ["SPY", "EWY", "TLT", "GLD", "SHV"]
W = 30
SHV_IDX = 4
TRADABLE_IDX = [0, 1, 2, 3]
HOLD_ACTION = 8


def _dqn_config(tmp_path, delta: float = 0.1) -> str:
    """실제 config.yaml을 그대로 복사해 algorithm만 DQN으로 바꾼 임시 config를 만든다."""
    cfg = load_config()
    cfg["model"]["algorithm"] = "DQN"
    cfg["model"]["action_delta"] = delta
    path = tmp_path / "config_dqn.yaml"
    path.write_text(yaml.safe_dump(cfg, allow_unicode=True), encoding="utf-8")
    return str(path)


def _state(n_rows: int) -> pd.DataFrame:
    cols = schema.feature_names(W)
    idx = pd.date_range("2020-01-01", periods=n_rows, freq="B")
    return pd.DataFrame(np.zeros((n_rows, len(cols)), dtype=np.float32), index=idx, columns=cols)


class _ScriptedDiscreteModel:
    """정해둔 이산 action을 순서대로 내는 정책 — 학습 env와 백테스트에 같은 수열을 먹인다."""

    def __init__(self, actions):
        self.actions = list(actions)
        self.i = 0

    def predict(self, obs, deterministic: bool = True):
        a = self.actions[self.i % len(self.actions)]
        self.i += 1
        return np.array(a), None


# ── loader — 알고리즘 판정이 한 곳에서만 나오는지 ──

def test_get_algorithm_defaults_to_ppo_and_normalizes_case():
    assert get_algorithm({}) == "PPO"
    assert get_algorithm({"model": {"algorithm": "dqn"}}) == "DQN"


def test_is_discrete_only_for_dqn():
    assert is_discrete({"model": {"algorithm": "DQN"}}) is True
    assert is_discrete({"model": {"algorithm": "PPO"}}) is False
    assert is_discrete({}) is False


def test_load_policy_rejects_unsupported_algorithm():
    with pytest.raises(NotImplementedError, match="지원하지 않는 알고리즘"):
        load_policy("nonexistent.zip", "SAC")


# ── config — Δ 단일 출처 ──

def test_action_delta_default_and_validation():
    assert get_action_delta({}) == pytest.approx(0.1)
    assert get_action_delta({"model": {"action_delta": 0.25}}) == pytest.approx(0.25)
    for bad in (0.0, -0.1, 1.5):
        with pytest.raises(ValueError, match="action_delta"):
            get_action_delta({"model": {"action_delta": bad}})


# ── 학습 env 선택 — cfg만 보고 갈리는지 ──

def test_load_fold_env_wraps_discrete_for_dqn(monkeypatch, tmp_path):
    """algorithm=DQN이면 load_fold_env가 이산 어댑터를 씌워야 한다.

    evaluate()처럼 알고리즘을 모르는 호출부도 cfg만으로 학습과 같은 env를 받는 것이 핵심.
    """
    from src.models import train as train_mod

    n = 12
    state = _state(n)
    targets = pd.DataFrame(
        np.zeros((n, 5), dtype=np.float32),
        index=state.index,
        columns=[f"fwd_ret_{a}" for a in ASSETS],
    )
    monkeypatch.setattr(train_mod.fs, "load_features", lambda *a, **k: state)
    monkeypatch.setattr(train_mod.fs, "load_targets", lambda *a, **k: targets)

    base_cfg = {
        "assets": ASSETS,
        "window": W,
        "transaction_cost": 0.001,
        "data": {"feature_store_dir": str(tmp_path)},
    }
    dqn_env = train_mod.load_fold_env({**base_cfg, "model": {"algorithm": "DQN"}}, 1)
    ppo_env = train_mod.load_fold_env({**base_cfg, "model": {"algorithm": "PPO"}}, 1)

    assert isinstance(dqn_env, DiscretePortfolioEnv)
    assert dqn_env.action_space.n == 9
    assert not isinstance(ppo_env, DiscretePortfolioEnv)


# ── 핵심: 학습 env와 백테스트가 같은 비중 궤적을 내는지 ──

def test_backtest_reproduces_training_weight_path(tmp_path):
    """같은 action 수열이면 학습 env의 비중과 백테스트의 비중이 일치해야 한다."""
    actions = [0, 0, 1, HOLD_ACTION, 4, 2, 3, HOLD_ACTION, 5, 1]
    n = len(actions) + 1  # env는 마지막 행에서 종료되므로 여유를 한 행 둔다
    state = _state(n)
    targets = pd.DataFrame(
        np.zeros((n, 5), dtype=np.float32),
        index=state.index,
        columns=[f"fwd_ret_{a}" for a in ASSETS],
    )

    # ① 학습 경로 — DiscretePortfolioEnv를 직접 스텝
    env = DiscretePortfolioEnv(
        PortfolioEnv(state, targets, cfg={"assets": ASSETS, "window": W, "transaction_cost": 0.001}),
        delta=0.1,
    )
    env.reset()
    train_weights = []
    for a in actions:
        _, _, terminated, truncated, _ = env.step(a)
        train_weights.append(env.unwrapped._current_weight.copy())
        if terminated or truncated:
            break

    # ② 백테스트 경로 — 같은 수열을 내는 정책으로 run_policy
    price_returns = pd.DataFrame(np.zeros((n, 5)), index=state.index, columns=ASSETS)
    nav_df = run_policy(
        price_returns,
        state,
        _ScriptedDiscreteModel(actions),
        BacktestEngine(initial_nav=1.0),
        config_path=_dqn_config(tmp_path),
    )

    k = len(train_weights)
    assert k >= len(actions) - 1  # 비교 구간이 사라지지 않았는지
    np.testing.assert_allclose(
        nav_df[ASSETS].to_numpy()[:k], np.asarray(train_weights), atol=1e-6
    )


def test_backtest_turnover_is_structurally_capped_by_delta(tmp_path):
    """개선안 B의 근거 — 하루 이전폭이 Δ면 turnover는 2Δ를 넘을 수 없다(콜드스타트 제외).

    첫 스텝만 SHV 100% → 첫 배분이라 예외다. 이 상한이 회전율 0.86(이슈 #34 §1)에
    직접 대응하는 구조적 제약이므로 숫자로 박제한다.

    허용오차 1e-6은 softmax(log(w)) 왕복의 float32 근사오차다(docs/env_spec.md §4-3).
    """
    delta = 0.1
    actions = [0, 1, 2, 3, 4, 5, 6, 7, HOLD_ACTION, 0, 1, 2]
    n = len(actions)
    state = _state(n)
    price_returns = pd.DataFrame(np.zeros((n, 5)), index=state.index, columns=ASSETS)

    nav_df = run_policy(
        price_returns,
        state,
        _ScriptedDiscreteModel(actions),
        BacktestEngine(initial_nav=1.0),
        config_path=_dqn_config(tmp_path, delta=delta),
    )

    assert (nav_df["turnover"].to_numpy() <= 2 * delta + 1e-6).all()


def test_pure_function_matches_wrapper_action(tmp_path):
    """어댑터가 순수 함수를 그대로 쓰는지 — 두 구현이 갈라지면 여기서 잡힌다."""
    env = DiscretePortfolioEnv(
        PortfolioEnv(
            _state(5),
            pd.DataFrame(
                np.zeros((5, 5), dtype=np.float32),
                index=_state(5).index,
                columns=[f"fwd_ret_{a}" for a in ASSETS],
            ),
            cfg={"assets": ASSETS, "window": W, "transaction_cost": 0.001},
        ),
        delta=0.1,
    )
    env.reset()
    w_prev = env.unwrapped._current_weight.copy()

    for idx in range(9):
        np.testing.assert_allclose(
            env.action(idx),
            discrete_action_to_logits(
                w_prev, idx, shv_idx=SHV_IDX, tradable_idx=TRADABLE_IDX, delta=0.1
            ),
            atol=0,
        )


def test_discrete_action_out_of_range_raises():
    w = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
    for bad in (-1, 9):
        with pytest.raises(ValueError, match="이산 action"):
            discrete_action_to_logits(
                w, bad, shv_idx=SHV_IDX, tradable_idx=TRADABLE_IDX, delta=0.1
            )


def test_logits_roundtrip_to_weights():
    """log(clip(w)) → softmax가 원래 비중을 복원하는지(어댑터의 softmax 계약)."""
    w = np.array([0.3, 0.1, 0.0, 0.2, 0.4])
    logits = discrete_action_to_logits(
        w, HOLD_ACTION, shv_idx=SHV_IDX, tradable_idx=TRADABLE_IDX, delta=0.1
    )
    np.testing.assert_allclose(_softmax(np.asarray(logits, dtype=np.float64)), w, atol=1e-6)
