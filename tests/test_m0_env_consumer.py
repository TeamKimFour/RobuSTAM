"""M0(K_asset=0, D=156) — env 소비자 관점 심층 검증 (PR #69 리뷰 후속).

민지 PR #69에서 `VARIANTS`에 `pytest.param(156, 0, 1, ...)`을 추가해 구조 계약
(observation_space·reset·단일 step·PPO 왕복)은 커버됐다. 다만 K_asset=0은 자산
지표 블록이 통째로 비는 유일한 콤보라, 그 특성 때문에 어긋날 수 있는 지점들이
파라메트라이즈 구조 검증만으로는 안 잡힌다.

이 파일은 형우 담당 영역(env가 M0을 정상적으로 소비하는가)에서 M0에서만 유효한
추가 계약을 검증한다:
  - schema 슬라이스가 K_asset=0에서 정확히 빈 슬라이스가 되는가
  - `market`이 `A·W` 위치에서 바로 시작하는가(자산 지표 블록 없이)
  - `feature_names`가 자산 지표 이름을 한 개도 만들지 않는가
  - `assemble_state_matrix`가 asset_feat 0컬럼 + market K_m컬럼을 소화해
    형우 PortfolioEnv가 요구하는 컬럼 순서로 정확히 출력하는가
  - **에피소드 끝까지** 다중 스텝을 진행해도 shape·prev_weight·룩어헤드 규약이 유지되는가
  - hold/완전 스왑에서 turnover·cost가 이론값과 일치하는가(비용이 M0에서만 어긋나는 사고 방지)
  - DiscretePortfolioEnv의 hold(=8) / +Δ / -Δ 이산 액션이 M0에서도 SHV 상대 계정
    계약을 정확히 지키는가

RunPod에서 도현이 M0으로 학습을 돌리게 되므로(PR #69 도현 코멘트), 이 층위의
계약이 어긋나면 학습이 조용히 잘못된 obs·비용을 학습하게 된다 — 그걸 방지한다.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src import config_loader as cl
from src.data import schema

pytest.importorskip("gymnasium")

from src.env.discrete_env import DiscretePortfolioEnv  # noqa: E402
from src.env.portfolio_env import PortfolioEnv  # noqa: E402

ASSETS = ["SPY", "EWY", "TLT", "GLD", "SHV"]
FWD_RET_COLS = [f"fwd_ret_{a}" for a in ASSETS]

# M0 = K_asset=0, K_market=1 (EBR만) → D = 5·30 + 0 + 1 + 5 = 156.
# state_spec.md v1.2 §2 표 · docs/feature_candidates.md M0 정의.
W_M0 = 30
K_ASSET_M0 = 0
K_MARKET_M0 = 1
D_M0 = 156
HOLD_ACTION = 8  # DiscretePortfolioEnv: 2·A_tr = 8 (유지)


def _cfg_m0(c: float = 0.001) -> dict:
    """M0(K_asset=0) config. 시장 지표 1종만 남아 자산 지표 블록이 완전히 빈다."""
    return {
        "assets": ASSETS,
        "window": W_M0,
        "transaction_cost": c,
        "features": {
            "asset": [],  # ← K_asset=0의 핵심
            "market": ["Equity_Bond_Ratio"],
        },
    }


def _state_df(n_rows: int = 10) -> pd.DataFrame:
    """M0 D=156 wide state_df. 각 (t, col)에 고유값을 심어 어느 행을 읽었는지 역추적 가능."""
    cfg = _cfg_m0()
    cols = schema.feature_names(
        W_M0,
        asset_features=cl.get_asset_features(cfg),
        market_features=cl.get_market_features(cfg),
    )
    dates = pd.date_range("2020-01-01", periods=n_rows, freq="B")
    data = np.arange(n_rows * len(cols), dtype=np.float32).reshape(n_rows, len(cols))
    return pd.DataFrame(data, index=dates, columns=cols)


def _zero_returns_df(state_df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        np.zeros((len(state_df), 5), dtype=np.float32),
        index=state_df.index,
        columns=FWD_RET_COLS,
    )


# ── 1. schema 슬라이스 계약 (K_asset=0 특수형) ──────────────────────

def test_asset_feature_slice_is_empty_at_k_asset_zero():
    """모든 자산 인덱스에서 asset_feature_slice가 빈 슬라이스(start==stop)."""
    for i in range(schema.N_ASSETS):
        sl = schema.asset_feature_slice(i, W_M0, n_asset_features=K_ASSET_M0)
        assert sl.start == sl.stop, f"asset {i}: {sl}"
        # 시작 위치는 수익률 블록 바로 뒤 A·W (자산별 오프셋이 0)
        assert sl.start == schema.N_ASSETS * W_M0


def test_market_slice_starts_at_returns_block_end():
    """K_asset=0이면 시장 블록이 A·W 바로 뒤에서 시작 (자산 블록 없음)."""
    sl = schema.market_feature_slice(
        W_M0, n_asset_features=K_ASSET_M0, n_market_features=K_MARKET_M0
    )
    assert sl == slice(schema.N_ASSETS * W_M0, schema.N_ASSETS * W_M0 + K_MARKET_M0)
    assert sl.stop - sl.start == K_MARKET_M0


def test_prev_weight_slice_is_last_five_at_m0():
    """prev_weight는 D의 마지막 5칸."""
    sl = schema.prev_weight_slice(
        W_M0, n_asset_features=K_ASSET_M0, n_market_features=K_MARKET_M0
    )
    assert sl == slice(D_M0 - schema.N_ASSETS, D_M0)


def test_index_map_features_blocks_are_all_empty_at_m0():
    """index_map의 features_XXX 5블록이 모두 빈 슬라이스이며 서로 같은 위치."""
    im = schema.index_map(W_M0, K_ASSET_M0, K_MARKET_M0)
    starts = []
    for asset in schema.ASSETS:
        sl = im[f"features_{asset}"]
        assert sl.start == sl.stop, f"{asset} 자산 지표 슬라이스가 비어야 함: {sl}"
        starts.append(sl.start)
    # 자산별 오프셋이 K_asset=0이라 모두 동일 위치
    assert len(set(starts)) == 1


def test_feature_names_has_no_asset_feature_columns_at_m0():
    """schema.feature_names가 feat_XXX_YYY 컬럼을 하나도 만들지 않는다."""
    cfg = _cfg_m0()
    names = schema.feature_names(
        W_M0,
        asset_features=cl.get_asset_features(cfg),
        market_features=cl.get_market_features(cfg),
    )
    assert len(names) == D_M0
    assert sum(1 for n in names if n.startswith("feat_")) == 0
    assert sum(1 for n in names if n.startswith("mkt_")) == K_MARKET_M0
    assert sum(1 for n in names if n.startswith("prevw_")) == schema.N_ASSETS


# ── 2. assemble → env 결합 (자산 지표 0컬럼 파이프라인) ────────────

def test_assemble_state_matrix_flows_through_env_at_m0():
    """assemble이 asset_feat 0컬럼·market 1컬럼을 소화해 env가 요구하는 컬럼 순서로 낸다.

    자산 지표 블록이 완전히 비는 유일한 콤보라, 여기서 조립이 어긋나면 env가
    바로 스키마 mismatch를 던진다.
    """
    from src.data.assemble import assemble_state_matrix

    cfg = _cfg_m0()
    n = W_M0 + 5
    dates = pd.date_range("2020-01-01", periods=n, freq="B")
    rng = np.random.default_rng(0)
    log_ret = pd.DataFrame(
        rng.normal(0.0, 0.01, size=(n, 5)).astype(np.float32),
        index=dates, columns=ASSETS,
    )
    asset_feat = pd.DataFrame(index=dates)  # 0컬럼 — M0의 특징
    market_feat = pd.DataFrame(
        {"mkt_Equity_Bond_Ratio": rng.normal(0.0, 0.01, size=n)},
        index=dates,
    )

    state = assemble_state_matrix(log_ret, asset_feat, market_feat, cfg)
    assert state.shape[1] == D_M0
    assert list(state.columns) == schema.feature_names(
        W_M0,
        asset_features=cl.get_asset_features(cfg),
        market_features=cl.get_market_features(cfg),
    )
    # env가 이 조립본을 받는지 최종 결합 검증
    fwd = pd.DataFrame(
        rng.normal(0.0, 0.01, size=(len(state), 5)).astype(np.float32),
        index=state.index, columns=FWD_RET_COLS,
    )
    env = PortfolioEnv(state, fwd, cfg=cfg)
    assert env.observation_space.shape == (D_M0,)


# ── 3. PortfolioEnv 다중 스텝 — 에피소드 끝까지 M0에서 안정 ─────────

def test_env_runs_full_episode_at_m0():
    """N-1 스텝을 돌아 terminated가 마지막 스텝에만 나와야 한다."""
    n = 6
    s = _state_df(n)
    r = _zero_returns_df(s)
    env = PortfolioEnv(s, r, cfg=_cfg_m0())
    env.reset()

    action = np.zeros(5, dtype=np.float64)
    for t in range(n - 2):
        obs, _, terminated, _, _ = env.step(action)
        assert obs.shape == (D_M0,)
        assert not terminated, f"t={t}: 중간에 종료됨"
    obs, _, terminated, _, _ = env.step(action)
    assert terminated
    assert obs.shape == (D_M0,)


def test_prev_weight_slice_position_holds_over_steps_at_m0():
    """다중 스텝을 진행해도 prev_weight가 항상 마지막 5칸에 갱신된다."""
    n = 5
    s = _state_df(n)
    r = _zero_returns_df(s)
    env = PortfolioEnv(s, r, cfg=_cfg_m0())
    obs, _ = env.reset()

    # reset 직후: SHV 100%
    np.testing.assert_array_equal(
        obs[D_M0 - schema.N_ASSETS:D_M0],
        np.array([0.0, 0.0, 0.0, 0.0, 1.0], dtype=np.float32),
    )

    # 균등 로짓 → 균등 비중 [0.2]*5
    for _ in range(n - 2):
        obs, _, terminated, _, info = env.step(np.array([1.0] * 5, dtype=np.float64))
        assert obs.shape == (D_M0,)
        np.testing.assert_allclose(
            obs[D_M0 - schema.N_ASSETS:D_M0],
            np.full(5, 0.2, dtype=np.float32),
            atol=1e-6,
        )
        np.testing.assert_allclose(info["weights"], np.full(5, 0.2), atol=1e-6)
        if terminated:
            break


# ── 4. 룩어헤드 방어 at M0 (여러 스텝 교차검증) ─────────────────────

def test_lookahead_defense_over_multiple_steps_at_m0():
    """M0에서도 각 스텝 log_return이 정확히 returns_df.iloc[t]와 일치.

    자산 지표 블록이 비면 상태 조회 오프셋이 잘못될 여지가 있어 별도 검증.
    """
    n = 6
    s = _state_df(n)
    r_arr = np.zeros((n, 5), dtype=np.float64)
    for t in range(n):
        r_arr[t] = (t + 1) * 0.001  # 각 t 고유값
    r = pd.DataFrame(r_arr, index=s.index, columns=FWD_RET_COLS)
    env = PortfolioEnv(s, r, cfg=_cfg_m0(c=0.0))
    env.reset()

    # 균등 비중 유지 → log_return = log(1 + expm1(r_log)) = r_log[t]
    for t in range(n - 1):
        _, _, terminated, _, info = env.step(np.array([1.0] * 5, dtype=np.float64))
        expected = (t + 1) * 0.001
        assert info["log_return"] == pytest.approx(expected, abs=1e-9), (
            f"M0 t={t}: log_return이 {expected}와 다름 → 룩어헤드 회귀 의심"
        )
        if terminated:
            break


def test_observation_carries_only_current_row_state_at_m0():
    """obs의 정적 부분(prev_weight 제외)이 state_df.iloc[t]와 일치.

    M0에서 자산 지표 블록이 사라진 뒤 market이 정확히 A·W 위치에 실려 있는지도
    함께 검증한다 — 인덱스 계산이 어긋나면 여기서 걸린다.
    """
    n = 5
    s = _state_df(n)
    r = _zero_returns_df(s)
    env = PortfolioEnv(s, r, cfg=_cfg_m0())
    prev_w_slice = slice(D_M0 - schema.N_ASSETS, D_M0)

    obs, _ = env.reset()
    row0 = s.iloc[0].to_numpy(dtype=np.float32)
    for i in range(D_M0):
        if prev_w_slice.start <= i < prev_w_slice.stop:
            continue
        assert obs[i] == pytest.approx(row0[i]), f"reset: obs[{i}] != state_df.iloc[0][{i}]"

    logits = np.array([1.0] * 5, dtype=np.float64)
    for t in range(1, n - 1):
        obs, _, terminated, _, _ = env.step(logits)
        expected = s.iloc[t].to_numpy(dtype=np.float32)
        for i in range(D_M0):
            if prev_w_slice.start <= i < prev_w_slice.stop:
                continue
            assert obs[i] == pytest.approx(expected[i]), (
                f"M0 t={t}: obs[{i}]={obs[i]} != state_df.iloc[{t}][{i}]={expected[i]}"
            )
        if terminated:
            break


# ── 5. 회전율·비용 정확도 at M0 (비용 계산이 D와 무관함을 명시) ────

def test_hold_action_zero_turnover_zero_cost_at_m0():
    """유지 시 turnover=0, cost=0, reward=포트폴리오 수익. M0에서도 성립."""
    s = _state_df()
    r = _zero_returns_df(s)
    env = PortfolioEnv(s, r, cfg=_cfg_m0(c=0.001))
    env.reset()
    # SHV 100% 유지 로짓
    logits = np.array([-1e6, -1e6, -1e6, -1e6, 1e6], dtype=np.float64)
    _, reward, _, _, info = env.step(logits)
    assert info["turnover"] == pytest.approx(0.0, abs=1e-6)
    assert info["cost"] == pytest.approx(0.0, abs=1e-9)
    assert reward == pytest.approx(0.0, abs=1e-9)


def test_full_swap_yields_cost_2c_at_m0():
    """SHV→SPY 완전 스왑 시 turnover=2, cost=2c."""
    c = 0.001
    s = _state_df()
    r = _zero_returns_df(s)
    env = PortfolioEnv(s, r, cfg=_cfg_m0(c=c))
    env.reset()
    logits = np.array([1e6, -1e6, -1e6, -1e6, -1e6], dtype=np.float64)
    _, reward, _, _, info = env.step(logits)
    assert info["turnover"] == pytest.approx(2.0, abs=1e-5)
    assert info["cost"] == pytest.approx(2 * c, abs=1e-8)
    assert reward == pytest.approx(-2 * c, abs=1e-8)


def test_cumulative_cost_matches_theory_over_episode_at_m0():
    """N번 완전 스왑 반복 → 누적 cost = N × 2c. M0에서도 이론값 유지."""
    n = 6
    c = 0.001
    s = _state_df(n)
    r = _zero_returns_df(s)
    env = PortfolioEnv(s, r, cfg=_cfg_m0(c=c))
    env.reset()

    total_cost = 0.0
    steps = 0
    for t in range(n - 1):
        if t % 2 == 0:
            logits = np.array([1e6, -1e6, -1e6, -1e6, -1e6], dtype=np.float64)
        else:
            logits = np.array([-1e6, -1e6, -1e6, -1e6, 1e6], dtype=np.float64)
        _, _, terminated, _, info = env.step(logits)
        total_cost += info["cost"]
        steps += 1
        if terminated:
            break
    assert total_cost == pytest.approx(steps * 2 * c, abs=1e-6)


# ── 6. DiscretePortfolioEnv 이산 액션 정확도 at M0 ──────────────────

def test_discrete_hold_semantic_at_m0():
    """이산 hold(=8) → turnover≈0, cost≈0. M0에서도 SHV 유지 계약 성립."""
    s = _state_df()
    r = _zero_returns_df(s)
    env = DiscretePortfolioEnv(PortfolioEnv(s, r, cfg=_cfg_m0(c=0.001)))
    env.reset()
    _, reward, _, _, info = env.step(HOLD_ACTION)
    assert info["turnover"] == pytest.approx(0.0, abs=1e-6)
    assert info["cost"] == pytest.approx(0.0, abs=1e-9)
    assert reward == pytest.approx(0.0, abs=1e-9)


def test_discrete_plus_delta_semantic_at_m0():
    """이산 +Δ SPY(=0) → SHV→SPY delta 이동 → turnover=2Δ, cost=2Δc."""
    delta = 0.1
    c = 0.001
    s = _state_df()
    r = _zero_returns_df(s)
    env = DiscretePortfolioEnv(
        PortfolioEnv(s, r, cfg=_cfg_m0(c=c)), delta=delta
    )
    env.reset()
    _, reward, _, _, info = env.step(0)  # +Δ to SPY
    assert info["turnover"] == pytest.approx(2 * delta, abs=1e-4)
    assert info["cost"] == pytest.approx(2 * delta * c, abs=1e-7)
    assert reward == pytest.approx(-2 * delta * c, abs=1e-7)


def test_discrete_lookahead_via_hold_at_m0():
    """이산 어댑터를 거쳐도 각 t가 returns_df.iloc[t]만 참조. M0에서 재검증."""
    n = 6
    s = _state_df(n)
    r_arr = np.zeros((n, 5), dtype=np.float64)
    for t in range(n):
        r_arr[t] = (t + 1) * 0.001
    r = pd.DataFrame(r_arr, index=s.index, columns=FWD_RET_COLS)
    env = DiscretePortfolioEnv(
        PortfolioEnv(s, r, cfg=_cfg_m0(c=0.0)), delta=0.1
    )
    env.reset()

    for t in range(n - 1):
        _, _, terminated, _, info = env.step(HOLD_ACTION)
        expected = (t + 1) * 0.001
        assert info["log_return"] == pytest.approx(expected, abs=1e-6), (
            f"discrete M0 t={t}: 룩어헤드 회귀 의심"
        )
        if terminated:
            break
