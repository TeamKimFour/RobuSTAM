"""가변 State 차원 호환성 테스트 — D∈{166, 171, 172} 등 config 변형에서 환경이 흔들리지 않음을 보장.

민지 Feature Store가 지표 개수를 바꿔 D=187 이외의 값(예: 166/171/172)으로 배포해도
PortfolioEnv·DiscretePortfolioEnv·observation space·prev_weight 위치·모델 왕복이 config만
바꿔 그대로 소비되는지 검증한다. 187 하드코딩 회귀도 함께 방지한다.

각 케이스의 (K_asset, K_market) 조합은 D=166/171/172를 재현하기 위한 것으로,
실제 민지 스키마와 다를 수 있다. 계약 검증에는 정확한 개수 조합보다 "config대로 소비되는가"가
핵심이라 여기서는 D를 재현할 수 있는 임의 조합을 쓴다.
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

# (D, K_asset, K_market) — 모두 W=30 기준.
# 5·30 + 5·K_asset + K_market + 5 = D 를 만족한다.
VARIANTS = [
    pytest.param(166, 2, 1, id="D=166 (K_asset=2, K_market=1)"),
    pytest.param(171, 3, 1, id="D=171 (K_asset=3, K_market=1)"),
    pytest.param(172, 3, 2, id="D=172 (K_asset=3, K_market=2)"),
]


def _make_cfg(k_asset: int, k_market: int, W: int = 30, c: float = 0.001) -> dict:
    asset_feats = [f"asset_feat_{i}" for i in range(k_asset)]
    market_feats = [f"mkt_feat_{i}" for i in range(k_market)]
    return {
        "assets": ASSETS,
        "window": W,
        "transaction_cost": c,
        "features": {"asset": asset_feats, "market": market_feats},
    }


def _make_state_df(cfg: dict, n_rows: int = 6) -> pd.DataFrame:
    W = cl.get_window(cfg)
    cols = schema.feature_names(
        W,
        asset_features=cl.get_asset_features(cfg),
        market_features=cl.get_market_features(cfg),
    )
    dates = pd.date_range("2020-01-01", periods=n_rows, freq="B")
    data = np.arange(n_rows * len(cols), dtype=np.float32).reshape(n_rows, len(cols))
    return pd.DataFrame(data, index=dates, columns=cols)


def _make_returns_df(state_df: pd.DataFrame, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    data = rng.normal(0.0, 0.01, size=(len(state_df), 5)).astype(np.float32)
    return pd.DataFrame(data, index=state_df.index, columns=FWD_RET_COLS)


# ── config → state_dim 산출 (SSOT: config_loader) ────────────────────

@pytest.mark.parametrize("D, k_a, k_m", VARIANTS)
def test_config_loader_computes_variant_state_dim(D, k_a, k_m):
    """config.features 길이에서 D를 정확히 뽑는다 (187 하드코딩 부재 검증)."""
    cfg = _make_cfg(k_a, k_m)
    assert cl.get_state_dim(cfg) == D
    assert cl.get_n_asset_features(cfg) == k_a
    assert cl.get_n_market_features(cfg) == k_m


@pytest.mark.parametrize("D, k_a, k_m", VARIANTS)
def test_schema_state_dim_matches_config_loader(D, k_a, k_m):
    """schema.state_dim(W, K_a, K_m) 과 config_loader.get_state_dim(cfg) 결과가 일치."""
    cfg = _make_cfg(k_a, k_m)
    assert schema.state_dim(30, n_asset_features=k_a, n_market_features=k_m) == D
    assert cl.get_state_dim(cfg) == schema.state_dim(
        30, n_asset_features=k_a, n_market_features=k_m
    )


@pytest.mark.parametrize("D, k_a, k_m", VARIANTS)
def test_feature_names_length_matches_state_dim(D, k_a, k_m):
    """schema.feature_names 길이가 정확히 D."""
    cfg = _make_cfg(k_a, k_m)
    names = schema.feature_names(
        cl.get_window(cfg),
        asset_features=cl.get_asset_features(cfg),
        market_features=cl.get_market_features(cfg),
    )
    assert len(names) == D


@pytest.mark.parametrize("D, k_a, k_m", VARIANTS)
def test_prev_weight_slice_is_last_A_slots(D, k_a, k_m):
    """직전 비중은 항상 마지막 A칸 [D-A : D]."""
    sl = schema.prev_weight_slice(30, n_asset_features=k_a, n_market_features=k_m)
    assert sl == slice(D - schema.N_ASSETS, D)


@pytest.mark.parametrize("D, k_a, k_m", VARIANTS)
def test_index_map_covers_state_exactly_once(D, k_a, k_m):
    """모든 블록 슬라이스가 [0, D)를 빈틈·겹침 없이 덮는다."""
    covered: list[int] = []
    for sl in schema.index_map(30, k_a, k_m).values():
        covered.extend(range(sl.start, sl.stop))
    assert sorted(covered) == list(range(D))


# ── PortfolioEnv 계약 (observation_space·reset·step) ────────────────

@pytest.mark.parametrize("D, k_a, k_m", VARIANTS)
def test_portfolio_env_observation_space_shape(D, k_a, k_m):
    cfg = _make_cfg(k_a, k_m)
    s = _make_state_df(cfg)
    r = _make_returns_df(s)
    env = PortfolioEnv(s, r, cfg=cfg)
    assert env.observation_space.shape == (D,)
    assert env.state_dim == D


@pytest.mark.parametrize("D, k_a, k_m", VARIANTS)
def test_portfolio_env_reset_returns_D_shaped_obs(D, k_a, k_m):
    cfg = _make_cfg(k_a, k_m)
    s = _make_state_df(cfg)
    r = _make_returns_df(s)
    env = PortfolioEnv(s, r, cfg=cfg)
    obs, info = env.reset()
    assert obs.shape == (D,)
    assert obs.dtype == np.float32
    # 초기 비중: SHV 100% (팀 회의 확정)
    prev_w = obs[env._prev_weight_slice]
    np.testing.assert_array_equal(
        prev_w, np.array([0.0, 0.0, 0.0, 0.0, 1.0], dtype=np.float32)
    )
    assert info == {}


@pytest.mark.parametrize("D, k_a, k_m", VARIANTS)
def test_portfolio_env_step_preserves_D_and_updates_prev_weight(D, k_a, k_m):
    cfg = _make_cfg(k_a, k_m)
    s = _make_state_df(cfg)
    r = _make_returns_df(s)
    env = PortfolioEnv(s, r, cfg=cfg)
    env.reset()
    logits = np.array([1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float64)
    obs, _, _, _, info = env.step(logits)
    assert obs.shape == (D,)
    # 균등 로짓 → 균등 비중
    np.testing.assert_allclose(
        obs[env._prev_weight_slice],
        np.full(5, 0.2, dtype=np.float32),
        atol=1e-6,
    )
    np.testing.assert_allclose(info["weights"], np.full(5, 0.2), atol=1e-6)


@pytest.mark.parametrize("D, k_a, k_m", VARIANTS)
def test_portfolio_env_rejects_state_columns_from_wrong_config(D, k_a, k_m):
    """A config로 만든 state_df를 B config env에 넣으면 스키마 검증이 실패해야 한다."""
    cfg_a = _make_cfg(k_a, k_m)
    # 다른 K_market으로 다른 D를 갖는 config
    cfg_b = _make_cfg(k_a, k_m + 5)
    s_a = _make_state_df(cfg_a)
    r_a = _make_returns_df(s_a)
    with pytest.raises(ValueError, match="schema.feature_names"):
        PortfolioEnv(s_a, r_a, cfg=cfg_b)


# ── DiscretePortfolioEnv (관측은 원본 통과) ─────────────────────────

@pytest.mark.parametrize("D, k_a, k_m", VARIANTS)
def test_discrete_env_observation_shape_passes_through(D, k_a, k_m):
    cfg = _make_cfg(k_a, k_m)
    s = _make_state_df(cfg)
    r = _make_returns_df(s)
    env = DiscretePortfolioEnv(PortfolioEnv(s, r, cfg=cfg))
    # 이산 어댑터는 관측을 그대로 통과 — obs는 항상 (D,)
    obs, _ = env.reset()
    assert obs.shape == (D,)
    obs2, _, _, _, _ = env.step(env.action_space.sample())
    assert obs2.shape == (D,)
    # 행동공간은 D와 무관: 2·A_tr + 1 = 9 유지
    assert env.action_space.n == 9


# ── 카논(187) 후방호환 회귀 방지 ──────────────────────────────────

def test_canonical_187_still_works_without_features_section():
    """features 섹션이 없는 config(=구 계약)에서도 D=187이 그대로 나온다."""
    cfg = {
        "assets": ASSETS,
        "window": 30,
        "transaction_cost": 0.001,
    }
    assert cl.get_state_dim(cfg) == 187
    assert cl.get_n_asset_features(cfg) == 6
    assert cl.get_n_market_features(cfg) == 2


# ── SB3 PPO 저장·로딩 왕복 (D별 policy.zip 호환성) ─────────────────
# stable_baselines3가 없으면 스킵 (RunPod GPU/CI 전용). 지역 개발환경에서 sb3 미설치가 기본.

@pytest.mark.parametrize("D, k_a, k_m", VARIANTS)
def test_ppo_save_load_roundtrip_preserves_variant_dim(D, k_a, k_m, tmp_path):
    """PPO(MlpPolicy)를 D별 env에 붙여 save→load 왕복 후 predict가 유효한지 검증.

    SB3의 MlpPolicy는 obs_dim에 맞춰 첫 은닉층을 자동 결정하므로 D가 바뀌어도
    구조 코드 변경 없이 동작해야 한다. 이 테스트는 그 계약이 실제로 성립하는지,
    저장된 .zip이 같은 D의 env로 다시 붙는지 확인한다(형우 검증 항목: 모델 저장·로딩).
    """
    sb3 = pytest.importorskip("stable_baselines3")

    import gymnasium as gym
    from gymnasium import spaces

    cfg = _make_cfg(k_a, k_m)
    # SB3는 유한 행동 bound를 요구 — train._BoundedActionWrapper와 동일 계약을 인라인으로 재현
    class _Bounded(gym.ActionWrapper):
        def __init__(self, env: gym.Env, bound: float = 10.0):
            super().__init__(env)
            self.action_space = spaces.Box(
                low=-bound, high=bound, shape=env.action_space.shape, dtype=np.float32
            )

        def action(self, a):
            return a

    def _mk_env() -> gym.Env:
        s = _make_state_df(cfg, n_rows=12)
        r = _make_returns_df(s)
        return _Bounded(PortfolioEnv(s, r, cfg=cfg))

    env = _mk_env()
    model = sb3.PPO("MlpPolicy", env, n_steps=8, batch_size=8, n_epochs=1, verbose=0, seed=0)

    save_path = tmp_path / f"ppo_D{D}.zip"
    model.save(str(save_path))
    assert save_path.is_file()

    # 재로드 후 새 env에 붙여 예측 실행 — obs shape가 D이고 action shape가 (5,)인지 확인
    fresh_env = _mk_env()
    loaded = sb3.PPO.load(str(save_path), env=fresh_env)
    obs, _ = fresh_env.reset()
    assert obs.shape == (D,), f"reload된 env obs D 불일치: {obs.shape}"

    action, _ = loaded.predict(obs, deterministic=True)
    assert action.shape == (5,), f"reload된 policy action shape 불일치: {action.shape}"
    assert np.isfinite(action).all()


@pytest.mark.parametrize("D_train, ka_t, km_t", VARIANTS)
@pytest.mark.parametrize("D_load, ka_l, km_l", VARIANTS)
def test_ppo_cross_dim_load_is_rejected(D_train, ka_t, km_t, D_load, ka_l, km_l, tmp_path):
    """다른 D로 저장된 policy를 다른 D env에 붙이면 실패해야 한다 (shape mismatch 감지).

    민지 스키마가 바뀌었는데 예전 policy를 그대로 쓰면 조용히 잘못된 예측이 나가는 게
    가장 위험하다. SB3는 obs_space가 다르면 로드 시 명시 에러를 내야 한다.
    """
    if D_train == D_load:
        pytest.skip("동일 D는 정상 왕복 케이스 — 다른 테스트가 검증")
    sb3 = pytest.importorskip("stable_baselines3")

    import gymnasium as gym
    from gymnasium import spaces

    class _Bounded(gym.ActionWrapper):
        def __init__(self, env: gym.Env, bound: float = 10.0):
            super().__init__(env)
            self.action_space = spaces.Box(
                low=-bound, high=bound, shape=env.action_space.shape, dtype=np.float32
            )

        def action(self, a):
            return a

    def _mk_env(k_a, k_m):
        cfg = _make_cfg(k_a, k_m)
        s = _make_state_df(cfg, n_rows=12)
        r = _make_returns_df(s)
        return _Bounded(PortfolioEnv(s, r, cfg=cfg))

    env_train = _mk_env(ka_t, km_t)
    model = sb3.PPO("MlpPolicy", env_train, n_steps=8, batch_size=8, n_epochs=1, verbose=0, seed=0)
    save_path = tmp_path / f"ppo_D{D_train}.zip"
    model.save(str(save_path))

    env_load = _mk_env(ka_l, km_l)
    with pytest.raises(Exception):  # SB3가 obs_space mismatch 검출
        sb3.PPO.load(str(save_path), env=env_load)
