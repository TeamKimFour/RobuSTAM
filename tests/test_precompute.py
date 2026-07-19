"""precompute 테스트 — build_today_obs(민지 글루)의 차원·정규화 재현·prev_weight,
그리고 generate_latest(목 모델)의 latest.json 계약.

(pandas·pandas_ta·pyarrow 필요) 실데이터·네트워크 없이 tmp에 합성 가격으로 build를 돌려
Feature Store를 채운 뒤 precompute를 검증한다.
"""

import numpy as np
import pytest

pd = pytest.importorskip("pandas")
pytest.importorskip("pandas_ta")
pytest.importorskip("pyarrow")

from src.data import feature_store as fs
from src.data import schema
from src.data.build import build
from src.inference.precompute import build_today_obs, generate_latest

ASSETS = list(schema.ASSETS)
W = 30


def _setup(tmp_path):
    """합성 원시가격 + config를 만들고 build를 돌려 (cfg, run_id)를 반환한다.

    test_block 끝을 데이터 이후로 둬, state_all의 마지막 행이 test split 마지막 행과
    같아지게 한다(정규화 재현 대조용).
    """
    import yaml

    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    fsdir = tmp_path / "fs"

    idx = pd.bdate_range("2015-01-01", periods=650)
    rng = np.random.default_rng(0)
    prices = pd.DataFrame(
        100 * np.exp(np.cumsum(rng.normal(0.0002, 0.01, (650, 5)), axis=0)),
        index=idx, columns=ASSETS,
    )
    prices.index.name = "date"
    prices.to_parquet(raw_dir / "prices_raw.parquet")

    cfg = {
        "assets": ASSETS, "window": W, "transaction_cost": 0.001,
        "data": {
            "raw_dir": str(raw_dir), "feature_store_dir": str(fsdir),
            "meta_db": str(fsdir / "meta.sqlite"),
            "precompute_path": str(tmp_path / "precompute" / "latest.json"),
        },
        "features": {
            "asset": list(schema.ASSET_FEATURES), "market": list(schema.MARKET_FEATURES),
            "params": {
                "sma_fast": 5, "sma_slow": 20, "rsi_length": 14, "macd": [12, 26, 9],
                "vol_window": 20, "bbands": [20, 2], "roc_length": 10,
                "annualize_vol": False, "ebr_ma_window": 20, "gvr_long_window": 60,
            },
        },
        "split": {
            "mode": "expanding", "anchor_start": "2015-01-01",
            "test_blocks": [["2017-01-01", "2018-06-01"]],  # 끝을 데이터 이후로
            "valid_days": 60, "embargo_days": 34,
        },
        "normalize": {
            "method": "zscore", "returns_scope": "per_asset",
            "feature_scope": "per_column", "eps": 1e-8,
        },
    }
    cfg_file = tmp_path / "config.yaml"
    cfg_file.write_text(yaml.dump(cfg), encoding="utf-8")
    run_id = build(str(cfg_file), verbose=False)
    cfg["inference"] = {
        "model_path": "", "scaler_run_id": run_id, "scaler_fold_id": 1,
        "model_version": "test-v1",
    }
    return cfg, run_id


def test_obs_shape_and_no_nan(tmp_path):
    cfg, _ = _setup(tmp_path)
    obs, date = build_today_obs(cfg, prev_weights=None)
    assert obs.shape == (schema.state_dim(W),) == (187,)
    assert obs.dtype == np.float32
    assert not np.isnan(obs).any()
    assert isinstance(date, str) and len(date) == 10  # YYYY-MM-DD


def test_normalization_reproduction(tmp_path):
    """핵심: precompute의 정규화 State가 build가 적재한 값과 정확히 일치해야 한다."""
    cfg, _ = _setup(tmp_path)
    test_df = fs.load_features(cfg["data"]["feature_store_dir"], 1, "test")
    last_row = test_df.iloc[-1].to_numpy()
    last_date = str(test_df.index[-1].date())

    obs, date = build_today_obs(cfg, prev_weights=None)

    assert date == last_date
    # 수익률+지표+시장 [0:182]은 build 적재값과 동일(같은 scaler_stats 재현)
    np.testing.assert_allclose(obs[:182], last_row[:182], rtol=1e-5, atol=1e-6)


def test_prev_weight_cold_start(tmp_path):
    cfg, _ = _setup(tmp_path)
    obs, _ = build_today_obs(cfg, prev_weights=None)
    # 콜드스타트 = SHV(마지막 자산) 100%
    np.testing.assert_array_equal(obs[schema.prev_weight_slice(W)], [0, 0, 0, 0, 1])


def test_prev_weight_injection(tmp_path):
    cfg, _ = _setup(tmp_path)
    prev = np.array([0.1, 0.2, 0.3, 0.15, 0.25])
    obs, _ = build_today_obs(cfg, prev_weights=prev)
    np.testing.assert_allclose(obs[schema.prev_weight_slice(W)], prev, rtol=0, atol=1e-6)


def test_empty_scaler_run_id_auto_resolves(tmp_path):
    """scaler_run_id를 비우면 같은 config_hash의 최신 build run을 자동 선택한다(이슈 #33).

    build run_id는 타임스탬프를 포함해 머신 간 재현이 불가능하지만, 같은 config로 재빌드하면
    통계가 사실상 동일하다. 덕분에 CI·서버가 각자 build해서 쓰면 되고 `meta.sqlite`를 옮길
    필요가 없다. 명시했을 때와 결과가 같아야 한다.
    """
    cfg, run_id = _setup(tmp_path)
    obs_pinned, _ = build_today_obs(cfg, prev_weights=None)

    cfg["inference"]["scaler_run_id"] = ""  # 비우면 자동 해석
    obs_auto, _ = build_today_obs(cfg, prev_weights=None)

    np.testing.assert_array_equal(obs_auto, obs_pinned)


def test_auto_resolve_ignores_other_config_hash(tmp_path):
    """자동 선택은 config_hash가 다른 run을 집으면 안 된다.

    그냥 "최신 run"을 쓰면 window·자산구성이 다른 빌드의 통계를 조용히 집어 차원·의미가
    어긋난다. 더 최신이지만 해시가 다른 미끼 run을 심어 무시하는지 확인한다.
    """
    cfg, run_id = _setup(tmp_path)
    db = cfg["data"]["meta_db"]
    # 더 나중에 만들어졌지만 config_hash가 다른 run(예: W=20 빌드)을 심는다.
    fs.write_run(db, "decoy-9999", "2099-01-01T00:00:00", "deadbeefcafe", 20, 137, ASSETS)

    cfg["inference"]["scaler_run_id"] = ""
    resolved = fs.latest_run_id_for_config(db, fs.config_hash(cfg))

    assert resolved == run_id != "decoy-9999"
    # 실제 경로도 정상 동작해야 한다(미끼 통계가 없으므로 잘못 집으면 여기서 깨진다).
    obs, _ = build_today_obs(cfg, prev_weights=None)
    assert obs.shape == (187,)


def test_no_matching_build_raises(tmp_path):
    """해당 config_hash의 build가 아예 없으면 명확한 에러를 낸다."""
    cfg, _ = _setup(tmp_path)
    cfg["inference"]["scaler_run_id"] = ""
    cfg["window"] = 20  # config_hash가 달라져 매칭되는 run이 없음
    with pytest.raises(ValueError, match="build run이 없습니다"):
        build_today_obs(cfg, prev_weights=None)


def test_generate_latest_with_mock_model(tmp_path):
    """generate_latest(목 policy) → latest.json 계약(합=1·자산 5개·순서)."""
    pytest.importorskip("gymnasium")  # portfolio_env._softmax 재사용에 필요
    cfg, _ = _setup(tmp_path)

    class _MockModel:
        def predict(self, obs, deterministic=True):
            # SPY에 강한 선호(로짓), softmax 후 최대 비중이 SPY여야 함
            return np.array([5.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32), None

    payload = generate_latest(cfg, model=_MockModel(), now="2020-01-01T00:00:00+00:00")

    assert set(payload["weights"]) == set(ASSETS)
    assert abs(sum(payload["weights"].values()) - 1.0) < 1e-6
    assert payload["model_version"] == "test-v1"
    assert max(payload["weights"], key=payload["weights"].get) == "SPY"

    # 파일이 원자적으로 기록됐는지
    from pathlib import Path
    assert Path(cfg["data"]["precompute_path"]).is_file()
