"""build.py 통합 테스트 — 합성 데이터로 전체 파이프라인 적재 검증.

(pandas·pandas_ta·pyarrow 필요) 실데이터·네트워크 없이 tmp에 합성 가격을 만들어 build를 돌린다.
"""

from pathlib import Path

import numpy as np
import pytest

pd = pytest.importorskip("pandas")
pytest.importorskip("pandas_ta")
pytest.importorskip("pyarrow")

from src.data import feature_store as fs
from src.data import schema
from src.data.build import build

ASSETS = list(schema.ASSETS)


def _setup(tmp_path):
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
        "assets": ASSETS, "window": 30, "transaction_cost": 0.001,
        "data": {
            "raw_dir": str(raw_dir), "feature_store_dir": str(fsdir),
            "meta_db": str(fsdir / "meta.sqlite"),
        },
        "feature_combos": {
            "full": {"asset": list(schema.ASSET_FEATURES), "market": list(schema.MARKET_FEATURES)},
            "M0": {"asset": [], "market": ["Equity_Bond_Ratio"]},
            "M1": {"asset": ["MACD_Hist", "Rolling_Vol_20"], "market": ["Equity_Bond_Ratio"]},
            "M2": {
                "asset": ["MACD_Hist", "Rolling_Vol_20", "RSI_28"],
                "market": ["Equity_Bond_Ratio"],
            },
            "M3": {
                "asset": ["MACD_Hist", "Rolling_Vol_20", "RSI_28"],
                "market": ["Equity_Bond_Ratio", "Drawdown"],
            },
        },
        "active_combo": "full",
        "features": {
            "asset": list(schema.ASSET_FEATURES), "market": list(schema.MARKET_FEATURES),
            "params": {
                "sma_fast": 5, "sma_slow": 20, "rsi_length": 14, "rsi_28_length": 28,
                "macd": [12, 26, 9], "vol_window": 20, "bbands": [20, 2], "roc_length": 10,
                "annualize_vol": False, "ebr_ma_window": 20, "gvr_long_window": 60,
                "drawdown_lookback": 60,
            },
        },
        "split": {
            "mode": "expanding", "anchor_start": "2015-01-01",
            "test_blocks": [["2017-01-01", "2017-03-31"]],
            "valid_days": 60, "embargo_days": 34,
        },
        "normalize": {
            "method": "zscore", "returns_scope": "per_asset",
            "feature_scope": "per_column", "eps": 1e-8,
        },
    }
    cfg_file = tmp_path / "config.yaml"
    cfg_file.write_text(yaml.dump(cfg), encoding="utf-8")
    return str(cfg_file), str(fsdir), cfg


def test_build_end_to_end(tmp_path):
    cfg_path, fsdir, cfg = _setup(tmp_path)
    run_id = build(cfg_path, verbose=False)
    db = cfg["data"]["meta_db"]

    # 파티션 로드 + 187 차원
    tr = fs.load_features(fsdir, 1, "train")
    assert tr.shape[1] == schema.state_dim(30) == 187
    assert list(tr.columns) == schema.feature_names(30)

    # train 정규화 확인 (지표 컬럼 평균≈0)
    assert abs(tr["feat_SPY_RSI_14"].mean()) < 1e-6

    # targets
    tg = fs.load_targets(fsdir, 1, "train")
    assert list(tg.columns) == [f"fwd_ret_{a}" for a in ASSETS]

    # 메타 채워짐
    assert len(fs.read_scaler_stats(db, run_id, 1)) == 182  # 150 + 30 + 2 (prev_weight 제외)
    assert len(fs.read_folds(db, run_id)) == 1
    assert len(fs.read_feature_columns(db, run_id)) == 187


def test_build_default_combo_uses_legacy_flat_path(tmp_path):
    """combo 인자 없이 build()를 부르면 active_combo(=full)를 기존 플랫 경로에 적재해야 한다.

    daily.yml·train.py가 기대하는 data/feature_store/fold=*/... 레이아웃이 콤보 도입으로
    바뀌지 않는다는 하위호환 회귀 가드.
    """
    cfg_path, fsdir, cfg = _setup(tmp_path)
    build(cfg_path, verbose=False)
    tr = fs.load_features(fsdir, 1, "train")  # combo 하위 디렉토리 없이 바로 fsdir 밑
    assert tr.shape[1] == 187


@pytest.mark.parametrize(
    "combo,expected_dim",
    [("M0", 156), ("M1", 166), ("M2", 171), ("M3", 172)],
)
def test_build_per_combo_dimensions(tmp_path, combo, expected_dim):
    """build(combo=...)가 각 콤보의 정확한 차원으로 Feature Store를 만들어야 한다."""
    cfg_path, fsdir, cfg = _setup(tmp_path)
    build(cfg_path, combo=combo, verbose=False)

    from src import config_loader as cl

    out_dir = cl.get_feature_store_dir(cfg, combo)
    tr = fs.load_features(out_dir, 1, "train")
    assert tr.shape[1] == expected_dim
    assert out_dir == str(Path(fsdir) / combo)  # full이 아니므로 하위 디렉토리로 분리됨


def test_build_combos_do_not_collide(tmp_path):
    """서로 다른 콤보를 같은 raw 데이터로 빌드해도 산출물이 별도 디렉토리에 공존해야 한다."""
    cfg_path, fsdir, cfg = _setup(tmp_path)
    build(cfg_path, combo="M0", verbose=False)
    build(cfg_path, combo="M1", verbose=False)

    from src import config_loader as cl

    tr_m0 = fs.load_features(cl.get_feature_store_dir(cfg, "M0"), 1, "train")
    tr_m1 = fs.load_features(cl.get_feature_store_dir(cfg, "M1"), 1, "train")
    assert tr_m0.shape[1] == 156
    assert tr_m1.shape[1] == 166
    # M0 빌드가 M1 산출물을 덮어쓰지 않았는지(별도 meta.sqlite도 분리됨)
    db_m0 = cl.get_meta_db(cfg, "M0")
    db_m1 = cl.get_meta_db(cfg, "M1")
    assert db_m0 != db_m1
    assert Path(db_m0).is_file() and Path(db_m1).is_file()
