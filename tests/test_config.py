"""config_loader 테스트.

차원 산출 등 순수 함수는 PyYAML 없이도 실행된다(dict를 직접 넘김).
실제 파일 로드는 PyYAML이 필요하므로 importorskip으로 가드한다.
"""

from pathlib import Path

import pytest

from src import config_loader as cl
from src.data import schema as s


def _fake_cfg(window=30):
    return {
        "assets": ["SPY", "EWY", "TLT", "GLD", "SHV"],
        "window": window,
        "transaction_cost": 0.001,
    }


def test_get_state_dim_matches_window():
    """get_state_dim이 W에 따라 변해야 한다 (187 하드코딩 금지)."""
    assert cl.get_state_dim(_fake_cfg(20)) == 137
    assert cl.get_state_dim(_fake_cfg(30)) == 187
    assert cl.get_state_dim(_fake_cfg(60)) == 337


def test_state_dim_agrees_with_schema():
    """config_loader와 schema가 동일 차원을 산출해야 한다 (교차검증)."""
    for w in (20, 30, 60):
        assert cl.get_state_dim(_fake_cfg(w)) == s.state_dim(w)
    assert cl.N_ASSET_FEATURES == s.N_ASSET_FEATURES
    assert cl.N_MARKET_FEATURES == s.N_MARKET_FEATURES


def test_basic_getters():
    cfg = _fake_cfg(30)
    assert cl.get_window(cfg) == 30
    assert cl.get_assets(cfg) == ["SPY", "EWY", "TLT", "GLD", "SHV"]
    assert cl.get_transaction_cost(cfg) == 0.001


def test_get_precompute_path_default_when_missing():
    """data 섹션이 없는 fake cfg에서도 기본 경로를 반환해야 한다."""
    assert cl.get_precompute_path(_fake_cfg()) == "data/precompute/latest.json"


def test_get_precompute_path_reads_data_section():
    cfg = _fake_cfg()
    cfg["data"] = {"precompute_path": "data/precompute/custom.json"}
    assert cl.get_precompute_path(cfg) == "data/precompute/custom.json"


def test_validate_rejects_wrong_asset_order():
    bad = _fake_cfg()
    bad["assets"] = ["EWY", "SPY", "TLT", "GLD", "SHV"]
    with pytest.raises(ValueError):
        cl._validate(bad)


def test_validate_rejects_nonpositive_window():
    bad = _fake_cfg(0)
    with pytest.raises(ValueError):
        cl._validate(bad)


def test_load_real_config_file():
    """실제 config/config.yaml 로드 — PyYAML 필요."""
    pytest.importorskip("yaml")
    cfg = cl.load_config("config/config.yaml")
    assert cl.get_state_dim(cfg) == 187
    assert cl.get_assets(cfg) == ["SPY", "EWY", "TLT", "GLD", "SHV"]
    assert cl.get_transaction_cost(cfg) == 0.001
    assert cl.get_precompute_path(cfg) == "data/precompute/latest.json"


def test_model_fold_id_default_is_one_indexed():
    """src/data/splits.py::make_folds가 fold_id를 1부터 매겨서(fold=0 없음),
    config.model.fold_id 기본값도 1이어야 한다 (회귀 테스트: 한때 0으로 잘못
    설정돼 RunPod 실행 중 FileNotFoundError가 났었음 — docs/model_training.md §4-5)."""
    pytest.importorskip("yaml")
    cfg = cl.load_config("config/config.yaml")
    assert cfg["model"]["fold_id"] == 1


# ── 피처 콤보(M0~M3) — config_loader.resolve_combo / 경로 헬퍼 / 검증 ────────


def _combo_cfg(active="full"):
    return {
        "assets": ["SPY", "EWY", "TLT", "GLD", "SHV"],
        "window": 30,
        "transaction_cost": 0.001,
        "feature_combos": {
            "full": {"asset": ["A1", "A2"], "market": ["MA_KNOWN"]},
            "M0": {"asset": [], "market": ["MA_KNOWN"]},
        },
        "active_combo": active,
    }


def test_resolve_combo_without_feature_combos_returns_cfg_unchanged():
    """feature_combos 섹션이 없는 구 계약 cfg는 resolve_combo가 그대로 돌려줘야 한다."""
    cfg = _fake_cfg()
    assert cl.resolve_combo(cfg) is cfg


def test_resolve_combo_uses_active_combo_by_default():
    cfg = _combo_cfg(active="M0")
    resolved = cl.resolve_combo(cfg)
    assert resolved["features"]["asset"] == []
    assert resolved["features"]["market"] == ["MA_KNOWN"]
    assert cfg.get("features") is None  # 원본 cfg는 불변


def test_resolve_combo_explicit_combo_overrides_active():
    cfg = _combo_cfg(active="full")
    resolved = cl.resolve_combo(cfg, combo="M0")
    assert resolved["features"]["asset"] == []


def test_resolve_combo_unknown_combo_raises():
    cfg = _combo_cfg()
    with pytest.raises(ValueError):
        cl.resolve_combo(cfg, combo="does-not-exist")


def test_get_feature_store_dir_full_keeps_legacy_path():
    cfg = {"data": {"feature_store_dir": "data/feature_store"}, "active_combo": "full"}
    assert cl.get_feature_store_dir(cfg) == "data/feature_store"
    assert cl.get_feature_store_dir(cfg, combo="full") == "data/feature_store"
    assert cl.get_feature_store_dir(cfg, combo=None) == "data/feature_store"


def test_get_feature_store_dir_other_combo_nests_under_base():
    cfg = {"data": {"feature_store_dir": "data/feature_store"}, "active_combo": "full"}
    assert cl.get_feature_store_dir(cfg, combo="M0") == str(Path("data/feature_store") / "M0")


def test_get_meta_db_full_keeps_legacy_path():
    cfg = {"data": {"meta_db": "data/feature_store/meta.sqlite"}, "active_combo": "full"}
    assert cl.get_meta_db(cfg) == "data/feature_store/meta.sqlite"


def test_get_meta_db_other_combo_nests_under_base():
    cfg = {"data": {"meta_db": "data/feature_store/meta.sqlite"}, "active_combo": "full"}
    assert cl.get_meta_db(cfg, combo="M1") == str(Path("data/feature_store/M1/meta.sqlite"))


def test_validate_combos_rejects_unknown_active_combo():
    cfg = _combo_cfg(active="does-not-exist")
    with pytest.raises(ValueError):
        cl._validate(cfg)


def test_validate_combos_rejects_unknown_feature_name():
    cfg = _combo_cfg()
    cfg["feature_combos"]["full"]["asset"] = ["Totally_Unknown_Feature"]
    with pytest.raises(ValueError):
        cl._validate(cfg)


def test_validate_combos_rejects_duplicate_feature_in_combo():
    cfg = _combo_cfg()
    cfg["feature_combos"]["full"]["asset"] = ["A1", "A1"]
    with pytest.raises(ValueError):
        cl._validate(cfg)


def test_validate_combos_rejects_missing_active_combo_key():
    cfg = _combo_cfg()
    del cfg["active_combo"]
    with pytest.raises(KeyError):
        cl._validate(cfg)


def test_real_config_defines_m0_through_m3():
    """실제 config.yaml의 feature_combos가 M0~M3 4개 차원(156/166/171/172)을 정확히 낸다."""
    pytest.importorskip("yaml")
    cfg = cl.load_config("config/config.yaml")
    assert cfg["active_combo"] == "full"
    assert cl.get_state_dim(cfg) == 187  # active_combo=full 기본값 하위호환

    expected_dims = {"M0": 156, "M1": 166, "M2": 171, "M3": 172}
    for combo, dim in expected_dims.items():
        resolved = cl.resolve_combo(cfg, combo)
        assert cl.get_state_dim(resolved) == dim, combo
