"""config_loader 테스트.

차원 산출 등 순수 함수는 PyYAML 없이도 실행된다(dict를 직접 넘김).
실제 파일 로드는 PyYAML이 필요하므로 importorskip으로 가드한다.
"""

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
