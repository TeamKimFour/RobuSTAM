"""schema.py 인덱스맵 회귀 테스트 — 서드파티 의존 없음(CI에서 항상 실행).

docs/state_spec.md가 명시적으로 경고한 실패모드를 전용 테스트로 고정한다:
  - 187 하드코딩 (W 변경 시 차원 자동 산출되어야 함)
  - 시장지표를 자산별로 복제 (1,355안의 핵심 오류)
  - 자산 순서 변경
"""

from src.data import schema as s


def test_state_dim_formula():
    """W에서 차원이 산출되어야 한다 (187 하드코딩 금지)."""
    assert s.state_dim(20) == 137
    assert s.state_dim(30) == 187
    assert s.state_dim(60) == 337


def test_asset_order_fixed():
    """자산 순서는 고정 — CLAUDE.md §2."""
    assert s.ASSETS == ("SPY", "EWY", "TLT", "GLD", "SHV")
    assert s.N_ASSETS == 5


def test_feature_counts():
    """자산별 지표 6개, 시장지표 2개."""
    assert s.N_ASSET_FEATURES == 6
    assert s.N_MARKET_FEATURES == 2
    assert len(s.ASSET_FEATURES) == 6
    assert len(s.MARKET_FEATURES) == 2


def test_index_map_covers_exactly_once():
    """모든 슬라이스가 [0, state_dim)을 빈틈·겹침 없이 정확히 한 번씩 덮어야 한다."""
    for W in (20, 30, 60):
        covered: list[int] = []
        for sl in s.index_map(W).values():
            covered.extend(range(sl.start, sl.stop))
        assert sorted(covered) == list(range(s.state_dim(W))), f"W={W} 피복 실패"


def test_market_slice_is_single_set_not_per_asset():
    """시장지표는 정확히 2칸 (자산별 5번 복제하면 10칸 → 1,355 오류)."""
    sl = s.market_feature_slice(30)
    assert sl.stop - sl.start == s.N_MARKET_FEATURES == 2


def test_spec_literal_slices_W30():
    """docs/state_spec.md §3의 구체적 인덱스값과 정확히 일치."""
    W = 30
    assert s.returns_slice(0, W) == slice(0, 30)      # SPY 수익률
    assert s.returns_slice(4, W) == slice(120, 150)   # SHV 수익률
    assert s.asset_feature_slice(0, W) == slice(150, 156)  # SPY 지표
    assert s.asset_feature_slice(4, W) == slice(174, 180)  # SHV 지표
    assert s.market_feature_slice(W) == slice(180, 182)    # 시장지표
    assert s.prev_weight_slice(W) == slice(182, 187)       # 직전비중


def test_feature_names_length_and_blocks():
    """컬럼명 개수 == state_dim, 블록별 접두사 확인."""
    W = 30
    names = s.feature_names(W)
    assert len(names) == s.state_dim(W) == 187
    assert names[0] == "ret_SPY_lag29"
    assert names[s.returns_slice(0, W).stop - 1] == "ret_SPY_lag0"
    assert names[150] == "feat_SPY_MA_Cross_5_20"
    assert names[180] == "mkt_Equity_Bond_Ratio"
    assert names[182] == "prevw_SPY"


def test_asset_idx_out_of_range():
    """잘못된 자산 인덱스는 거부."""
    import pytest

    with pytest.raises(IndexError):
        s.returns_slice(5, 30)
