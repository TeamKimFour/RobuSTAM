"""splits.py 테스트 — Expanding walk-forward 경계·embargo·비겹침 (pandas 필요)."""

import pytest

pd = pytest.importorskip("pandas")

from src.data.splits import make_folds


def _cfg():
    return {
        "split": {
            "mode": "expanding",
            "anchor_start": "2010-01-01",
            "test_blocks": [
                ["2020-01-01", "2021-12-31"],
                ["2022-01-01", "2023-12-31"],
                ["2024-01-01", "2025-12-31"],
            ],
            "valid_days": 252,
            "embargo_days": 34,
        }
    }


def _index():
    # 2009-12 ~ 2025-12 영업일 (지표 warm-up 후 가용일 근사)
    return pd.bdate_range("2009-12-28", "2025-12-31")


def test_three_folds():
    folds = make_folds(_index(), _cfg())
    assert len(folds) == 3
    assert [f.fold_id for f in folds] == [1, 2, 3]


def test_embargo_gap():
    """train_end와 test_start 사이 거래일 수 >= embargo_days."""
    idx = _index()
    for f in make_folds(idx, _cfg()):
        gap = idx.get_loc(f.test_start) - idx.get_loc(f.train_end)
        assert gap >= f.embargo_days + 1


def test_no_overlap_and_valid_tail():
    idx = _index()
    for f in make_folds(idx, _cfg()):
        assert f.train_end < f.valid_start
        assert f.valid_end < f.test_start
        # valid는 train_all의 꼬리 252 거래일
        valid_len = idx.get_loc(f.valid_end) - idx.get_loc(f.valid_start) + 1
        assert valid_len == 252


def test_expanding_anchor_fixed_and_growing():
    folds = make_folds(_index(), _cfg())
    # anchor 고정: 모든 fold train_start 동일
    assert folds[0].train_start == folds[1].train_start == folds[2].train_start
    # train_end 단조 증가
    assert folds[0].train_end < folds[1].train_end < folds[2].train_end


def test_raises_when_embargo_too_large():
    cfg = _cfg()
    cfg["split"]["test_blocks"] = [["2010-02-01", "2010-03-01"]]  # anchor 직후 → train 부족
    with pytest.raises(ValueError):
        make_folds(_index(), cfg)
