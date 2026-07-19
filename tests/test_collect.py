"""collect.py 정렬 로직 테스트 — 네트워크 없이 합성 데이터로 검증.

실제 yfinance 다운로드(fetch_prices)는 네트워크 의존이라 여기서 테스트하지 않는다.
순수 정렬 로직 _align만 검증한다(pandas 필요).
"""

import pytest

pd = pytest.importorskip("pandas")
np = pytest.importorskip("numpy")

from src.data import collect as c


def _close_with_gap():
    """B자산에 하루 결측이 있는 종가 표."""
    idx = pd.date_range("2020-01-01", periods=4, freq="D")
    return pd.DataFrame(
        {
            "SPY": [100, 101, 102, 103],
            "EWY": [50, 51, 52, 53],
            "TLT": [90, 91, 92, 93],
            "GLD": [180, 181, 182, 183],
            "SHV": [110, np.nan, 110, 110],  # 둘째 날 결측
        },
        index=idx,
    )


ASSETS = ["SPY", "EWY", "TLT", "GLD", "SHV"]


def test_align_drops_incomplete_days_no_ffill():
    """결측일은 drop되고, ffill로 채워지지 않아야 한다."""
    aligned = c._align(_close_with_gap(), ASSETS)
    # 4일 중 결측 있던 1일이 빠져 3일
    assert len(aligned) == 3
    # 결측치 없음
    assert not aligned.isna().any().any()
    # ffill 안 됨: 둘째 날이 통째로 사라졌고, SHV가 가짜로 채워지지 않음
    assert pd.Timestamp("2020-01-02") not in aligned.index


def test_align_preserves_asset_order():
    """컬럼이 자산 고정 순서로 정렬되어야 한다."""
    shuffled = _close_with_gap()[["GLD", "SHV", "SPY", "TLT", "EWY"]]
    aligned = c._align(shuffled, ASSETS)
    assert list(aligned.columns) == ASSETS


def test_align_raises_on_missing_asset():
    incomplete = _close_with_gap().drop(columns=["TLT"])
    with pytest.raises(ValueError):
        c._align(incomplete, ASSETS)


# ── 수집 신뢰성: 재시도 · 품질 게이트 · 캐시 보호 ──
# 배경: CI에서 Yahoo 레이트리밋으로 0행이 와도 "성공"으로 끝나 빈 캐시를 남기고,
# 한참 뒤 지표 계산에서 TypeError로 터졌다(daily.yml 2026-07-17·18 장애).

def _good_prices(n=1200):
    """검증을 통과하는 합성 가격(완만한 상승, 결측·극단이동 없음)."""
    idx = pd.bdate_range("2015-01-01", periods=n)
    base = np.linspace(100, 200, n)
    return pd.DataFrame({a: base + i for i, a in enumerate(ASSETS)}, index=idx)


def _no_sleep(_seconds):
    """재시도 대기를 건너뛴다(테스트 속도)."""


def test_download_once_handles_empty_response(monkeypatch):
    """레이트리밋 빈 응답에서 컬럼 접근으로 죽지 않고 빈 DataFrame을 돌려준다."""
    import sys
    import types

    fake = types.ModuleType("yfinance")
    fake.download = lambda **kwargs: pd.DataFrame()
    monkeypatch.setitem(sys.modules, "yfinance", fake)

    out = c._download_once(ASSETS, "2020-01-01", "2020-02-01")
    assert out.empty


def test_fetch_retries_then_succeeds(monkeypatch, tmp_path):
    """첫 시도가 빈 응답이어도 재시도로 회복되면 성공한다."""
    calls = {"n": 0}

    def flaky(assets, start, end):
        calls["n"] += 1
        return pd.DataFrame(columns=assets) if calls["n"] == 1 else _good_prices()

    monkeypatch.setattr(c, "_download_once", flaky)
    prices = c.fetch_prices(
        ASSETS, "2015-01-01", "2020-01-01", raw_dir=str(tmp_path),
        min_rows=1000, sleep=_no_sleep,
    )
    assert calls["n"] == 2
    assert len(prices) == 1200
    assert (tmp_path / c.RAW_FILENAME).is_file()  # 성공 시에만 캐시 기록


def test_fetch_raises_when_all_attempts_empty(monkeypatch, tmp_path):
    """재시도 모두 빈 응답이면 조용히 넘어가지 않고 ValueError로 실패한다."""
    monkeypatch.setattr(c, "_download_once", lambda a, s, e: pd.DataFrame(columns=a))
    with pytest.raises(ValueError, match="수집 실패"):
        c.fetch_prices(ASSETS, "2015-01-01", "2020-01-01", raw_dir=str(tmp_path),
                       retries=2, sleep=_no_sleep)
    assert not (tmp_path / c.RAW_FILENAME).exists()  # 캐시 미생성


def test_fetch_raises_on_insufficient_rows(monkeypatch, tmp_path):
    """행수가 기준 미달이면 품질 게이트가 막는다."""
    monkeypatch.setattr(c, "_download_once", lambda a, s, e: _good_prices(n=50))
    with pytest.raises(ValueError, match="행 수 부족"):
        c.fetch_prices(ASSETS, "2015-01-01", "2020-01-01", raw_dir=str(tmp_path),
                       min_rows=1000, sleep=_no_sleep)
    assert not (tmp_path / c.RAW_FILENAME).exists()


def test_failure_does_not_overwrite_good_cache(monkeypatch, tmp_path):
    """★ 핵심: 나쁜 수집이 기존 좋은 캐시를 덮어쓰지 못한다(S3 오염 방지)."""
    cache = tmp_path / c.RAW_FILENAME
    _good_prices().to_parquet(cache)
    before = cache.read_bytes()

    monkeypatch.setattr(c, "_download_once", lambda a, s, e: pd.DataFrame(columns=a))
    with pytest.raises(ValueError):
        c.fetch_prices(ASSETS, "2015-01-01", "2020-01-01", raw_dir=str(tmp_path),
                       retries=2, sleep=_no_sleep)

    assert cache.read_bytes() == before  # 원본 그대로
