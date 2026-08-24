"""src.backtest.export 테스트 — FE JSON 계약과 fold 연속화 로직을 검증한다.

실제 모델·Feature Store 없이 `run_fold_fn`을 가짜로 주입해, 익스포터가
NAV를 이어붙이고 지표를 재계산해서 계약대로 JSON을 조립하는지만 확인한다.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from src.backtest import export as ex


def _fake_nav(dates: pd.DatetimeIndex, initial_nav: float, growth: float) -> pd.DataFrame:
    """(nav, turnover, cost) 컬럼을 가진 fake NAV DataFrame. NAV는 매일 growth 배로 증가."""
    factors = np.array([(1.0 + growth) ** i for i in range(len(dates))])
    nav = initial_nav * factors
    return pd.DataFrame({"nav": nav, "turnover": 0.02, "cost": 100.0}, index=dates)


def _fake_run_fold_factory(initial_nav: float):
    """fold별 fake 결과를 반환하는 run_fold_fn 대체품을 만든다.

    combo 키워드를 받아 콤보별로 다른 값을 낸다 — combo 전달 누락(PR #72 리뷰에서 잡힌
    export.py 버그)이 회귀되면 콤보별 policy 결과가 전부 동일해져 다른 테스트가 이를
    감지할 수 있다.
    """
    periods = {
        1: pd.bdate_range("2020-01-02", periods=5),
        2: pd.bdate_range("2022-01-03", periods=5),
        3: pd.bdate_range("2024-01-02", periods=5),
    }
    # RL policy만 다른 fold별 성장률로, 나머지는 동일.
    growth_map = {"RL policy": (0.01, 0.005, 0.002), "1/N": (0.003,) * 3,
                  "60:40": (0.004,) * 3, "B&H": (0.006,) * 3}
    # 콤보별 성장률 배수 — combo가 실제로 전달됐다면 RL policy NAV·metrics가 콤보마다 달라진다.
    combo_scale = {None: 1.0, "full": 1.0, "M0": 1.5, "M1": 1.2, "M2": 0.9, "M3": 0.7}

    def fake(fold_id, model, config_path, split, *, initial_nav=initial_nav, combo=None):
        idx = periods[fold_id]
        scale = combo_scale.get(combo, 1.0)
        nav_by_strategy = {}
        for name in ex.STRATEGY_ORDER:
            # RL policy만 콤보에 따라 성장률이 스케일된다(다른 콤보 정책 = 다른 결과).
            g = growth_map[name][fold_id - 1] * (scale if name == "RL policy" else 1.0)
            nav_by_strategy[name] = _fake_nav(idx, initial_nav, g)
        strategies = {
            name: {
                "total_return": 0.1, "sharpe": 1.0, "mdd": -0.05,
                "avg_turnover": 0.02 * (scale if name == "RL policy" else 1.0),
                "total_cost": 500.0 * (scale if name == "RL policy" else 1.0),
            }
            for name in ex.STRATEGY_ORDER
        }
        return {
            "fold_id": fold_id,
            "period": (str(idx[0].date()), str(idx[-1].date())),
            "strategies": strategies,
            "nav_by_strategy": nav_by_strategy,
            "s3_prefixes": {name: None for name in ex.STRATEGY_ORDER},
            "comparison": {name: {"beats_target": False, "sharpe_improvement_pct": 0.0,
                                  "mdd_defense_pct": 0.0} for name in ex.STRATEGY_ORDER
                           if name != "RL policy"},
        }

    return fake


def _spy_run_fold_factory(initial_nav: float):
    """호출 시 (fold_id, combo) 튜플을 기록하는 스파이 — combo가 실제로 전달됐는지 회귀 검증용."""
    fake = _fake_run_fold_factory(initial_nav)
    calls: list[tuple[int, str | None]] = []

    def spy(fold_id, model, config_path, split, *, initial_nav=initial_nav, combo=None):
        calls.append((fold_id, combo))
        return fake(fold_id, model, config_path, split, initial_nav=initial_nav, combo=combo)

    return spy, calls


@pytest.fixture()
def cfg_path(tmp_path):
    cfg = {
        "assets": ["SPY", "EWY", "TLT", "GLD", "SHV"],
        "window": 30,
        "transaction_cost": 0.001,
        "split": {"test_blocks": [
            ["2020-01-01", "2021-12-31"],
            ["2022-01-01", "2023-12-31"],
            ["2024-01-01", "2025-12-31"],
        ]},
    }
    p = tmp_path / "config.yaml"
    p.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    return str(p)


# ── ① 순수 함수: NAV 이어붙이기 ──
def test_concat_folds_multiplies_by_previous_end_value():
    """fold2의 첫 값 1.0은 fold1의 마지막 값(=1.2)만큼 스케일링돼 1.2가 돼야 한다."""
    fold1 = [{"date": "2020-01-01", "value": 1.0}, {"date": "2020-01-02", "value": 1.2}]
    fold2 = [{"date": "2022-01-01", "value": 1.0}, {"date": "2022-01-02", "value": 1.1}]
    out = ex._concat_folds([fold1, fold2])
    assert [p["value"] for p in out] == pytest.approx([1.0, 1.2, 1.2, 1.32])


def test_concat_folds_skips_empty_folds():
    out = ex._concat_folds([[], [{"date": "2020-01-01", "value": 1.0}]])
    assert len(out) == 1


# ── ② 순수 함수: 지표 재계산 ──
def test_metrics_from_points_recovers_analytic_values():
    """단조 증가 곡선의 total_return·MDD·CAGR을 이론값과 대조한다.

    Sharpe는 부동소수 잡음(1.01·1.01·... 누적)으로 σ가 0이 아니게 나오므로 값을 못 박지 않고
    다른 지표만 확인한다. std=0 경계는 아래 별도 테스트에서 다룬다.
    """
    pts = [{"date": f"2020-01-{i+1:02d}", "value": 1.01 ** i} for i in range(20)]
    m = ex._metrics_from_points(pts)
    assert m["total_return"] == pytest.approx(1.01 ** 19 - 1.0)
    assert m["mdd"] == pytest.approx(0.0, abs=1e-12)  # 단조 증가라 낙폭 0
    # CAGR: 19영업일 상승, 연 252영업일 기준
    expected_cagr = (1.01 ** 19) ** (252 / 19) - 1.0
    assert m["cagr"] == pytest.approx(expected_cagr, rel=1e-6)


def test_metrics_from_points_returns_zero_on_flat_curve():
    """모든 값이 동일하면 std=0 경계 → sharpe·vol 모두 0."""
    pts = [{"date": f"2020-01-{i+1:02d}", "value": 1.0} for i in range(10)]
    m = ex._metrics_from_points(pts)
    assert m["vol"] == 0.0
    assert m["sharpe"] == 0.0
    assert m["mdd"] == 0.0
    assert m["total_return"] == 0.0


def test_metrics_from_points_handles_short_series():
    assert ex._metrics_from_points([{"date": "2020-01-01", "value": 1.0}]) == {
        "cagr": 0.0, "sharpe": 0.0, "mdd": 0.0, "vol": 0.0, "total_return": 0.0,
    }


# ── ③ _summarize_verdict: 벤치마크별 fold 통과 집계 (합격/불합격 이분법 없음) ──
def test_summarize_verdict_counts_passed_folds():
    fold_results = [
        {"comparison": {"1/N": {"beats_target": True}, "60:40": {"beats_target": False},
                        "B&H": {"beats_target": True}}},
        {"comparison": {"1/N": {"beats_target": True}, "60:40": {"beats_target": False},
                        "B&H": {"beats_target": False}}},
        {"comparison": {"1/N": {"beats_target": False}, "60:40": {"beats_target": False},
                        "B&H": {"beats_target": True}}},
    ]
    out = ex._summarize_verdict(fold_results)

    assert out["1/N"] == {"folds_passed": 2, "folds_total": 3, "pass_rate": pytest.approx(2 / 3)}
    assert out["60:40"] == {"folds_passed": 0, "folds_total": 3, "pass_rate": 0.0}
    assert out["B&H"] == {"folds_passed": 2, "folds_total": 3, "pass_rate": pytest.approx(2 / 3)}
    # 이분법 합격/불합격 필드는 넣지 않는다(팀 §1 기준 미확정 — 이슈 #46).
    for v in out.values():
        assert "overall_pass" not in v


def test_summarize_verdict_handles_empty_fold_results():
    """fold가 하나도 없으면 0으로 나누기 없이 folds_total=0·pass_rate=0.0이어야 한다."""
    out = ex._summarize_verdict([])
    for name in ("1/N", "60:40", "B&H"):
        assert out[name] == {"folds_passed": 0, "folds_total": 0, "pass_rate": 0.0}


# ── ④ build_export: 계약 준수 ──
def test_build_export_produces_valid_contract(cfg_path):
    bundle = ex.build_export(
        model=None, config_path=cfg_path, initial_nav=1_000_000,
        run_fold_fn=_fake_run_fold_factory(1_000_000),
    )

    # 최상위 키
    for key in (
        "generated_at", "initial_nav", "periods", "strategies",
        "fold_table", "comparison", "verdict_summary",
    ):
        assert key in bundle, f"missing key: {key}"

    # _fake_run_fold_factory는 3 fold 전부 beats_target=False로 고정 — 통과 0건이어야 한다.
    for name in ("1/N", "60:40", "B&H"):
        assert bundle["verdict_summary"][name] == {
            "folds_passed": 0, "folds_total": 3, "pass_rate": 0.0,
        }

    # 3 fold × 4 전략
    assert len(bundle["periods"]) == 3
    assert [p["fold_id"] for p in bundle["periods"]] == [1, 2, 3]
    assert len(bundle["strategies"]) == 4
    assert [s["name"] for s in bundle["strategies"]] == list(ex.STRATEGY_ORDER)

    # 각 전략은 nav·metrics 필드
    for s in bundle["strategies"]:
        assert s["color"].startswith("#")
        assert len(s["nav"]) == 15  # 5일 × 3 fold, 이어붙임
        assert s["nav"][0]["value"] == pytest.approx(1.0)  # 초기 정규화
        for k in ("cagr", "sharpe", "mdd", "vol", "total_return", "avg_turnover", "total_cost"):
            assert k in s["metrics"]

    # fold_table은 RL policy 지표
    assert len(bundle["fold_table"]) == 3
    for row in bundle["fold_table"]:
        assert set(row.keys()) == {"fold_id", "period", "sharpe", "cagr", "mdd"}


def test_build_export_nav_is_monotonic_when_all_folds_growing(cfg_path):
    """모든 fold가 상승 곡선이면 이어붙인 NAV도 단조 증가해야 한다."""
    bundle = ex.build_export(
        model=None, config_path=cfg_path, initial_nav=1_000_000,
        run_fold_fn=_fake_run_fold_factory(1_000_000),
    )
    for s in bundle["strategies"]:
        values = [p["value"] for p in s["nav"]]
        assert all(values[i + 1] >= values[i] for i in range(len(values) - 1)), s["name"]


# ── ⑤ write_export: 원자적 쓰기 ──
def test_write_export_creates_file_with_valid_json(tmp_path, cfg_path):
    bundle = ex.build_export(
        model=None, config_path=cfg_path, initial_nav=1_000_000,
        run_fold_fn=_fake_run_fold_factory(1_000_000),
    )
    out = tmp_path / "sub" / "backtest.json"
    ex.write_export(bundle, str(out))
    assert out.is_file()
    parsed = json.loads(out.read_text(encoding="utf-8"))
    assert parsed["initial_nav"] == 1_000_000
    assert not (tmp_path / "sub" / "backtest.json.tmp").exists()  # 임시파일 잔여 없음


# ── ⑥ upload_to_s3: S3 키 계약·skip 동작 ──
def test_s3_key_matches_frontend_contract():
    """FE prebuild(scripts/fetch-backtest.mjs)와 반드시 같은 키 규칙을 써야 한다."""
    assert ex._s3_key("robustam") == "robustam/frontend/backtest.json"
    assert ex._s3_key("") == "frontend/backtest.json"
    assert ex._s3_key("/robustam/") == "robustam/frontend/backtest.json"  # 슬래시 trim


@pytest.fixture()
def _cfg_with_bucket(tmp_path):
    cfg = {
        "assets": ["SPY", "EWY", "TLT", "GLD", "SHV"],
        "window": 30,
        "transaction_cost": 0.001,
        "split": {"test_blocks": [["2020-01-01", "2021-12-31"]]},
        "data": {"s3_bucket": "test-bucket", "s3_prefix": "robustam"},
    }
    p = tmp_path / "cfg.yaml"
    p.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    return str(p)


def test_upload_skips_when_bucket_missing(tmp_path, monkeypatch, capsys):
    """버킷이 config·환경변수 어디에도 없으면 조용히 skip한다."""
    monkeypatch.delenv("S3_BUCKET", raising=False)
    cfg = {"data": {}, "assets": ["SPY", "EWY", "TLT", "GLD", "SHV"],
           "window": 30, "transaction_cost": 0.001}
    p = tmp_path / "cfg.yaml"
    p.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    payload = tmp_path / "backtest.json"
    payload.write_text('{"generated_at": "x"}', encoding="utf-8")

    url = ex.upload_to_s3(str(payload), str(p))
    assert url is None
    assert "skip" in capsys.readouterr().out.lower()


def test_upload_puts_object_at_frontend_key(tmp_path, monkeypatch, _cfg_with_bucket):
    """계약 키(`{prefix}/frontend/backtest.json`)에 정확히 올라간다 (moto)."""
    moto = pytest.importorskip("moto")
    boto3 = pytest.importorskip("boto3")
    from moto import mock_aws

    monkeypatch.delenv("S3_BUCKET", raising=False)
    monkeypatch.delenv("S3_PREFIX", raising=False)
    payload = tmp_path / "backtest.json"
    payload.write_bytes(b'{"strategies": []}')

    with mock_aws():
        s3 = boto3.client("s3", region_name="us-east-1")
        s3.create_bucket(Bucket="test-bucket")
        url = ex.upload_to_s3(str(payload), _cfg_with_bucket, client=s3)

        assert url is not None
        assert url.endswith("robustam/frontend/backtest.json")
        obj = s3.get_object(Bucket="test-bucket", Key="robustam/frontend/backtest.json")
        assert obj["Body"].read() == b'{"strategies": []}'
        assert obj["ContentType"] == "application/json"


# ── ⑦ combos 필드: 콤보별 policy 병행 익스포트 (도현·찬휘 3순위) ──
def test_build_export_omits_combos_when_no_models_by_combo(cfg_path):
    """models_by_combo가 없으면 combos 필드가 아예 없어야 한다 (하위호환)."""
    bundle = ex.build_export(
        model=None, config_path=cfg_path, initial_nav=1_000_000,
        run_fold_fn=_fake_run_fold_factory(1_000_000),
    )
    assert "combos" not in bundle


def test_build_export_populates_combos_when_models_provided(cfg_path):
    """models_by_combo가 있으면 각 콤보의 nav·metrics·fold_table을 조립한다."""
    fake = _fake_run_fold_factory(1_000_000)
    # 실제 모델 객체는 필요 없다 — fake run_fold는 model 인자를 무시한다.
    models = {"full": object(), "M0": object(), "M1": object()}
    bundle = ex.build_export(
        model=None, config_path=cfg_path, initial_nav=1_000_000,
        run_fold_fn=fake, models_by_combo=models,
    )
    assert "combos" in bundle
    # COMBO_ORDER 순서 유지 (full → M0 → M1 → M2 → M3)
    assert [c["combo"] for c in bundle["combos"]] == ["full", "M0", "M1"]
    for c in bundle["combos"]:
        assert c["display_name"].startswith(c["combo"].capitalize()[0].upper()) or c["display_name"].startswith(c["combo"][0])
        assert c["color"].startswith("#")
        assert c["nav"], "nav 시계열이 비어 있음"
        assert c["nav"][0]["value"] == pytest.approx(1.0)  # 정규화 시작값
        # 전 구간 지표
        for k in ("cagr", "sharpe", "mdd", "vol", "total_return", "avg_turnover", "total_cost"):
            assert k in c["metrics"], f"combo {c['combo']}에 metrics.{k} 누락"
        # fold별 상세
        assert len(c["fold_table"]) == 3
        for row in c["fold_table"]:
            assert set(row.keys()) == {
                "fold_id", "period", "sharpe", "cagr", "mdd", "avg_turnover", "total_cost",
            }


def test_build_export_ignores_unknown_combos_in_models_dict(cfg_path):
    """models_by_combo에 알 수 없는 콤보명이 섞여 있으면 COMBO_ORDER의 것만 뽑아낸다."""
    fake = _fake_run_fold_factory(1_000_000)
    models = {"full": object(), "GARBAGE": object()}
    bundle = ex.build_export(
        model=None, config_path=cfg_path, initial_nav=1_000_000,
        run_fold_fn=fake, models_by_combo=models,
    )
    assert [c["combo"] for c in bundle["combos"]] == ["full"]


# ── 회귀 테스트: combo가 run_fold_fn까지 실제로 전달되는지 (PR #72 리뷰 대응) ──
def test_build_export_passes_combo_kwarg_to_run_fold(cfg_path):
    """`_build_combo_snapshot`이 각 콤보 이름을 그대로 `run_fold_fn(combo=...)`에 넘겨야 한다.

    이 인자가 빠지면 runner.run_fold가 config.active_combo(기본 'full')로 fallback해
    콤보별로 다른 Feature Store를 읽어야 할 상황이 전부 'full'로 뭉개진다.
    상위 `strategies`(=default combo=None) 호출과, 콤보별 호출을 모두 기록해서
    콤보 이름이 정확히 매칭되는지 검증한다.
    """
    spy, calls = _spy_run_fold_factory(1_000_000)
    models = {"full": object(), "M0": object(), "M2": object()}
    ex.build_export(
        model=None, config_path=cfg_path, initial_nav=1_000_000,
        run_fold_fn=spy, models_by_combo=models,
    )

    # 상위 strategies 호출은 combo 인자 없음(=None) — 3 fold
    top_level_calls = [c for c in calls if c[1] is None]
    assert sorted(top_level_calls) == [(1, None), (2, None), (3, None)]

    # 콤보별 호출은 combo 이름이 정확히 함께 전달되어야 한다 — 3 콤보 × 3 fold
    for combo in ("full", "M0", "M2"):
        combo_calls = sorted(c for c in calls if c[1] == combo)
        assert combo_calls == [(1, combo), (2, combo), (3, combo)], (
            f"combo={combo!r}가 run_fold_fn에 제대로 전달되지 않음: {combo_calls}"
        )


def test_build_export_produces_distinct_metrics_per_combo(cfg_path):
    """콤보별로 다른 policy가 학습됐다면 combos 항목의 지표가 콤보마다 달라야 한다.

    _fake_run_fold_factory는 combo에 따라 RL policy 성장률을 스케일한다 —
    combo 전달 누락 회귀 시 모든 콤보가 같은(=None fallback) 결과를 내 이 테스트가 실패.
    """
    fake = _fake_run_fold_factory(1_000_000)
    models = {"M0": object(), "M2": object()}
    bundle = ex.build_export(
        model=None, config_path=cfg_path, initial_nav=1_000_000,
        run_fold_fn=fake, models_by_combo=models,
    )
    combos_out = {c["combo"]: c for c in bundle["combos"]}
    assert set(combos_out) == {"M0", "M2"}

    # M0 스케일 1.5x, M2 스케일 0.9x — 총 수익률이 달라야 한다.
    m0_total = combos_out["M0"]["metrics"]["total_return"]
    m2_total = combos_out["M2"]["metrics"]["total_return"]
    assert m0_total != pytest.approx(m2_total), (
        f"콤보별 total_return이 동일 — combo 인자 전달 누락 회귀 의심 "
        f"(M0={m0_total}, M2={m2_total})"
    )
    # 부가 지표(avg_turnover)도 콤보에 따라 스케일된다 — 정합 확인
    assert combos_out["M0"]["metrics"]["avg_turnover"] != pytest.approx(
        combos_out["M2"]["metrics"]["avg_turnover"]
    )


def test_parse_combo_arg_rejects_bad_format():
    """`name=path` 형식이 아니면 SystemExit."""
    with pytest.raises(SystemExit, match="name=path"):
        ex._parse_combo_arg(["fullruns/full.zip"])


def test_parse_combo_arg_rejects_unknown_combo():
    """오타로 조용히 스킵되면 산출물이 어긋나므로 명시적으로 거부한다."""
    with pytest.raises(SystemExit, match="알 수 없는 콤보"):
        ex._parse_combo_arg(["M4=runs/m4.zip"])


def test_parse_combo_arg_returns_empty_for_none():
    assert ex._parse_combo_arg(None) == {}
    assert ex._parse_combo_arg([]) == {}


def test_parse_combo_arg_handles_multiple_pairs():
    out = ex._parse_combo_arg(["full=a.zip", "M0=b.zip", " M3 = c.zip "])
    assert out == {"full": "a.zip", "M0": "b.zip", "M3": "c.zip"}


def test_upload_env_bucket_overrides_config(tmp_path, monkeypatch, _cfg_with_bucket):
    """S3_BUCKET 환경변수가 config의 버킷을 override한다(s3_sync와 같은 관례)."""
    moto = pytest.importorskip("moto")
    boto3 = pytest.importorskip("boto3")
    from moto import mock_aws

    monkeypatch.setenv("S3_BUCKET", "override-bucket")
    monkeypatch.delenv("S3_PREFIX", raising=False)
    payload = tmp_path / "backtest.json"
    payload.write_bytes(b'{}')

    with mock_aws():
        s3 = boto3.client("s3", region_name="us-east-1")
        s3.create_bucket(Bucket="override-bucket")
        url = ex.upload_to_s3(str(payload), _cfg_with_bucket, client=s3)

    assert url is not None
    assert "override-bucket" in url
