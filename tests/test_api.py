"""FastAPI 추론 서버 테스트.

httpx2는 requirements-dev에만 있으므로 importorskip으로 가드(미설치 CI에선 skip) —
test_s3_sync의 moto 가드와 동일 패턴.
"""

import json

import pytest

pytest.importorskip("httpx2")

from fastapi.testclient import TestClient

from src.api.main import app, get_config

ASSETS = ["SPY", "EWY", "TLT", "GLD", "SHV"]


def _fake_cfg(precompute_path: str) -> dict:
    return {
        "assets": ASSETS,
        "window": 30,
        "transaction_cost": 0.001,
        "data": {"precompute_path": precompute_path},
    }


def _client_with_cfg(cfg: dict) -> TestClient:
    app.dependency_overrides[get_config] = lambda: cfg
    return TestClient(app)


@pytest.fixture(autouse=True)
def _clear_overrides():
    yield
    app.dependency_overrides.clear()


def test_health():
    client = TestClient(app)
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_root():
    client = TestClient(app)
    resp = client.get("/")
    assert resp.status_code == 200
    assert resp.json()["service"] == "RobuSTAM API"


def test_latest_inference_missing_file_returns_503(tmp_path):
    missing = tmp_path / "latest.json"
    client = _client_with_cfg(_fake_cfg(str(missing)))

    resp = client.get("/inference/latest")

    assert resp.status_code == 503


def test_latest_inference_success(tmp_path):
    path = tmp_path / "latest.json"
    payload = {
        "date": "2026-07-14",
        "generated_at": "2026-07-13T09:15:00+09:00",
        "model_version": "ppo_v1",
        "weights": {"SPY": 0.40, "EWY": 0.15, "TLT": 0.20, "GLD": 0.15, "SHV": 0.10},
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    client = _client_with_cfg(_fake_cfg(str(path)))

    resp = client.get("/inference/latest")

    assert resp.status_code == 200
    body = resp.json()
    assert body["date"] == "2026-07-14"
    assert body["weights"] == payload["weights"]


def test_latest_inference_rejects_asset_mismatch(tmp_path):
    path = tmp_path / "latest.json"
    payload = {
        "date": "2026-07-14",
        "generated_at": "2026-07-13T09:15:00+09:00",
        "model_version": "ppo_v1",
        # QQQ가 config 자산 목록에 없음 — 계약 위반
        "weights": {"SPY": 0.40, "EWY": 0.15, "TLT": 0.20, "GLD": 0.15, "QQQ": 0.10},
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    client = _client_with_cfg(_fake_cfg(str(path)))

    resp = client.get("/inference/latest")

    assert resp.status_code == 500


def test_latest_inference_rejects_weights_not_summing_to_one(tmp_path):
    path = tmp_path / "latest.json"
    payload = {
        "date": "2026-07-14",
        "generated_at": "2026-07-13T09:15:00+09:00",
        "model_version": "ppo_v1",
        "weights": {"SPY": 0.40, "EWY": 0.15, "TLT": 0.20, "GLD": 0.15, "SHV": 0.50},
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    client = _client_with_cfg(_fake_cfg(str(path)))

    resp = client.get("/inference/latest")

    assert resp.status_code == 500
