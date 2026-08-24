"""FastAPI 추론 서버 테스트.

httpx2는 requirements-dev에만 있으므로 importorskip으로 가드(미설치 CI에선 skip) —
test_s3_sync의 moto 가드와 동일 패턴.

/models/runs·/models/deployed는 진짜 MLflow(Supabase 포함)에 연결하지 않는다 —
`src.api.main.MlflowClient`를 가짜 클라이언트로 monkeypatch해서 격리한다.
"""

import json

import pytest

pytest.importorskip("httpx2")

from fastapi.testclient import TestClient

from src.api import main
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


# ── /models/runs·/models/deployed — MlflowClient는 전부 가짜로 격리 ──

RUN_ID = "50db7bc4c4a541bb843c6d58d190bee5"  # MLflow 32자리 hex 규약


def _fake_model_cfg(model_path: str, *, mlflow_experiment: str | None = None) -> dict:
    cfg: dict = {
        "assets": ASSETS,
        "window": 30,
        "transaction_cost": 0.001,
        "inference": {
            "model_path": model_path,
            "scaler_fold_id": 1,
            "model_version": "ppo-fold1-50db7bc4",
        },
        "model": {"action_bound": 10.0},
    }
    if mlflow_experiment is not None:
        cfg["model"]["mlflow_experiment"] = mlflow_experiment
    return cfg


class _FakeRunInfo:
    def __init__(self, run_id: str, status: str = "FINISHED"):
        self.run_id = run_id
        self.status = status


class _FakeRunData:
    def __init__(self, params: dict, metrics: dict):
        self.params = params
        self.metrics = metrics


class _FakeRun:
    def __init__(self, run_id: str, status: str, params: dict, metrics: dict):
        self.info = _FakeRunInfo(run_id, status)
        self.data = _FakeRunData(params, metrics)


class _FakeExperiment:
    def __init__(self, experiment_id: str = "1"):
        self.experiment_id = experiment_id


def _fake_run(run_id: str = RUN_ID, status: str = "FINISHED", **param_overrides) -> _FakeRun:
    """train.py가 실제로 로깅하는 params를 흉내낸 fake run(기본 valid_sharpe=0.42)."""
    params = {
        "fold_id": "1", "seed": "42", "total_timesteps": "200000",
        "algorithm": "PPO", "policy": "MlpPolicy", "train_cost_multiplier": "1.0",
        # 실제로는 조건부로만 찍히는 값들 — hyperparams에서 걸러지는지 검증용으로 얹는다.
        "learning_rate": "3e-4", "vol_penalty_coef": "0.0", "state_dim": "187",
    }
    params.update(param_overrides)
    return _FakeRun(run_id, status, params, {"valid_sharpe": 0.42})


class _FakeMlflowClient:
    def __init__(
        self, *, experiment=None, runs=None, run_by_id=None,
        experiment_error=None, search_error=None, get_run_error=None,
    ):
        self._experiment = experiment
        self._runs = runs or []
        self._run_by_id = run_by_id or {}
        self._experiment_error = experiment_error
        self._search_error = search_error
        self._get_run_error = get_run_error

    def get_experiment_by_name(self, name):
        if self._experiment_error:
            raise self._experiment_error
        return self._experiment

    def search_runs(self, experiment_ids):
        if self._search_error:
            raise self._search_error
        return self._runs

    def get_run(self, run_id):
        if self._get_run_error:
            raise self._get_run_error
        return self._run_by_id[run_id]


def _patch_mlflow(monkeypatch, fake_client: _FakeMlflowClient, *, tracking_uri="http://fake-mlflow"):
    monkeypatch.setattr(main, "_MLFLOW_TRACKING_URI", tracking_uri)
    monkeypatch.setattr(main, "MlflowClient", lambda *a, **k: fake_client)


def test_models_runs_returns_503_when_tracking_uri_missing(monkeypatch):
    monkeypatch.setattr(main, "_MLFLOW_TRACKING_URI", None)
    client = _client_with_cfg(_fake_model_cfg("ppo_fold1_x_" + RUN_ID + ".zip"))

    resp = client.get("/models/runs")

    assert resp.status_code == 503


def test_models_runs_returns_503_on_experiment_lookup_failure(monkeypatch):
    _patch_mlflow(monkeypatch, _FakeMlflowClient(experiment_error=ConnectionError("연결 실패")))
    client = _client_with_cfg(_fake_model_cfg("ppo_fold1_x_" + RUN_ID + ".zip"))

    resp = client.get("/models/runs")

    assert resp.status_code == 503
    assert "MLflow 연결 실패" in resp.json()["detail"]


def test_models_runs_returns_503_when_experiment_missing(monkeypatch):
    """experiment가 아직 없으면(학습 전) 빈 배열이 아니라 503으로 명확히 알린다."""
    _patch_mlflow(monkeypatch, _FakeMlflowClient(experiment=None))
    client = _client_with_cfg(_fake_model_cfg("ppo_fold1_x_" + RUN_ID + ".zip"))

    resp = client.get("/models/runs")

    assert resp.status_code == 503


def test_models_runs_returns_list_and_marks_deployed_run(monkeypatch):
    other_run_id = "a" * 32
    runs = [
        _fake_run(run_id=RUN_ID, status="FINISHED"),
        _fake_run(run_id=other_run_id, status="RUNNING", fold_id="2", seed="7"),
    ]
    _patch_mlflow(monkeypatch, _FakeMlflowClient(experiment=_FakeExperiment(), runs=runs))
    model_path = f"mlruns/models/ppo_full_fold1_b461a860_{RUN_ID}.zip"
    client = _client_with_cfg(_fake_model_cfg(model_path))

    resp = client.get("/models/runs")

    assert resp.status_code == 200
    body = resp.json()
    assert len(body) == 2
    deployed = next(r for r in body if r["run_id"] == RUN_ID)
    other = next(r for r in body if r["run_id"] == other_run_id)
    assert deployed["deployed"] is True
    assert deployed["fold_id"] == 1
    assert deployed["seed"] == 42
    assert deployed["total_timesteps"] == 200000
    assert deployed["valid_sharpe"] == pytest.approx(0.42)
    assert deployed["status"] == "FINISHED"
    assert other["deployed"] is False
    assert other["status"] == "RUNNING"
    # validMdd·testSharpe는 train.py가 로깅하지 않는 값이라 응답에 아예 없어야 한다.
    for r in body:
        assert "valid_mdd" not in r and "validMdd" not in r
        assert "test_sharpe" not in r and "testSharpe" not in r


def test_deployed_run_id_parses_new_convention_with_combo_segment():
    cfg = _fake_model_cfg(f"mlruns/models/ppo_full_fold1_b461a860_{RUN_ID}.zip")
    assert main._deployed_run_id(cfg) == RUN_ID


def test_deployed_run_id_parses_legacy_convention_without_combo_segment():
    """PR #69(콤보) 이전 배포 model_path엔 콤보 세그먼트가 없다 — 그래도 파싱돼야 한다."""
    cfg = _fake_model_cfg(f"mlruns/models/ppo_fold1_b461a86027e2-20260718102003_{RUN_ID}.zip")
    assert main._deployed_run_id(cfg) == RUN_ID


def test_models_deployed_returns_only_six_hyperparam_keys(monkeypatch):
    model_path = f"mlruns/models/ppo_full_fold1_b461a860_{RUN_ID}.zip"
    fake_run = _fake_run(run_id=RUN_ID)
    _patch_mlflow(
        monkeypatch,
        _FakeMlflowClient(run_by_id={RUN_ID: fake_run}),
    )
    client = _client_with_cfg(_fake_model_cfg(model_path))

    resp = client.get("/models/deployed")

    assert resp.status_code == 200
    body = resp.json()
    assert body["run_id"] == RUN_ID
    assert body["model_version"] == "ppo-fold1-50db7bc4"
    assert body["scaler_fold_id"] == 1
    assert body["valid_sharpe"] == pytest.approx(0.42)
    assert body["hyperparams"] == {
        "algorithm": "PPO",
        "policy": "MlpPolicy",
        "total_timesteps": "200000",
        "seed": "42",
        "train_cost_multiplier": "1.0",
        "action_bound": "10.0",
    }
    # 팀이 노출하지 않기로 한 값들은 run.data.params에 있어도 새어나가면 안 된다.
    assert "learning_rate" not in body["hyperparams"]
    assert "vol_penalty_coef" not in body["hyperparams"]
    assert "state_dim" not in body["hyperparams"]


def test_models_deployed_returns_503_when_run_id_unparseable(monkeypatch):
    _patch_mlflow(monkeypatch, _FakeMlflowClient())
    client = _client_with_cfg(_fake_model_cfg("mlruns/models/not-a-valid-name.zip"))

    resp = client.get("/models/deployed")

    assert resp.status_code == 503


def test_models_deployed_returns_503_on_mlflow_connection_failure(monkeypatch):
    model_path = f"mlruns/models/ppo_full_fold1_b461a860_{RUN_ID}.zip"
    _patch_mlflow(monkeypatch, _FakeMlflowClient(get_run_error=ConnectionError("연결 실패")))
    client = _client_with_cfg(_fake_model_cfg(model_path))

    resp = client.get("/models/deployed")

    assert resp.status_code == 503
    assert "MLflow 연결 실패" in resp.json()["detail"]
