"""
API tests. A small XGBoost package is trained on the synthetic data, then served
through FastAPI's TestClient, so these run offline and without Docker.
The container itself is tested in CI (serve-docker job).
"""

import json
import os
import uuid

import mlflow
import numpy as np
import pytest
from fastapi.testclient import TestClient

import serve
from config import INFORMATIVE_SENSORS, MLFLOW_TRACKING_URI, PREDICTOR_DIR
from data_processor import load_raw
from export_model import export
from model_trainer import train
from registry import register_run
from rul_predictor import load_predictor


@pytest.fixture(scope="module")
def package():
    train(n_estimators=30, learning_rate=0.1, device="cpu", log_to_mlflow=False)
    return PREDICTOR_DIR


@pytest.fixture
def client(package):
    serve.MODEL.uri = package
    serve.MODEL.predictor = None
    with TestClient(serve.app) as c:
        yield c


def _payload(units=(1, 2, 3), drop=None, all_cycles=False):
    raw = load_raw("test")
    rows = raw[raw["unit"].isin(units)]
    if drop:
        rows = rows.drop(columns=drop)
    return {"cycles": rows.to_dict(orient="records"), "all_cycles": all_cycles}, rows


def test_health_and_model_info(client):
    r = client.get("/health")
    assert r.status_code == 200 and r.json()["status"] == "ok"
    info = client.get("/model").json()
    assert info["model_type"] == "xgboost" and info["source"] == PREDICTOR_DIR


def test_predict_matches_the_packaged_model(client, package):
    body, rows = _payload()
    r = client.post("/predict", json=body)
    assert r.status_code == 200
    assert r.headers["X-Model-Type"] == "xgboost"
    preds = r.json()["predictions"]
    assert [p["unit"] for p in preds] == [1, 2, 3]
    expected = load_predictor(package, fallback=False).predict(rows)["rul"].values
    np.testing.assert_allclose([p["rul"] for p in preds], expected, rtol=1e-6)
    assert all(p["rul_lower"] <= p["rul"] <= p["rul_upper"] for p in preds)


def test_all_cycles_returns_one_row_per_cycle(client):
    body, rows = _payload(units=(1,), all_cycles=True)
    preds = client.post("/predict", json=body).json()["predictions"]
    assert len(preds) == len(rows)
    assert [p["last_cycle"] for p in preds] == list(rows["cycle"].astype(int))


@pytest.mark.parametrize("bad", ["missing_sensor", "gap", "empty", "text"])
def test_bad_requests_are_rejected_with_422(client, bad):
    if bad == "missing_sensor":
        body, _ = _payload(drop=["s4"])
    elif bad == "gap":
        body, _ = _payload(units=(1,))
        del body["cycles"][5]
    elif bad == "empty":
        body = {"cycles": []}
    else:
        body, _ = _payload(units=(1,))
        body["cycles"][0]["s2"] = "high"
    r = client.post("/predict", json=body)
    assert r.status_code == 422
    if bad == "missing_sensor":
        assert "s4" in r.text
    if bad == "gap":
        assert "consecutive" in r.text


def test_oversized_request_is_rejected(client, monkeypatch):
    monkeypatch.setattr(serve, "MAX_ROWS", 10)
    body, _ = _payload(units=(1,))
    assert client.post("/predict", json=body).status_code == 413


def test_no_model_means_503_not_a_crash(package):
    serve.MODEL.uri = "/nonexistent/model"
    serve.MODEL.predictor = None
    with TestClient(serve.app) as c:
        assert c.get("/health").status_code == 503
        body, _ = _payload(units=(1,))
        assert c.post("/predict", json=body).status_code == 503


def test_reload_needs_the_admin_token_when_one_is_set(client, monkeypatch):
    monkeypatch.setattr(serve, "ADMIN_TOKEN", "secret")
    assert client.post("/reload").status_code == 401
    r = client.post("/reload", headers={"X-Admin-Token": "secret"})
    assert r.status_code == 200 and r.json()["model_type"] == "xgboost"


def test_failed_reload_keeps_serving_the_old_model(client):
    serve.MODEL.uri = "/nonexistent/model"
    assert client.post("/reload").status_code == 503
    assert client.get("/health").status_code == 200


def test_export_from_registry_alias_and_serve_it(package, tmp_path):
    """The Docker path: registry champion -> exported folder -> API reports its version."""
    name = f"rul-serve-{uuid.uuid4().hex[:8]}"
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment("serve-tests")
    with mlflow.start_run() as run:
        mlflow.log_artifacts(package, artifact_path="model")
    version = register_run(run.info.run_id, "xgboost",
                           {"val_rmse": 9.1, "test_rmse": 12.5, "nasa_score": 250.0}, 15.0, name=name)
    mlflow.MlflowClient().set_registered_model_alias(name, "champion", version)

    out = str(tmp_path / "champion")
    meta = export(f"models:/{name}@champion", out)
    assert meta["version"] == str(version) and meta["val_rmse"] == 9.1
    assert json.load(open(os.path.join(out, "serving_info.json")))["model_type"] == "xgboost"

    serve.MODEL.uri = out
    serve.MODEL.predictor = None
    with TestClient(serve.app) as c:
        info = c.get("/model").json()
        assert info["version"] == str(version) and info["val_rmse"] == 9.1
        body, _ = _payload(units=(2,))
        r = c.post("/predict", json=body)
        assert r.status_code == 200 and r.headers["X-Model-Version"] == str(version)


def test_sensor_list_is_what_the_api_expects():
    # The README example and serve_smoke.py rely on the 14 informative sensors
    assert len(INFORMATIVE_SENSORS) == 14


def test_integer_columns_from_json_floats_are_cast_back():
    """FD001's s17 holds whole numbers, so models trained on it declare s17 as an integer column.
    JSON sends floats; the API must cast them back or MLflow rejects the request.
    (The real-data case runs end to end in serve_smoke.py, in the serve-docker CI job.)"""
    from fastapi import HTTPException

    rows = [{"unit": 1.0, "cycle": 1.0, "s17": 392.0, "s2": 642.5},
            {"unit": 1.0, "cycle": 2.0, "s17": 393.0, "s2": 642.7}]
    frame = serve._to_frame(rows, ["unit", "cycle", "s17"])
    assert str(frame["s17"].dtype) == "int64" and str(frame["unit"].dtype) == "int64"
    assert str(frame["s2"].dtype) == "float64"

    rows[0]["s17"] = 392.5
    with pytest.raises(HTTPException) as err:
        serve._to_frame(rows, ["unit", "cycle", "s17"])
    assert err.value.status_code == 422 and "s17" in err.value.detail
