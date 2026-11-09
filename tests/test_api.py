"""
tests/test_api.py
-----------------
Automated tests. They run locally (`pytest -v`) and inside the CI/CD pipeline
(GitHub Actions) before every deployment.

They cover: health, valid prediction, the three routing modes, and the
SECURITY behaviours (invalid type, missing field, out-of-range, extra field).
"""
import pytest
from fastapi.testclient import TestClient
from api.main import app

# a valid example request (same as the Swagger example)
VALID = {"lights": 10, "RH_1": 45.5, "RH_2": 40.2, "RH_3": 41.0, "RH_out": 80,
         "Windspeed": 4.0, "Visibility": 40, "hour": 18, "dayofweek": 2}


@pytest.fixture(scope="module")
def client():
    # 'with' makes FastAPI run the startup code that loads the models
    with TestClient(app) as c:
        yield c


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200 and r.json()["status"] == "healthy"


def test_predict_valid(client):
    r = client.post("/predict", json=VALID)
    assert r.status_code == 200
    body = r.json()
    assert body["model_used"] in ("decision_tree", "random_forest")
    assert body["prediction_wh"] >= 0


@pytest.mark.parametrize("mode,expected", [("dt", "decision_tree"), ("rf", "random_forest")])
def test_forced_modes(client, mode, expected):
    r = client.post(f"/predict?mode={mode}", json=VALID)
    assert r.status_code == 200 and r.json()["model_used"] == expected


def test_invalid_type_rejected(client):
    bad = dict(VALID, RH_1="hello")                 # text instead of number
    assert client.post("/predict", json=bad).status_code == 422


def test_missing_field_rejected(client):
    bad = dict(VALID); bad.pop("RH_2")              # required field removed
    assert client.post("/predict", json=bad).status_code == 422


def test_out_of_range_rejected(client):
    bad = dict(VALID, RH_1=500)                     # humidity cannot be 500 %
    assert client.post("/predict", json=bad).status_code == 422


def test_extra_field_rejected(client):
    bad = dict(VALID, home_address="secret")        # unexpected field (data minimisation)
    assert client.post("/predict", json=bad).status_code == 422


def test_metrics_and_drift_endpoints(client):
    assert client.get("/metrics").status_code == 200
    assert client.get("/drift").status_code == 200
