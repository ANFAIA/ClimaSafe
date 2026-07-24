"""Tests de la API REST de ClimaSafeAI (chat/app.py)."""

import pytest
from fastapi.testclient import TestClient

from chat.app import app, _state


@pytest.fixture(autouse=True)
def reset_state():
    _state["models"] = {}
    _state["scaler"] = None
    _state["encoders"] = {}
    _state["feature_names"] = []
    _state["target_encoder"] = None
    _state["model_loaded"] = False
    yield


@pytest.fixture
def client():
    return TestClient(app)


def test_api_status_sin_modelo(client):
    response = client.get("/api/status")
    assert response.status_code == 200
    data = response.json()
    assert data["project"] == "ClimaSafeAI"
    assert data["ml_type"] == "supervisado"
    assert data["model_loaded"] is False
    assert data["models"] == []
    assert data["feature_count"] == 0


def test_api_status_con_modelo(client):
    _state["models"] = {"RandomForest": "fake"}
    _state["model_loaded"] = True
    _state["feature_names"] = ["feat_0", "feat_1"]

    response = client.get("/api/status")
    assert response.status_code == 200
    data = response.json()
    assert data["model_loaded"] is True
    assert "RandomForest" in data["models"]
    assert data["feature_count"] == 2
    assert data["features"] == ["feat_0", "feat_1"]


def test_api_status_tiene_ml_type(client):
    response = client.get("/api/status")
    assert response.status_code == 200
    assert response.json()["ml_type"] == "supervisado"


def test_api_root_devuelve_html(client):
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]


def test_api_predict_sin_modelo_real_devuelve_error(client):
    response = client.post("/api/predict", json={
        "provincia": "Madrid",
        "lat": 40.4168,
        "lon": -3.7038,
        "perfil": {},
    })
    assert response.status_code == 200
    data = response.json()
    assert "error" in data


def test_api_predict_con_fecha_pasada_devuelve_error(client):
    response = client.post("/api/predict", json={
        "provincia": "Madrid",
        "lat": 40.4168,
        "lon": -3.7038,
        "perfil": {},
    }, params={"date": "2020-01-01"})
    assert response.status_code == 200
    data = response.json()
    assert "error" in data
    assert "ya pasó" in data["error"].lower()


def test_api_predict_con_fecha_lejana_devuelve_error(client):
    response = client.post("/api/predict", json={
        "provincia": "Madrid",
        "lat": 40.4168,
        "lon": -3.7038,
        "perfil": {},
    }, params={"date": "2030-06-15"})
    assert response.status_code == 200
    data = response.json()
    assert "error" in data
    assert "2 días" in data["error"]
