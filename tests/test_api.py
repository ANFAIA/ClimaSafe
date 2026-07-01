
"""
test_api.py — Tests de la API REST de ClimaSafeAI.

Usa el cliente de test de FastAPI (httpx) para verificar los endpoints
sin necesidad de levantar un servidor real.
"""
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

import api.main as api_module
from api.main import app, _state


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture(autouse=True)
def reset_state(patch_paths):
    """Resetea el estado global entre tests y parchea rutas."""
    _state["models"]        = {}
    _state["scaler"]        = None
    _state["encoders"]      = {}
    _state["feature_names"] = ["feat_0", "feat_1", "feat_2", "feat_3"]
    _state["target_encoder"] = None
    _state["model_loaded"]  = False
    yield
    _state["models"]        = {}
    _state["model_loaded"]  = False


@pytest.fixture
def client():
    return TestClient(app)


def _inject_model(patch_paths):
    """Entrena un modelo mínimo e inyecta en _state para tests."""

    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler
    import joblib

    np.random.seed(42)
    X = np.random.randn(60, 4)
    y = (X[:, 0] + X[:, 1] > 0).astype(int)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = RandomForestClassifier(n_estimators=5, random_state=42)
    model.fit(X_scaled, y)

    _state["scaler"]         = scaler
    _state["models"]["RandomForest"] = model
    _state["model_loaded"]   = True


# ---------------------------------------------------------------------------
# Tests de /health
# ---------------------------------------------------------------------------
def test_health_sin_modelo(client):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["model_loaded"] is False


def test_health_con_modelo(client, patch_paths):
    _inject_model(patch_paths)
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["model_loaded"] is True


# ---------------------------------------------------------------------------
# Tests de /info
# ---------------------------------------------------------------------------
def test_info_devuelve_metadata(client):
    response = client.get("/info")
    assert response.status_code == 200
    data = response.json()
    assert data["ml_type"] == "supervisado"
    assert "feature_names" in data
    assert isinstance(data["feature_names"], list)


# ---------------------------------------------------------------------------
# Tests de /predict
# ---------------------------------------------------------------------------
def test_predict_sin_modelo_devuelve_503(client):
    payload = {"features": {"feat_0": 1.0, "feat_1": -0.5, "feat_2": 0.3, "feat_3": 0.1}}
    response = client.post("/predict", json=payload)
    assert response.status_code == 503


def test_predict_con_modelo_ok(client, patch_paths):
    _inject_model(patch_paths)
    payload = {"features": {"feat_0": 1.0, "feat_1": -0.5, "feat_2": 0.3, "feat_3": 0.1}}
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "prediction" in data or "cluster" in data
    assert "model_name" in data


def test_predict_features_faltantes_devuelve_422(client, patch_paths):
    _inject_model(patch_paths)
    # Enviar solo 2 de las 4 features esperadas
    payload = {"features": {"feat_0": 1.0}}
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_predict_payload_vacio_devuelve_422(client, patch_paths):
    _inject_model(patch_paths)
    response = client.post("/predict", json={})
    assert response.status_code == 422


def test_predict_clasificacion_tiene_probabilidad(client, patch_paths):
    _inject_model(patch_paths)
    payload = {"features": {"feat_0": 1.0, "feat_1": -0.5, "feat_2": 0.3, "feat_3": 0.1}}
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["probability"] is not None
    assert 0.0 <= data["probability"] <= 1.0



