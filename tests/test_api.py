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


class _DBEspia:
    """DBManager falso: solo apunta qué escrituras se han intentado."""

    def __init__(self):
        self.escrituras = []

    def buscar_por_alias(self, alias):
        return None

    def crear_perfil(self, datos):
        self.escrituras.append(("crear_perfil", datos))
        return 999

    def actualizar_perfil(self, perfil_id, datos):
        self.escrituras.append(("actualizar_perfil", perfil_id))

    def guardar_consulta(self, **kwargs):
        self.escrituras.append(("guardar_consulta", kwargs))


@pytest.fixture
def db_espia(monkeypatch):
    """Sustituye el DBManager real y el ensemble por dobles de prueba."""
    from climasafeai.models import ensemble

    espia = _DBEspia()
    monkeypatch.setattr("chat.app._db", espia)
    monkeypatch.setattr(
        ensemble, "predict_ensemble",
        lambda **kwargs: {
            "clase_final": 0,
            "modelos": {},
            "weather": {"lat": kwargs.get("lat"), "lon": kwargs.get("lon"), "perfil_horario": []},
        },
    )
    return espia


def test_api_predict_persiste_por_defecto(client, db_espia):
    """Una consulta normal del usuario sí crea perfil y consulta."""
    response = client.post("/api/predict", json={
        "provincia": "Madrid",
        "lat": 40.4168,
        "lon": -3.7038,
        "perfil": {"edad": 30},
    })
    assert response.status_code == 200
    acciones = [a for a, _ in db_espia.escrituras]
    assert "crear_perfil" in acciones
    assert "guardar_consulta" in acciones


PERFIL_HORARIO_DEMO = [
    {"hora": h, "HI": hi, "temp": hi - 3}
    for h, hi in zip(range(8, 21), [24, 26, 29, 32, 35, 37, 38, 37, 35, 32, 29, 27, 25])
]


def test_curvas_edad_una_curva_por_edad(client):
    """Con el perfil horario en el body no hace falta ni meteo ni modelos."""
    response = client.post("/api/curvas-edad", json={
        "perfil": {"sexo": "hombre", "nivel_actividad": "ligera"},
        "perfil_horario": PERFIL_HORARIO_DEMO,
        "edades": [25, 55, 85],
    })
    assert response.status_code == 200
    data = response.json()
    assert [c["edad"] for c in data["curvas"]] == [25, 55, 85]
    for c in data["curvas"]:
        assert len(c["curva"]) == len(PERFIL_HORARIO_DEMO)
        assert 0.0 <= c["pico"] <= 1.0
    assert data["umbrales"]["precaucion"] < data["umbrales"]["peligro"]


def test_curvas_edad_mayor_edad_mas_riesgo(client):
    """Las curvas no pueden salir todas iguales: la edad tiene que moverlas."""
    response = client.post("/api/curvas-edad", json={
        "perfil": {"sexo": "hombre", "nivel_actividad": "ligera"},
        "perfil_horario": PERFIL_HORARIO_DEMO,
        "edades": [25, 85],
    })
    picos = {c["edad"]: c["pico"] for c in response.json()["curvas"]}
    assert picos[85] > picos[25]


def test_curvas_edad_no_escribe_en_bbdd(client, db_espia):
    """Son perfiles derivados del real: no deben tocar SQLite."""
    response = client.post("/api/curvas-edad", json={
        "perfil": {"edad": 40},
        "perfil_horario": PERFIL_HORARIO_DEMO,
    })
    assert response.status_code == 200
    assert db_espia.escrituras == []


def test_curvas_edad_usa_edades_por_defecto(client):
    response = client.post("/api/curvas-edad", json={
        "perfil": {},
        "perfil_horario": PERFIL_HORARIO_DEMO,
    })
    from chat.app import EDADES_COMPARATIVA
    assert [c["edad"] for c in response.json()["curvas"]] == list(EDADES_COMPARATIVA)


def test_curvas_edad_sin_perfil_horario_devuelve_error(client, monkeypatch):
    """Sin perfil horario y sin meteo utilizable, error claro y sin reventar."""
    from climasafeai.data import weather_fetcher

    monkeypatch.setattr(weather_fetcher, "fetch_weather_data", lambda **kwargs: {"df_hora": None})
    response = client.post("/api/curvas-edad", json={"perfil": {}, "provincia": "Madrid"})
    assert response.status_code == 200
    assert "error" in response.json()


def test_api_predict_con_persistir_false_no_escribe_en_bbdd(client, db_espia):
    """La comparativa de edades manda perfiles inventados: no deben guardarse."""
    response = client.post("/api/predict", json={
        "provincia": "Madrid",
        "lat": 40.4168,
        "lon": -3.7038,
        "perfil": {"edad": 55},
        "persistir": False,
    })
    assert response.status_code == 200
    assert db_espia.escrituras == []
