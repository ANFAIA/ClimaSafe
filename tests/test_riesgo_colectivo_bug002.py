"""Tests de BUG-002: el riesgo colectivo por etiqueta referenciaba nombres que
no existen (F821) y devolvía un error por cada persona del grupo.

Cubre las dos ramas rotas:
  - `POST /api/riesgo-colectivo` con `tipo=etiqueta`: `predict_ensemble`, `lat`,
    `lon`, `provincia` y `date_obj` no estaban definidos en esa rama, así que el
    `except` de cada perfil los convertía en `{"alias": ..., "error": ...}`.
  - `predict_group_risk` del MCP con `tipo=numero`: referenciaba `c[...]`, una
    variable que solo existe en la versión de `chat/app.py` de la que se copió.

Sigue el patrón de tests/test_web_rutinas.py: DBManager falso y `predict_ensemble`
sustituido por un doble que captura los kwargs.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

import agents.tools.prediction_mcp_tool as mcp
from chat.app import app


# ── Dobles ──────────────────────────────────────────────────────────────────


def _perfil(pid: int, alias: str, edad: int) -> dict:
    return {
        "id": pid,
        "alias": alias,
        "edad": edad,
        "sexo": "hombre",
        "lat": 42.24,
        "lon": -8.72,
        "provincia": "Pontevedra",
        "tags": "cuadrilla",
        "aclimatado": False,
        "porcentaje_grasa": 22.0,
        "fototipo": "III",
        "comorbilidades": [],
        "farmacos": [],
        "situacion_social": [],
    }


class _DBConTag:
    """Solo lo que usa la rama de etiqueta: buscar perfiles por tag."""

    def __init__(self):
        self.perfiles = [_perfil(1, "alex", 40), _perfil(2, "aldan", 57)]

    def buscar_por_tag(self, tag: str) -> list[dict]:
        return [p for p in self.perfiles if tag in p["tags"].split(",")]


def _stub_predict(monkeypatch, capturado: dict | None = None):
    def _fake(**kwargs):
        if capturado is not None:
            capturado.update(kwargs)
        return {
            "clase_final": 1,
            "clase_final_label": "PRECAUCION",
            "perfil": {"calor": {"prob_personalizada": 0.35}},
            "weather": {
                "perfil_horario": [
                    {"hora": 8, "HI": 27.0, "temp": 28.0},
                    {"hora": 9, "HI": 28.0, "temp": 29.0},
                    {"hora": 10, "HI": 30.0, "temp": 31.0},
                ],
                "lat": kwargs.get("lat"),
                "lon": kwargs.get("lon"),
            },
            "modelos": {},
            "explicacion": "",
            "recomendaciones": [],
        }

    monkeypatch.setattr("climasafeai.models.ensemble.predict_ensemble", _fake)
    return capturado


@pytest.fixture
def client():
    return TestClient(app)


# ── La rama de etiqueta de la web ───────────────────────────────────────────


def test_riesgo_colectivo_etiqueta_no_devuelve_error_por_persona(client, monkeypatch):
    """El fallo original: cada perfil salía con clave 'error' (NameError)."""
    monkeypatch.setattr("chat.app._db", _DBConTag())
    capturado: dict = {}
    _stub_predict(monkeypatch, capturado)

    res = client.post(
        "/api/riesgo-colectivo",
        json={
            "tipo": "etiqueta",
            "tag": "cuadrilla",
            "lat": 42.24,
            "lon": -8.72,
            "provincia": "Pontevedra",
            "fecha": "2026-08-07",
            "actividad": "moderada",
            "hora_inicio": 8,
            "duracion": 8,
        },
    )
    assert res.status_code == 200
    data = res.json()
    assert "error" not in data
    assert data["total_personas"] == 2
    assert all("error" not in r for r in data["detalle"]), data["detalle"]
    assert [r["clase"] for r in data["detalle"]] == ["PRECAUCION", "PRECAUCION"]
    assert data["en_precaucion"] == 2

    # La ubicación y la fecha del body llegan a la predicción, no se pierden
    from datetime import date

    assert capturado["lat"] == 42.24
    assert capturado["lon"] == -8.72
    assert capturado["provincia"] == "Pontevedra"
    assert capturado["target_date"] == date(2026, 8, 7)


def test_riesgo_colectivo_etiqueta_sin_fecha_predice_para_hoy(client, monkeypatch):
    monkeypatch.setattr("chat.app._db", _DBConTag())
    capturado: dict = {}
    _stub_predict(monkeypatch, capturado)

    res = client.post(
        "/api/riesgo-colectivo",
        json={"tipo": "etiqueta", "tag": "cuadrilla", "lat": 42.24, "lon": -8.72,
              "provincia": "Pontevedra"},
    )
    assert res.status_code == 200
    assert all("error" not in r for r in res.json()["detalle"])
    assert capturado["target_date"] is None


# ── La rama de número del MCP ───────────────────────────────────────────────


def test_predict_group_risk_numero_devuelve_factores_y_resumen(monkeypatch):
    """Antes reventaba con NameError: name 'c' is not defined."""
    class _DBVacia:
        pass

    monkeypatch.setattr("climasafeai.db.manager.DBManager", lambda *a, **k: _DBVacia())
    _stub_predict(monkeypatch)

    res = mcp.predict_group_risk(
        lat=42.24, lon=-8.72, provincia="Pontevedra",
        tipo="numero", cantidad=100, edad_min=30, edad_max=60,
        actividad="moderada", hora_inicio=8, duracion=8,
    )

    assert "error" not in res
    assert res["total_personas"] == 100
    assert isinstance(res["factores_detalle"], list) and res["factores_detalle"]
    assert isinstance(res["resumen"], str) and res["resumen"]
