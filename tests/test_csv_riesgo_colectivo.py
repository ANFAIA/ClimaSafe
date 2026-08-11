"""Tests de CSV-001: /api/riesgo-colectivo/csv.

Un CSV de personas devuelve una fila de riesgo por persona más el bloque de
estadísticas del grupo. El factor 'orgullo colectivo' (ver ORGULLO_COLECTIVO
en chat/app.py) solo se aplica cuando `tipo_actividad` es competicion/deporte.
Un CSV con columnas faltantes o valores inválidos devuelve un 400 explicativo,
nunca un 500.

Sigue el patrón de tests/test_riesgo_colectivo_bug002.py: TestClient y
`predict_ensemble` sustituido por un doble que captura los perfiles.
"""

import pytest
from fastapi.testclient import TestClient

from chat.app import app


@pytest.fixture
def client():
    return TestClient(app)


def _stub_predict(monkeypatch, capturado: list | None = None):
    """predict_ensemble falso: clase según la edad, prob_personalizada 0.5 fija."""

    def _fake(**kwargs):
        if capturado is not None:
            capturado.append(dict(kwargs))
        edad = (kwargs.get("perfil") or {}).get("edad") or 40
        if edad >= 60:
            clase, label = 2, "PELIGRO"
        elif edad >= 35:
            clase, label = 1, "PRECAUCION"
        else:
            clase, label = 0, "SEGURO"
        return {
            "clase_final": clase,
            "clase_final_label": label,
            "perfil": {"calor": {"prob_personalizada": 0.5}},
            "weather": {"perfil_horario": []},
            "modelos": {},
        }

    monkeypatch.setattr("climasafeai.models.ensemble.predict_ensemble", _fake)
    return capturado


CSV_EJEMPLO = """nombre,edad,sexo,grasa,nivel_actividad,hora_inicio,duracion,deporte
ana,25,mujer,28,moderada,10,2,
luis,40,hombre,22,ligera,9,8,
marta,70,mujer,30,ligera,8,4,correr
"""


# ── Caso feliz ──────────────────────────────────────────────────────────────


def test_csv_devuelve_fila_por_persona_y_estadisticas(client, monkeypatch):
    capturado: list = []
    _stub_predict(monkeypatch, capturado)

    res = client.post(
        "/api/riesgo-colectivo/csv",
        json={
            "csv": CSV_EJEMPLO,
            "lat": 40.4,
            "lon": -3.7,
            "provincia": "Madrid",
            "tipo_actividad": "deporte",
        },
    )
    assert res.status_code == 200
    data = res.json()
    assert "error" not in data

    # Una fila por persona, con su clase y probabilidad
    assert data["total_personas"] == 3
    assert [r["nombre"] for r in data["detalle"]] == ["ana", "luis", "marta"]
    assert [r["clase"] for r in data["detalle"]] == ["SEGURO", "PRECAUCION", "PELIGRO"]
    assert all("prob_riesgo" in r for r in data["detalle"])

    # Bloque de estadísticas del grupo
    assert data["seguros"] == 1
    assert data["en_precaucion"] == 1
    assert data["en_peligro"] == 1
    assert data["pct_peligro"] == 33.3

    # Los perfiles llegan al ensemble con las columnas mapeadas
    # (grasa→porcentaje_grasa, duracion→duracion_actividad_h)
    ana = next(c for c in capturado if c["perfil"]["edad"] == 25)
    assert ana["perfil"]["sexo"] == "mujer"
    assert ana["perfil"]["porcentaje_grasa"] == 28.0
    assert ana["perfil"]["hora_inicio"] == 10.0
    assert ana["perfil"]["duracion_actividad_h"] == 2.0
    # El deporte de marta fija la intensidad por MET antes de predecir
    marta = next(c for c in capturado if c["perfil"]["edad"] == 70)
    assert marta["perfil"]["deporte"] == "correr"


def test_csv_permite_fecha_y_ubicacion_opcionales(client, monkeypatch):
    capturado: list = []
    _stub_predict(monkeypatch, capturado)

    res = client.post(
        "/api/riesgo-colectivo/csv",
        json={
            "csv": "nombre,edad,sexo\nbea,80,mujer\n",
            "lat": 42.24,
            "lon": -8.72,
            "provincia": "Pontevedra",
            "fecha": "2026-08-07",
        },
    )
    assert res.status_code == 200
    assert res.json()["total_personas"] == 1
    from datetime import date

    assert capturado[0]["lat"] == 42.24
    assert capturado[0]["provincia"] == "Pontevedra"
    assert capturado[0]["target_date"] == date(2026, 8, 7)


def test_csv_sin_campo_csv_ni_ubicacion_errores(client, monkeypatch):
    _stub_predict(monkeypatch)
    res = client.post("/api/riesgo-colectivo/csv", json={"lat": 40.4, "lon": -3.7})
    assert res.status_code == 400
    assert "csv" in res.json()["error"]

    res = client.post(
        "/api/riesgo-colectivo/csv",
        json={"csv": CSV_EJEMPLO, "lat": 40.4},
    )
    assert res.status_code == 400
    assert "lat" in res.json()["error"]


# ── CSV malformado: error explicativo, no 500 ───────────────────────────────


def test_csv_con_columnas_faltantes_error_explicativo(client, monkeypatch):
    _stub_predict(monkeypatch)
    res = client.post(
        "/api/riesgo-colectivo/csv",
        json={"csv": "nombre,sexo\nana,mujer\n", "lat": 40.4, "lon": -3.7},
    )
    assert res.status_code == 400
    detalle = res.json()["error"]
    assert "edad" in detalle
    assert "faltan" in detalle.lower()


def test_csv_con_valor_invalido_error_explicativo(client, monkeypatch):
    _stub_predict(monkeypatch)
    res = client.post(
        "/api/riesgo-colectivo/csv",
        json={"csv": "nombre,edad,sexo\nana,veinte,mujer\n", "lat": 40.4, "lon": -3.7},
    )
    assert res.status_code == 400
    detalle = res.json()["error"]
    assert "ana" in detalle  # dice en qué fila
    assert "edad" in detalle  # y qué campo


def test_csv_con_edad_vacia_error_400(client, monkeypatch):
    _stub_predict(monkeypatch)
    # raise_server_exceptions=False: si el bug reaparece y vuelve el 500,
    # el test falla con el assert de status en vez de con una excepción.
    res = TestClient(app, raise_server_exceptions=False).post(
        "/api/riesgo-colectivo/csv",
        json={"csv": "nombre,edad,sexo\nana,,mujer\n", "lat": 40.4, "lon": -3.7},
    )
    assert res.status_code == 400
    detalle = res.json()["error"]
    assert "ana" in detalle  # dice en qué fila
    assert "edad" in detalle  # y qué campo
    assert "vacía" in detalle


def test_csv_con_sexo_invalido_error_explicativo(client, monkeypatch):
    _stub_predict(monkeypatch)
    res = client.post(
        "/api/riesgo-colectivo/csv",
        json={"csv": "nombre,edad,sexo\nana,25,otro\n", "lat": 40.4, "lon": -3.7},
    )
    assert res.status_code == 400
    assert "sexo" in res.json()["error"]


def test_csv_texto_no_csv_error_explicativo(client, monkeypatch):
    _stub_predict(monkeypatch)
    res = client.post(
        "/api/riesgo-colectivo/csv",
        json={"csv": "esto no es un csv", "lat": 40.4, "lon": -3.7},
    )
    assert res.status_code == 400
    msg = res.json()["error"].lower()
    # O bien no reconoce ninguna persona ("CSV vacío") o bien faltan columnas;
    # en ambos casos el error dice qué pasa, no es un 500 mudo.
    assert "csv" in msg
    assert ("vacío" in msg) or ("faltan columnas" in msg)


# ── Orgullo colectivo: solo con competición/deporte ─────────────────────────


def test_orgullo_colectivo_se_aplica_en_competicion_y_deporte(client, monkeypatch):
    _stub_predict(monkeypatch)

    for tipo in ("competicion", "deporte"):
        res = client.post(
            "/api/riesgo-colectivo/csv",
            json={"csv": CSV_EJEMPLO, "lat": 40.4, "lon": -3.7, "tipo_actividad": tipo},
        )
        assert res.status_code == 200
        data = res.json()
        # prob base 0.5 en odds ×1.2 → 0.5455
        assert data["orgullo_colectivo"]["aplicado"] is True
        assert data["orgullo_colectivo"]["factor"] == 1.2
        assert all(r["factor_orgullo"] == 1.2 for r in data["detalle"])
        assert all(r["prob_riesgo"] == 0.5455 for r in data["detalle"])


def test_orgullo_colectivo_no_se_aplica_sin_competicion_deporte(client, monkeypatch):
    _stub_predict(monkeypatch)

    for tipo in ("ligera", "trabajo", "", None):
        res = client.post(
            "/api/riesgo-colectivo/csv",
            json={
                "csv": CSV_EJEMPLO,
                "lat": 40.4,
                "lon": -3.7,
                **({"tipo_actividad": tipo} if tipo is not None else {}),
            },
        )
        assert res.status_code == 200
        data = res.json()
        assert data["orgullo_colectivo"]["aplicado"] is False
        assert data["orgullo_colectivo"]["factor"] == 1.0
        assert all(r["factor_orgullo"] == 1.0 for r in data["detalle"])
        assert all(r["prob_riesgo"] == 0.5 for r in data["detalle"])
