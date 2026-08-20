"""Tests de WEB-009 — la página individual precarga el perfil guardado de la BD.

El fallo que blindan: los selects del formulario (ocupacion, deporte, ...) mandan
``''`` para «sin elegir», y ese string vacío violaba los CHECK de las columnas
de `perfiles` (``ocupacion IN (...)``, ``fototipo BETWEEN 1 AND 6``, ``sexo``...).
`crear_perfil`/`actualizar_perfil` reventaban con ``sqlite3.IntegrityError`` → 500
→ el perfil nunca se guardaba → la página individual no tenía nada que precargar.

Se usa una BD SQLite REAL con el esquema real (data/schema.sql) y el TestClient,
igual que el flujo de producción. Sigue el patrón de tests/test_web_predict.py.
"""

import pathlib
import tempfile

import pytest
from fastapi.testclient import TestClient

from climasafeai.db.manager import DBManager


@pytest.fixture
def db():
    """Una BD SQLite real con el esquema real del proyecto."""
    gestor = DBManager(tempfile.mktemp(suffix=".db"))
    esquema = pathlib.Path("data/schema.sql").read_text(encoding="utf-8")
    with gestor.conn() as c:
        c.executescript(esquema)
    return gestor


@pytest.fixture
def client(db, monkeypatch):
    import chat.app as web

    monkeypatch.setattr(web, "_db", db)
    return TestClient(web.app)


# El payload exacto que genera getPerfil() de chat/static/index.html al guardar
# un perfil sin elegir los selects opcionales: strings vacíos en ocupacion/deporte
# y hora/duración null (parseFloat('') → NaN → null en JSON).
PAYLOAD_SPA_SIN_OPCIONALES = {
    "edad": 67,
    "fecha_nacimiento": "1959-03-14",
    "sexo": "hombre",
    "porcentaje_grasa": 28.5,
    "nivel_actividad": "moderada",
    "hora_inicio": None,
    "duracion_actividad_h": None,
    "aclimatado": False,
    "entrenado": False,
    "fototipo": "3",
    "deporte": "",
    "ocupacion": "",
    "provincia": "Madrid",
    "comorbilidades": ["cardiovascular", "diabetes"],
    "farmacos": ["diureticos_asa"],
    "situacion_social": ["vive_solo", "sin_aire_acondicionado"],
    "falta_sueno": True,
    "enfermedad_reciente": False,
    "fiesta": False,
    "alias": "jose-67",
}


# ── Criterio 1 y 2: guardar con selects vacíos no revienta y el GET devuelve
# ── los campos que el formulario necesita precargar ─────────────────────────


def test_guardar_perfil_con_selects_vacios_no_revienta(client, db):
    """El payload real de la SPA (ocupacion/deporte '') se guarda: antes 500."""
    res = client.post("/api/perfil", json=PAYLOAD_SPA_SIN_OPCIONALES)
    assert res.status_code == 200, res.text
    pid = res.json()["perfil_id"]
    # El perfil quedó guardado (no NULL porque el INSERT nunca llegó a hacerse)
    guardado = db.obtener_perfil(pid)
    assert guardado["alias"] == "jose-67"
    assert guardado["ocupacion"] is None  # '' → None, no viola el CHECK
    assert guardado["deporte"] is None


def test_get_perfil_devuelve_los_campos_del_formulario(client, db):
    """GET /api/perfil/{id} devuelve todo lo que rellenarFormulario() precarga.

    Campos del formulario individual: edad, sexo, grasa, aclimatado, fototipo,
    comorbilidades, medicación (farmacos), situación social y ubicación.
    """
    pid = client.post("/api/perfil", json=PAYLOAD_SPA_SIN_OPCIONALES).json()["perfil_id"]

    res = client.get(f"/api/perfil/{pid}")
    assert res.status_code == 200
    p = res.json()

    assert p["edad"] == 67
    assert p["sexo"] == "hombre"
    assert p["porcentaje_grasa"] == 28.5
    assert p["aclimatado"] is False
    assert p["fototipo"] == 3
    assert set(p["comorbilidades"]) == {"cardiovascular", "diabetes"}
    assert set(p["farmacos"]) == {"diureticos_asa"}
    assert set(p["situacion_social"]) == {"vive_solo", "sin_aire_acondicionado"}
    assert p["provincia"] == "Madrid"


def test_get_perfil_devuelve_ocupacion_si_se_eligio(client):
    """Con ocupación elegida, el GET la devuelve y el formulario la rellena."""
    payload = dict(PAYLOAD_SPA_SIN_OPCIONALES)
    payload["ocupacion"] = "mantenimiento"
    pid = client.post("/api/perfil", json=payload).json()["perfil_id"]
    p = client.get(f"/api/perfil/{pid}").json()
    assert p["ocupacion"] == "mantenimiento"


# ── Criterio 3: un perfil inexistente no revienta ──────────────────────────


def test_perfil_inexistente_da_404_sin_reventar(client):
    """La página sigue pidiendo datos: 404 con cuerpo {'error': ...}."""
    res = client.get("/api/perfil/99999")
    assert res.status_code == 404
    body = res.json()
    assert body["error"] == "Perfil no encontrado"
    # El frontend comprueba `if (p.error)`: nunca debe recibir 'detail'
    assert "detail" not in body


# ── /api/predict también persiste el perfil con el mismo payload ───────────


def test_predict_persiste_perfil_con_selects_vacios(client, db, monkeypatch):
    """/api/predict guarda el perfil tras predecir; los '' no lo tumban (500)."""
    import chat.app as web
    import climasafeai.models.ensemble as ens

    def _fake_predict(**kwargs):
        return {
            "clase_final": 1,
            "clase_final_label": "PRECAUCION",
            "tipo": "calor",
            "perfil": {"calor": {"prob_personalizada": 0.5}, "frio": {"prob_personalizada": 0.4}},
            "perfil_usuario": {},
            "weather": {},
            "modelos": {},
            "explicacion": {},
        }

    monkeypatch.setattr(ens, "predict_ensemble", _fake_predict)

    res = client.post(
        "/api/predict",
        json={
            "provincia": "Madrid",
            "lat": 40.4,
            "lon": -3.7,
            "perfil": {
                "edad": 67,
                "sexo": "hombre",
                "ocupacion": "",
                "deporte": "",
                "comorbilidades": [],
                "alias": "jose-67",
            },
        },
    )
    assert res.status_code == 200, res.text
    assert "perfil_id" in res.json()
    assert db.obtener_perfil(res.json()["perfil_id"])["alias"] == "jose-67"
