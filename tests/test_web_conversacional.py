"""Tests del agente conversacional web (UX-001, estilo SymptomAI).

La conversación pregunta de una en una y construye el perfil de forma
progresiva; al terminar llama a predict_ensemble, el MISMO camino que usa
POST /api/predict (con _normalize_perfil y _aplicar_deporte_a_nivel). Se
mockea predict_ensemble para capturar el perfil que recibe y devolver un
resultado fijo, sin tocar red ni modelos.
"""

import pytest
from fastapi.testclient import TestClient

from chat.app import _state, app


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


class _FakeDB:
    """DB con el catálogo de factores que consume el chat (vía /api/factores)."""

    def obtener_factores(self, solo_implementados=True, tipo=None):
        return {
            "calor": {
                "comorbilidades": [
                    {"clave": "cardiovascular", "nombre": "Cardiovascular"},
                    {"clave": "diabetes", "nombre": "Diabetes"},
                    {"clave": "renal", "nombre": "Renal"},
                ],
                "farmacos": [
                    {"clave": "diureticos_asa", "nombre": "Diuréticos"},
                    {"clave": "antipsicoticos", "nombre": "Antipsicóticos"},
                ],
                "situacional": [
                    {"clave": "vive_solo", "nombre": "Vivo solo"},
                    {"clave": "alcohol", "nombre": "Alcohol (se oculta en el chat)"},
                    {"clave": "no_sale", "nombre": "Casi no salgo de casa"},
                ],
            }
        }


@pytest.fixture
def db_fake(monkeypatch):
    db = _FakeDB()
    monkeypatch.setattr("chat.app._db", db)
    return db


def _stub_predict(monkeypatch, capturado=None):
    """Sustituye predict_ensemble capturando kwargs y devolviendo un resultado fijo."""

    def _fake(**kwargs):
        if capturado is not None:
            capturado.update(kwargs)
        perfil = kwargs.get("perfil") or {}
        return {
            "clase_final": 1,
            "clase_final_label": "PRECAUCION",
            "perfil": {"calor": {"prob_personalizada": 0.35, "factores": []}},
            "perfil_usuario": perfil,
            "weather": {
                "lat": kwargs.get("lat"),
                "lon": kwargs.get("lon"),
                "provincia": kwargs.get("provincia"),
                "uv_index": 4.0,
                "current": {"t2m_c": 31.0, "rh": 40, "wind_speed_kmh": 10},
                "perfil_horario": [
                    {"hora": 8, "HI": 27.0, "temp": 28.0},
                    {"hora": 9, "HI": 28.0, "temp": 29.0},
                ],
            },
            "modelos": {},
            "recomendaciones": ["Mantente hidratado", "Evita las horas de más calor"],
            "explicacion": {},
        }

    monkeypatch.setattr("climasafeai.models.ensemble.predict_ensemble", _fake)
    return capturado


# Respuesta tipo para una conversación completa: 12 preguntas.
RESPUESTAS_BASE = [
    "Madrid",  # ubicacion
    "45",  # edad
    "mujer",  # sexo
    "no",  # aclimatado
    "moderada",  # nivel_actividad
    "2",  # duracion_actividad_h
    "8",  # hora_inicio
    "1,3",  # comorbilidades (cardiovascular, renal)
    "ninguna",  # farmacos
    "2",  # situacion_social (no_sale; el 1 es vive_solo)
    "saltar",  # porcentaje_grasa (opcional)
    "4",  # fototipo
]


def _turnos_chat(client, respuestas):
    """Recorre la conversación y devuelve la respuesta del servidor a cada turno."""
    estado = None
    out = []
    for msg in respuestas:
        r = client.post("/api/chat", json={"mensaje": msg, "estado": estado})
        data = r.json()
        out.append(data)
        assert "error" not in data or data.get("fin"), (msg, data)
        if data.get("fin"):
            break  # la respuesta final no lleva 'paso'
        estado = {"paso": data["paso"], "perfil": data["perfil"], "ubicacion": data["ubicacion"]}
    return out


# ── Inicio y avance progresivo ───────────────────────────────────────────


def test_chat_primera_pregunta_sin_mensaje(client, db_fake):
    data = client.post("/api/chat", json={}).json()
    assert data["paso"] == 0
    assert data["total"] == 12
    assert "Dónde" in data["pregunta"]
    assert data["perfil"] == {}
    assert "error" not in data


def test_chat_sin_mensaje_devuelve_la_pregunta_actual(client, db_fake):
    turnos = _turnos_chat(client, ["Madrid", "45"])
    data = turnos[-1]
    assert data["campo"] == "sexo"
    # Re-enviar sin mensaje: misma pregunta, sin avanzar ni perder el perfil.
    estado = {"paso": data["paso"], "perfil": data["perfil"], "ubicacion": data["ubicacion"]}
    r = client.post("/api/chat", json={"mensaje": "", "estado": estado}).json()
    assert r["campo"] == "sexo"
    assert r["paso"] == data["paso"]
    assert r["perfil"]["edad"] == 45


def test_chat_avanza_paso_a_paso_acumulando_perfil(client, db_fake):
    turnos = _turnos_chat(client, ["Madrid", "45", "mujer"])
    # Cada respuesta devuelve la siguiente pregunta y el perfil acumulado.
    assert turnos[0]["campo"] == "edad"  # tras ubicacion
    assert turnos[1]["campo"] == "sexo"  # tras edad
    assert turnos[2]["campo"] == "aclimatado"  # tras sexo
    assert turnos[2]["perfil"] == {"edad": 45, "sexo": "mujer"}
    # El eco confirma el valor legible de la última respuesta.
    assert "45" in turnos[1]["respuesta"]
    assert "Mujer" in turnos[2]["respuesta"]


# ── Validación de respuestas ─────────────────────────────────────────────


def test_chat_respuesta_invalida_no_avanza(client, db_fake):
    estado = {"paso": 1, "perfil": {"sexo": "hombre"}, "ubicacion": {"provincia": "Madrid"}}
    data = client.post("/api/chat", json={"mensaje": "abc", "estado": estado}).json()
    assert "error" in data
    assert data["paso"] == 1
    assert data["perfil"] == {"sexo": "hombre"}


def test_chat_saltar_no_avanza_en_pregunta_obligatoria(client, db_fake):
    estado = {"paso": 1, "perfil": {}, "ubicacion": {"provincia": "Madrid"}}
    data = client.post("/api/chat", json={"mensaje": "saltar", "estado": estado}).json()
    assert "error" in data
    assert data["paso"] == 1


def test_chat_numero_fuera_de_rango(client, db_fake):
    estado = {"paso": 5, "perfil": {"edad": 45}, "ubicacion": {"provincia": "Madrid"}}
    data = client.post("/api/chat", json={"mensaje": "30", "estado": estado}).json()
    assert "error" in data
    assert "entre 0 y 24" in data["error"]


def test_chat_cancelar_resetea(client, db_fake):
    estado = {
        "paso": 2,
        "perfil": {"sexo": "hombre", "edad": 45},
        "ubicacion": {"provincia": "Madrid"},
    }
    data = client.post("/api/chat", json={"mensaje": "cancelar", "estado": estado}).json()
    assert data["cancelado"] is True
    assert data["perfil"] == {}


# ── Opciones del catálogo (misma fuente que el formulario) ───────────────


def test_chat_opciones_multiselect_del_catalogo(client, db_fake):
    """Las opciones de comorbilidades salen de /api/factores, no de listas a mano."""
    turnos = _turnos_chat(
        client, ["Madrid", "45", "hombre", "si", "ligera", "1", "8", "ninguna", "ninguna"]
    )
    # El turno que responde a farmacos (paso 8) devuelve la pregunta de
    # situacion_social (paso 9) con sus opciones del catálogo.
    data = turnos[-1]
    assert data["campo"] == "situacion_social"
    claves = [o["clave"] for o in data["opciones"]]
    assert "vive_solo" in claves and "no_sale" in claves
    assert "alcohol" not in claves  # mismo filtro que el formulario


def test_chat_multiselect_numeros_fuera_de_rango(client, db_fake):
    estado = {"paso": 7, "perfil": {"edad": 45}, "ubicacion": {"provincia": "Madrid"}}
    data = client.post("/api/chat", json={"mensaje": "9", "estado": estado}).json()
    assert "error" in data
    assert "1-3" in data["error"]


# ── Integración con el sistema de predicción ─────────────────────────────


def test_chat_flujo_completo_llama_a_predict_ensemble(client, db_fake, monkeypatch):
    capturado = {}
    _stub_predict(monkeypatch, capturado)

    turnos = _turnos_chat(client, RESPUESTAS_BASE)
    data = turnos[-1]

    assert data["fin"] is True
    assert data["resultado"]["clase_final_label"] == "PRECAUCION"

    # El perfil que llega a predict_ensemble es el de la conversación, con los
    # sets normalizados (mismo camino que /api/predict).
    perfil = capturado["perfil"]
    assert perfil["edad"] == 45
    assert perfil["sexo"] == "mujer"
    assert perfil["aclimatado"] is False
    assert perfil["nivel_actividad"] == "moderada"
    assert perfil["duracion_actividad_h"] == 2.0
    assert perfil["hora_inicio"] == 8.0
    assert perfil["comorbilidades"] == {"cardiovascular", "renal"}
    assert perfil["farmacos"] == set()
    assert perfil["situacion_social"] == {"no_sale"}
    assert "porcentaje_grasa" not in perfil  # 'saltar' en el paso opcional
    assert perfil["fototipo"] == "4"
    assert capturado["provincia"] == "Madrid"
    assert capturado["lat"] is None and capturado["lon"] is None


def test_chat_ubicacion_con_coordenadas_llega_a_predict(client, db_fake, monkeypatch):
    capturado = {}
    _stub_predict(monkeypatch, capturado)

    respuestas = ["40.4168,-3.7038"] + RESPUESTAS_BASE[1:]
    turnos = _turnos_chat(client, respuestas)
    assert turnos[-1]["fin"] is True
    assert capturado["lat"] == 40.4168
    assert capturado["lon"] == -3.7038


def test_chat_final_incluye_recomendaciones_contextuales(client, db_fake, monkeypatch):
    _stub_predict(monkeypatch)

    turnos = _turnos_chat(client, RESPUESTAS_BASE)
    data = turnos[-1]

    # Las recomendaciones del agente son las de la herramienta (mismo resultado
    # que /api/predict) y el mensaje final las resume.
    assert data["resultado"]["recomendaciones"] == [
        "Mantente hidratado",
        "Evita las horas de más calor",
    ]
    assert "riesgo" in data["mensaje_final"].lower()
    assert "PRECAUCIÓN" in data["mensaje_final"]


def test_chat_error_de_prediccion_no_tumba_la_conversacion(client, db_fake, monkeypatch):
    def _fake(**kwargs):
        raise RuntimeError("sin red")

    monkeypatch.setattr("climasafeai.models.ensemble.predict_ensemble", _fake)

    turnos = _turnos_chat(client, RESPUESTAS_BASE)
    data = turnos[-1]
    assert data["fin"] is True
    assert "sin red" in data["error"]


def test_chat_estado_desfasado_devuelve_error(client, db_fake):
    data = client.post("/api/chat", json={"mensaje": "hola", "estado": {"paso": 99}}).json()
    assert "error" in data
