"""Tests de BOT-008 (web): endpoints de rutinas, avisos y pronóstico por ventana.

Sigue el patrón de tests/test_api.py: el SQLite real se sustituye por un fake
DBManager y `predict_ensemble` por un doble que captura el perfil.
"""

from pathlib import Path

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


class _FakeDB:
    """DBManager falso con un perfil web (sin Telegram) y uno sin ubicación."""

    def __init__(self):
        self.perfiles = {
            1: {
                "id": 1,
                "alias": "alex",
                "edad": 40,
                "sexo": "hombre",
                "lat": 40.4,
                "lon": -3.7,
                "provincia": "Madrid",
                "telegram_chat_id": None,
                "aclimatado": False,
                "porcentaje_grasa": None,
                "fototipo": None,
                "comorbilidades": [],
                "farmacos": [],
                "situacion_social": [],
            },
            2: {
                "id": 2,
                "alias": "sin-ubi",
                "edad": 40,
                "sexo": "hombre",
                "lat": None,
                "lon": None,
                "provincia": None,
                "telegram_chat_id": None,
                "aclimatado": False,
                "comorbilidades": [],
                "farmacos": [],
                "situacion_social": [],
            },
            3: {
                "id": 3,
                "alias": "con-telegram",
                "edad": 35,
                "sexo": "mujer",
                "lat": 40.4,
                "lon": -3.7,
                "provincia": "Madrid",
                "telegram_chat_id": "555",
                "aclimatado": False,
                "comorbilidades": [],
                "farmacos": [],
                "situacion_social": [],
            },
        }
        self.rutinas: list[dict] = []
        self.avisos: dict[str, str] = {}
        self._next_id = 1

    def obtener_perfil(self, perfil_id):
        p = self.perfiles.get(perfil_id)
        return dict(p) if p else None

    def listar_rutinas(self, chat_id):
        return [dict(r) for r in self.rutinas if r["chat_id"] == str(chat_id)]

    def crear_rutina(
        self, chat_id, nombre, dias, hora_inicio, hora_fin, ocupacion=None, deporte=None
    ):
        rid = self._next_id
        self._next_id += 1
        self.rutinas.append(
            {
                "id": rid,
                "chat_id": str(chat_id),
                "nombre": nombre,
                "dias": dias,
                "hora_inicio": hora_inicio,
                "hora_fin": hora_fin,
                "ocupacion": ocupacion,
                "deporte": deporte,
            }
        )
        return rid

    def eliminar_rutina(self, rutina_id):
        antes = len(self.rutinas)
        self.rutinas = [r for r in self.rutinas if r["id"] != rutina_id]
        return len(self.rutinas) < antes

    def rutinas_por_dia(self, chat_id, weekday):
        return [
            r
            for r in self.listar_rutinas(chat_id)
            if weekday in {int(d) for d in r["dias"].split(",") if d.strip()}
        ]

    def obtener_hora_aviso(self, chat_id):
        return self.avisos.get(str(chat_id))

    def guardar_hora_aviso(self, chat_id, hora):
        if hora is None:
            self.avisos.pop(str(chat_id), None)
        else:
            self.avisos[str(chat_id)] = hora

    def eliminar_perfil(self, perfil_id):
        return self.perfiles.pop(perfil_id, None) is not None


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
        return {
            "clase_final": 1,
            "clase_final_label": "PRECAUCION",
            "perfil": {"calor": {"prob_personalizada": 0.35}},
            "weather": {
                "perfil_horario": [
                    {"hora": 8, "HI": 27.0, "temp": 28.0},
                    {"hora": 9, "HI": 28.0, "temp": 29.0},
                ],
                "lat": kwargs.get("lat"),
                "lon": kwargs.get("lon"),
            },
            "modelos": {},
        }

    monkeypatch.setattr("climasafeai.models.ensemble.predict_ensemble", _fake)
    return capturado


# ── Rutinas: listar / crear / borrar ───────────────────────────────────────


def test_listar_rutinas_vacias(client, db_fake):
    res = client.get("/api/perfil/1/rutinas")
    assert res.status_code == 200
    assert res.json() == {"rutinas": []}


def test_listar_rutinas_con_datos(client, db_fake):
    db_fake.rutinas.append(
        {
            "id": 1,
            "chat_id": "web_1",
            "nombre": "trabajo",
            "dias": "1,2,3,4,5",
            "hora_inicio": 8.0,
            "hora_fin": 16.0,
            "ocupacion": None,
            "deporte": None,
        }
    )
    res = client.get("/api/perfil/1/rutinas")
    data = res.json()
    assert len(data["rutinas"]) == 1
    assert data["rutinas"][0]["nombre"] == "trabajo"
    assert data["rutinas"][0]["dias"] == "1,2,3,4,5"


def test_listar_rutinas_perfil_no_existe(client, db_fake):
    res = client.get("/api/perfil/999/rutinas")
    assert res.json()["error"] == "Perfil no encontrado"


def test_crear_rutina_ok(client, db_fake):
    res = client.post(
        "/api/perfil/1/rutinas",
        json={
            "nombre": "entreno",
            "dias": "1,3,5",
            "hora_inicio": 18,
            "hora_fin": 20,
            "deporte": "correr",
        },
    )
    assert res.status_code == 200
    rid = res.json()["id"]
    rutinas = db_fake.rutinas
    assert len(rutinas) == 1
    assert rutinas[0]["chat_id"] == "web_1"
    assert rutinas[0]["dias"] == "1,3,5"
    assert rutinas[0]["deporte"] == "correr"
    assert rutinas[0]["id"] == rid


def test_crear_rutina_normaliza_dias_desordenados(client, db_fake):
    res = client.post(
        "/api/perfil/1/rutinas",
        json={
            "nombre": "trabajo",
            "dias": "5,2,1",
            "hora_inicio": 8,
            "hora_fin": 16,
        },
    )
    assert res.status_code == 200
    assert db_fake.rutinas[0]["dias"] == "1,2,5"


def test_crear_rutina_dias_invalidos(client, db_fake):
    res = client.post(
        "/api/perfil/1/rutinas",
        json={
            "nombre": "trabajo",
            "dias": "1,8",
            "hora_inicio": 8,
            "hora_fin": 16,
        },
    )
    assert "error" in res.json()
    assert db_fake.rutinas == []


def test_crear_rutina_horario_invalido(client, db_fake):
    res = client.post(
        "/api/perfil/1/rutinas",
        json={
            "nombre": "trabajo",
            "dias": "1,2",
            "hora_inicio": 20,
            "hora_fin": 8,
        },
    )
    assert "error" in res.json()
    assert db_fake.rutinas == []


def test_crear_rutina_sin_nombre(client, db_fake):
    res = client.post(
        "/api/perfil/1/rutinas",
        json={
            "dias": "1,2",
            "hora_inicio": 8,
            "hora_fin": 16,
        },
    )
    assert "error" in res.json()


def test_borrar_rutina_ok(client, db_fake):
    db_fake.rutinas.append(
        {
            "id": 7,
            "chat_id": "web_1",
            "nombre": "trabajo",
            "dias": "1,2,3",
            "hora_inicio": 8.0,
            "hora_fin": 16.0,
            "ocupacion": None,
            "deporte": None,
        }
    )
    res = client.delete("/api/perfil/1/rutinas/7")
    assert res.json() == {"ok": True}
    assert db_fake.rutinas == []


def test_borrar_rutina_ajena(client, db_fake):
    # La rutina es de otro chat: el perfil no puede borrarla
    db_fake.rutinas.append(
        {
            "id": 7,
            "chat_id": "web_999",
            "nombre": "trabajo",
            "dias": "1,2,3",
            "hora_inicio": 8.0,
            "hora_fin": 16.0,
            "ocupacion": None,
            "deporte": None,
        }
    )
    res = client.delete("/api/perfil/1/rutinas/7")
    assert res.json()["error"] == "Rutina no encontrada"
    assert len(db_fake.rutinas) == 1


# ── Avisos: ver / configurar / desactivar ──────────────────────────────────


def test_avisos_sin_configurar(client, db_fake):
    res = client.get("/api/perfil/1/avisos")
    assert res.json() == {"hora": None}


def test_avisos_configurar(client, db_fake):
    res = client.post("/api/perfil/1/avisos", json={"hora": "08:00"})
    assert res.json() == {"ok": True}
    assert db_fake.avisos == {"web_1": "08:00"}
    assert client.get("/api/perfil/1/avisos").json() == {"hora": "08:00"}


def test_avisos_desactivar(client, db_fake):
    db_fake.avisos["web_1"] = "08:00"
    res = client.post("/api/perfil/1/avisos", json={"hora": None})
    assert res.json() == {"ok": True}
    assert db_fake.avisos == {}


def test_avisos_hora_invalida(client, db_fake):
    res = client.post("/api/perfil/1/avisos", json={"hora": "25:99"})
    assert "error" in res.json()
    assert db_fake.avisos == {}


def test_avisos_normaliza_hora(client, db_fake):
    res = client.post("/api/perfil/1/avisos", json={"hora": "8:05"})
    assert res.json() == {"ok": True}
    assert db_fake.avisos == {"web_1": "08:05"}


# ── Pronóstico del día por ventana ─────────────────────────────────────────


def test_pronostico_sin_ubicacion_error_claro(client, db_fake):
    res = client.post("/api/perfil/2/pronostico-dia", json={})
    assert res.status_code == 400
    assert "ubicación" in res.json()["error"].lower()


def test_pronostico_llama_predict_con_la_ventana_de_cada_rutina(client, db_fake, monkeypatch):
    db_fake.rutinas = [
        {
            "id": 1,
            "chat_id": "web_1",
            "nombre": "trabajo",
            "dias": "1",
            "hora_inicio": 8.0,
            "hora_fin": 16.0,
            "ocupacion": None,
            "deporte": None,
        },
        {
            "id": 2,
            "chat_id": "web_1",
            "nombre": "entreno",
            "dias": "1",
            "hora_inicio": 18.0,
            "hora_fin": 20.0,
            "ocupacion": None,
            "deporte": "correr",
        },
    ]
    capturado: dict = {}
    _stub_predict(monkeypatch, capturado)

    res = client.post("/api/perfil/1/pronostico-dia", json={"weekday": 1})
    assert res.status_code == 200
    data = res.json()
    assert data["weekday"] == 1
    assert len(data["rutinas"]) == 2

    # Primera ventana: la define la rutina, no el perfil
    assert capturado["perfil"]["hora_inicio"] == 18.0
    assert capturado["perfil"]["duracion_actividad_h"] == 2.0
    assert capturado["lat"] == 40.4
    assert capturado["lon"] == -3.7

    # El deporte de la rutina fija la intensidad (correr = 10.5 MET)
    assert capturado["perfil"]["deporte"] == "correr"
    assert capturado["perfil"]["nivel_actividad"] == "muy_intensa"

    ventana_entreno = data["rutinas"][1]
    assert ventana_entreno["clase"] == "PRECAUCION"
    assert ventana_entreno["prob_riesgo"] == 0.35
    assert ventana_entreno["temp_media"] == 28.5


def test_pronostico_weekday_invalido(client, db_fake):
    res = client.post("/api/perfil/1/pronostico-dia", json={"weekday": 9})
    assert "error" in res.json()


def test_pronostico_solo_rutinas_del_dia(client, db_fake, monkeypatch):
    db_fake.rutinas = [
        {
            "id": 1,
            "chat_id": "web_1",
            "nombre": "trabajo",
            "dias": "1",
            "hora_inicio": 8.0,
            "hora_fin": 16.0,
            "ocupacion": None,
            "deporte": None,
        },
        {
            "id": 2,
            "chat_id": "web_1",
            "nombre": "futbol",
            "dias": "6,7",
            "hora_inicio": 18.0,
            "hora_fin": 20.0,
            "ocupacion": None,
            "deporte": "futbol",
        },
    ]
    _stub_predict(monkeypatch)

    res = client.post("/api/perfil/1/pronostico-dia", json={"weekday": 1})
    assert len(res.json()["rutinas"]) == 1
    assert res.json()["rutinas"][0]["nombre"] == "trabajo"


# ── Deporte cambia el resultado: /api/predict y /api/riesgo-colectivo ──────


def test_predict_con_deporte_deriva_nivel_actividad(client, monkeypatch):
    capturado: dict = {}
    _stub_predict(monkeypatch, capturado)

    res = client.post(
        "/api/predict",
        json={
            "provincia": "Madrid",
            "lat": 40.4,
            "lon": -3.7,
            "perfil": {
                "edad": 40,
                "sexo": "hombre",
                "deporte": "correr",
                "nivel_actividad": "ligera",
            },
            "persistir": False,
        },
    )
    assert res.status_code == 200
    # El MET del deporte manda sobre el nivel_actividad por defecto
    assert capturado["perfil"]["deporte"] == "correr"
    assert capturado["perfil"]["nivel_actividad"] == "muy_intensa"


def test_predict_sin_deporte_respeta_nivel_actividad(client, monkeypatch):
    capturado: dict = {}
    _stub_predict(monkeypatch, capturado)

    res = client.post(
        "/api/predict",
        json={
            "provincia": "Madrid",
            "lat": 40.4,
            "lon": -3.7,
            "perfil": {"edad": 40, "sexo": "hombre", "nivel_actividad": "ligera"},
            "persistir": False,
        },
    )
    assert res.status_code == 200
    assert capturado["perfil"]["nivel_actividad"] == "ligera"


def test_predict_deporte_desconocido_no_toca_nivel(client, monkeypatch):
    capturado: dict = {}
    _stub_predict(monkeypatch, capturado)

    res = client.post(
        "/api/predict",
        json={
            "provincia": "Madrid",
            "lat": 40.4,
            "lon": -3.7,
            "perfil": {
                "edad": 40,
                "sexo": "hombre",
                "deporte": "padel",
                "nivel_actividad": "moderada",
            },
            "persistir": False,
        },
    )
    assert res.status_code == 200
    assert capturado["perfil"]["nivel_actividad"] == "moderada"


def test_riesgo_colectivo_con_deporte_deriva_nivel(client, monkeypatch):
    capturado: dict = {}
    _stub_predict(monkeypatch, capturado)

    res = client.post(
        "/api/riesgo-colectivo",
        json={
            "tipo": "numero",
            "lat": 40.4,
            "lon": -3.7,
            "provincia": "Madrid",
            "cantidad": 100,
            "edad_min": 30,
            "edad_max": 40,
            "pct_hombres": 100,
            "actividad": "ligera",
            "deporte": "correr",
        },
    )
    assert res.status_code == 200
    data = res.json()
    assert "error" not in data
    assert capturado["perfil"]["deporte"] == "correr"
    assert capturado["perfil"]["nivel_actividad"] == "muy_intensa"
    # El perfil del mapa de zona también lleva la intensidad derivada
    assert data["perfil_mapa"]["nivel_actividad"] == "muy_intensa"


# ── WEB-004: XSS almacenado en la lista de rutinas ──────────────────────────

_PAYLOAD = '<img src=x onerror="alert(1)">'
_INDEX = Path(__file__).resolve().parents[1] / "chat" / "static" / "index.html"


def test_crear_rutina_no_escapa_el_nombre_en_la_api(client, db_fake):
    """El nombre se guarda y se devuelve tal cual: escapar es cosa del render.

    Lo que se comprueba aquí es que la API lo trata como dato (JSON), no como
    HTML — el arreglo vive en index.html, no en una sanitización del servidor.
    """
    res = client.post(
        "/api/perfil/1/rutinas",
        json={"nombre": _PAYLOAD, "dias": "1", "hora_inicio": 8, "hora_fin": 9},
    )
    assert res.status_code == 200
    assert db_fake.rutinas[0]["nombre"] == _PAYLOAD

    data = client.get("/api/perfil/1/rutinas").json()
    assert data["rutinas"][0]["nombre"] == _PAYLOAD


def test_index_escapa_los_campos_de_rutina_al_pintarlos():
    """Regresión de WEB-004: el render de rutinas concatenaba en innerHTML.

    Una rutina llamada `<img src=x onerror=...>` — creable desde la web o desde
    Telegram, comparten tabla — ejecutaba su código al abrir la página.
    """
    html = _INDEX.read_text(encoding="utf-8")
    assert "function esc(v)" in html, "falta el helper de escapado"

    for campo in ("r.nombre", "r.deporte", "r.ocupacion", "v.nombre", "v.deporte"):
        assert f"esc({campo})" in html, f"{campo} se pinta sin escapar"
        assert f"+ {campo} +" not in html, f"{campo} sigue concatenándose crudo"


# ── WEB-005: los errores salían con HTTP 200 ────────────────────────────────


def test_perfil_inexistente_da_404(client, db_fake):
    res = client.get("/api/perfil/999")
    assert res.status_code == 404
    assert res.json()["error"] == "Perfil no encontrado"


def test_rutinas_de_perfil_inexistente_da_404(client, db_fake):
    assert client.get("/api/perfil/999/rutinas").status_code == 404
    assert client.get("/api/perfil/999/avisos").status_code == 404
    res = client.post("/api/perfil/999/pronostico-dia", json={})
    assert res.status_code == 404


def test_body_invalido_da_400(client, db_fake):
    """Días fuera de rango, horario del revés y hora mal formada → 400, no 200."""
    res = client.post(
        "/api/perfil/1/rutinas",
        json={"nombre": "trabajo", "dias": "1,8", "hora_inicio": 8, "hora_fin": 16},
    )
    assert res.status_code == 400
    assert "dias" in res.json()["error"]

    res = client.post(
        "/api/perfil/1/rutinas",
        json={"nombre": "trabajo", "dias": "1,2", "hora_inicio": 20, "hora_fin": 8},
    )
    assert res.status_code == 400

    res = client.post("/api/perfil/1/avisos", json={"hora": "25:99"})
    assert res.status_code == 400
    assert "HH:MM" in res.json()["error"]

    assert db_fake.rutinas == []


def test_borrar_rutina_ajena_da_404(client, db_fake):
    db_fake.crear_rutina("otro_chat", "suya", "1", 8.0, 9.0)
    res = client.delete("/api/perfil/1/rutinas/1")
    assert res.status_code == 404
    assert len(db_fake.rutinas) == 1


def test_el_cuerpo_del_error_sigue_siendo_error_no_detail(client, db_fake):
    """El frontend hace `if (d.error)` en una veintena de sitios: no se rompe."""
    body = client.get("/api/perfil/999").json()
    assert "error" in body and "detail" not in body


# ── WEB-006: borrar un perfil dejaba rutinas y aviso huérfanos ──────────────


def test_borrar_perfil_web_se_lleva_rutinas_y_aviso(client, db_fake):
    db_fake.crear_rutina("web_1", "trabajo", "1,2,3", 8.0, 16.0)
    db_fake.crear_rutina("web_1", "correr", "6", 18.0, 19.0)
    db_fake.guardar_hora_aviso("web_1", "08:00")
    # Rutina de otro perfil: no se toca
    db_fake.crear_rutina("web_2", "ajena", "1", 8.0, 9.0)

    assert len(db_fake.listar_rutinas("web_1")) == 2

    res = client.delete("/api/perfil/1")
    assert res.status_code == 200
    assert res.json()["rutinas_borradas"] == 2

    assert db_fake.listar_rutinas("web_1") == []
    assert db_fake.obtener_hora_aviso("web_1") is None
    assert db_fake.obtener_perfil(1) is None
    assert len(db_fake.listar_rutinas("web_2")) == 1


def test_borrar_perfil_de_telegram_conserva_sus_rutinas(client, db_fake):
    """El chat sigue existiendo: sus rutinas son del usuario del bot, no de la web.

    El aviso sí se quita — se personalizaba con el perfil que se está borrando,
    y es lo que hacía que el bot siguiera escribiendo a un perfil fantasma.
    """
    db_fake.crear_rutina("555", "trabajo", "1,2", 8.0, 16.0)
    db_fake.guardar_hora_aviso("555", "07:30")

    res = client.delete("/api/perfil/3")
    assert res.status_code == 200
    assert res.json()["rutinas_borradas"] == 0

    assert len(db_fake.listar_rutinas("555")) == 1
    assert db_fake.obtener_hora_aviso("555") is None


def test_borrar_perfil_inexistente_da_404(client, db_fake):
    assert client.delete("/api/perfil/999").status_code == 404
