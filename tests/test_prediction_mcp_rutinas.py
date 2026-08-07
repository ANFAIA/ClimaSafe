"""Tests de BOT-008 (parte MCP): tools de rutinas y deporte cambia el resultado.

Cubre:
  - `_aplicar_deporte_a_perfil`: el MET del deporte manda sobre nivel_actividad
  - `predict_risk` con deporte: el perfil que llega a predict_ensemble lleva el
    nivel derivado del Compendium
  - tools MCP de rutinas (crear/listar/borrar) y hora de aviso, contra un SQLite
    temporal (mismo patrón que tests/test_bot_rutinas_avisos.py)
  - `riesgo_rutinas_dia_mcp`: calcula por ventana con predict_ensemble y da
    error claro si el perfil no tiene lat/lon

El módulo MCP crea DBManager() en cada llamada; se redirige a un SQLite temporal
monkeypatcheando climasafeai.db.manager.DBManager.

Desde MCP-003 las tools exigen identidad del llamante: la fixture `como` emite
un token real y lo pone en CLIMASAFE_MCP_TOKEN, que es la vía del transporte
stdio. Pasar el alias/chat_id de otro ya no da acceso, así que los tests que
operan sobre dos perfiles cambian de identidad explícitamente.
"""

from __future__ import annotations

import json

import pytest

import agents.tools.prediction_mcp_tool as mcp
import climasafeai.db.manager as dbm


# ── Fixtures ────────────────────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def sin_token_ambiental(monkeypatch):
    """Ningún test hereda el token MCP del entorno del desarrollador."""
    monkeypatch.delenv(mcp.ENV_TOKEN_MCP, raising=False)


@pytest.fixture
def db(tmp_path, monkeypatch):
    """DBManager real sobre un SQLite temporal, inyectado en las tools MCP."""
    import climasafeai.db.manager as manager_mod

    class _DB(manager_mod.DBManager):
        def __init__(self, path=None):
            super().__init__(path or str(tmp_path / "test.db"))

    db = _DB()
    db.initialize()
    monkeypatch.setattr(manager_mod, "DBManager", _DB)
    return db


@pytest.fixture
def como(db, monkeypatch):
    """Cambia la identidad del llamante del MCP emitiendo un token real."""

    def _como(alias: str) -> dict:
        match = db.buscar_por_alias(alias)
        assert match, f"no existe el perfil '{alias}'"
        monkeypatch.setenv(mcp.ENV_TOKEN_MCP, db.emitir_token_mcp(match["id"]))
        return db.obtener_perfil(match["id"])

    return _como


def _json(s: str) -> dict:
    return json.loads(s)


# ── El deporte cambia el resultado ──────────────────────────────────────────


class TestDeporteCambiaResultado:
    def test_aplicar_deporte_cambia_nivel_actividad(self):
        p = mcp._aplicar_deporte_a_perfil({"nivel_actividad": "ligera", "deporte": "correr"})
        assert p["nivel_actividad"] == "muy_intensa"  # correr = 10.5 MET

        p = mcp._aplicar_deporte_a_perfil({"nivel_actividad": "reposo", "deporte": "pasear"})
        assert p["nivel_actividad"] == "moderada"  # pasear = 3.5 MET

    def test_deporte_desconocido_no_toca_nivel(self):
        p = mcp._aplicar_deporte_a_perfil({"nivel_actividad": "ligera", "deporte": "padel"})
        assert p["nivel_actividad"] == "ligera"

    def test_sin_deporte_no_toca_nivel(self):
        p = mcp._aplicar_deporte_a_perfil({"nivel_actividad": "intensa"})
        assert p["nivel_actividad"] == "intensa"

    def test_predict_risk_deriva_nivel_del_met_antes_de_prediccion(self, monkeypatch):
        capturado: dict = {}

        def _fake_try_prediction(lat, lon, provincia, perfil, target_date=None, resolucion=60):
            capturado.update(perfil)
            return {"clase_final": 0, "clase_final_label": "SEGURO",
                    "weather": {"current": {"t2m_c": 20.0}, "perfil_horario": []},
                    "modelos": {"Formula": {"frio": {"wind_chill_c": 18}, "calor": {"heat_index_c": 22}}},
                    "perfil": {"calor": {"prob_personalizada": 0.1}, "frio": {"prob_personalizada": 0.1}}}

        monkeypatch.setattr(mcp, "_try_prediction", _fake_try_prediction)
        mcp.predict_risk(lat=42.29, lon=-8.81, provincia="Pontevedra", edad=40,
                         nivel_actividad="ligera", deporte="correr")
        assert capturado["deporte"] == "correr"
        assert capturado["nivel_actividad"] == "muy_intensa"

    def test_predict_risk_pasa_resolucion_a_try_prediction(self, monkeypatch):
        """DATA-007 criterio 3: `predict_risk` acepta `resolucion` (min por
        punto) y se lo pasa a `_try_prediction`; sin él usa el default 60."""
        capturado: dict = {}

        def _fake_try_prediction(lat, lon, provincia, perfil, target_date=None, resolucion=60):
            capturado["resolucion"] = resolucion
            return {"clase_final": 0, "clase_final_label": "SEGURO",
                    "weather": {"current": {"t2m_c": 20.0}, "perfil_horario": []},
                    "modelos": {"Formula": {"frio": {"wind_chill_c": 18}, "calor": {"heat_index_c": 22}}},
                    "perfil": {"calor": {"prob_personalizada": 0.1}, "frio": {"prob_personalizada": 0.1}}}

        monkeypatch.setattr(mcp, "_try_prediction", _fake_try_prediction)

        mcp.predict_risk(lat=42.29, lon=-8.81, provincia="Pontevedra", resolucion=15)
        assert capturado["resolucion"] == 15

        capturado.clear()
        mcp.predict_risk(lat=42.29, lon=-8.81, provincia="Pontevedra")
        assert capturado["resolucion"] == 60


# ── Tools MCP de rutinas ────────────────────────────────────────────────────


class TestRutinasMCP:
    def test_crear_listar_borrar(self, db, como):
        pid = db.crear_perfil({"alias": "Aldán", "edad": 57, "sexo": "hombre",
                               "telegram_chat_id": "111"})
        como("Aldán")
        r = _json(mcp.crear_rutina_mcp(alias="Aldán", nombre="trabajo", dias="1,2,3,4,5",
                                       hora_inicio=8.0, hora_fin=16.0))
        assert r["success"] is True and r["id"] > 0

        r = _json(mcp.crear_rutina_mcp(perfil_id=pid, nombre="correr", dias="6,7",
                                       hora_inicio=18.0, hora_fin=20.0, deporte="correr"))
        assert r["success"] is True

        lista = _json(mcp.listar_rutinas_mcp(alias="Aldán"))
        assert len(lista) == 2
        assert lista[0]["hora_inicio"] == 8.0  # ordenadas por hora
        assert lista[1]["deporte"] == "correr"
        assert lista[1]["dias"] == "6,7"

        # por chat_id también
        assert len(_json(mcp.listar_rutinas_mcp(chat_id="111"))) == 2

        r = _json(mcp.borrar_rutina_mcp(lista[0]["id"], alias="Aldán"))
        assert r["success"] is True
        assert len(_json(mcp.listar_rutinas_mcp(alias="Aldán"))) == 1

        # chats distintos no se mezclan
        db.crear_perfil({"alias": "Otro", "edad": 30, "sexo": "mujer",
                         "telegram_chat_id": "222"})
        como("Otro")
        assert _json(mcp.listar_rutinas_mcp(chat_id="222"))["rutinas"] == []

    def test_sin_identificador_se_opera_sobre_el_perfil_del_token(self, db, como):
        """MCP-003: tener token ya es ser alguien; no hay que nombrarse otra vez."""
        db.crear_perfil({"alias": "A", "edad": 40, "sexo": "hombre", "telegram_chat_id": "1"})
        como("A")
        r = _json(mcp.crear_rutina_mcp(nombre="x", dias="1", hora_inicio=8.0, hora_fin=9.0))
        assert r["success"] is True
        assert [x["nombre"] for x in _json(mcp.listar_rutinas_mcp())] == ["x"]

    def test_crear_sin_identidad_no_escribe(self, db):
        db.crear_perfil({"alias": "A", "edad": 40, "sexo": "hombre", "telegram_chat_id": "1"})
        r = _json(mcp.crear_rutina_mcp(nombre="x", dias="1", hora_inicio=8.0, hora_fin=9.0,
                                       chat_id="1"))
        assert r.get("error") and "anónimos" in r["error"]
        assert db.listar_rutinas("1") == []

    def test_crear_perfil_sin_chat_error_claro(self, db, como):
        db.crear_perfil({"alias": "SinChat", "edad": 40, "sexo": "hombre"})
        como("SinChat")
        r = _json(mcp.crear_rutina_mcp(alias="SinChat", nombre="x", dias="1",
                                       hora_inicio=8.0, hora_fin=9.0))
        assert r.get("error") and "chat de Telegram" in r["error"]

    def test_crear_dias_invalidos(self, db, como):
        db.crear_perfil({"alias": "A", "edad": 40, "sexo": "hombre", "telegram_chat_id": "1"})
        como("A")
        r = _json(mcp.crear_rutina_mcp(chat_id="1", nombre="x", dias="1,8",
                                       hora_inicio=8.0, hora_fin=9.0))
        assert r["success"] is False and "dias" in r["error"]
        assert _json(mcp.listar_rutinas_mcp(chat_id="1"))["rutinas"] == []

    def test_crear_ventana_invalida(self, db, como):
        db.crear_perfil({"alias": "A", "edad": 40, "sexo": "hombre", "telegram_chat_id": "1"})
        como("A")
        r = _json(mcp.crear_rutina_mcp(chat_id="1", nombre="x", dias="1",
                                       hora_inicio=20.0, hora_fin=8.0))
        assert r["success"] is False and "ventana" in r["error"]

    def test_borrar_inexistente(self, db, como):
        db.crear_perfil({"alias": "A", "edad": 40, "sexo": "hombre", "telegram_chat_id": "1"})
        como("A")
        r = _json(mcp.borrar_rutina_mcp(9999, chat_id="1"))
        assert r["success"] is False and "9999" in r["error"]

    def test_borrar_sin_identidad_no_borra(self, db, como):
        db.crear_perfil({"alias": "A", "edad": 40, "sexo": "hombre", "telegram_chat_id": "1"})
        como("A")
        rid = _json(mcp.crear_rutina_mcp(chat_id="1", nombre="x", dias="1",
                                         hora_inicio=8.0, hora_fin=9.0))["id"]
        import os
        os.environ.pop(mcp.ENV_TOKEN_MCP, None)
        r = _json(mcp.borrar_rutina_mcp(rid))
        assert r.get("error") and "anónimos" in r["error"]
        assert len(db.listar_rutinas("1")) == 1

    def test_borrar_rutina_ajena_no_la_borra(self, db, como):
        """MCP-001: sin comprobar propiedad, enumerar ids borraba rutinas de otros.

        MCP-003 añade una capa antes: con la identidad de Intruso, nombrar el
        chat de Dueño ya se rechaza sin llegar a la consulta.
        """
        db.crear_perfil({"alias": "Dueño", "edad": 40, "sexo": "hombre", "telegram_chat_id": "111"})
        db.crear_perfil({"alias": "Intruso", "edad": 30, "sexo": "mujer", "telegram_chat_id": "222"})
        como("Dueño")
        rid = _json(mcp.crear_rutina_mcp(chat_id="111", nombre="trabajo", dias="1",
                                         hora_inicio=8.0, hora_fin=16.0))["id"]

        como("Intruso")
        # nombrando el chat del dueño: lo corta el control de acceso (MCP-003)
        r = _json(mcp.borrar_rutina_mcp(rid, chat_id="111"))
        assert set(r) == {"error"} and "denegado" in r["error"]
        # nombrando el suyo, con el id de la rutina ajena: lo corta MCP-001
        r = _json(mcp.borrar_rutina_mcp(rid, chat_id="222"))
        assert r["success"] is False
        assert len(db.listar_rutinas("111")) == 1

        como("Dueño")
        r = _json(mcp.borrar_rutina_mcp(rid, chat_id="111"))
        assert r["success"] is True
        assert _json(mcp.listar_rutinas_mcp(chat_id="111"))["rutinas"] == []


class TestHoraAvisoMCP:
    def test_ver_configurar_y_desactivar(self, db, como):
        db.crear_perfil({"alias": "A", "edad": 40, "sexo": "hombre", "telegram_chat_id": "1"})
        como("A")
        assert _json(mcp.configurar_hora_aviso_mcp(chat_id="1"))["hora"] is None

        r = _json(mcp.configurar_hora_aviso_mcp(chat_id="1", hora="08:00"))
        assert r["success"] is True and r["hora"] == "08:00"
        assert _json(mcp.configurar_hora_aviso_mcp(chat_id="1"))["hora"] == "08:00"
        # por alias también consulta
        assert _json(mcp.configurar_hora_aviso_mcp(alias="A"))["hora"] == "08:00"

        r = _json(mcp.configurar_hora_aviso_mcp(chat_id="1", hora="off"))
        assert r["success"] is True and r["hora"] is None
        assert _json(mcp.configurar_hora_aviso_mcp(chat_id="1"))["hora"] is None

    def test_hora_invalida(self, db, como):
        db.crear_perfil({"alias": "A", "edad": 40, "sexo": "hombre", "telegram_chat_id": "1"})
        como("A")
        r = _json(mcp.configurar_hora_aviso_mcp(chat_id="1", hora="25:99"))
        assert r["success"] is False and "inválida" in r["error"]
        r = _json(mcp.configurar_hora_aviso_mcp(chat_id="1", hora="8"))
        assert r["success"] is False


# ── Riesgo por ventana de rutina ────────────────────────────────────────────


class TestRiesgoRutinasDiaMCP:
    def test_calcula_riesgo_por_ventana_con_la_rutina(self, db, como, monkeypatch):
        db.crear_perfil({"alias": "Aldán", "edad": 57, "sexo": "hombre",
                         "lat": 42.29, "lon": -8.81, "provincia": "Pontevedra",
                         "telegram_chat_id": "111"})
        como("Aldán")
        _json(mcp.crear_rutina_mcp(chat_id="111", nombre="trabajo", dias="1,2,3,4,5",
                                   hora_inicio=8.0, hora_fin=16.0))
        _json(mcp.crear_rutina_mcp(chat_id="111", nombre="correr", dias="1",
                                   hora_inicio=18.0, hora_fin=20.0, deporte="correr"))

        capturado: dict = {}

        def _fake_try_prediction(lat, lon, provincia, perfil, target_date=None, resolucion=60):
            capturado.update({"lat": lat, "lon": lon, "provincia": provincia, "perfil": perfil})
            return {
                "clase_final": 1, "clase_final_label": "PRECAUCIÓN",
                "weather": {
                    "current": {"t2m_c": 28.0},
                    "perfil_horario": [{"hora": 8, "temp": 27.0}, {"hora": 9, "temp": 28.0},
                                       {"hora": 18, "temp": 30.0}, {"hora": 19, "temp": 31.0}],
                },
                "modelos": {"Formula": {"frio": {"wind_chill_c": 20}, "calor": {"heat_index_c": 28}}},
                "perfil": {"calor": {"prob_personalizada": 0.35}, "frio": {"prob_personalizada": 0.05}},
            }

        monkeypatch.setattr(mcp, "_try_prediction", _fake_try_prediction)

        out = _json(mcp.riesgo_rutinas_dia_mcp(alias="Aldán", weekday=1))
        assert out["num_rutinas"] == 2
        assert capturado["lat"] == 42.29 and capturado["lon"] == -8.81
        assert capturado["provincia"] == "Pontevedra"

        ventana_trabajo = next(v for v in out["ventanas"] if v["nombre"] == "trabajo")
        assert ventana_trabajo["ventana"] == "8:00-16:00"
        assert ventana_trabajo["nivel_actividad"] == "ligera"
        assert ventana_trabajo["clase_final_label"] == "PRECAUCIÓN"
        assert ventana_trabajo["temp_media_ventana_c"] == 27.5

        ventana_correr = next(v for v in out["ventanas"] if v["nombre"] == "correr")
        # el deporte deriva la intensidad del MET: correr = muy_intensa
        assert ventana_correr["nivel_actividad"] == "muy_intensa"
        assert ventana_correr["deporte"] == "correr"

    def test_sin_ubicacion_error_claro(self, db, como, monkeypatch):
        db.crear_perfil({"alias": "SinUbic", "edad": 40, "sexo": "hombre",
                         "telegram_chat_id": "1"})
        como("SinUbic")

        def _no_deberia_calcular(*args, **kwargs):
            raise AssertionError("no se calcula sin ubicación")

        monkeypatch.setattr(mcp, "_try_prediction", _no_deberia_calcular)

        out = _json(mcp.riesgo_rutinas_dia_mcp(alias="SinUbic"))
        assert out.get("error") and "ubicación" in out["error"]

    def test_alias_inexistente_no_dice_si_existe(self, db, como, monkeypatch):
        """MCP-003 criterio 4: el error de un alias que no existe es idéntico al
        de uno ajeno, y no repite el alias probado. Si no, enumerar diría quién
        está dado de alta."""
        db.crear_perfil({"alias": "A", "edad": 40, "sexo": "hombre", "telegram_chat_id": "1"})
        db.crear_perfil({"alias": "Vecino", "edad": 50, "sexo": "mujer", "telegram_chat_id": "2"})
        como("A")

        def _no_deberia_calcular(*args, **kwargs):
            raise AssertionError("no se calcula sin perfil propio")

        monkeypatch.setattr(mcp, "_try_prediction", _no_deberia_calcular)
        inexistente = mcp.riesgo_rutinas_dia_mcp(alias="NoExiste")
        ajeno = mcp.riesgo_rutinas_dia_mcp(alias="Vecino")
        assert inexistente == ajeno
        assert "NoExiste" not in inexistente
        assert "denegado" in _json(inexistente)["error"]

    def test_weekday_invalido(self, db, como):
        db.crear_perfil({"alias": "A", "edad": 40, "sexo": "hombre",
                         "lat": 40.4, "lon": -3.7, "provincia": "Madrid",
                         "telegram_chat_id": "1"})
        como("A")
        out = _json(mcp.riesgo_rutinas_dia_mcp(alias="A", weekday=9))
        assert out.get("error") and "weekday" in out["error"]

    def test_sin_rutinas_ese_dia(self, db, como, monkeypatch):
        db.crear_perfil({"alias": "A", "edad": 40, "sexo": "hombre",
                         "lat": 40.4, "lon": -3.7, "provincia": "Madrid",
                         "telegram_chat_id": "1"})
        como("A")
        _json(mcp.crear_rutina_mcp(chat_id="1", nombre="correr", dias="6,7",
                                   hora_inicio=18.0, hora_fin=20.0, deporte="correr"))

        def _no_deberia_calcular(*args, **kwargs):
            raise AssertionError("sin rutinas ese día no se calcula")

        monkeypatch.setattr(mcp, "_try_prediction", _no_deberia_calcular)
        out = _json(mcp.riesgo_rutinas_dia_mcp(alias="A", weekday=3))
        assert out["num_rutinas"] == 0 and out["ventanas"] == []
