"""Tests de MCP-002: el servidor MCP arranca en SOLO LECTURA por defecto.

Cubre, criterio a criterio:
  1. la clasificación lectura/escritura queda explícita (`__climasafe_escritura__`)
  2. sin `CLIMASAFE_MCP_WRITE_TOKEN` ninguna tool de escritura toca la BD
  3. con el token de escritura en el entorno, las tools de escritura funcionan
  4. las tools de lectura nunca piden el token de escritura
  5. el token no aparece en la respuesta de ninguna tool (ni en el error)

Misma mecánica que tests/test_mcp_control_acceso.py: identidad real emitida con
`DBManager.emitir_token_mcp` y puesta en `CLIMASAFE_MCP_TOKEN`. El token de
escritura se configura (o no) con `CLIMASAFE_MCP_WRITE_TOKEN`, que es la vía de
arranque de un host en stdio (un proceso = un llamante): el que lo lleve puede
escribir, el que no, solo lee.
"""

from __future__ import annotations

import asyncio
import json

import pytest

import agents.tools.prediction_mcp_tool as mcp


# ── Fixtures ────────────────────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def sin_token_ambiental(monkeypatch):
    """Ningún test hereda los tokens del entorno del desarrollador."""
    monkeypatch.delenv(mcp.ENV_TOKEN_MCP, raising=False)
    monkeypatch.delenv(mcp.ENV_TOKEN_ESCRITURA, raising=False)


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
    """Identidad del llamante: token MCP real en el entorno (vía stdio)."""

    def _como(alias: str) -> dict:
        match = db.buscar_por_alias(alias)
        assert match, f"no existe el perfil '{alias}'"
        monkeypatch.setenv(mcp.ENV_TOKEN_MCP, db.emitir_token_mcp(match["id"]))
        return db.obtener_perfil(match["id"])

    return _como


@pytest.fixture
def con_escritura(monkeypatch):
    """El servidor arrancó con el token de escritura: la escritura está abierta."""
    monkeypatch.setenv(mcp.ENV_TOKEN_ESCRITURA, "secreto-de-escritura-002")


@pytest.fixture
def prediccion_falsa(monkeypatch):
    """Evita llamar al ensemble en las tools que predicen."""

    def _fake(lat, lon, provincia, perfil, target_date=None, resolucion=60):
        return {
            "clase_final": 1, "clase_final_label": "PRECAUCIÓN",
            "weather": {"provincia": provincia, "current": {"t2m_c": 28.0},
                        "perfil_horario": [{"hora": h, "temp": 27.0, "HI": 30.0} for h in range(24)]},
            "modelos": {"Formula": {"frio": {"wind_chill_c": 20}, "calor": {"heat_index_c": 28}}},
            "perfil": {"calor": {"prob_personalizada": 0.35}, "frio": {"prob_personalizada": 0.05}},
        }

    monkeypatch.setattr(mcp, "_try_prediction", _fake)


def _json(s: str) -> dict:
    return json.loads(s)


# Clasificación congelada: la lista de MCP-002. Una tool nueva entra aquí a
# propósito; olvidarse de clasificarla rompe la suite.
TOOLS_ESCRITURA = {
    "crear_perfil_mcp", "crear_rutina_mcp", "borrar_rutina_mcp",
    "vincular_chat_id_mcp", "configurar_hora_aviso_mcp",
}
TOOLS_LECTURA = {
    "predict_risk_mcp", "grafica_riesgo_horario_mcp",
    "listar_usuarios_mcp", "cargar_perfil_mcp",
    "cargar_perfil_por_chat_id_mcp", "listar_rutinas_mcp",
    "riesgo_rutinas_dia_mcp",
}


def _foto_de_la_bd(db):
    """Todo lo que una llamada de escritura podría alterar, para comprobar cero efecto."""
    return {
        "rutinas": {chat: sorted((r["id"], r["nombre"]) for r in db.listar_rutinas(chat))
                    for chat in ("111", "222")},
        "avisos": {chat: db.obtener_hora_aviso(chat) for chat in ("111", "222")},
        "chats": {p["alias"]: p["telegram_chat_id"] for p in db.listar_perfiles()},
    }


# ── Criterio 1: clasificación explícita ─────────────────────────────────────


class TestClasificacion:
    def test_cada_tool_queda_clasificada_lectura_o_escritura(self):
        registradas = {t.name for t in asyncio.run(mcp._mcp.list_tools())}
        assert registradas == TOOLS_ESCRITURA | TOOLS_LECTURA

        escritura = {
            nombre for nombre in registradas
            if getattr(getattr(mcp, nombre), "__climasafe_escritura__", False)
        }
        assert escritura == TOOLS_ESCRITURA, (
            f"tools marcadas como escritura: {sorted(escritura)}"
        )
        assert registradas - escritura == TOOLS_LECTURA

    def test_ninguna_tool_de_lectura_pide_token_de_escritura(self, db, como, prediccion_falsa):
        """Sin token de escritura, todas las de lectura funcionan con su identidad."""
        db.crear_perfil({"alias": "Ana", "edad": 57, "sexo": "mujer",
                         "telegram_chat_id": "111"})
        como("Ana")

        assert _json(mcp.cargar_perfil_mcp())["alias"] == "Ana"
        assert _json(mcp.listar_rutinas_mcp())["rutinas"] == []
        assert _json(mcp.predict_risk_mcp(lat=42.29, lon=-8.81, provincia="Pontevedra"))[
            "clase_final_label"] == "PRECAUCIÓN"


# ── Criterio 2: escritura sin token → rechazo y BD intacta ──────────────────


class TestEscrituraSinToken:
    def test_ninguna_tool_de_escritura_modifica_la_bd_sin_token(self, db, como):
        db.crear_perfil({"alias": "Ana", "edad": 57, "sexo": "mujer",
                         "telegram_chat_id": "111"})
        db.crear_rutina("111", "trabajo", "1,2,3,4,5", 8.0, 16.0)
        como("Ana")
        antes = _foto_de_la_bd(db)

        llamadas = [
            ("crear_perfil_mcp", lambda: mcp.crear_perfil_mcp(alias="Carla", edad=44, sexo="mujer")),
            ("crear_rutina_mcp", lambda: mcp.crear_rutina_mcp(
                chat_id="111", nombre="nueva", dias="1", hora_inicio=8.0, hora_fin=9.0)),
            ("borrar_rutina_mcp", lambda: mcp.borrar_rutina_mcp(1, chat_id="111")),
            ("vincular_chat_id_mcp", lambda: mcp.vincular_chat_id_mcp(chat_id="333")),
            ("configurar_hora_aviso_mcp", lambda: mcp.configurar_hora_aviso_mcp(
                chat_id="111", hora="07:00")),
        ]
        for nombre, llamada in llamadas:
            datos = _json(llamada())
            assert set(datos) == {"error"}, f"{nombre} devolvió datos: {datos}"
            assert "SOLO LECTURA" in datos["error"], f"{nombre}: {datos['error']}"

        assert _foto_de_la_bd(db) == antes, "una llamada rechazada tocó la BD"
        assert db.buscar_por_alias("Carla") is None
        assert db.buscar_por_telegram("333") is None

    def test_la_identidad_sigue_siendo_necesaria_aunque_haya_token_de_escritura(self, con_escritura):
        """La capa de escritura está ENCIMA de la identidad, no la sustituye."""
        datos = _json(mcp.crear_perfil_mcp(alias="X", edad=1, sexo="hombre"))
        assert set(datos) == {"error"} and "anónimos" in datos["error"]

    def test_sin_identidad_ni_escritura_manda_el_error_de_identidad(self, db):
        """El mensaje del llamante anónimo no cambia: MCP-003 sigue intacto."""
        datos = _json(mcp.crear_perfil_mcp(alias="X", edad=1, sexo="hombre"))
        assert set(datos) == {"error"} and "anónimos" in datos["error"]


# ── Criterio 3: escritura con token → funciona ──────────────────────────────


class TestEscrituraConToken:
    def test_las_tools_de_escritura_funcionan_con_el_token(self, db, como, con_escritura):
        db.crear_perfil({"alias": "Ana", "edad": 57, "sexo": "mujer",
                         "telegram_chat_id": "111"})
        como("Ana")

        r = _json(mcp.crear_perfil_mcp(alias="Carla", edad=44, sexo="mujer"))
        assert r["success"] is True and r["uid"].startswith("usr_")
        assert db.buscar_por_alias("Carla") is not None

        r = _json(mcp.crear_rutina_mcp(chat_id="111", nombre="trabajo", dias="1,2,3,4,5",
                                       hora_inicio=8.0, hora_fin=16.0))
        assert r["success"] is True
        assert [x["nombre"] for x in db.listar_rutinas("111")] == ["trabajo"]

        r = _json(mcp.configurar_hora_aviso_mcp(chat_id="111", hora="07:00"))
        assert r["success"] is True and r["hora"] == "07:00"
        assert db.obtener_hora_aviso("111") == "07:00"

        r = _json(mcp.borrar_rutina_mcp(db.listar_rutinas("111")[0]["id"], chat_id="111"))
        assert r["success"] is True
        assert db.listar_rutinas("111") == []

        # el vínculo de un chat nuevo se hace ANTES del borrado de rutinas:
        # al vincular otro chat, el anterior deja de ser "suyo" (MCP-003).
        r = _json(mcp.vincular_chat_id_mcp(chat_id="333"))
        assert r["success"] is True
        assert db.buscar_por_telegram("333")["alias"] == "Ana"

    def test_sin_token_escritura_rechazada_y_con_token_aceptada(self, db, como, con_escritura):
        """La traza del criterio: una llamada con token y otra sin el."""
        db.crear_perfil({"alias": "Ana", "edad": 57, "sexo": "mujer",
                         "telegram_chat_id": "111"})
        como("Ana")

        # sin el token en el entorno: rechazada y la BD intacta
        import os

        os.environ.pop(mcp.ENV_TOKEN_ESCRITURA, None)
        r = _json(mcp.crear_rutina_mcp(chat_id="111", nombre="x", dias="1",
                                       hora_inicio=8.0, hora_fin=9.0))
        assert set(r) == {"error"} and "SOLO LECTURA" in r["error"]
        assert db.listar_rutinas("111") == []

        # con el token: aceptada
        os.environ[mcp.ENV_TOKEN_ESCRITURA] = "secreto-de-escritura-002"
        r = _json(mcp.crear_rutina_mcp(chat_id="111", nombre="x", dias="1",
                                       hora_inicio=8.0, hora_fin=9.0))
        assert r["success"] is True
        assert [x["nombre"] for x in db.listar_rutinas("111")] == ["x"]


# ── Criterio 4: el token no sale en la respuesta ────────────────────────────


class TestTokenNoFiltrado:
    def test_el_token_no_aparece_en_el_error_ni_en_la_respuesta(self, db, como, con_escritura):
        db.crear_perfil({"alias": "Ana", "edad": 57, "sexo": "mujer",
                         "telegram_chat_id": "111"})
        como("Ana")
        secreto = "secreto-de-escritura-002"

        # en el error de solo lectura (el token NO está configurado aquí)
        import os

        os.environ.pop(mcp.ENV_TOKEN_ESCRITURA, None)
        salida = mcp.crear_perfil_mcp(alias="Carla", edad=44, sexo="mujer")
        assert secreto not in salida

        # en la respuesta de éxito (el token SÍ está configurado)
        os.environ[mcp.ENV_TOKEN_ESCRITURA] = secreto
        salida = mcp.crear_perfil_mcp(alias="Carla", edad=44, sexo="mujer")
        assert secreto not in salida
        assert _json(salida)["success"] is True

    def test_el_error_de_solo_lectura_nombra_la_variable_no_el_valor(self):
        """El mensaje ayuda al operador: dice QUÉ variable configurar, no
        ningún valor real de token."""
        assert "CLIMASAFE_MCP_WRITE_TOKEN" in mcp.ERROR_SOLO_LECTURA
        assert "<secreto>" in mcp.ERROR_SOLO_LECTURA
