"""Tests de MCP-003: control de acceso por identidad en el servidor MCP.

Cubre, criterio a criterio:
  1. sin identidad ninguna tool devuelve datos de ningún perfil
  2. acceso propio permitido / acceso ajeno denegado sin devolver ni un campo
  3. `listar_usuarios_mcp` solo para rol admin
  4. los identificadores son opacos: enumerar o adivinar no da acceso
  5. minimización: qué campos salen de un perfil propio y cuáles no salen nunca
  6. TODA tool registrada pasa por el punto único (marca `__climasafe_acceso__`)
  7. dos identidades conviven sobre la misma base de datos sin verse
  8. propio-permitido / ajeno-denegado para cada tool de lectura

La identidad NO se monkeypatchea: se emiten tokens reales con
`DBManager.emitir_token_mcp` y se ponen en `CLIMASAFE_MCP_TOKEN`, que es
exactamente la vía que usa el transporte stdio (el que configuran `.mcp.json` y
`opencode.json`). Así el test recorre `_identidad_actual` de verdad.
"""

from __future__ import annotations

import asyncio
import inspect
import json

import pytest

import agents.tools.prediction_mcp_tool as mcp


# ── Fixtures ────────────────────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def sin_token_ambiental(monkeypatch):
    """Ningún test hereda el token del entorno del desarrollador."""
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
    """Cambia la identidad del llamante emitiendo un token MCP real."""

    def _como(alias: str | None, rol: str | None = None) -> dict | None:
        if alias is None:
            monkeypatch.delenv(mcp.ENV_TOKEN_MCP, raising=False)
            return None
        match = db.buscar_por_alias(alias)
        assert match, f"no existe el perfil '{alias}'"
        monkeypatch.setenv(mcp.ENV_TOKEN_MCP, db.emitir_token_mcp(match["id"], rol=rol))
        return db.obtener_perfil(match["id"])

    return _como


@pytest.fixture
def dos_usuarios(db, como):
    """A y B, con chat y rutina cada uno. Devuelve (perfil_a, perfil_b)."""
    ida = db.crear_perfil({"alias": "Ana", "edad": 57, "sexo": "mujer",
                           "lat": 42.29, "lon": -8.81, "provincia": "Pontevedra",
                           "telegram_chat_id": "111",
                           "comorbilidades": ["cardiovascular"],
                           "farmacos": ["diureticos_asa"]})
    idb = db.crear_perfil({"alias": "Bruno", "edad": 30, "sexo": "hombre",
                           "lat": 40.4, "lon": -3.7, "provincia": "Madrid",
                           "telegram_chat_id": "222",
                           "comorbilidades": ["diabetes"],
                           "farmacos": ["antipsicoticos"]})
    db.crear_rutina("111", "trabajo de Ana", "1,2,3,4,5", 8.0, 16.0)
    db.crear_rutina("222", "trabajo de Bruno", "1,2,3,4,5", 9.0, 17.0)
    return db.obtener_perfil(ida), db.obtener_perfil(idb)


def _json(s: str) -> dict | list:
    return json.loads(s)


def _es_denegado(salida: str) -> bool:
    datos = _json(salida)
    return isinstance(datos, dict) and "denegado" in str(datos.get("error", "")).lower()


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


# ── Criterio 6: el punto único ──────────────────────────────────────────────


class TestPuntoUnico:
    """El control vive en UNA función y ninguna tool se la salta."""

    def test_toda_tool_registrada_pasa_por_el_control_de_acceso(self):
        tools = asyncio.run(mcp._mcp.list_tools())
        assert tools, "el servidor MCP no registró ninguna tool"
        sin_marca = []
        for t in tools:
            fn = getattr(mcp, t.name, None)
            if fn is None or not hasattr(fn, "__climasafe_acceso__"):
                sin_marca.append(t.name)
        assert not sin_marca, (
            f"estas tools no pasan por _requiere_identidad ni se declararon "
            f"públicas con _acceso_publico: {sorted(sin_marca)}"
        )

    def test_el_guardian_detecta_una_tool_que_se_lo_salta(self):
        """Si el decorador se olvida, el test de arriba falla: aquí se prueba
        que falla de verdad, sin dejar el agujero abierto en el servidor."""

        def tool_recien_llegada_mcp(uid: str) -> str:
            return "datos de cualquiera"

        assert not hasattr(tool_recien_llegada_mcp, "__climasafe_acceso__")

        registradas = {t.name for t in asyncio.run(mcp._mcp.list_tools())}
        candidatas = registradas | {"tool_recien_llegada_mcp"}
        sin_marca = [
            n for n in candidatas
            if not hasattr(getattr(mcp, n, tool_recien_llegada_mcp), "__climasafe_acceso__")
        ]
        assert sin_marca == ["tool_recien_llegada_mcp"]

    def test_los_niveles_declarados_son_los_esperados(self):
        niveles = {
            t.name: getattr(getattr(mcp, t.name), "__climasafe_acceso__", None)
            for t in asyncio.run(mcp._mcp.list_tools())
        }
        assert niveles == {
            "predict_risk_mcp": "publico",
            "grafica_riesgo_horario_mcp": "publico",
            "listar_usuarios_mcp": "admin",
            "crear_perfil_mcp": "identidad",
            "cargar_perfil_mcp": "perfil_propio",
            "cargar_perfil_por_chat_id_mcp": "perfil_propio",
            "vincular_chat_id_mcp": "perfil_propio",
            "listar_rutinas_mcp": "perfil_propio",
            "crear_rutina_mcp": "perfil_propio",
            "borrar_rutina_mcp": "perfil_propio",
            "configurar_hora_aviso_mcp": "perfil_propio",
            "riesgo_rutinas_dia_mcp": "perfil_propio",
        }

    def test_el_decorador_no_cambia_la_firma_publica_de_la_tool(self):
        """`inspect.signature` sigue `__wrapped__`, así que el inputSchema que
        genera FastMCP no se contamina con el envoltorio."""
        por_nombre = {t.name: t for t in asyncio.run(mcp._mcp.list_tools())}
        props = (por_nombre["listar_rutinas_mcp"].inputSchema or {}).get("properties", {})
        assert set(props) == {"alias", "perfil_id", "chat_id"}
        assert "args" not in props and "kwargs" not in props


# ── Criterio 1: sin identidad no sale nada ──────────────────────────────────


class TestSinIdentidad:
    def test_ninguna_tool_de_perfil_devuelve_datos_sin_identidad(self, db, dos_usuarios):
        a, _ = dos_usuarios
        llamadas = [
            ("cargar_perfil_mcp", lambda: mcp.cargar_perfil_mcp()),
            ("cargar_perfil_mcp(uid ajeno)", lambda: mcp.cargar_perfil_mcp(uid=a["uid"])),
            ("cargar_perfil_por_chat_id_mcp", lambda: mcp.cargar_perfil_por_chat_id_mcp(chat_id="111")),
            ("listar_usuarios_mcp", lambda: mcp.listar_usuarios_mcp()),
            ("listar_rutinas_mcp", lambda: mcp.listar_rutinas_mcp(chat_id="111")),
            ("riesgo_rutinas_dia_mcp", lambda: mcp.riesgo_rutinas_dia_mcp(alias="Ana")),
            ("vincular_chat_id_mcp", lambda: mcp.vincular_chat_id_mcp(chat_id="999")),
            ("crear_perfil_mcp", lambda: mcp.crear_perfil_mcp(alias="X", edad=1, sexo="hombre")),
            ("crear_rutina_mcp", lambda: mcp.crear_rutina_mcp(nombre="x", dias="1", hora_inicio=8.0,
                                                             hora_fin=9.0, chat_id="111")),
            ("borrar_rutina_mcp", lambda: mcp.borrar_rutina_mcp(1, chat_id="111")),
            ("configurar_hora_aviso_mcp", lambda: mcp.configurar_hora_aviso_mcp(chat_id="111")),
        ]
        for nombre, llamada in llamadas:
            datos = _json(llamada())
            assert set(datos) == {"error"}, f"{nombre} devolvió campos sin identidad: {datos}"
            assert "anónimos" in datos["error"], nombre

        # y nada se escribió
        assert db.buscar_por_alias("X") is None
        assert len(db.listar_rutinas("111")) == 1

    def test_token_inventado_no_vale(self, db, dos_usuarios, monkeypatch):
        monkeypatch.setenv(mcp.ENV_TOKEN_MCP, "token-que-nadie-emitio")
        assert set(_json(mcp.cargar_perfil_mcp())) == {"error"}

    def test_perfil_sin_token_emitido_no_puede_entrar(self, db, monkeypatch):
        """`mcp_token_hash` NULL significa 'sin acceso por MCP', no 'sin token'."""
        db.crear_perfil({"alias": "SinAcceso", "edad": 40, "sexo": "hombre"})
        monkeypatch.setenv(mcp.ENV_TOKEN_MCP, "")
        assert set(_json(mcp.cargar_perfil_mcp())) == {"error"}

    def test_las_tools_publicas_no_necesitan_identidad(self, prediccion_falsa):
        """Reciben todo por parámetro y no tocan la BD de perfiles."""
        out = _json(mcp.predict_risk_mcp(lat=42.29, lon=-8.81, provincia="Pontevedra"))
        assert out["clase_final_label"] == "PRECAUCIÓN"


# ── Criterios 2, 4, 5, 8: propio sí, ajeno no ───────────────────────────────


class TestAccesoPropioYAjeno:
    def test_cargar_perfil_propio_vs_ajeno(self, dos_usuarios, como):
        a, b = dos_usuarios
        como("Ana")

        propio = _json(mcp.cargar_perfil_mcp(uid=a["uid"]))
        assert propio["alias"] == "Ana"
        assert propio["comorbilidades"] == ["cardiovascular"]

        ajeno = _json(mcp.cargar_perfil_mcp(uid=b["uid"]))
        assert set(ajeno) == {"error"} and "denegado" in ajeno["error"]

    def test_cargar_perfil_sin_uid_carga_el_del_token(self, dos_usuarios, como):
        como("Bruno")
        assert _json(mcp.cargar_perfil_mcp())["alias"] == "Bruno"

    def test_cargar_perfil_por_chat_id_propio_vs_ajeno(self, dos_usuarios, como):
        como("Ana")
        assert _json(mcp.cargar_perfil_por_chat_id_mcp(chat_id="111"))["alias"] == "Ana"
        assert _es_denegado(mcp.cargar_perfil_por_chat_id_mcp(chat_id="222"))

    def test_listar_rutinas_propias_vs_ajenas(self, dos_usuarios, como):
        como("Ana")
        mias = _json(mcp.listar_rutinas_mcp(chat_id="111"))
        assert [r["nombre"] for r in mias] == ["trabajo de Ana"]
        assert _es_denegado(mcp.listar_rutinas_mcp(chat_id="222"))
        assert _es_denegado(mcp.listar_rutinas_mcp(alias="Bruno"))

    def test_riesgo_rutinas_propio_vs_ajeno(self, dos_usuarios, como, prediccion_falsa):
        a, _ = dos_usuarios
        como("Ana")
        mio = _json(mcp.riesgo_rutinas_dia_mcp(alias="Ana", weekday=1))
        assert mio["uid"] == a["uid"] and mio["num_rutinas"] == 1
        assert _es_denegado(mcp.riesgo_rutinas_dia_mcp(alias="Bruno", weekday=1))

    def test_escritura_ajena_denegada(self, dos_usuarios, db, como):
        """Criterio 2 también en las tools que escriben: nadie toca lo de otro."""
        como("Ana")
        ajena = db.listar_rutinas("222")[0]
        assert _es_denegado(mcp.crear_rutina_mcp(nombre="cuña", dias="1", hora_inicio=8.0,
                                                 hora_fin=9.0, chat_id="222"))
        assert _es_denegado(mcp.borrar_rutina_mcp(ajena["id"], chat_id="222"))
        assert _es_denegado(mcp.configurar_hora_aviso_mcp(chat_id="222", hora="07:00"))
        assert len(db.listar_rutinas("222")) == 1
        assert db.obtener_hora_aviso("222") is None

    def test_riesgo_rutinas_solo_devuelve_el_uid(self, dos_usuarios, como, prediccion_falsa):
        """alias, perfil_id y chat_id eran justo las llaves que dejaron de valer."""
        como("Ana")
        out = _json(mcp.riesgo_rutinas_dia_mcp(weekday=1))
        assert "alias" not in out and "perfil_id" not in out and "chat_id" not in out
        assert out["uid"].startswith("usr_")


# ── El caso mixto: sujeto propio + sujeto ajeno en la MISMA llamada ─────────
#
# El guardián validaba solo el primer sujeto no vacío y `_resolver_chat` mira
# `chat_id` primero: nombrando `alias=<propio>, chat_id=<ajeno>` el guardián
# aprobaba uno y la tool usaba el otro. Se leían, creaban y borraban rutinas
# ajenas con un token legítimo.

# Valores para los parámetros obligatorios que NO son sujeto, para poder llamar
# a cualquier tool desde el test del invariante. Si aparece una tool con un
# obligatorio nuevo, el test falla pidiendo que se añada aquí: es deliberado.
_VALORES_EJEMPLO = {
    "rutina_id": 1,
    "nombre": "inyectada",
    "dias": "1",
    "hora_inicio": 1.0,
    "hora_fin": 2.0,
    "hora": "03:00",
    "weekday": 1,
}


def _valor_de_sujeto(clave: str, perfil: dict):
    return {
        "uid": perfil.get("uid"),
        "perfil_id": perfil.get("id"),
        "alias": perfil.get("alias"),
        "chat_id": perfil.get("telegram_chat_id"),
    }[clave]


def _tools_con_dos_sujetos():
    """(nombre, fn, claves) de cada tool que acepta más de un sujeto."""
    for t in asyncio.run(mcp._mcp.list_tools()):
        fn = getattr(mcp, t.name)
        if getattr(fn, "__climasafe_acceso__", None) != "perfil_propio":
            continue
        claves = getattr(fn, "__climasafe_sujeto__", ())
        if len(claves) > 1:
            yield t.name, fn, claves


def _foto_de_la_bd(db):
    """Todo lo que una llamada podría alterar, para comprobar cero efecto."""
    return {
        "rutinas": {chat: sorted((r["id"], r["nombre"]) for r in db.listar_rutinas(chat))
                    for chat in ("111", "222")},
        "avisos": {chat: db.obtener_hora_aviso(chat) for chat in ("111", "222")},
        "chats": {p["alias"]: p["telegram_chat_id"] for p in db.listar_perfiles()},
    }


class TestSujetoMixto:
    def test_cada_tool_con_dos_sujetos_rechaza_propio_mas_ajeno(self, db, dos_usuarios, como):
        """Invariante congelado: para TODA tool que acepte más de un sujeto,
        nombrar uno propio y uno ajeno es rechazo, en cualquier orden y con
        cualquier combinación de parámetros. Una tool nueva con dos sujetos
        entra sola en este test."""
        a, b = dos_usuarios
        como("Ana")

        tools = list(_tools_con_dos_sujetos())
        assert tools, "ninguna tool declara más de un sujeto: el test se quedó ciego"

        antes = _foto_de_la_bd(db)
        comprobadas = []
        for nombre, fn, claves in tools:
            firma = inspect.signature(fn)
            requeridos = [
                p.name for p in firma.parameters.values()
                if p.default is inspect.Parameter.empty and p.name not in claves
            ]
            faltan = [p for p in requeridos if p not in _VALORES_EJEMPLO]
            assert not faltan, f"{nombre}: añade un valor de ejemplo para {faltan}"

            for propia in claves:
                for ajena in claves:
                    if propia == ajena:
                        continue
                    if _valor_de_sujeto(propia, a) is None or _valor_de_sujeto(ajena, b) is None:
                        continue
                    kwargs = {p: _VALORES_EJEMPLO[p] for p in requeridos}
                    kwargs[propia] = _valor_de_sujeto(propia, a)   # el mío, delante
                    kwargs[ajena] = _valor_de_sujeto(ajena, b)     # el de Bruno, detrás
                    salida = _json(fn(**kwargs))
                    assert isinstance(salida, dict) and set(salida) == {"error"}, (
                        f"{nombre}({kwargs}) devolvió datos en vez de un error: {salida}"
                    )
                    assert "denegado" in salida["error"], f"{nombre}({kwargs}): {salida}"
                    comprobadas.append(f"{nombre}[{propia}+{ajena}]")

        assert len(comprobadas) >= 8, comprobadas
        assert _foto_de_la_bd(db) == antes, "una llamada rechazada tocó la BD"

    def test_la_traza_del_reviewer_ya_no_pasa(self, db, dos_usuarios, como):
        """Las tres llamadas exactas con las que se reprodujo el bypass."""
        como("Ana")
        rid_bruno = db.listar_rutinas("222")[0]["id"]
        antes = _foto_de_la_bd(db)

        assert _es_denegado(mcp.listar_rutinas_mcp(alias="Ana", chat_id="222"))
        assert _es_denegado(mcp.configurar_hora_aviso_mcp(hora="03:00", alias="Ana", chat_id="222"))
        assert _es_denegado(mcp.borrar_rutina_mcp(rid_bruno, alias="Ana", chat_id="222"))
        assert _es_denegado(mcp.crear_rutina_mcp(nombre="inyectada", dias="1", hora_inicio=1.0,
                                                 hora_fin=2.0, alias="Ana", chat_id="222"))

        # Bruno intacto, leído de la BD y no del valor devuelto
        assert [r["nombre"] for r in db.listar_rutinas("222")] == ["trabajo de Bruno"]
        assert db.obtener_hora_aviso("222") is None
        assert _foto_de_la_bd(db) == antes

    def test_da_igual_el_orden_de_los_parametros(self, dos_usuarios, como):
        """El ajeno delante o detrás: el guardián no depende del orden."""
        a, b = dos_usuarios
        como("Ana")
        assert _es_denegado(mcp.listar_rutinas_mcp(alias="Ana", chat_id="222"))
        assert _es_denegado(mcp.listar_rutinas_mcp(alias="Bruno", chat_id="111"))
        assert _es_denegado(mcp.listar_rutinas_mcp(perfil_id=a["id"], chat_id="222"))
        assert _es_denegado(mcp.listar_rutinas_mcp(perfil_id=b["id"], chat_id="111"))

    def test_nombrar_dos_sujetos_propios_tambien_es_error(self, dos_usuarios, como):
        """Aunque los dos sean míos, uno sobra: si el guardián mira un parámetro
        y la tool otro, la pregunta '¿cuál manda?' vuelve a existir."""
        como("Ana")
        r = _json(mcp.listar_rutinas_mcp(alias="Ana", chat_id="111"))
        assert isinstance(r, dict) and set(r) == {"error"}, f"devolvió datos: {r}"
        assert "un solo sujeto" in r["error"]
        # y con uno solo sigue funcionando
        assert [x["nombre"] for x in _json(mcp.listar_rutinas_mcp(chat_id="111"))] == ["trabajo de Ana"]

    def test_vincular_chat_id_declara_un_solo_sujeto_a_proposito(self):
        """Su `chat_id` es el chat que se vincula, no quién llama: por eso queda
        fuera del invariante de arriba. La exención es explícita, no un olvido."""
        assert mcp.vincular_chat_id_mcp.__climasafe_sujeto__ == ("uid",)


class TestIdentificadoresOpacos:
    """Criterio 4: enumerar o adivinar identificadores no da acceso a nada."""

    def test_el_uid_es_opaco_y_no_secuencial(self, dos_usuarios):
        a, b = dos_usuarios
        assert a["uid"].startswith("usr_") and len(a["uid"]) > 20
        assert a["uid"] != b["uid"]
        assert str(a["id"]) not in a["uid"]

    def test_adivinar_uid_alias_id_o_chat_no_abre_nada(self, dos_usuarios, como):
        a, _ = dos_usuarios
        como("Bruno")
        for intento in (a["uid"], "usr_" + "a" * 26, "1", str(a["id"]), ""):
            salida = mcp.cargar_perfil_mcp(uid=intento) if intento else mcp.cargar_perfil_mcp()
            datos = _json(salida)
            assert datos.get("alias") != "Ana", f"'{intento}' filtró el perfil de Ana"

    def test_el_error_no_distingue_perfil_ajeno_de_inexistente(self, dos_usuarios, como):
        """Si se distinguieran, enumerar diría quién existe."""
        a, _ = dos_usuarios
        como("Bruno")
        existe = mcp.cargar_perfil_mcp(uid=a["uid"])
        no_existe = mcp.cargar_perfil_mcp(uid="usr_noexisteestoenlabasededatos")
        assert existe == no_existe

    def test_los_perfiles_previos_a_la_migracion_tambien_tienen_uid(self, db):
        """Backfill: un perfil creado antes de MCP-003 no puede quedarse en NULL."""
        with db.conn() as c:
            c.execute("UPDATE perfiles SET uid = NULL")
            c.execute("INSERT INTO perfiles (alias, edad) VALUES ('Viejo', 70)")
        db.initialize()  # vuelve a migrar: el backfill es idempotente
        uids = [p["uid"] for p in db.listar_perfiles()]
        assert all(u and u.startswith("usr_") for u in uids), uids
        assert len(set(uids)) == len(uids), "uids repetidos tras el backfill"

        antes = {p["id"]: p["uid"] for p in db.listar_perfiles()}
        db.initialize()
        assert {p["id"]: p["uid"] for p in db.listar_perfiles()} == antes


class TestMinimizacionDeCampos:
    """Criterio 5: qué ve un llamante legítimo y qué no sale nunca."""

    NUNCA_DE_UN_PERFIL_AJENO = (
        "farmacos", "comorbilidades", "situacion_social", "porcentaje_grasa",
        "fototipo", "fecha_nacimiento", "lat", "lon", "edad", "sexo", "alias",
    )

    def test_del_perfil_propio_no_salen_credenciales_ni_claves_de_fila(self, dos_usuarios, como):
        como("Ana")
        propio = _json(mcp.cargar_perfil_mcp())
        assert "mcp_token_hash" not in propio
        assert "telegram_chat_id" not in propio
        assert "id" not in propio
        # lo que sí ve de lo suyo
        assert propio["uid"].startswith("usr_")
        assert propio["farmacos"] == ["diureticos_asa"]
        assert propio["comorbilidades"] == ["cardiovascular"]

    def test_de_un_perfil_ajeno_no_sale_ni_un_campo(self, dos_usuarios, como):
        a, _ = dos_usuarios
        como("Bruno")
        for salida in (mcp.cargar_perfil_mcp(uid=a["uid"]),
                       mcp.cargar_perfil_por_chat_id_mcp(chat_id="111")):
            datos = _json(salida)
            assert set(datos) == {"error"}
            for campo in self.NUNCA_DE_UN_PERFIL_AJENO:
                assert campo not in datos

    def test_el_hash_del_token_no_sale_del_getter_generico(self, db, dos_usuarios, como):
        a, _ = dos_usuarios
        como("Ana")  # emite el token, así que la fila sí tiene hash
        assert "mcp_token_hash" not in db.obtener_perfil(a["id"])
        with db.conn() as c:
            fila = dict(c.execute("SELECT * FROM perfiles WHERE id=?", (a["id"],)).fetchone())
        assert fila["mcp_token_hash"], "el hash sí está en BD, solo no se devuelve"


# ── Criterio 3: listar_usuarios_mcp ─────────────────────────────────────────


class TestListarUsuarios:
    def test_un_llamante_normal_recibe_error_no_una_lista_vacia(self, dos_usuarios, como):
        como("Ana")
        datos = _json(mcp.listar_usuarios_mcp())
        assert isinstance(datos, dict), "una lista vacía no demuestra nada: debe ser error"
        assert "administración" in datos["error"]

    def test_un_admin_si_la_puede_usar(self, dos_usuarios, como):
        como("Ana", rol="admin")
        datos = _json(mcp.listar_usuarios_mcp())
        assert isinstance(datos, list) and len(datos) == 2
        assert {d["alias"] for d in datos} == {"Ana", "Bruno"}
        assert all(d["uid"].startswith("usr_") for d in datos)
        assert all("telegram_chat_id" not in d for d in datos)


# ── Criterio 7: dos identidades sobre la misma BD ───────────────────────────


class TestMultiusuario:
    def test_dos_sesiones_conviven_sin_verse(self, db, dos_usuarios, como, prediccion_falsa):
        a, b = dos_usuarios

        como("Ana")
        assert _json(mcp.cargar_perfil_mcp())["uid"] == a["uid"]
        assert [r["nombre"] for r in _json(mcp.listar_rutinas_mcp())] == ["trabajo de Ana"]
        assert _es_denegado(mcp.cargar_perfil_mcp(uid=b["uid"]))
        r = _json(mcp.configurar_hora_aviso_mcp(hora="07:00"))
        assert r["success"] is True

        como("Bruno")
        assert _json(mcp.cargar_perfil_mcp())["uid"] == b["uid"]
        assert [r["nombre"] for r in _json(mcp.listar_rutinas_mcp())] == ["trabajo de Bruno"]
        assert _es_denegado(mcp.cargar_perfil_mcp(uid=a["uid"]))
        assert _json(mcp.configurar_hora_aviso_mcp())["hora"] is None  # el de Ana no se ve

        # la misma BD, los dos perfiles intactos
        assert db.obtener_hora_aviso("111") == "07:00"
        assert db.obtener_hora_aviso("222") is None

    def test_reemitir_el_token_invalida_el_anterior(self, dos_usuarios, db, como, monkeypatch):
        como("Ana")
        viejo = mcp._token_del_transporte()
        db.emitir_token_mcp(db.buscar_por_alias("Ana")["id"])
        monkeypatch.setenv(mcp.ENV_TOKEN_MCP, viejo)
        assert set(_json(mcp.cargar_perfil_mcp())) == {"error"}


# ── vincular_chat_id_mcp: la escalada en dos pasos ──────────────────────────


class TestVincularChatId:
    def test_no_se_puede_reasignar_el_perfil_de_otro(self, dos_usuarios, db, como):
        a, _ = dos_usuarios
        como("Bruno")
        r = _json(mcp.vincular_chat_id_mcp(chat_id="999", uid=a["uid"]))
        assert r["error"] and "denegado" in r["error"]
        assert db.obtener_perfil(a["id"])["telegram_chat_id"] == "111"

    def test_no_se_puede_robar_el_chat_de_otro_para_el_perfil_propio(self, dos_usuarios, db, como):
        """El segundo paso de la escalada: las rutinas cuelgan del chat_id, así
        que apropiarse del chat daría acceso a las rutinas de Ana."""
        a, b = dos_usuarios
        como("Bruno")
        r = _json(mcp.vincular_chat_id_mcp(chat_id="111"))
        assert r["success"] is False and "otro perfil" in r["error"]
        assert db.obtener_perfil(b["id"])["telegram_chat_id"] == "222"
        assert db.obtener_perfil(a["id"])["telegram_chat_id"] == "111"
        assert [x["nombre"] for x in _json(mcp.listar_rutinas_mcp())] == ["trabajo de Bruno"]

    def test_vincular_un_chat_libre_al_perfil_propio_si_funciona(self, db, como):
        db.crear_perfil({"alias": "Nuevo", "edad": 25, "sexo": "mujer"})
        como("Nuevo")
        r = _json(mcp.vincular_chat_id_mcp(chat_id="333"))
        assert r["success"] is True
        assert db.buscar_por_alias("Nuevo")
        assert db.obtener_perfil(db.buscar_por_alias("Nuevo")["id"])["telegram_chat_id"] == "333"


class TestCrearPerfil:
    def test_crear_perfil_exige_identidad_y_devuelve_uid_no_id(self, db, dos_usuarios, como):
        como("Ana")
        r = _json(mcp.crear_perfil_mcp(alias="Carla", edad=44, sexo="mujer"))
        assert r["success"] is True
        assert r["uid"].startswith("usr_") and "id" not in r
        assert "mcp-token" in r["mensaje"]

    def test_el_perfil_nuevo_nace_sin_acceso_mcp(self, db, dos_usuarios, como, monkeypatch):
        como("Ana")
        uid = _json(mcp.crear_perfil_mcp(alias="Carla", edad=44, sexo="mujer"))["uid"]
        creado = db.buscar_por_uid(uid)
        with db.conn() as c:
            fila = dict(c.execute("SELECT * FROM perfiles WHERE id=?", (creado["id"],)).fetchone())
        assert fila["mcp_token_hash"] is None
        assert fila["rol"] == "usuario"

    def test_no_se_puede_crear_un_perfil_sobre_el_chat_de_otro(self, db, dos_usuarios, como):
        como("Ana")
        r = _json(mcp.crear_perfil_mcp(alias="Fantasma", edad=44, sexo="mujer", chat_id="222"))
        assert r["success"] is False and "otro perfil" in r["error"]
        assert db.buscar_por_alias("Fantasma") is None
