"""Tests de SEC-001 — Protección de la BD de perfiles.

Cubre los criterios de la feature:
1. Permisos 600 del fichero SQLite + fuera del control de versiones.
2. Ningún log escribe datos de salud identificables tras una conversación
   del bot y una predicción desde la web.
3. Backup y restauración de ida y vuelta.
6. make test pasa (esta suite es parte de ella).
Los criterios 4 y 5 (quién escribe en la BD y decisión de cifrado) viven en
documentacion/seguridad_bd.md.
"""

import asyncio
import logging
import os
import re
import stat
import subprocess
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from climasafeai.db.manager import DBManager


# ── Criterio 1 — permisos del fichero ─────────────────────────────────────


def test_db_creada_con_permisos_600(tmp_path):
    """La BD nace con permisos 600, no con el umask del proceso (644)."""
    ruta = tmp_path / "permisos.db"
    db = DBManager(ruta)
    db.initialize()
    assert stat.S_IMODE(os.stat(ruta).st_mode) == 0o600

    # Un uso posterior de la BD mantiene el permiso (se re-aplica en conn()).
    db.crear_perfil({"alias": "a", "edad": 30})
    assert stat.S_IMODE(os.stat(ruta).st_mode) == 0o600


def test_db_creada_sin_initialize_tambien_queda_a_600(tmp_path):
    """Una conexión sin initialize() (creación implícita del fichero) también
    deja el fichero a 600."""
    ruta = tmp_path / "nueva.db"
    db = DBManager(ruta)
    with db.conn() as c:
        c.execute("CREATE TABLE t (x INTEGER)")
    assert stat.S_IMODE(os.stat(ruta).st_mode) == 0o600


def test_db_fuera_del_control_de_versiones():
    """git check-ignore data/climasafe.db → ignorada (regla data/**/*.db)."""
    repo = Path(__file__).resolve().parents[1]
    resultado = subprocess.run(
        ["git", "check-ignore", "-v", "data/climasafe.db"],
        cwd=repo,
        capture_output=True,
        text=True,
    )
    assert resultado.returncode == 0, resultado.stderr
    assert "data/**/*.db" in resultado.stdout


# ── Criterio 3 — backup y restauración ────────────────────────────────────


def test_backup_y_restauracion_ida_y_vuelta(tmp_path):
    db = DBManager(tmp_path / "roundtrip.db")
    db.initialize()
    pid = db.crear_perfil(
        {
            "alias": "ana",
            "edad": 45,
            "sexo": "mujer",
            "comorbilidades": ["cardiovascular"],
            "farmacos": ["diureticos_asa"],
        }
    )
    backup = tmp_path / "backups" / "copia.db"

    info = db.backup(backup)
    assert backup.exists()
    assert stat.S_IMODE(os.stat(backup).st_mode) == 0o600

    # La BD cambia después del backup
    db.actualizar_perfil(pid, {"edad": 46})
    assert db.obtener_perfil(pid)["edad"] == 46

    # Restauramos: los datos del backup sobreviven intactos
    db.restaurar(backup)
    perfil = db.obtener_perfil(pid)
    assert perfil["edad"] == 45
    assert perfil["alias"] == "ana"
    assert perfil["comorbilidades"] == ["cardiovascular"]
    assert perfil["farmacos"] == ["diureticos_asa"]
    # El fichero restaurado sigue con 600
    assert stat.S_IMODE(os.stat(db.db_path).st_mode) == 0o600

    # Segundo round-trip: restaurar otra vez vuelve al mismo estado del backup
    db.crear_perfil({"alias": "luis", "edad": 60})
    db.restaurar(backup)
    assert db.obtener_perfil(pid)["edad"] == 45
    assert db.buscar_por_alias("luis") is None


def test_restaurar_backup_inexistente_es_error(tmp_path):
    db = DBManager(tmp_path / "restore_missing.db")
    db.initialize()
    with pytest.raises(FileNotFoundError):
        db.restaurar(tmp_path / "no_existe.db")


# ── Criterio 2 — sin datos de salud en los logs ──────────────────────────


class _CapturaLog(logging.Handler):
    """Handler que acumula los mensajes ya formateados y filtrados."""

    def __init__(self):
        super().__init__()
        self.mensajes: list[str] = []

    def emit(self, record: logging.LogRecord):
        self.mensajes.append(self.format(record))


CHAT_ID = 123456789  # formato realista de chat_id de Telegram
COORDENADAS = (40.4168, -3.7038)  # Madrid, coordenadas exactas


def _instalar_filtros_de_produccion():
    """El mismo par de filtros que _setup_logging aplica a los handlers reales."""
    from climasafeai.bot.telegram_bot import _OcultarChatId, _OcultarToken

    captura = _CapturaLog()
    captura.setFormatter(logging.Formatter("%(message)s"))
    captura.addFilter(_OcultarToken())
    captura.addFilter(_OcultarChatId())
    root = logging.getLogger()
    root.addHandler(captura)
    root.setLevel(logging.INFO)
    return captura


@pytest.fixture
def bot_con_captura(monkeypatch, tmp_path):
    """Bot montado con BD real en tmp_path + captura de logs con los filtros
    de producción."""
    import climasafeai.bot.telegram_bot as mod
    from climasafeai.bot import geocoding as geo

    captura = _instalar_filtros_de_produccion()
    db = DBManager(tmp_path / "bot_logs.db")
    db.initialize()

    async def _fake_tg(method, **kwargs):
        return {"ok": True, "result": {}}

    monkeypatch.setattr(mod, "_db", db)
    monkeypatch.setattr(mod, "_tg", _fake_tg)
    monkeypatch.setattr(mod, "_modelo_por_defecto", lambda: mod.MODELO_DETERMINISTA)
    # Geocodificación inversa sin red
    monkeypatch.setattr(geo, "provincia_desde_coords", lambda lat, lon: {"provincia": "Madrid", "nombre": "Madrid"})

    yield mod, captura, db
    logging.getLogger().removeHandler(captura)


@pytest.mark.asyncio
async def test_conversacion_del_bot_no_escribe_pii_al_log(bot_con_captura):
    mod, captura, db = bot_con_captura

    # Conversación por el mismo camino que un usuario real: mensajes, ubicación
    # compartida y pulsaciones de botones (callbacks). El chat_id y los datos de
    # salud van en claro por la entrada; el log no debe contenerlos.
    await mod.procesar_update({"message": {"chat": {"id": CHAT_ID}, "text": "/start"}})
    await mod.procesar_update({"message": {"chat": {"id": CHAT_ID}, "text": "soy una persona de 45 años con diabetes"}})
    mod._conversaciones[CHAT_ID] = {"estado": mod.Estado.UBICACION, "data": {}}
    await mod.procesar_update(
        {
            "message": {
                "chat": {"id": CHAT_ID},
                "location": {"latitude": COORDENADAS[0], "longitude": COORDENADAS[1]},
            }
        }
    )
    await mod.procesar_update(
        {
            "callback_query": {
                "id": "1",
                "message": {"chat": {"id": CHAT_ID}, "message_id": 1},
                "data": "edit_edad",
            }
        }
    )

    texto = "\n".join(captura.mensajes)
    # chat_id completo y cualquier número largo de Telegram
    assert str(CHAT_ID) not in texto
    assert not re.search(r"\b\d{6,}\b", texto)
    # datos de salud del mensaje en lenguaje natural
    assert "45 años" not in texto
    assert "diabetes" not in texto
    assert "metformina" not in texto
    # ubicación exacta
    assert str(COORDENADAS[0]) not in texto
    assert str(COORDENADAS[1]) not in texto
    # campo de salud del callback
    assert "edit_edad" not in texto


def test_prediccion_web_no_escribe_pii_al_log(monkeypatch, tmp_path):
    import chat.app as web_app
    from chat.app import app as fastapi_app

    captura = _instalar_filtros_de_produccion()
    db = DBManager(tmp_path / "web_logs.db")
    db.initialize()
    monkeypatch.setattr(web_app, "_db", db)

    def _fake_predict(**kwargs):
        return {
            "clase_final": 1,
            "clase_final_label": "PRECAUCION",
            "tipo": "calor",
            "perfil": {"calor": {"prob_personalizada": 0.35, "factores": []}},
            "perfil_usuario": kwargs.get("perfil") or {},
            "explicacion": {"indice_original": 3.2, "indice_personalizado": 3.5},
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
            "recomendaciones": ["Mantente hidratado"],
        }

    monkeypatch.setattr("climasafeai.models.ensemble.predict_ensemble", _fake_predict)

    client = TestClient(fastapi_app)
    resp = client.post(
        "/api/predict",
        json={
            "provincia": "Madrid",
            "lat": COORDENADAS[0],
            "lon": COORDENADAS[1],
            "perfil": {
                "alias": "web_user_demo",
                "edad": 45,
                "sexo": "mujer",
                "comorbilidades": ["cardiovascular", "diabetes"],
                "farmacos": ["metformina"],
                "situacion_social": ["vive_solo"],
                "aclimatado": False,
                "fototipo": "4",
                "nivel_actividad": "moderada",
                "duracion_actividad_h": 2,
                "hora_inicio": 8,
            },
        },
    )
    assert resp.status_code == 200, resp.text

    texto = "\n".join(captura.mensajes)
    # La predicción persiste el perfil en la BD: los datos de salud no pueden
    # colarse en el log por ninguna vía (ni siquiera por la excepción de
    # guardado, que sí se loguea).
    assert "web_user_demo" not in texto
    assert "metformina" not in texto
    assert "diabetes" not in texto
    assert "cardiovascular" not in texto
    assert "vive_solo" not in texto
    assert str(COORDENADAS[0]) not in texto
    assert str(COORDENADAS[1]) not in texto
