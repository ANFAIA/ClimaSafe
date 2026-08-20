#!/usr/bin/env python3
"""scripts/demo_seguridad_logs.py — Evidencia del criterio 2 de SEC-001.

Simula una conversación completa del bot y una predicción desde la web por los
mismos caminos de código que un uso real (procesar_update y POST /api/predict),
con el logging de producción activo (logs/bot.log con los filtros _OcultarToken
y _OcultarChatId) y hace grep sobre los logs para demostrar que no hay datos de
salud identificables.

Uso:
    python scripts/demo_seguridad_logs.py
"""

from __future__ import annotations

import asyncio
import logging
import re
import sys
from pathlib import Path

PROYECTO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROYECTO))

CHAT_ID = 123456789
COORDENADAS = (40.4168, -3.7038)  # Madrid, exactas
PATRONES_PII = [
    (r"123456789", "chat_id completo"),
    (r"\b\d{6,}\b", "identificador numérico largo (chat_id)"),
    (r"45 años", "edad en lenguaje natural"),
    (r"diabetes", "comorbilidad"),
    (r"metformina", "medicación"),
    (r"40\.4168", "latitud exacta"),
    (r"-3\.7038", "longitud exacta"),
    (r"edit_edad", "campo de salud (callback)"),
]


def main() -> int:
    # ── Logging de producción: logs/bot.log con los filtros del bot ──
    import climasafeai.bot.telegram_bot as bot_mod
    from climasafeai.bot import geocoding as geo
    from climasafeai.db.manager import DBManager

    bot_mod._setup_logging()
    log_file = PROYECTO / "logs" / "bot.log"
    # Solo se grepea el tramo NUEVO del log (lo que esta demo escribe): el
    # fichero histórico puede arrastrar líneas anteriores a SEC-001.
    offset_inicial = log_file.stat().st_size if log_file.exists() else 0

    db = DBManager(PROYECTO / "data" / "climasafe.db")
    db.initialize()
    bot_mod._db = db

    async def _fake_tg(method, **kwargs):
        return {"ok": True, "result": {}}

    bot_mod._tg = _fake_tg
    bot_mod._modelo_por_defecto = lambda: bot_mod.MODELO_DETERMINISTA
    geo.provincia_desde_coords = lambda lat, lon: {"provincia": "Madrid", "nombre": "Madrid"}
    bot_mod._conversaciones.clear()

    # ── Conversación simulada del bot ──
    async def conversacion():
        await bot_mod.procesar_update({"message": {"chat": {"id": CHAT_ID}, "text": "/start"}})
        await bot_mod.procesar_update(
            {"message": {"chat": {"id": CHAT_ID}, "text": "soy una persona de 45 años con diabetes"}}
        )
        bot_mod._conversaciones[CHAT_ID] = {"estado": bot_mod.Estado.UBICACION, "data": {}}
        await bot_mod.procesar_update(
            {
                "message": {
                    "chat": {"id": CHAT_ID},
                    "location": {"latitude": COORDENADAS[0], "longitude": COORDENADAS[1]},
                }
            }
        )
        await bot_mod.procesar_update(
            {
                "callback_query": {
                    "id": "1",
                    "message": {"chat": {"id": CHAT_ID}, "message_id": 1},
                    "data": "edit_edad",
                }
            }
        )

    asyncio.run(conversacion())

    # ── Predicción desde la web (mismo camino que /api/predict) ──
    import chat.app as web_app
    from chat.app import app as fastapi_app
    from fastapi.testclient import TestClient

    web_app._db = DBManager(PROYECTO / "data" / "climasafe.db")
    web_app._db.initialize()

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
                "perfil_horario": [{"hora": 8, "HI": 27.0, "temp": 28.0}],
            },
            "modelos": {},
        }

    import climasafeai.models.ensemble as ensemble_mod
    ensemble_mod.predict_ensemble = _fake_predict

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
    print(f"POST /api/predict → {resp.status_code}")

    # ── Grep sobre el tramo nuevo de los logs ──
    with open(log_file, "r", encoding="utf-8", errors="replace") as fh:
        fh.seek(offset_inicial)
        contenido = fh.read()
    print(f"\n=== grep sobre {log_file} (tramo nuevo: {len(contenido)} caracteres) ===")
    fallos = 0
    for patron, nombre in PATRONES_PII:
        coincidencias = re.findall(patron, contenido)
        estado = "OK  " if not coincidencias else "FALLO"
        if coincidencias:
            fallos += 1
        print(f"  [{estado}] {nombre:45s} {patron!r} → {coincidencias[:3] or 'sin coincidencias'}")
    print()

    if fallos:
        print(f"\nRESULTADO: {fallos} patrón(es) de PII encontrado(s) en los logs nuevos.")
        return 1
    print("\nRESULTADO: ningún patrón de PII en los logs nuevos. (criterio 2 OK)")
    return 0


if __name__ == "__main__":
    sys.exit(main())