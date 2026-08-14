"""
climasafeai.bot.evaluacion_programada — Cálculo y encolado de la evaluación programada.

Calcula la evaluación de riesgo (hoy / mañana / resumen diario) y la encola en
la cola de MSG-003 (`ColaMensajes`); el envío lo hace el worker de la cola con
el `MessageAdapter` de MSG-001. No hay contenedor: quién decide CUÁNDO se
lanza (CRON_SCHEDULE) es `programador_evaluaciones`, que corre este módulo
como productor y la cola como consumidor.

    from climasafeai.bot.cola_mensajes import ColaMensajes
    from climasafeai.bot.evaluacion_programada import crear_adapter, encolar_evaluacion

    cola = ColaMensajes(adapter=crear_adapter())
    await encolar_evaluacion(cola)   # el envío ocurre cuando la cola se procesa

Configuración (variables de entorno):

- CRON_TAREA   — ``resumen`` (por defecto, hoy+mañana), ``hoy`` o ``manana``.
- MSG_ADAPTER  — canal de salida: ``telegram`` (por defecto), ``hermes`` o
  ``webhook``. Las URLs de hermes/webhook salen de HERMES_BASE_URL /
  WEBHOOK_URL (igual que en ``messaging.py``); telegram usa
  TELEGRAM_BOT_TOKEN.
- MSG_DESTINO  — destinos separados por coma (chat_id de Telegram, user id de
  Hermes, destino del webhook). Obligatorio.
- PERFIL_ID    — opcional, id de un perfil de la DB. Aporta lat/lon/provincia y
  sus factores (edad, comorbilidades...). Sin PERFIL_ID se usan LAT, LON,
  PROVINCIA, EDAD y SEXO.

El patrón de cálculo es el del aviso diario del bot (``_enviar_aviso_diario``
en ``telegram_bot.py``): `predict_ensemble` para el día concreto y
`prediccion_semanal` para el resumen. No toca la web ni el bot: solo calcula
y encola.
"""

from __future__ import annotations

import asyncio
import logging
import os
from datetime import date, timedelta
from typing import Any

from climasafeai.bot.cola_mensajes import ColaMensajes
from climasafeai.bot.messaging import HermesAdapter, MessageAdapter, TelegramAdapter, WebhookAdapter
from climasafeai.db.manager import DBManager
from climasafeai.models.ensemble import predict_ensemble, prediccion_semanal
from climasafeai.models.recomendaciones import recomendacion_resumen

logger = logging.getLogger(__name__)

_db = DBManager()

_ADAPTERS: dict[str, type[MessageAdapter]] = {
    "telegram": TelegramAdapter,
    "hermes": HermesAdapter,
    "webhook": WebhookAdapter,
}

TAREAS = ("hoy", "manana", "resumen")

# Campos del perfil guardado en la DB que sí consume personalizar_riesgo;
# el resto (uid, created_at...) no pinta en la predicción.
_CAMPOS_PERFIL_PREDICCION = (
    "sexo",
    "edad",
    "aclimatado",
    "nivel_actividad",
    "entrenado",
    "deporte",
    "duracion_actividad_h",
    "hora_inicio",
    "porcentaje_grasa",
    "fototipo",
    "falta_sueno",
    "enfermedad_reciente",
    "fiesta",
    "ocupacion",
)


def crear_adapter(nombre: str | None = None) -> MessageAdapter:
    """Instancia el adaptador que pide MSG_ADAPTER (telegram por defecto)."""
    nombre = (nombre or os.getenv("MSG_ADAPTER", "telegram")).lower()
    try:
        return _ADAPTERS[nombre]()
    except KeyError:
        raise SystemExit(f"MSG_ADAPTER desconocido: {nombre} (valores: {', '.join(_ADAPTERS)})")


def inicializar_db() -> None:
    """Crea la DB de perfiles si no existe (idempotente, no toca los datos)."""
    _db.initialize()


def destinos_env() -> list[str]:
    """Destinos de MSG_DESTINO (separados por coma); aborta si no hay ninguno."""
    destinos = [d.strip() for d in os.getenv("MSG_DESTINO", "").split(",") if d.strip()]
    if not destinos:
        raise SystemExit("MSG_DESTINO es obligatorio (destinos separados por coma)")
    return destinos


def _prob_dominante(resultado: dict) -> float:
    """Probabilidad personalizada del canal que más riesgo marca (calor o frío)."""
    perfil_aplicado = resultado.get("perfil", {})
    probs = [
        perfil_aplicado.get(canal, {}).get("prob_personalizada") for canal in ("calor", "frio")
    ]
    probs = [p for p in probs if p is not None]
    return max(probs) if probs else 0.0


def _formatear_dia(titulo: str, resultado: dict) -> str:
    """'☀️ Riesgo hoy — PRECAUCION (35%)' + recomendación, como el aviso diario."""
    clase = resultado.get("clase_final_label", "?")
    prob = _prob_dominante(resultado)
    return f"☀️ *{titulo}*\nRiesgo: {clase} ({prob:.0%})\n→ {recomendacion_resumen(resultado)}"


def _formatear_resumen(serie: dict, provincia: str) -> str:
    """Resumen diario: riesgo de hoy y mañana con su banda conformal."""
    hoy = date.today().isoformat()
    lineas = [f"📋 *Resumen diario* ({provincia})"]
    for dia in serie.get("dias", []):
        prob = dia.get("prob")
        prob_txt = f"{prob:.0%}" if prob is not None else "n/d"
        etiqueta = "Hoy" if dia["fecha"] == hoy else "Mañana"
        lineas.append(
            f"• {etiqueta} ({dia['fecha']}): {dia.get('clase')} ({prob_txt}), "
            f"confianza {dia.get('confianza_conformal')}"
        )
    return "\n".join(lineas)


def _perfil_para_prediccion(perfil: dict) -> dict:
    """Perfil de la DB normalizado para predict_ensemble (arrays → sets)."""
    p: dict[str, Any] = {
        "sexo": perfil.get("sexo", "hombre"),
        "edad": perfil.get("edad"),
        "aclimatado": bool(perfil.get("aclimatado", False)),
        "nivel_actividad": "ligera",
        "comorbilidades": set(perfil.get("comorbilidades") or []),
        "farmacos": set(perfil.get("farmacos") or []),
    }
    for campo in _CAMPOS_PERFIL_PREDICCION:
        if perfil.get(campo) is not None:
            p[campo] = perfil[campo]
    if perfil.get("situacion_social"):
        p["situacion_social"] = set(perfil["situacion_social"])
    return p


def _cargar_ubicacion_y_perfil() -> tuple[dict, float, float, str]:
    """(perfil_predicción, lat, lon, provincia) desde PERFIL_ID o variables env."""
    perfil_ids = [pid.strip() for pid in os.getenv("PERFIL_ID", "").split(",") if pid.strip()]
    if perfil_ids:
        perfil_db = _db.obtener_perfil(int(perfil_ids[0]))
        if perfil_db is None:
            raise SystemExit(f"PERFIL_ID {perfil_ids[0]} no existe en la base de datos")
        lat = perfil_db.get("lat")
        lon = perfil_db.get("lon")
        if lat is None or lon is None:
            raise SystemExit(f"El perfil {perfil_ids[0]} no tiene lat/lon configurados")
        return (
            _perfil_para_prediccion(perfil_db),
            float(lat),
            float(lon),
            perfil_db.get("provincia") or "Madrid",
        )

    lat = os.getenv("LAT")
    lon = os.getenv("LON")
    if lat is None or lon is None:
        raise SystemExit("Falta PERFIL_ID o LAT/LON: no hay ubicación para evaluar")
    perfil = {
        "sexo": os.getenv("SEXO", "hombre"),
        "nivel_actividad": "ligera",
        "comorbilidades": set(),
        "farmacos": set(),
    }
    if os.getenv("EDAD"):
        perfil["edad"] = int(os.getenv("EDAD", ""))
    return perfil, float(lat), float(lon), os.getenv("PROVINCIA", "Madrid")


async def _calcular_texto(tarea: str, lat: float, lon: float, provincia: str, perfil: dict) -> str:
    """Calcula el texto de la evaluación pedida (sin red: la hace el modelo)."""
    if tarea == "resumen":
        serie = await asyncio.to_thread(
            prediccion_semanal, lat=lat, lon=lon, provincia=provincia, perfil=perfil, dias=2
        )
        return _formatear_resumen(serie, provincia)
    titulo = "Riesgo hoy" if tarea == "hoy" else "Riesgo mañana"
    target_date = date.today() if tarea == "hoy" else date.today() + timedelta(days=1)
    resultado = await asyncio.to_thread(
        predict_ensemble,
        lat=lat,
        lon=lon,
        provincia=provincia,
        perfil=perfil,
        target_date=target_date,
    )
    return _formatear_dia(titulo, resultado)


async def encolar_evaluacion(cola: ColaMensajes, tarea: str | None = None) -> int:
    """Calcula la evaluación pedida y la encola en `cola` para cada destino.

    Devuelve cuántos mensajes se encolaron. El envío lo hace el worker de la
    cola (MSG-003) con su `MessageAdapter`; aquí solo se produce el mensaje.
    """
    tarea = (tarea or os.getenv("CRON_TAREA", "resumen")).lower()
    if tarea not in TAREAS:
        raise SystemExit(f"CRON_TAREA desconocida: {tarea} (valores: {', '.join(TAREAS)})")

    destinos = destinos_env()
    perfil, lat, lon, provincia = _cargar_ubicacion_y_perfil()
    texto = await _calcular_texto(tarea, lat, lon, provincia, perfil)
    logger.info("Evaluación %s (%s) encolada para %s", tarea, provincia, destinos)
    return cola.encolar_lote([(destino, texto) for destino in destinos])
