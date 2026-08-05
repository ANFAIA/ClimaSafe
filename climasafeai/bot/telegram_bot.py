"""
climasafeai.bot.telegram_bot — Bot de Telegram determinista.

Reemplaza spacebot para la recogida de formulario (10 campos) usando botones
nativos de Telegram. El LLM solo se usa para redactar la respuesta final;
si falla o no hay API key, se usa una plantilla de texto.

Uso:
    uv run python -m climasafeai.bot.telegram_bot
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import signal
import sys
import textwrap
from datetime import date, datetime
from enum import Enum, auto

from climasafeai.bot.geocoding import buscar_lugar, provincia_desde_coords
from climasafeai.features.personalizacion import _OCUPACION_NIVELES, nivel_actividad_de_deporte
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any

import httpx

from climasafeai.db.manager import DBManager
from climasafeai.llm.rag_qwen import (
    LLMConfig,
    ask_con_perfil,
    ask_with_rag,
    check_ollama,
    lineas_parte,
)
from climasafeai.models.ensemble import predict_ensemble
from climasafeai.models.recomendaciones import recomendacion_resumen

logger = logging.getLogger(__name__)

# ── Constantes ─────────────────────────────────────────────────────────────


def _telegram_api() -> str:
    """Lazy: TELEGRAM_API no se evalúa hasta la primera llamada."""
    token = os.getenv("TELEGRAM_BOT_TOKEN", "")
    return f"https://api.telegram.org/bot{token}"
ACTIVIDADES = ["reposo", "ligera", "moderada", "intensa", "muy_intensa"]

# Solo estos dos fármacos tienen coeficiente en la base de conocimiento
# (`_factores_implementados("calor", "farmacos")`): diuréticos de asa x1.3 y
# antipsicóticos x1.8. Preguntar por otros sería recoger un dato que nadie usa.
MEDICACION_OPTS = ["diureticos_asa", "antipsicoticos"]
MEDICACION_LABELS = {
    "diureticos_asa": "Diuréticos (furosemida…)",
    "antipsicoticos": "Antipsicóticos",
}

# La ocupación describe ESTA salida, no el oficio: quien trabaja en el campo pero
# sale a pasear el domingo no lleva ocupación.
# Orden de intensidad ascendente (igual que `_OCUPACION_NIVELES`): se muestra así
# en los botones del cuestionario de tipo de trabajo.
OCUPACIONES = {
    "oficina": "Oficina / interior",
    "reparto": "Reparto / conducción",
    "mantenimiento": "Mantenimiento / jardinería",
    "construccion": "Construcción",
    "campo": "Campo / agricultura",
}

# Deportes predefinidos (mismos que la web + más). Se muestran como botones inline
# en lugar de texto libre. "otro" permite escribir un deporte no listado.
# Solo deportes con MET medido en el Compendium of Physical Activities: elegirlos
# fija la intensidad con un número publicado en vez de dejar que el usuario adivine
# si su partido de fútbol es "moderada" o "intensa". El pádel y la natación no
# aparecen con MET utilizable, así que no se ofrecen: mejor no darlos que
# inventarles un valor. El resto se escribe a mano con "Otro" y entonces la
# intensidad la elige el usuario, como antes.
DEPORTES: dict[str, str] = {
    "pasear": "Pasear",
    "caminar": "Caminar",
    "senderismo": "Senderismo",
    "trekking_mochila": "Trekking con mochila",
    "correr_suave": "Trotar",
    "correr": "Correr",
    "ciclismo_suave": "Bici tranquila",
    "ciclismo": "Bici",
    "ciclismo_fuerte": "Bici fuerte",
    "btt": "BTT / montaña",
    "futbol": "Futbol",
    "futbol_competicion": "Futbol de competicion",
    "tenis_dobles": "Tenis dobles",
    "tenis": "Tenis individual",
}

# Nombres de rutina que son entrenamiento físico genérico, no una salida laboral:
# no tienen MET propio en el Compendium (por eso no están en DEPORTES), pero se
# guardan directo como hoy, sin el cuestionario de tipo de trabajo (BOT-015).
_NOMBRES_ENTRENAMIENTO = {"entreno", "entrenamiento"}

# Cómo llega el usuario a la salida. Son los tres factores situacionales que el
# modelo sí sabe puntuar; el de fiesta es de los que más pesan de todo el sistema.
ESTADO_PREVIO_OPTS = {
    "falta_sueno": "Dormi poco",
    "fiesta": "Fiesta o alcohol reciente",
    "enfermedad_reciente": "Enfermedad reciente",
}
COMORBILIDADES_OPTS = [
    "cardiovascular", "diabetes", "respiratoria", "renal",
    "obesidad", "salud_mental",
]
COMORBILIDADES_LABELS: dict[str, str] = {
    "cardiovascular": "Cardiovascular",
    "diabetes": "Diabetes",
    "respiratoria": "Respiratoria",
    "renal": "Renal",
    "obesidad": "Obesidad",
    "salud_mental": "Salud mental",
}

# Escala Fitzpatrick para fototipo (factor UV, solo actúa con UV>3).
# Se pregunta como opcional (también con 'saltar').
FOTOTIPO_OPTS = {
    "1": "Tipo I — siempre me quemo, nunca me bronceo",
    "2": "Tipo II — siempre me quemo, a veces me bronceo",
    "3": "Tipo III — a veces me quemo, siempre me bronceo",
    "4": "Tipo IV — nunca me quemo, siempre me bronceo",
    "5": "Tipo V — piel naturalmente oscura",
    "6": "Tipo VI — piel muy oscura",
}

# Situación social: factores de aislamiento/vivienda que el modelo sí puntúa
# (vive_solo ×1.5, no_sale ×2.0, sin_aire_acondicionado ×2.5).
# Son condiciones PERMANENTES o de la vivienda, no del día concreto.
SITUACION_SOCIAL_OPTS = {
    "vive_solo": "Vivo solo",
    "no_sale": "Casi no salgo de casa",
    "sin_aire_acondicionado": "Sin aire acondicionado en casa",
    "vivienda_fria": "Vivienda mal aislada / sin calefaccion",
}

# ── BOT-007: edición de perfil y rutinas ────────────────────────────────────

# Campos que /perfil permite editar: alias de campo → columna real de la BD.
# OJO con los nombres reales: `porcentaje_grasa` y `farmacos`, no `grasa` ni
# `medicacion` (ya costó dos veces en este proyecto).
_CAMPOS_EDITABLES = {
    "edad",
    "sexo",
    "grasa",
    "fototipo",
    "aclimatado",
    "comorbilidades",
    "medicacion",
    "situacion_social",
}
_CAMPOS_ARRAY = {"comorbilidades", "medicacion", "situacion_social"}
_CAMPO_A_COLUMNA = {
    "edad": "edad",
    "sexo": "sexo",
    "grasa": "porcentaje_grasa",
    "fototipo": "fototipo",
    "aclimatado": "aclimatado",
    "comorbilidades": "comorbilidades",
    "medicacion": "farmacos",
    "situacion_social": "situacion_social",
}
_CAMPO_LABEL = {
    "edad": "Edad",
    "sexo": "Sexo",
    "grasa": "% grasa",
    "fototipo": "Fototipo",
    "aclimatado": "Aclimatado",
    "comorbilidades": "Comorbilidades",
    "medicacion": "Medicación",
    "situacion_social": "Situación social",
}
_PREGUNTAS_EDIT = {
    "edad": "¿Cuál es tu fecha de nacimiento? (DD/MM/AAAA, ej: 15/03/1990)",
    "sexo": "¿Nuevo sexo? Elige abajo:",
    "grasa": "¿Nuevo % de grasa corporal? (ej: 20.5, o 'saltar' para dejarlo vacío):",
    "fototipo": "¿Nuevo fototipo? Elige abajo:",
    "aclimatado": "¿Estás aclimatado al calor?",
    "comorbilidades": "¿Qué comorbilidades tienes? Pulsa las que apliquen y luego 'Terminé':",
    "medicacion": "¿Qué medicación tomas? Pulsa la que aplique y luego 'Terminé':",
    "situacion_social": "¿Cuál es tu situación social? Pulsa lo que aplique y luego 'Terminé':",
}

# Días de la semana en formato 1-7 (1=lunes) para rutinas.
_NOMBRE_DIAS = {1: "L", 2: "M", 3: "X", 4: "J", 5: "V", 6: "S", 7: "D"}
_DIA_NUM = {v: k for k, v in _NOMBRE_DIAS.items()}

AVISO_SIN_PERFIL = (
    "🌡️ *Aviso diario*\n"
    "Todavía no tienes un perfil guardado. Usa /start para crear uno y "
    "configura tus rutinas con /rutinas, así podré avisarte del riesgo cada día."
)
AVISO_SIN_UBICACION = (
    "🌡️ *Aviso diario*\n"
    "Tu perfil no tiene ubicación guardada. Usa /start y comparte tu ubicación "
    "para poder calcular tu riesgo, y añade rutinas con /rutinas."
)


# Strings de modelo LiteLLM. El usuario puede escribir cualquier modelo
# soportado por LiteLLM: "ollama/qwen2.5:1.5b", "groq/llama-3.3-70b-versatile", etc.
MODELO_LOCAL = "ollama/qwen2.5:1.5b"
MODELO_API = "groq/llama-3.3-70b-versatile"
MODELO_DETERMINISTA = "__determinista__"  # valor centinela: sin LLM


def _modelo_por_defecto() -> str:
    """Auto-detecta el mejor modelo local; si no hay Ollama, determinista.

    Antes devolvía `MODELO_LOCAL` fijo (el 1.5B) aunque hubiera un 7B instalado, y
    la diferencia se nota: el 1.5B contesta con titulares en negrita y viñetas —
    indistinguible de la plantilla— mientras el 7B da el parte de una línea.
    `check_ollama()` ya calcula cuál es el mejor, así que se usa.
    """
    st = check_ollama()
    if st.get("available"):
        return st.get("best_model") or MODELO_LOCAL
    return MODELO_DETERMINISTA


class Estado(Enum):
    IDLE = auto()
    SEXO = auto()
    EDAD = auto()
    GRASA = auto()
    FOTOTIPO = auto()
    ACLIMATADO = auto()
    ACTIVIDAD = auto()
    ENTRENADO = auto()
    DURACION = auto()
    HORA_INICIO = auto()
    TRABAJO = auto()
    TIPO_TRABAJO = auto()
    DEPORTE = auto()
    COMORBILIDADES = auto()
    MEDICACION = auto()
    ESTADO_PREVIO = auto()
    SITUACION_SOCIAL = auto()
    UBICACION = auto()
    GUARDAR_PERFIL = auto()
    DONE = auto()


# Estados de opción múltiple: se acumulan clics hasta que el usuario pulsa
# "Terminé". El resto avanza con el primer clic.
MULTISELECCION = {Estado.COMORBILIDADES, Estado.MEDICACION, Estado.ESTADO_PREVIO, Estado.SITUACION_SOCIAL}

# Mapas de callback_data → nombre visible para los toasts de toggle
_COMORB_TOAST = {k: v for k, v in COMORBILIDADES_LABELS.items()}
_MED_TOAST = {k: v for k, v in MEDICACION_LABELS.items()}
_PREVIO_TOAST = {k: v for k, v in ESTADO_PREVIO_OPTS.items()}
_SOCIAL_TOAST = {k: v for k, v in SITUACION_SOCIAL_OPTS.items()}
_TOAST_POR_ESTADO: dict[Estado, dict[str, str]] = {
    Estado.COMORBILIDADES: _COMORB_TOAST,
    Estado.MEDICACION: _MED_TOAST,
    Estado.ESTADO_PREVIO: _PREVIO_TOAST,
    Estado.SITUACION_SOCIAL: _SOCIAL_TOAST,
}


def _nombre_opcion(estado: Estado, callback_data: str) -> str:
    """Nombre visible de una opción de multiselect para el toast."""
    mapa = _TOAST_POR_ESTADO.get(estado, {})
    return mapa.get(callback_data, callback_data.replace("_", " ").title())


# ── Ayudantes Telegram ─────────────────────────────────────────────────────

_HTTP_CLIENT: httpx.AsyncClient | None = None


async def _tg(method: str, **kwargs: Any) -> dict:
    """Llama a la Bot API de Telegram y devuelve el JSON.

    Reutiliza un mismo cliente HTTP (conexión keep-alive) en vez de crear
    uno nuevo por llamada.
    """
    global _HTTP_CLIENT
    if _HTTP_CLIENT is None:
        _HTTP_CLIENT = httpx.AsyncClient(timeout=20)
    url = f"{_telegram_api()}/{method}"
    r = await _HTTP_CLIENT.post(url, json=kwargs)
    if r.status_code == 429:
        retry_after = int(r.headers.get("retry-after", 5))
        logger.warning("Rate limited (429), esperando %ds", retry_after)
        await asyncio.sleep(retry_after)
        r = await _HTTP_CLIENT.post(url, json=kwargs)
    r.raise_for_status()
    return r.json()


def _kb_yesno() -> list[list[dict]]:
    return [[{"text": "Si", "callback_data": "si"},
             {"text": "No", "callback_data": "no"}]]


def _kb_sexo() -> list[list[dict]]:
    return [[{"text": "Hombre", "callback_data": "hombre"},
             {"text": "Mujer", "callback_data": "mujer"}]]


def _kb_actividad() -> list[list[dict]]:
    return [[{"text": a.capitalize(), "callback_data": a}] for a in ACTIVIDADES]


def _kb_trabajo() -> list[list[dict]]:
    return [
        [{"text": "Por trabajo", "callback_data": "trabajo"}],
        [{"text": "Por mi cuenta (ocio o deporte)", "callback_data": "propia"}],
    ]


def _kb_tipo_trabajo(prefijo: str = "") -> list[list[dict]]:
    """Teclado de tipo de trabajo.

    En /start el callback es la clave pura (``"campo"``); en el cuestionario de
    rutina (BOT-015) se pasa ``prefijo="rutina_tipo_"`` para distinguir ese
    callback del flujo normal y del chat RAG posterior.
    """
    return [[{"text": v, "callback_data": f"{prefijo}{k}"}] for k, v in OCUPACIONES.items()]


def _kb_estado_previo() -> list[list[dict]]:
    kb = [[{"text": v, "callback_data": k}] for k, v in ESTADO_PREVIO_OPTS.items()]
    kb.append([{"text": "Nada de eso / Termine", "callback_data": "__done__"}])
    return kb


def _kb_medicacion() -> list[list[dict]]:
    kb = [[{"text": MEDICACION_LABELS[m], "callback_data": m}] for m in MEDICACION_OPTS]
    kb.append([{"text": "Ninguna / Termine", "callback_data": "__done__"}])
    return kb


def _kb_comorbilidades() -> list[list[dict]]:
    kb = [[{"text": c.replace("_", " ").title(), "callback_data": c}] for c in COMORBILIDADES_OPTS]
    kb.append([{"text": "Ninguna / Termine", "callback_data": "__done__"}])
    return kb


def _kb_fototipo() -> list[list[dict]]:
    return [[{"text": f"{k} — {v.split('—')[1].strip()}", "callback_data": k}] for k, v in FOTOTIPO_OPTS.items()]


def _kb_situacion_social() -> list[list[dict]]:
    kb = [[{"text": v, "callback_data": k}] for k, v in SITUACION_SOCIAL_OPTS.items()]
    kb.append([{"text": "Nada de eso / Termine", "callback_data": "__done__"}])
    return kb


def _kb_deporte() -> list[list[dict]]:
    items = list(DEPORTES.items())
    kb = [[{"text": v, "callback_data": k} for k, v in items[i:i + 2]]
          for i in range(0, len(items), 2)]
    kb.append([{"text": "Otro (lo escribo)", "callback_data": "__otro__"}])
    kb.append([{"text": "Saltar", "callback_data": "__saltar__"}])
    return kb


def _kb_modelos(modelo_actual: str) -> list[list[dict]]:
    """Teclado para elegir modelo LLM."""
    opciones = [
        ("Qwen 2.5 local (CPU)", MODELO_LOCAL),
        ("API externa (Groq)", MODELO_API),
        ("Determinista (sin LLM)", MODELO_DETERMINISTA),
    ]
    return [[{"text": f"{'→ ' if m == modelo_actual else '  '}{t}",
              "callback_data": f"modelo_{m}"}] for t, m in opciones]


def _kb_ubicacion() -> dict:
    """Teclado de respuesta con el botón nativo de compartir ubicación.

    No es un teclado inline: `request_location` solo existe en los ReplyKeyboard.
    Telegram devuelve lat/lon exactas, sin ambigüedad de "¿qué Aldán?".
    """
    return {
        "keyboard": [[{"text": "Enviar mi ubicacion", "request_location": True}]],
        "resize_keyboard": True,
        "one_time_keyboard": True,
    }


FIELD_LABELS: dict[Estado, tuple[str, str, Any]] = {
    Estado.SEXO: ("¿Cuál es tu sexo?", "sexo", _kb_sexo()),
    Estado.EDAD: (
        "¿Cuál es tu fecha de nacimiento? (DD/MM/AAAA, ej: 15/03/1990)",
        "fecha_nacimiento",
        None,
    ),
    Estado.GRASA: (
        "¿Sabes tu % de grasa corporal? Escríbelo, o 'saltar' si no lo sabes.\n"
        "_(es lo único que el modelo usa de tu constitución física)_",
        "porcentaje_grasa",
        None,
    ),
    Estado.FOTOTIPO: (
        "¿Cuál es tu fototipo de piel? (solo importa si el sol da fuerte, "
        "o escribe 'saltar' para saltarlo)\n"
        "1 → Siempre me quemo\n2 → Siempre me quemo, a veces bronceo\n"
        "3 → A veces me quemo, siempre bronceo\n4 → Nunca me quemo\n"
        "5 → Piel oscura\n6 → Piel muy oscura",
        "fototipo",
        _kb_fototipo(),
    ),
    Estado.ACLIMATADO: (
        "¿Estás aclimatado al calor? (vives en clima cálido o llevas semanas de calor — "
        "hacer deporte no cuenta)",
        "aclimatado",
        _kb_yesno(),
    ),
    Estado.ACTIVIDAD: ("¿Qué intensidad tendrá la actividad?", "nivel_actividad", _kb_actividad()),
    Estado.ENTRENADO: (
        "¿Estás acostumbrado a esa actividad en concreto?",
        "entrenado",
        _kb_yesno(),
    ),
    Estado.DURACION: ("¿Cuántas horas durará? (ej: 2, 3.5)", "duracion_h", None),
    Estado.HORA_INICIO: ("¿A qué hora empiezas? (ej: 8, 14, 10:30)", "hora_inicio", None),
    Estado.TRABAJO: ("¿Sales por trabajo o por tu cuenta?", "_por_trabajo", _kb_trabajo()),
    Estado.TIPO_TRABAJO: ("¿Qué tipo de trabajo?", "ocupacion", _kb_tipo_trabajo()),
    Estado.DEPORTE: (
        "¿Qué deporte o actividad vas a hacer?",
        "deporte",
        _kb_deporte(),
    ),
    Estado.COMORBILIDADES: (
        "¿Tienes alguna de estas condiciones? (pulsa todas las que apliquen, "
        "luego 'Ninguna / Terminé')",
        "comorbilidades",
        _kb_comorbilidades(),
    ),
    Estado.MEDICACION: (
        "¿Tomas alguno de estos? (pulsa los que apliquen, luego 'Ninguna / Terminé')",
        "farmacos",
        _kb_medicacion(),
    ),
    Estado.ESTADO_PREVIO: (
        "¿Cómo llegas a la salida? (pulsa lo que aplique, luego 'Terminé')",
        "estado_previo",
        _kb_estado_previo(),
    ),
    Estado.SITUACION_SOCIAL: (
        "¿Cómo es tu situación habitual? (pulsa lo que aplique, luego 'Terminé')",
        "situacion_social",
        _kb_situacion_social(),
    ),
    Estado.UBICACION: (
        "¿Dónde vas a estar? Pulsa el botón para mandarme tu ubicación, "
        "o escribe el nombre del sitio (ej: Aldán).",
        "ubicacion",
        None,  # el teclado se monta aparte: es un ReplyKeyboard, no inline
    ),
    Estado.GUARDAR_PERFIL: (
        "Guardar perfil",
        "_guardar_perfil",
        None,  # se monta aparte
    ),
}


# ── Formateo de respuesta ─────────────────────────────────────────────────

# BOT-013: la cabecera y las frases de riesgo salen de `lineas_parte`, las
# mismas que se le dan al LLM. Antes esta plantilla ponía "Riesgo PRECAUCIÓN
# (19%)" y el 19% parecía explicar la clase cuando ni siquiera sale de ahí.
RESPONSE_TEMPLATE = textwrap.dedent("""\
    {cabecera}

    {explicacion}
    🌡️ Temperatura prevista: {temp:.1f} °C
    ☀️ Índice UV (media): {uv}
    ❗ Recomendación: {recomendacion}
""")


def _temps_en_ventana(perfil_horario: list[dict], perfil_usuario: dict) -> list[float]:
    """Temperaturas previstas en las horas en las que el usuario estará fuera."""
    inicio = perfil_usuario.get("hora_inicio")
    duracion = perfil_usuario.get("duracion_actividad_h")
    if inicio is not None and duracion is not None:
        en_ventana = [
            h["temp"] for h in perfil_horario
            if inicio <= h["hora"] < inicio + duracion and h.get("temp") is not None
        ]
        if en_ventana:
            return en_ventana
    return [h["temp"] for h in perfil_horario if h.get("temp") is not None]


def _format_uv(uv) -> str:
    """Índice UV legible: 6 en vez de 6.0; n/d si no hay dato."""
    if uv is None:
        return "n/d"
    return f"{uv:.1f}".rstrip("0").rstrip(".")


def _format_template(result: dict, lugar: str | None = None) -> str:
    """Respuesta sin LLM: plantilla fija con el parte completo.

    Incluye la ubicación del usuario, el nivel de riesgo con su porcentaje, la
    temperatura prevista (media en las horas de actividad), el índice UV medio
    y una recomendación adaptada al contexto (frío/calor/UV), no solo la clase.
    """
    w = result.get("weather", {})
    cur = w.get("current", {})
    perfil_h = w.get("perfil_horario") or []
    perfil_u = result.get("perfil_usuario") or {}
    temps = _temps_en_ventana(perfil_h, perfil_u)
    temp = round(sum(temps) / len(temps), 1) if temps else (cur.get("t2m_c") or 0)
    uv = w.get("uv_index")
    if uv is None:
        uv = cur.get("uv_index")

    cabecera, *explicacion = lineas_parte(result, lugar)
    return RESPONSE_TEMPLATE.format(
        cabecera=cabecera,
        explicacion="\n".join(f"🔎 {linea}" for linea in explicacion),
        temp=temp,
        uv=_format_uv(uv),
        recomendacion=recomendacion_resumen(result),
    )


# ── Lógica del bot (stateless por conversación) ───────────────────────────

# Lo primero que ve el usuario, y lo que devuelve /help. Solo los comandos que
# necesita: los de modelo (/model, /qwen, /api, /determinista) son de
# depuración. Desde CHAT-003 no hay /chat: /start con botones es el único
# camino, y tras el parte (con LLM activo) el chat queda abierto para dudas.
BIENVENIDA = (
    "Soy *ClimaSafeAI*. Te calculo el riesgo de una salida:\n\n"
    "*/start* — Cuestionario con botones: te pregunto por ti y por el día y "
    "te calculo el riesgo por calor.\n"
    "Tras el parte puedes quedarte preguntándome las dudas que te queden "
    "(p. ej. qué es SPF), y volver a /start para otra salida.\n\n"
    "Tus datos y rutinas:\n"
    "*/perfil* — Ver y editar tus datos personales guardados (edad, sexo, "
    "% grasa, fototipo, aclimatación, comorbilidades, medicación, situación social).\n"
    "*/rutinas* — Ver tus rutinas semanales.\n"
    "*/rutinas_anadir L-V trabajo 8-16* — Añadir una rutina (días, nombre, "
    "hora inicio-fin). Ej: `L-V entreno 18-20`.\n"
    "*/avisos HH:MM* — Hora del aviso diario del pronóstico según tus rutinas "
    "(`/avisos off` lo desactiva).\n\n"
    "Vuelve a ver esto con /help."
)

CHAT_CIERRE = (
    "¿Te queda alguna duda? Pregúntamela (p. ej. qué es SPF) y te la resuelvo. "
    "Para calcular otra salida, escribe /start."
)

# Estado en memoria: {chat_id: {"estado": Estado, "data": dict, ...}}
_conversaciones: dict[int, dict[str, Any]] = {}

# Base de datos de perfiles (SQLite local)
_db = DBManager()


def _perfil_a_data(perfil: dict) -> dict:
    """Convierte un perfil de la BD al formato `conv["data"]`."""
    data: dict[str, Any] = {"_perfil_cargado": True}
    if perfil.get("sexo"):
        data["sexo"] = perfil["sexo"]
    if perfil.get("edad"):
        data["edad"] = perfil["edad"]
    if perfil.get("porcentaje_grasa") is not None:
        data["porcentaje_grasa"] = perfil["porcentaje_grasa"]
    if perfil.get("fototipo"):
        data["fototipo"] = str(perfil["fototipo"])
    if perfil.get("aclimatado") is not None:
        data["aclimatado"] = bool(perfil["aclimatado"])
    # Siempre guardar estos campos aunque vacíos, para que _saltar_si_prellenado
    # los detecte y salte los estados correspondientes.
    data["comorbilidades"] = set(perfil.get("comorbilidades") or [])
    data["farmacos"] = set(perfil.get("farmacos") or [])
    data["situacion_social"] = set(perfil.get("situacion_social") or [])
    return data


def _data_a_perfil(data: dict, alias: str, chat_id: str) -> dict:
    """Convierte `conv["data"]` al formato que espera DBManager.crear_perfil()."""
    p: dict[str, Any] = {"alias": alias, "telegram_chat_id": chat_id}
    if data.get("sexo"):
        p["sexo"] = data["sexo"]
    if data.get("edad"):
        p["edad"] = data["edad"]
    if data.get("porcentaje_grasa") is not None:
        p["porcentaje_grasa"] = data["porcentaje_grasa"]
    if data.get("fototipo"):
        p["fototipo"] = int(data["fototipo"])
    if data.get("aclimatado") is not None:
        p["aclimatado"] = data["aclimatado"]
    if data.get("comorbilidades"):
        p["comorbilidades"] = list(data["comorbilidades"])
    if data.get("farmacos"):
        p["farmacos"] = list(data["farmacos"])
    if data.get("situacion_social"):
        p["situacion_social"] = list(data["situacion_social"])
    return p


# ── Fecha de nacimiento → edad (BOT-010) ──────────────────────────────────

# La BD guarda `edad` (int); la fecha de nacimiento solo se usa para calcular
# esa edad en el momento de pedirla. Formatos aceptados: 'DD/MM/AAAA',
# 'DD-MM-AAAA' o solo el año ('1965', interpretado como 1 de enero).
FORMATO_FECHA_MSG = (
    "No entendí la fecha. Escríbela como DD/MM/AAAA (ej: 15/03/1990), "
    "con guiones (15-03-1990) o solo el año (1965)."
)
RANGO_FECHA_MSG = "El año debe estar entre 1900 y el año actual."
FUTURA_FECHA_MSG = "Esa fecha es de futuro. Escríbela como DD/MM/AAAA (ej: 15/03/1990)."


def _edad_desde_fecha_nacimiento(fecha: date, hoy: date | None = None) -> int:
    """Edad exacta a día de hoy: resta 1 si aún no ha cumplido años este año.

    El 29 de febrero solo cuenta el día exacto en años bisiestos; en un año no
    bisiesto aún no ha cumplido hasta el 1 de marzo.
    """
    hoy = hoy or date.today()
    edad = hoy.year - fecha.year
    if (hoy.month, hoy.day) < (fecha.month, fecha.day):
        edad -= 1
    return edad


def _parsear_fecha_nacimiento(texto: str, hoy: date | None = None) -> tuple[bool, int | str]:
    """Valida una fecha de nacimiento y devuelve (ok, edad) o (False, mensaje).

    No puede ser futura y el año va de 1900 a hoy. Un formato malo devuelve un
    mensaje claro, nunca una excepción.
    """
    hoy = hoy or date.today()
    t = (texto or "").strip()
    if not t:
        return False, FORMATO_FECHA_MSG
    try:
        if re.fullmatch(r"\d{4}", t):
            anio = int(t)
            if anio > hoy.year:
                return False, FUTURA_FECHA_MSG
            if anio < 1900:
                return False, RANGO_FECHA_MSG
            fecha = date(anio, 1, 1)  # solo el año → 1 de enero
        else:
            if "/" in t:
                partes = t.split("/")
            elif "-" in t:
                partes = t.split("-")
            else:
                return False, FORMATO_FECHA_MSG
            if len(partes) != 3:
                return False, FORMATO_FECHA_MSG
            dia, mes, anio = (int(p) for p in partes)
            if anio > hoy.year:
                return False, FUTURA_FECHA_MSG
            if anio < 1900:
                return False, RANGO_FECHA_MSG
            fecha = date(anio, mes, dia)  # ValueError en 31/02, 15/13…
    except ValueError:
        return False, FORMATO_FECHA_MSG
    if fecha > hoy:
        return False, FUTURA_FECHA_MSG
    return True, _edad_desde_fecha_nacimiento(fecha, hoy)


# ── BOT-007: resumen de perfil y edición ───────────────────────────────────


def _resumen_perfil(perfil: dict) -> str:
    """Texto legible con los datos actuales del perfil."""
    lineas = ["*Tu perfil:*"]
    if perfil.get("alias"):
        lineas.append(f"  Alias: {perfil['alias']}")
    lineas.append(f"  Edad: {perfil.get('edad') or '—'}")
    lineas.append(f"  Sexo: {perfil.get('sexo') or '—'}")
    grasa = perfil.get("porcentaje_grasa")
    lineas.append(f"  % grasa: {grasa if grasa is not None else '—'}")
    lineas.append(f"  Fototipo: {perfil.get('fototipo') or '—'}")
    acl = perfil.get("aclimatado")
    lineas.append("  Aclimatado: " + ("sí" if acl is True else ("no" if acl is False else "—")))
    lineas.append(f"  Comorbilidades: {', '.join(perfil.get('comorbilidades') or []) or 'ninguna'}")
    lineas.append(f"  Medicación: {', '.join(perfil.get('farmacos') or []) or 'ninguna'}")
    lineas.append(f"  Situación social: {', '.join(perfil.get('situacion_social') or []) or '—'}")
    if perfil.get("lat") is not None:
        lineas.append(
            f"  Ubicación: {perfil.get('provincia') or f'{perfil["lat"]:.2f}, {perfil["lon"]:.2f}'}"
        )
    lineas.append("\nPulsa un campo para editarlo:")
    return "\n".join(lineas)


def _kb_edicion_perfil() -> list[list[dict]]:
    """Teclado inline de /perfil: un botón por campo editable."""
    return [
        [{"text": "Edad", "callback_data": "edit_edad"}],
        [{"text": "Sexo", "callback_data": "edit_sexo"}],
        [{"text": "% grasa", "callback_data": "edit_grasa"}],
        [{"text": "Fototipo", "callback_data": "edit_fototipo"}],
        [{"text": "Aclimatado", "callback_data": "edit_aclimatado"}],
        [{"text": "Comorbilidades", "callback_data": "edit_comorbilidades"}],
        [{"text": "Medicación", "callback_data": "edit_medicacion"}],
        [{"text": "Situación social", "callback_data": "edit_situacion_social"}],
    ]


def _kb_edit_campo(campo: str) -> list[list[dict]] | None:
    """Teclado del mini-formulario de edición; None si el campo es texto libre."""
    if campo == "sexo":
        return _kb_sexo()
    if campo == "aclimatado":
        return _kb_yesno()
    if campo == "fototipo":
        return _kb_fototipo()
    if campo == "comorbilidades":
        return _kb_comorbilidades()
    if campo == "medicacion":
        return _kb_medicacion()
    if campo == "situacion_social":
        return _kb_situacion_social()
    return None


def _guardar_campo_valor(chat_id: int, campo: str, valor) -> tuple[bool, str]:
    """Guarda un campo del perfil del chat. Devuelve (ok, mensaje)."""
    match = _db.buscar_por_telegram(str(chat_id))
    if not match:
        return False, "No hay perfil guardado. Usa /start para crear uno."
    try:
        _db.actualizar_perfil(match["id"], {_CAMPO_A_COLUMNA[campo]: valor})
    except Exception:
        logger.exception("Error al guardar %s del perfil de %s", campo, chat_id)
        return False, "No se pudo guardar el cambio."
    return True, f"✓ {_CAMPO_LABEL[campo]} actualizado."


def _parsear_y_guardar_campo(chat_id: int, campo: str, texto: str) -> tuple[bool, str]:
    """Valida la respuesta de texto a una edición y la guarda.

    Devuelve (ok, mensaje). Los campos de opción también aceptan texto aquí
    por si el usuario escribe en vez de pulsar el botón.
    """
    valor = None
    try:
        if campo == "edad":
            # Pide fecha de nacimiento y guarda la edad calculada (BOT-010);
            # la columna de la BD sigue siendo `edad` (int), no cambia el esquema.
            ok, edad = _parsear_fecha_nacimiento(texto)
            if not ok:
                return False, edad  # edad contiene el mensaje de error
            valor = edad
        elif campo == "grasa":
            v = float(texto.replace("%", "").replace(",", ".").strip())
            if not (3 <= v <= 65):
                return False, "El % de grasa debe estar entre 3 y 65."
            valor = v
        elif campo == "fototipo":
            v = int(texto.strip())
            if not (1 <= v <= 6):
                return False, "El fototipo debe estar entre 1 y 6."
            valor = v
        elif campo == "aclimatado":
            t = texto.strip().lower()
            if t in ("si", "sí", "s"):
                valor = True
            elif t in ("no", "n"):
                valor = False
            else:
                return False, "Responde 'si' o 'no'."
        elif campo == "sexo":
            t = texto.strip().lower()
            if t not in ("hombre", "mujer"):
                return False, "Escribe 'hombre' o 'mujer'."
            valor = t
        elif campo in _CAMPOS_ARRAY:
            t = texto.strip().lower()
            if t in ("ninguna", "ninguno", "nada", "-"):
                valor = []
            else:
                valor = [p.strip() for p in t.split(",") if p.strip()]
    except (ValueError, TypeError):
        return False, "Valor inválido. Escribe un número o 'cancelar'."
    return _guardar_campo_valor(chat_id, campo, valor)


# ── BOT-007: rutinas semanales ─────────────────────────────────────────────


def _formato_hora(h: float) -> str:
    """8.0 → '8:00', 18.5 → '18:30'."""
    hh = int(h)
    mm = int(round((h - hh) * 60))
    return f"{hh}:{mm:02d}"


def _hora_a_float(texto: str) -> float:
    """'8' → 8.0, '8:30' → 8.5, '16:00' → 16.0."""
    t = texto.strip()
    if ":" in t:
        hh, mm = t.split(":", 1)
        return int(hh) + int(mm) / 60
    return float(t)


def _dias_a_lista(dias_raw: str) -> list[int]:
    """'L-V' → [1,2,3,4,5]; 'L,M,X' → [1,2,3]; 'S-D' → [6,7]."""
    dias: list[int] = []
    for parte in dias_raw.replace(" ", "").split(","):
        if not parte:
            continue
        if "-" in parte:
            a, b = parte.split("-", 1)
            if a not in _DIA_NUM or b not in _DIA_NUM:
                return []
            ini, fin = _DIA_NUM[a], _DIA_NUM[b]
            if ini <= fin:
                dias.extend(range(ini, fin + 1))
            else:  # rango que cruza la semana, ej: 'D-L' → 7,1
                dias.extend(range(ini, 8))
                dias.extend(range(1, fin + 1))
        else:
            if parte not in _DIA_NUM:
                return []
            dias.append(_DIA_NUM[parte])
    return sorted(set(dias))


def _rango_dias(a: int, b: int) -> str:
    if a == b:
        return _NOMBRE_DIAS[a]
    return f"{_NOMBRE_DIAS[a]}-{_NOMBRE_DIAS[b]}"


def _formatear_dias(dias_texto: str) -> str:
    """'1,2,3,4,5' → 'L-V'; '1,3,5' → 'L,X,V'."""
    nums = sorted({int(d) for d in dias_texto.split(",") if d.strip()})
    if not nums:
        return "?"
    partes: list[str] = []
    inicio = prev = nums[0]
    for n in nums[1:]:
        if n == prev + 1:
            prev = n
            continue
        partes.append(_rango_dias(inicio, prev))
        inicio = prev = n
    partes.append(_rango_dias(inicio, prev))
    return ",".join(partes)


def _parsear_rutina(texto: str) -> dict | None:
    """Convierte 'L-V trabajo 8-16' → dict de rutina. None si no parsea.

    Formato: <dias> <nombre> <hora_inicio-hora_fin>. Días con letras L M X J V
    S D y rangos con guion ('L-V'). El nombre es libre; si coincide con un
    deporte conocido, se guarda como deporte para afinar la predicción.
    """
    partes = texto.strip().split()
    if len(partes) < 3 or "-" not in partes[-1]:
        return None
    dias_raw = partes[0]
    nombre = " ".join(partes[1:-1]).lower()
    h_inicio_raw, h_fin_raw = partes[-1].split("-", 1)
    try:
        hora_inicio = _hora_a_float(h_inicio_raw)
        hora_fin = _hora_a_float(h_fin_raw)
    except (ValueError, TypeError):
        return None
    if not (0 <= hora_inicio < 24 and 0 < hora_fin <= 24 and hora_fin > hora_inicio):
        return None
    dias = _dias_a_lista(dias_raw)
    if not dias:
        return None
    deporte = nombre if nombre in DEPORTES else None
    return {
        "nombre": nombre,
        "dias": ",".join(str(d) for d in dias),
        "hora_inicio": hora_inicio,
        "hora_fin": hora_fin,
        "deporte": deporte,
    }


def _etiqueta_ocupacion(ocupacion: str | None) -> str | None:
    """'Construcción x2.2' para una ocupación conocida, o None si no lo es."""
    if not ocupacion or ocupacion not in _OCUPACION_NIVELES:
        return None
    coef, _ = _OCUPACION_NIVELES[ocupacion]
    return f"{OCUPACIONES.get(ocupacion, ocupacion.capitalize())} x{coef}"


def _resumen_rutinas(rutinas: list[dict]) -> str:
    lineas = ["*Tus rutinas:*"]
    for r in rutinas:
        extra = ""
        etiqueta_ocp = _etiqueta_ocupacion(r.get("ocupacion"))
        if etiqueta_ocp:
            extra = f" ({etiqueta_ocp})"
        lineas.append(
            f"  {r['id']}. {r['nombre'].capitalize()} — "
            f"{_formatear_dias(r['dias'])}, {_formato_hora(r['hora_inicio'])}-{_formato_hora(r['hora_fin'])}{extra}"
        )
    lineas.append("\nAñade una con: /rutinas_anadir <dias> <nombre> <inicio-fin>")
    lineas.append("Ej: /rutinas_anadir L-V trabajo 8-16")
    return "\n".join(lineas)


def _kb_borrar_rutinas(rutinas: list[dict]) -> list[list[dict]]:
    return [
        [
            {
                "text": f"Borrar: {r['nombre'].capitalize()} ({_formatear_dias(r['dias'])})",
                "callback_data": f"del_rutina_{r['id']}",
            }
        ]
        for r in rutinas
    ]


def _validar_hora(texto: str) -> str | None:
    """Valida 'HH:MM' y devuelve la hora normalizada 'HH:MM'; None si no vale."""
    t = texto.strip()
    try:
        if ":" not in t:
            return None
        hh, mm = t.split(":", 1)
        h, m = int(hh), int(mm)
        if not (0 <= h <= 23 and 0 <= m <= 59):
            return None
        return f"{h:02d}:{m:02d}"
    except (ValueError, TypeError):
        return None


async def procesar_mensaje(chat_id: int, texto: str | None) -> str | None:
    """Procesa un mensaje de texto del usuario. Devuelve texto a enviar."""
    conv = _conversaciones.setdefault(chat_id, {"estado": Estado.IDLE, "data": {}})
    # Modelo por defecto al crear conversación
    if "modelo" not in conv:
        conv["modelo"] = _modelo_por_defecto()
    estado = conv["estado"]
    data = conv["data"]

    if texto is None:
        return None

    # ── Comandos ──────────────────────────────────────────────────────

    if texto.startswith("/start"):
        conv["data"] = {}
        # Un /start nuevo sobreescribe el parte anterior y reabre el chat de
        # preguntas con el contexto nuevo (CHAT-003).
        conv.pop("ultima_prediccion", None)
        conv.pop("ultimo_resultado", None)
        conv.pop("_rutina_pendiente", None)
        # ¿Tiene perfil guardado?
        match = _db.buscar_por_telegram(str(chat_id))
        if match:
            perfil = _db.obtener_perfil(match["id"])
            conv["data"].update(_perfil_a_data(perfil))
            conv["estado"] = Estado.ACTIVIDAD  # saltar personales
            alias = perfil.get("alias") or match.get("alias", "")
            return f"Hola de nuevo, {alias}! Se cargaron tus datos previos. Solo las preguntas del dia:"
        conv["estado"] = Estado.SEXO
        # Sin perfil guardado: es la primera vez que este chat usa el bot, así
        # que se explica qué hay antes de soltarle la primera pregunta.
        return BIENVENIDA

    # ── Cambio de modelo ──────────────────────────────────────────────
    modelo_str = conv.get("modelo", MODELO_DETERMINISTA)
    if modelo_str == MODELO_DETERMINISTA:
        modelo_mostrar = "Determinista (sin LLM)"
    else:
        modelo_mostrar = modelo_str

    if texto.startswith("/model"):
        return (
            f"Modelo actual: *{modelo_mostrar}*\n\n"
            "Atajos:\n"
            "  /qwen — Qwen 2.5 local\n"
            "  /api — API externa (Groq)\n"
            "  /determinista — Sin LLM\n\n"
            "O escribe cualquier modelo LiteLLM:\n"
            "  ollama/qwen2.5:7b, groq/llama-3.3-70b, gpt-4o…\n\n"
            "Usa los botones de abajo."
        )

    # Atajos de modelo
    if texto.startswith("/qwen"):
        # Si tiene 7b en Ollama, usar ese; si no, 1.5b
        st = check_ollama()
        if "qwen2.5:7b" in st.get("models", []):
            conv["modelo"] = "ollama/qwen2.5:7b"
            return "Modo cambiado a *Qwen 2.5 7B* (GPU + RAG)."
        conv["modelo"] = MODELO_LOCAL
        return "Modo cambiado a *Qwen 2.5 1.5B* (CPU + RAG)."
    if texto.startswith("/api"):
        conv["modelo"] = MODELO_API
        return "Modo cambiado a *API externa* (Groq Llama 3 70B)."
    if texto.startswith("/determinista"):
        conv["modelo"] = MODELO_DETERMINISTA
        return "Modo cambiado a *Determinista* (plantilla, sin LLM)."

    # ¿Es un modelo LiteLLM escrito directamente?
    if texto and ("/" in texto and not texto.startswith("/")):
        # Podría ser "ollama/qwen2.5:7b" escrito por el usuario
        if any(prefix in texto for prefix in ("ollama/", "groq/", "gpt-", "gemini/")):
            conv["modelo"] = texto.strip()
            return f"Modelo cambiado a *{texto.strip()}*."

    if texto.startswith("/help"):
        return BIENVENIDA

    # ── BOT-007: /perfil, /rutinas y /avisos ─────────────────────────

    if texto.startswith("/perfil"):
        conv.pop("_editando", None)
        conv.pop("_rutina_pendiente", None)
        conv["data"].pop("_edit_set", None)
        match = _db.buscar_por_telegram(str(chat_id))
        if not match:
            return "No tienes perfil guardado. Usa /start para crear uno."
        return _resumen_perfil(_db.obtener_perfil(match["id"]))

    if texto.startswith("/rutinas_anadir"):
        resto = texto[len("/rutinas_anadir") :].strip()
        if not resto:
            return (
                "Usa el formato: /rutinas_anadir <dias> <nombre> <inicio-fin>\n"
                "Ej: /rutinas_anadir L-V trabajo 8-16\n"
                "    /rutinas_anadir L-V entreno 18-20\n"
                "Días: L M X J V S D (rangos con guion, ej: L-V)"
            )
        rutina = _parsear_rutina(resto)
        if rutina is None:
            return (
                "No entendí la rutina. Formato: /rutinas_anadir <dias> <nombre> <inicio-fin>\n"
                "Ej: /rutinas_anadir L-V trabajo 8-16"
            )
        # BOT-015: los deportes (y el entrenamiento genérico) se guardan directo;
        # una rutina que no es deporte se trata como salida laboral y se pregunta
        # el tipo de trabajo antes de guardar: la intensidad laboral (x1.0 a x2.7)
        # es un factor que el modelo sí sabe puntuar, no puede quedar en genérico.
        if rutina["deporte"] or rutina["nombre"] in _NOMBRES_ENTRENAMIENTO:
            _db.crear_rutina(str(chat_id), **rutina)
            return "Rutina añadida:\n" + _resumen_rutinas(_db.listar_rutinas(str(chat_id)))
        conv["_rutina_pendiente"] = rutina
        return FIELD_LABELS[Estado.TIPO_TRABAJO][0]

    if texto.startswith("/rutinas"):
        conv.pop("_editando", None)
        conv.pop("_rutina_pendiente", None)
        rutinas = _db.listar_rutinas(str(chat_id))
        if not rutinas:
            return (
                "No tienes rutinas guardadas.\n"
                "Añade una con: /rutinas_anadir <dias> <nombre> <inicio-fin>\n"
                "Ej: /rutinas_anadir L-V trabajo 8-16"
            )
        return _resumen_rutinas(rutinas)

    if texto.startswith("/avisos"):
        conv.pop("_editando", None)
        conv.pop("_rutina_pendiente", None)
        resto = texto[len("/avisos") :].strip()
        if not resto:
            hora_actual = _db.obtener_hora_aviso(str(chat_id))
            if hora_actual:
                return (
                    f"Tu aviso diario está configurado a las {hora_actual}.\n"
                    "Cambia con /avisos HH:MM o desactívalo con /avisos off."
                )
            return (
                "No tienes aviso diario configurado.\n"
                "Usa /avisos HH:MM (ej: /avisos 08:00) para que cada día te "
                "avise del riesgo de tus rutinas."
            )
        if resto.lower() in ("off", "no", "desactivar"):
            _db.guardar_hora_aviso(str(chat_id), None)
            return "Aviso diario desactivado."
        hora = _validar_hora(resto)
        if hora is None:
            return "Formato de hora inválido. Usa HH:MM (ej: /avisos 08:00)."
        _db.guardar_hora_aviso(str(chat_id), hora)
        return f"Aviso diario configurado a las {hora}. Cada día a esa hora te avisaré del riesgo de tus rutinas."

    # ── BOT-007: respuesta de texto a una edición de perfil ─────────
    editando = conv.get("_editando")
    if editando:
        conv.pop("_editando", None)
        conv["data"].pop("_edit_set", None)
        if texto.lower() in ("cancelar", "cancel"):
            return "Edición cancelada."
        ok, mensaje = _parsear_y_guardar_campo(chat_id, editando, texto)
        return mensaje

    # ── Chat abierto de preguntas tras el parte (CHAT-003) ────────────
    # Al terminar /start con LLM activo, el chat queda abierto: cualquier
    # mensaje es una duda sobre el parte que se acaba de entregar, y se
    # responde con RAG usando el parte como contexto.
    if estado == Estado.DONE and data.get("_prediccion_hecha"):
        if texto.lower() in ("salir", "exit", "/salir"):
            conv["estado"] = Estado.IDLE
            data.pop("_prediccion_hecha", None)
            return "Saliendo del chat de preguntas. /start para una nueva salida."
        # El parte entregado es el contexto: sin él el RAG rechazaría las
        # dudas sobre lo que el propio bot recomendó (p. ej. qué es SPF).
        # BOT-011: además del parte, se pasan los datos REALES de la predicción
        # (%, factores, ubicación) para que la respuesta sea personalizada.
        contexto = _contexto_parte_conversacion(conv)
        return await _preguntar_al_rag(texto, conv, contexto)

    if estado == Estado.IDLE:
        return BIENVENIDA

    # Si venimos de "Otro (lo escribo)" en DEPORTE, el texto es el nombre del deporte
    if estado == Estado.DEPORTE and data.pop("_texto_deporte", False):
        if texto and texto.lower() == "saltar":
            data["deporte"] = None
        else:
            data["deporte"] = texto.strip() if texto else None
        conv["estado"] = _siguiente(estado, data)
        return None

    # Si venimos de GUARDAR_PERFIL → "Si", el texto es el alias
    if estado == Estado.GUARDAR_PERFIL and data.pop("_esperando_alias", False):
        alias = texto.strip() if texto else f"user_{chat_id}"
        try:
            datos_perfil = _data_a_perfil(data, alias, str(chat_id))
            _db.crear_perfil(datos_perfil)
        except Exception as exc:
            logger.exception("Error al guardar perfil")
        conv["estado"] = _siguiente(estado, data)  # → DONE
        return None

    if texto and texto.lower() == "saltar":
        data[FIELD_LABELS[estado][1]] = None
        conv["estado"] = _siguiente(estado, data)
        return None

    campo = FIELD_LABELS[estado][1]

    # Ubicación escrita a mano: la resuelve Nominatim, nunca el LLM. Si no la
    # encuentra se dice y se vuelve a preguntar; inventarse unas coordenadas daría
    # una predicción con el tiempo de otro sitio sin que salte ninguna alarma.
    if estado == Estado.UBICACION:
        lugar = buscar_lugar(texto or "")
        if lugar is None:
            return ("No he encontrado ese sitio. Prueba con otro nombre "
                    "(ej: 'Aldán, Pontevedra') o pulsa el botón de ubicación.")
        data["lat"] = lugar["lat"]
        data["lon"] = lugar["lon"]
        data["provincia"] = lugar["provincia"] or "Madrid"
        data["lugar"] = lugar["nombre"]
        conv["estado"] = _siguiente(estado, data)
        return None

    # Validar según el campo
    if campo == "fecha_nacimiento":
        ok, edad = _parsear_fecha_nacimiento(texto or "")
        if not ok:
            return edad  # mensaje de error claro, nunca un crash
        data["fecha_nacimiento"] = texto.strip()
        data["edad"] = edad
    elif campo in ("duracion_h", "hora_inicio", "porcentaje_grasa"):
        try:
            if campo in ("duracion_h",):
                v = float(texto.strip()) if texto else None
                if v is not None and (v <= 0 or v > 24):
                    return "La duración debe estar entre 0.5 y 24 horas."
            elif campo == "hora_inicio":
                v = float(texto.replace(":", ".").strip()) if texto else None
                if v is not None and (v < 0 or v >= 24):
                    return "La hora debe estar entre 0 y 23."
            elif campo == "porcentaje_grasa":
                v = float(texto.replace("%", "").replace(",", ".").strip()) if texto else None
                if v is not None and not (3 <= v <= 65):
                    return "El % de grasa corporal debe estar entre 3 y 65, o escribe 'saltar'."
        except (ValueError, TypeError):
            return f"Valor inválido para '{campo}'. Escribe un número o 'saltar'."

        if v is not None:
            if campo == "hora_inicio":
                data["hora_inicio"] = int(v)
            else:
                data[campo] = v
        else:
            data[campo] = None
    else:
        data[campo] = texto.strip() if texto else ""

    conv["estado"] = _siguiente(estado, data)
    return None


def _siguiente(actual: Estado, data: dict | None = None) -> Estado:
    """Siguiente estado, saltándose las ramas que no tocan.

    Trabajo y deporte son cosas distintas: quien sale a trabajar elige el tipo de
    trabajo (que sí pesa en el riesgo, hasta x2.7) y quien sale por su cuenta dice
    qué va a hacer (que no pesa: eso lo marca la intensidad).
    """
    order = list(Estado)
    idx = order.index(actual)
    sig = order[idx + 1] if idx + 1 < len(order) else Estado.DONE

    if data is not None:
        por_trabajo = data.get("_por_trabajo")
        if sig == Estado.TIPO_TRABAJO and not por_trabajo:
            return _siguiente(sig, data)
        if sig == Estado.DEPORTE and por_trabajo:
            return _siguiente(sig, data)
    return sig


async def procesar_callback(chat_id: int, callback_data: str) -> tuple[str | None, bool]:
    """Procesa un callback de botón inline. Devuelve (texto_respuesta, es_final)."""
    conv = _conversaciones.get(chat_id)
    if not conv:
        return "Envía /start para comenzar.", False

    # ── BOT-007: borrar una rutina ───────────────────────────────────
    if callback_data.startswith("del_rutina_"):
        try:
            rid = int(callback_data.split("_", 2)[2])
        except ValueError:
            return "Rutina inválida.", False
        _db.eliminar_rutina(rid)
        return "Rutina eliminada. /rutinas para ver las que quedan.", False

    # ── BOT-015: elegir el tipo de trabajo de una rutina pendiente ──
    if callback_data.startswith("rutina_tipo_"):
        rutina = conv.get("_rutina_pendiente")
        if not rutina:
            return "No hay ninguna rutina pendiente. Usa /rutinas_anadir.", False
        ocupacion = callback_data[len("rutina_tipo_"):]
        if ocupacion not in OCUPACIONES:
            return "Tipo de trabajo inválido.", False
        rutina["ocupacion"] = ocupacion
        conv.pop("_rutina_pendiente", None)
        _db.crear_rutina(str(chat_id), **rutina)
        return "Rutina añadida:\n" + _resumen_rutinas(_db.listar_rutinas(str(chat_id))), False

    # ── BOT-007: iniciar la edición de un campo del perfil ───────────
    if callback_data.startswith("edit_"):
        campo = callback_data[5:]
        if campo not in _CAMPOS_EDITABLES:
            return "Campo desconocido.", False
        conv["_editando"] = campo
        return _PREGUNTAS_EDIT.get(campo, "Escribe el nuevo valor:"), False

    # ── BOT-007: respuesta a un campo de opción en edición ───────────
    editando = conv.get("_editando")
    if editando:
        if callback_data == "__done__":
            conv.pop("_editando", None)
            if editando in _CAMPOS_ARRAY:
                conjunto = conv["data"].pop("_edit_set", set())
                ok, mensaje = _guardar_campo_valor(chat_id, editando, conjunto)
                return mensaje, False
            return "Edición cancelada.", False
        return await _procesar_callback_edicion(chat_id, callback_data, editando)

    estado = conv["estado"]
    data = conv["data"]

    # Tras el parte con LLM el chat queda abierto en DONE: los botones del
    # cuestionario ya no valen, se responde por texto.
    if estado == Estado.DONE:
        return "Escribe tu duda, o /start para una nueva salida.", False

    campo = FIELD_LABELS[estado][1]

    if estado in MULTISELECCION:
        if callback_data == "__done__":
            conv["estado"] = _siguiente(estado, data)
            return None, False
        elegidos = data.setdefault(campo, set())
        if callback_data in elegidos:
            elegidos.discard(callback_data)
        else:
            elegidos.add(callback_data)
        # No avanzamos, esperamos más clics o "Terminé"
        return None, False

    if estado == Estado.FOTOTIPO:
        data["fototipo"] = callback_data
    elif estado == Estado.SEXO:
        data["sexo"] = callback_data
    elif estado == Estado.ACLIMATADO:
        data["aclimatado"] = callback_data == "si"
    elif estado == Estado.ENTRENADO:
        data["entrenado"] = callback_data == "si"
    elif estado == Estado.ACTIVIDAD:
        data["nivel_actividad"] = callback_data
    elif estado == Estado.TRABAJO:
        data["_por_trabajo"] = callback_data == "trabajo"
        if not data["_por_trabajo"]:
            data["ocupacion"] = None   # sin exposición laboral que sumar
    elif estado == Estado.TIPO_TRABAJO:
        data["ocupacion"] = callback_data
    elif estado == Estado.DEPORTE:
        if callback_data == "__saltar__":
            data["deporte"] = None
        elif callback_data == "__otro__":
            data["_texto_deporte"] = True
            return "Escribe el nombre del deporte:", False
        else:
            data["deporte"] = DEPORTES.get(callback_data, callback_data)
            # El MET del deporte es más fiable que el adjetivo que elija el
            # usuario: 8 MET de tenis individual son muy_intensa aunque él haya
            # dicho "moderada". Si el deporte no está en la tabla, se respeta.
            nivel = nivel_actividad_de_deporte(callback_data)
            if nivel:
                data["nivel_actividad"] = nivel
                data["_nivel_desde_deporte"] = True
    elif estado == Estado.GUARDAR_PERFIL:
        if callback_data == "guardar_si":
            data["_esperando_alias"] = True
            return "Como quieres llamarte?", False
        # guardar_no → avanza a DONE (el siguiente estado)

    sig = _siguiente(estado, data)
    conv["estado"] = sig
    return None, sig == Estado.DONE


async def _procesar_callback_edicion(
    chat_id: int, callback_data: str, editando: str
) -> tuple[str | None, bool]:
    """Callback de un campo de opción en el mini-formulario de edición.

    Los campos simples guardan con el primer clic; los de lista (comorbilidades,
    medicación, situación social) acumulan en ``_edit_set`` hasta 'Terminé'.
    """
    conv = _conversaciones[chat_id]
    data = conv["data"]

    if editando == "sexo":
        conv.pop("_editando", None)
        ok, mensaje = _guardar_campo_valor(chat_id, "sexo", callback_data)
        return mensaje, False
    if editando == "aclimatado":
        conv.pop("_editando", None)
        ok, mensaje = _guardar_campo_valor(chat_id, "aclimatado", callback_data == "si")
        return mensaje, False
    if editando == "fototipo":
        try:
            valor = int(callback_data)
        except ValueError:
            return "Fototipo inválido (1-6).", False
        conv.pop("_editando", None)
        ok, mensaje = _guardar_campo_valor(chat_id, "fototipo", valor)
        return mensaje, False

    # Listas: multiselect acumulativo hasta 'Terminé'
    elegidos = data.setdefault("_edit_set", set())
    if callback_data in elegidos:
        elegidos.discard(callback_data)
    else:
        elegidos.add(callback_data)
    return None, False


async def ejecutar_prediccion(chat_id: int) -> str:
    """Ejecuta la predicción con los datos recogidos y devuelve texto de respuesta."""
    conv = _conversaciones.get(chat_id)
    if not conv:
        return "Error: no hay datos. Envía /start."

    data = conv["data"]

    # OJO con los nombres: el modelo lee `farmacos` y `porcentaje_grasa`, no
    # `medicacion` ni `grasa_corporal`. Escribir mal una clave no da error — el
    # factor se salta en silencio y el riesgo sale más bajo de lo que debe.
    # `peso` y `altura` no los lee nadie, por eso no se piden.
    perfil = {
        "sexo": data.get("sexo", "hombre"),
        "edad": data.get("edad"),
        "aclimatado": data.get("aclimatado", False),
        "nivel_actividad": data.get("nivel_actividad", "ligera"),
        "duracion_actividad_h": data.get("duracion_h", 1),
        "hora_inicio": data.get("hora_inicio", 10),
        "comorbilidades": data.get("comorbilidades", set()),
        "farmacos": data.get("farmacos", set()),
    }
    if data.get("porcentaje_grasa") is not None:
        perfil["porcentaje_grasa"] = data["porcentaje_grasa"]
    if data.get("fototipo") is not None:
        perfil["fototipo"] = data["fototipo"]
    if data.get("deporte"):
        perfil["deporte"] = data["deporte"]   # solo etiqueta, no cambia el riesgo

    # Situación social permanente: aislamiento, vivienda, etc.
    sit_social = data.get("situacion_social") or set()
    if sit_social:
        perfil["situacion_social"] = set(sit_social)

    # Cómo llega a la salida: fiesta x1.8, enfermedad reciente x1.3, mala noche x1.2
    previo = data.get("estado_previo") or set()
    for clave in ESTADO_PREVIO_OPTS:
        if clave in previo:
            perfil[clave] = True
    if data.get("entrenado") is not None:
        perfil["entrenado"] = data["entrenado"]
    # Solo si la salida es de trabajo: pesa de x1.0 (oficina) a x2.7 (campo)
    if data.get("ocupacion"):
        perfil["ocupacion"] = data["ocupacion"]

    try:
        result = predict_ensemble(
            lat=data.get("lat"),
            lon=data.get("lon"),
            provincia=data.get("provincia", "Madrid"),
            perfil=perfil,
        )
    except Exception as exc:
        logger.exception("Error en predicción")
        return f"Error al calcular el riesgo: {exc}"

    # Se guarda el resultado por si el chat de preguntas abierto tras el parte
    # (CHAT-003) necesita el detalle de esta predicción.
    conv["ultimo_resultado"] = result

    # Respuesta según el modelo elegido
    modelo = conv.get("modelo", MODELO_DETERMINISTA)
    lugar = data.get("lugar")
    if modelo != MODELO_DETERMINISTA:
        config = LLMConfig(model=modelo)
        texto = await asyncio.to_thread(ask_con_perfil, perfil, result, config, lugar)
        if texto:
            logger.info("Respuesta redactada por %s", modelo)
        else:
            # Sin esta línea la degradación es invisible: el usuario ve la
            # plantilla y cree que el LLM está funcionando.
            logger.warning("%s no contestó; se responde con la plantilla", modelo)
    else:
        texto = None
        logger.info("Modo determinista: respuesta con plantilla")
    if not texto:
        texto = _format_template(result, lugar)

    return texto


def _contexto_parte_conversacion(conv: dict) -> str:
    """Contexto del chat abierto tras el parte (BOT-011).

    Lleva el parte entregado más los datos REALES de esa predicción
    (probabilidad de cada canal, factores, ubicación) y pide al LLM una
    respuesta concisa y personalizada, no un texto genérico de tres párrafos.
    `ultimo_resultado` puede faltar (dicts mínimos de test): entonces solo se
    pasa el parte, como antes.
    """
    partes = [f"Parte que le acabas de dar al usuario:\n{conv.get('ultima_prediccion', '')}"]
    result = conv.get("ultimo_resultado") or {}
    if result:
        w = result.get("weather") or {}
        cur = w.get("current") or {}
        perfil = result.get("perfil") or {}
        calor = perfil.get("calor") or {}
        frio = perfil.get("frio") or {}
        prob_calor = calor.get("prob_personalizada")
        prob_frio = frio.get("prob_personalizada")
        factores = calor.get("factores") or frio.get("factores") or []
        lineas = [
            f"Ubicación: {w.get('provincia') or '?'}",
            f"Clase de riesgo: {result.get('clase_final_label') or '?'}",
        ]
        if prob_calor is not None:
            lineas.append(f"Probabilidad personalizada (calor): {prob_calor:.0%}")
        if prob_frio is not None:
            lineas.append(f"Probabilidad personalizada (frío): {prob_frio:.0%}")
        if factores:
            nombres = []
            for f in factores:
                if isinstance(f, dict):
                    nombre = f.get("nombre", str(f))
                    coef = f.get("factor")
                    # Coeficiente real del pipeline (xN), no solo el nombre:
                    # sin él el LLM se inventa cuánto pesa cada factor.
                    nombres.append(f"{nombre} (x{coef})" if coef is not None else nombre)
                else:
                    nombres.append(str(f))
            lineas.append("Factores que suben tu riesgo: " + ", ".join(nombres))
        else:
            lineas.append("Factores que suben tu riesgo: ninguno relevante")
        # Ocupación de esta salida (obra, oficina, reparto...) con su etiqueta
        # y coeficiente, para que la respuesta se adapte al contexto.
        ocp = (result.get("perfil_usuario") or {}).get("ocupacion")
        if ocp in _OCUPACION_NIVELES:
            coef, label = _OCUPACION_NIVELES[ocp]
            lineas.append(f"Ocupación: {label} (x{coef})")
        if cur.get("t2m_c") is not None:
            lineas.append(f"Temperatura prevista: {cur['t2m_c']:.1f} °C")
        partes.append("DATOS REALES DE ESTA PREDICCIÓN:\n" + "\n".join(lineas))
    partes.append(
        "El usuario pregunta sobre su parte. Responde en 2-3 frases con los "
        "datos reales de ESTA predicción (porcentaje, factores, ubicación), "
        "sin textos genéricos. Adapta la respuesta al contexto de la persona "
        "(trabajo en obra, oficina, deporte...): nunca le digas 'reduce la "
        "exposición en interiores' ni repitas consejos genéricos que no "
        "apliquen a su situación."
    )
    return "\n\n".join(partes)


async def _preguntar_al_rag(texto: str, conv: dict, contexto: str | None = None) -> str:
    """Pregunta libre al LLM con RAG. `contexto` es el parte ya entregado."""
    config = LLMConfig(model=conv["modelo"])
    # Extraer el último resultado de la predicción para obtener el perfil de usuario y factores
    ultimo_resultado = conv.get("ultimo_resultado") or {}
    # Perfil del usuario desde el último resultado (si está presente); de lo contrario, reconstruir desde los datos de la conversación
    perfil_usuario = ultimo_resultado.get("perfil_usuario") or {}
    if not perfil_usuario:
        # Reconstruir el perfil igual que en ejecutar_prediccion
        data = conv.get("data") or {}
        perfil_usuario = {
            "sexo": data.get("sexo", "hombre"),
            "edad": data.get("edad"),
            "aclimatado": data.get("aclimatado", False),
            "nivel_actividad": data.get("nivel_actividad", "ligera"),
            "duracion_actividad_h": data.get("duracion_h", 1),
            "hora_inicio": data.get("hora_inicio", 10),
            "comorbilidades": data.get("comorbilidades", set()),
            "farmacos": data.get("farmacos", set()),
        }
        if data.get("porcentaje_grasa") is not None:
            perfil_usuario["porcentaje_grasa"] = data["porcentaje_grasa"]
        if data.get("fototipo") is not None:
            perfil_usuario["fototipo"] = data["fototipo"]
        if data.get("deporte"):
            perfil_usuario["deporte"] = data["deporte"]
        sit_social = data.get("situacion_social") or set()
        if sit_social:
            perfil_usuario["situacion_social"] = set(sit_social)
        previo = data.get("estado_previo") or set()
        for clave in ESTADO_PREVIO_OPTS:
            if clave in previo:
                perfil_usuario[clave] = True
        if data.get("entrenado") is not None:
            perfil_usuario["entrenado"] = data["entrenado"]
        if data.get("ocupacion"):
            perfil_usuario["ocupacion"] = data["ocupacion"]
    # Llamar a ask_with_rag con el perfil para adaptación contextual
    res = await asyncio.to_thread(ask_with_rag, texto, 3, 3, config, contexto, perfil_usuario)
    return res.get("answer") or (
        f"El LLM no respondió. Revisa que {conv['modelo']} esté disponible."
    )


async def _finalizar_parte(chat_id: int) -> None:
    """Entrega el parte al terminar /start y decide si el chat queda abierto.

    Con LLM activo (modelo no determinista) la respuesta la redacta
    `ejecutar_prediccion` vía `ask_con_perfil`, y tras el parte el chat queda
    abierto para preguntas libres con RAG: se guarda `ultima_prediccion` como
    contexto y se invita a preguntar (p. ej. qué es SPF) o a volver a /start.
    En modo determinista se responde con la plantilla y se cierra como antes.
    """
    conv = _conversaciones.get(chat_id)
    if not conv:
        return
    texto = await ejecutar_prediccion(chat_id)
    if conv.get("modelo", MODELO_DETERMINISTA) != MODELO_DETERMINISTA:
        conv["data"]["_prediccion_hecha"] = True
        conv["ultima_prediccion"] = texto
        await enviar_mensaje(chat_id, f"{texto}\n\n{CHAT_CIERRE}")
        return
    await enviar_mensaje(chat_id, texto)
    _conversaciones.pop(chat_id, None)


async def enviar_mensaje(
    chat_id: int,
    texto: str,
    kb: list[list[dict]] | None = None,
    reply_markup: dict | None = None,
) -> None:
    """Envía un mensaje. `kb` es un teclado inline; `reply_markup` va tal cual.

    Los dos no son intercambiables: `request_location` solo funciona en un
    ReplyKeyboard, no en los botones inline.
    """
    payload: dict[str, Any] = {
        "chat_id": chat_id,
        "text": texto,
        "parse_mode": "Markdown",
    }
    if reply_markup:
        payload["reply_markup"] = json.dumps(reply_markup)
    elif kb:
        payload["reply_markup"] = json.dumps({"inline_keyboard": kb})
    try:
        await _tg("sendMessage", **payload)
    except httpx.HTTPStatusError as exc:
        if exc.response.status_code != 400:
            raise
        # Un asterisco o un guion bajo sueltos en el texto que redacta el LLM y
        # Telegram rechaza el mensaje ENTERO con un 400: el usuario no recibe
        # nada. Se reenvía en plano — mejor sin negritas que sin mensaje.
        logger.warning("Telegram rechazó el formato del mensaje; se reenvía en texto plano")
        payload.pop("parse_mode", None)
        await _tg("sendMessage", **payload)


_CAMPOS_PERFIL = {
    Estado.COMORBILIDADES: "comorbilidades",
    Estado.MEDICACION: "farmacos",
    Estado.SITUACION_SOCIAL: "situacion_social",
}


def _saltar_si_prellenado(conv: dict) -> None:
    """Avanza el estado si los datos vienen de perfil cargado."""
    if not conv["data"].get("_perfil_cargado"):
        return
    estado = conv["estado"]
    campo = _CAMPOS_PERFIL.get(estado)
    if campo and campo in conv["data"]:
        conv["estado"] = _siguiente(estado, conv["data"])
        _saltar_si_prellenado(conv)  # recursivo: puede haber varios seguidos


async def enviar_siguiente_pregunta(chat_id: int) -> None:
    """Envía la pregunta correspondiente al estado actual con sus botones."""
    conv = _conversaciones.get(chat_id)
    if not conv:
        return
    estado = conv["estado"]

    # Saltar estados personales si ya tienen datos cargados (perfil)
    _saltar_si_prellenado(conv)

    estado = conv["estado"]
    if estado == Estado.DONE:
        await _finalizar_parte(chat_id)
        return

    if estado == Estado.GUARDAR_PERFIL:
        # Preguntar solo si no tiene perfil aún
        match = _db.buscar_por_telegram(str(chat_id))
        if match:
            # Ya tiene perfil → saltar directamente a DONE
            conv["estado"] = _siguiente(estado, conv["data"])
            await enviar_siguiente_pregunta(chat_id)
            return
        await enviar_mensaje(
            chat_id,
            "Quieres guardar tu perfil para la proxima?",
            kb=[[{"text": "Si", "callback_data": "guardar_si"}],
                [{"text": "No", "callback_data": "guardar_no"}]],
        )
        return

    info = FIELD_LABELS.get(estado)
    if not info:
        return
    texto_pregunta, _, kb = info
    if estado == Estado.UBICACION:
        await enviar_mensaje(chat_id, texto_pregunta, reply_markup=_kb_ubicacion())
    else:
        await enviar_mensaje(chat_id, texto_pregunta, kb)


# ── Bucle de polling ──────────────────────────────────────────────────────

_shutdown_event = asyncio.Event()


async def polling_loop() -> None:
    """Bucle principal: long polling contra Telegram Bot API."""
    logger.info("Bot iniciado — polling cada 30s con timeout 25s")

    # Verificar que el token funciona al arranque
    try:
        me = await _tg("getMe")
        logger.info("Bot autenticado: @%s", me.get("result", {}).get("username", "?"))
    except httpx.HTTPStatusError as exc:
        logger.error("Token inválido o sin acceso: %s", exc)
        # Sin token no hay nada que avisar: se para también la task de avisos
        _shutdown_event.set()
        return

    offset = 0
    while not _shutdown_event.is_set():
        try:
            params = {"timeout": 25, "offset": offset, "allowed_updates": ["message", "callback_query"]}
            resp = await _tg("getUpdates", **params)
            for update in resp.get("result", []):
                update_id = update.get("update_id", 0)
                offset = max(offset, update_id + 1)
                try:
                    await procesar_update(update)
                except Exception as exc:
                    logger.exception("Error procesando update %s: %s", update_id, exc)
        except httpx.TimeoutException:
            continue  # timeout normal del long polling
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code == 401:
                logger.error("Token revocado. Deteniendo bot.")
                _shutdown_event.set()
                return
            logger.error("Error HTTP %s: %s", exc.response.status_code, exc)
            await asyncio.sleep(5)
        except httpx.ConnectError:
            logger.warning("Sin conexión. Reintentando en 30s...")
            await asyncio.sleep(30)
        except Exception as exc:
            logger.exception("Error en polling loop: %s", exc)
            await asyncio.sleep(10)


# ── BOT-007: aviso diario de riesgo por rutinas ────────────────────────────

# chat_id + fecha de los avisos ya enviados, para no repetir el del día
# aunque el bot reinicie o el check pase varias veces en el mismo minuto.
_avisos_enviados: set[str] = set()


def _perfil_prediccion_desde_rutina(perfil: dict, rutina: dict) -> dict:
    """Perfil para predict_ensemble: datos del chat + la ventana de la rutina.

    La ventana a evaluar la define la rutina (hora_inicio + duración), no el
    perfil; el resto de factores (edad, comorbilidades, fármacos...) vienen del
    perfil guardado del chat.
    """
    p = {
        "sexo": perfil.get("sexo", "hombre"),
        "edad": perfil.get("edad"),
        "aclimatado": perfil.get("aclimatado", False),
        "nivel_actividad": "ligera",
        "duracion_actividad_h": rutina["hora_fin"] - rutina["hora_inicio"],
        "hora_inicio": rutina["hora_inicio"],
        "comorbilidades": set(perfil.get("comorbilidades") or []),
        "farmacos": set(perfil.get("farmacos") or []),
    }
    if perfil.get("porcentaje_grasa") is not None:
        p["porcentaje_grasa"] = perfil["porcentaje_grasa"]
    if perfil.get("fototipo") is not None:
        p["fototipo"] = perfil["fototipo"]
    if perfil.get("situacion_social"):
        p["situacion_social"] = set(perfil["situacion_social"])
    if rutina.get("deporte"):
        p["deporte"] = rutina["deporte"]
        nivel = nivel_actividad_de_deporte(rutina["deporte"])
        if nivel:
            p["nivel_actividad"] = nivel
    if rutina.get("ocupacion"):
        p["ocupacion"] = rutina["ocupacion"]
    return p


def _etiqueta_rutina(rutina: dict) -> str:
    """'Trabajo 8:00-16:00 (Campo x2.7)' o 'Entreno 18:00-20:00 (correr)'."""
    nombre = rutina["nombre"].capitalize()
    ventana = f"{_formato_hora(rutina['hora_inicio'])}-{_formato_hora(rutina['hora_fin'])}"
    actividad = ""
    if rutina.get("deporte"):
        actividad = f" ({rutina['deporte']})"
    elif rutina.get("ocupacion"):
        etiqueta = _etiqueta_ocupacion(rutina["ocupacion"])
        actividad = f" ({etiqueta})" if etiqueta else f" (trabajo: {rutina['ocupacion']})"
    return f"{nombre} {ventana}{actividad}"


async def _enviar_aviso_diario(chat_id: str, weekday: int) -> None:
    """Calcula y envía el aviso de riesgo de las rutinas de un chat para hoy.

    Sin perfil o sin ubicación no calcula: avisa de que falta configurar.
    """
    chat_id_int = int(chat_id)
    match = _db.buscar_por_telegram(chat_id)
    if not match:
        await enviar_mensaje(chat_id_int, AVISO_SIN_PERFIL)
        return
    perfil = _db.obtener_perfil(match["id"])
    if perfil.get("lat") is None or perfil.get("lon") is None:
        await enviar_mensaje(chat_id_int, AVISO_SIN_UBICACION)
        return

    rutinas = _db.rutinas_por_dia(chat_id, weekday)
    if not rutinas:
        return  # sin rutinas hoy: nada que avisar

    partes = ["☀️ *Aviso diario* — riesgo de tus rutinas de hoy:"]
    for r in rutinas:
        perfil_pred = _perfil_prediccion_desde_rutina(perfil, r)
        try:
            result = await asyncio.to_thread(
                predict_ensemble,
                lat=perfil["lat"],
                lon=perfil["lon"],
                provincia=perfil.get("provincia") or "Madrid",
                perfil=perfil_pred,
            )
        except Exception:
            logger.exception("Error prediciendo rutina %s de %s", r["id"], chat_id)
            partes.append(f"• {_etiqueta_rutina(r)}: no se pudo calcular el riesgo")
            continue
        clase = result.get("clase_final_label", "?")
        prob = result.get("perfil", {}).get("calor", {}).get("prob_personalizada") or 0
        temps = _temps_en_ventana(
            result.get("weather", {}).get("perfil_horario") or [],
            {
                "hora_inicio": r["hora_inicio"],
                "duracion_actividad_h": r["hora_fin"] - r["hora_inicio"],
            },
        )
        temp_txt = f"{round(sum(temps) / len(temps), 1):.0f} °C" if temps else "temp n/d"
        rec = recomendacion_resumen(result)
        partes.append(
            f"• {_etiqueta_rutina(r)}: riesgo {clase} ({prob:.0%}), {temp_txt}\n  → {rec}"
        )
    await enviar_mensaje(chat_id_int, "\n".join(partes))


async def tarea_avisos_diarios() -> None:
    """Task de fondo: cada ~30s mira si toca enviar el aviso diario de algún chat.

    Dispara cuando la hora configurada coincide con la actual y el minuto es 0,
    una vez por chat y día. Corre junto a ``polling_loop`` desde ``main()``.
    """
    logger.info("Task de avisos diarios iniciado")
    while not _shutdown_event.is_set():
        try:
            await asyncio.sleep(30)
            ahora = datetime.now()
            if ahora.minute != 0:
                continue
            hoy = ahora.strftime("%Y-%m-%d")
            weekday = ahora.isoweekday()  # 1=lunes ... 7=domingo
            for aviso in _db.chats_con_aviso():
                hhmm = _hora_aviso_dict(aviso["hora"])
                if hhmm is None or hhmm != (ahora.hour, ahora.minute):
                    continue
                clave = f"{aviso['chat_id']}:{hoy}"
                if clave in _avisos_enviados:
                    continue
                _avisos_enviados.add(clave)
                try:
                    await _enviar_aviso_diario(aviso["chat_id"], weekday)
                except Exception as exc:
                    logger.exception("Error en aviso diario de %s: %s", aviso["chat_id"], exc)
        except asyncio.CancelledError:
            break
        except Exception as exc:
            logger.exception("Error en task de avisos: %s", exc)


def _hora_aviso_dict(hora_texto: str) -> tuple[int, int] | None:
    """'08:00' → (8, 0). None si el formato no es 'HH:MM'."""
    try:
        hh, mm = hora_texto.split(":")
        return int(hh), int(mm)
    except (ValueError, AttributeError):
        return None


async def _recibir_ubicacion(chat_id: int, location: dict) -> None:
    """Guarda la ubicación compartida y sigue con la conversación."""
    conv = _conversaciones.get(chat_id)
    if not conv or conv["estado"] != Estado.UBICACION:
        await enviar_mensaje(chat_id, "Gracias, pero ahora mismo no te estaba pidiendo la ubicación.")
        return

    lat, lon = location.get("latitude"), location.get("longitude")
    logger.info("Ubicación de %s: %s, %s", chat_id, lat, lon)

    sitio = provincia_desde_coords(lat, lon) or {}
    conv["data"]["lat"] = lat
    conv["data"]["lon"] = lon
    conv["data"]["provincia"] = sitio.get("provincia") or "Madrid"
    conv["data"]["lugar"] = sitio.get("nombre") or f"{lat:.4f}, {lon:.4f}"
    conv["estado"] = _siguiente(Estado.UBICACION, conv["data"])
    await enviar_siguiente_pregunta(chat_id)


async def procesar_update(update: dict) -> None:
    """Procesa una update de Telegram."""
    msg = update.get("message")
    cb = update.get("callback_query")

    if msg:
        chat_id = msg["chat"]["id"]
        texto = msg.get("text", "")

        # Ubicación compartida con el botón nativo: lat/lon exactas de Telegram.
        # Solo falta el nombre de la provincia, que el modelo necesita, y eso lo
        # da la geocodificación inversa.
        if msg.get("location"):
            await _recibir_ubicacion(chat_id, msg["location"])
            return

        logger.info("Mensaje de %s: %s", chat_id, texto[:50])

        if texto.startswith("/start"):
            respuesta = await procesar_mensaje(chat_id, texto)
            if respuesta:
                await enviar_mensaje(chat_id, respuesta)
            await enviar_siguiente_pregunta(chat_id)
            return

        if texto.startswith("/model"):
            conv_actual = _conversaciones.get(chat_id, {})
            modelo_act = conv_actual.get("modelo", MODELO_DETERMINISTA)
            texto_resp = await procesar_mensaje(chat_id, texto)
            if texto_resp:
                await enviar_mensaje(chat_id, texto_resp, kb=_kb_modelos(modelo_act))
            return

        if texto.startswith("/perfil"):
            respuesta = await procesar_mensaje(chat_id, texto)
            if respuesta:
                kb = _kb_edicion_perfil() if _db.buscar_por_telegram(str(chat_id)) else None
                await enviar_mensaje(chat_id, respuesta, kb)
            return

        if texto.startswith("/rutinas_anadir"):
            respuesta = await procesar_mensaje(chat_id, texto)
            if respuesta:
                # BOT-015: si la rutina es de trabajo queda pendiente y la
                # respuesta es la pregunta del tipo, con sus botones inline.
                conv_actual = _conversaciones.get(chat_id, {})
                kb = _kb_tipo_trabajo("rutina_tipo_") if conv_actual.get("_rutina_pendiente") else None
                await enviar_mensaje(chat_id, respuesta, kb)
            return

        if texto.startswith("/rutinas") and not texto.startswith("/rutinas_anadir"):
            respuesta = await procesar_mensaje(chat_id, texto)
            if respuesta:
                rutinas = _db.listar_rutinas(str(chat_id))
                kb = _kb_borrar_rutinas(rutinas) if rutinas else None
                await enviar_mensaje(chat_id, respuesta, kb)
            return

        respuesta = await procesar_mensaje(chat_id, texto)
        if respuesta:
            await enviar_mensaje(chat_id, respuesta)
        else:
            await enviar_siguiente_pregunta(chat_id)

    elif cb:
        chat_id = cb["message"]["chat"]["id"]
        data = cb.get("data", "")
        logger.info("Callback de %s: %s", chat_id, data[:30])

        # Cambio de modelo inline
        if data and data.startswith("modelo_"):
            modelo = data[7:]
            conv_post = _conversaciones.setdefault(chat_id, {"estado": Estado.IDLE, "data": {}})
            conv_post["modelo"] = modelo
            try:
                await _tg("answerCallbackQuery", {
                    "callback_query_id": cb["id"],
                    "text": f"Modelo: {modelo}" if modelo != MODELO_DETERMINISTA else "Modo determinista",
                })
            except Exception:
                pass
            await _tg("editMessageText", {
                "chat_id": chat_id,
                "message_id": cb["message"]["message_id"],
                "text": f"✓ Modelo cambiado a: `{modelo}`" if modelo != MODELO_DETERMINISTA else "✓ Modo determinista (sin LLM)",
                "parse_mode": "Markdown",
            })
            return

        # Guardar estado antes de procesar para detectar toggles multiselect
        conv_pre = _conversaciones.get(chat_id)
        estado_pre = conv_pre["estado"] if conv_pre else None

        texto_respuesta, es_final = await procesar_callback(chat_id, data)

        # Responder al callback (con toast si es toggle multiselect)
        try:
            payload: dict[str, object] = {"callback_query_id": cb["id"]}
            if (
                estado_pre in MULTISELECCION
                and data != "__done__"
                and (nombre_visible := _nombre_opcion(estado_pre, data))
            ):
                payload["text"] = nombre_visible
            await _tg("answerCallbackQuery", **payload)
        except Exception:
            pass  # no crítico

        if texto_respuesta:
            conv_edit = _conversaciones.get(chat_id)
            kb_edit = (
                _kb_edit_campo(conv_edit["_editando"])
                if conv_edit and conv_edit.get("_editando")
                else None
            )
            await enviar_mensaje(chat_id, texto_respuesta, kb_edit)
        if es_final:
            await _finalizar_parte(chat_id)
        else:
            # Solo avanzar si el estado cambió (toggle multiselect no cambia)
            conv_post = _conversaciones.get(chat_id)
            estado_post = conv_post["estado"] if conv_post else None
            if estado_post != estado_pre:
                await enviar_siguiente_pregunta(chat_id)


# ── Entry point ───────────────────────────────────────────────────────────

class _OcultarToken(logging.Filter):
    """Tapa el token de Telegram en cualquier linea que pase por el log.

    La Bot API lleva el token en la RUTA (api.telegram.org/bot<TOKEN>/sendMessage) y
    httpx registra la URL entera en INFO. El resultado era el token en claro en cada
    linea de logs/bot.log — un fichero que se rota y se queda en disco, y con el que
    cualquiera puede controlar el bot.
    """

    _MARCA = "bot<TOKEN_OCULTO>"

    def filter(self, record: logging.LogRecord) -> bool:
        token = os.getenv("TELEGRAM_BOT_TOKEN")
        if token:
            if isinstance(record.msg, str) and token in record.msg:
                record.msg = record.msg.replace(f"bot{token}", self._MARCA).replace(token, "<OCULTO>")
            if record.args:
                record.args = tuple(
                    a.replace(f"bot{token}", self._MARCA).replace(token, "<OCULTO>")
                    if isinstance(a, str) else a
                    for a in (record.args if isinstance(record.args, tuple) else (record.args,))
                )
        return True


def _setup_logging() -> None:
    """Logging a archivo rotativo siempre, y a consola solo si es un terminal real.

    Idempotente: `run_bot.sh` reinicia el proceso en bucle y, sin esta guarda, cada
    arranque anadia otro par de handlers al root y cada linea salia repetida.

    El UNICO escritor de logs/bot.log es el RotatingFileHandler. No hay console
    handler cuando stdout no es un terminal (produccion: run_bot.sh redirige
    stdout al mismo bot.log con `>> "$LOGFILE" 2>&1`): si lo hubiera, cada linea
    se escribiria dos veces en el mismo fichero — una por el handler y otra por
    la redireccion. En un terminal interactivo (desarrollo) nadie redirige, asi
    que la consola se puede anadir sin duplicar.
    """
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)

    root = logging.getLogger()
    if any(getattr(h, "_climasafe", False) for h in root.handlers):
        return
    # basicConfig() de terceros deja un handler suelto que duplica todo
    for h in list(root.handlers):
        root.removeHandler(h)

    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    filtro = _OcultarToken()

    # Archivo rotativo (5 MB × 3 backups)
    file_handler = RotatingFileHandler(
        log_dir / "bot.log", maxBytes=5 * 1024 * 1024, backupCount=3
    )
    file_handler.setFormatter(formatter)

    # Consola: solo con stdout interactivo (desarrollo). Redirigido a fichero
    # (produccion via run_bot.sh, o nohup) no se anade para no duplicar.
    handlers = [file_handler]
    if sys.stdout.isatty():
        console = logging.StreamHandler(sys.stdout)
        console.setFormatter(formatter)
        handlers.insert(0, console)

    for h in handlers:
        h.addFilter(filtro)
        h._climasafe = True          # marca para no volver a anadirlos
        root.addHandler(h)
    root.setLevel(logging.INFO)

    # httpx registra la URL completa de cada peticion, token incluido. El filtro ya
    # lo tapa, pero no hace falta ese ruido: una linea por peticion HTTP no aporta
    # nada cuando ya se loguea el mensaje y la respuesta.
    logging.getLogger("httpx").setLevel(logging.WARNING)


def main() -> None:
    from dotenv import load_dotenv
    load_dotenv()

    _setup_logging()

    if not os.getenv("TELEGRAM_BOT_TOKEN"):
        logger.error("TELEGRAM_BOT_TOKEN no está configurado en .env")
        sys.exit(1)

    # Graceful shutdown con SIGINT/SIGTERM
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    def _signal_handler():
        if _shutdown_event.is_set():
            return  # la segunda senal (kill + pkill de run_bot.sh) no re-loguea
        logger.info("Señal de parada recibida. Cerrando bot...")
        _shutdown_event.set()
        asyncio.ensure_future(_shutdown_client())

    async def _shutdown_client():
        global _HTTP_CLIENT
        if _HTTP_CLIENT:
            await _HTTP_CLIENT.aclose()
            _HTTP_CLIENT = None
        loop.stop()

    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, _signal_handler)

    try:
        loop.run_until_complete(
            asyncio.gather(
                polling_loop(),
                tarea_avisos_diarios(),
            )
        )
    except asyncio.CancelledError:
        pass
    finally:
        # Limpiar cliente HTTP
        try:
            loop.run_until_complete(_shutdown_client())
        except Exception:
            pass
        loop.close()
        logger.info("Bot detenido.")


if __name__ == "__main__":
    main()
