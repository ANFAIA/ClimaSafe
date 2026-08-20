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
import unicodedata
from datetime import date, datetime
from enum import Enum, auto

from climasafeai.bot.geocoding import buscar_lugar, provincia_desde_coords
from climasafeai.features.personalizacion import (
    DEPORTE_MET,
    _OCUPACION_NIVELES,
    nivel_actividad_de_deporte,
    pico_riesgo_actividad,
    recomendar_horario,
    riesgo_horario_acumulado,
)
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
from climasafeai.models.recomendaciones import _canal_dominante, recomendacion_resumen

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
# no tienen MET propio en el Compendium (por eso no están en DEPORTES). BOT-016:
# tampoco se guardan directo, se pregunta la actividad concreta con DEPORTES.
_NOMBRES_ENTRENAMIENTO = {"entreno", "entrenamiento"}

# BOT-016: al añadir una rutina de deporte (o entrenamiento genérico) se pregunta
# qué actividad concreta es; la respuesta se guarda en la columna `deporte`.
PREGUNTA_TIPO_ACTIVIDAD = "¿Qué tipo de actividad deportiva?"

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
# soportado por LiteLLM: "ollama/qwen2.5:1.5b", "groq/openai/gpt-oss-20b", etc.
MODELO_LOCAL = "ollama/qwen2.5:1.5b"
# HOST-001: el free tier de Groq retiró llama-3.3-70b-versatile (404 real
# verificado el 18-08-2026); gpt-oss-20b responde en el free tier (8K TPM /
# 200K TPD según la tabla pública de rate limits de Groq).
MODELO_API = "groq/openai/gpt-oss-20b"
# HOST-001: alternativa para cuentas sin key de Groq (verificado con la
# GEMINI_API_KEY real el 18-08-2026: gemini-2.5-flash da 404 "no longer
# available", gemini-3.6-flash responde).
MODELO_API_GEMINI = "gemini/gemini-3.6-flash"
MODELO_DETERMINISTA = "__determinista__"  # valor centinela: sin LLM


def _modelo_por_defecto() -> str:
    """Auto-detecta el mejor modelo local; si no hay Ollama, LLM remoto o determinista.

    Antes devolvía `MODELO_LOCAL` fijo (el 1.5B) aunque hubiera un 7B instalado, y
    la diferencia se nota: el 1.5B contesta con titulares en negrita y viñetas —
    indistinguible de la plantilla— mientras el 7B da el parte de una línea.
    `check_ollama()` ya calcula cuál es el mejor, así que se usa.

    HOST-001: si Ollama no está (portátil apagado o bot en un host sin LLM
    local), se cae al LLM remoto gratuito si hay clave; solo sin ninguna clave
    se queda en determinista. Sin este salto, un bot desplegado sin Ollama
    contestaría siempre con plantilla y el LLM remoto no se usaría nunca.
    """
    st = check_ollama()
    if st.get("available"):
        return st.get("best_model") or MODELO_LOCAL
    if os.getenv("GROQ_API_KEY"):
        return MODELO_API
    if os.getenv("GEMINI_API_KEY"):
        return MODELO_API_GEMINI
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
    CONFIRMAR_DIA = auto()  # BOT-019: resumen de la salida deducida de la rutina
    REPETIR_SALIDA = auto()  # BOT-017: ofrecer repetir la última salida guardada
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


def _kb_tipo_deporte(prefijo: str = "") -> list[list[dict]]:
    """Teclado de actividad deportiva para el cuestionario de rutina (BOT-016).

    Mismos deportes que el formulario de /start, con el prefijo
    ``rutina_deporte_`` para distinguir ese callback del flujo normal: al elegir
    uno se guarda su clave, y el MET del Compendium fija la intensidad.
    """
    return [[{"text": v, "callback_data": f"{prefijo}{k}"}] for k, v in DEPORTES.items()]


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
    Estado.CONFIRMAR_DIA: (
        # Placeholder: el texto real se construye en `_mensaje_confirmar_dia`
        # con la salida deducida de la rutina (BOT-019).
        "He dado por supuesto la salida de hoy. ¿Es correcto?",
        "_confirmar_dia",
        None,  # se monta aparte: botones Sí / Cambiar
    ),
    Estado.REPETIR_SALIDA: (
        # Placeholder: el texto real se construye en `_mensaje_repetir_salida`
        # con la última salida guardada (BOT-017).
        "La última vez saliste a... ¿repetimos?",
        "_repetir_salida",
        None,  # se monta aparte: botones Sí / No
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
# mismas que se le dan al LLM. BOT-020: esa cabecera abre con la clasificación
# y la probabilidad en %, y el resto del parte (jornada, factores, tabla
# horaria, recomendaciones) se monta aquí, en la plantilla determinista.


def _linea_comparacion_salida_anterior(
    result: dict, salida_anterior: dict | None
) -> str | None:
    """Criterio 5 (BOT-020): 'es un nivel más alto que la simulación anterior de...'.

    Punto de extensión de BOT-017: hoy el perfil no guarda la última salida, así
    que nadie llama con `salida_anterior`; cuando exista el dato, se pasará aquí
    con {'clase_final': int, 'actividad': str} y la comparación sale sola.
    """
    if not salida_anterior:
        return None
    actual = result.get("clase_final") or 0
    previa = salida_anterior.get("clase_final") or 0
    actividad = salida_anterior.get("actividad")
    if not actividad:
        return None
    if actual > previa:
        return f"Es un nivel más alto que la simulación anterior de {actividad}."
    if actual < previa:
        return f"Es un nivel más bajo que la simulación anterior de {actividad}."
    return None


def _resumen_jornada(perfil_u: dict) -> str | None:
    """Criterio 2 (BOT-020): la jornada en una línea, en lenguaje llano.

    '(actividad, horario, duración, intensidad, aclimatado, falta de sueño)'
    con los campos de los que haya dato. La ocupa quien la llama.
    """
    inicio = perfil_u.get("hora_inicio")
    duracion = perfil_u.get("duracion_actividad_h")
    if inicio is None or duracion is None:
        return None
    bits: list[str] = []
    ocp = perfil_u.get("ocupacion")
    if ocp in _OCUPACION_NIVELES:
        # "Construcción / albañilería (carga pesada, PPE, sol directo)" → "construcción"
        oficio = _OCUPACION_NIVELES[ocp][1].split("/")[0].split("(")[0].strip().lower()
        bits.append(f"trabajo de {oficio}")
    elif perfil_u.get("deporte"):
        bits.append(str(perfil_u["deporte"]))
    bits.append(f"{inicio:.0f}:00-{inicio + duracion:.0f}:00")
    bits.append(f"{duracion:g}h")
    if perfil_u.get("nivel_actividad"):
        bits.append(f"actividad {perfil_u['nivel_actividad']}")
    if perfil_u.get("aclimatado") is True:
        bits.append("aclimatado")
    elif perfil_u.get("aclimatado") is False:
        bits.append("no aclimatado")
    if "falta_sueno" in perfil_u:
        bits.append("con falta de sueño" if perfil_u["falta_sueno"] else "sin falta de sueño")
    return ", ".join(bits) or None


def _nombre_factor_llano(nombre: str) -> str:
    """El nombre técnico del pipeline a lenguaje llano para el parte (BOT-020)."""
    n = nombre.strip()
    low = n.lower()
    if low.startswith("trabajo "):
        # "trabajo Construcción / albañilería (carga pesada, PPE, sol directo)"
        oficio = n[len("trabajo "):].split("/")[0].split("(")[0].strip().lower()
        return f"trabajo de {oficio} al aire libre"
    if low.startswith("duración"):
        return f"la duración de {float(n.split()[1]):g}h"
    if low.startswith("hora inicio"):
        return "el horario que solapa con el pico de calor"
    if low == "no aclimatado":
        return "no estar aclimatado"
    if low.startswith("falta de sueño"):
        return "la falta de sueño"
    if low.startswith("edad "):
        return f"la edad de {n.split()[1]} años"
    if low.startswith("actividad "):
        return f"la actividad {n.split('actividad ', 1)[1]}"
    if low.startswith("sexo "):
        return "el sexo"
    if low.startswith("fiesta"):
        return "el consumo de alcohol reciente"
    if low.startswith("enfermedad reciente"):
        return "una enfermedad reciente"
    if low.startswith("grasa corporal"):
        return "la grasa corporal"
    if low.startswith("fatiga acumulada"):
        return "la fatiga acumulada"
    if low.startswith("uv "):
        return "la radiación UV"
    if low.startswith("aislamiento"):
        return "el aislamiento"
    return n


def _lineas_factores(result: dict) -> list[str]:
    """Criterio 3 (BOT-020): los factores con su multiplicador, de mayor a menor.

    Solo los que suben el riesgo (>1): los protectores (<1) no "pesan" y los
    neutros no aportan nada al parte.
    """
    calor = (result.get("perfil") or {}).get("calor") or {}
    candidatos = [
        f for f in calor.get("factores") or []
        if isinstance(f, dict) and isinstance(f.get("factor"), (int, float)) and f["factor"] > 1.0
    ]
    if not candidatos:
        return []
    ordenados = sorted(candidatos, key=lambda f: f["factor"], reverse=True)
    return [
        f"• {_nombre_factor_llano(f['nombre'])} (factor x{f['factor']})"
        for f in ordenados
    ]


def _formato_hora(hora) -> str:
    """8 → '8:00'; 8.5 → '8:30' (la curva puede traer puntos sub-horarios)."""
    h = float(hora)
    return f"{int(h)}:{int(round((h - int(h)) * 60)):02d}"


def _tabla_horaria(perfil_h: list[dict], perfil_u: dict) -> list[str]:
    """Criterio 4 (BOT-020): riesgo por hora de la jornada, inicio-pico-fin.

    La curva sale de `riesgo_horario_acumulado` (contrato intacto); aquí solo
    se eligen las filas: la primera de la ventana, la de riesgo máximo y la
    última.
    """
    inicio = perfil_u.get("hora_inicio")
    duracion = perfil_u.get("duracion_actividad_h")
    if not perfil_h or inicio is None or duracion is None:
        return []
    curva = riesgo_horario_acumulado(perfil_h, perfil_u)
    ventana = [c for c in curva if inicio <= c["hora"] < inicio + duracion]
    if not ventana:
        return []
    pico = max(ventana, key=lambda c: c["riesgo"])
    filas: list[tuple] = [(ventana[0], "inicio")]
    if pico["hora"] != ventana[0]["hora"]:
        filas.append((pico, "pico"))
    ultima = ventana[-1]
    if ultima["hora"] != pico["hora"] and ultima["hora"] != ventana[0]["hora"]:
        filas.append((ultima, "fin"))
    lineas = ["El riesgo por horas de la jornada:", "Hora | Riesgo | Heat Index"]
    for c, etiqueta in filas:
        lineas.append(f"{_formato_hora(c['hora'])} ({etiqueta}) | {c['riesgo']:.2f} | {c['hi']:.1f}°C")
    return lineas


def _lineas_franjas(perfil_h: list[dict], perfil_u: dict) -> list[str]:
    """BOT-012: franja de mayor riesgo del día y franja recomendada para la actividad.

    Criterio 2: reutiliza `riesgo_horario_acumulado`, `pico_riesgo_actividad`
    y `recomendar_horario` de personalizacion (los mismos que la web y el MCP);
    aquí solo se eligen las frases, no se recalcula el riesgo. Criterio 3: sin
    perfil horario se dice en vez de callarse o inventar una franja.
    """
    if not perfil_h:
        return ["No hay datos horarios para hoy: no puedo decir la franja de mayor riesgo ni la recomendada."]
    curva = riesgo_horario_acumulado(perfil_h, perfil_u)
    if not curva:
        return ["No hay datos horarios para hoy: no puedo decir la franja de mayor riesgo ni la recomendada."]
    lineas = []
    # Franja de mayor riesgo: la hora de la curva con más riesgo del día, con
    # el valor que ya da pico_riesgo_actividad para este perfil.
    pico_dia = max(curva, key=lambda c: c["riesgo"])
    riesgo_pico = pico_riesgo_actividad(curva, perfil_u) or pico_dia["riesgo"]
    lineas.append(
        f"⚠️ Franja de mayor riesgo del día: en torno a las "
        f"{_formato_hora(pico_dia['hora'])} (riesgo {riesgo_pico:.2f} de 1)"
    )
    rec = recomendar_horario(perfil_h, perfil_u)
    if rec and rec.get("hora_inicio") is not None:
        lineas.append(
            f"✅ Franja recomendada para la actividad: "
            f"{_formato_hora(rec['hora_inicio'])}-{_formato_hora(rec['hora_fin'])}"
        )
    return lineas


def _linea_recomendaciones(result: dict) -> str:
    """Criterio 4 (BOT-020): las recomendaciones de la herramienta tal cual."""
    nivel = result.get("clase_final_label") or ""
    cabecera = (
        f"Recomendaciones de la herramienta (nivel {nivel}, no las suavizo):"
        if nivel else "Recomendaciones de la herramienta (no las suavizo):"
    )
    recs = result.get("recomendaciones") or []
    if not recs:
        return f"{cabecera}\n{recomendacion_resumen(result)}"
    return cabecera + "\n" + "\n".join(f"{i}. {r}" for i, r in enumerate(recs, 1))


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


def _format_template(result: dict, lugar: str | None = None, salida_anterior: dict | None = None) -> str:
    """Respuesta sin LLM: plantilla fija con el parte completo.

    BOT-020: el parte abre con la clasificación y su probabilidad (criterio 1),
    compara con la salida anterior si la hay (criterio 5), resume la jornada
    (criterio 2), lista los factores con su multiplicador de mayor a menor
    (criterio 3) y cierra con la tabla horaria y las recomendaciones tal cual
    (criterio 4). BOT-012: tras la tabla añade la franja de mayor riesgo del
    día y la franja recomendada para la actividad. Las frases BOT-013 vienen
    de `lineas_parte`, las mismas que ve el modo LLM.
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

    # Cabecera (criterio 1) + frases BOT-013: compartidas con la vía LLM.
    cabecera, *explicacion = lineas_parte(result, lugar)
    bloque = [cabecera]

    # Criterio 5: comparación con la salida anterior, si la hay (BOT-017).
    comparacion = _linea_comparacion_salida_anterior(result, salida_anterior)
    if comparacion:
        bloque.append(comparacion)

    bloque.extend(explicacion)

    # Criterio 2: la jornada en una línea. Criterio 3: factores por peso.
    jornada = _resumen_jornada(perfil_u)
    factores = _lineas_factores(result)
    if jornada and factores:
        bloque.append(f"Con esta jornada ({jornada}), el riesgo lo marcan estos factores, de mayor a menor:")
        bloque.extend(factores)
    elif jornada:
        bloque.append(f"Con esta jornada ({jornada}).")
    elif factores:
        bloque.append("Factores que pesan, de mayor a menor:")
        bloque.extend(factores)

    # Criterio 4: tabla horaria y recomendaciones de la herramienta.
    bloque.extend(_tabla_horaria(perfil_h, perfil_u))
    # BOT-012: la franja de mayor riesgo del día y la recomendada para la
    # actividad (o el aviso de que no hay datos horarios).
    bloque.extend(_lineas_franjas(perfil_h, perfil_u))
    bloque.append(_linea_recomendaciones(result))

    bloque.append(f"🌡️ Temperatura prevista: {temp:.1f} °C")
    bloque.append(f"☀️ Índice UV (media): {_format_uv(uv)}")
    return "\n".join(bloque)


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

# HOST-001: plantilla determinista del chat de preguntas libres. Si el LLM
# remoto está caído o agotó su cuota, el chat libre no puede redactar una
# respuesta personalizada; en vez de un error visible se responde con esta
# plantilla, que remite al parte oficial (que ya es determinista).
CHAT_LIBRE_SIN_LLM = (
    "El modelo de redacción no responde ahora mismo (puede estar caído o "
    "haber agotado su cuota), así que te respondo sin él. El parte que te "
    "acabo de dar es la información de esta salida; puedes repasarlo o "
    "escribir /start para calcular otra."
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
    # BOT-019: la ubicación guardada del perfil permite darla por supuesta en
    # la salida deducida de la rutina (visible en la confirmación).
    if perfil.get("lat") is not None and perfil.get("lon") is not None:
        data["lat"] = perfil["lat"]
        data["lon"] = perfil["lon"]
        data["provincia"] = perfil.get("provincia") or "Madrid"
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
    # BOT-019: la ubicación de la salida se guarda en el perfil para no tener
    # que volver a preguntarla en salidas posteriores con rutina.
    if data.get("lat") is not None and data.get("lon") is not None:
        p["lat"] = data["lat"]
        p["lon"] = data["lon"]
        p["provincia"] = data.get("provincia")
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


def _etiqueta_deporte(deporte: str | None) -> str | None:
    """'Futbol MET 7' para un deporte con MET medido, o None si no lo es."""
    if not deporte:
        return None
    entrada = DEPORTE_MET.get(deporte)
    if not entrada:
        return None
    met, _ = entrada
    return f"{DEPORTES.get(deporte, deporte.capitalize())} MET {met:g}"


def _resumen_rutinas(rutinas: list[dict]) -> str:
    lineas = ["*Tus rutinas:*"]
    for r in rutinas:
        extra = ""
        etiqueta_ocp = _etiqueta_ocupacion(r.get("ocupacion"))
        if etiqueta_ocp:
            extra = f" ({etiqueta_ocp})"
        else:
            etiqueta_dep = _etiqueta_deporte(r.get("deporte"))
            if etiqueta_dep:
                extra = f" ({etiqueta_dep})"
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


# ── BOT-019: salida del día deducida de la rutina ──────────────────────────


def _rutina_de_hoy(chat_id: int) -> dict | None:
    """La única rutina del chat para hoy; None si no hay o hay varias.

    Con varias rutinas en el mismo día no se deduce nada: /start pregunta
    por UNA salida y elegir la primera sería arbitrario.
    """
    rutinas = _db.rutinas_por_dia(str(chat_id), datetime.now().isoweekday())
    return rutinas[0] if len(rutinas) == 1 else None


def _prefill_desde_rutina(data: dict, rutina: dict) -> None:
    """Rellena los datos del día desde una rutina (BOT-019).

    Mismo criterio que `_perfil_prediccion_desde_rutina` (aviso diario):
    - deporte → intensidad por su MET (el número del Compendium vale más que
      el adjetivo del usuario, como ya hace el formulario en DEPORTE);
    - trabajo → intensidad "ligera" (el peso lo pone la ocupación, hasta x2.7);
    - rutina genérica ("entreno") → "ligera" por defecto.
    `entrenado=True` porque una rutina semanal es, por definición, una
    actividad a la que se está acostumbrado.
    """
    data["hora_inicio"] = rutina["hora_inicio"]
    data["duracion_h"] = rutina["hora_fin"] - rutina["hora_inicio"]
    data["entrenado"] = True
    if rutina.get("deporte"):
        data["_por_trabajo"] = False
        data["ocupacion"] = None
        data["deporte"] = DEPORTES.get(rutina["deporte"], rutina["deporte"])
        nivel = nivel_actividad_de_deporte(rutina["deporte"])
        if nivel:
            data["nivel_actividad"] = nivel
    elif rutina.get("ocupacion"):
        data["_por_trabajo"] = True
        data["ocupacion"] = rutina["ocupacion"]
        data["deporte"] = None
        data["nivel_actividad"] = "ligera"
    else:
        data["_por_trabajo"] = False
        data["ocupacion"] = None
        data["deporte"] = None
        data["nivel_actividad"] = "ligera"


_DIA_DERIVADO = {
    "hora_inicio", "duracion_h", "nivel_actividad", "entrenado",
    "deporte", "ocupacion", "_por_trabajo", "_confirmar_dia_msg",
    "_repetir_salida_msg",
}


def _limpiar_dia_derivado(data: dict) -> None:
    """Quita lo deducido de la rutina para volver a preguntar el día completo."""
    for k in _DIA_DERIVADO:
        data.pop(k, None)


def _kb_confirmar_dia() -> list[list[dict]]:
    return [
        [{"text": "Sí, es correcto", "callback_data": "confirmar_si"}],
        [{"text": "Cambiar algo", "callback_data": "confirmar_no"}],
    ]


def _mensaje_confirmar_dia(data: dict, rutina: dict) -> str:
    """Resumen de lo dado por supuesto: visible y corregible (criterio BOT-019)."""
    lineas = [f"*Hoy toca*: {_etiqueta_rutina(rutina)}", "", "He dado por supuesto:"]
    if data.get("nivel_actividad"):
        lineas.append(f"• Intensidad: {data['nivel_actividad'].replace('_', ' ')}")
    if data.get("entrenado") is not None:
        lineas.append("• Estás acostumbrado: sí (viene de tu rutina semanal)")
    if data.get("duracion_h") is not None:
        lineas.append(f"• Duración: {data['duracion_h']:g} h")
    if data.get("hora_inicio") is not None:
        lineas.append(f"• Empieza: {_formato_hora(data['hora_inicio'])}")
    if data.get("deporte"):
        lineas.append(f"• Actividad: {data['deporte']}")
    if data.get("ocupacion"):
        etiqueta = _etiqueta_ocupacion(data["ocupacion"])
        lineas.append(f"• Salida de trabajo: {etiqueta or OCUPACIONES.get(data['ocupacion'], data['ocupacion'])}")
    if data.get("lat") is not None and data.get("lon") is not None:
        lineas.append(f"• Ubicación: {data.get('provincia') or f'{data['lat']:.2f}, {data['lon']:.2f}'}")
    lineas.append("")
    lineas.append("¿Es lo que vas a hacer?")
    return "\n".join(lineas)


def _guardar_ubicacion_perfil(chat_id: int, lat: float, lon: float, provincia: str) -> None:
    """Guarda la ubicación de la salida como ubicación del perfil (BOT-019).

    Así una salida posterior con rutina puede darla por supuesta (se muestra
    en la confirmación y se cambia con "Cambiar algo"). No crítico: si falla
    solo se pierde el atajo, nunca la predicción.
    """
    try:
        match = _db.buscar_por_telegram(str(chat_id))
        if match:
            _db.actualizar_perfil(match["id"], {"lat": lat, "lon": lon, "provincia": provincia})
    except Exception:
        logger.exception("No se pudo guardar la ubicación en el perfil de %s", chat_id)


# ── BOT-017: repetir la última salida guardada ─────────────────────────────


def _etiqueta_actividad_salida(data: dict) -> str | None:
    """Etiqueta legible de la salida para el parte y el resumen (BOT-017).

    'Correr' para un deporte; 'trabajo de campo' para una salida laboral;
    None si no hay actividad que nombrar.
    """
    if data.get("ocupacion") in _OCUPACION_NIVELES:
        # "Construcción / albañilería (carga pesada, PPE, sol directo)" → "campo"
        oficio = _OCUPACION_NIVELES[data["ocupacion"]][1].split("/")[0].split("(")[0].strip().lower()
        return f"trabajo de {oficio}"
    if data.get("deporte"):
        return str(data["deporte"])
    return None


def _prefill_desde_ultima_salida(data: dict, ultima: dict) -> None:
    """Rellena el día desde la última salida guardada (BOT-017).

    La intensidad y el trabajo/deporte se guardan tal cual se predijeron la
    última vez (el MET del deporte ya está dentro de `nivel_actividad`), así
    que repetir es idéntico salvo el tiempo, que `predict_ensemble` siempre
    toma de HOY.
    """
    data["hora_inicio"] = ultima.get("hora_inicio")
    data["duracion_h"] = ultima.get("duracion_h")
    data["nivel_actividad"] = ultima.get("nivel_actividad")
    data["entrenado"] = ultima.get("entrenado")
    if ultima.get("ocupacion"):
        data["_por_trabajo"] = True
        data["ocupacion"] = ultima["ocupacion"]
        data["deporte"] = None
    elif ultima.get("deporte"):
        data["_por_trabajo"] = False
        data["deporte"] = ultima["deporte"]
        data["ocupacion"] = None
    else:
        data["_por_trabajo"] = False
        data["deporte"] = None
        data["ocupacion"] = None
    if ultima.get("lat") is not None and ultima.get("lon") is not None:
        data["lat"] = ultima["lat"]
        data["lon"] = ultima["lon"]
        data["provincia"] = ultima.get("provincia") or "Madrid"


def _kb_repetir_salida() -> list[list[dict]]:
    return [
        [{"text": "Sí, repetir", "callback_data": "repetir_si"}],
        [{"text": "No, pregúntamelo", "callback_data": "repetir_no"}],
    ]


def _mensaje_repetir_salida(ultima: dict) -> str:
    """Resumen de la última salida guardada con la pregunta de repetirla."""
    lineas = ["*La última vez saliste a*:"]
    if ultima.get("actividad"):
        lineas.append(f"• Actividad: {ultima['actividad']}")
    if ultima.get("nivel_actividad"):
        lineas.append(f"• Intensidad: {ultima['nivel_actividad'].replace('_', ' ')}")
    if ultima.get("duracion_h") is not None:
        lineas.append(f"• Duración: {ultima['duracion_h']:g} h")
    if ultima.get("hora_inicio") is not None:
        lineas.append(f"• Empieza: {_formato_hora(ultima['hora_inicio'])}")
    if ultima.get("provincia"):
        lineas.append(f"• Ubicación: {ultima['provincia']}")
    lineas.append("")
    lineas.append("¿Vas a hacer lo mismo hoy?")
    return "\n".join(lineas)


def _guardar_ultima_salida(chat_id: int, data: dict, result: dict) -> None:
    """Guarda la salida de esta predicción como la última del perfil (BOT-017).

    Criterio 1: al terminar un /start el perfil recuerda qué salida fue
    (actividad u ocupación, intensidad, duración, hora y ubicación) para
    ofrecer repetirla en el próximo /start. La clase se guarda para que
    BOT-020 pueda comparar "es un nivel más alto que la simulación anterior".
    No crítico: si falla solo se pierde el atajo, nunca la predicción.
    """
    try:
        match = _db.buscar_por_telegram(str(chat_id))
        if not match:
            return
        salida: dict[str, Any] = {
            "actividad": _etiqueta_actividad_salida(data),
            "nivel_actividad": data.get("nivel_actividad"),
            "duracion_h": data.get("duracion_h"),
            "hora_inicio": data.get("hora_inicio"),
            "lat": data.get("lat"),
            "lon": data.get("lon"),
            "provincia": data.get("provincia"),
            "entrenado": data.get("entrenado"),
            "clase_final": result.get("clase_final"),
        }
        if data.get("ocupacion"):
            salida["ocupacion"] = data["ocupacion"]
        if data.get("deporte"):
            salida["deporte"] = data["deporte"]
        _db.actualizar_perfil(match["id"], {"ultima_salida": salida})
    except Exception:
        logger.exception("No se pudo guardar la última salida del perfil de %s", chat_id)


# ── BOT-021: frase libre que describe una salida ──────────────────────────

# Estados en los que el usuario puede describir la salida con una frase en vez
# de rellenar el formulario campo a campo. Son los que abren el "día" de /start.
_ESTADOS_FRASE_LIBRE = {Estado.ACTIVIDAD, Estado.CONFIRMAR_DIA, Estado.REPETIR_SALIDA}

# Campo de la salida que falta → estado del formulario que lo pregunta. Cuando
# una frase no lo dice todo, se hace SOLO esta pregunta, nunca el formulario.
_ESTADO_POR_CAMPO = {
    "actividad": Estado.ACTIVIDAD,
    "duracion_h": Estado.DURACION,
    "hora_inicio": Estado.HORA_INICIO,
    "ubicacion": Estado.UBICACION,
}

# Frases que remiten a la salida anterior / a la rutina en vez de describirla:
# 'como ayer', 'igual que el martes', 'como siempre'... El contexto (última
# salida de BOT-017 o rutina de BOT-019) ya está prefillado en `data` cuando el
# bot llega a CONFIRMAR_DIA o REPETIR_SALIDA, así que basta con no tocarlo.
_SENAL_REFERENCIA = re.compile(
    r"(?:como|igual que|lo mismo que|lo de)\s+"
    r"(?:ayer|siempre|de costumbre|todos los dias|todos los días|"
    r"el (?:lunes|martes|miercoles|miércoles|jueves|viernes|sabado|sábado|domingo))"
)

# Adjetivos de intensidad → nivel de actividad. Solo valen cuando la frase no
# nombra un deporte: si nombra deporte, el MET del Compendium manda (igual que
# en los botones del formulario).
_NIVEL_ADJETIVOS = {
    "muy intensa": "muy_intensa", "muy intenso": "muy_intensa", "extremo": "muy_intensa",
    "intensa": "intensa", "intenso": "intensa", "fuerte": "intensa",
    "moderada": "moderada", "moderado": "moderada", "normal": "moderada",
    "ligera": "ligera", "ligero": "ligera", "suave": "ligera",
    "tranquila": "ligera", "tranquilo": "ligera", "reposo": "reposo",
}


def _deporte_en_frase(t: str) -> str | None:
    """El deporte de DEPORTES mencionado en la frase (clave canónica), o None.

    Se buscan primero las etiquetas largas ('tenis individual', 'tenis dobles',
    'bici fuerte'...) para que 'voy a jugar tenis dobles' no caiga en 'tenis'.
    """
    etiquetas = sorted(
        ((clave, etiqueta.lower()) for clave, etiqueta in DEPORTES.items()),
        key=lambda kv: len(kv[1]),
        reverse=True,
    )
    for clave, etiqueta in etiquetas:
        if etiqueta in t or clave in t:
            return clave
    return None


def _nivel_en_frase(t: str) -> str | None:
    for adjetivo, nivel in sorted(_NIVEL_ADJETIVOS.items(), key=lambda kv: len(kv[0]), reverse=True):
        if adjetivo in t:
            return nivel
    return None


def _duracion_en_frase(t: str) -> float | None:
    """Duración que la frase menciona, en horas. '40 min' → 0.667, 'hora y media' → 1.5."""
    m = re.search(r"(\d+(?:[.,]\d+)?)\s*(?:horas?|h)\b", t)
    if m:
        return float(m.group(1).replace(",", "."))
    m = re.search(r"(\d+)\s*(?:min|minutos?|mins?)\b", t)
    if m:
        return int(m.group(1)) / 60
    if "hora y media" in t:
        return 1.5
    if re.search(r"\bmedia hora\b", t):
        return 0.5
    if re.search(r"\buna hora\b", t):
        return 1.0
    return None


def _hora_en_frase(t: str) -> int | None:
    """Hora de inicio que la frase menciona. 'esta tarde' → 17 (igual que /chat)."""
    m = re.search(r"a las?\s+(\d{1,2})(?::(\d{2}))?\b", t)
    if m and 0 <= int(m.group(1)) <= 23:
        return int(m.group(1))
    if re.search(r"\b(?:esta|por la)\s+tarde\b", t):
        return 17
    if re.search(r"\b(?:esta|por la|a la)\s+noche\b", t):
        return 21
    if "mediodia" in t or "mediodía" in t:
        return 13
    if re.search(r"\b(?:por la|esta)\s+(?:mañana|manana)\b", t):
        return 9
    return None


def _interpretar_salida_frase(texto: str) -> dict:
    """Lee de una frase libre los campos de la salida que se reconocen.

    Interpretación determinista (regex/plantillas): no depende del LLM. Devuelve
    solo las claves que la frase menciona — lo que no menciona lo aporta el
    contexto (última salida guardada o rutina de hoy, ya prefillados en `data`).
    `referencia` marca que la frase remite a la salida anterior ('como ayer',
    'igual que el martes', 'como siempre').
    """
    t = unicodedata.normalize("NFC", texto.lower())
    salida: dict[str, Any] = {}
    if _SENAL_REFERENCIA.search(t):
        salida["referencia"] = True
    deporte = _deporte_en_frase(t)
    if deporte:
        salida["deporte"] = deporte
    nivel = _nivel_en_frase(t)
    if nivel:
        salida["nivel_actividad"] = nivel
    duracion = _duracion_en_frase(t)
    if duracion is not None:
        salida["duracion_h"] = duracion
    hora = _hora_en_frase(t)
    if hora is not None:
        salida["hora_inicio"] = hora
    return salida


def _frase_describe_salida(salida: dict) -> bool:
    """¿La frase merece el atajo de BOT-021 o es un valor suelto del formulario?

    Un adjetivo suelto ('intensa') o un número ('2') son respuestas del
    formulario, no una frase de salida. La frase libre se reconoce cuando
    nombra la actividad, la duración, la hora o remite a una salida anterior.
    """
    if salida.get("referencia"):
        return True
    return bool(
        salida.get("deporte")
        or salida.get("duracion_h") is not None
        or salida.get("hora_inicio") is not None
    )


def _campos_salida_faltantes(data: dict) -> list[str]:
    """Campos de la salida sin los que no se puede predecir (BOT-021).

    En el orden del formulario. La actividad puede venir como intensidad o como
    deporte con MET propio; la ubicación viene del perfil, de la última salida
    o de la rutina. Solo faltan los que no tiene nadie.
    """
    faltan = []
    if not data.get("nivel_actividad"):
        faltan.append("actividad")
    if data.get("duracion_h") is None:
        faltan.append("duracion_h")
    if data.get("hora_inicio") is None:
        faltan.append("hora_inicio")
    if data.get("lat") is None or data.get("lon") is None:
        faltan.append("ubicacion")
    return faltan


def _avanzar_tras_campo(conv: dict, estado: Estado) -> None:
    """Avanza tras responder un campo del formulario (BOT-021).

    Si se está completando una frase libre se pregunta el siguiente campo que
    falte (uno a uno, nunca el formulario entero) o se predice directo cuando
    ya está todo. Si no, sigue el formulario normal.
    """
    data = conv["data"]
    if data.get("_frase_libre"):
        faltan = _campos_salida_faltantes(data)
        if faltan:
            conv["estado"] = _ESTADO_POR_CAMPO[faltan[0]]
            return
        data.pop("_frase_libre", None)
        conv["estado"] = Estado.DONE
        return
    conv["estado"] = _siguiente(estado, data)


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
            # BOT-017: la salida anterior guardada alimenta la comparación del
            # parte ("es un nivel más alto que la simulación anterior de...").
            conv["salida_anterior"] = perfil.get("ultima_salida") or None
            # BOT-019: si hay una única rutina hoy, la salida del día se deduce
            # (hora, duración, intensidad, trabajo/deporte) y solo se confirma.
            # El resto de casos sigue preguntando como siempre.
            rutina_hoy = _rutina_de_hoy(chat_id)
            if rutina_hoy:
                _prefill_desde_rutina(conv["data"], rutina_hoy)
                conv["estado"] = Estado.CONFIRMAR_DIA
                conv["data"]["_confirmar_dia_msg"] = _mensaje_confirmar_dia(conv["data"], rutina_hoy)
                return None  # el resumen + botones lo envía enviar_siguiente_pregunta
            # BOT-017: sin rutina hoy, si el perfil guarda la última salida se
            # ofrece repetirla antes de preguntar campo a campo. Una salida
            # sin hora (blob parcial) no cuenta: mejor el flujo normal.
            ultima_salida = perfil.get("ultima_salida")
            if ultima_salida and ultima_salida.get("hora_inicio") is not None:
                _prefill_desde_ultima_salida(conv["data"], ultima_salida)
                conv["estado"] = Estado.REPETIR_SALIDA
                conv["data"]["_repetir_salida_msg"] = _mensaje_repetir_salida(ultima_salida)
                return None  # el resumen + botones lo envía enviar_siguiente_pregunta
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
        return "Modo cambiado a *API externa* (Groq gpt-oss-20b)."
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
        # BOT-016: ni el trabajo ni el deporte se guardan directo. El trabajo
        # pregunta el tipo (ocupación, x1.0 a x2.7); el deporte (o el entreno
        # genérico) pregunta la actividad concreta, cuyo MET del Compendium
        # fija la intensidad mejor que el adjetivo que elegiría el usuario.
        conv["_rutina_pendiente"] = rutina
        if rutina["deporte"] or rutina["nombre"] in _NOMBRES_ENTRENAMIENTO:
            return PREGUNTA_TIPO_ACTIVIDAD
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
        return await _preguntar_al_rag(texto, conv, contexto, str(chat_id))

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
        except Exception:
            logger.exception("Error al guardar perfil")
        conv["estado"] = _siguiente(estado, data)  # → DONE
        return None

    if texto and texto.lower() == "saltar":
        data[FIELD_LABELS[estado][1]] = None
        _avanzar_tras_campo(conv, estado)
        return None

    # ── BOT-021: frase libre que describe la salida ──────────────────
    # En vez de contestar el formulario campo a campo, el usuario puede
    # escribir la salida en una frase ('voy al tenis como ayer', 'esta tarde
    # correr 40 min', 'igual que el martes'). La interpretación es
    # determinista (regex/plantillas, sin depender del LLM) y usa como
    # contexto la última salida guardada (BOT-017) o la rutina de hoy
    # (BOT-019), que ya están prefilladas en `data`. Si falta algo se hace
    # SOLO la pregunta del campo que falta, nunca el formulario entero.
    if estado in _ESTADOS_FRASE_LIBRE or data.get("_frase_libre"):
        salida = _interpretar_salida_frase(texto)
        if salida and (data.get("_frase_libre") or _frase_describe_salida(salida)):
            if salida.get("deporte"):
                data["_por_trabajo"] = False
                data["ocupacion"] = None
                data["deporte"] = DEPORTES.get(salida["deporte"], salida["deporte"])
                # El MET del deporte manda sobre cualquier adjetivo, igual
                # que en los botones del formulario.
                nivel = nivel_actividad_de_deporte(salida["deporte"])
                if nivel:
                    data["nivel_actividad"] = nivel
                elif salida.get("nivel_actividad"):
                    data["nivel_actividad"] = salida["nivel_actividad"]
            elif salida.get("nivel_actividad"):
                data["nivel_actividad"] = salida["nivel_actividad"]
            if salida.get("duracion_h") is not None:
                data["duracion_h"] = salida["duracion_h"]
            if salida.get("hora_inicio") is not None:
                data["hora_inicio"] = salida["hora_inicio"]
            faltan = _campos_salida_faltantes(data)
            if faltan:
                data["_frase_libre"] = True
                conv["estado"] = _ESTADO_POR_CAMPO[faltan[0]]
                return FIELD_LABELS[conv["estado"]][0]
            data.pop("_frase_libre", None)
            conv["estado"] = Estado.DONE
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
        _guardar_ubicacion_perfil(chat_id, data["lat"], data["lon"], data["provincia"])
        _avanzar_tras_campo(conv, estado)
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

    _avanzar_tras_campo(conv, estado)
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
    # BOT-019/BOT-017: CONFIRMAR_DIA y REPETIR_SALIDA solo se alcanzan por
    # asignación directa desde /start (hay rutina hoy / hay última salida
    # guardada). El resto de flujos los saltan sin tocar.
    if sig in (Estado.CONFIRMAR_DIA, Estado.REPETIR_SALIDA):
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

    # ── BOT-016: elegir la actividad deportiva de una rutina pendiente ──
    if callback_data.startswith("rutina_deporte_"):
        rutina = conv.get("_rutina_pendiente")
        if not rutina:
            return "No hay ninguna rutina pendiente. Usa /rutinas_anadir.", False
        deporte = callback_data[len("rutina_deporte_"):]
        if deporte not in DEPORTES:
            return "Actividad inválida.", False
        rutina["deporte"] = deporte
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

    # BOT-019: confirmar la salida deducida de la rutina. "Cambiar algo"
    # limpia lo deducido y vuelve a preguntar el día completo desde ACTIVIDAD,
    # sin reiniciar /start.
    if estado == Estado.CONFIRMAR_DIA:
        if callback_data == "confirmar_no":
            _limpiar_dia_derivado(data)
            conv["estado"] = Estado.ACTIVIDAD
            return None, False
        sig = _siguiente(estado, data)
        conv["estado"] = sig
        return None, sig == Estado.DONE

    # BOT-017: repetir la última salida guardada. "Sí" salta directo a la
    # predicción (criterio 3: no se vuelve a preguntar nada; el tiempo lo da
    # predict_ensemble de HOY). "No" limpia la salida repetida y sigue el
    # formulario exactamente como hoy, campo a campo.
    if estado == Estado.REPETIR_SALIDA:
        if callback_data == "repetir_no":
            _limpiar_dia_derivado(data)
            # La ubicación repetida tampoco cuenta: hoy, sin rutina, se
            # pregunta siempre aunque el perfil la tenga guardada.
            for k in ("lat", "lon", "provincia", "lugar"):
                data.pop(k, None)
            conv["estado"] = Estado.ACTIVIDAD
            return None, False
        conv["estado"] = Estado.DONE
        return None, True

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

    _avanzar_tras_campo(conv, estado)
    return None, conv["estado"] == Estado.DONE


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

    # BOT-017: al terminar un /start, el perfil recuerda esta salida para
    # ofrecer repetirla en el próximo /start.
    _guardar_ultima_salida(chat_id, data, result)

    # Respuesta según el modelo elegido
    modelo = conv.get("modelo", MODELO_DETERMINISTA)
    lugar = data.get("lugar")
    if modelo != MODELO_DETERMINISTA:
        # BOT-006: la recomendación post-predicción se redacta con el LLM
        # local (Ollama) si está disponible — el que ve el contexto real de
        # esta predicción y no cuesta tokens — y con la plantilla determinista
        # de BOT-005 si no lo hay. Se comprueba en el momento de predecir, no
        # al arrancar la conversación: Ollama puede haberse caído (o
        # levantado) entre medias. El LLM remoto (HOST-001) queda para el chat
        # libre y los comandos de modelo, no para el parte.
        st = check_ollama()
        if st.get("available"):
            config = LLMConfig(model=st.get("best_model") or MODELO_LOCAL)
            texto = await asyncio.to_thread(
                ask_con_perfil, perfil, result, config, lugar, str(chat_id)
            )
            if texto:
                logger.info("Respuesta redactada por %s", config.model)
            else:
                # Sin esta línea la degradación es invisible: el usuario ve la
                # plantilla y cree que el LLM está funcionando.
                logger.warning("%s no contestó; se responde con la plantilla", config.model)
        else:
            texto = None
            logger.info("Sin LLM local: respuesta con plantilla")
    else:
        texto = None
        logger.info("Modo determinista: respuesta con plantilla")
    if not texto:
        # BOT-020/BOT-017: con la salida anterior guardada, la plantilla
        # compara "es un nivel más alto que la simulación anterior de...".
        texto = _format_template(result, lugar, conv.get("salida_anterior"))

    return texto


def _contexto_parte_conversacion(conv: dict) -> str:
    """Contexto del chat abierto tras el parte (BOT-011).

    Lleva el parte entregado más los datos REALES de esa predicción
    (probabilidad y factores del canal dominante, ubicación) y pide al LLM una
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
        # BOT-014: el contexto solo lleva el canal dominante (el de mayor
        # probabilidad personalizada), no los dos: un parte de calor no mete
        # la probabilidad de frío ni sus factores, para que el LLM no mezcle
        # canales al responder. Reutiliza _canal_dominante de recomendaciones.
        canal = _canal_dominante(result)
        if canal is None:
            # Sin probabilidades de canal (dicts mínimos de test): se degrada
            # al canal que trae factores, como antes de BOT-014.
            canal = "calor" if calor.get("factores") else "frio"
        elif canal == "ninguno":
            # Ambos canales por debajo del umbral de relevancia: se muestra el
            # de mayor probabilidad, el que más se acerca a contar algo.
            canal = "calor" if (calor.get("prob_personalizada") or 0) >= (frio.get("prob_personalizada") or 0) else "frio"
        canal_data = calor if canal == "calor" else frio
        prob_canal = canal_data.get("prob_personalizada")
        factores = canal_data.get("factores") or []
        lineas = [
            f"Ubicación: {w.get('provincia') or '?'}",
            f"Clase de riesgo: {result.get('clase_final_label') or '?'}",
        ]
        if prob_canal is not None:
            lineas.append(f"Probabilidad personalizada ({canal}): {prob_canal:.0%}")
        if factores:
            nombres = []
            # BOT-014: ordenados por su coeficiente, de mayor a menor, y CON el
            # coeficiente (xN), para que el LLM pueda decir cuál pesa más.
            for f in sorted(
                factores,
                key=lambda f: f.get("factor")
                if isinstance(f, dict) and isinstance(f.get("factor"), (int, float))
                else -1.0,
                reverse=True,
            ):
                if isinstance(f, dict):
                    nombre = f.get("nombre", str(f))
                    coef = f.get("factor")
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


async def _preguntar_al_rag(
    texto: str, conv: dict, contexto: str | None = None, sesion_id: str = "default"
) -> str:
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
    res = await asyncio.to_thread(
        ask_with_rag, texto, 3, 3, config, contexto, perfil_usuario, sesion_id
    )
    # HOST-001: si el LLM no contesta (servicio caído o cuota agotada), se
    # responde con la plantilla determinista. Antes se devolvía un error
    # visible ("El LLM no respondió. Revisa que ... esté disponible"), que no
    # resolvía la duda y delataba el modelo interno; el parte ya entregado es
    # la información oficial de la salida.
    return res.get("answer") or CHAT_LIBRE_SIN_LLM


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


# Telegram rechaza con 400 los mensajes de más de 4096 caracteres.
MAX_TG_LEN = 4096

# BOT-022: aviso que se añade cuando Telegram sigue rechazando el mensaje y
# hay que recortarlo en vez de perder el update mudo.
AVISO_TG_RECORTE = "\n\n[mensaje recortado: Telegram rechazó el envío completo]"


def _partir_texto(texto: str, max_len: int = MAX_TG_LEN) -> list[str]:
    """Parte un texto en trozos de ≤ max_len para Telegram.

    Se corta por saltos de línea para no partir una frase por la mitad; una
    línea que ella sola supere el límite se corta a la fuerza. Los saltos de
    línea que separan trozos se conservan al final del trozo, así que el texto
    unido de todos los trozos es idéntico al original.
    """
    trozos: list[str] = []
    resto = texto
    while len(resto) > max_len:
        corte = resto.rfind("\n", 0, max_len)
        if corte == -1:
            # Línea única gigante: corte forzoso.
            trozos.append(resto[:max_len])
            resto = resto[max_len:]
            continue
        trozos.append(resto[: corte + 1])
        resto = resto[corte + 1 :]
    if resto:
        trozos.append(resto)
    return trozos


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
        # BOT-022: el reenvío en plano vive DENTRO del manejo de errores. Antes
        # estaba fuera del try: si Telegram volvía a responder 400 —p.ej. el
        # texto superaba los 4096 caracteres— la excepción subía por
        # _finalizar_parte hasta el polling_loop y el parte del 13-08 murió
        # así: el usuario no recibió nada. Ahora el segundo 400 se maneja: se
        # parte el texto en mensajes múltiples (o se recorta con aviso), el
        # update nunca se pierde mudo.
        try:
            await _tg("sendMessage", **payload)
        except httpx.HTTPStatusError as exc2:
            if exc2.response.status_code != 400:
                raise
            logger.warning(
                "Doble 400 al enviar a %s (%d caracteres); se parte el mensaje",
                chat_id,
                len(texto),
            )
            trozos = _partir_texto(texto)
            if len(trozos) == 1:
                # No era la longitud: recortar no lo arregla, pero el usuario
                # recibe el texto con un aviso en vez de un update muerto.
                trozos = [
                    f"{texto[: MAX_TG_LEN - len(AVISO_TG_RECORTE)]}{AVISO_TG_RECORTE}"
                ]
            for trozo in trozos:
                await _tg("sendMessage", **{**payload, "text": trozo})


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
        return
    # BOT-019: con la salida deducida de la rutina, la ubicación del perfil se
    # da por supuesta (se mostró en la confirmación y se cambia con "Cambiar
    # algo"). Sin rutina la ubicación se sigue preguntando siempre.
    if (
        estado == Estado.UBICACION
        and conv["data"].get("_confirmar_dia_msg")
        and conv["data"].get("lat") is not None
        and conv["data"].get("lon") is not None
    ):
        conv["estado"] = _siguiente(estado, conv["data"])
        _saltar_si_prellenado(conv)


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

    if estado == Estado.CONFIRMAR_DIA:
        msg = conv["data"].get("_confirmar_dia_msg") or FIELD_LABELS[estado][0]
        await enviar_mensaje(chat_id, msg, kb=_kb_confirmar_dia())
        return

    if estado == Estado.REPETIR_SALIDA:
        msg = conv["data"].get("_repetir_salida_msg") or FIELD_LABELS[estado][0]
        await enviar_mensaje(chat_id, msg, kb=_kb_repetir_salida())
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
    # SEC-001: ni lat/lon exactas ni chat_id completo al log — son datos
    # identificables. El chat_id lo tapa _OcultarChatId, las coordenadas no
    # deben aparecer ni para eso.
    logger.info("Ubicación recibida de %s", chat_id)

    sitio = provincia_desde_coords(lat, lon) or {}
    conv["data"]["lat"] = lat
    conv["data"]["lon"] = lon
    conv["data"]["provincia"] = sitio.get("provincia") or "Madrid"
    conv["data"]["lugar"] = sitio.get("nombre") or f"{lat:.4f}, {lon:.4f}"
    _guardar_ubicacion_perfil(chat_id, lat, lon, conv["data"]["provincia"])
    _avanzar_tras_campo(conv, Estado.UBICACION)
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

        # SEC-001: el texto del mensaje puede contener edad, medicación o
        # cualquier dato de salud en lenguaje natural; no se escribe al log,
        # solo la longitud (suficiente para depurar el flujo). El chat_id lo
        # tapa _OcultarChatId.
        logger.info("Mensaje de %s (%d caracteres)", chat_id, len(texto))

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
                # BOT-016: si la rutina queda pendiente la respuesta es la
                # pregunta del tipo — ocupación o actividad deportiva — y el
                # teclado es el que corresponde a esa rama.
                conv_actual = _conversaciones.get(chat_id, {})
                kb = None
                pendiente = conv_actual.get("_rutina_pendiente")
                if pendiente:
                    es_deporte = bool(pendiente.get("deporte")) or pendiente.get("nombre") in _NOMBRES_ENTRENAMIENTO
                    kb = _kb_tipo_deporte("rutina_deporte_") if es_deporte else _kb_tipo_trabajo("rutina_tipo_")
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
        # SEC-001: el callback_data son nombres de campos de salud (edit_edad,
        # enfermedad_reciente...); no se escribe. El chat_id lo tapa
        # _OcultarChatId.
        logger.info("Callback de %s", chat_id)

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
    """Tapa el token de Telegram y las claves *_API_KEY en cualquier linea del log.

    La Bot API lleva el token en la RUTA (api.telegram.org/bot<TOKEN>/sendMessage) y
    httpx registra la URL entera en INFO. El resultado era el token en claro en cada
    linea de logs/bot.log — un fichero que se rota y se queda en disco, y con el que
    cualquiera puede controlar el bot.

    HOST-001: al usar LLM remoto (Groq/Gemini), un error de litellm puede
    traer la URL de la llamada y, con Gemini, la key viaja en el query string
    (?key=...). El filtro se extiende a cualquier variable *_API_KEY del
    entorno para que las claves del LLM tampoco acaben en el log.
    """

    _MARCA = "bot<TOKEN_OCULTO>"

    def _secretos(self) -> list[str]:
        """Token de Telegram + valores de las claves *_API_KEY del entorno."""
        secretos: list[str] = []
        token = os.getenv("TELEGRAM_BOT_TOKEN")
        if token:
            secretos.append(token)
        for nombre, valor in os.environ.items():
            if nombre != "TELEGRAM_BOT_TOKEN" and nombre.endswith("_API_KEY") and valor:
                secretos.append(valor)
        return secretos

    def filter(self, record: logging.LogRecord) -> bool:
        for secreto in self._secretos():
            if not secreto:
                continue
            if isinstance(record.msg, str) and secreto in record.msg:
                record.msg = record.msg.replace(f"bot{secreto}", self._MARCA).replace(secreto, "<OCULTO>")
            if record.args:
                record.args = tuple(
                    a.replace(f"bot{secreto}", self._MARCA).replace(secreto, "<OCULTO>")
                    if isinstance(a, str) else a
                    for a in (record.args if isinstance(record.args, tuple) else (record.args,))
                )
        return True


class _OcultarChatId(logging.Filter):
    """Tapa los chat_id de Telegram (números de 6+ dígitos) en cualquier línea.

    SEC-001: el chat_id completo es un dato identificable — con él se puede
    emparejar una línea del log con la persona concreta. Se aplica junto a
    _OcultarToken en todos los handlers del bot: aunque una línea futura meta
    un chat_id sin querer, al disco nunca llega el número completo. Los chat_id
    llegan como int (msg['chat']['id']), así que no basta con mirar strings.
    """

    _RE = re.compile(r"\b\d{6,}\b")
    _REEMPLAZO = "<CHAT_ID>"

    def filter(self, record: logging.LogRecord) -> bool:
        if isinstance(record.msg, str):
            record.msg = self._RE.sub(self._REEMPLAZO, record.msg)
        if record.args:
            args = record.args if isinstance(record.args, tuple) else (record.args,)
            record.args = tuple(
                self._REEMPLAZO
                if isinstance(a, int) and self._RE.search(str(a))
                else self._RE.sub(self._REEMPLAZO, a)
                if isinstance(a, str) and self._RE.search(a)
                else a
                for a in args
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
    filtro_pii = _OcultarChatId()

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
        h.addFilter(filtro_pii)
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
