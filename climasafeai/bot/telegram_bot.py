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
import signal
import sys
import textwrap
from enum import Enum, auto

from climasafeai.bot.geocoding import buscar_lugar, provincia_desde_coords
from climasafeai.features.personalizacion import nivel_actividad_de_deporte
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
OCUPACIONES = {
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
    # Fuera del flujo de formulario: chat libre con LLM
    CHAT = auto()


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


def _kb_tipo_trabajo() -> list[list[dict]]:
    return [[{"text": v, "callback_data": k}] for k, v in OCUPACIONES.items()]


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


def _kb_salir_chat() -> list[list[dict]]:
    return [[{"text": "Salir del chat", "callback_data": "__salir_chat__"}]]


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
    Estado.EDAD: ("¿Cuántos años tienes?", "edad", None),
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

RESPONSE_TEMPLATE = textwrap.dedent("""\
    {ubicacion} — Riesgo {clase} ({prob:.0%})

    🟡 Nivel de riesgo: {clase} ({prob:.0%})
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
    clase = result.get("clase_final_label", "?")
    prob = result.get("perfil", {}).get("calor", {}).get("prob_personalizada") or 0

    return RESPONSE_TEMPLATE.format(
        ubicacion=lugar or w.get("provincia") or "?",
        clase=clase,
        prob=prob,
        temp=temp,
        uv=_format_uv(uv),
        recomendacion=recomendacion_resumen(result),
    )


# ── Lógica del bot (stateless por conversación) ───────────────────────────

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
        # ¿Tiene perfil guardado?
        match = _db.buscar_por_telegram(str(chat_id))
        if match:
            perfil = _db.obtener_perfil(match["id"])
            conv["data"].update(_perfil_a_data(perfil))
            conv["estado"] = Estado.ACTIVIDAD  # saltar personales
            alias = perfil.get("alias") or match.get("alias", "")
            return f"Hola de nuevo, {alias}! Se cargaron tus datos previos. Solo las preguntas del dia:"
        conv["estado"] = Estado.SEXO
        return None  # quien llama debe enviar el campo siguiente

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

    if texto.startswith("/chat"):
        # Entrar en modo chat libre
        if conv["modelo"] == MODELO_DETERMINISTA:
            return (
                "El modo chat requiere un LLM activo. "
                "Cambia a /qwen o /api primero."
            )
        conv["estado"] = Estado.CHAT
        return (
            "Modo chat activado. Pregúntame sobre riesgo térmico, "
            "factores, recomendaciones…\n"
            "/start para volver al formulario."
        )

    if texto.startswith("/help"):
        return (
            "Comandos:\n"
            "/start — Formulario de predicción\n"
            "/chat — Chatear con el LLM\n"
            "/model — Ver / cambiar modelo\n"
            "  /qwen — Qwen local  (1.5B CPU / 7B GPU)\n"
            "  /api — API externa  (Groq)\n"
            "  /determinista — Sin LLM (plantilla)\n"
            "/help — Esto"
        )

    # ── Modo chat libre ────────────────────────────────────────────────
    if estado == Estado.CHAT:
        if texto.lower() in ("salir", "exit"):
            conv["estado"] = Estado.IDLE
            return "Saliendo del chat. /start para el formulario."
        # Responder con LLM + RAG (o raw si el modelo no es Ollama)
        config = LLMConfig(model=conv["modelo"])
        res = ask_with_rag(texto, k_factores=3, k_docs=3, config=config)
        ans = res.get("answer")
        if ans:
            return ans
        return f"El LLM no respondió. Revisa que {conv['modelo']} esté disponible."

    if estado == Estado.IDLE:
        return "Envía /start para comenzar."

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
    if campo in ("edad", "duracion_h", "hora_inicio", "porcentaje_grasa"):
        try:
            if campo in ("edad",):
                v = int(texto.strip()) if texto else None
                if v is not None and (v < 1 or v > 120):
                    return "La edad debe estar entre 1 y 120."
            elif campo in ("duracion_h",):
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

    estado = conv["estado"]
    data = conv["data"]
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
        texto = await ejecutar_prediccion(chat_id)
        await enviar_mensaje(chat_id, texto)
        # Limpiar conversación
        _conversaciones.pop(chat_id, None)
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
                return
            logger.error("Error HTTP %s: %s", exc.response.status_code, exc)
            await asyncio.sleep(5)
        except httpx.ConnectError:
            logger.warning("Sin conexión. Reintentando en 30s...")
            await asyncio.sleep(30)
        except Exception as exc:
            logger.exception("Error en polling loop: %s", exc)
            await asyncio.sleep(10)


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

        # Salir del chat
        if data == "__salir_chat__":
            conv_post = _conversaciones.get(chat_id)
            if conv_post:
                conv_post["estado"] = Estado.IDLE
            try:
                await _tg("answerCallbackQuery", {"callback_query_id": cb["id"]})
            except Exception:
                pass
            await enviar_mensaje(chat_id, "Saliendo del chat. /start para el formulario.")
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
            await enviar_mensaje(chat_id, texto_respuesta)
        if es_final:
            texto_final = await ejecutar_prediccion(chat_id)
            await enviar_mensaje(chat_id, texto_final)
            _conversaciones.pop(chat_id, None)
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
    """Logging a consola + archivo rotativo en logs/bot.log.

    Idempotente: `run_bot.sh` reinicia el proceso en bucle y, sin esta guarda, cada
    arranque anadia otro par de handlers al root y cada linea salia repetida.
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

    # Consola
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(formatter)

    # Archivo rotativo (5 MB × 3 backups)
    file_handler = RotatingFileHandler(
        log_dir / "bot.log", maxBytes=5 * 1024 * 1024, backupCount=3
    )
    file_handler.setFormatter(formatter)

    for h in (console, file_handler):
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
        loop.run_until_complete(polling_loop())
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
