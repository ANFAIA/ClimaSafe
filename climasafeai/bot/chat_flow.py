"""
climasafeai.bot.chat_flow — cuestionario conversacional de /chat (CHAT-001).

El LLM hace aquí dos cosas y solo dos: leer lo que el usuario ya ha dicho para
rellenar la ficha, y redactar la pregunta de lo que falte. Todo lo que decide el
resultado queda fuera de su alcance:

- El **riesgo** lo calcula `predict_ensemble`, igual que /start y que la web. Si
  el texto redactado nombra una clase distinta a la del pipeline se tira
  (`clase_coherente`) y se responde con la plantilla.
- Las **coordenadas** salen de Nominatim o del botón de Telegram. Un modelo
  devuelve coordenadas plausibles inventadas y la predicción saldría con el
  tiempo de otro sitio sin que salte ninguna alarma.
- La **intensidad** sale del MET del Compendium, no del adjetivo que elija el
  modelo: por eso `actividad` es una lista cerrada (las mismas 14 de /start).

Los campos que se piden son un subconjunto de los 17 estados de /start —
conversar 17 preguntas es insufrible. Ver `CAMPOS_REQUERIDOS`.
"""

from __future__ import annotations

import json
import logging
import unicodedata
from typing import Any

from climasafeai.bot.geocoding import buscar_lugar
from climasafeai.features.personalizacion import (
    DEPORTE_MET,
    nivel_actividad_de_deporte,
)
from climasafeai.llm.rag_qwen import LLMConfig, _chat_litellm
from climasafeai.models.ensemble import CLASES

logger = logging.getLogger(__name__)

# Lo mínimo sin lo que la predicción sería una invención. El resto de campos de
# /start (grasa, fototipo, comorbilidades, fármacos, situación social,
# aclimatación, entrenamiento, ocupación) NO se preguntan en el chat: se cogen
# del perfil guardado si lo hay, y si no se quedan en el valor por defecto que
# ya usa `ejecutar_prediccion`. Preguntarlos uno a uno conversando cuesta 17
# turnos y hace que nadie termine el chat.
CAMPOS_REQUERIDOS = ("lugar", "edad", "sexo", "actividad", "duracion_h", "hora_inicio")

# Cómo se pide cada campo si el LLM no propone nada utilizable.
PREGUNTAS: dict[str, str] = {
    "lugar": "¿dónde vas a estar? (dime el sitio, o mándame tu ubicación)",
    "edad": "¿cuántos años tienes?",
    "sexo": "¿eres hombre o mujer?",
    "actividad": "¿qué vas a hacer? (pasear, caminar, senderismo, correr, bici, fútbol, tenis…)",
    "duracion_h": "¿cuántas horas vas a estar fuera?",
    "hora_inicio": "¿a qué hora empiezas?",
}

# Se piden como mucho dos cosas por mensaje: seis preguntas de golpe no las
# contesta nadie.
_MAX_PREGUNTAS_POR_MENSAJE = 2

AVISO_LUGAR_NO_ENCONTRADO = (
    "No he encontrado ese sitio. Prueba con otro nombre (ej: «Aldán, Pontevedra») "
    "o mándame tu ubicación con el botón."
)

# Señales de que el usuario quiere saber si puede salir a hacer algo, aunque el
# LLM no lo capte. Un modelo pequeño clasifica "voy a salir a pasear" como
# consulta y el bot se va al RAG sin pedir ningún dato: la intención es de vida o
# muerte aquí, así que no se deja solo en manos del prompt.
_SENALES_ACTIVIDAD_FUTURA = (
    "salir", "salgo", "pase", "pasear", "paseo", "caminar", "camino",
    "correr", "corro", "bici", "bicicleta", "senderismo", "montaña", "ruta",
    "trabajar", "trabajo", "entrenar", "entreno", "jugar", "juego", "futbol",
    "tenis", "nadar", "piscina", "playa", "hacer deporte", "voy a", "vamos a",
    "quiero hacer", "pienso ir", "voy a salir",
)

# Palabras que el LLM confunde con sitios ("lugar": "paseo") y que Nominatim sí
# geocodificaría como una calle cualquiera: inventarse una ubicación da una
# predicción con el tiempo de otro sitio sin que salte ninguna alarma.
_SENALES_LUGAR_FALSO = (
    "paseo", "pasear", "caminar", "correr", "bici", "deporte", "trabajo",
    "entrenamiento", "senderismo", "excursion", "ruta", "playa", "montaña",
)


def _es_actividad_futura(texto: str) -> bool:
    """¿El mensaje describe salir a hacer algo? Decisión determinista.

    La devuelve True aunque el LLM diga consulta: la intención de riesgo se
    fuerza por señales léxicas y no se discute con el modelo.
    """
    t = unicodedata.normalize("NFC", texto.lower())
    for senal in _SENALES_ACTIVIDAD_FUTURA:
        if senal in t:
            return True
    return False


SYSTEM_EXTRACCION = """\
Eres el recepcionista de ClimaSafeAI. Tu unico trabajo es rellenar una ficha con
lo que el usuario ya ha dicho y preguntar con naturalidad lo que falte.
NUNCA calculas, estimas ni mencionas el riesgo: de eso se encarga el modelo.

Devuelve SOLO un objeto JSON, sin texto alrededor, con estas claves:
  intencion: "riesgo" si el usuario quiere saber si puede salir o hacer algo;
             "consulta" si solo pregunta por un concepto.
  lugar: el sitio TAL CUAL lo escribio el usuario, o null. Jamas inventes un
         sitio ni unas coordenadas.
  edad: entero, o null.
  sexo: "hombre", "mujer" o null.
  actividad: la mas parecida de esta lista cerrada, o null si no dice ninguna:
             {deportes}
  duracion_h: horas que estara fuera (numero), o null.
  hora_inicio: hora de inicio de 0 a 23. "por la manana"=9, "al mediodia"=13,
               "por la tarde"=17, "por la noche"=21. null si no dice nada.
  faltan: lista con los nombres de los campos que SIGUEN sin valor despues de
          contar los datos ya conocidos.
  pregunta: una sola frase, cercana, en espanol y sin listas, pidiendo como
            mucho dos campos de `faltan`. Cadena vacia si no falta nada.

Si un dato no lo ha dicho el usuario y no esta en los datos conocidos, es null.
No rellenes huecos por tu cuenta."""


def _system_extraccion() -> str:
    return SYSTEM_EXTRACCION.format(deportes=", ".join(sorted(DEPORTE_MET)))


# ── Estado de la ficha ─────────────────────────────────────────────────────


def campos_que_faltan(data: dict) -> list[str]:
    """Campos requeridos que siguen sin valor, en el orden en que se piden.

    Función pura: es la que garantiza que no se vuelve a preguntar lo que el
    usuario ya dijo, sin depender de que el LLM se acuerde.
    """
    faltan = []
    for campo in CAMPOS_REQUERIDOS:
        if campo == "lugar":
            if data.get("lat") is None or data.get("lon") is None:
                faltan.append(campo)
        elif campo == "actividad":
            if not data.get("nivel_actividad"):
                faltan.append(campo)
        elif data.get(campo) is None:
            faltan.append(campo)
    return faltan


# ── Normalización de lo que devuelve el LLM ────────────────────────────────


def _entero(valor: Any) -> int | None:
    try:
        return int(float(str(valor).strip()))
    except (TypeError, ValueError):
        return None


def _numero(valor: Any) -> float | None:
    try:
        return float(str(valor).replace(",", ".").strip())
    except (TypeError, ValueError):
        return None


def _es_lugar_falso(texto_lugar: str) -> bool:
    """¿El "lugar" extraído es en realidad una actividad?

    El LLM pequeño escribe `"lugar": "paseo"` cuando el usuario dice "salir a
    pasear". Si ese texto llegara a Nominatim se geocodificaría como una calle
    cualquiera llamada Paseo y la predicción saldría con el tiempo de otro
    sitio. Se descarta: vale más volver a preguntar el sitio.
    """
    t = unicodedata.normalize("NFC", texto_lugar.strip().lower())
    if t in DEPORTE_MET:
        return True
    return any(t == s or t.startswith(s + " ") or f" {s} " in f" {t} "
               for s in _SENALES_LUGAR_FALSO)


def normalizar_slots(bruto: dict) -> dict:
    """Se queda solo con los valores que pasan las validaciones de /start.

    Lo que el LLM devuelva fuera de rango (edad 500, hora 40, un deporte que no
    está en el Compendium) se descarta en silencio: vale más volver a preguntar
    que meter un dato inventado en el modelo.
    """
    out: dict[str, Any] = {}
    if not isinstance(bruto, dict):
        return out

    lugar = bruto.get("lugar")
    if (
        isinstance(lugar, str)
        and lugar.strip()
        and not _es_lugar_falso(lugar)
    ):
        out["lugar"] = lugar.strip()

    edad = _entero(bruto.get("edad"))
    if edad is not None and 1 <= edad <= 120:
        out["edad"] = edad

    sexo = bruto.get("sexo")
    if isinstance(sexo, str) and sexo.strip().lower() in ("hombre", "mujer"):
        out["sexo"] = sexo.strip().lower()

    actividad = bruto.get("actividad")
    if isinstance(actividad, str) and actividad.strip().lower() in DEPORTE_MET:
        out["actividad"] = actividad.strip().lower()

    duracion = _numero(bruto.get("duracion_h"))
    if duracion is not None and 0 < duracion <= 24:
        out["duracion_h"] = duracion

    hora = _entero(bruto.get("hora_inicio"))
    if hora is not None and 0 <= hora <= 23:
        out["hora_inicio"] = hora

    return out


def aplicar_slots(data: dict, slots: dict) -> str | None:
    """Vuelca los slots en `data` con las mismas claves que usa /start.

    Devuelve un aviso para el usuario si algo no se pudo aplicar (hoy solo el
    lugar, cuando Nominatim no lo encuentra), o None si todo fue bien.
    """
    if slots.get("edad") is not None:
        data["edad"] = slots["edad"]
    if slots.get("sexo"):
        data["sexo"] = slots["sexo"]
    if slots.get("duracion_h") is not None:
        data["duracion_h"] = slots["duracion_h"]
    if slots.get("hora_inicio") is not None:
        data["hora_inicio"] = slots["hora_inicio"]

    if slots.get("actividad"):
        clave = slots["actividad"]
        data["deporte"] = clave
        # El MET manda sobre cualquier adjetivo, igual que en los botones.
        data["nivel_actividad"] = nivel_actividad_de_deporte(clave)

    texto_lugar = slots.get("lugar")
    if texto_lugar and texto_lugar != data.get("_lugar_txt"):
        # Solo se geocodifica cuando el texto cambia: el extractor repite el
        # lugar en cada turno y Nominatim admite una petición por segundo.
        data["_lugar_txt"] = texto_lugar
        lugar = buscar_lugar(texto_lugar)
        if lugar is None:
            data.pop("_lugar_txt", None)
            return AVISO_LUGAR_NO_ENCONTRADO
        data["lat"] = lugar["lat"]
        data["lon"] = lugar["lon"]
        data["provincia"] = lugar["provincia"] or "Madrid"
        data["lugar"] = lugar["nombre"]
    return None


# ── Llamada al LLM ─────────────────────────────────────────────────────────


def _parse_json(raw: str | None) -> dict:
    """Saca el objeto JSON de la respuesta del LLM, con o sin ```fences```."""
    if not raw:
        return {}
    texto = raw.strip()
    inicio, fin = texto.find("{"), texto.rfind("}")
    if inicio == -1 or fin <= inicio:
        return {}
    try:
        cargado = json.loads(texto[inicio:fin + 1])
    except json.JSONDecodeError:
        # SEC-001: el texto del LLM puede repetir los datos de salud del
        # usuario; no se escribe, solo la longitud.
        logger.warning("El extractor no devolvió JSON válido (texto de %d caracteres)", len(texto))
        return {}
    return cargado if isinstance(cargado, dict) else {}


def _datos_conocidos(data: dict) -> dict:
    """Lo que ya está en la ficha, para que el LLM no lo vuelva a preguntar."""
    conocidos: dict[str, Any] = {}
    if data.get("lugar"):
        conocidos["lugar"] = data["lugar"]
    for campo in ("edad", "sexo", "duracion_h", "hora_inicio"):
        if data.get(campo) is not None:
            conocidos[campo] = data[campo]
    if data.get("deporte"):
        conocidos["actividad"] = data["deporte"]
    return conocidos


def extraer(texto: str, data: dict, config: LLMConfig) -> dict:
    """Pasa el mensaje por el LLM y devuelve intención, slots y pregunta.

    Es una única llamada por turno: extraer y redactar la siguiente pregunta se
    piden a la vez para no pagar dos veces el prompt.
    """
    user = (
        f"Datos conocidos: {json.dumps(_datos_conocidos(data), ensure_ascii=False)}\n"
        f'Mensaje del usuario: "{texto}"'
    )
    bruto = _parse_json(_chat_litellm(
        [{"role": "system", "content": _system_extraccion()},
         {"role": "user", "content": user}],
        config,
    ))
    faltan_llm = bruto.get("faltan")
    pregunta = bruto.get("pregunta")
    return {
        "intencion": "riesgo" if _es_actividad_futura(texto)
                     else ("consulta" if bruto.get("intencion") == "consulta" else "riesgo"),
        "slots": normalizar_slots(bruto),
        "faltan_llm": [str(f) for f in faltan_llm] if isinstance(faltan_llm, list) else None,
        "pregunta": pregunta.strip() if isinstance(pregunta, str) else "",
    }


# ── Redacción de la siguiente pregunta ─────────────────────────────────────


def _pregunta_plantilla(faltan: list[str]) -> str:
    pendientes = [PREGUNTAS[c] for c in faltan[:_MAX_PREGUNTAS_POR_MENSAJE]]
    if not pendientes:
        return "Ya tengo todo lo que necesito. Cuéntame qué vas a hacer y dónde."
    if len(pendientes) == 1:
        return f"Para calcularlo me falta un dato: {pendientes[0]}"
    return (f"Para calcularlo me faltan dos cosas: {pendientes[0]} "
            f"Y también {pendientes[1]}")


def mensaje_de_pregunta(
    faltan: list[str],
    pregunta_llm: str,
    faltan_llm: list[str] | None,
    aviso: str | None = None,
) -> str:
    """Texto que se le manda al usuario cuando la ficha aún no está completa.

    La frase del LLM solo se usa si su idea de lo que falta coincide con la
    nuestra. Si no coincide, va a preguntar por algo que el usuario ya dijo (o a
    dar por hecho algo que no dijo), así que se descarta y se usa la plantilla.
    """
    if faltan_llm is not None and set(faltan_llm) == set(faltan) and pregunta_llm:
        cuerpo = pregunta_llm
    else:
        cuerpo = _pregunta_plantilla(faltan)
    return f"{aviso}\n\n{cuerpo}" if aviso else cuerpo


# ── El LLM redacta, pero no decide la clase ────────────────────────────────


def _sin_tildes(texto: str) -> str:
    return "".join(
        c for c in unicodedata.normalize("NFD", texto.upper())
        if unicodedata.category(c) != "Mn"
    )


def clase_coherente(texto: str | None, clase_pipeline: str) -> bool:
    """¿El texto redactado dice la misma clase de riesgo que `predict_ensemble`?

    El LLM redacta el parte, pero la clase no es suya: si nombra otra distinta —
    o no nombra ninguna— el texto no vale y el bot responde con la plantilla, que
    sale del `result` del pipeline.
    """
    if not texto:
        return False
    plano = _sin_tildes(texto)
    nombradas = {_sin_tildes(c) for c in CLASES if _sin_tildes(c) in plano}
    return nombradas == {_sin_tildes(clase_pipeline)}
