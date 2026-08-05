# ClimaSafeAI — RAG + LLM unificado (LiteLLM)
#
# Integra la búsqueda semántica en sqlite-vec (factores + documentación)
# con cualquier LLM via LiteLLM: local (Ollama) o remoto (Groq, OpenAI, Gemini…).
# El modelo se elige con un string: "ollama/qwen2.5:1.5b", "groq/llama-3.3-70b", etc.
#
# Tres modos de servicio:
#   1. LLM fine-tuneado → (futuro)
#   2. LLM + RAG        → ask_with_rag()  ← principal
#   3. Determinista     → bot sin LLM (fallback)

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import litellm

from climasafeai.db.manager import DBManager
from climasafeai.features.personalizacion import _OCUPACION_NIVELES, recomendar_horario
from climasafeai.models.recomendaciones import recomendacion_resumen

logger = logging.getLogger(__name__)

# ── Constantes de modelo ────────────────────────────────────────────────

# Strings LiteLLM: "proveedor/nombre_modelo"
MODELO_LOCAL_CPU = "ollama/qwen2.5:1.5b"
MODELO_LOCAL_QWEN3 = "ollama/qwen3:1.7b"
MODELO_LOCAL_GPU = "ollama/qwen2.5:7b"
MODELO_FINE_TUNED = "ollama/qwen2.5:climasafe"
MODELO_API_DEFECTO = "groq/llama-3.3-70b-versatile"


@dataclass
class LLMConfig:
    """Configuración del LLM (funciona con cualquier proveedor LiteLLM).

    El modelo sigue el formato de LiteLLM:
      - Local Ollama:  "ollama/qwen2.5:1.5b"
      - Groq:          "groq/llama-3.3-70b-versatile"
      - OpenAI:        "gpt-4o"
      - Gemini:        "gemini/gemini-1.5-flash"
    """
    model: str = MODELO_LOCAL_CPU
    temperature: float = 0.3
    max_tokens: int = 1024

    @classmethod
    def desde_modelo(cls, model: str) -> "LLMConfig":
        """Crea una config para un modelo concreto."""
        return cls(model=model)

    @classmethod
    def mejor_disponible(cls) -> "LLMConfig":
        """Detecta el mejor modelo Ollama disponible.

        El orden no es por tamaño, es por lo que midió el benchmark de LLM-003
        (4 modelos × 100 ejemplos de data/llm/val.jsonl, corrida definitiva):

            modelo             clase  formato  inventa cifras  error del índice
            qwen3:1.7b          38%     100%        13%              0.297
            qwen2.5:1.5b        32%     100%       100%              7.098

        qwen2.5:1.5b se inventa alguna cifra en TODAS las respuestas y da
        índices fuera del rango 0-1. En un asistente de salud eso no es "peor",
        es inservible, así que qwen3:1.7b va por delante pese a pesar lo mismo.
        """
        modelos = _modelos_ollama()
        for candidate in [MODELO_FINE_TUNED, MODELO_LOCAL_GPU, MODELO_LOCAL_QWEN3, MODELO_LOCAL_CPU]:
            # fine-tuned: "ollama/qwen2.5:climasafe" → Ollama lo ve como "qwen2.5:climasafe"
            nombre_corto = candidate.split("/", 1)[1] if "/" in candidate else candidate
            if nombre_corto in modelos:
                return cls(model=candidate)
        return cls(model=MODELO_LOCAL_CPU)


# ── Consultas LiteLLM ──────────────────────────────────────────────────


def _chat_litellm(
    messages: list[dict[str, str]],
    config: LLMConfig,
) -> str | None:
    """Envía un chat a cualquier LLM via LiteLLM.

    LiteLLM normaliza la API de 100+ proveedores (Ollama, Groq, OpenAI…).
    El proveedor se deduce del prefijo del model name.
    """
    try:
        resp = litellm.completion(
            model=config.model,
            messages=messages,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
        )
        # El coste de una conversación solo se puede comparar si queda escrito:
        # `usage` viene del proveedor, no es una estimación nuestra.
        usage = getattr(resp, "usage", None)
        if usage is not None:
            logger.info(
                "tokens %s: %s prompt + %s completion = %s",
                config.model,
                getattr(usage, "prompt_tokens", "?"),
                getattr(usage, "completion_tokens", "?"),
                getattr(usage, "total_tokens", "?"),
            )
        return resp.choices[0].message.content.strip()
    except Exception as exc:
        logger.error("Error en LiteLLM (%s): %s", config.model, exc)
        return None


def _modelos_ollama() -> list[str]:
    """Lista los modelos disponibles en Ollama (vacía si no responde)."""
    try:
        import httpx
        r = httpx.get("http://localhost:11434/api/tags", timeout=5)
        return [m["name"] for m in r.json().get("models", [])]
    except Exception:
        return []


# ── Prompts ─────────────────────────────────────────────────────────────

SYSTEM_RAG = """\
Eres un asistente experto en riesgo térmico (calor y frío) para ClimaSafeAI.

Tu función es responder preguntas basándote EXCLUSIVAMENTE en el contexto
proporcionado abajo, que incluye:
- **Factores de riesgo** extraídos de la literatura científica (con DOI).
- **Documentación del proyecto** (arquitectura, modelos, decisiones técnicas).

NORMAS:
1. No inventes información que no esté en el contexto.
2. Si el contexto no es suficiente para responder, dilo claramente.
3. Cita las fuentes: para factores indica el nombre del factor y su categoría
   (calor/frío); para documentación indica el documento y la sección.
4. Cuando cites una fuente, usa el formato: [Fuente: <nombre>].
5. Responde en español, directo, sin rodeos.
6. Si la pregunta no está relacionada con riesgo térmico, indica
   educadamente que solo respondes sobre clima y salud térmica.
7. Separa conceptos de calor y frío cuando ambos apliquen.
8. Menciona valores o umbrales concretos si aparecen en el contexto.
9. Si el contexto incluye una sección "SITUACIÓN DEL USUARIO" con una persona
   concreta (su parte, sus factores, su ocupación), adapta la respuesta a esa
   situación y no repitas consejos genéricos de los documentos que no apliquen
   (p. ej. "reduce la exposición en interiores" para alguien que trabaja al
   aire libre)."""

SYSTEM_RAW = """\
Eres un asistente experto en riesgo térmico (calor y frío) para ClimaSafeAI,
un sistema de predicción de riesgo personalizado.

Responde preguntas sobre calor, frío, riesgos para la salud, factores de
vulnerabilidad y recomendaciones. Usa tus conocimientos pero sé cauto: si no
estás seguro de un dato, dilo.

Responde en español, directo y conciso."""

# `ask_con_perfil` iba sin system prompt: solo el mensaje de usuario. Con
# qwen2.5 colaba, pero qwen3:1.7b redactaba el parte de Pontevedra en portugués
# —"Mantenha-se hidratado, use protector solar SPF 30+ e evite exposição
# prolongada"— porque el topónimo gallego y el español comparten espacio con el
# portugués en un modelo pequeño y multilingüe. La instrucción de idioma tiene
# que ser explícita y estar en el system, no confiada al contexto.
SYSTEM_PARTE = """\
Eres ClimaSafeAI, un sistema de predicción de riesgo térmico personalizado.

Responde SIEMPRE en español de España. Nunca en portugués, gallego, catalán,
inglés ni ninguna otra lengua, aunque el topónimo del usuario te suene a otro
idioma. Si dudas, español.

No inventes cifras: usa solo las que te den. No inventes horas ni franjas
horarias: si mencionas horas o picos de calor, usa solo las que vienen en los
datos. Si mencionas un término técnico (SPF, HI, WC), explícalo en lenguaje
llano entre paréntesis. La recomendación se adapta al contexto de la persona
(trabajo en obra, oficina, deporte...): nunca des consejos genéricos ni digas
"reduce la exposición en interiores".

Cuando el mensaje traiga FRASES OBLIGATORIAS, cópialas literalmente y en su
orden: ya vienen redactadas en lenguaje llano y sus cifras están calculadas.
Nunca traduzcas tú un multiplicador a palabras ni escribas "multiplica el
riesgo por N": eso no le dice nada a quien lee. Sin rodeos y sin emojis."""

CONTEXT_TEMPLATE = """\
=== FACTORES DE RIESGO RELEVANTES ===
{factores_ctx}

=== DOCUMENTACIÓN RELEVANTE DEL PROYECTO ===
{docs_ctx}

=== PREGUNTA ===
{question}"""


# ── Funciones principales ────────────────────────────────────────────


def _format_factores(results: list[dict]) -> str:
    if not results:
        return "(ninguno)"
    lines = []
    for i, r in enumerate(results, 1):
        tipo = r.get("tipo", "?")
        cat = r.get("categoria", "?")
        clave = r.get("clave", "?")
        texto = r.get("texto", "")
        dist = r.get("distance", "?")
        lines.append(
            f"{i}. [{tipo}/{cat}] {clave}: {texto[:300]} (distancia: {dist:.3f})"
        )
    return "\n".join(lines)


def _format_docs(results: list[dict]) -> str:
    if not results:
        return "(ninguno)"
    lines = []
    for i, r in enumerate(results, 1):
        titulo = r.get("titulo", "?")
        seccion = r.get("seccion", "?")
        texto = r.get("texto", "")
        dist = r.get("distance", "?")
        # `__intro__` es el centinela que el chunker le pone al texto anterior al
        # primer `##` (ver RAG._chunks_desde_md), no una sección real. Colándose
        # aquí, el modelo lo citaba tal cual: "Fuente: __intro__".
        etiqueta = titulo if seccion in ("__intro__", "?", "", None) else f"{titulo} / {seccion}"
        lines.append(
            f"{i}. [{etiqueta}]: {texto[:400]} (distancia: {dist:.3f})"
        )
    return "\n".join(lines)


def _format_perfil(perfil: dict | None) -> str:
    """Formatear datos del perfil para el prompt, incluyendo factores con multiplicadores y ocupación."""
    if not perfil:
        return ""

    lineas = []

    # Factores de riesgo desde el resultado del perfil
    perfil_data = perfil.get("perfil") or {}
    factores_calor = perfil_data.get("calor", {}).get("factores", [])
    factores_frio = perfil_data.get("frio", {}).get("factores", [])

    todos_los_factores = []
    for f in factores_calor + factores_frio:
        if isinstance(f, dict):
            nombre = f.get("nombre", str(f))
            coef = f.get("factor")
            if coef is not None:
                todos_los_factores.append(f"{nombre} (x{coef})")
            else:
                todos_los_factores.append(nombre)
        else:
            todos_los_factores.append(str(f))

    if todos_los_factores:
        lineas.append(f"Factores de riesgo personales: {', '.join(todos_los_factores)}")

    # Ocupación del perfil
    ocupacion = perfil.get("ocupacion")
    if ocupacion in _OCUPACION_NIVELES:
        coef, label = _OCUPACION_NIVELES[ocupacion]
        lineas.append(f"Ocupación: {label} (x{coef})")

    return "\n".join(lineas)


def ask_with_rag(
    question: str,
    k_factores: int = 5,
    k_docs: int = 5,
    config: LLMConfig | None = None,
    contexto: str | None = None,
    perfil: dict | None = None,
) -> dict[str, Any]:
    """RAG completo: busca en factores + documentación y responde con cualquier LLM.

    Args:
        question: Pregunta del usuario en lenguaje natural.
        k_factores: Cuántos factores de riesgo recuperar.
        k_docs: Cuántos fragmentos de documentación recuperar.
        config: Configuración del modelo (LiteLLM string: ollama/groq/openai/…).
        contexto: Texto que se añade al prompt pero NO a la búsqueda semántica.
            Lo usa el chat para las dudas posteriores a una predicción: el parte
            que se acaba de dar es contexto, no términos de búsqueda.
        perfil: Perfil del usuario (dict) para adaptar el consejo al contexto personal.

    Returns:
        dict con answer, sources_factores, sources_docs, model, error.
    """
    if config is None:
        config = LLMConfig()

    # 1. Buscar en RAG
    db = DBManager()
    db.init_rag()
    factores = db.search_factores(question, k=k_factores)
    docs = db.search_documentos(question, k=k_docs)

    # 2. Construir contexto
    factores_ctx = _format_factores(factores)
    docs_ctx = _format_docs(docs)

    has_context = bool(factores) or bool(docs)
    if not has_context:
        return ask_raw(question, config=config, contexto=contexto)

    user_prompt = CONTEXT_TEMPLATE.format(
        factores_ctx=factores_ctx,
        docs_ctx=docs_ctx,
        question=question,
    )
    if contexto:
        user_prompt = f"=== SITUACIÓN DEL USUARIO ===\n{contexto}\n\n{user_prompt}"
    
    # Si tenemos un perfil, incluirlo para adaptación contextual
    if perfil:
        perfil_ctx = _format_perfil(perfil)
        user_prompt += (
            "\n\n=== ADAPTACIÓN CONTEXTUAL ===\n"
            "Se necesita un consejo adaptado al contexto personal del usuario "
            "(ocupación, factores de riesgo). Evita consejos genéricos como "
            "'reducir la exposición en interiores'. Elige el enfoque apropiado "
            "(obra, oficina, deporte, etc.)."
            f"{perfil_ctx}"
        )

    # 3. Consultar LLM (LiteLLM unifica todos los proveedores)
    # Usar SYSTEM_PARTE si tenemos un perfil para adaptación contextual
    system_prompt = SYSTEM_PARTE if perfil else SYSTEM_RAG
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    answer = _chat_litellm(messages, config)

    return {
        "answer": answer,
        "sources_factores": factores,
        "sources_docs": docs,
        "model": config.model,
        "error": None if answer else f"No se pudo obtener respuesta de {config.model}",
    }


def ask_raw(
    question: str,
    config: LLMConfig | None = None,
    contexto: str | None = None,
) -> dict[str, Any]:
    """LLM raw sin RAG — sin contexto aumentado.

    Args:
        question: Pregunta del usuario.
        config: Configuración del modelo.
        contexto: Situación del usuario que se antepone a la pregunta.

    Returns:
        dict con answer, model, error.
    """
    if config is None:
        config = LLMConfig()

    messages = [
        {"role": "system", "content": SYSTEM_RAW},
        {"role": "user", "content": f"{contexto}\n\n{question}" if contexto else question},
    ]
    answer = _chat_litellm(messages, config)

    return {
        "answer": answer,
        "sources_factores": [],
        "sources_docs": [],
        "model": config.model,
        "error": None if answer else f"No se pudo obtener respuesta de {config.model}",
    }


# ── El parte en lenguaje llano (BOT-013) ───────────────────────────────
#
# Todo lo que el parte DEBE decir sí o sí se redacta aquí, en Python, y se le
# da al LLM ya masticado. Un modelo de 1.5B no deriva "algo más del doble" de
# un x2.2 de forma fiable, pero sí copia una frase que le venga hecha.

_CLASES_LLANAS = {
    "SEGURO": "SEGURO, el nivel más bajo de tres: seguro / precaución / peligro",
    "PRECAUCION": "PRECAUCIÓN, el nivel intermedio de tres: seguro / precaución / peligro",
    "PRECAUCIÓN": "PRECAUCIÓN, el nivel intermedio de tres: seguro / precaución / peligro",
    "PELIGRO": "PELIGRO, el nivel más alto de tres: seguro / precaución / peligro",
}

# El nivel NO se deduce del porcentaje: sale de `apply_class_thresholds` con los
# umbrales de la provincia y de los overrides físicos de `predict_ensemble`
# (HI>=39 → PELIGRO, HI>=27 + UV>3 → PRECAUCION). Enseñar los dos números
# juntos sin decir esto es lo que hacía pensar que "PRECAUCION: 19%" era un
# 2 sobre 10 de peligro.
LINEA_CLASE_VS_PORCENTAJE = (
    "Esa cifra no es lo que decide tu nivel: el nivel se calcula aparte, con los "
    "umbrales de tu provincia y con reglas de seguridad por calor y radiación, "
    "así que una cifra baja puede venir con un nivel alto."
)

_CONFIANZA_LLANA = {
    "alta": (
        "El modelo está seguro de este nivel: con los datos de hoy no le encaja otro."
    ),
    "media": (
        "El modelo duda entre este nivel y el de al lado, así que tómatelo como "
        "una orientación, no como una medida exacta."
    ),
    "baja": (
        "Aviso: hoy el modelo tiene poca confianza — con estos datos no puede "
        "descartar ningún nivel, así que fíate poco de la cifra y ve por el lado "
        "seguro."
    ),
}


def _clase_llana(clase: str | None) -> str:
    """Ancla la clase en su escala: 'PRECAUCION' solo no dice si es mucho o poco."""
    return _CLASES_LLANAS.get((clase or "").strip().upper(), clase or "?")


def _frecuencia_natural(prob: float) -> str:
    """El porcentaje como frecuencia natural: '19 de cada 100' se entiende, '19%' no."""
    n = round(max(0.0, min(1.0, float(prob or 0))) * 100)
    if n < 1:
        cuantos = "en menos de 1"
    elif n == 1:
        cuantos = "en 1"
    else:
        cuantos = f"en unos {n}"
    return (
        f"De cada 100 días como el de hoy y con la misma salida, {cuantos} el "
        "calor te pasaría factura."
    )


def _coeficiente_llano(coef: float) -> str:
    """x2.2 no significa nada sin un 'comparado con qué' ni sin traducción."""
    c = float(coef)
    if c < 1.05:
        return "prácticamente el mismo"
    if c < 1.75:
        return f"un {round((c - 1) * 100)}% más alto"
    if c < 2.15:
        return "el doble"
    if c < 2.55:
        return "algo más del doble"
    if c < 2.85:
        return "casi el triple"
    if c < 3.25:
        return "el triple"
    return f"más de {int(c)} veces mayor"


def _linea_factor_dominante(factores: list) -> str | None:
    """El factor que más pesa, en llano y CON LÍNEA BASE (contra quién se compara)."""
    candidatos = [
        f for f in factores
        if isinstance(f, dict) and isinstance(f.get("factor"), (int, float))
    ]
    if not candidatos:
        return None
    dom = max(candidatos, key=lambda f: f["factor"])
    if dom["factor"] <= 1.0:
        return None
    llano = _coeficiente_llano(dom["factor"])
    nombre = str(dom.get("nombre", "")).strip()
    if nombre.lower().startswith("trabajo "):
        # "trabajo Construcción / albañilería (carga pesada, PPE, sol directo)"
        # → "construcción": la etiqueta larga es del pipeline, no del parte.
        oficio = nombre[len("trabajo "):].split("(")[0].split("/")[0].strip().lower()
        return (
            "Lo que más pesa en tu caso no es el tiempo, es tu trabajo: en "
            f"{oficio} tu riesgo es {llano} que el de alguien como tú a cubierto."
        )
    return (
        f"Lo que más pesa en tu caso no es el tiempo, es {nombre}: con ese factor "
        f"tu riesgo es {llano} que el de alguien como tú sin él."
    )


def _confianza_conformal(resultado_prediccion: dict) -> str | None:
    """Confianza conformal (alta/media/baja) del canal que da la cifra del parte.

    Es `None` cuando no existe el artefacto `conformal_<clase>.joblib`: el
    ensemble ya lo deja a None en ese caso y el parte se calla, no se inventa
    una confianza.
    """
    modelos = resultado_prediccion.get("modelos") or {}
    for clave in ("XGBoost_calor", "RandomForest_frio"):
        m = modelos.get(clave)
        if isinstance(m, dict) and m.get("conformal_confianza"):
            return str(m["conformal_confianza"])
    return None


def lineas_parte(resultado_prediccion: dict, lugar: str | None = None) -> list[str]:
    """Las frases que un parte no puede perder, ya redactadas en llano.

    Las comparten las dos vías del parte: el prompt de `ask_con_perfil` (que se
    las da hechas al LLM para que solo las copie) y la plantilla determinista
    del bot cuando no hay LLM. Si divergieran, el usuario leería dos partes
    distintos del mismo día según si Ollama estaba levantado.
    """
    w = resultado_prediccion.get("weather") or {}
    p = (resultado_prediccion.get("perfil") or {}).get("calor") or {}
    ubicacion = lugar or w.get("provincia") or "?"

    lineas = [
        f"{ubicacion} — {_clase_llana(resultado_prediccion.get('clase_final_label'))}.",
        _frecuencia_natural(p.get("prob_personalizada") or 0),
        LINEA_CLASE_VS_PORCENTAJE,
    ]
    linea_factor = _linea_factor_dominante(p.get("factores") or [])
    if linea_factor:
        lineas.append(linea_factor)
    linea_confianza = _CONFIANZA_LLANA.get(
        (_confianza_conformal(resultado_prediccion) or "").lower()
    )
    if linea_confianza:
        lineas.append(linea_confianza)
    return lineas


# Trozo distintivo de cada frase que el parte no puede perder. Sirve para saber
# si el LLM la copió: comparar la frase entera no vale porque los modelos
# pequeños cambian una palabra ("como hoy" por "como el de hoy") al copiar.
_MARCAS_REPONIBLES = (
    "te pasaría factura",           # la frecuencia natural: es LA cifra del parte
    "no es lo que decide tu nivel",  # el nivel no sale del porcentaje
    "Lo que más pesa",              # el factor dominante con su línea base
    "El modelo está seguro",        # confianza alta
    "duda entre este nivel",        # confianza media
    "poca confianza",               # confianza baja: es un aviso de seguridad
)


def _reponer_obligatorias(respuesta: str, obligatorias: list[str]) -> str:
    """Añade al final las frases obligatorias que el LLM se saltó.

    qwen2.5:1.5b copia 4 de las 5 y se salta una distinta cada vez: en una
    tirada se comió la que separa el nivel del porcentaje, en la siguiente la
    de la frecuencia. Ninguna de las dos puede faltar, así que no se deja a su
    criterio.
    """
    faltan = [
        linea
        for linea in obligatorias
        for marca in _MARCAS_REPONIBLES
        if marca in linea and marca not in respuesta
    ]
    return " ".join([respuesta.rstrip(), *faltan]) if faltan else respuesta


def ask_con_perfil(
    perfil: dict,
    resultado_prediccion: dict,
    config: LLMConfig | None = None,
    lugar: str | None = None,
) -> str | None:
    """Redacta la respuesta de una predicción usando el LLM.

    Es el reemplazo unificado de _format_with_llm del bot: funciona con
    cualquier modelo LiteLLM (local o API). `lugar` es el nombre con el que
    el usuario identificó su ubicación (p.ej. "Moaña, Pontevedra"); el prompt
    lo incluye para que la explicación diga dónde aplica el riesgo.

    La llamada es síncrona (LiteLLM) — el bot la ejecuta en un hilo.
    """
    if config is None:
        config = LLMConfig()

    w = resultado_prediccion.get("weather", {})
    cur = w.get("current", {})
    perfil_h = w.get("perfil_horario") or []
    hi = max(h["HI"] for h in perfil_h) if perfil_h else (cur.get("t2m_c") or 0)
    p = resultado_prediccion.get("perfil", {}).get("calor", {})
    factores = p.get("factores", [])
    clase = resultado_prediccion.get("clase_final_label", "?")
    prob = p.get("prob_personalizada") or 0
    ubicacion = lugar or w.get("provincia", "?")
    uv = w.get("uv_index")
    uv_label = f"{uv:.1f}".rstrip("0").rstrip(".") if uv is not None else "n/d"
    resumen = recomendacion_resumen(resultado_prediccion)

    # Factores con su coeficiente ("construcción (x2.2)"), no solo el nombre:
    # sin el multiplicador el LLM se inventa cuánto pesa cada factor.
    factores_ctx = ", ".join(
        f"{f['nombre']} (x{f['factor']})" if isinstance(f, dict) else str(f)
        for f in factores
    ) or "ninguno"

    # Ocupación del perfil con su etiqueta y coeficiente (si está en la tabla
    # _OCUPACION_NIVELES). Quien trabaja en obra no recibe consejos de oficina.
    ocp = perfil.get("ocupacion")
    ocupacion_ctx = ""
    if ocp in _OCUPACION_NIVELES:
        coef, label = _OCUPACION_NIVELES[ocp]
        ocupacion_ctx = f"{label} (x{coef})"

    # Franjas horarias REALES: la recomendada la da recomendar_horario y el
    # pico sale de la hora con mayor HI del perfil_horario. El LLM no inventa
    # horas: solo puede usar estas.
    franja_ctx = ""
    if perfil_h:
        lineas_franja = []
        rec = recomendar_horario(perfil_h, perfil)
        if rec and rec.get("hora_inicio") is not None:
            lineas_franja.append(
                f"Franja recomendada (menor riesgo): {rec['hora_inicio']:.0f}:00"
                f"-{rec['hora_fin']:.0f}:00"
            )
        pico_h = max(perfil_h, key=lambda h: h["HI"])
        lineas_franja.append(
            f"Pico de calor (evitar si puedes): {pico_h['hora']:.0f}:00 "
            f"(HI {pico_h['HI']:.1f}°C)"
        )
        franja_ctx = "\n".join(lineas_franja)
    bloque_franja = f"\n{franja_ctx}" if franja_ctx else ""

    # Las frases que el parte no puede perder se redactan aquí y el LLM solo
    # las copia (BOT-013). Lo que sí queda a su cargo es la recomendación final.
    confianza = _confianza_conformal(resultado_prediccion)
    obligatorias = lineas_parte(resultado_prediccion, lugar)
    bloque_obligatorio = "\n".join(obligatorias)

    prompt = f"""\
Ubicación: {ubicacion}
Clase de riesgo: {clase} (la deciden los umbrales de la provincia y las reglas
de seguridad por calor/UV, NO la probabilidad de abajo)
Probabilidad personalizada del canal de calor: {prob:.0%} (NO la escribas como
porcentaje: abajo te la damos ya redactada)
Temperatura: {(cur.get("t2m_c") or 0):.1f}°C (HI pico: {hi:.1f}°C)
Índice UV: {uv_label}
Humedad: {(cur.get("rh") or 0):.0f}%
Ocupación: {ocupacion_ctx or 'n/d'}
Factores de riesgo: {factores_ctx}{bloque_franja}
Confianza del modelo: {confianza or 'no medida'}
Recomendación contextual: {resumen}

Escribe el parte así: copia TAL CUAL las FRASES OBLIGATORIAS, en su orden y una
por línea, y termina con la FRASE DE CIERRE adaptada al contexto de la persona.
No reformules las frases obligatorias, no recalcules sus cifras y no añadas
ningún porcentaje que no esté en ellas. Si hace frío recomienda abrigo; solo
menciona protección solar si el índice UV lo justifica. Sin rodeos, sin emojis,
sin viñetas ni títulos, y no repitas estas instrucciones en tu respuesta.

FRASES OBLIGATORIAS:
{bloque_obligatorio}

FRASE DE CIERRE:
{resumen}
"""
    messages = [
        {"role": "system", "content": SYSTEM_PARTE},
        {"role": "user", "content": prompt},
    ]
    try:
        respuesta = _chat_litellm(messages, config)
    except Exception as exc:
        logger.warning("LLM falló al redactar respuesta: %s", exc)
        return None

    return _reponer_obligatorias(respuesta, obligatorias) if respuesta else respuesta


# ── Detección del mejor modelo ─────────────────────────────────────────


def check_ollama() -> dict[str, Any]:
    """Verifica disponibilidad de Ollama y modelos disponibles."""
    modelos = _modelos_ollama()
    return {
        "available": bool(modelos),
        "models": modelos,
        "best_model": LLMConfig.mejor_disponible().model if modelos else "",
    }
