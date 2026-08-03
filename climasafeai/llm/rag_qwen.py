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
        sobre los 100 ejemplos de data/llm/val.jsonl:

            modelo             clase  formato  inventa cifras  error del índice
            qwen3:1.7b          32%     100%        14%              0.385
            qwen2.5:1.5b        25%     100%       100%              8.417

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
8. Menciona valores o umbrales concretos si aparecen en el contexto."""

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

No inventes cifras: usa solo las que te den. Sin rodeos y sin emojis."""

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


def ask_with_rag(
    question: str,
    k_factores: int = 5,
    k_docs: int = 5,
    config: LLMConfig | None = None,
    contexto: str | None = None,
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

    # 3. Consultar LLM (LiteLLM unifica todos los proveedores)
    messages = [
        {"role": "system", "content": SYSTEM_RAG},
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

    prompt = f"""\
Ubicación: {ubicacion}
Clase de riesgo: {clase}
Probabilidad personalizada: {prob:.0%}
Temperatura: {(cur.get("t2m_c") or 0):.1f}°C (HI pico: {hi:.1f}°C)
Índice UV: {uv_label}
Humedad: {(cur.get("rh") or 0):.0f}%
Factores de riesgo: {', '.join(f['nombre'] for f in factores) if factores else 'ninguno'}
Recomendación contextual: {resumen}

Escribe un parte médico de una línea con la ubicación, la clase, la probabilidad
explicada en lenguaje llano y la recomendación principal adaptada al contexto:
si hace frío recomienda abrigo, y solo menciona protección solar si el índice
UV lo justifica.
La probabilidad se explica como porcentaje de riesgo térmico durante la
actividad, p. ej. "21% de probabilidad de riesgo térmico durante la actividad
(mayor cuanto más se acerque a 100%)".
Sin rodeos, sin emojis.
Ejemplo: "{ubicacion} — Riesgo {clase}: {prob:.0%} de probabilidad de riesgo térmico durante la actividad (mayor cuanto más se acerque a 100%). {resumen}"
"""
    messages = [
        {"role": "system", "content": SYSTEM_PARTE},
        {"role": "user", "content": prompt},
    ]
    try:
        return _chat_litellm(messages, config)
    except Exception as exc:
        logger.warning("LLM falló al redactar respuesta: %s", exc)
        return None


# ── Detección del mejor modelo ─────────────────────────────────────────


def check_ollama() -> dict[str, Any]:
    """Verifica disponibilidad de Ollama y modelos disponibles."""
    modelos = _modelos_ollama()
    return {
        "available": bool(modelos),
        "models": modelos,
        "best_model": LLMConfig.mejor_disponible().model if modelos else "",
    }
