# ClimaSafeAI — Contador de tokens y coste por llamada al LLM (ARNES-004)
#
# Cada llamada al LLM registra tokens de entrada, de salida, latencia y coste
# estimado. El coste se calcula contra la tabla PRECIOS_MODELOS, que vive SOLO
# aquí: actualizar un precio es tocar una línea de este fichero.
#
# Para los modelos locales (Ollama) el coste es cero, pero los tokens y la
# latencia se registran igual: es el dato que justifica elegir local.

from __future__ import annotations

import logging
import os
from typing import Any

logger = logging.getLogger(__name__)

# ── Tope de presupuesto de tokens por petición (ARNES-010) ─────────────
#
# El gasto de una petición se acota para que un descontrol no se traduzca en
# una factura. El tope es configurable por variable de entorno
# (CLIMASAFE_MAX_TOKENS_PETICION, en tokens) y el valor por defecto se elige
# contra la referencia que motivó la feature: spacebot gasta ~7.900 tokens por
# mensaje, así que 10.000 deja margen a la petición legítima (≈ +27%) y corta
# los picos. El corte ocurre en `_chat_litellm` (climasafeai.llm.rag_qwen),
# que es la única puerta al LLM: la excepción PresupuestoExcedidoError es el
# error claro con el que se corta — nunca en silencio.

ENV_TOPE_TOKENS = "CLIMASAFE_MAX_TOKENS_PETICION"
TOPE_TOKENS_POR_PETICION = 10_000


class PresupuestoExcedidoError(Exception):
    """Se lanza cuando una petición al LLM supera el tope de tokens.

    El mensaje lleva la cifra medida, la etiqueta de qué se midió y el tope:
    es el error claro con el que se corta la petición.
    """


def tope_tokens_peticion() -> int:
    """Tope de tokens por petición, desde CLIMASAFE_MAX_TOKENS_PETICION.

    Si la variable no está puesta o no es un entero positivo, se usa
    TOPE_TOKENS_POR_PETICION: un valor inválido no puede dejar el tope en 0
    (cortaría todas las peticiones) ni romper el arranque.
    """
    raw = os.environ.get(ENV_TOPE_TOKENS, "").strip()
    if raw:
        try:
            valor = int(raw)
        except ValueError:
            logger.warning("%s='%s' no es un entero; se usa el tope por defecto", ENV_TOPE_TOKENS, raw)
        else:
            if valor > 0:
                return valor
            logger.warning("%s=%s no es positivo; se usa el tope por defecto", ENV_TOPE_TOKENS, valor)
    return TOPE_TOKENS_POR_PETICION


def comprobar_presupuesto(tokens: int, *, etiqueta: str, tope: int | None = None) -> None:
    """Lanza PresupuestoExcedidoError si `tokens` supera el tope de la petición.

    Args:
        tokens: Tokens medidos (estimados o reales, según `etiqueta`).
        etiqueta: Qué se midió ("payload estimado", "usage real (prompt+completion)").
        tope: Límite en tokens; si es None se lee de la configuración.
    """
    tope = tope_tokens_peticion() if tope is None else tope
    if tokens > tope:
        raise PresupuestoExcedidoError(
            f"Presupuesto de tokens superado: {tokens} tokens ({etiqueta}) > tope {tope}."
        )

# ── Tabla de precios (única fuente de verdad) ────────────────────────────
#
# Precios por MILLÓN de tokens, en USD, según las tarifas públicas de cada
# proveedor. La clave es el nombre del modelo SIN prefijo de proveedor
# LiteLLM ("groq/openai/gpt-oss-20b" → "gpt-oss-20b"); `precios_de` hace la
# normalización. Para actualizar un precio basta editar esta línea.
#
# Los modelos locales (Ollama) y los que no aparecen aquí cuestan cero.

PRECIOS_MODELOS: dict[str, dict[str, float]] = {
    # Modelos remotos que usa el proyecto (ver MODELO_API_DEFECTO en rag_qwen).
    "gpt-oss-20b": {"prompt": 0.20, "completion": 0.80},
    # Referencia de modelos remotos habituales (editar al gusto).
    "gpt-4o": {"prompt": 2.50, "completion": 10.00},
    "gpt-4o-mini": {"prompt": 0.15, "completion": 0.60},
    "gemini-3.6-flash": {"prompt": 0.10, "completion": 0.40},
    "llama-3.3-70b": {"prompt": 0.59, "completion": 0.79},
}

# Prefijos de proveedor que LiteLLM antepone al nombre del modelo. Se quitan
# para buscar en PRECIOS_MODELOS; pueden ir anidados ("groq/openai/gpt-oss-20b").
_PREFIJOS_PROVEEDOR = ("ollama/", "groq/", "openai/", "gemini/", "anthropic/")

# Claves del acumulado por sesión (cero para una sesión sin llamadas).
_CLAVES_ACUM = ("llamadas", "prompt_tokens", "completion_tokens", "coste", "latencia_s")

# Acumulado en memoria por sesión: {sesion_id: {clave: valor}}.
_ACUMULADOS: dict[str, dict[str, float]] = {}


def _base_modelo(modelo: str) -> str:
    """Quita el prefijo de proveedor de LiteLLM, incluidos los anidados.

    "groq/openai/gpt-oss-20b" → "gpt-oss-20b"
    "ollama/qwen3:climasafe"  → "qwen3:climasafe"
    """
    base = modelo
    for prefijo in _PREFIJOS_PROVEEDOR:
        while base.startswith(prefijo):
            base = base[len(prefijo):]
    return base


def es_local(modelo: str) -> bool:
    """True para modelos locales servidos por Ollama (coste cero)."""
    return modelo.startswith("ollama/")


def precios_de(modelo: str) -> tuple[float, float]:
    """(precio_prompt, precio_completion) POR TOKEN para un modelo.

    Los locales (Ollama) y los que no están en PRECIOS_MODELOS cuestan cero:
    se sigue midiendo tokens y latencia, pero no hay coste.
    """
    if es_local(modelo):
        return 0.0, 0.0
    precio = PRECIOS_MODELOS.get(_base_modelo(modelo))
    if precio is None:
        return 0.0, 0.0
    return precio["prompt"] / 1_000_000, precio["completion"] / 1_000_000


def coste_llamada(modelo: str, prompt_tokens: int, completion_tokens: int) -> float:
    """Coste estimado de una llamada en USD, desde PRECIOS_MODELOS."""
    precio_prompt, precio_completion = precios_de(modelo)
    return prompt_tokens * precio_prompt + completion_tokens * precio_completion


def registrar_llamada(
    modelo: str,
    prompt_tokens: int,
    completion_tokens: int,
    latencia_s: float,
    sesion_id: str = "default",
) -> dict[str, Any]:
    """Registra una llamada en el acumulado de su sesión y devuelve el detalle.

    Args:
        modelo: Nombre LiteLLM del modelo ("ollama/qwen2.5:1.5b", "groq/...").
        prompt_tokens: Tokens de entrada, del `usage` que devuelve LiteLLM.
        completion_tokens: Tokens de salida, del `usage` que devuelve LiteLLM.
        latencia_s: Segundos que tardó la llamada (medidos en el caller).
        sesion_id: Identificador del acumulado (p. ej. el chat_id del bot).

    Returns:
        dict con el detalle de ESTA llamada (modelo, tokens, coste, latencia).
    """
    coste = coste_llamada(modelo, prompt_tokens, completion_tokens)
    acum = _ACUMULADOS.setdefault(sesion_id, {k: 0.0 for k in _CLAVES_ACUM})
    acum["llamadas"] += 1
    acum["prompt_tokens"] += prompt_tokens
    acum["completion_tokens"] += completion_tokens
    acum["coste"] += coste
    acum["latencia_s"] += latencia_s

    detalle = {
        "modelo": modelo,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
        "coste": round(coste, 6),
        "latencia_s": round(latencia_s, 3),
    }
    logger.info(
        "LLM %s: %d tokens (%d prompt + %d completion) en %.2fs, coste $%.6f",
        modelo,
        detalle["total_tokens"],
        prompt_tokens,
        completion_tokens,
        latencia_s,
        coste,
    )
    return detalle


def resumen_sesion(sesion_id: str = "default") -> dict[str, Any]:
    """Acumulado consultable de una sesión (todo a cero si no ha llamado)."""
    if sesion_id not in _ACUMULADOS:
        return {k: 0 for k in _CLAVES_ACUM}
    acum = {k: v for k, v in _ACUMULADOS[sesion_id].items()}
    acum["coste"] = round(acum["coste"], 6)
    return acum
