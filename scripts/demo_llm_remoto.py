#!/usr/bin/env python3
"""HOST-001 — Demo real: el bot responde con LLM remoto sin Ollama.

Ejecuta las funciones EXACTAS que usa el bot (telegram_bot.py llama a
`ask_con_perfil` para el parte y a `ask_with_rag` para el chat libre) contra
el proveedor remoto con la key real del .env, con Ollama PARADO.

  - 1ª llamada: parte redactado por Groq (gpt-oss-20b, free tier).
  - 2ª llamada: pregunta libre del chat con RAG + contexto del parte.
  - 3ª llamada (a propósito fallida): modelo retirado de Groq
    (llama-3.3-70b-versatile, 404 real) → `ask_con_perfil` devuelve None →
    el parte cae a la plantilla determinista `_format_template`, igual que
    haría el bot con el servicio caído o sin cuota.

Uso:
  .venv/bin/python scripts/demo_llm_remoto.py

NO imprime las claves. NO es parte de la suite de tests: hace llamadas reales
con la cuota gratuita (3 llamadas Groq). Si Ollama está corriendo, aborta: la
demostración exige el proceso local parado.
"""

from __future__ import annotations

import logging
import os
import sys

from dotenv import load_dotenv

load_dotenv()

# ── Logging mínimo: solo las líneas de tokens y errores de nuestras librerías
logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
logging.getLogger("litellm").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("climasafeai.llm.rag_qwen").setLevel(logging.INFO)

from climasafeai.bot.telegram_bot import MODELO_API, MODELO_DETERMINISTA, _format_template  # noqa: E402
from climasafeai.llm.rag_qwen import LLMConfig, ask_con_perfil, ask_with_rag, check_ollama  # noqa: E402

# Resultado de predicción realista (misma forma que la que produce el ensemble
# y que usan los tests): PELIGRO en Pontevedra, un día de 36 °C.
RESULTADO = {
    "clase_final": 2,
    "clase_final_label": "PELIGRO",
    "perfil": {
        "calor": {
            "prob_personalizada": 0.72,
            "factores": [
                {"nombre": "vive_solo", "factor": 1.5},
                {"nombre": "diureticos_asa", "factor": 1.8},
            ],
        }
    },
    "perfil_usuario": {
        "hora_inicio": 17,
        "duracion_actividad_h": 2,
        "ocupacion": "obra",
        "sexo": "hombre",
        "edad": 57,
    },
    "weather": {
        "provincia": "Pontevedra",
        "current": {"t2m_c": 36.0, "rh": 50},
        "uv_index": 7,
        "perfil_horario": [
            {"hora": 17, "HI": 39.0, "temp": 36.0},
            {"hora": 18, "HI": 38.0, "temp": 35.0},
        ],
    },
    "modelos": {"Formula": {"frio": {"wind_chill_c": 30}, "calor": {"heat_index_c": 39}}},
    "recomendaciones": ["Evita la actividad"],
}

PERFIL = {"ocupacion": "obra", "edad": 57, "sexo": "hombre"}


def _estado_ollama() -> None:
    st = check_ollama()
    if st.get("available"):
        print("❌  Ollama está CORRIENDO. La demo exige el proceso local parado.")
        print("    Páralo (p.ej. `kill $(pgrep ollama)` o `ollama stop`) y repite.")
        sys.exit(1)
    print("✔  Ollama NO está corriendo (check_ollama(): available=False).")


def _proveedor() -> str:
    if os.getenv("GROQ_API_KEY"):
        return MODELO_API
    if os.getenv("GEMINI_API_KEY"):
        return "gemini/gemini-3.6-flash"
    print("❌  No hay GROQ_API_KEY ni GEMINI_API_KEY en el entorno.")
    sys.exit(1)
    return ""  # inalcanzable


def main() -> None:
    print("=" * 72)
    print("HOST-001 · Demo: bot con LLM remoto, Ollama parado")
    print("=" * 72)
    _estado_ollama()
    modelo = _proveedor()
    print(f"Proveedor remoto: {modelo}")

    # ── 1. El parte por el LLM remoto (ask_con_perfil: la llamada del bot) ──
    print("\n─ 1/3 · PARTE redactado por el LLM remoto ─")
    texto = ask_con_perfil(PERFIL, RESULTADO, LLMConfig(model=modelo), lugar="Aldán")
    if not texto:
        print("FALLO: el LLM remoto no contestó el parte.")
        sys.exit(2)
    print(texto)

    # ── 2. Pregunta libre del chat con RAG (ask_with_rag: la llamada del bot) ─
    print("\n─ 2/3 · PREGUNTA LIBRE del chat (RAG + contexto del parte) ─")
    contexto = f"Parte que le acabas de dar al usuario:\n{texto}"
    res = ask_with_rag(
        "¿qué es el SPF y cada cuánto debo aplicármelo si trabajo en obra?",
        k_factores=3,
        k_docs=3,
        config=LLMConfig(model=modelo),
        contexto=contexto,
        perfil=PERFIL,
    )
    if not res.get("answer"):
        print(f"FALLO: el LLM remoto no contestó la pregunta libre. error={res.get('error')}")
        sys.exit(3)
    print(res["answer"])
    print(f"\n(fuentes RAG: {len(res.get('sources_factores', []))} factores, "
          f"{len(res.get('sources_docs', []))} documentos)")

    # ── 3. Fallo provocado: modelo retirado → plantilla determinista ──
    print("\n─ 3/3 · SERVICIO CAÍDO (modelo retirado: groq/llama-3.3-70b-versatile) ─")
    print("Llamando a ask_con_perfil con un modelo que ya no existe en Groq...")
    muerto = "groq/llama-3.3-70b-versatile"
    texto_none = ask_con_perfil(PERFIL, RESULTADO, LLMConfig(model=muerto), lugar="Aldán")
    if texto_none is not None:
        print(f"INESPERADO: el modelo retirado contestó: {texto_none!r}")
        sys.exit(4)
    plantilla = _format_template(RESULTADO, "Aldán")
    print("ask_con_perfil devolvió None → el bot responde con la plantilla determinista:")
    print(plantilla)

    print("\n✔  Demo completa: LLM remoto respondiendo con Ollama parado y")
    print("   degradación a plantilla cuando el proveedor falla.")


if __name__ == "__main__":
    main()
