#!/usr/bin/env python3
"""BOT-006 — Demo real: recomendación post-predicción con LLM local (Ollama).

Ejecuta el camino EXACTO del bot tras la predicción: `ejecutar_prediccion`
construye el `perfil` desde la conversación y llama a `ask_con_perfil` con el
mejor modelo local de `check_ollama()`. Para esta demo se pincha qwen2.5 (la
feature pide "Ollama con qwen2.5"): la llamada es la misma que haría el bot
solo que con el modelo pedido, no con el best_model de esta máquina (que es
qwen3:climasafe, el fine-tune de LLM-014).

  1. check_ollama(): la detección real (criterio 1) — modelos y best_model.
  2. Parte redactado por qwen2.5:1.5b con el contexto real (ubicación,
     actividad, perfil, temperatura, UV), con latencia y tokens (criterios 2 y 4).
  3. Contraste: la plantilla determinista de BOT-005, sin LLM (criterio 3).

Uso:
  .venv/bin/python scripts/demo_llm_local.py

REQUIERE Ollama corriendo (p.ej. `ollama serve &`). NO imprime claves. NO es
parte de la suite de tests: hace UNA llamada real al LLM local.
"""

from __future__ import annotations

import asyncio
import logging
import sys
import time

# ── Logging mínimo: la línea de tokens de _chat_litellm vive en INFO.
logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
logging.getLogger("litellm").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("climasafeai.llm.rag_qwen").setLevel(logging.INFO)

from climasafeai.bot.telegram_bot import _format_template  # noqa: E402
from climasafeai.llm.rag_qwen import LLMConfig, ask_con_perfil, check_ollama  # noqa: E402

# Resultado de predicción realista (misma forma que produce el ensemble):
# PELIGRO en Pontevedra, 36 °C, UV 7, jornada de 2 h a las 17:00.
RESULTADO = {
    "clase_final": 2,
    "clase_final_label": "PELIGRO",
    "perfil": {
        "calor": {
            "prob_personalizada": 0.72,
            "factores": [
                {"nombre": "diureticos_asa", "factor": 1.8},
                {"nombre": "vive_solo", "factor": 1.5},
                {"nombre": "no_aclimatado", "factor": 1.3},
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

# El perfil que `ejecutar_prediccion` construye desde la conversación.
PERFIL = {
    "sexo": "hombre",
    "edad": 57,
    "aclimatado": False,
    "nivel_actividad": "moderada",
    "duracion_actividad_h": 2,
    "hora_inicio": 17,
    "comorbilidades": {"cardiovascular"},
    "farmacos": {"diureticos_asa"},
    "situacion_social": {"vive_solo"},
    "ocupacion": "obra",
    "entrenado": True,
}


def _entrada_usuario() -> str:
    """La conversación del usuario que alimenta esta predicción (lo que el
    bot recogió por botones antes de predecir)."""
    return "\n".join(
        [
            "Entrada del usuario (formulario /start):",
            "  • Sexo: hombre · Edad: 57 · % grasa: 20.5 · Fototipo: 3",
            "  • Aclimatado: no · Actividad: moderada (2 h) · Empieza: 17:00",
            "  • Entrenado: sí · Trabajo: obra (x2.7) · Estado previo: noche de fiesta",
            "  • Comorbilidades: cardiovascular · Medicación: diuréticos de asa (x1.8)",
            "  • Situación social: vive solo (x1.5) · Ubicación: Aldán, Pontevedra",
        ]
    )


def main() -> None:
    print("=" * 72)
    print("BOT-006 · Demo: recomendación post-predicción con LLM local")
    print("=" * 72)

    st = check_ollama()
    if not st.get("available"):
        print("❌  Ollama NO responde (check_ollama(): available=False).")
        print("    Arranca el servidor: `ollama serve &` (o como en DEPLOY_IA.md).")
        sys.exit(1)
    print("✔  LLM local detectado (check_ollama(): available=True).")
    print(f"    Modelos en Ollama: {', '.join(st['models'])}")
    print(f"    best_model del bot (LLM-003): {st['best_model']}")

    modelo_demo = "ollama/qwen2.5:1.5b"  # la feature pide qwen2.5
    print(f"\n─ PARTE redactado por el LLM local ({modelo_demo}) ─")
    print(_entrada_usuario())

    t0 = time.perf_counter()
    texto = ask_con_perfil(PERFIL, RESULTADO, LLMConfig(model=modelo_demo), lugar="Aldán")
    latencia = time.perf_counter() - t0

    if not texto:
        print("FALLO: el LLM local no contestó el parte.")
        sys.exit(2)
    print("\nParte de la predicción:")
    print(texto)
    print(f"\n⏱  Latencia de la llamada al LLM local: {latencia:.2f} s")
    print("   (tokens: ver la línea 'tokens ...' del log de rag_qwen más arriba)")

    # Contraste: lo que responde el bot sin LLM local (criterio 3).
    print("\n─ CONTRASTE · plantilla determinista de BOT-005 (sin LLM) ─")
    print(_format_template(RESULTADO, "Aldán"))

    print("\n✔  Demo completa: LLM local detectado, parte redactado por qwen2.5")
    print("   con el contexto real, latencia y tokens medidos, y plantilla a la vista.")


if __name__ == "__main__":
    main()
