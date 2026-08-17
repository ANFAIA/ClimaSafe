"""Guardia del estudio base vs instruct (LLM-016 / 2026-08-17).

LLM-016 es un estudio documental: `documentacion/llm/base-vs-instruct.md`
explica la diferencia entre modelo base y modelo de instrucciones y decide
sobre qué variante aplicar el próximo LoRA. Estos tests garantizan que el
documento existe, cubre los cuatro temas del criterio y está enlazado desde
docs_site/llm.md.
"""

import pathlib

_ROOT = pathlib.Path(__file__).resolve().parents[1]
_DOC = _ROOT / "documentacion" / "llm" / "base-vs-instruct.md"
_LLM_PAGE = _ROOT / "docs_site" / "llm.md"

# Términos que el estudio tiene que tocar sí o sí (criterio 1 de LLM-016).
# En minúsculas: el texto se normaliza con .lower() antes de buscar.
_TERMINOS_CLAVE = [
    "chat template",
    "thinking",
    "lora",
    "base",
    "instruct",
]


def test_estudio_base_vs_instruct_existe():
    assert _DOC.exists(), "falta documentacion/llm/base-vs-instruct.md (LLM-016)"
    assert _DOC.stat().st_size > 2000, "el estudio está vacío o es un stub"


def test_estudio_cubre_los_temas_clave():
    texto = _DOC.read_text(encoding="utf-8").lower()
    for termino in _TERMINOS_CLAVE:
        assert termino in texto, f"el estudio no menciona '{termino}'"


def test_estudio_incluye_comparacion_real():
    texto = _DOC.read_text(encoding="utf-8")
    assert "qwen3:climasafe" in texto, "falta la salida del fine-tuned"
    assert "qwen3:1.7b" in texto, "falta la salida del instruct"
    assert "ollama list" in texto, "falta la limitación de modelos disponibles"


def test_estudio_tiene_conclusion_practica():
    texto = _DOC.read_text(encoding="utf-8")
    assert "Conclusión práctica" in texto, "falta la sección de conclusión"
    assert "Instruct" in texto, "la conclusión no menciona partir de instruct"


def test_estudio_enlazado_desde_docs_site():
    pagina = _LLM_PAGE.read_text(encoding="utf-8")
    assert "base-vs-instruct.md" in pagina, "docs_site/llm.md no enlaza el estudio"
