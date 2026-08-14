"""Guardia del sitio de documentación (DOC-005 / 2026-08-14).

Bug: `exclude_docs` con `*`/`**/*` impedía que MkDocs copiara los assets del
tema Material → la página servida quedaba sin estilos ("documentación rota").
La solución: el contenido curado vive en docs_site/ y NO se usa exclude_docs.

Estos tests garantizan que `make docs` produce un sitio con los assets del
tema, las 6 páginas curadas y sin el footer "Made with Material".
"""

import pathlib
import subprocess
import sys

_ROOT = pathlib.Path(__file__).resolve().parents[1]
_SITE = _ROOT / "site"


def _build() -> None:
    subprocess.run(
        [sys.executable, "-m", "mkdocs", "build", "-f", str(_ROOT / "mkdocs.yml")],
        cwd=_ROOT, check=True, capture_output=True, text=True,
    )


def test_sitio_incluye_assets_del_tema():
    _build()
    css = sorted((_SITE / "assets" / "stylesheets").glob("*.min.css"))
    js = sorted((_SITE / "assets" / "javascripts").glob("bundle.*.min.js"))
    assert css, "faltan los stylesheets de Material en site/ (docs rota)"
    assert js, "faltan los javascripts de Material en site/"
    assert (_SITE / "assets" / "extra.css").exists(), "falta extra.css (estética del sitio)"


def test_sitio_tiene_las_6_paginas_curadas():
    _build()
    # index.md es la raíz del sitio; el resto viven en su subcarpeta.
    assert (_SITE / "index.html").exists(), "falta la portada (index.md)"
    for p in ["modelos-pesos", "riesgo-personalizacion", "arquitectura", "papers", "llm"]:
        assert (_SITE / p / "index.html").exists(), f"falta la página curada {p}"


def test_sin_footer_made_with_material():
    _build()
    for html in (_SITE / "index.html", _SITE / "modelos-pesos" / "index.html"):
        txt = html.read_text(encoding="utf-8")
        assert "Made with" not in txt, f"footer de Material presente en {html}"
        assert "cs-footer" in txt, f"falta el footer del sitio en {html}"
