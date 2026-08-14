"""Guardia de vendor/ de la demo (WEB-012).

Bug 2026-08-14: la demo fallaba con "no available backend found" porque
onnxruntime-web no encontraba sus ficheros: faltaban ort-wasm-simd-threaded.jsep.mjs
y .jsep.wasm en vendor/, y `wasmPaths = "./vendor/"` duplicaba el prefijo
(vendor/vendor/...) al resolverse relativo al directorio del script.

Estos tests garantizan que vendor/ contiene todos los ficheros que ort.min.js
referencia y que main.js no vuelve a prefijar wasmPaths.
"""

import pathlib
import re

_ROOT = pathlib.Path(__file__).resolve().parents[1]
_VENDOR = _ROOT / "web" / "probar-ya" / "vendor"


def _referenciados() -> list[str]:
    ort = (_VENDOR / "ort.min.js").read_text(encoding="utf-8")
    return sorted(set(re.findall(r"ort-wasm[a-zA-Z0-9._-]*\.(?:wasm|mjs)", ort)))


def test_vendor_tiene_todos_los_ficheros_onnxruntime():
    refs = _referenciados()
    assert refs, "no se encontraron ficheros ort-wasm referenciados en ort.min.js"
    faltan = [f for f in refs if not (_VENDOR / f).exists()]
    assert not faltan, f"faltan en vendor/: {faltan}"


def test_vendor_incluye_jsep():
    assert (_VENDOR / "ort-wasm-simd-threaded.jsep.mjs").exists(), "falta el jsep.mjs (import dinámico)"
    assert (_VENDOR / "ort-wasm-simd-threaded.jsep.wasm").exists(), "falta el jsep.wasm"


def test_wasm_paths_no_duplica_vendor():
    main = (_ROOT / "web" / "probar-ya" / "js" / "main.js").read_text(encoding="utf-8")
    assert 'wasmPaths = "./vendor/"' not in main, "wasmPaths con prefijo duplicado reintroducido"


def test_demo_formulario_tiene_los_campos_del_bot_y_mcp():
    """La demo recoge los mismos campos que piden el bot de Telegram y el MCP."""
    html = (_ROOT / "web" / "probar-ya" / "index.html").read_text(encoding="utf-8")
    for campo in ("falta_sueno", "enfermedad_reciente", "fiesta", "ocupacion",
                  "comorb", "farmaco", "social", "fototipo", "entrenado", "aclimatado"):
        assert campo in html, f"falta el campo '{campo}' en el formulario de la demo"


def test_demo_nav_y_volver_apuntan_fuera_de_probar_ya():
    """Los enlaces de la demo son relativos y vuelven al proyecto/home, no a README."""
    html = (_ROOT / "web" / "probar-ya" / "index.html").read_text(encoding="utf-8")
    assert 'href="../../projects/climasafe.html"' in html, "falta el botón Volver al proyecto"
    assert 'href="../../index.html"' in html, "el '← index' no apunta al home real"
    assert "cómo regenerar / desplegar" not in html, "el botón sin sentido sigue en el hero"
