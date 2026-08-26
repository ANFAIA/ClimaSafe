"""Unidades puras JS del modo LLM local (WEB-016): js/llm.js en node.

Corre web/probar-ya/test/llm_unit.mjs, que prueba SIN red ni DOM las piezas
testables de la integración transformers.js:
  - elección device/dtype (webgpu→q4f16, wasm→q4)
  - construcción del contexto a partir del resultado del pipeline ML
  - construcción de los mensajes de chat (system prohíbe inventar cifras)
  - limpieza de la salida del LLM

La ejecución real del modelo en el navegador NO es automatizable aquí: se
documenta en documentacion/wasm/llm_navegador.md (verificación manual).
"""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
TEST_DIR = ROOT / "web" / "probar-ya" / "test"
NODE = shutil.which("node")

pytestmark = pytest.mark.skipif(NODE is None, reason="node no disponible en este entorno")


def test_unidades_llm_js() -> None:
    proc = subprocess.run(
        [NODE, "llm_unit.mjs"],
        cwd=TEST_DIR,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert proc.returncode == 0, (
        f"node llm_unit.mjs falló (rc={proc.returncode})\n"
        f"stdout: {proc.stdout}\nstderr: {proc.stderr}"
    )
    print(proc.stdout)
    assert "pruebas de unidades LLM en verde" in proc.stdout
