"""PACK-001 — Empaquetado y presentación: versión única.

La versión sale de una sola fuente (pyproject.toml); `chat/app.py` y
`chat/entrypoint.sh` la consumen sin hardcodear.
"""

import re
import tomllib

from chat.app import PROJECT_DIR, _VERSION


def _version_pyproject() -> str:
    with open(PROJECT_DIR / "pyproject.toml", "rb") as fh:
        return tomllib.load(fh)["project"]["version"]


def test_version_app_coincide_con_pyproject():
    """La versión que expone /api/status es la de pyproject.toml."""
    assert _VERSION == _version_pyproject()


def test_version_app_no_esta_hardcodeada():
    """El valor viejo hardcodeado (0.0.1) ya no existe."""
    assert _VERSION != "0.0.1"


def test_app_no_tiene_versiones_hardcodeadas():
    """DEPLOY-001: ningún literal de versión fuera del fallback de _resolve_version.

    El único "0.0.x" permitido en chat/app.py es "0.0.0" (el fallback de
    _resolve_version cuando no hay paquete instalado ni pyproject.toml).
    Cualquier otro literal es una segunda fuente de verdad y falla.
    """
    src = (PROJECT_DIR / "chat" / "app.py").read_text()
    for literal in re.findall(r'"0\.0\.\d+"', src):
        assert literal == '"0.0.0"', f"Versión hardcodeada en chat/app.py: {literal}"


def test_fastapi_version_usa_resolve_version():
    """La versión del FastAPI app es la resuelta, no un literal."""
    from chat.app import app

    assert app.version == _VERSION


def test_welcome_message_usa_resolve_version():
    """El mensaje de bienvenida muestra la versión resuelta."""
    from chat.app import _welcome_message

    assert f"`{_VERSION}`" in _welcome_message()


def test_entrypoint_usa_version_de_pyproject():
    """entrypoint.sh no hardcodea la versión: la extrae de pyproject.toml."""
    script = (PROJECT_DIR / "chat" / "entrypoint.sh").read_text()
    assert "v0.0.1" not in script
    assert "VERSION=" in script
    assert "pyproject.toml" in script
