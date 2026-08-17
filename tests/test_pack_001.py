"""PACK-001 — Empaquetado y presentación: versión única.

La versión sale de una sola fuente (pyproject.toml); `chat/app.py` y
`chat/entrypoint.sh` la consumen sin hardcodear.
"""

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


def test_entrypoint_usa_version_de_pyproject():
    """entrypoint.sh no hardcodea la versión: la extrae de pyproject.toml."""
    script = (PROJECT_DIR / "chat" / "entrypoint.sh").read_text()
    assert "v0.0.1" not in script
    assert "VERSION=" in script
    assert "pyproject.toml" in script
