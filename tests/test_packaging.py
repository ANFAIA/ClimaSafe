"""
tests/test_packaging.py — Verificaciones estáticas del empaquetado (PACK-002).

La máquina de desarrollo no tiene demonio Docker (permiso denegado), así que
estos tests validan lo automatizable sin daemon:

  - Dockerfile multi-stage que sirve el servicio (web/bot), no el despliegue
    interno de agentes, con los paths que copia presentes en el repo.
  - Sin secretos (.env) ni en Dockerfile ni en el contexto (.dockerignore).
  - docker-compose.yml válido: web por defecto, bot tras perfil explícito.
  - Instalable por pip: entry points declarados y README referenciado.

El build/push real queda como acción humana documentada en
documentacion/despliegue/packaging.md (criterios de PACK-002).
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

try:
    import yaml

    HAVE_YAML = True
except ModuleNotFoundError:  # pragma: no cover - yaml viene con el entorno dev
    HAVE_YAML = False

ROOT = Path(__file__).resolve().parents[1]
DOCKERFILE = ROOT / "Dockerfile"
COMPOSE = ROOT / "docker-compose.yml"
IGNOREFILE = ROOT / ".dockerignore"


# ---------------------------------------------------------------------------
# Criterio 2 y 3 — imagen Docker del servicio (revierte el descarte de MSG-004)
# ---------------------------------------------------------------------------
def test_dockerfile_existe_y_es_multistage() -> None:
    assert DOCKERFILE.is_file(), "Falta Dockerfile en la raíz"
    content = DOCKERFILE.read_text(encoding="utf-8")
    stages = re.findall(r"^FROM\s+(\S+)", content, flags=re.MULTILINE)
    assert len(stages) >= 2, f"Se esperaba multi-stage, hay {len(stages)} FROM"


def test_dockerfile_sirve_el_servicio_no_el_arnes() -> None:
    content = DOCKERFILE.read_text(encoding="utf-8")
    # Sirve la interfaz web existente (entrypoint dskit → uvicorn chat.app:app)
    assert "chat/entrypoint.sh" in content
    # NO embarca el estado del arnés ni el pipeline de entrenamiento
    for prohibido in ("progress", "featureslist.json", ".opencode", "main.py"):
        copy_lines = [
            ln
            for ln in content.splitlines()
            if ln.strip().startswith(("COPY", "ADD")) and prohibido in ln
        ]
        assert not copy_lines, f"La imagen no debe llevar {prohibido}: {copy_lines}"


def test_dockerfile_copy_paths_existen() -> None:
    """Cada fuente de COPY (sin --from=builder) existe en el repo.

    Los globs tipo `models/*.joblib` son COPY válido en Docker: aquí se
    expanden y deben coincidir con al menos un fichero real.
    """
    content = DOCKERFILE.read_text(encoding="utf-8")
    for line in content.splitlines():
        stripped = line.strip()
        if not stripped.startswith("COPY") or "--from=" in stripped:
            continue
        parts = [p for p in stripped.split()[1:] if not p.startswith("--")]
        *sources, _dest = parts
        for src in sources:
            matches = sorted(ROOT.glob(src)) if any(c in src for c in "*?[") else [ROOT / src]
            assert matches, f"COPY referencia una ruta sin coincidencias: {src}"
            for m in matches:
                assert m.exists(), f"COPY referencia una ruta inexistente: {src}"


def test_dockerfile_sin_secretos() -> None:
    content = DOCKERFILE.read_text(encoding="utf-8")
    for ln in content.splitlines():
        s = ln.strip()
        if s.startswith(("COPY", "ADD")):
            assert ".env" not in s, f"Secreto en la imagen: {s}"
            assert not re.search(r"\b(TOKEN|API_KEY|PASSWORD)\s*=", s), f"Secreto inline: {s}"


def test_dockerignore_corta_secretos_y_pesados() -> None:
    assert IGNOREFILE.is_file(), "Falta .dockerignore"
    rules = IGNOREFILE.read_text(encoding="utf-8").splitlines()
    assert ".env" in rules, ".dockerignore debe bloquear .env"
    joined = "\n".join(rules)
    for necesario in ("data/raw", "models/*", ".venv", "web"):
        assert necesario in joined, f".dockerignore debe excluir {necesario}"


@pytest.mark.skipif(not HAVE_YAML, reason="pyyaml no instalado")
def test_compose_valido_web_por_defecto_bot_en_perfil() -> None:
    data = yaml.safe_load(COMPOSE.read_text(encoding="utf-8"))
    services = data["services"]
    # Web arranca con `docker compose up -d` a secas (sin perfil)
    assert "profiles" not in services["web"]
    # El bot solo con perfil explícito y exige token por entorno, jamás hardcodeado
    bot = services["bot"]
    assert "bot" in bot.get("profiles", [])
    token = bot.get("environment", {}).get("TELEGRAM_BOT_TOKEN", "")
    assert "${TELEGRAM_BOT_TOKEN:" in token, "el token debe venir del .env, no por defecto"
    assert ":" not in re.sub(r"\$\{[^}]*\}", "", token), "token hardcodeado detectado"


# ---------------------------------------------------------------------------
# Criterio 1 — instalable por pip para usuarios técnicos
# ---------------------------------------------------------------------------
def test_wheel_declara_entry_points_y_readme() -> None:
    try:
        import tomllib
    except ModuleNotFoundError:  # pragma: no cover
        import tomli as tomllib  # type: ignore[no-redef]

    data = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    scripts = data["project"]["scripts"]
    assert "climasafeai-bot" in scripts
    assert "climasafeai-mcp" in scripts
    assert (ROOT / data["project"]["readme"]).is_file()
    assert data["project"]["version"], "versión única en pyproject.toml"
