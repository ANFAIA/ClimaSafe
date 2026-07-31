"""
GIT-001 — Al cerrar una feature: bump de versión en README y propuesta de
commit sin co-autoría.

El flujo es: `harness finish` cierra la feature, sube un punto de versión
patch (pyproject.toml + README) y deja lista una propuesta de commit
Conventional Commits sin línea de co-autoría. El commit NO se ejecuta solo:
lo aprueba el humano y entonces se ejecuta tal cual.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

from agents.agents.git_agent import GitAgent
from agents.agents.harness_agent import HarnessAgent
from agents.config import ProjectConfig
from agents.context import SharedContext
from agents.tools.git_tool import GitTool


def _write_backlog(root: Path) -> None:
    (root / "featureslist.json").write_text(json.dumps({
        "features": [{
            "id": "GIT-001",
            "title": "Cierre con release",
            "description": "bump + propuesta al cerrar",
            "acceptance_criteria": ["el README sube de versión"],
            "status": "in_progress",
            "started": "2026-07-31",
        }],
    }, ensure_ascii=False), encoding="utf-8")


def _write_fake_gate(root: Path) -> None:
    """init.sh mínimo que la puerta de `harness finish` pueda leer (JSON)."""
    (root / "init.sh").write_text(
        '#!/usr/bin/env bash\n'
        'echo \'{"ready": true, "checks": [{"check": "pytest", "status": "ok", "detail": "3 passed"}]}\'\n',
        encoding="utf-8",
    )


def _write_versioned_files(root: Path) -> None:
    (root / "pyproject.toml").write_text('[project]\nname = "x"\nversion = "0.1.0"\n', encoding="utf-8")
    (root / "README.md").write_text(
        "![Version](https://img.shields.io/badge/Version-0.1.0-green)\n\n**Versión:** 0.1.0\n",
        encoding="utf-8",
    )


def _harness_ctx(root: Path) -> SharedContext:
    return SharedContext(root=root, config=ProjectConfig(project_slug="mi_paquete"))


def test_finish_bumps_patch_version_and_suggests_commit(project_root):
    _write_backlog(project_root)
    _write_fake_gate(project_root)
    _write_versioned_files(project_root)
    # los ficheros nuevos son untracked y `git diff` no los ve: se añaden,
    # igual que los cambios del implementer están en el índice al cerrar
    subprocess.run(["git", "add", "-A"], cwd=project_root, check=True)

    result = HarnessAgent(context=_harness_ctx(project_root)).finish(id="GIT-001", evidence="salida real")

    assert result.success
    # criterio 1: pyproject y README suben un punto de versión patch
    assert 'version = "0.1.1"' in (project_root / "pyproject.toml").read_text()
    readme = (project_root / "README.md").read_text()
    assert "Version-0.1.1-green" in readme
    assert "**Versión:** 0.1.1" in readme
    assert result.data["version_bump"]["new_version"] == "0.1.1"

    # criterio 2: el agente git propone un commit Conventional Commits sin co-autoría
    suggestion = result.data["commit_suggestion"]
    assert GitTool.parse_conventional_commit(suggestion) is not None
    assert "GIT-001" in suggestion
    assert "Co-authored-by" not in suggestion


def test_finish_without_version_still_closes_with_warning(project_root):
    _write_backlog(project_root)
    _write_fake_gate(project_root)
    subprocess.run(["git", "add", "-A"], cwd=project_root, check=True)
    # sin pyproject.toml versionable: el cierre no se bloquea, solo avisa

    result = HarnessAgent(context=_harness_ctx(project_root)).finish(id="GIT-001", evidence="salida real")

    assert result.success
    assert result.data["id"] == "GIT-001"
    assert any("No se pudo subir la versión" in w for w in result.warnings)
    # la propuesta de commit sí se pudo generar (hay cambios: backlog + progress)
    assert "GIT-001" in result.data["commit_suggestion"]


def test_next_patch_version():
    assert HarnessAgent._next_patch_version("0.1.0") == "0.1.1"
    assert HarnessAgent._next_patch_version("2.10.3") == "2.10.4"
    assert HarnessAgent._next_patch_version("0.1") is None
    assert HarnessAgent._next_patch_version(None) is None


def test_suggest_commit_message_with_hint_uses_hint(context):
    (context.root / "mi_paquete" / "nuevo.py").write_text("x = 1\n")
    subprocess.run(["git", "add", "."], cwd=context.root, check=True)
    result = GitAgent(context=context).suggest_commit_message(hint="cierra TEST-001")

    assert result.success
    assert result.data["suggested_message"].endswith("cierra TEST-001")
    assert not result.warnings  # con hint real no hay aviso de placeholder


def test_approved_suggestion_is_committed_exactly_without_coauthorship(context):
    """Criterio 3: si el usuario aprueba la propuesta, el commit se ejecuta tal cual."""
    (context.root / "mi_paquete" / "nuevo.py").write_text("x = 1\n")
    subprocess.run(["git", "add", "."], cwd=context.root, check=True)
    agent = GitAgent(context=context)

    suggestion = agent.suggest_commit_message(hint="cierra TEST-001")
    assert suggestion.success
    message = suggestion.data["suggested_message"]

    # el usuario aprueba la propuesta sin tocarla → el commit usa ese mensaje exacto
    commit = agent.commit_with_changelog(message=message)
    assert commit.success

    log = subprocess.run(
        ["git", "log", "-1", "--format=%B"], cwd=context.root,
        capture_output=True, text=True, check=True,
    ).stdout
    assert log.strip() == message
    assert "Co-authored-by" not in log
