"""agents.tests.conftest — Fixtures compartidas para los tests del sistema de agentes."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from agents.context import SharedContext
from agents.config import ProjectConfig


@pytest.fixture
def project_root(tmp_path: Path) -> Path:
    """
    Un proyecto mínimo en un directorio temporal: estructura de carpetas del
    template + un repo git real inicializado (para los tests de GitAgent).
    No reutiliza el repo real de este template a propósito, para que los
    tests no dependan del estado de git de quien los ejecute.
    """
    (tmp_path / "data" / "raw").mkdir(parents=True)
    (tmp_path / "models").mkdir()
    (tmp_path / "reports" / "figures").mkdir(parents=True)
    (tmp_path / "tests").mkdir()
    (tmp_path / "mi_paquete").mkdir()

    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=tmp_path, check=True)
    (tmp_path / "README.md").write_text("# Proyecto de prueba\n")
    subprocess.run(["git", "add", "."], cwd=tmp_path, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "chore: commit inicial"], cwd=tmp_path, check=True)

    return tmp_path


@pytest.fixture
def context(project_root: Path) -> SharedContext:
    config = ProjectConfig(project_slug="mi_paquete")
    return SharedContext(root=project_root, config=config)
