"""
GIT-001 + ARNES-014 — Al cerrar una feature: bump de versión en README y
commit automático del cierre SOLO con el flag explícito `--commit`, sin
co-autoría.

El flujo es: `harness finish` cierra la feature, sube un punto de versión
patch (pyproject.toml + README) y propone el mensaje de commit. El commit
automático SOLO ocurre cuando el lider pasa `--commit` y, aun así, acotado a
las rutas del ticket (`--changes` + los ficheros del propio cierre) con
mensaje Conventional Commits sin línea de co-autoría. Sin el flag, `finish`
propone el mensaje y NO commitea — ningún otro agente ni asistente commitea
por su cuenta. Si `--changes` viene vacío o trae rutas inexistentes, se avisa
y no se commitea; los cambios ajenos al ticket se avisan y el commit continúa
acotado a las rutas del ticket + cierre, dejando lo ajeno sin tocar.
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


# ── ARNES-014: commit automático al cerrar, acotado a las rutas del ticket ──

def _head(root: Path) -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True, check=True,
    ).stdout.strip()


def _setup_ticket(project_root: Path, changes: str = "mi_paquete/nuevo.py") -> None:
    """
    Estado sano para el commit automático: backlog + puerta + versión + una ruta
    del ticket. `init.sh` es infraestructura del test, no del ticket: se
    commitea aparte para que el working tree solo tenga cambios autorizados.
    """
    _write_backlog(project_root)
    _write_fake_gate(project_root)
    _write_versioned_files(project_root)
    (project_root / changes.split(";")[0]).parent.mkdir(parents=True, exist_ok=True)
    (project_root / changes.split(";")[0]).write_text("x = 1\n")
    subprocess.run(["git", "add", "init.sh"], cwd=project_root, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "chore: puerta de test"], cwd=project_root, check=True)


def test_finish_without_commit_flag_proposes_but_does_not_commit(project_root):
    """Sin el flag --commit, finish propone el mensaje (GIT-001) pero NO commitea (ARNES-014)."""
    _setup_ticket(project_root)
    before = _head(project_root)

    result = HarnessAgent(context=_harness_ctx(project_root)).finish(
        id="GIT-001", evidence="salida real", changes="mi_paquete/nuevo.py",
    )

    assert result.success
    assert result.data["id"] == "GIT-001"
    # propone el mensaje, como antes de ARNES-014…
    assert "GIT-001" in result.data["commit_suggestion"]
    # …pero no hay commit automático y HEAD no avanza
    assert "auto_commit" not in result.data
    assert _head(project_root) == before


def test_finish_with_commit_flag_commits_only_ticket_paths(project_root):
    """Con --commit, finish commitea SOLO --changes + los ficheros del cierre, y el mensaje va sin co-autoría."""
    _setup_ticket(project_root)
    before = _head(project_root)

    result = HarnessAgent(context=_harness_ctx(project_root)).finish(
        id="GIT-001", evidence="salida real", changes="mi_paquete/nuevo.py", commit=True,
    )

    assert result.success
    auto = result.data["auto_commit"]
    assert auto["success"]
    assert _head(project_root) != before  # con el flag sí commitea

    # el commit incluye las rutas del ticket + las del cierre, y nada más
    files = set(auto["files"])
    assert "mi_paquete/nuevo.py" in files
    assert "featureslist.json" in files
    assert any(f.startswith("progress/") for f in files)
    assert "pyproject.toml" in files
    assert "README.md" in files
    assert "init.sh" not in files
    assert all(f in {"mi_paquete/nuevo.py", "featureslist.json", "pyproject.toml", "README.md"}
               or f.startswith("progress/") for f in files)

    # mensaje Conventional Commits, id como subject, sin co-autoría
    assert GitTool.parse_conventional_commit(auto["message"]) is not None
    assert "GIT-001" in auto["message"]
    assert "Co-authored-by" not in auto["message"]

    log = subprocess.run(
        ["git", "log", "-1", "--format=%B"], cwd=project_root,
        capture_output=True, text=True, check=True,
    ).stdout
    assert log.strip() == auto["message"]  # el commit usa el mensaje exacto, sin pie de herramienta
    assert "Co-authored-by" not in log


def test_finish_empty_changes_does_not_commit(project_root):
    """--changes vacío, aun con --commit → no se commitea nada y se avisa; la feature se cierra igual."""
    _setup_ticket(project_root)
    before = _head(project_root)

    result = HarnessAgent(context=_harness_ctx(project_root)).finish(
        id="GIT-001", evidence="salida real", changes="", commit=True,
    )

    assert result.success
    assert result.data["id"] == "GIT-001"
    assert "auto_commit" not in result.data
    assert any("sin --changes" in w for w in result.warnings)
    assert _head(project_root) == before


def test_finish_missing_path_does_not_commit(project_root):
    """--changes con rutas inexistentes, aun con --commit → no se commitea nada y se avisa."""
    _setup_ticket(project_root)
    before = _head(project_root)

    result = HarnessAgent(context=_harness_ctx(project_root)).finish(
        id="GIT-001", evidence="salida real", commit=True,
        changes="mi_paquete/nuevo.py;mi_paquete/no_existe.py",
    )

    assert result.success
    assert "auto_commit" not in result.data
    assert any("no existen" in w for w in result.warnings)
    assert _head(project_root) == before


def test_finish_foreign_changes_do_not_block_the_commit(project_root):
    """Cambios ajenos en el árbol + --commit → se avisa y el commit continúa acotado; lo ajeno queda sin tocar."""
    _setup_ticket(project_root)
    # un cambio ajeno al ticket, sin declarar en --changes
    (project_root / "mi_paquete" / "ajeno.py").write_text("y = 2\n")
    before = _head(project_root)

    result = HarnessAgent(context=_harness_ctx(project_root)).finish(
        id="GIT-001", evidence="salida real", changes="mi_paquete/nuevo.py", commit=True,
    )

    assert result.success
    assert "auto_commit" in result.data
    assert any("fuera del ticket" in w for w in result.warnings)
    # el commit avanzó (solo con las rutas del ticket + cierre)
    assert _head(project_root) != before
    # lo ajeno sigue en el árbol, sin commitear
    status = subprocess.run(
        ["git", "status", "--porcelain", "-uall"], cwd=project_root,
        capture_output=True, text=True, check=True,
    ).stdout
    assert "mi_paquete/ajeno.py" in status
    # el commit no contiene la ruta ajena
    commit_files = subprocess.run(
        ["git", "show", "--name-only", "--format=", "HEAD"], cwd=project_root,
        capture_output=True, text=True, check=True,
    ).stdout
    assert "mi_paquete/ajeno.py" not in commit_files


def test_finish_without_flag_leaves_foreign_changes_untouched(project_root):
    """Sin --commit no hay commit de ningún tipo: ni el ticket ni lo ajeno se tocan."""
    _setup_ticket(project_root)
    (project_root / "mi_paquete" / "ajeno.py").write_text("y = 2\n")
    before = _head(project_root)

    result = HarnessAgent(context=_harness_ctx(project_root)).finish(
        id="GIT-001", evidence="salida real", changes="mi_paquete/nuevo.py",
    )

    assert result.success
    assert "auto_commit" not in result.data
    assert _head(project_root) == before


def test_auto_commit_nothing_staged_does_not_commit(project_root):
    """Tras filtrar no queda nada staged (rutas ya commiteadas) → no se commitea y se avisa."""
    _setup_ticket(project_root)
    # progress/ ya existe y está commiteado, como en un cierre previo
    (project_root / "progress").mkdir()
    (project_root / "progress" / ".keep").write_text("")
    subprocess.run(["git", "add", "-A"], cwd=project_root, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "chore: todo commiteado"], cwd=project_root, check=True)
    before = _head(project_root)

    result = HarnessAgent(context=_harness_ctx(project_root))._auto_commit({"id": "GIT-001"}, "mi_paquete/nuevo.py")

    assert "auto_commit" not in result["data"]
    assert any("nada staged" in w for w in result["warnings"])
    assert _head(project_root) == before
