"""Tests de CI/CD (DEPLOY-002).

Cubre:
  1. Validez y estructura de los workflows de GitHub Actions (ci, release,
     pages, paper_scout).
  2. release_notes.py: agrupación por tipo Conventional Commits y modo
     release-notes.
  3. release_ci.sh: idempotencia, tag LIGERO (sin autor github-actions), no
     commitea en el repo, CHANGELOG intacto por defecto.
  4. pages_deploy.sh (dry-run): produce projects/climasafe/documentation/ +
     projects/climasafe-src/ en el repo destino,
     sin push.
  5. Targets del Makefile: lint, docs, pages-deploy-dry, format-check.
"""

from __future__ import annotations

import os
import pathlib
import re
import shutil
import subprocess
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
WORKFLOWS = ROOT / ".github" / "workflows"
SCRIPTS = ROOT / "scripts"

yaml = pytest.importorskip("yaml")


# ─────────────────────────────────────────────────────────────────────────────
# 1. Workflows de GitHub Actions
# ─────────────────────────────────────────────────────────────────────────────

def _cargar_workflow(nombre: str) -> dict:
    with open(WORKFLOWS / nombre, encoding="utf-8") as f:
        return yaml.safe_load(f)


@pytest.mark.parametrize("nombre", ["ci.yml", "release.yml", "pages.yml", "paper_scout.yml"])
def test_workflows_yaml_validos(nombre):
    cfg = _cargar_workflow(nombre)
    assert isinstance(cfg, dict), f"{nombre} no es YAML válido de workflow"
    assert "jobs" in cfg, f"{nombre} sin jobs"
    assert ("on" in cfg or True in cfg), f"{nombre} sin triggers (on)"


def test_ci_triggers_y_pasos():
    cfg = _cargar_workflow("ci.yml")
    # PyYAML 1.1: la clave "on" se parsea como True.
    on = cfg.get("on") or cfg.get(True)
    assert "pull_request" in on and "push" in on, "ci.yml debe correr en PR y push"
    job = cfg["jobs"]["quality"]
    pasos = " ".join(str(s.get("name", "")) + " " + str(s.get("run", "")) for s in job["steps"])
    assert "make test" in pasos, "ci.yml debe ejecutar make test"
    assert "ruff check" in pasos, "ci.yml debe lintear los .py cambiados"


def test_release_no_commitea_bot():
    cfg = _cargar_workflow("release.yml")
    run = " ".join(str(s.get("run", "")) for s in cfg["jobs"]["release"]["steps"])
    assert "release_ci.sh" in run, "release.yml debe llamar a release_ci.sh"
    # Regla de oro (2026-08-14): CI no configura identidad de git (el tag es
    # ligero y no se commitea nada), así que el bot jamás firma.
    script = (SCRIPTS / "release_ci.sh").read_text(encoding="utf-8")
    assert "GIT_NAME" not in script, "release_ci.sh no debe definir identidad de bot"
    assert "user.name" not in script, "release_ci.sh no debe configurar user.name"


def test_release_solo_en_repo_anfaia():
    """DEPLOY-003: el release (tags/versiones) vive SOLO en ANFAIA/ClimaSafe.
    Si el repo se espeja a otro remoto, el job debe saltarse, no etiquetar."""
    cfg = _cargar_workflow("release.yml")
    job = cfg["jobs"]["release"]
    assert job.get("if") == "github.repository == 'ANFAIA/ClimaSafe'", (
        "release.yml debe acotarse al repo ANFAIA con un guard de repository"
    )


def test_no_release_please():
    """DEPLOY-003: release-please se evaluó y se descartó (sería un segundo
    escritor de la versión en pyproject.toml y commitea con identidad de bot).
    Guard contra reintroducirlo por accidente: no debe USARSE en ningún
    workflow ni existir su configuración de manifiesto."""
    for yml in WORKFLOWS.glob("*.yml"):
        contenido = yml.read_text(encoding="utf-8")
        assert not re.search(r"uses:\s*.*release-please", contenido), (
            f"{yml.name} usa release-please — releer documentacion/despliegue/releases.md"
        )
    for config in ("release-please-config.json", ".release-please-manifest.json"):
        assert not (ROOT / config).exists(), f"{config} no debe existir"


def test_pages_usa_secreto_y_script():
    cfg = _cargar_workflow("pages.yml")
    run = " ".join(str(s.get("run", "")) for s in cfg["jobs"]["pages"]["steps"])
    assert "pages_deploy.sh" in run, "pages.yml debe llamar a pages_deploy.sh"
    assert "PAGES_DEPLOY_TOKEN" in run or "secrets.PAGES_DEPLOY_TOKEN" in run


# ─────────────────────────────────────────────────────────────────────────────
# 2. release_notes.py
# ─────────────────────────────────────────────────────────────────────────────


def _git(*args, cwd: pathlib.Path) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True)


@pytest.fixture()
def repo_git(tmp_path: pathlib.Path):
    """Repo git temporal: commit base + tag v0.0.1 + 3 commits Conventional."""
    _git("init", "-b", "main", cwd=tmp_path)
    _git("config", "user.name", "test", cwd=tmp_path)
    _git("config", "user.email", "test@example.com", cwd=tmp_path)
    (tmp_path / "base.txt").write_text("base", encoding="utf-8")
    _git("add", ".", cwd=tmp_path)
    _git("commit", "-m", "chore: base", cwd=tmp_path)
    _git("tag", "v0.0.1", cwd=tmp_path)
    for msg in ["feat: primera feature", "fix: arreglo un bug", "docs: documenta algo"]:
        (tmp_path / "f.txt").write_text(msg + "\n", encoding="utf-8")
        _git("add", ".", cwd=tmp_path)
        _git("commit", "-m", msg, cwd=tmp_path)
    return tmp_path


def test_release_notes_agrupa_por_tipo(repo_git):
    res = subprocess.run(
        [sys.executable, str(SCRIPTS / "release_notes.py"), "v0.0.1..HEAD", "0.0.2"],
        cwd=repo_git, capture_output=True, text=True,
    )
    assert res.returncode == 0, res.stderr
    assert "### Añadido" in res.stdout and "- primera feature" in res.stdout
    assert "### Corrección de bugs" in res.stdout and "- arreglo un bug" in res.stdout
    assert "### Documentación" in res.stdout and "- documenta algo" in res.stdout


def test_release_notes_modo_release(repo_git):
    res = subprocess.run(
        [sys.executable, str(SCRIPTS / "release_notes.py"), "v0.0.1..HEAD", "0.0.2", "--release-notes"],
        cwd=repo_git, capture_output=True, text=True,
    )
    assert res.returncode == 0
    assert "v0.0.2" in res.stdout  # cabecera de release notes


# ─────────────────────────────────────────────────────────────────────────────
# 3. release_ci.sh
# ─────────────────────────────────────────────────────────────────────────────


def _release_ci_en_repo_temporal(repo_git: pathlib.Path) -> subprocess.CompletedProcess:
    """release_ci.sh se ejecuta sobre su propia ubicación: se copia al repo
    temporal junto a release_notes.py para probarlo de forma aislada."""
    (repo_git / "scripts").mkdir(exist_ok=True)
    shutil.copy(SCRIPTS / "release_ci.sh", repo_git / "scripts" / "release_ci.sh")
    shutil.copy(SCRIPTS / "release_notes.py", repo_git / "scripts" / "release_notes.py")
    env = dict(os.environ, PUSH="no", GITHUB_TOKEN="")
    return subprocess.run(
        ["bash", "scripts/release_ci.sh"], cwd=repo_git,
        capture_output=True, text=True, env=env,
    )


def test_release_ci_tag_ligero_sin_commit_ni_changelog(repo_git):
    (repo_git / "pyproject.toml").write_text(
        '[project]\nversion = "0.0.2"\n', encoding="utf-8")
    _git("add", ".", cwd=repo_git)
    _git("commit", "-m", "chore: version 0.0.2", cwd=repo_git)
    commits_antes = len(_git("log", "--oneline", cwd=repo_git).stdout.splitlines())
    changelog_antes = (repo_git / "CHANGELOG.md").read_text() if (repo_git / "CHANGELOG.md").exists() else None

    res = _release_ci_en_repo_temporal(repo_git)
    assert res.returncode == 0, res.stderr

    # Tag LIGERO (apunta al commit, sin objeto de tag ni autor bot)
    tipo = _git("cat-file", "-t", "v0.0.2", cwd=repo_git).stdout.strip()
    assert tipo == "commit", f"el tag debe ser ligero, es {tipo!r}"
    # No hay commits nuevos
    commits_despues = len(_git("log", "--oneline", cwd=repo_git).stdout.splitlines())
    assert commits_despues == commits_antes, "release_ci.sh no debe commitear en el repo"
    # CHANGELOG intacto por defecto (UPDATE_CHANGELOG no activado)
    changelog_despues = (repo_git / "CHANGELOG.md").read_text() if (repo_git / "CHANGELOG.md").exists() else None
    assert changelog_despues == changelog_antes, "sin UPDATE_CHANGELOG=yes no debe tocar CHANGELOG.md"


def test_release_ci_idempotente(repo_git):
    (repo_git / "pyproject.toml").write_text(
        '[project]\nversion = "0.0.2"\n', encoding="utf-8")
    _git("add", ".", cwd=repo_git)
    _git("commit", "-m", "chore: version 0.0.2", cwd=repo_git)
    r1 = _release_ci_en_repo_temporal(repo_git)
    assert r1.returncode == 0, r1.stderr
    r2 = _release_ci_en_repo_temporal(repo_git)
    assert r2.returncode == 0, r2.stderr
    assert "ya existe" in r2.stdout, "la segunda ejecución debe ser idempotente"


# ─────────────────────────────────────────────────────────────────────────────
# 4. pages_deploy.sh (dry-run, sin push)
# ─────────────────────────────────────────────────────────────────────────────


def test_pages_deploy_dry_run_layout(tmp_path: pathlib.Path):
    dest = tmp_path / "pages"
    dest.mkdir()
    _git("init", "-b", "main", cwd=dest)
    _git("config", "user.name", "test", cwd=dest)
    _git("config", "user.email", "test@example.com", cwd=dest)
    (dest / "index.html").write_text("<h1>home</h1>", encoding="utf-8")
    _git("add", ".", cwd=dest)
    _git("commit", "-m", "init", cwd=dest)

    env = dict(os.environ, PAGES_DIR=str(dest), PAGES_REMOTE="origin", PUSH="no")
    res = subprocess.run(
        ["bash", str(SCRIPTS / "pages_deploy.sh")],
        cwd=ROOT, capture_output=True, text=True, env=env,
    )
    assert res.returncode == 0, res.stderr

    # Layout esperado en el repo destino
    assert (dest / "projects" / "climasafe" / "documentation" / "index.html").exists(), "docs no copiadas"
    assert (dest / "projects" / "climasafe-src" / "mkdocs.yml").exists(), "fuente de docs no copiada"
    assert (dest / "projects" / "climasafe-src" / "overrides" / "main.html").exists(), "overrides no copiados"
    # El home del repo destino sigue intacto
    assert (dest / "index.html").read_text() == "<h1>home</h1>"
    # PUSH=no hace el commit en local sin push: el árbol queda limpio pero el
    # commit "deploy(climasafe)" debe existir.
    log = _git("log", "--oneline", "-1", cwd=dest).stdout
    assert "deploy(climasafe)" in log, f"falta el commit de deploy: {log}"


# ─────────────────────────────────────────────────────────────────────────────
# 5. Makefile
# ─────────────────────────────────────────────────────────────────────────────


def test_makefile_targets_existen():
    mk = (ROOT / "Makefile").read_text(encoding="utf-8")
    for target in ["test:", "lint:", "docs:", "pages-deploy-dry:", "format-check:"]:
        assert target in mk, f"falta el target {target} en el Makefile"


def test_no_queda_deuda_gitignore_site():
    # La documentación rota (2026-08-14) dejaba site/ sin versionar; el guard
    # de docs ya cubre el build. site/ debe estar gitignored.
    res = subprocess.run(["git", "check-ignore", "-v", "site/"], cwd=ROOT,
                         capture_output=True, text=True)
    assert res.returncode == 0, "site/ no está en .gitignore"
