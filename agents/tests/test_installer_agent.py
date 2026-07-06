from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from agents.agents.installer_agent import InstallerAgent
from agents.core.registry import agent_registry


VALID_AGENT_SOURCE = '''
from agents.core.base_agent import AgentResult, BaseAgent
from agents.core.registry import register_agent

@register_agent
class SaludoTestAgent(BaseAgent):
    name = "saludo_test"
    description = "agente de prueba"
    capabilities = ["saludo"]

    def actions(self):
        return {"saludar": self.saludar}

    def saludar(self):
        return AgentResult(True, self.name, "saludar", "hola")
'''

INCOMPLETE_AGENT_SOURCE = '''
from agents.core.base_agent import BaseAgent
from agents.core.registry import register_agent

@register_agent
class IncompletoAgent(BaseAgent):
    name = "incompleto_test"

    def actions(self):
        return {}
'''


@pytest.fixture(autouse=True)
def _cleanup_registry():
    yield
    agent_registry._agents.pop("saludo_test", None)
    agent_registry._agents.pop("incompleto_test", None)
    agent_registry._agents.pop("buena_adaptacion_test", None)
    agent_registry._agents.pop("mala_adaptacion_test", None)


def _make_external_repo(tmp_path: Path, source: str, filename: str = "agente.py") -> Path:
    repo_dir = tmp_path / "repo_externo"
    repo_dir.mkdir()
    (repo_dir / filename).write_text(source)
    subprocess.run(["git", "init", "-q"], cwd=repo_dir, check=True)
    subprocess.run(["git", "config", "user.email", "a@a.com"], cwd=repo_dir, check=True)
    subprocess.run(["git", "config", "user.name", "a"], cwd=repo_dir, check=True)
    subprocess.run(["git", "add", "."], cwd=repo_dir, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "init"], cwd=repo_dir, check=True)
    return repo_dir


def test_install_from_git_valid_agent_end_to_end_via_cli(tmp_path):
    """
    A diferencia de los demás tests de este archivo, este NO usa la fixture
    `context` de conftest.py: esa fixture crea una raíz de proyecto sintética
    que no coincide con la ubicación real del paquete `agents` que pytest ya
    tiene importado en este proceso — 'confirmar que quedó registrado' no
    puede comprobarse así, `agent_registry.discover()` solo ve el paquete
    `agents` real ya cargado, no uno hipotético en otra ruta.

    En producción esto sí es siempre así (la raíz del proyecto ES donde vive
    agents/, ver `context.py`) — lo comprobé manualmente contra un repo real
    antes de escribir este test. Aquí se reproduce esa misma realidad
    lanzando un proceso Python nuevo (`python -m agents ...`) sobre una
    copia real y completa de agents/, igual que se usaría de verdad.
    """
    import shutil
    import subprocess
    import sys

    agents_pkg_root = Path(__file__).resolve().parents[2]  # .../template
    project_dir = tmp_path / "proyecto_real"
    shutil.copytree(agents_pkg_root / "agents", project_dir / "agents", ignore=shutil.ignore_patterns("__pycache__"))

    repo = _make_external_repo(tmp_path, VALID_AGENT_SOURCE)

    result = subprocess.run(
        [sys.executable, "-m", "agents", "run", "installer", "install_from_git", "--repo_url", str(repo)],
        cwd=project_dir, capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "registrado como 'saludo_test'" in result.stdout
    assert (project_dir / "agents" / "external" / "agente.py").exists()

    verify = subprocess.run(
        [sys.executable, "-m", "agents", "run", "saludo_test", "saludar"],
        cwd=project_dir, capture_output=True, text=True, timeout=30,
    )
    assert verify.returncode == 0, verify.stdout + verify.stderr
    assert "hola" in verify.stdout


def test_install_from_git_warns_on_incomplete_structure(context, tmp_path):
    repo = _make_external_repo(tmp_path, INCOMPLETE_AGENT_SOURCE)
    agent = InstallerAgent(context=context)

    result = agent.install_from_git(repo_url=str(repo))

    assert result.success  # se instala igualmente — es un aviso, no un bloqueo
    assert any("description" in w for w in result.warnings)


def test_install_from_git_no_candidates_fails(context, tmp_path):
    repo_dir = tmp_path / "repo_vacio"
    repo_dir.mkdir()
    (repo_dir / "no_es_agente.py").write_text("x = 1\n")
    subprocess.run(["git", "init", "-q"], cwd=repo_dir, check=True)
    subprocess.run(["git", "config", "user.email", "a@a.com"], cwd=repo_dir, check=True)
    subprocess.run(["git", "config", "user.name", "a"], cwd=repo_dir, check=True)
    subprocess.run(["git", "add", "."], cwd=repo_dir, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "init"], cwd=repo_dir, check=True)

    agent = InstallerAgent(context=context)
    result = agent.install_from_git(repo_url=str(repo_dir))
    assert not result.success


def test_install_refuses_overwrite_without_force(context, tmp_path):
    repo = _make_external_repo(tmp_path, VALID_AGENT_SOURCE)
    agent = InstallerAgent(context=context)
    agent.install_from_git(repo_url=str(repo))

    result = agent.install_from_git(repo_url=str(repo))  # segunda vez, sin force
    assert not result.success


def test_verify_reports_unregistered_agent(context):
    agent = InstallerAgent(context=context)
    result = agent.verify(agent_name="no_existe_este_agente")
    assert not result.success


AGENT_USING_CTX = '''
from agents.core.base_agent import AgentResult, BaseAgent
from agents.core.registry import register_agent

@register_agent
class BuenaAdaptacionAgent(BaseAgent):
    name = "buena_adaptacion_test"
    description = "usa self.ctx"
    capabilities = ["x"]

    def actions(self):
        return {"hacer": self.hacer}

    def hacer(self):
        path = self.ctx.raw_data_dir / "archivo.csv"
        return AgentResult(True, self.name, "hacer", str(path))
'''

AGENT_WITH_HARDCODED_PATH = '''
from pathlib import Path
from agents.core.base_agent import AgentResult, BaseAgent
from agents.core.registry import register_agent

@register_agent
class MalaAdaptacionAgent(BaseAgent):
    name = "mala_adaptacion_test"
    description = "ruta fija"
    capabilities = ["x"]

    def actions(self):
        return {"hacer": self.hacer}

    def hacer(self):
        path = Path("data/raw/archivo.csv")
        return AgentResult(True, self.name, "hacer", str(path))
'''


def test_install_warns_about_hardcoded_paths_without_self_ctx(context, tmp_path):
    repo = _make_external_repo(tmp_path, AGENT_WITH_HARDCODED_PATH, filename="mala_adaptacion.py")
    agent = InstallerAgent(context=context)
    result = agent.install_from_git(repo_url=str(repo))

    assert result.success  # se instala igual — es un aviso, no un bloqueo
    assert any("self.ctx" in w for w in result.warnings)
    assert any("data/raw/archivo.csv" in w for w in result.warnings)


def test_install_no_path_warning_when_using_self_ctx(context, tmp_path):
    repo = _make_external_repo(tmp_path, AGENT_USING_CTX, filename="buena_adaptacion.py")
    agent = InstallerAgent(context=context)
    result = agent.install_from_git(repo_url=str(repo))

    assert result.success
    assert not any("self.ctx" in w for w in result.warnings)
    assert not any("Rutas literales" in w for w in result.warnings)


def test_normalize_github_shorthand_expands_user_repo():
    from agents.tools.agent_installer_tool import AgentInstallerTool
    assert AgentInstallerTool.normalize_github_shorthand("torvalds/linux") == "https://github.com/torvalds/linux.git"


def test_normalize_github_shorthand_leaves_full_url_untouched():
    from agents.tools.agent_installer_tool import AgentInstallerTool
    url = "https://github.com/torvalds/linux.git"
    assert AgentInstallerTool.normalize_github_shorthand(url) == url


def test_normalize_github_shorthand_leaves_existing_local_path_untouched(tmp_path):
    from agents.tools.agent_installer_tool import AgentInstallerTool
    local = tmp_path / "algo"
    local.mkdir()
    assert AgentInstallerTool.normalize_github_shorthand(str(local)) == str(local)
