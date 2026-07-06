from __future__ import annotations

from agents.agents.dependency_agent import DependencyAgent
from agents.tools.dependency_tool import (
    normalize_package_name,
    parse_dependency_name,
    parse_pyproject_dependencies,
    parse_uv_lock_versions,
)

# --- tests puros de parseo, sin red -----------------------------------------

def test_parse_dependency_name_strips_version_and_extras():
    assert parse_dependency_name("requests>=2.20.0") == "requests"
    assert parse_dependency_name("pandas[extra]") == "pandas"
    assert parse_dependency_name('scikit-learn; python_version >= "3.10"') == "scikit-learn"


def test_normalize_package_name_pep503():
    assert normalize_package_name("Scikit_Learn") == "scikit-learn"
    assert normalize_package_name("Python.Dotenv") == "python-dotenv"


def test_parse_pyproject_dependencies():
    text = '[project]\ndependencies = [\n    "requests>=2.0",\n    "pandas",\n]\n'
    assert parse_pyproject_dependencies(text) == ["requests>=2.0", "pandas"]


def test_parse_pyproject_dependencies_missing_block_returns_empty():
    assert parse_pyproject_dependencies("[project]\nname = 'x'\n") == []


def test_parse_uv_lock_versions():
    lock_text = (
        '[[package]]\nname = "requests"\nversion = "2.34.2"\nsource = {registry = "x"}\n\n'
        '[[package]]\nname = "Pandas"\nversion = "3.0.3"\nsource = {registry = "x"}\n'
    )
    versions = parse_uv_lock_versions(lock_text)
    assert versions["requests"] == "2.34.2"
    assert versions["pandas"] == "3.0.3"  # normalizado a minúsculas


# --- tests de agente sin red (rutas de error) -------------------------------

def test_check_outdated_without_pyproject_fails(context):
    agent = DependencyAgent(context=context)
    result = agent.check_outdated()
    assert not result.success


def test_check_outdated_without_dependencies_returns_empty(context):
    (context.root / "pyproject.toml").write_text('[project]\nname = "x"\ndependencies = []\n')
    agent = DependencyAgent(context=context)
    result = agent.check_outdated()
    assert result.success
    assert result.data == []


def test_check_vulnerabilities_without_lock_fails(context):
    (context.root / "pyproject.toml").write_text('[project]\nname = "x"\ndependencies = ["requests"]\n')
    agent = DependencyAgent(context=context)
    result = agent.check_vulnerabilities()
    assert not result.success
