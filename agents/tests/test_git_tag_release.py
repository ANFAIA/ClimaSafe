from __future__ import annotations

import subprocess

from agents.agents.documentation_agent import DocumentationAgent
from agents.agents.git_agent import GitAgent


def _write_versioned_files(root):
    (root / "pyproject.toml").write_text('[project]\nname = "x"\nversion = "1.0.0"\n')
    (root / "README.md").write_text(
        "![Version](https://img.shields.io/badge/Version-1.0.0-green)\n\n**Versión:** 1.0.0\n"
    )


def test_bump_version_updates_pyproject_and_readme(context):
    _write_versioned_files(context.root)
    agent = DocumentationAgent(context=context)
    result = agent.bump_version(new_version="2.0.0")

    assert result.success
    assert 'version = "2.0.0"' in (context.root / "pyproject.toml").read_text()
    readme = (context.root / "README.md").read_text()
    assert "Version-2.0.0-green" in readme
    assert "**Versión:** 2.0.0" in readme


def test_bump_version_warns_when_pattern_missing(context):
    (context.root / "pyproject.toml").write_text('[project]\nname = "x"\n')  # sin "version = ..."
    (context.root / "README.md").write_text("# Proyecto sin badge de versión\n")
    agent = DocumentationAgent(context=context)
    result = agent.bump_version(new_version="2.0.0")

    assert not result.success
    assert any("pyproject.toml" in w for w in result.warnings)


def test_tag_release_full_flow(context):
    _write_versioned_files(context.root)
    subprocess.run(["git", "add", "-A"], cwd=context.root, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "chore: añade archivos versionados"], cwd=context.root, check=True)

    agent = GitAgent(context=context)
    result = agent.tag_release(version="1.1.0")

    assert result.success
    assert 'version = "1.1.0"' in (context.root / "pyproject.toml").read_text()
    tags = subprocess.run(["git", "tag"], cwd=context.root, capture_output=True, text=True, check=True).stdout
    assert "1.1.0" in tags.splitlines()

    show = subprocess.run(
        ["git", "show", "--stat", "1.1.0"], cwd=context.root, capture_output=True, text=True, check=True
    ).stdout
    assert "pyproject.toml" in show and "README.md" in show and "CHANGELOG.md" in show


def test_tag_release_refuses_to_overwrite_existing_tag(context):
    _write_versioned_files(context.root)
    subprocess.run(["git", "add", "-A"], cwd=context.root, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "chore: setup"], cwd=context.root, check=True)
    subprocess.run(["git", "tag", "1.1.0"], cwd=context.root, check=True)

    agent = GitAgent(context=context)
    result = agent.tag_release(version="1.1.0")
    assert not result.success


def test_tag_release_generates_cicd_when_missing(context):
    (context.root / "Makefile").write_text("test:\n\t@echo test\nlint:\n\t@echo lint\n")
    _write_versioned_files(context.root)
    subprocess.run(["git", "add", "-A"], cwd=context.root, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "chore: setup"], cwd=context.root, check=True)

    agent = GitAgent(context=context)
    result = agent.tag_release(version="1.1.0")

    assert result.success
    assert (context.root / ".github" / "workflows" / "ci.yml").exists()
    show = subprocess.run(
        ["git", "show", "--stat", "1.1.0"], cwd=context.root, capture_output=True, text=True, check=True
    ).stdout
    assert ".github/workflows/ci.yml" in show
    assert any("No había ningún workflow de CI" in w for w in result.warnings)
