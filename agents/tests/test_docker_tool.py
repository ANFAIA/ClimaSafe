from __future__ import annotations

from pathlib import Path

from agents.tools.docker_tool import DockerTool


def test_lint_flags_unpinned_base_image(tmp_path: Path):
    dockerfile = tmp_path / "Dockerfile"
    dockerfile.write_text("FROM python\nCOPY . .\n")
    findings = DockerTool.lint_dockerfile(dockerfile)
    assert any("sin versión fijada" in f.message for f in findings)


def test_lint_does_not_flag_pinned_base_image(tmp_path: Path):
    dockerfile = tmp_path / "Dockerfile"
    dockerfile.write_text("FROM python:3.12-slim\nUSER appuser\nCOPY . .\n")
    findings = DockerTool.lint_dockerfile(dockerfile)
    assert not any("sin versión fijada" in f.message for f in findings)


def test_lint_flags_missing_user(tmp_path: Path):
    dockerfile = tmp_path / "Dockerfile"
    dockerfile.write_text("FROM python:3.12-slim\nCOPY . .\n")
    findings = DockerTool.lint_dockerfile(dockerfile)
    assert any("USER" in f.message for f in findings)


def test_lint_missing_file_returns_single_warning(tmp_path: Path):
    findings = DockerTool.lint_dockerfile(tmp_path / "no_existe" / "Dockerfile")
    assert len(findings) == 1
    assert findings[0].severity == "warning"
