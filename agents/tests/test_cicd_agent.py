from __future__ import annotations

from agents.agents.cicd_agent import CICDAgent


def _write_makefile(root, targets=("lint", "test")):
    content = "\n".join(f"{t}:\n\t@echo {t}\n" for t in targets)
    (root / "Makefile").write_text(content)


def test_generate_workflow_creates_valid_file(context):
    _write_makefile(context.root)
    agent = CICDAgent(context=context)
    result = agent.generate_workflow()

    assert result.success
    path = context.root / ".github" / "workflows" / "ci.yml"
    assert path.exists()
    assert "make lint" in path.read_text()
    assert "make test" in path.read_text()


def test_generate_workflow_refuses_overwrite_without_flag(context):
    _write_makefile(context.root)
    agent = CICDAgent(context=context)
    agent.generate_workflow()
    result = agent.generate_workflow()
    assert not result.success


def test_generate_workflow_fails_without_project_slug(context):
    context.config.__dict__  # no-op, solo para dejar constancia de que se usa config real
    from agents.config import ProjectConfig
    from agents.context import SharedContext
    ctx_sin_slug = SharedContext(root=context.root, config=ProjectConfig(project_slug=""))
    _write_makefile(ctx_sin_slug.root)
    agent = CICDAgent(context=ctx_sin_slug)
    result = agent.generate_workflow()
    assert not result.success


def test_validate_workflow_detects_missing_runs_on(context):
    workflows_dir = context.root / ".github" / "workflows"
    workflows_dir.mkdir(parents=True)
    (workflows_dir / "ci.yml").write_text("name: CI\non:\n  push:\njobs:\n  build:\n    steps:\n      - run: echo hola\n")

    agent = CICDAgent(context=context)
    result = agent.validate_workflow()
    assert not result.success
    assert any("runs-on" in p for p in result.data["problems"])


def test_validate_workflow_cross_references_makefile(context):
    _write_makefile(context.root, targets=("lint",))  # sin 'test'
    workflows_dir = context.root / ".github" / "workflows"
    workflows_dir.mkdir(parents=True)
    (workflows_dir / "ci.yml").write_text(
        "name: CI\non:\n  push:\njobs:\n  build:\n    runs-on: ubuntu-latest\n    steps:\n      - run: make test\n"
    )
    agent = CICDAgent(context=context)
    result = agent.validate_workflow()
    assert any("test" in w and "Makefile" in w for w in result.warnings)


def test_validate_workflow_missing_file(context):
    agent = CICDAgent(context=context)
    result = agent.validate_workflow(filename="no_existe.yml")
    assert not result.success


def test_list_workflows_empty_by_default(context):
    agent = CICDAgent(context=context)
    result = agent.list_workflows()
    assert result.success
    assert result.data == []
