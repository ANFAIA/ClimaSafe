from __future__ import annotations

from agents.agents.test_agent import TestAgent


def test_list_untested_modules_detects_missing_and_present(context):
    (context.root / "mi_paquete" / "con_test.py").write_text("def f(): return 1\n")
    (context.root / "mi_paquete" / "sin_test.py").write_text("def g(): return 2\n")
    (context.root / "tests" / "test_con_test.py").write_text("from mi_paquete.con_test import f\n")

    agent = TestAgent(context=context)
    result = agent.list_untested_modules()

    assert result.success
    assert result.data == ["sin_test"]


def test_list_untested_modules_excludes_init(context):
    (context.root / "mi_paquete" / "__init__.py").write_text("x = 1\n")
    agent = TestAgent(context=context)
    result = agent.list_untested_modules()
    assert result.success
    assert "__init__" not in result.data


def test_list_untested_modules_missing_package_dir_fails(context):
    import shutil
    shutil.rmtree(context.root / "mi_paquete")
    agent = TestAgent(context=context)
    result = agent.list_untested_modules()
    assert not result.success


def test_coverage_report_without_project_slug_fails(context):
    from agents.config import ProjectConfig
    from agents.context import SharedContext
    ctx_sin_slug = SharedContext(root=context.root, config=ProjectConfig(project_slug=""))
    agent = TestAgent(context=ctx_sin_slug)
    result = agent.coverage_report()
    assert not result.success
