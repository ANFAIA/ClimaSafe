from __future__ import annotations

import subprocess

from agents.agents.git_agent import GitAgent
from agents.tools.git_tool import GitTool


def test_status_on_clean_repo(context):
    agent = GitAgent(context=context)
    result = agent.status()
    assert result.success
    assert result.data["changes"] == []


def test_suggest_commit_message_with_changes(context):
    (context.root / "mi_paquete" / "nuevo.py").write_text("x = 1\n")
    subprocess.run(["git", "add", "."], cwd=context.root, check=True)

    agent = GitAgent(context=context)
    result = agent.suggest_commit_message(staged=True)
    assert result.success
    assert "mi_paquete" in result.data["suggested_message"]


def test_analyze_diff_warns_when_source_touched_without_tests(context):
    (context.root / "mi_paquete" / "nuevo.py").write_text("x = 1\n")
    subprocess.run(["git", "add", "."], cwd=context.root, check=True)

    agent = GitAgent(context=context)
    result = agent.analyze_diff(staged=True)
    assert result.success
    assert any("tests/" in w for w in result.warnings)


def test_parse_conventional_commit_valid():
    parsed = GitTool.parse_conventional_commit("feat(api): añade endpoint de salud")
    assert parsed == {"type": "feat", "scope": "api", "breaking": False, "subject": "añade endpoint de salud"}


def test_parse_conventional_commit_invalid():
    assert GitTool.parse_conventional_commit("cambié cosas") is None


def test_guess_commit_type_prefers_test_for_test_files():
    assert GitTool.guess_commit_type(["tests/test_x.py"]) == "test"


def test_guess_commit_type_defaults_to_feat_for_unknown_paths():
    assert GitTool.guess_commit_type(["mi_paquete/models/nuevo.py"]) == "feat"
