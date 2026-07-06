from __future__ import annotations

import subprocess

from agents.agents.git_agent import GitAgent


def test_commit_with_changelog_includes_changelog_in_same_commit(context):
    (context.root / "mi_paquete" / "nuevo.py").write_text("x = 1\n")
    subprocess.run(["git", "add", "."], cwd=context.root, check=True)

    agent = GitAgent(context=context)
    result = agent.commit_with_changelog(message="feat(mi_paquete): añade nuevo.py")

    assert result.success
    assert result.data["changelog_updated"] is True
    assert (context.root / "CHANGELOG.md").exists()

    log = subprocess.run(
        ["git", "show", "--stat", "HEAD"], cwd=context.root, capture_output=True, text=True, check=True
    ).stdout
    assert "CHANGELOG.md" in log
    assert "nuevo.py" in log


def test_commit_with_changelog_without_pending_changes_fails_since_tag(context):
    import subprocess

    agent = GitAgent(context=context)

    # primera llamada: SÍ hay algo real que resumir (el commit inicial de la
    # fixture) -> crea CHANGELOG.md por primera vez -> commit válido
    first = agent.commit_with_changelog(message="chore: primer changelog")
    assert first.success
    assert first.data["changelog_updated"] is True

    # fijamos un tag en el HEAD actual: a partir de aquí, generate_changelog
    # con since_tag=ese_tag no tiene NINGÚN commit nuevo que resumir (a
    # diferencia de llamar sin tag, que siempre re-resume todo el historial
    # y por tanto "cambia" en cada llamada porque el propio commit anterior
    # ya pasó a formar parte de ese historial)
    subprocess.run(["git", "tag", "v0.0.1"], cwd=context.root, check=True)

    second = agent.commit_with_changelog(message="chore: sin cambios reales", since_tag="v0.0.1")
    assert not second.success
