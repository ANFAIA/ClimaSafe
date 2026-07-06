from __future__ import annotations

from agents.core.registry import agent_registry
from agents.orchestrator import Orchestrator


def test_all_core_agents_are_discovered():
    agents = agent_registry.all()
    expected = {"git", "data", "graph", "docker", "ml", "review", "documentation"}
    assert expected.issubset(agents.keys())


def test_template_agent_is_not_discovered():
    # _template_agent.py empieza por '_' — no debe registrarse nunca.
    agents = agent_registry.all()
    assert "template_example" not in agents


def test_orchestrator_routes_git_query(context):
    orchestrator = Orchestrator(context=context)
    decision = orchestrator.select_agent("genera el changelog del proyecto")
    assert decision.agent_name == "git"


def test_orchestrator_routes_docker_query(context):
    orchestrator = Orchestrator(context=context)
    decision = orchestrator.select_agent("revisa el Dockerfile de este proyecto")
    assert decision.agent_name == "docker"


def test_orchestrator_returns_no_agent_for_irrelevant_query(context):
    orchestrator = Orchestrator(context=context)
    decision = orchestrator.select_agent("xyz completamente fuera de dominio 123")
    assert decision.agent_name is None


def test_dispatch_without_action_lists_available_actions(context):
    orchestrator = Orchestrator(context=context)
    result = orchestrator.dispatch("estado del repositorio git")
    assert result.success
    assert "available_actions" in result.data


def test_run_unknown_agent_returns_failed_result(context):
    orchestrator = Orchestrator(context=context)
    result = orchestrator.run("agente_que_no_existe", "cualquier_accion")
    assert not result.success


def test_dispatch_auto_runs_zero_arg_action(context):
    # "estado" no coincide textualmente con ninguna acción de git (todas en
    # inglés), pero "revisa el dockerfile" sí solapa con 'lint_dockerfile'
    # -> docker es el agente, lint_dockerfile la acción, sin argumentos
    # obligatorios -> debe ejecutarse sola, sin pedir action= explícito.
    (context.root / "Dockerfile").write_text("FROM python:3.12-slim\nUSER app\n")
    orchestrator = Orchestrator(context=context)
    result = orchestrator.dispatch("revisa el dockerfile de este proyecto")
    assert result.agent == "docker"
    assert result.action == "lint_dockerfile"


def test_dispatch_reports_required_args_instead_of_guessing_them(context):
    # 'genera el eda del dataset' debería apuntar a data.eda_report, que
    # necesita 'filename' obligatoriamente -> no debe ejecutarse sola,
    # debe reportar qué argumento falta en vez de inventarlo.
    orchestrator = Orchestrator(context=context)
    result = orchestrator.dispatch("haz un eda_report del dataset")
    assert result.success  # no es un error, es información de qué falta
    assert result.data.get("selected_action") == "eda_report"
    assert "filename" in result.data.get("required_args", [])


def test_dispatch_disambiguates_commit_actions_via_aliases(context):
    # "haz un commit" y "sugiéreme un mensaje de commit" comparten la palabra
    # "commit" con dos acciones distintas de GitAgent — sin action_aliases()
    # empatarían; con alias, cada consulta debe apuntar a la acción correcta.
    orchestrator = Orchestrator(context=context)

    commit_result = orchestrator.dispatch("haz un commit de mis cambios")
    assert commit_result.data.get("selected_action") == "commit_with_changelog"

    suggest_result = orchestrator.dispatch("sugiéreme un mensaje de commit")
    assert suggest_result.action == "suggest_commit_message"
