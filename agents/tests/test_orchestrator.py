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
