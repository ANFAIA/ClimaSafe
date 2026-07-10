"""Tests del PlanAgent: encargo → preguntas → delegación, sin inventar nada."""

from __future__ import annotations

from agents.agents.plan_agent import PlanAgent
from agents.orchestrator import Orchestrator


def _write_pyproject(project_root):
    (project_root / "pyproject.toml").write_text(
        '[project]\nname = "mi_paquete"\nversion = "0.1.0"\nrequires-python = ">=3.10"\n'
    )


def test_intake_asks_for_missing_args_and_execute_refuses(context):
    """
    Un paso cuya acción necesita un argumento obligatorio (tag_release →
    version) debe convertirse en pregunta, y execute debe NEGARSE mientras
    haya preguntas — pedir información, nunca adivinarla.
    """
    plan = PlanAgent(context=context)

    result = plan.intake(brief="haz un tag release del proyecto git")
    assert result.success
    assert result.data["steps"], f"El intake no asignó ningún paso: {result.message}"
    step = result.data["steps"][0]
    assert step["agent"] == "git"
    assert "version" in step["missing"]
    assert result.needs, "Debería haber preguntas pendientes (falta 'version')"

    order_id = result.data["id"]
    refused = plan.execute(order=order_id)
    assert not refused.success
    assert refused.needs, "execute debe devolver las preguntas pendientes, no ejecutar con huecos"

    answered = plan.answer(order=order_id, step0_version="9.9.9")
    assert answered.success
    assert not answered.needs, f"No deberían quedar preguntas: {answered.needs}"


def test_intake_execute_happy_path_delegates_and_records(context, project_root):
    """Encargo sin huecos: se ejecuta delegando en el agente dueño y queda auditado."""
    _write_pyproject(project_root)
    plan = PlanAgent(context=context)

    result = plan.intake(brief="verifica la python version del entorno")
    assert result.success
    assert result.data["steps"][0]["agent"] == "env"
    assert not result.needs, f"No debería preguntar nada: {result.needs}"

    executed = plan.execute(order=result.data["id"])
    assert executed.success, executed.message
    assert executed.data["status"] == "completado"
    assert executed.data["results"][0]["agent"] == "env"

    # La delegación pasó por run() → quedó en el log de auditoría.
    from agents import audit
    entries = audit.read_entries(context)
    assert any(e["agent"] == "env" and e["action"] == "check_python_version" for e in entries)


def test_status_lists_orders(context):
    plan = PlanAgent(context=context)
    plan.intake(brief="verifica la python version del entorno")
    listing = plan.status()
    assert listing.success
    assert len(listing.data) == 1


def test_plan_routes_via_orchestrator(context):
    """'planificar' y compañía rutean al agente plan, no a otro."""
    orchestrator = Orchestrator(context=context)
    decision = orchestrator.select_agent("planifica este encargo")
    assert decision.agent_name == "plan"
