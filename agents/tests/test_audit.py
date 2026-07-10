"""Tests de la auditoría: toda ejecución vía run() queda registrada y es analizable."""

from __future__ import annotations

from agents import audit
from agents.agents.audit_agent import AuditAgent
from agents.orchestrator import Orchestrator


def test_run_records_audit_entry(context, project_root):
    (project_root / "pyproject.toml").write_text(
        '[project]\nname = "mi_paquete"\nversion = "0.1.0"\nrequires-python = ">=3.10"\n'
    )
    orchestrator = Orchestrator(context=context)
    result = orchestrator.run("env", "check_python_version")
    assert result.success

    entries = audit.read_entries(context)
    assert len(entries) == 1
    entry = entries[0]
    assert entry["agent"] == "env"
    assert entry["action"] == "check_python_version"
    assert entry["success"] is True
    assert entry["duration_ms"] >= 0
    # No se guardan los valores de los kwargs (pueden contener secretos), solo nombres.
    assert entry["kwarg_names"] == []


def test_failed_actions_are_recorded_too(context):
    orchestrator = Orchestrator(context=context)
    result = orchestrator.run("env", "check_python_version")  # sin pyproject.toml → falla
    assert not result.success

    entries = audit.read_entries(context)
    assert entries and entries[-1]["success"] is False


def test_audit_agent_report_and_suggestions(context, project_root):
    (project_root / "pyproject.toml").write_text(
        '[project]\nname = "mi_paquete"\nversion = "0.1.0"\nrequires-python = ">=3.10"\n'
    )
    orchestrator = Orchestrator(context=context)
    orchestrator.run("env", "check_python_version")
    orchestrator.run("env", "check_python_version")

    report = AuditAgent(context=context).report()
    assert report.success
    assert report.data[0]["accion"] == "env.check_python_version"
    assert report.data[0]["runs"] == 2

    suggestions = AuditAgent(context=context).suggest_improvements()
    assert suggestions.success
    # Con solo 2 runs y todo OK, no debe acusar a env de nada; como los demás
    # agentes no se han usado, sí debe señalarlos como "sin uso".
    assert not any("env.check_python_version' falla" in s for s in suggestions.data)
    assert any("sin ninguna ejecución auditada" in s for s in suggestions.data)


def test_audit_report_on_empty_log(context):
    result = AuditAgent(context=context).report()
    assert result.success
    assert result.data == []
