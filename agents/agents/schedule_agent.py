"""
agents.agents.schedule_agent — Gestión de programación cron y tareas periódicas.

Usa ScheduleTool para validar, describir y calcular próximas ejecuciones de
expresiones cron. Puede integrarse con workflows CI/CD, backups, o cualquier
tarea programada del proyecto.
"""

from __future__ import annotations

from agents.core.base_agent import AgentResult, BaseAgent
from agents.core.registry import register_agent
from agents.tools.schedule_tool import ScheduleTool


@register_agent
class ScheduleAgent(BaseAgent):
    name = "schedule"
    description = "Valida, describe y analiza expresiones cron para tareas programadas."
    capabilities = ["cron", "schedule", "programar", "temporizador", "scheduler", "calendarizar"]

    def action_aliases(self) -> dict:
        return {
            "validate": ["valida", "comprueba", "es valido"],
            "to_human": ["explica", "traduce", "describe", "legible"],
            "next_runs": ["proximas", "siguientes", "cuando se ejecuta", "next"],
        }

    def actions(self) -> dict:
        return {
            "validate": self.validate_cron,
            "to_human": self.to_human,
            "next_runs": self.next_runs,
        }

    def validate_cron(self, *, expression: str) -> AgentResult:
        """Valida si una expresión cron es correcta."""
        result = ScheduleTool.validate(expression)
        return AgentResult(
            result["valid"], self.name, "validate",
            "Expresiión cron válida." if result["valid"] else f"Cron inválido: {result['error']}",
            data=result,
        )

    def to_human(self, *, expression: str) -> AgentResult:
        """Convierte una expresión cron a texto legible."""
        valid = ScheduleTool.validate(expression)
        if not valid["valid"]:
            return AgentResult(False, self.name, "to_human", valid["error"], data=valid)
        human = ScheduleTool.to_human(expression)
        return AgentResult(True, self.name, "to_human", human, data={"expression": expression, "human": human})

    def next_runs(self, *, expression: str, count: int = 5) -> AgentResult:
        """Calcula las próximas ejecuciones de una expresión cron."""
        valid = ScheduleTool.validate(expression)
        if not valid["valid"]:
            return AgentResult(False, self.name, "next_runs", valid["error"], data=valid)
        result = ScheduleTool.next_run(expression)
        runs = result["next_runs"][:count]
        return AgentResult(
            True, self.name, "next_runs",
            f"Próximas {len(runs)} ejecución(es): {', '.join(runs)}",
            data={"expression": expression, "next_runs": runs, "from": result["from"]},
        )
