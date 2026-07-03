"""
agents.agents._template_agent — Plantilla de ejemplo para crear un agente nuevo.

Este archivo empieza con `_` a propósito: el auto-descubrimiento del
registro (`agents/core/registry.py`) ignora los módulos que empiezan por
guion bajo, así que esta plantilla NUNCA se registra ni aparece en la CLI.

Para crear un agente real:
  1. Copia este archivo a `agents/agents/mi_agente.py` (sin el `_` inicial).
  2. Cambia `name`, `description` y `capabilities`.
  3. Implementa tus métodos de acción — cada uno debe devolver `AgentResult`.
  4. Regístralos en `actions()`.
  5. Si necesitas una herramienta que no existe, créala en `agents/tools/`
     en vez de meter la lógica directamente en el agente (así otros agentes
     futuros pueden reutilizarla).

No hace falta tocar `orchestrator.py`, `cli.py` ni ningún otro agente: el
registro y el descubrimiento son automáticos.
"""

from __future__ import annotations

from agents.core.base_agent import AgentResult, BaseAgent
from agents.core.registry import register_agent

# from agents.tools.filesystem_tool import FilesystemTool  # ejemplo de herramienta reutilizada


@register_agent
class TemplateAgent(BaseAgent):
    name = "template_example"  # identificador único — se usa en la CLI y el registro
    description = "Ejemplo de agente — copia este archivo para crear uno nuevo."
    capabilities = ["ejemplo", "plantilla"]  # palabras clave para el ruteo del Orchestrator

    def actions(self) -> dict:
        return {
            "hello": self.hello,
        }

    def hello(self, *, name: str = "mundo") -> AgentResult:
        return AgentResult(
            success=True,
            agent=self.name,
            action="hello",
            message=f"Hola, {name}. Este es un agente de ejemplo — la raíz del proyecto es {self.ctx.root}.",
            data={"root": str(self.ctx.root), "project_slug": self.ctx.config.project_slug},
        )
