"""
agents.helpers — Funciones auxiliares para agentes.

Incluye el helper de delegación entre agentes, que permite que un agente
invoque la acción de otro agente programáticamente.
"""

from __future__ import annotations

from typing import Any

from agents.context import get_context
from agents.orchestrator import Orchestrator


_ORCH: Orchestrator | None = None


def _orch() -> Orchestrator:
    global _ORCH
    if _ORCH is None:
        _ORCH = Orchestrator(context=get_context())
    return _ORCH


def delegate_to(agent: str, action: str, **kwargs) -> Any:
    """
    Delega una tarea a otro agente y devuelve su resultado.

    Útil cuando un agente necesita información de otro: por ejemplo,
    ``DataAgent`` puede delegar en ``GitAgent`` para saber qué archivos
    cambiaron, o ``DoctorAgent`` puede delegar en múltiples agentes para
    el checkup.

    Ejemplo::

        from agents.helpers import delegate_to

        result = delegate_to("git", "analyze_diff")
        files = result.data  # dict con añadidos/modificados/borrados
    """
    return _orch().run(agent, action, **kwargs)
