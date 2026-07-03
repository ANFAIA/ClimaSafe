"""
agents — Sistema de agentes especializados para este proyecto dskit.

Arquitectura tipo "plugin": cada agente vive en `agents/agents/`, se registra
solo (decorador `@register_agent`) y el `Orchestrator` decide a quién delegar
una tarea según las `capabilities` que cada agente declara.

Uso rápido
----------
    from agents import Orchestrator

    orch = Orchestrator()
    result = orch.dispatch("genera el changelog del último commit")
    print(result.message)

O agente a agente, sin pasar por el orquestador:

    from agents.agents.git_agent import GitAgent
    git = GitAgent()
    print(git.suggest_commit_message().data)

Ver `agents/README.md` para la guía completa de arquitectura y extensión.
"""

from agents.context import SharedContext, get_context
from agents.core.registry import agent_registry, register_agent
from agents.core.base_agent import BaseAgent, AgentResult
from agents.orchestrator import Orchestrator

__all__ = [
    "SharedContext",
    "get_context",
    "agent_registry",
    "register_agent",
    "BaseAgent",
    "AgentResult",
    "Orchestrator",
]

__version__ = "0.1.0"
