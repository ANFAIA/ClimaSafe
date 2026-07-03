"""agents.core — Contratos base: BaseAgent, AgentResult y el registro de agentes."""

from agents.core.base_agent import AgentResult, BaseAgent
from agents.core.registry import agent_registry, register_agent

__all__ = ["AgentResult", "BaseAgent", "agent_registry", "register_agent"]
