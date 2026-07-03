"""
agents.orchestrator — Decide qué agente(s) usar para una tarea.

Ruteo determinista, no un LLM
------------------------------
El `Orchestrator` puntúa cada agente registrado con `agent.can_handle(query)`
(coincidencia de palabras clave, ver `BaseAgent.can_handle`) y elige el de
mayor puntuación. Es una decisión de diseño explícita: el objetivo de este
sistema es automatizar tareas reales con herramientas, no envolver un
chatbot (ver `agents/README.md`).

Si en el futuro quieres un ruteo más inteligente (p. ej. la llamada a un LLM
que decida el agente, sea de Anthropic, OpenAI o cualquier otro proveedor),
el punto de extensión es
`Orchestrator.select_agent` — puedes sobreescribirlo o inyectar una función
de scoring distinta sin tocar ningún agente.
"""

from __future__ import annotations

from dataclasses import dataclass

from agents.context import SharedContext, get_context
from agents.core.base_agent import AgentResult, BaseAgent
from agents.core.registry import agent_registry
from agents.exceptions import AgentNotFoundError

MIN_CONFIDENCE = 0.15


@dataclass
class RoutingDecision:
    query: str
    agent_name: str | None
    confidence: float
    candidates: list[tuple[str, float]]


class Orchestrator:
    def __init__(self, context: SharedContext | None = None):
        self.ctx = context or get_context()
        self._instances: dict[str, BaseAgent] = {}

    def _get_instance(self, name: str) -> BaseAgent:
        if name not in self._instances:
            agent_cls = agent_registry.get(name)
            self._instances[name] = agent_cls(context=self.ctx)
        return self._instances[name]

    def list_agents(self) -> list[dict]:
        agent_registry.discover()
        return [self._get_instance(name).describe() for name in sorted(agent_registry.all())]

    def select_agent(self, query: str) -> RoutingDecision:
        agent_registry.discover()
        scores = []
        for name in agent_registry.all():
            instance = self._get_instance(name)
            scores.append((name, instance.can_handle(query)))
        scores.sort(key=lambda pair: -pair[1])

        best_name, best_score = (scores[0] if scores else (None, 0.0))
        if best_score < MIN_CONFIDENCE:
            best_name = None
        return RoutingDecision(query=query, agent_name=best_name, confidence=best_score, candidates=scores)

    def dispatch(self, query: str, *, action: str | None = None, **kwargs) -> AgentResult:
        """
        Rutea `query` en lenguaje natural al agente con mayor puntuación de
        `can_handle` y ejecuta `action` sobre él (si no se especifica, usa la
        primera acción declarada por el agente — ver nota más abajo).
        """
        decision = self.select_agent(query)
        if decision.agent_name is None:
            return AgentResult(
                False, "orchestrator", "dispatch",
                f"Ningún agente registrado alcanza la confianza mínima ({MIN_CONFIDENCE}) para: '{query}'. "
                f"Candidatos evaluados: {decision.candidates}. Prueba a invocar el agente directamente "
                f"si sabes cuál necesitas: agents.run(nombre, accion, **kwargs).",
            )

        agent = self._get_instance(decision.agent_name)
        if action is None:
            # Sin acción explícita: el ruteo por palabras clave identifica el agente
            # correcto, pero no la acción — se necesita más contexto para adivinarla
            # bien. Devolvemos las acciones disponibles en vez de arriesgar una
            # ejecución equivocada.
            return AgentResult(
                True, "orchestrator", "dispatch",
                f"Agente seleccionado: '{decision.agent_name}' (confianza {decision.confidence:.2f}). "
                f"Especifica una acción con action=... — acciones disponibles: {sorted(agent.actions())}",
                data={"selected_agent": decision.agent_name, "available_actions": sorted(agent.actions())},
            )

        return self.run(decision.agent_name, action, **kwargs)

    def run(self, agent_name: str, action: str, **kwargs) -> AgentResult:
        """Ejecuta una acción concreta de un agente concreto, sin pasar por el ruteo por keywords."""
        from agents.exceptions import ActionNotSupportedError

        try:
            agent = self._get_instance(agent_name)
        except AgentNotFoundError as exc:
            return AgentResult(False, "orchestrator", "run", str(exc))
        try:
            return agent.run(action, **kwargs)
        except ActionNotSupportedError as exc:
            return AgentResult(False, agent_name, action, str(exc))
