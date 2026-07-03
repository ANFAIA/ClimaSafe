"""
agents.core.base_agent — Contrato que debe cumplir todo agente del sistema.

Un agente:
1. Declara quién es (`name`, `description`) y qué sabe hacer (`capabilities`,
   una lista de palabras clave usada por el `Orchestrator` para el ruteo).
2. Expone sus acciones como métodos públicos normales, usables directamente
   sin pasar por el orquestador (`GitAgent().suggest_commit_message()`).
3. Implementa `run(action, **kwargs)` como despacho uniforme a esos métodos,
   para que el `Orchestrator` y la CLI puedan invocar cualquier agente sin
   conocer su API interna de antemano.

Todo método público de un agente debe devolver un `AgentResult`, nunca lanzar
una excepción hacia arriba directamente: los errores esperables (herramienta
no instalada, archivo no encontrado...) se capturan y se devuelven como
`AgentResult(success=False, ...)`. Esto mantiene al `Orchestrator` y a la CLI
simples: no necesitan un `try/except` por cada agente.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from agents.context import SharedContext, get_context
from agents.exceptions import ActionNotSupportedError


@dataclass
class AgentResult:
    """Resultado uniforme que devuelve cualquier acción de cualquier agente."""

    success: bool
    agent: str
    action: str
    message: str
    data: Any = None
    warnings: list[str] = field(default_factory=list)

    def __bool__(self) -> bool:
        return self.success

    def __repr__(self) -> str:
        status = "OK" if self.success else "FAIL"
        return f"<AgentResult {status} {self.agent}.{self.action}: {self.message}>"


class BaseAgent(ABC):
    """
    Clase base de todos los agentes.

    Subclases obligatorias a definir:
        name          : str  — identificador único, usado en el registro y la CLI
        description   : str  — una línea, qué hace este agente
        capabilities  : list[str] — palabras clave para el ruteo del Orchestrator

    Subclases deben implementar:
        actions()  -> dict[str, Callable[..., AgentResult]]
            Mapa {nombre_de_accion: metodo_bound}. Es la fuente de verdad que
            usan `run()`, la CLI y `describe()` — defínelo una sola vez.
    """

    name: str = "base"
    description: str = "Agente base (no debe instanciarse directamente)."
    capabilities: list[str] = []

    def __init__(self, context: SharedContext | None = None):
        self.ctx = context or get_context()

    @abstractmethod
    def actions(self) -> dict[str, Any]:
        """Devuelve {nombre_accion: metodo} para despacho uniforme vía run()."""
        raise NotImplementedError

    def run(self, action: str, **kwargs) -> AgentResult:
        """Despacho genérico: `agent.run("suggest_commit_message")`."""
        available = self.actions()
        if action not in available:
            raise ActionNotSupportedError(
                f"El agente '{self.name}' no soporta la acción '{action}'. "
                f"Acciones disponibles: {sorted(available)}"
            )
        return available[action](**kwargs)

    def can_handle(self, query: str) -> float:
        """
        Puntúa 0..1 cuánto de relevante es este agente para `query`, en base a
        coincidencias de `capabilities` en el texto (case-insensitive).

        Es una heurística simple y determinista a propósito (ver filosofía en
        `agents/README.md`: estos agentes no son un chatbot). Si en el futuro
        quieres un ruteo más inteligente, este es el único método a sobreescribir
        o a sustituir por una llamada a un LLM — el resto del sistema no cambia.
        """
        if not self.capabilities:
            return 0.0
        text = query.lower()
        hits = sum(1 for kw in self.capabilities if kw.lower() in text)
        return min(1.0, hits / max(1, len(self.capabilities)) * 2)

    def describe(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "capabilities": self.capabilities,
            "actions": sorted(self.actions()),
        }
