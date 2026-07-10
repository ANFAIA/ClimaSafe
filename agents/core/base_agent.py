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

import inspect
import re
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
    # Preguntas pendientes: si a la acción le falta información que solo el
    # humano puede dar, se devuelve success=False con la lista de preguntas
    # aquí — NUNCA se inventa un valor. Es el mecanismo estándar de "pedir
    # información" del sistema (lo usa PlanAgent, y cualquier agente puede
    # usarlo igual).
    needs: list[str] = field(default_factory=list)

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

    def run(self, action: str, /, **kwargs) -> AgentResult:
        """
        Despacho genérico: `agent.run("suggest_commit_message")`.

        Toda ejecución que pasa por aquí (CLI, Orchestrator, GStack,
        delegate_to) queda registrada en el log de auditoría
        (`agents/workspace/audit/audit.jsonl`, ver `agents/audit.py`) — es
        la base para medir y mejorar a los agentes con el agente `audit`.
        """
        import time

        from agents import audit

        available = self.actions()
        if action not in available:
            raise ActionNotSupportedError(
                f"El agente '{self.name}' no soporta la acción '{action}'. "
                f"Acciones disponibles: {sorted(available)}"
            )

        start = time.perf_counter()
        try:
            result = available[action](**kwargs)
        except Exception as exc:
            # Los agentes no deberían dejar escapar excepciones (ver docstring
            # del módulo), pero si ocurre, se audita igualmente antes de
            # propagarla — un fallo no auditado es invisible para `audit`.
            audit.record(
                self.ctx, agent=self.name, action=action, success=False,
                duration_ms=(time.perf_counter() - start) * 1000,
                message="excepción no controlada", error=f"{type(exc).__name__}: {exc}",
                kwarg_names=sorted(kwargs),
            )
            raise

        audit.record(
            self.ctx, agent=self.name, action=action, success=result.success,
            duration_ms=(time.perf_counter() - start) * 1000,
            message=result.message, warnings=len(result.warnings),
            kwarg_names=sorted(kwargs),
        )
        return result

    def can_handle(self, query: str) -> float:
        """
        Puntúa 0..1 cuánto de relevante es este agente para `query`, en base a
        coincidencias de `capabilities` en el texto (case-insensitive, por
        PALABRA/FRASE COMPLETA con límites `\\b` — no subcadena).

        Importante: usar subcadena en vez de límites de palabra es un bug
        real que encontré probando esto — "ci" (palabra clave de un agente
        de CI/CD) es subcadena literal de "dependencias" (depen-CI-as), lo
        que enrutaba consultas sobre dependencias al agente equivocado por
        pura coincidencia de caracteres, no de palabras.

        También encontré (al arreglar lo anterior) que normalizar por
        `len(capabilities)` penaliza a los agentes con una lista de
        capacidades más completa: dos agentes con 1 acierto cada uno, pero
        uno con 5 keywords declaradas y otro con 12, no deberían competir en
        desventaja el segundo solo por haber documentado más sinónimos. Por
        eso cada acierto suma un valor fijo (0.4), sin dividir por el total
        de capacidades del agente.

        Es una heurística simple y determinista a propósito (ver filosofía en
        `agents/README.md`: estos agentes no son un chatbot). Si en el futuro
        quieres un ruteo más inteligente, este es el único método a sobreescribir
        o a sustituir por una llamada a un LLM — el resto del sistema no cambia.
        """
        if not self.capabilities:
            return 0.0
        text = query.lower()
        matched = [kw for kw in self.capabilities if re.search(rf"\b{re.escape(kw.lower())}\b", text)]
        if not matched:
            return 0.0
        # Desempate por especificidad: dos agentes con el mismo nº de aciertos
        # empataban y ganaba el descubierto antes (orden alfabético) — puro
        # azar. Caso real: "pre-commit" acierta 'pre-commit' (env) y también
        # 'commit' (git, el guion es límite de palabra). La coincidencia más
        # larga (más específica) gana el empate; el bonus (≤0.1) nunca puede
        # alterar el ranking entre números de aciertos distintos (0.4 cada uno).
        specificity = min(0.1, sum(len(kw) for kw in matched) * 0.001)
        return min(1.0, len(matched) * 0.4 + specificity)

    def action_aliases(self) -> dict[str, list[str]]:
        """
        Hook opcional: {nombre_de_accion: [palabras clave adicionales]}.
        `best_action` ya adivina razonablemente bien cuando el nombre de la
        acción comparte palabras con la consulta (ideal para consultas en
        inglés o con términos técnicos como "commit", "docker", "changelog").
        Falla más cuando la consulta está en español y el nombre de la acción
        no comparte ninguna palabra reconocible (p. ej. "suggest_commit_message"
        no tiene ninguna palabra en común con "sugiere un mensaje").

        No es obligatorio definir esto para cada acción — solo tiene sentido
        donde hay ambigüedad real entre dos acciones parecidas (ver
        `GitAgent.action_aliases` para el caso concreto que motivó esto:
        diferenciar "haz un commit" de "sugiéreme un mensaje de commit").
        """
        return {}

    def best_action(self, query: str) -> str | None:
        """
        Adivina qué acción de `self.actions()` encaja mejor con `query`, por
        solapamiento de palabras entre el texto y el propio nombre de la
        acción (p. ej. "generate_changelog" -> {"generate", "changelog"}),
        más las palabras extra de `action_aliases()` si el agente las define.

        Deliberadamente no requiere que cada agente declare keywords por
        acción — usa el nombre de la acción tal cual, así funciona igual de
        bien para acciones de agentes futuros sin tocar este método. Es una
        heurística de convención, no de comprensión real: nombra bien tus
        acciones (verbo_sustantivo, en snake_case) y esto funciona; nómbralas
        de forma opaca y esto no adivinará nada útil.
        """
        text_words = set(re.findall(r"[a-záéíóúñ]+", query.lower()))
        if not text_words:
            return None
        aliases = self.action_aliases()
        best_name, best_score = None, 0
        for action_name in self.actions():
            action_words = set(action_name.split("_"))
            for alias in aliases.get(action_name, []):
                action_words |= set(re.findall(r"[a-záéíóúñ]+", alias.lower()))
            score = len(text_words & action_words)
            if score > best_score:
                best_name, best_score = action_name, score
        return best_name

    def can_auto_run(self, action_name: str) -> bool:
        """
        True si `action_name` se puede ejecutar sin argumentos adicionales
        (todos sus parámetros, aparte de `self`, tienen valor por defecto).
        El `Orchestrator` solo ejecuta una acción adivinada automáticamente
        cuando esto es cierto — si la acción necesita un argumento
        obligatorio (p. ej. `message` en `commit_with_changelog`), no hay
        forma honesta de adivinarlo desde una frase en lenguaje natural sin
        un LLM de por medio, así que no se intenta.
        """
        method = self.actions().get(action_name)
        if method is None:
            return False
        signature = inspect.signature(method)
        for param in signature.parameters.values():
            if param.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
                continue
            if param.default is inspect.Parameter.empty:
                return False
        return True

    def describe(self) -> dict[str, Any]:
        from agents.contracts import contract_for

        info = {
            "name": self.name,
            "description": self.description,
            "capabilities": self.capabilities,
            "actions": sorted(self.actions()),
        }
        contract = contract_for(self.name)
        if contract is not None:
            info["contract"] = contract.as_dict()
        return info
