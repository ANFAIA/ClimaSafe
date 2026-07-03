"""
agents.exceptions — Jerarquía de excepciones propia del sistema de agentes.

Usar excepciones específicas (en vez de `Exception` genérica) permite que el
`Orchestrator` y la CLI distingan errores esperables (p. ej. "git no
instalado") de bugs reales, y respondan de forma distinta en cada caso.
"""

from __future__ import annotations


class AgentSystemError(Exception):
    """Excepción base de todo el sistema de agentes."""


class AgentNotFoundError(AgentSystemError):
    """Se pidió un agente que no está registrado."""


class ActionNotSupportedError(AgentSystemError):
    """Se pidió una acción que el agente no expone."""


class ToolNotFoundError(AgentSystemError):
    """Se pidió una herramienta que no está registrada."""


class ToolExecutionError(AgentSystemError):
    """Una herramienta falló al ejecutarse (proceso externo, IO, parsing...)."""

    def __init__(self, message: str, *, returncode: int | None = None, stderr: str | None = None):
        super().__init__(message)
        self.returncode = returncode
        self.stderr = stderr


class MissingDependencyError(AgentSystemError):
    """
    Una herramienta necesita un binario externo (git, docker) o un paquete
    Python opcional (duckdb, shap...) que no está disponible en este entorno.
    """
