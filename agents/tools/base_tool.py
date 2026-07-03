"""agents.tools.base_tool — Contrato mínimo que deben cumplir las herramientas."""

from __future__ import annotations

from abc import ABC


class BaseTool(ABC):
    """
    Clase base opcional para herramientas con estado (p. ej. una conexión
    DuckDB reutilizable). Las herramientas simples pueden ser directamente
    funciones sueltas en su módulo — no todo tiene que heredar de aquí. Lo
    único obligatorio para que una herramienta se pueda descubrir por nombre
    es registrarla con `@register_tool("nombre")`.
    """

    name: str = "base_tool"
    description: str = ""
