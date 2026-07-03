"""
agents.tools.registry — Registro de herramientas por nombre.

A diferencia del registro de agentes (que auto-descubre módulos porque el
`Orchestrator` necesita listarlos todos), las herramientas normalmente se
importan directamente donde se usan:

    from agents.tools.git_tool import GitTool

El registro existe para dos casos concretos:
  - la CLI (`python -m agents tools --list`) quiere listar qué hay disponible
  - un agente quiere resolver una herramienta por nombre en tiempo de
    ejecución (p. ej. configurable por el usuario)

Registrar una herramienta es opcional pero recomendado.
"""

from __future__ import annotations

from typing import Any, Callable


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, Any] = {}

    def register(self, name: str) -> Callable[[Any], Any]:
        def decorator(obj: Any) -> Any:
            if name in self._tools and self._tools[name] is not obj:
                raise ValueError(f"Ya existe una herramienta registrada como '{name}'.")
            self._tools[name] = obj
            return obj

        return decorator

    def get(self, name: str) -> Any:
        from agents.exceptions import ToolNotFoundError

        if name not in self._tools:
            raise ToolNotFoundError(
                f"No existe ninguna herramienta registrada como '{name}'. "
                f"Disponibles: {sorted(self._tools)}"
            )
        return self._tools[name]

    def all(self) -> dict[str, Any]:
        return dict(self._tools)


tool_registry = ToolRegistry()


def register_tool(name: str) -> Callable[[Any], Any]:
    """Decorador: `@register_tool("git")` sobre una clase o función factory."""
    return tool_registry.register(name)
