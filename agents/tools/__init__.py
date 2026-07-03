"""
agents.tools — Herramientas reutilizables por cualquier agente.

Regla del sistema: los agentes NUNCA duplican lógica de herramientas. Si dos
agentes necesitan leer un CSV o correr un comando `git`, ambos importan la
misma herramienta de aquí. Ver `agents/README.md` para la lista completa y
la filosofía ("sin dependencias innecesarias": todo lo que puede vivir en la
librería estándar de Python, vive ahí).
"""

from agents.tools.registry import register_tool, tool_registry

__all__ = ["register_tool", "tool_registry"]
