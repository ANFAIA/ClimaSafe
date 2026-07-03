"""
agents.core.registry — Registro automático de agentes.

Cómo añadir un agente nuevo sin tocar el resto del sistema
------------------------------------------------------------
1. Crea `agents/agents/mi_agente.py`.
2. Define una clase que herede de `BaseAgent` y decórala:

    from agents.core.base_agent import BaseAgent, AgentResult
    from agents.core.registry import register_agent

    @register_agent
    class MiAgente(BaseAgent):
        name = "mi_agente"
        description = "..."
        capabilities = ["palabra1", "palabra2"]

        def actions(self):
            return {"hacer_algo": self.hacer_algo}

        def hacer_algo(self) -> AgentResult:
            ...

3. Nada más. `AgentRegistry.discover()` importa todos los módulos de
   `agents.agents` (y `agents.external`, y los entry points instalados)
   automáticamente, lo que ejecuta el decorador y registra la clase. El
   `Orchestrator` y la CLI lo verán sin cambios de código.
"""

from __future__ import annotations

import importlib
import pkgutil
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agents.core.base_agent import BaseAgent


class AgentRegistry:
    """Registro global {nombre: clase_agente}. Un único singleton: `agent_registry`."""

    def __init__(self) -> None:
        self._agents: dict[str, type["BaseAgent"]] = {}
        self._discovered = False

    def discover(self, *, force: bool = False) -> None:
        """
        Importa todos los módulos de `agents.agents` (núcleo del template),
        `agents.external` (agentes locales de terceros, ver
        `agents/external/README.md`) y cualquier entry point del grupo
        `dskit.agents` expuesto por paquetes instalados, para que se
        auto-registren.
        """
        if self._discovered and not force:
            return

        self._discover_package("agents.agents")
        self._discover_package("agents.external")
        self._discover_entry_points()

        self._discovered = True

    @staticmethod
    def _discover_package(package_name: str) -> None:
        try:
            package = importlib.import_module(package_name)
        except ImportError:
            return  # el paquete no existe en este checkout — no es un error fatal
        for module_info in pkgutil.iter_modules(package.__path__, prefix=f"{package_name}."):
            if module_info.name.rsplit(".", 1)[-1].startswith("_"):
                continue  # módulos privados / plantillas de ejemplo (p.ej. _template_agent.py)
            importlib.import_module(module_info.name)

    @staticmethod
    def _discover_entry_points() -> None:
        """
        Importa agentes expuestos como entry point `dskit.agents` por
        paquetes instalados en el entorno (ver `agents/external/README.md`,
        Opción 2). Usa `importlib.metadata` de la librería estándar — sin
        dependencias nuevas.

        Nota de compatibilidad: el argumento `group=` de
        `entry_points()` se añadió en Python 3.10. Si en algún momento este
        proyecto soporta una versión anterior, revisa la documentación de
        `importlib.metadata` para la API equivalente en esa versión — no
        está cubierta aquí a propósito, para no fingir una compatibilidad
        que no se ha probado.
        """
        import importlib.metadata as metadata

        try:
            eps = metadata.entry_points(group="dskit.agents")
        except TypeError:
            # Firma antigua (pre-3.10): entry_points() sin `group` devuelve un dict-like.
            eps = metadata.entry_points().get("dskit.agents", [])

        for entry_point in eps:
            try:
                entry_point.load()  # el propio import ejecuta @register_agent sobre la clase
            except Exception:  # noqa: BLE001 — un entry point de terceros roto no debe tumbar el resto
                continue

    def register(self, agent_cls: type["BaseAgent"]) -> type["BaseAgent"]:
        name = getattr(agent_cls, "name", None)
        if not name or name == "base":
            raise ValueError(
                f"{agent_cls.__name__} debe definir un atributo de clase `name` propio "
                f"antes de poder registrarse."
            )
        if name in self._agents and self._agents[name] is not agent_cls:
            raise ValueError(
                f"Ya hay un agente registrado con name='{name}' "
                f"({self._agents[name].__name__}). Los nombres deben ser únicos."
            )
        self._agents[name] = agent_cls
        return agent_cls

    def get(self, name: str) -> type["BaseAgent"]:
        self.discover()
        from agents.exceptions import AgentNotFoundError

        if name not in self._agents:
            raise AgentNotFoundError(
                f"No existe ningún agente registrado con name='{name}'. "
                f"Disponibles: {sorted(self._agents)}"
            )
        return self._agents[name]

    def all(self) -> dict[str, type["BaseAgent"]]:
        self.discover()
        return dict(self._agents)


agent_registry = AgentRegistry()


def register_agent(agent_cls: type["BaseAgent"]) -> type["BaseAgent"]:
    """Decorador: `@register_agent` sobre una clase que hereda de BaseAgent."""
    return agent_registry.register(agent_cls)
