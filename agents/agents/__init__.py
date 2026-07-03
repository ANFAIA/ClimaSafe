"""
agents.agents — Agentes concretos del sistema.

Cada módulo de este paquete define un agente (`@register_agent` sobre una
clase que hereda `BaseAgent`) con una única responsabilidad. El registro
(`agents.core.registry.agent_registry`) importa todos los módulos de este
paquete automáticamente — no hace falta listar nada a mano.

Módulos con prefijo `_` (como `_template_agent.py`) se ignoran en el
auto-descubrimiento: úsalo para plantillas de ejemplo o trabajo en curso que
no quieras que se registre todavía.
"""
