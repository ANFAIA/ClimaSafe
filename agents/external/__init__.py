"""
agents.external — Agentes de terceros o personales, fuera del núcleo del template.

Cualquier módulo `.py` que coloques aquí, con una clase `@register_agent`
igual que en `agents/agents/`, se auto-descubre exactamente igual (mismo
mecanismo, ver `agents/core/registry.py`). La única diferencia es de
intención: esta carpeta es para agentes que NO forman parte del template
"oficial" — cosas que bajas de un repo de terceros, que te pasa un
compañero, o que escribes tú para un proyecto concreto y no quieres que
vivan junto a git_agent.py, data_agent.py, etc.

Ver `agents/README.md`, sección "Agentes externos", para la lista curada de
proyectos de terceros que pueden servir de inspiración o integrarse aquí
(memoria a largo plazo, deep research, generación de skills...).
"""
