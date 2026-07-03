# agents/external/

Aquí van agentes que **no** forman parte del núcleo del template: de
terceros, de otro proyecto tuyo, o experimentos que no quieres mezclar con
`agents/agents/`.

## Opción 1 — un archivo suelto (la más simple)

Copia un módulo `.py` aquí con una clase `@register_agent`, exactamente con
la misma forma que cualquier agente de `agents/agents/` (ver
`agents/agents/_template_agent.py` como plantilla). El registro
(`agents/core/registry.py`) lo descubre automáticamente la próxima vez que
llames a `Orchestrator()` o `python -m agents list` — no hace falta tocar
ningún import.

```
agents/external/
  mi_agente_de_otro_proyecto.py   # define @register_agent class ...
```

## Opción 2 — un paquete pip instalado (para agentes que mantienes aparte)

Si tu agente externo vive en su propio repo/paquete, expón un *entry point*
en el grupo `dskit.agents` de su `pyproject.toml`:

```toml
[project.entry-points."dskit.agents"]
mi_agente = "mi_paquete.mi_modulo:MiAgente"
```

Con el paquete instalado en el mismo entorno (`uv pip install -e ../mi-paquete`
o desde PyPI), `agent_registry.discover()` lo encuentra automáticamente vía
`importlib.metadata.entry_points()` — no necesitas copiar ningún archivo
aquí. Esta vía es mejor cuando el agente tiene sus propias dependencias o lo
usas en varios proyectos.

## Qué NO cambia

Ni el `Orchestrator`, ni la CLI, ni ningún agente del núcleo necesitan saber
que un agente es "externo" — una vez registrado, se comporta exactamente
igual que `GitAgent` o `DataAgent`. La distinción entre `agents/agents/` y
`agents/external/` es solo organizativa, para ti.

## Ideas para agentes externos (no incluidos, ver `agents/README.md`)

La sección "Agentes externos y lecturas recomendadas" del README raíz de
`agents/` recoge proyectos de terceros (memoria a largo plazo, deep
research, generación de nuevas skills...) que pueden servir de base para un
agente aquí. Ninguno viene integrado por defecto — hay que evaluarlos,
instalarlos y adaptarlos tú.
