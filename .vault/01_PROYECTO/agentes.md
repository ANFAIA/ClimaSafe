---
type: proyecto
created: 2026-07-09
tags:
  - proyecto
  - agentes
status: draft
---

# Sistema de Agentes

15 agentes auto-registrados que heredan de `BaseAgent`.

## Arquitectura

- `BaseAgent` + `AgentResult` en `agents/core/base_agent.py`
- `@register_agent` para registro automático
- `Registry` centralizado en `agents/core/registry.py`
- `Orchestrator` en `agents/orchestrator.py` con ruteo por keywords
- CLI en `agents/cli.py`

## Invocación

```bash
python -m agents run <agente> <acción>
```

O mediante lenguaje natural vía el orquestador.

## Agentes

(Lista completa de agentes con su propósito — crear notas individuales en [[05_AGENTES/]])

## Ver también

- [[arquitectura]]
- [[05_AGENTES/]] — fichas individuales