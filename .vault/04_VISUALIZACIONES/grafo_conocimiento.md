---
type: visualizacion
created: 2026-07-09
tags:
  - grafo
  - graphify
  - conocimiento
status: active
---

# Grafo de Conocimiento (Graphify)

Generado el 2026-07-09 con Graphify + Gemini sobre todo el proyecto.

## Métricas

| Métrica     | Valor                         |
| ----------- | ----------------------------- |
| Nodos       | 1077                          |
| Aristas     | 2216                          |
| Comunidades | 71                            |
| Extracción  | 95% AST, 5% inferida (Gemini) |
| Coste       | ~$0.0                         |
| Commit      | `4c5415f4`                    |

## God Nodes (mayor centralidad)

| Nodo | Conexiones | Rol |
|------|-----------|-----|
| `AgentResult` | 101 | Puente entre casi todos los agentes |
| `BaseAgent` | 53 | Clase base del sistema de agentes |
| `SharedContext` | 49 | Contexto compartido entre agentes |
| `MissingDependencyError` | 34 | Excepción central |
| `GitAgent` | 33 | Automatización git |

## Comunidades destacadas

- Feature Engineering Pipeline
- Model Training and Evaluation
- Agent System Overview (14 agentes)
- Meteorological Index Calculations (ERA5, índices de calor)
- Risk Label Assignment (percentiles → SEGURO/PRECAUCION/PELIGRO)

## Archivos de salida

- `graphify-out/graph.json` — grafo completo (1077 nodos, 2216 aristas)
- `graphify-out/graph.html` — visualización interactiva
- `graphify-out/GRAPH_REPORT.md` — reporte detallado

## Próximos pasos

- [ ] Revisar 16 nodos aislados y conectarlos
- [ ] Resolver colisiones (data_agent, ml_agent)
- [ ] Verificar 20 aristas inferidas de `AgentResult`
- [ ] Mantener actualizado con `graphify update .`

## Ver también

- [[01_PROYECTO/arquitectura]]
- [[01_PROYECTO/agentes]]
- [[00_META/IA_index]]