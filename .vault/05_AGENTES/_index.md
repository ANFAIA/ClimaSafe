---
type: agente
created: 2026-07-09
tags:
  - agentes
  - indice
  - core
status: active
---

# Agentes — Índice

15 agentes auto-registrados que heredan de `BaseAgent`. Cada uno es un archivo `.py` en `agents/agents/` decorado con `@register_agent`.

## Arquitectura

- [[../../agents/core/base_agent|BaseAgent]] — clase base
- [[../../agents/core/registry|Registry]] — registro automático
- [[../../agents/orchestrator|Orchestrator]] — ruteo por keywords
- [[../../agents/cli|CLI]] — invocación desde terminal
- [[../../agents/exceptions|Excepciones]] — jerarquía propia

## Agentes

| Agente | Propósito | Nodo en grafo |
|--------|-----------|--------------|
| [[APIAgent]] | Valida API REST | 14 conexiones |
| [[CICDAgent]] | GitHub Actions | 17 conexiones |
| [[DataAgent]] | Calidad de datos | 27 conexiones |
| [[DependencyAgent]] | Dependencias PyPI | 31 conexiones |
| [[DockerAgent]] | Dockerfile/compose | 7 conexiones |
| [[DocumentationAgent]] | Docs, README, CHANGELOG | 29 conexiones |
| [[GitAgent]] | Git, changelog, PRs | 33 conexiones |
| [[GraphAgent]] | Inspección de figuras | 6 conexiones |
| [[InstallerAgent]] | Instala agentes externos | 22 conexiones |
| [[MLAgent]] | Análisis de modelos | 7 conexiones |
| [[MLflowAgent]] | MLflow runs | 12 conexiones |
| [[NotebookAgent]] | Notebooks Jupyter | 19 conexiones |
| [[ReviewAgent]] | Code review | 18 conexiones |
| [[SecretsAgent]] | Escaneo de secretos | 16 conexiones |
| [[TestAgent]] | Pytest + cobertura | 7 conexiones |

## Ver también

- [[01_PROYECTO/agentes]]
- [[01_PROYECTO/arquitectura]]
- [[04_VISUALIZACIONES/grafo_conocimiento]]