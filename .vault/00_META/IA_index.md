---
type: meta
status: active
created: 2026-07-09
ia: readme
---

# Guía para la IA

Este vault organiza el proyecto **ClimaSafeAI**. Úsalo como fuente de verdad para entender contexto, decisiones y estado.

## Convenciones

- `#etiqueta` para temas transversales
- `[[wikilinks]]` para conectar notas relacionadas
- `propiedades` (frontmatter) para metadatos estructurados
- Cada nota debe tener al menos: `tags`, `created`, `type`

## Estructura

| Carpeta | Propósito |
|---------|-----------|
| `00_META/templates/` | Plantillas reutilizables |
| `01_PROYECTO/` | Arq., decisiones, roadmap |
| `02_DATOS/` | Fuentes, features, datasets |
| `03_MODELOS/` | Experimentos, métricas |
| `04_VISUALIZACIONES/` | Gráficos, dashboards |
| `05_AGENTES/` | Documentación de agentes |
| `06_OBSERVACIONES/` | Notas diarias, ideas sueltas |
| `07_REFERENCIAS/` | Papers, enlaces externos |

## Flujo de trabajo recomendado

1. Al empezar una tarea, leer `IA_index.md`
2. Buscar notas existentes por tag o propiedad
3. Actualizar o crear notas a medida que se trabaja
4. Mantener enlaces entre notas para que el grafo sea navegable