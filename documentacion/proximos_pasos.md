# Próximos pasos — hoja de ruta

**Última revisión:** 2026-07-21

---

## Lo hecho ✅

### Web
| # | Qué | Archivos |
|---|-----|----------|
| ✅ | `GET /api/pending-factors` — devuelve factores con `implementado: false` | `chat/app.py:470` |
| ✅ | `GET /api/factores` — devuelve factores implementados agrupados por tipo/categoría | `chat/app.py:453` |
| ✅ | `POST /api/predict` — predicción completa con normalización de perfil y factores personalizados | `chat/app.py:478` |
| ✅ | `GET /api/status` ahora incluye `has_pending_factors` | `chat/app.py:407` |
| ✅ | Banner amarillo en frontend si hay factores pendientes | `chat/static/index.html` |

### Agentes
| # | Qué | Archivos |
|---|-----|----------|
| ✅ | Paper scout v1 — busca papers, clasifica con LLM, guarda markdown + candidatos a `factores_riesgo.json` con `implementado: false` | `agents/paper_scout.py` |
| ✅ | MCP server — 6 tools para gestionar factores (`get_factors`, `suggest_factor`, `approve_factor`, `reject_factor`, `update_factor`, `pending_factors`). Arrancar con `uv run python -m agents.tools.factors_mcp_tool` | `agents/tools/factors_mcp_tool.py` |
| ✅ | Calibración isotónica post-hoc para RandomForest frío | `climasafeai/models/calibrate.py` |

### Datos
| # | Qué | Archivos |
|---|-----|----------|
| ✅ | `data/factores_riesgo.json` con 20 factores implementados (calor + frío), estructura por categorías | `data/factores_riesgo.json` |
| ✅ | Carpeta `modelos/` con papers de referencia (transformers, GNN, diffusion, N-BEATS) | `documentacion/modelos/` |

---

## Lo que queda por hacer

### Alta prioridad

| # | Tarea | Archivos/detalle | Depende de |
|---|-------|-----------------|------------|
| 1 | **Diseñar esquema SQLite de perfiles** (tablas: `perfiles`, `factores_riesgo`, `historial_consultas`) | Crear `data/schema.sql` o migración con script Python. Campos: edad, sexo, comorbilidades, factores, historial de consultas, fechas. | — |
| 2 | **Migrar MCP server de JSON a SQLite** — las 6 tools de `factors_mcp_tool.py` deben leer/escribir en SQLite en vez de `factores_riesgo.json` | `agents/tools/factors_mcp_tool.py` | #1 |
| 3 | **Migrar `GET /api/factores` a SQLite** — el endpoint debe leer de SQLite en vez del JSON | `chat/app.py` | #1 |
| 4 | **Agente que auto-apruebe factores del scout** — cuando el scout encuentra un factor con calidad alta + DOI confirmado, que lo active automáticamente sin necesidad de `scout --review` manual | `agents/paper_scout.py` o nuevo agente. Usar las tools MCP (`suggest_factor` → `approve_factor`) | — |

### Media prioridad

| # | Tarea | Archivos/detalle | Depende de |
|---|-------|-----------------|------------|
| 5 | **GitHub Action con cron 14 días** para ejecutar `python -m agents scout` automáticamente | Crear `.github/workflows/scout.yml` | — |
| 6 | **Adaptación temporal de perfil** — si el usuario dijo "no aclimatado" y pasaron ≥14 días, preguntar si sigue igual. Usar `historial_consultas` en SQLite. | `chat/app.py` + lógica en `personalizacion.py` | #1 |
| 7 | **Conectar paper scout a graphify** — que los papers encontrados alimenten el grafo de conocimiento | `agents/paper_scout.py` + `agents/tools/graphify_tool.py` | — |
| 8 | **Organizar `documentacion/`** — índice general, limpiar duplicados, unificar criterios | `documentacion/` | — |

### Baja prioridad

| # | Tarea | Archivos/detalle | Depende de |
|---|-------|-----------------|------------|
| 9 | **Añadir sqlite-vec** para búsqueda semántica de perfiles similares | `pyproject.toml` + esquema SQLite | #1 |
| 10 | **Inyectar markdown de perfil** como contexto inicial del LLM | MCP server + `chat/app.py` | #1 |
| 11 | **Endpoint web para aprobar factores** — alternativa gráfica al CLI `scout --review` | `chat/app.py` + `index.html` | — |
| 12 | **Dockerizar stack completo** (api + llm-proxy + SQLite persistente) | `Dockerfile`, `docker-compose.yml` | #1 |

---

## Resumen visual

```
Alta ──┬── 1. Esquema SQLite
       ├── 2. Migrar MCP a SQLite
       ├── 3. Migrar GET /api/factores a SQLite
       └── 4. Auto-aprobación de factores (calidad alta)

Media ─┬── 5. GH Action cron scout
        ├── 6. Adaptación temporal perfil
        ├── 7. Conectar scout a graphify
        └── 8. Organizar documentación

Baja ───┬── 9. sqlite-vec
         ├── 10. Contexto LLM desde SQL
         ├── 11. Aprobación web de factores
         └── 12. Docker
```

---

## Referencias

- `conclusion-base-conocimiento.md` — decisión técnica de base de conocimiento
- `arquitectura/agentes_ia.md` — arquitectura de agentes existente
- `ml/conclusiones_modelos.md` — métricas y comparación de modelos actuales
- `arquitectura/diseño_modelo.md` — diseño general del sistema de predicción
- `README.md` — índice general de toda la documentación
