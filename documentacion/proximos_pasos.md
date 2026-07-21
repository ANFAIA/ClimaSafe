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
| ✅ | Checkboxes dinámicos desde `/api/factores` — cualquier factor nuevo aparece solo | `chat/static/index.html` |
| ✅ | Sección medicación (antipsicóticos, diuréticos de asa) | `chat/static/index.html` |
| ✅ | Sección exposición laboral (estrés térmico, esfuerzo térmico) | `chat/static/index.html` |
| ✅ | Checkbox alcohol reciente | `chat/static/index.html` |
| ✅ | Modal de aprobación de factores pendientes ("Salieron nuevos papers") | `chat/static/index.html` |
| ✅ | POST `/api/approve-factor` para implementar factor desde la web | `chat/app.py` |

### Agentes
| # | Qué | Archivos |
|---|-----|----------|
| ✅ | Paper scout v1 — busca papers, clasifica con LLM, guarda markdown + candidatos a `factores_riesgo.json` con `implementado: false` | `agents/paper_scout.py` |
| ✅ | Prompt del scout desambiguado, max_tokens 4096, extracción estructurada de factores (clave, categoria, tipo, coef, poblacion) | `agents/paper_scout.py` |
| ✅ | Auto-aprobación de factores con calidad=alta + DOI confirmado | `agents/paper_scout.py` (`_agregar_factor_json`) |
| ✅ | MCP server — 6 tools para gestionar factores | `agents/tools/factors_mcp_tool.py` |
| ✅ | Calibración isotónica post-hoc para RandomForest frío | `climasafeai/models/calibrate.py` |

### Personalización / ML
| # | Qué | Archivos |
|---|-----|----------|
| ✅ | Clase final desde probabilidad personalizada (si no hay override por HI) | `climasafeai/models/ensemble.py` |
| ✅ | Grasa baja como protectora en calor (×0.9) | `data/factores_riesgo.json` + `personalizacion.py` |
| ✅ | Categoría `ocupacional` en JSON con 2 factores | `data/factores_riesgo.json` |
| ✅ | `alcohol` en `calor.situacional` (×1.8, Semenza 1996) | `data/factores_riesgo.json` |
| ✅ | `respiratoria` en `calor.comorbilidades` (×1.3, Bunker 2016) | `data/factores_riesgo.json` |
| ✅ | DOIs corregidos (Semenza ...201→...203, cardiovascular frío cita Fan 2023) | `data/factores_riesgo.json` |

### Datos
| # | Qué | Archivos |
|---|-----|----------|
| ✅ | `data/factores_riesgo.json` con 26 factores implementados (calor + frío), 5 categorías | `data/factores_riesgo.json` |
| ✅ | Carpeta `modelos/` con papers de referencia (transformers, GNN, diffusion, N-BEATS) | `documentacion/modelos/` |
| ✅ | 68 papers revisados manualmente en 5 categorías | `documentacion/papers/` |

### Infraestructura
| # | Qué | Archivos |
|---|-----|----------|
| ✅ | GH Action paper_scout.yml con cron 14d + commit de papers y JSON | `.github/workflows/paper_scout.yml` |

---

## Lo que queda por hacer

### Alta prioridad

| # | Tarea | Archivos/detalle | Depende de |
|---|-------|-----------------|------------|
| 1 | **Diseñar esquema SQLite de perfiles** (tablas: `perfiles`, `factores_riesgo`, `historial_consultas`) | Crear `data/schema.sql` o migración con script Python. Campos: edad, sexo, comorbilidades, factores, historial de consultas, fechas. | — |
| 2 | **Migrar MCP server de JSON a SQLite** — las 6 tools de `factors_mcp_tool.py` deben leer/escribir en SQLite en vez de `factores_riesgo.json` | `agents/tools/factors_mcp_tool.py` | #1 |
| 3 | **Migrar `GET /api/factores` y `/api/approve-factor` a SQLite** — los endpoints deben leer de SQLite en vez del JSON | `chat/app.py` | #1 |
| 5 | **Probar scout con prompt nuevo** — cuando se restablezca cuota Gemini, ejecutar `uv run python -m agents scout --dry-run --query exertional` y verificar que extrae factores con clave, categoria, tipo, coef, poblacion | `agents/paper_scout.py` | Cuota Gemini |
| 6 | **Verificar auto-aprobación** — factores con calidad=alta+DOI deben guardarse con `implementado: true` en el JSON | `agents/paper_scout.py` | #5 |

### Media prioridad

| # | Tarea | Archivos/detalle | Depende de |
|---|-------|-----------------|------------|
| 7 | **Adaptación temporal de perfil** — si el usuario dijo "no aclimatado" y pasaron ≥14 días, preguntar si sigue igual. Usar `historial_consultas` en SQLite. | `chat/app.py` + lógica en `personalizacion.py` | #1 |
| 8 | **Endpoint web para guardar/cargar perfil** — que un usuario pueda guardar su perfil y recuperarlo después | `chat/app.py` + `index.html` | #1 |
| 9 | **Mostrar que `grasa_alta_esfuerzo` solo aplica en esfuerzo** — el factor de obesidad solo se activa con actividad moderada/intensa/muy_intensa. Añadir tooltip o nota al checkbox. | `chat/static/index.html` | — |
| 10 | **Añadir factor UV según fototipo** — el campo `fototipo` se envía pero se elimina en normalize_perfil. Podría modular riesgo UV (piel clara → más riesgo). | `personalizacion.py` + `factores_riesgo.json` | — |
| 11 | **Conectar paper scout a graphify** — que los papers encontrados alimenten el grafo de conocimiento | `agents/paper_scout.py` + `agents/tools/graphify_tool.py` | — |
| 12 | **Organizar `documentacion/`** — índice general, limpiar duplicados, unificar criterios | `documentacion/` | — |

### Baja prioridad

| # | Tarea | Archivos/detalle | Depende de |
|---|-------|-----------------|------------|
| 13 | **Validar thresholds de clase personalizada** (0.45 y 0.65) con datos reales — se eligieron a ojo en la sesión del 2026-07-21 | `climasafeai/models/ensemble.py` | — |
| 14 | **Unificar `alcohol_reciente`** — actualmente se puede enviar como `alcohol_reciente: true` (checkbox viejo) O como `situacion_social: ["alcohol"]` (dinámico). Sobran caminos. | `personalizacion.py` | — |
| 15 | **Limpiar `_normalize_perfil`** — los mapeos `hipertension→cardiovascular`, `respiratorio→respiratoria`, etc. ya no se envían desde la web dinámica. Código muerto. | `chat/app.py` | — |
| 16 | **Inyectar markdown de perfil** como contexto inicial del LLM del chat | MCP server + `chat/app.py` | #1 |
| 17 | **Añadir sqlite-vec** para búsqueda semántica de perfiles similares | `pyproject.toml` + esquema SQLite | #1 |
| 18 | **Cafeína como posible factor** — John 2024 (DOI: 10.1007/s00421-024-05460-z) muestra aumento de producción de calor (+7.9%) y temperatura central (+0.6°C), pero evidencia mixta. Investigar más. | `agents/paper_scout.py` (buscar) + opcional `factores_riesgo.json` | — |
| 19 | **Dockerizar stack completo** (api + llm-proxy + SQLite persistente) | `Dockerfile`, `docker-compose.yml` | #1 |

---

## Resumen visual

```
Alta ──┬── 1. Esquema SQLite
       ├── 2. Migrar MCP a SQLite
       ├── 3. Migrar API a SQLite
       ├── 5. Probar scout (cuota Gemini)
       └── 6. Verificar auto-aprobación

Media ─┬── 7.  Adaptación temporal perfil
        ├── 8.  Guardar/cargar perfil web
        ├── 9.  Tooltip grasa_alta_esfuerzo
        ├── 10. Factor UV según fototipo
        ├── 11. Conectar scout a graphify
        └── 12. Organizar documentación

Baja ───┬── 13. Validar thresholds
         ├── 14. Unificar alcohol_reciente
         ├── 15. Limpiar _normalize_perfil
         ├── 16. Contexto LLM desde SQL
         ├── 17. sqlite-vec
         ├── 18. Cafeína como factor (investigar)
         └── 19. Docker
```

---

## Referencias

- `conclusion-base-conocimiento.md` — decisión técnica de base de conocimiento
- `arquitectura/agentes_ia.md` — arquitectura de agentes existente
- `ml/conclusiones_modelos.md` — métricas y comparación de modelos actuales
- `arquitectura/diseño_modelo.md` — diseño general del sistema de predicción
- `README.md` — índice general de toda la documentación
