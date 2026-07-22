# Base de datos — SQLite + RAG semántico

## Esquema

Archivo: `data/schema.sql` → `data/climasafe.db`

### Tablas principales

| Tabla | Propósito |
|-------|-----------|
| `perfiles` | Perfiles de usuario (edad, sexo, factores, ...) |
| `factores_riesgo` | Factores de riesgo con metadatos (categoría, fuente, calidad) |
| `consultas` | Historial de predicciones |
| `perfil_factor` | Relación MxM perfiles ↔ factores |
| `configuracion` | KV store para config |

### RAG semántico

`factores_vec` — tabla `vec0` de sqlite-vec con embeddings de `all-MiniLM-L6-v2` sobre `factores_riesgo`.

`factores_vec_src` — metadatos de cada embedding (factor_id, nombre, categoria).

## DBManager

`climasafeai/db/manager.py` — singleton que gestiona toda la base de datos.

### Métodos principales

| Método | Qué hace |
|--------|----------|
| `crear_perfil(datos)` | Inserta perfil, devuelve perfil_id |
| `actualizar_perfil(id, datos)` | UPDATE parcial |
| `obtener_perfil(id)` | SELECT por id |
| `guardar_consulta(...)` | Inserta histórico de predicción |
| `listar_historial(perfil_id)` | Consultas anteriores de un perfil |
| `init_rag()` | Inicializa tabla vec0 y sincroniza factores |
| `search_factores(query, k)` | Búsqueda semántica top-k |

### Inicialización

En `app.py` startup: `DBManager("data/climasafe.db")`.
`init_rag()` se llama automáticamente si la tabla vec0 no existe.

## RAG

`climasafeai/db/rag.py`

### `RAG` class

- `embed(text: str) → list[float]` — all-MiniLM-L6-v2, 384d
- `search_factores(query, k=5) → list[dict]` — búsqueda semántica con distancia coseno
- `sincronizar(sync_fn)` — reindexa todos los factores

### Endpoint

`POST /api/rag-search` — body `{ "query": "...", "k": 5 }`.

MCP tool: `search_factors_mcp` en `agents/tools/factors_mcp_tool.py`.

## Referencias

- `documentacion/conclusion-base-conocimiento.md` — decisión técnica SQLite + RAG
- `documentacion/arquitectura/pipeline_prediccion.md` — cómo se usa en el flujo de predicción
- `climasafeai/db/manager.py` — implementación
- `climasafeai/db/rag.py` — implementación RAG
