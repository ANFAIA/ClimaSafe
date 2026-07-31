# Paso 1 — RAG: indexar `documentacion/`

**Fecha:** 2026-07-30
**Feature:** LLM-001 (sub-paso 1)
**Estado:** Completado

## Qué se hizo

Extender el módulo `climasafeai/db/rag.py` — que ya indexaba `factores_riesgo` de la BD con `sqlite-vec` + `all-MiniLM-L6-v2` (384 dims) — para que también indexe los archivos `.md` de `documentacion/`.

## Decisiones de diseño

### 1. Chunking por secciones (`##`)

Cada archivo `.md` se divide por encabezados de segundo nivel (`##`). Cada sección se convierte en un fragmento independiente con:

- `ruta`: ruta completa al `.md`
- `titulo`: el primer `# Título` del documento (o nombre del archivo si no hay)
- `seccion`: el texto tras el `##` (o `__intro__` para el preámbulo)
- `texto`: el contenido de la sección
- `palabras`: conteo de palabras (para filtrar fragmentos muy cortos)

Se excluyen fragmentos con <10 palabras (ruido).

### 2. Virtual table independiente

`docs_vec` (embeddings) + `docs_vec_src` (metadatos), separada de `factores_vec`. Motivo:
- Distintos orígenes de datos (BD vs sistema de archivos)
- Consultas semánticas diferentes: buscar factores de riesgo vs buscar documentación del proyecto
- Posibilidad de limpiar y reindexar docs sin tocar factores

### 3. Indexado incremental

`sync_documentos()` solo indexa fragmentos nuevos, comparando `ruta+seccion` contra lo ya guardado en `docs_vec_src`. Para reindexar completo existe `resync_documentos()`.

### 4. Auto-indexado en `initialize()`

Al arrancar el RAG (`DBManager.init_rag()`), se llama a `sync_documentos()` automáticamente. En la práctica solo añade fragmentos nuevos ~ la primera carga es la pesada (509 fragmentos, 84k palabras).

### 5. MCP tools nuevas

| Tool | Descripción |
|------|-------------|
| `search_documentos_mcp` | Búsqueda semántica solo en documentación |
| `search_all_mcp` | Búsqueda combinada: factores + docs |

## Resultados

- **Fragmentos indexados:** 509
- **Palabras totales:** 83,628
- **Tiempo de primera indexación:** ~2 segundos (incluye carga del embedder)
- **Búsqueda semántica:** funcional, devuelve fragmentos con distancia coseno

## Pendiente

- Indexar `documentacion/llm/` (excluida explícitamente con `DOCS_EXCLUDE`)
- Integrar el RAG de documentación en el prompt del Qwen (Paso 2)
- Añadir filtro por documento/tema a la búsqueda
