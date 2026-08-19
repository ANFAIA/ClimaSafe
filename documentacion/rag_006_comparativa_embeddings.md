# RAG-006 — Comparativa de embeddings y solapamiento (decisión por los números)

Este documento registra la comparación de **modelos de embeddings** y
**solapamiento de chunks** contra la línea base de RAG-004, y por qué se eligió
lo que se eligió. Es la constancia escrita que exige RAG-006: cada cambio se
mide contra la línea base; si no mejora el recall, se revierte y se dice.

## Línea base de RAG-004 (a comparar)

Estado del retrieval antes de RAG-006, medido con `scripts/evaluar_rag.py`
sobre `data/climasafe.db` (modelo `all-MiniLM-L6-v2`, sin solapamiento, k=5):

- **factores**:   recall@5 = **0.780** (25 preguntas)
- **documentos**: recall@5 = **0.325** (39 preguntas)

## Cómo se midió

`scripts/comparar_modelos_rag.py` mide el recall@k por canal sobre
`data/rag/eval_set.json` para varios estados del índice, en este orden:

1. línea base (índice activo tal cual),
2. índice activo reindexado con solapamiento de 200 caracteres,
3. modelo alternativo sin solapamiento,
4. modelo alternativo con solapamiento.

## Resultados (k=5)

| Config | factores recall@5 | documentos recall@5 |
|--------|-------------------|---------------------|
| 1. Línea base (all-MiniLM-L6-v2, sin solape) | 0.780 | 0.325 |
| 2. + solape 200 (all-MiniLM-L6-v2) | 0.780 | **0.286** ↓ |
| 3. + modelo alt (distiluse-base-multilingual-cased-v2, sin solape) | **0.940** ↑ | **0.611** ↑ |
| 4. + modelo alt (distiluse-base-multilingual-cased-v2, solape 200) | 0.940 | 0.526 ↓ |

## Decisión

### Modelo de embeddings: `distiluse-base-multilingual-cased-v2`

Gana **por los números** en ambos canales frente a la línea base:

- documentos: 0.325 → **0.611** (sin solape)
- factores:   0.780 → **0.940** (sin solape)

Por eso se cambió el modelo por defecto del RAG a
`distiluse-base-multilingual-cased-v2` (dimensión 512), se reindexó
`data/climasafe.db` y se confirmó el resultado sobre la BD real: factores
0.940, documentos 0.611. El modelo anterior (`all-MiniLM-L6-v2`, 384) queda
disponible como modelo alternativo (tablas `*_minilm`) para poder re-comparar
sin perder nada.

### Solapamiento: **revertido** (no mejora)

El solapamiento **empeora** el recall de documentos en ambos modelos:

- all-MiniLM: 0.325 → 0.286
- distiluse:  0.611 → 0.526

Por qué empeora: el chunking es **por secciones** (`##`), y el solapamiento
antecede a cada sección la cola de la anterior. Eso diluye el embedding de la
sección con texto ajeno al centro semántico de la pregunta; además, los papers
del set de evaluación son en su mayoría de **una sola sección** (no hay sección
anterior de la que tomar cola), así que el solapamiento no aporta en los casos
que más fallaban.

Conclusión: **se revierte** — `CHUNK_OVERLAP = 0` (desactivado por defecto). El
código del solapamiento queda implementado y configurable por si se quisiera
re-medir con otro chunking (p. ej. por tamaño fijo en vez de por secciones).

### Conteo de fragmentos antes/después (con solapamiento)

El solapamiento no cambia el **número** de fragmentos (sigue siendo uno por
sección, con la misma clave de dedup `ruta::seccion`); lo que cambia es el
contenido de cada fragmento:

| | Fragmentos | Palabras |
|--|-----------|----------|
| Sin solapamiento | 614 | 101 538 |
| Con solapamiento (200) | 614 | 113 178 (+11 640) |

## Estado final del código

- `RAG_EMBEDDER_DEFAULT = "distiluse-base-multilingual-cased-v2"` (512 dims).
- `CHUNK_OVERLAP = 0` (solapamiento revertido).
- Esquema multi-modelo: el índice activo (tablas sin sufijo) usa el modelo por
  defecto; los alternativos viven en `docs_vec_<slug>` / `factores_vec_<slug>`
  con su propia dimensión, de modo que se pueden comparar sin perder el índice
  anterior.
