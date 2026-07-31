# Doc Agent — Documentación unificada del proyecto

Combina 3 fuentes de conocimiento: **graphify** (grafo estructural), **RAG** (búsqueda semántica vectorial), **vault** (notas Obsidian).

## Acciones

| Acción | Parámetros | Descripción |
|--------|-----------|-------------|
| `search` | `--query`, `--sources` (all/graph/rag/vault) | Busca en las fuentes indicadas |
| `graph_query` | `--question` | Consulta estructural al grafo graphify |
| `rag_search` | `--query`, `--top-k` | Búsqueda semántica vía ChromaDB |
| `vault_grep` | `--pattern` | Búsqueda textual directa en vault markdown |
| `index` | — | Construye grafo graphify + índice RAG en un paso |
| `status` | — | Estado de cada fuente |

## Fuentes

- **Graphify**: relaciones estructurales entre módulos, dependencias, nodos del proyecto
- **RAG**: embeddings semánticos de código, prompts, docs y vault (requiere chromadb)
- **Vault**: notas markdown de Obsidian generadas por `knowledge build`

## Integración

- `doc search` es el punto de entrada único para preguntas sobre el proyecto
- Cuando un agente necesita contexto, puede delegar a `doc` en vez de consultar cada fuente por separado
- El subagente `doc` en opencode permite chatear directamente con la documentación del proyecto

<!-- BEGIN AUTOGEN — lo regenera `make prompts-sync`; no lo edites a mano -->

## Acciones

| Acción | Argumentos |
|--------|------------|
| `run doc search` | `--query` (obligatorio) · `--sources` |
| `run doc graph_query` | `--question` (obligatorio) · `--budget`, `--no_cache` |
| `run doc rag_search` | `--query` (obligatorio) · `--top_k` |
| `run doc vault_grep` | `--pattern` (obligatorio) |
| `run doc neighbors` | `--node` (obligatorio) · `--limit` |
| `run doc list_references` | — |
| `run doc index` | — |
| `run doc status` | — |

## Límites

**Rol.** Buscador unificado de documentación: grafo graphify, índice RAG y notas del vault. Solo lee.

**No hace:**
- construir, podar o modificar el grafo → knowledge
- buscar fuera del proyecto (papers nuevos, web) → research
- escribir documentación (README, CHANGELOG, docs/) → documentation

**Necesita que le den:** la consulta o el nodo del que partir

**Se apoya en:** knowledge, documentation

<!-- END AUTOGEN -->
