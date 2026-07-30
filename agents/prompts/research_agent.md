# Research Agent — Papers relacionados con el proyecto

Deriva las palabras clave del proyecto (README, pyproject, grafo) y busca papers
relevantes en fuentes abiertas (**arXiv**, **OpenAlex**), sin API key.

## Acciones

### `project_keywords` — Palabras clave del proyecto
```bash
uv run python -m agents run research project_keywords --top 12
```

### `find_papers` — Papers relacionados con el proyecto
```bash
uv run python -m agents run research find_papers --backend openalex --max_results 10
uv run python -m agents run research find_papers --backend arxiv
```

### `search` — Búsqueda directa por consulta
```bash
uv run python -m agents run research search --query "graph neural networks" --backend arxiv
```

## Fuentes
- **arxiv**: preprints (Atom XML de `export.arxiv.org`).
- **openalex**: catálogo abierto con recuento de citas (`api.openalex.org`).

## Límite honesto
La relevancia es **léxica** (solapamiento de keywords con título+abstract), no
una lectura semántica del paper. Requiere internet; sin red cada búsqueda falla
de forma controlada. Resultados cacheados en `graphify-out/cache/`.

<!-- BEGIN AUTOGEN — lo regenera `make prompts-sync`; no lo edites a mano -->

## Acciones

| Acción | Argumentos |
|--------|------------|
| `run research project_keywords` | `--top` |
| `run research find_papers` | `--backend`, `--max_results`, `--top_keywords` |
| `run research search` | `--query` (obligatorio) · `--backend`, `--max_results`, `--no_cache` |

## Límites

**Rol.** Investigador externo: busca papers (arXiv/OpenAlex) relacionados con el proyecto. Solo lee.

**No hace:**
- indexar papers en el grafo o la bóveda → knowledge
- decidir qué paper adoptar — presenta candidatos, el humano (o supervisor) elige

**Se apoya en:** knowledge, supervisor

<!-- END AUTOGEN -->
