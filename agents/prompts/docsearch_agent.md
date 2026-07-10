# DocSearch Agent — Búsqueda y navegación por el grafo

Navega el grafo de conocimiento del `knowledge` agent: consultas en lenguaje
natural, vecinos de un nodo y poda de referencias/nodos innecesarios.

## Acciones

### `search` — Consulta la documentación en lenguaje natural
Delega en `graphify query` (cacheado).
```bash
uv run python -m agents run docsearch search --question "¿cómo se entrena el modelo?"
uv run python -m agents run docsearch search --question "..." --budget 1500
```

### `neighbors` — Vecinos de un nodo (navegación por el árbol)
Acepta id o label del nodo.
```bash
uv run python -m agents run docsearch neighbors --node "train_model"
```

### `list_references` — Lista los nodos de tipo referencia/cita/enlace
```bash
uv run python -m agents run docsearch list_references
```

### `prune` — Poda nodos del grafo
Por tipo, por id, o los que queden aislados. **`dry_run=True` por defecto**:
informa sin escribir. Pasa `dry_run=False` para aplicar (deja `graph.json.bak`).
```bash
# simular quitar todas las referencias
uv run python -m agents run docsearch prune --node_types '["reference"]'
# aplicar de verdad
uv run python -m agents run docsearch prune --node_types '["reference"]' --dry_run false
```

## Nota
La poda solo modifica la representación en `graphify-out/graph.json` (con
backup). Nunca borra archivos fuente del proyecto.
