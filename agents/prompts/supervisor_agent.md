# Supervisor Agent — Workers que compiten

Hace que varios workers trabajen **por separado** sobre la misma tarea, prueba
cada propuesta con una métrica, elige la mejor y la pule.

## Acciones

### `research` — Competición de búsqueda de papers
Lanza el `research` agent con cada backend (arXiv, OpenAlex) en paralelo, puntúa
cada propuesta (relevancia + cobertura + volumen), elige la ganadora y pule
fusionando lo mejor de ambas.
```bash
uv run python -m agents run supervisor research --max_results 10
uv run python -m agents run supervisor research --backends '["arxiv","openalex"]'
```

### `compete` — Competición genérica
Enfrenta candidatos arbitrarios; cada uno es `{agent, action, kwargs, label}`.
```bash
uv run python -m agents run supervisor compete --candidates \
  '[{"agent":"research","action":"find_papers","kwargs":{"backend":"arxiv"},"label":"arxiv"},
    {"agent":"research","action":"find_papers","kwargs":{"backend":"openalex"},"label":"openalex"}]'
```

## Cómo puntúa (determinista, no un juez LLM)
- **research**: `0.5·relevancia_media + 0.4·cobertura_keywords + 0.1·volumen`.
- **compete**: premia éxito + riqueza de `data`, penaliza warnings.

La métrica es el punto de extensión si quieres un arbitraje más sofisticado —
el resto del sistema no cambia.
