# Aprendizaje activo — paper scout

## Qué es

Clasificador ligero que filtra papers irrelevantes **antes** de la llamada LLM, ahorrando tokens y tiempo. Se re-entrena automáticamente con cada veredicto (approve/reject) del usuario.

## Arquitectura

```
Paper → ActiveLearner.predict() → "relevante" → LLM clasifica
                                 → "irrelevante" → descartado (sin LLM)
User approve/reject → ActiveLearner.add_ejemplo() → retrain
```

## Implementación

`climasafeai/ml/active_learner.py`

### `ActiveLearner`

- **Embedder**: `all-MiniLM-L6-v2` (384d, mismo que RAG)
- **Clasificador**: `LogisticRegression(C=1.0, class_weight="balanced")`
- **Estado inicial**: abstiene (devuelve `None`) hasta tener ≥5 muestras
- **Umbral de confianza**: `CONFIDENCE_THRESHOLD = 0.75` — por debajo, pasa al LLM aunque clasifique como irrelevante

### Métodos

| Método | Qué hace |
|--------|----------|
| `predict(text)` | `(relevante: bool | None, confianza: float)` |
| `add_ejemplo(text, es_relevante)` | Almacena y re-entrena si hay datos nuevos |
| `entrenar()` | Re-entrena LogisticRegression desde cero |
| `estado()` | N muestras por clase, total, accuracy estimado |

## Integración en paper scout

En `agents/paper_scout.py`:

1. `scout_run()` itera los papers y llama a `active_learner.predict(title + abstract)` antes del LLM
2. Si `predict` devuelve `False` con confianza ≥ 0.75 → descarta sin LLM
3. Si `predict` devuelve `True` o baja confianza → pasa a `_classify_batch()` (LLM)
4. Tras el veredicto del LLM, llama a `active_learner.store_many()` con los resultados
5. Al aprobar/rechazar un factor desde el modal de la web, se añade el abstract como ejemplo

## Re-entrenamiento manual

```bash
uv run python scripts/entrenar_active_learner.py
```

Carga los factores aceptados/rechazados desde `factores_riesgo` y re-entrena.

## Estado actual

- 27 ejemplos positivos + 15 negativos (sintéticos desde factores_riesgo)
- Accuracy esperada: ~70-80% (mejora con más datos reales)
- Bloqueado hasta que se restablezca cuota Gemini para generar más datos
