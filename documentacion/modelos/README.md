# Modelos — catálogo completo

Catálogo de todos los modelos ML del proyecto: actuales (desplegados) y
alternativos (exploración). Cada modelo tiene su propia carpeta con ficha técnica,
paper original y estado.

Alimentado por el **agente de búsqueda de papers** (ver `../proximos_pasos.md` §2).
El agente clasifica automáticamente nuevos modelos aquí y notifica al usuario
para exploración en notebook.

## Modelos actuales (desplegados)

| Modelo | Ficha | Asignado a | Rec_riesgo | Comentario |
|--------|-------|-----------|------------|------------|
| **RandomForest** | [`actuales/randomforest.md`](actuales/randomforest.md) | Frío | **0.527** | `class_weight="balanced"`, depth 8/12 |
| **XGBoost** | [`actuales/xgboost.md`](actuales/xgboost.md) | Calor | **0.614** | Con `sample_weight` balanceado (crítico) |
| **LSTM híbrida** | [`actuales/lstm.md`](actuales/lstm.md) | Explorado | inferior | No superó a RF/XGBoost con lags |

## Modelos alternativos (exploración)

| Modelo | Carpeta | Estado | Prioridad | Comentario |
|--------|---------|--------|-----------|------------|
| PatchTST (Transformer) | [`transformers/`](transformers/) | Pendiente | Baja | Forecasting series temporales |
| TimeSFormer | [`transformers/`](transformers/) | Pendiente | Baja | Forecasting con atención temporal |
| GNN (STGCN, GraphWaveNet) | [`gnn/`](gnn/) | Pendiente | Baja | Correlación espacial entre provincias |
| N-BEATS / N-HiTS | [`nbeats/`](nbeats/) | Pendiente | Baja | Forecasting univariante puro |
| Diffusion probabilístico | [`diffusion/`](diffusion/) | Pendiente | Baja | Generación de escenarios extremos |

## Criterios de selección

| Criterio | Peso | Descripción |
|----------|------|-------------|
| Compatibilidad con datos actuales | Alta | ¿Funciona con features tabulares + series temporales? |
| Mejora sobre XGBoost/RF/LSTM | Alta | ¿Supera en métricas reportadas en papers similares? |
| Esfuerzo de integración | Media | ¿Requiere cambiar el pipeline entero o se añade como rama? |
| Interpretabilidad | Media | ¿Podemos explicar por qué predice lo que predice? |
| Coste computacional | Baja | ¿Entrena en CPU o necesita GPU? |

## Flujo de incorporación

1. **Agente de papers** encuentra modelo → genera markdown en la carpeta correspondiente.
2. **Usuario revisa** el análisis de compatibilidad.
3. **Usuario experimenta** en notebook (no implementación directa).
4. Si es prometedor → se diseña un experimento formal vs baseline actual.
5. Si se adopta → se actualiza `../ml/conclusiones_modelos.md` con la comparativa.

## Ver también

- `../ml/conclusiones_modelos.md` — comparativa de modelos actuales (XGBoost, RF, LSTM).
- `../proximos_pasos.md` §5 — hoja de ruta de investigación de modelos.
