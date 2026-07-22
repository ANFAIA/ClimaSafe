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

## Mejoras futuras (evaluadas, no implementadas)

Técnicas exploradas conceptualmente o en notebook (`0-5-experimentos.ipynb`)
pero no integradas en producción. Se documentan para no perder la evaluación.

| Técnica | Dónde se exploró | Estado | Motivo del descarte / aplazamiento |
|---------|-----------------|--------|-------------------------------------|
| **Procesos Gaussianos (GP)** | Prototipo | El GP clásico escala O(n³), inviable con el dataset real (~20k filas). Un SVGP sparse sería viable para el índice personalizado (~1000 consultas), pero conformal prediction (ya implementado) cubre la misma necesidad sin añadir dependencias. |
| **TFT / N-BEATS** | Descartado | Forecast multi-horizonte para un problema que solo necesita predicción a 1-2 días. La LSTM actual es suficiente. Ver `../arquitectura/diseño_modelo.md` §7.2. |
| **RL (PPO puro + curriculum)** | Prototipo | Entorno sintético con reglas escritas a mano. Para que RL tuviera sentido necesitaría un ciclo de feedback real (usuarios → recomendaciones → resultado), que hoy no existe. |
| **Aprendizaje por refuerzo (RL)** | Conversación | Descartado | Misma razón: sin bucle de feedback real, RL es una solución en busca de problema. |
| **Teoría causal (Pearl, Do-calculus)** | Conversación | Marco mental | No hay nada que implementar. Es un marco conceptual que cambia cómo se diseñan features y se interpretan resultados, no una librería. |
| **Explicaciones contrafactuales** | Conversación | Pendiente de evaluar | Sí tendría valor real: "si hoy trabajaras a la sombra, tu riesgo bajaría de peligro a precaución". Se implementaría con `scipy.optimize` sobre los factores de `personalizacion.py`. Pendiente de priorizar. |

**Criterio general:** ninguna de estas técnicas se descarta para siempre. Si el proyecto crece (más usuarios, feedback real, datos a nivel de calle), algunas podrían re-evaluarse. Por ahora, las prioridades son las que están en `../proximos_pasos.md`.

## Ver también

- `../ml/conclusiones_modelos.md` — comparativa de modelos actuales (XGBoost, RF, LSTM).
- `../proximos_pasos.md` — hoja de ruta y próximos pasos.
- `0-5-experimentos.ipynb` — notebook con prototipos de GP y RL.
- `../arquitectura/diseño_modelo.md` — decisiones de arquitectura y modelos descartados.
