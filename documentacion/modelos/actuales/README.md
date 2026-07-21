# Modelos actuales en producción

Modelos entrenados y desplegados en el sistema de predicción.

| Modelo | Tipo | Target | Estado |
|--------|------|--------|--------|
| `XGBoost_calor` | Gradient Boosting | Riesgo por calor | Producción |
| `RandomForest_frio` | Random Forest | Riesgo por frío | Producción |
| `LSTM province_hybrid` | Red neuronal recurrente | Calor + frío (multi-tarea) | Producción |

## Detalle

- [`xgboost.md`](xgboost.md) — XGBoost para calor
- [`randomforest.md`](randomforest.md) — RandomForest para frío
- [`lstm.md`](lstm.md) — LSTM híbrida con embedding provincial
