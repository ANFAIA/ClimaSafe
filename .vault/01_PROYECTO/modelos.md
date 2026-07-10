---
type: proyecto
created: 2026-07-09
tags:
  - proyecto
  - modelos
  - ml
status: draft
---

# Modelos

## En producción

### XGBoost — Riesgo por Calor
- Seleccionado por recall en clase de riesgo
- Features: estadísticas 24h, lags, medias móviles de temperatura

### RandomForest — Riesgo por Frío
- Seleccionado por recall en clase de riesgo
- Features similares con ventanas temporales adaptadas

## Experimentos

- Grid search y tuning con optuna en `tuning/`
- Tracking con MLflow (`mlflow.db`)
- Notebooks exploratorios en `notebooks/`

## Pendiente

- [ ] Probar LSTM para secuencias temporales
- [ ] Monitoring de derivas en producción

## Ver también

- [[arquitectura]]
- [[02_DATOS/features]]