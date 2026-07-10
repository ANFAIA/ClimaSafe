---
type: modelo
created: 2026-07-09
tags:
  - modelo
  - xgboost
  - calor
  - produccion
status: active
---

# XGBoost — Calor

Modelo en producción para riesgo por calor.

## Resultados

| Métrica | Valor |
|---------|-------|
| cv_score | 0.9028 |
| recall_riesgo | (óptimo) |

## Hiperparámetros (último tuning)

- `n_estimators`, `max_depth`, `learning_rate`, `colsample_bytree`, `reg_alpha`, `reg_lambda`
- Tuning vía Optuna + GridSearch

## Features

Estadísticas 24h, lags (t-1, t-2, t-3, t-7), medias móviles (3, 7, 14 días) de temperatura.

## Tracking

- MLflow: `climasafeai_XGBoost` (v1-v41 + versiones _calor)
- Runs de entrenamiento: `XGBoost` (cv_score)
- Runs de eval: `XGBoost_eval` (acc, f1, prec, rec)

## Ver también

- [[XGBoost_frio]]
- [[RandomForest_calor]]
- [[02_DATOS/features]]