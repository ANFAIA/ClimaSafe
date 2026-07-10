---
type: modelo
created: 2026-07-09
tags:
  - modelo
  - randomforest
  - frio
  - produccion
status: active
---

# RandomForest — Frío

Modelo en producción para riesgo por frío. Seleccionado por recall en clase de riesgo.

## Resultados

| Métrica | Valor |
|---------|-------|
| cv_score | 0.8039 |

## Hiperparámetros

- `n_estimators`, `max_depth`, `min_samples_split`, `min_samples_leaf`, `bootstrap`, `class_weight`
- Tuning vía Optuna

## Tracking

MLflow: `climasafeai_RandomForest` (v1-v41 + versiones _frio)

## Ver también

- [[RandomForest_calor]]
- [[01_PROYECTO/modelos]]