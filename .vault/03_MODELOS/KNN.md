---
type: modelo
created: 2026-07-09
tags:
  - modelo
  - knn
  - baseline
status: draft
---

# KNN

Modelo baseline de referencia.

## Resultados

| Métrica | Calor | Frío |
|---------|-------|------|
| cv_score | 0.8922 | 0.9258 |

## Hiperparámetros

- `n_neighbors`, `leaf_size`, `metric`, `algorithm`
- Optimización vía GridSearchCV (búsqueda del mejor k)
- Mejores params guardados en `models/artifacts/best_params_KNN.joblib`

## Tracking

MLflow: `climasafeai_KNN` (v1-v41 + versiones _calor/_frio)

## Ver también

- [[XGBoost_calor]]
- [[RandomForest_frio]]
- [[01_PROYECTO/modelos]]