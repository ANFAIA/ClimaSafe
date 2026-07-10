---
type: modelo
created: 2026-07-09
tags:
  - modelo
  - optuna
  - tuning
status: draft
---

# Hiperparámetros (Optuna)

Optimización de hiperparámetros con Optuna + GridSearchCV.

## Búsquedas realizadas

- RandomForest: `n_estimators`, `max_depth`, `min_samples_split`, `min_samples_leaf`, `bootstrap`, `class_weight`
- XGBoost: `n_estimators`, `max_depth`, `learning_rate`, `colsample_bytree`, `reg_alpha`, `reg_lambda`, `eval_metric`
- KNN: GridSearchCV sobre `n_neighbors` (1-50)

## Artefactos guardados

- `models/artifacts/best_params_KNN.joblib`
- `reports/tuning_results.csv`
- `tuning/` — scripts de optimización

## Mejores cv_score por algoritmo

| Algoritmo | Calor | Frío |
|-----------|-------|------|
| RandomForest | 0.8424 | 0.8039 |
| XGBoost | 0.9028 | 0.9271 |
| KNN | 0.8922 | 0.9258 |

## Ver también

- [[MLflow]]
- [[XGBoost_calor]]
- [[01_PROYECTO/modelos]]