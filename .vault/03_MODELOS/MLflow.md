---
type: modelo
created: 2026-07-09
tags:
  - modelo
  - mlflow
  - experimentos
  - tracking
status: active
---

# MLflow — Experimentos

## Resumen

219 runs en 2 experimentos, 3 familias de modelos registradas.

## Experimentos

| ID | Nombre | Runs |
|----|--------|------|
| 0 | Default | 0 |
| 1 | **climasafeai** | **219** |

## Tipos de Run

| Nombre | Propósito |
|--------|-----------|
| `RandomForest` | Entrenamiento con cv_score |
| `XGBoost` | Entrenamiento con cv_score |
| `KNN` | Entrenamiento con cv_score |
| `RandomForest_eval` | Evaluación en test |
| `XGBoost_eval` | Evaluación en test |
| `KNN_eval` | Evaluación en test |

## Métricas registradas

`cv_score`, `acc_train`, `acc_test`, `f1_train`, `f1_test`, `prec_test`, `rec_test`, `rec_riesgo`, `f1_macro`, `roc_auc`

## Model Registry (~132 versiones)

| Familia | Versiones |
|---------|-----------|
| `climasafeai_RandomForest` | v1-v41 + _calor v1-v2 + _frio v1-v2 |
| `climasafeai_XGBoost` | v1-v41 + _calor v1-v2 + _frio v1-v2 |
| `climasafeai_KNN` | v1-v41 + _calor v1-v2 + _frio v1-v2 |

## Origen

Notebooks `0-2-Calor-Ejecucion.ipynb` y `0-2-Frio-Ejecucion.ipynb`.

## Ver también

- [[XGBoost_calor]] — mejor cv_score
- [[01_PROYECTO/modelos]]