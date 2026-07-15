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
- Features: 27 features completas (grupos A-D: meteorológicas, intradía, persistencia)
- Umbral calibrado: t1=0.40, t2=0.35 → Rec_riesgo 0.668
- Guardado en `models/XGBoost_calor.joblib`

### RandomForest — Riesgo por Frío
- Seleccionado por recall en clase de riesgo
- Features: 19 features base (sin persistencia avanzada que dañaba -0.020)
- Nocturnas (`t2m_min_noche_lag1/roll7`) y rachas severas (`dias_consec_wc_severo`, `horas_wc_severo_sum14`) añadidas en 2026-07-14 (+0.026)
- Umbral calibrado: t1=0.45, t2=0.40 → Rec_riesgo 0.612
- Guardado en `models/RandomForest_frio.joblib`

## Completado

### LSTM province_hybrid (en pipeline desde 2026-07-15)
- Arquitectura: `LSTMProvinceHybridMultiTask` — LSTM + embedding provincia (16) + INE (4) + daily (31) → fusión(128) → 2 cabezas
- `peso_riesgo_extra=8.0` optimizado para Rec_riesgo
- Rec_riesgo: **Calor 0.7316**, **Frío 0.6735** (argmax); **0.7367/0.7082** (th óptimos)
- Checkpoint: `models/LSTM_province_hybrid_seed42.pt`
- Thresholds propios en `CLASS_THRESHOLDS_LSTM` (calor 0.60/0.55, frío 0.40/0.35)
- Reemplazó a la LSTM híbrida en `main.py`

### LSTM híbrida (reemplazada)
- Arquitectura: `LSTMHybridMultiTask` — tronco LSTM + 31 features diarias, sin embedding provincia
- Predecesora de la province_hybrid. Checkpoint: `models/LSTM_hybrid.pt`

### LSTM multi-tarea (base — legado)
- Arquitectura: `LSTMMultiTask` — tronco LSTM + 2 cabezas, sin features diarias
- Rendía por debajo de los modelos tabulares
- Checkpoint: `models/LSTM_multitask.pt`

> [!note] Umbrales calibrados vs argmax
> Los umbrales en cascada mejoran significativamente Rec_riesgo (métrica principal) respecto a argmax: calor +0.035, frío +0.095. La evaluación en `main.py` muestra ambos para comparación.

## Experimentos

- Ablación features 27v19: las 8 de persistencia avanzada ayudan a calor (+0.007) pero dañan a frío (-0.020). Ver `documentacion/ablacion_features_27v19.md`.
- Features nocturnas y rachas severas para frío (+0.026). Ver `tuning/features_frio.py`.
- Grid search de umbrales en `tuning/calibrar_umbrales.py`.
- Notebooks exploratorios en `notebooks/` (legado — el pipeline ahora es `main.py`).

## Pendiente

- [ ] Monitoring de derivas en producción
- [ ] FastAPI para servir predicciones
- [ ] Docker + docker-compose
- [ ] Re-evaluar ensemble tabular + LSTM híbrida con umbrales calibrados

## Ver también

- [[arquitectura]]
- [[02_DATOS/features]]
- [[03_MODELOS/LSTM]]
- [[03_MODELOS/MLflow]]