---
type: observacion
created: 2026-07-15
tags:
  - observacion
  - lstm
  - optimizacion
  - ensemble
status: active
---

# Optimización LSTM province_hybrid

## Problema
Maximizar `Rec_riesgo` de la LSTM province_hybrid integrada en `main.py`. Baseline: Calor 0.7233, Frío 0.6239.

## Experimentos fallidos

### LR scheduler (ReduceLROnPlateau) + weight_decay=1e-5
- Calor 0.7207, Frío 0.5428 — weight_decay sobre-regulariza frío

### LR scheduler solo
- Calor 0.7156, Frío 0.5443 — sin weight_decay sigue empeorando

### Ensemble de 3 seeds (42, 43, 44)
- Logit avg: Calor 0.7103, Frío 0.5981
- Prob avg: Calor 0.7103, Frío 0.5961
- Weighted logit (0.8/0/0.2): Calor 0.7207, Frío 0.5959
- Majority vote: Calor 0.7041, Frío 0.5784
- **Ninguna estrategia de ensemble supera el baseline**

## Solución: peso_riesgo_extra=8.0

Aumentar `peso_riesgo_extra` de 3.0 a 8.0 — el multiplicador de pesos para clases de riesgo (1 y 2) en la pérdida CrossEntropy.

Resultados comparativos (seed=42):

| Config | Calor | Frío |
|--------|-------|------|
| peso=3.0 (baseline) | 0.7191 | 0.5874 |
| peso=5.0 | 0.7286 | 0.6471 |
| **peso=8.0** | **0.7306** | **0.6757** |
| peso=3.0, lr=5e-4 | 0.7210 | 0.6061 |
| peso=5.0, lr=5e-4 | 0.7256 | 0.6579 |

### Resultado final en pipeline completo
```json
{
  "calor_argmax": 0.7316,
  "frio_argmax": 0.6735,
  "calor_th_optimo": 0.7367,
  "frio_th_optimo": 0.7082
}
```

## Cambios aplicados

- `main.py` → `peso_riesgo_extra=8.0`
- `predict_model.py` → nuevo `CLASS_THRESHOLDS_LSTM` (calor 0.60/0.55, frío 0.40/0.35) separado de `CLASS_THRESHOLDS_RECOMENDADOS` (que sigue para XGBoost/RF)
- `lstm_province_hybrid.py` → parámetro `seed` en `train_lstm_province_hybrid()`, más `EnsembleLSTM` y `load_by_seed()` disponible

## Ver también
- [[01_PROYECTO/modelos]]
- [[03_MODELOS/LSTM]]
- [[2026-07-14_actualizacion_pipeline]]
