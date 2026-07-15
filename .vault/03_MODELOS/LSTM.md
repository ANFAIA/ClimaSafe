---
type: modelo
created: 2026-07-14
tags:
  - modelo
  - lstm
  - secuencias
  - multitask
status: active
---

# LSTM multi-tarea

Cuarta estimación del sistema: una única red con tronco LSTM compartido y dos cabezas de 3 clases (calor y frío) entrenada sobre secuencias de 24 horas crudas de ERA5.

Aprende la correlación empírica española sin heredar la calibración americana de las fórmulas Heat Index / Wind Chill.

## Arquitectura base

```
Input: (batch, 24, 5)
  │
  └── LSTM (2 capas, hidden=64, dropout=0.3)
        │
        └── h_T (último estado oculto, capa superior)
              │
              ├── head_calor: Linear(64 → 3) → logits calor
              │
              └── head_frio: Linear(64 → 3) → logits frío
```

- `n_features=5`: t2m_c, rh, wind_speed_kmh, heat_index_c, wind_chill_c
- Pérdida conjunta: CE(calor, pesos balanced) + CE(frío, pesos balanced)
- Early stopping: paciencia 5 épocas sobre val_loss conjunta
- Optimizer: Adam (lr=1e-3, batch_size=256)

## Pipeline de datos

```
data/raw/era5/*.nc → filtrar 5pt/provincia → procesar_era5_a_horario()
→ construir_secuencias_24h() → alinear con labels MoMo
→ guardar en data/processed/secuencias_24h.npz
```

Implementado en `climasafeai/data/sequences.py`.

## Split temporal

| Set | % | Criterio |
|-----|---|----------|
| Train | 70% | Fechas más antiguas |
| Val | 10% | Siguientes (early stopping) |
| Test | 20% | Últimas fechas |

Nunca aleatorio: un split aleatorio mezclaría días de la misma ola de calor entre train y test.

## Rendimiento (base)

| Cabeza | Acc_test | F1_macro | Rec_riesgo |
|--------|----------|----------|------------|
| Calor | ~0.49 | ~0.28 | ~0.26 |
| Frío | ~0.47 | ~0.25 | ~0.22 |

> La LSTM base rinde por debajo de [[XGBoost_calor]] y [[RandomForest_frio]] porque opera sobre secuencias horarias crudas sin el feature engineering exhaustivo (27 features diarias con lags/medias móviles/grados-día). Su fortaleza potencial está en capturar patrones intradía que el feature engineering manual no cubre.

## LSTM híbrida en pipeline (`climasafeai/models/lstm_hybrid.py`)

Desde 2026-07-14, `main.py` entrena la **LSTM híbrida** (no la base): concatena el último estado oculto h_T de la LSTM con un vector de 31 features diarias (27 clásicas + 4 nocturnas/rachas severas) antes de las cabezas de clasificación.

Arquitectura:
```
Input secuencia: (batch, 24, 5) ──→ LSTM (2 capas, 64 hidden)
                                       │
                                       └── h_T ──⊕── features diarias (31) ──→ fusión(64) ──→ cabezas calor/frío
```

- Checkpoint: `models/LSTM_hybrid.pt`
- Scaler: `models/artifacts/scaler_diarias_lstm_hybrid.joblib`
- Features diarias vía `alinear_features_diarias()` desde `dataset_calor_labeled.parquet`
- Rendimiento actual: ver tabla en `main.py` paso 9.

## LSTM province_hybrid en pipeline (`climasafeai/models/lstm_province_hybrid.py`)

Desde 2026-07-15, `main.py` entrena la **LSTM province_hybrid** que reemplazó a la LSTM híbrida.

Arquitectura:
```
Input secuencia: (batch, 24, 5) ──→ LSTM (2 capas, 64 hidden)
                                       │
                                       └── h_T ⊕ emb(provinica, 16) ⊕ x_ine(4) ⊕ x_diarias(31)
                                             │
                                             └── fusión(128) ──→ cabezas calor/frío
```

- Embedding de provincia (dim=16) + 4 features INE + 31 features diarias
- `peso_riesgo_extra=8.0` — multiplicador de pesos para clases de riesgo
- Checkpoint: `models/LSTM_province_hybrid_seed42.pt`

### Rendimiento actual (2026-07-15)

| Métrica | Calor | Frío |
|---------|-------|------|
| Acc_test | 0.6954 | 0.7191 |
| F1_macro | 0.5226 | 0.4709 |
| **Rec_riesgo argmax** | **0.7316** | **0.6735** |
| **Rec_riesgo th óptimo** | **0.7367** | **0.7082** |

Thresholds LSTM específicos: calor t1=0.60/t2=0.55, frío t1=0.40/t2=0.35 (en `CLASS_THRESHOLDS_LSTM`).

### Experimentos

- LR scheduler + weight_decay empeoró ambos (over-regularización)
- Ensemble de 3 seeds no superó modelo individual
- Incrementar `peso_riesgo_extra` de 3.0 → 8.0 dio la mayor ganancia: +0.008 calor, +0.050 frío
- Ver [[2026-07-15_optimizacion_lstm]]

## Variantes descartadas

### LSTM con embedding de provincia (legacy)

- Concatena h_T + emb(provincia, dim=16)
- Rendía por debajo de la province_hybrid

### Gating y atención multi-head

- Fusión ponderada en vez de concatenación — peor
- HPO atención (4 configs): ninguna supera a concat simple

### Ensemble (tabular + LSTM)

- Evaluado pero ninguna estrategia de ensamble superó al mejor modelo individual

## Referencias

- `climasafeai/models/lstm_province_hybrid.py` — implementación activa
- `climasafeai/models/lstm_hybrid.py` — predecesora (reemplazada)
- `climasafeai/models/lstm_model.py` — implementación base
- `climasafeai/models/lstm_province.py` — variante con embedding (legacy)
- `climasafeai/data/sequences.py` — generación de secuencias
