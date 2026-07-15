---
type: observacion
created: 2026-07-14
tags:
  - observacion
  - pipeline
  - main
  - lstm
status: active
---

# Actualización del pipeline principal

## Cambios en `main.py`

Reescritura completa del `run_full_pipeline()`:

**Antes:** pipeline genérico (EDA, Optuna, SHAP, PCA) sobre un único `dataset.csv` con `target` binario — obsoleto para el proyecto real.

**Ahora:** pipeline real de ClimaSafeAI con 9 pasos:

1. Descarga de datos crudos (MoMo + ERA5) — skip si existen
2. Preprocesado → parquets etiquetados — skip si existen y tienen columnas nuevas
3. Secuencias LSTM 24h (`secuencias_24h.npz`) — skip si existe
4. Preprocesado ML (train/test split por fecha + scaler) para calor y frío
5. Entrenamiento inline: XGBoost (calor, 27 features) + RandomForest (frío, 19 features)
6. Entrenamiento: LSTM province_hybrid (24h + embedding provincia + INE + 31 daily) via `train_lstm_province_hybrid()`
7. Evaluación tabulares dual (argmax + umbrales calibrados lado a lado)
8. Evaluación LSTM province_hybrid via `evaluate_lstm_province_hybrid()`
9. Recalibración de thresholds globales para todos los modelos
10. Tabla comparativa final en `reports/comparativa_final.csv`

Cada paso verifica archivos existentes antes de ejecutar.

## Modelos entrenados

| Modelo | Archivo | Features |
|--------|---------|----------|
| XGBoost Calor | `models/XGBoost_calor.joblib` | 27 |
| RandomForest Frío | `models/RandomForest_frio.joblib` | 23 (con nocturnas + rachas) |
| LSTM province_hybrid | `models/LSTM_province_hybrid_seed42.pt` | 24h seq + emb prov + INE + 31 daily |

## Cambios colaterales (2026-07-14)

- Features nocturnas (`t2m_min_noche`) y rachas severas (`horas_wc_severo`) para frío en `make_dataset.py`
- Selección por clase en `build_features.py` (27v23 vía `FRIO_EXTRA_COLS`/`COLS_TO_DROP_BY_CLASE`)
- Umbrales recalibrados: calor t1=0.40/t2=0.35, frío t1=0.45/t2=0.40
- `tuning/calibrar_umbrales.py` — script de grid search para recalibración
- `DAILY_FEATURE_COLS` en `lstm_hybrid.py` extendido de 27 a 31 features
- `experimento_label_sin_fuga.py` eliminado
- Templates `00_META/templates/` rellenados; archivos vacíos eliminados

## Actualización 2026-07-15

- LSTM province_hybrid reemplazó a LSTM híbrida en `main.py`
- `peso_riesgo_extra=8.0` — multiplicador para clases de riesgo en pérdida
- Thresholds LSTM separados en `CLASS_THRESHOLDS_LSTM` (calor 0.60/0.55, frío 0.40/0.35)
- Rec_riesgo final: **Calor 0.7367**, **Frío 0.7082** (supera baseline +0.013/+0.084)
- Ver [[2026-07-15_optimizacion_lstm]]

## Pendiente para siguiente iteración

- [ ] Servir predicciones vía FastAPI
- [ ] Dockerizar el pipeline

## Ver también

- [[01_PROYECTO/arquitectura]]
- [[01_PROYECTO/modelos]]
- [[03_MODELOS/LSTM]]
