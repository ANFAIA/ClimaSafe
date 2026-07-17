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

## Sistema ensemble (2026-07-16)

Las 4 estimaciones se combinan en `climasafeai/models/ensemble.py` con el criterio **más restrictivo**:

| Estimación | Archivo | Captura |
|---|---|---|
| XGBoost (calor) | `models/XGBoost_calor.joblib` | Riesgo poblacional calor (MoMo) |
| RandomForest (frío) | `models/RandomForest_frio.joblib` | Riesgo poblacional frío (MoMo) |
| LSTM province_hybrid | `models/LSTM_province_hybrid_seed42.pt` | Correlación temporal HI/WC ↔ mortalidad |
| Fórmula determinista | `formulas_riesgo_deterministico.md` | Riesgo individual (NWS/OMS) |

### Guardarraíl físico
- Ya NO anula completamente. PELIGRO(2)→PRECAUCION(1), PRECAUCION(1) se mantiene.
- Solo se activa cuando peak HI<27°C AND WC>0°C AND UV<6.

### Override (degradación)
- Degrada solo 1 nivel (2→1), nunca toca PRECAUCION(1).
- Menciona el HI pico durante la ventana de actividad, no solo condiciones actuales.

### Perfil horario
- `weather_fetcher.py` calcula HI/WC para cada hora en la ventana de actividad del usuario.
- El ensemble usa el pico de la ventana para la decisión final.

## Explicabilidad (2026-07-16)

Módulo `climasafeai/models/explicabilidad.py`:

- **SHAP**: `_FEATURE_NAME_MAP` con 31 entradas traduce nombres técnicos (ej. `heat_index_c_roll7` → "Media HI últimos 7 días").
- **Explicaciones ML filtradas**: solo se muestran si `prob_riesgo >= 0.35`.
- **Explicación fórmula**: `explicar_formula()` usa HI de ventana de actividad cuando existe, genera explicación tipo "Condiciones actuales seguras (HI=X°C), pero durante la actividad prevista (Y:00-Z:00) se esperan HI entre A y B°C".
- **Cabeceras**: "Riesgo por tendencia meteorológica" para XGBoost/RF.

## Recomendaciones (2026-07-16)

Módulo `climasafeai/models/recomendaciones.py`:

- `_riesgo_dominante()` compara heat_clases vs cold_clases para determinar si el riesgo es calor, frío o ambos.
- `_clasificar_clima()` filtra recomendaciones por tags según riesgo dominante.
- Señales de alarma separadas por tipo de riesgo.
- Catálogo en `data/recomendaciones.json`.

## LSTM province_hybrid (producción desde 2026-07-16)

- Arquitectura: `LSTMProvinceHybridMultiTask` — LSTM + embedding provincia (16) + INE (4) + daily (31) → fusión(128) → 2 cabezas
- `peso_riesgo_extra=8.0` optimizado para Rec_riesgo
- Rec_riesgo: **Calor 0.7316**, **Frío 0.6735** (argmax); **0.7367/0.7082** (th óptimos)
- Checkpoint: `models/LSTM_province_hybrid_seed42.pt`
- Thresholds propios en `CLASS_THRESHOLDS_LSTM` (calor 0.60/0.55, frío 0.40/0.35)
- `lstm_hybrid.py` + `lstm_province.py` fusionados → `lstm_province_hybrid.py`

### LSTM híbrida (reemplazada)
- Arquitectura: `LSTMHybridMultiTask` — tronco LSTM + 31 features diarias, sin embedding provincia
- Checkpoint: `models/LSTM_hybrid.pt`

### LSTM multi-tarea (base — legado)
- Arquitectura: `LSTMMultiTask` — tronco LSTM + 2 cabezas, sin features diarias
- Checkpoint: `models/LSTM_multitask.pt`

> [!note] Umbrales calibrados vs argmax
> Los umbrales en cascada mejoran significativamente Rec_riesgo respecto a argmax: calor +0.035, frío +0.095.

## Experimentos

- Ablación features 27v19: las 8 de persistencia avanzada ayudan a calor (+0.007) pero dañan a frío (-0.020). Ver `documentacion/ml/ablacion_features_27v19.md`.
- Features nocturnas y rachas severas para frío (+0.026). Ver `tuning/features_frio.py`.
- Grid search de umbrales en `tuning/calibrar_umbrales.py`.

## Modelos avanzados evaluados y descartados

| Modelo | Capa | Motivo descarte |
|---|---|---|
| **WeatherNext 2** (Google) | Meteorología | Pago por API, resolución 6h, mejora marginal para España |
| **TimesFM 2.5** (Google) | Predicción serie temporal | 200M params, GPU necesaria, propósito general |
| **Granite TTM-R3** (IBM) | Predicción serie temporal | Licencia CC-BY-NC-SA (no comercial) |
| **Prithvi-EO-2.0** (IBM/NASA) | Satélite | GPU, pipeline complejo, resolución excesiva para nivel provincial |

> [!success] Solo open-source gratuito
> Todo el stack corre en CPU, sin API keys obligatorias, sin dependencias de pago. Documentado en `documentacion/arquitectura/diseño_modelo.md#7`.

## Pendiente

- [ ] Monitoring de derivas en producción
- [ ] FastAPI para servir predicciones
- [ ] Docker + docker-compose

## Ver también

- [[arquitectura]]
- [[02_DATOS/features]]
- [[03_MODELOS/LSTM]]
- [[03_MODELOS/MLflow]]