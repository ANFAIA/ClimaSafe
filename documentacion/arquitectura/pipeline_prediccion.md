# Pipeline de predicción

Flujo completo desde que el usuario envía su perfil hasta que recibe la clase de riesgo.

```
Usuario → POST /api/predict → predict_ensemble() → resultado JSON
```

## 1. Entrada

`POST /api/predict` recibe `{ provincia, perfil, lat?, lon?, date? }`.

El perfil pasa por `_normalize_perfil()` que:
- Convierte claves legacy (ej. `actividad` → `nivel_actividad`)
- Elimina mapeos obsoletos
- Añade `_perfil_horario` si hay HI horario disponible

## 2. Ensemble (`predict_ensemble`)

### 2.1 Clima
`fetch_weather_data()` obtiene datos de Open-Meteo (archive + forecast). Devuelve:
- `current`: condiciones actuales (temp, HI, WC, UV, viento)
- `df_hora`: serie horaria con HI, WBGT, WindChill
- `df_features`: vector diario (27 features) para modelos tabulares

### 2.2 Modelos ML
Se ejecutan 4 modelos independientes:

| Modelo | Tipo | Path |
|--------|------|------|
| XGBoost_calor | Gradient boosting (calor) | `models/XGBoost_calor.joblib` |
| RandomForest_frio | Random forest (frío) | `models/RandomForest_frio.joblib` |
| LSTM híbrida | LSTM multi-tarea (24h seq + 27 features) | `models/lstm_province_hybrid.pt` |
| Formula | Determinista (HI, WC, WBGT) | En código |

Cada modelo devuelve `{ prob_riesgo, clase_threshold, _X? }`.

### 2.3 Override por HI
Si `HI_peak >= 32` o `UV > 3`, fuerza PRECAUCION aunque ML diga SEGURO.
Si `HI_peak >= 39`, fuerza PELIGRO.
Si `HI < 27` y `WC > 0` y `UV < 6`, puede bajar PELIGRO a PRECAUCION.

### 2.4 Personalización
`personalizar_riesgo()` modula la probabilidad poblacional con factores individuales:
- Multiplica odds por producto de factores
- Capped a ×3.0 (`CAP_FACTORES_DEFECTO`)
- Factores distintos para calor vs frío

### 2.5 Clase final
Toma `max(prob_pers_calor, prob_pers_frio)` y aplica:

| Clase | Threshold |
|-------|-----------|
| SEGURO | < 0.25 |
| PRECAUCION | ≥ 0.25 |
| PELIGRO | ≥ 0.55 |

Si hay override HI, prevalece sobre la clase personalizada.

## 3. Salida

```json
{
  "weather": { "current": {...}, "uv_index": ..., "provincia": "..." },
  "modelos": { "XGBoost_calor": {...}, "RandomForest_frio": {...}, "LSTM": {...}, "Formula": {...} },
  "perfil": { "calor": { "prob_poblacional": ..., "prob_personalizada": ..., "factores": [...] }, "frio": {...} },
  "perfil_usuario": { ... },
  "clase_final": 0|1|2,
  "clase_final_label": "SEGURO|PRECAUCION|PELIGRO",
  "explicacion": { "shap": [...], "modelo_determinante": "..." },
  "recomendaciones": ["..."],
  "override_fisico": {...} | null,
  "perfil_id": 123
}
```

## Archivos clave

| Archivo | Rol |
|---------|-----|
| `chat/app.py` | Endpoint + normalización de perfil |
| `climasafeai/models/ensemble.py` | Orquestador de modelos |
| `climasafeai/models/predict_model.py` | Thresholds + clases |
| `climasafeai/features/personalizacion.py` | Factores individuales |
| `climasafeai/features/build_features.py` | Feature engineering |
| `climasafeai/data/weather_fetcher.py` | Datos meteorológicos |
| `climasafeai/models/explicabilidad.py` | Explicaciones SHAP + contrafactuales |

Ver también: `documentacion/ml/contrafactuales.md` (generación de contrafactuales), `documentacion/riesgo/personalizacion_individual.md` (coeficientes de factores)
