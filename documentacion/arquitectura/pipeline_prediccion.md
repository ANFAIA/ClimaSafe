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

Si el perfil incluye `alias`, se busca o crea en DB (`buscar_por_alias`).

## 2. Ensemble (`predict_ensemble`)

### 2.1 Clima
`fetch_weather_data()` obtiene datos de Open-Meteo (archive + forecast). Devuelve:
- `current`: condiciones actuales (temp, HI, WC, UV, viento, rh)
- `df_hora`: serie horaria con HI, WBGT, WindChill
- `df_features`: vector diario (27 features) para modelos tabulares

### 2.2 Modelos ML
Se ejecutan 4 modelos independientes:

| Modelo | Tipo | Path | Confianza conformal |
|--------|------|------|---------------------|
| XGBoost_calor | Gradient boosting (calor) | `models/XGBoost_calor.joblib` | Sí (set size) |
| RandomForest_frio | Random forest (frío) | `models/RandomForest_frio.joblib` | Sí (set size) |
| LSTM híbrida | LSTM multi-tarea (24h seq + 27 features) | `models/lstm_province_hybrid.pt` | No |
| Formula | Determinista (HI, WC, WBGT) | En código | No |

Cada modelo devuelve `{ prob_riesgo, clase_threshold, conformal_confianza, _X? }`.

La **confianza conformal** (split conformal, α=0.1) clasifica cada predicción en:
- `alta`: set size = 1 (modelo seguro de la clase)
- `media`: set size = 2 (modelo duda entre dos clases)
- `baja`: set size = 3 (máxima incertidumbre)
- `desconocida`: no disponible

Se muestra en la UI como círculos de color (verde/amarillo/rojo).

### 2.3 Personalización

`personalizar_riesgo()` modula la probabilidad poblacional de cada modelo
(XGBoost calor, RF frío) con factores individuales del perfil:

- **Sexo**: calibración centrada (calor: hombre ×0.96, mujer ×1.04; frío: hombre ×1.15, mujer ×0.87)
- **Edad**: 65a×1.2, 75a×1.5, 85a×2.0 (calor y frío con tablas distintas)
- **Actividad**: según intensidad (×1.0–×2.0 calor, ×0.9–×1.2 frío)
- **Entrenado**: reduce a la mitad el exceso del factor actividad si ≥moderada
- **Grasa corporal**: continua 0.85–1.15 según desviación de la media del grupo edad+sexo (curvas CUN-BAE/ENPE)
- **Comorbilidades**: cardiovascular, diabetes, respiratoria, mental
- **Fármacos**: diuréticos de asa, antipsicóticos
- **Situación social**: vive_solo, no_sale, encamado, vivienda_fría, alcohol
- **Fiesta**: factor ×1.8 independiente (no se mezcla con situacional)
- **Ocupación**: oficina(1.0), reparto(1.35), mantenimiento(1.7), construcción(2.2), campo(2.7)
- **Aclimatación**: no_aclimatado ×1.6
- **Falta de sueño**: ×1.2
- **UV + fototipo**: factor según índice UV y fototipo Fitzpatrick
- **Fatiga acumulada**: si actividad ≥moderada, duración≥umbral y HI≥27°C
- **Hora del día**: solapamiento con ventana 12-18h (calor) o 4-8h (frío)
- **Viento en frío**: factor extra para actividad intensa con viento >10 km/h

Composición en **odds** (no multiplicación directa) para no salirse de [0,1].
Producto de factores capado a **×3.0** (`CAP_FACTORES_DEFECTO`). Cuando se
choca el cap, el flag `capado=true` se propaga al sistema de overrides.

El resultado es la **probabilidad personalizada** (`prob_pers`), separada en
calor y frío. La clase personalizada se calcula con:

```
clase_pers = max(prob_pers_calor, prob_pers_frio) → thresholds
```

| Clase | Threshold |
|-------|-----------|
| SEGURO | < 0.25 |
| PRECAUCION | ≥ 0.25 |
| PELIGRO | ≥ 0.55 |

### 2.4 Safety overrides físicos

Se aplican **después** de personalización. Solo sobreescriben `clase_pers`
si las condiciones lo requieren. Esto evita que un perfil joven/adaptado
reciba una alarma solo por HI alto, pero garantiza que un perfil vulnerable
esté protegido aunque el ML subestime su riesgo.

#### 2.4.1 Calor (HI)

| Condición | Acción | ¿Unconditional? |
|-----------|--------|----------------|
| HI ≥ 39°C | PELIGRO | Sí |
| HI ≥ 27°C + vulnerable + (HI ≥ 32°C o UV > 3) | PRECAUCION | Solo si vulnerable |

**Vulnerabilidad calor**: se cumple si el perfil tiene **al menos uno** de:
- Comorbilidades
- Fármacos de riesgo
- Edad ≥ 60 **y no** está (entrenado + aclimatado)
- `aclimatado == false`
- `factor_total > 1.8`
- `capado == true` (chocó el ×3.0)

#### 2.4.2 Frío (Wind Chill)

| Condición | Acción | ¿Unconditional? |
|-----------|--------|----------------|
| WC ≤ -25°C | PELIGRO | Sí |
| WC ≤ -10°C + vulnerable | PRECAUCION | Solo si vulnerable |

**Vulnerabilidad frío**: se cumple si el perfil tiene **al menos uno** de:
- Comorbilidades
- Edad ≥ 60 **y no** entrenado
- `situacion_social` incluye `vive_solo`, `no_sale` o `vivienda_fria`
- `factor_total > 1.8`
- `capado == true`

#### 2.4.3 Downgrade por ausencia de calor real

Si `HI < 27°C`, `WC > 0°C` y `UV < 6`, pero la clase personalizada es ≥ 1
(porque los modelos ML detectan tendencia de riesgo), se **reduce** PELIGRO
a PRECAUCION en lugar de mostrar una alarma que no corresponde al clima actual:

```
clase_pers == 2  →  PRECAUCION (con razón "sin calor actual")
clase_pers == 1  →  se mantiene como PRECAUCION (con nota aclaratoria)
```

### 2.5 Clase final

```
if override_fisico:
    clase_final = override_fisico["clase_final"]
else:
    clase_final = clase_pers
```

El override siempre prevalece. La explicación incluye `modelo_determinante`
que puede ser:
- `"Override — ..."`: cuando un override físico decidió
- `"Personalización (subió/bajó de X a Y)"`: cuando la personalización cambió
  la clase respecto al voto mayoritario de los modelos ML
- El nombre del modelo ML que más votó (por defecto)

### 2.6 Diagnóstico bayesiano (post-procesado)

Si los modelos están disponibles, se ejecuta un `BayesianRiskDiagnosis` que:
- Calcula la **contribución inversa** de temperatura, grasa corporal y edad
  al riesgo final (qué factores explican más la clase actual)
- Genera **contrafactuales**: "si tu temperatura fuera 5°C menor, tu riesgo
  bajaría de PELIGRO a PRECAUCION"
- Se muestra en la UI como sección "¿Por qué este riesgo?" con sliders

### 2.7 Recomendaciones

`generar_recomendaciones()` produce consejos contextuales:
- **Time-aware**: si la actividad es nocturna, omite "busca sombra en horas centrales"
- **Sport-aware**: usa el nombre del deporte en lugar de "actividad"
- **Fiesta-aware**: recomendación específica sobre alcohol+deshidratación
- **Falta-sueño-aware**: aviso de fatiga+tolerancia al calor
- **Duración**: pausas y agua según horas de actividad

## 3. Salida

```json
{
  "weather": {
    "current": { "t2m_c": 28.0, "rh": 65, "wind_speed_kmh": 12, ... },
    "uv_index": 7.9,
    "provincia": "Pontevedra",
    "lat": 42.28,
    "lon": -8.72,
    "df_hora": [ ... ],
    "df_features": { ... }
  },
  "modelos": {
    "XGBoost_calor": { "prob_riesgo": 0.12, "clase_threshold": 0, "conformal_confianza": "alta", ... },
    "RandomForest_frio": { "prob_riesgo": 0.03, "clase_threshold": 0, "conformal_confianza": "alta", ... },
    "LSTM": { "calor": { ... }, "frio": { ... }, "conformal_confianza": null },
    "Formula": { "calor": { "clase": 1, "heat_index_c": 28.5, ... }, "frio": { "clase": 0, "wind_chill_c": 25.2 } }
  },
  "perfil": {
    "calor": {
      "prob_poblacional": 0.035,
      "factor_total": 1.204,
      "producto_bruto": 1.204,
      "capado": false,
      "prob_personalizada": 0.0418,
      "factores": [
        { "nombre": "sexo hombre", "categoria": "fisiologico", "factor": 0.96 },
        { "nombre": "senderismo (actividad ligera)", "categoria": "fisiologico", "factor": 1.1 },
        { "nombre": "UV 7.9 + fototipo 2", "categoria": "fisiologico", "factor": 1.14 }
      ]
    },
    "frio": { ... }
  },
  "perfil_usuario": { ... },
  "clase_final": 0,
  "clase_final_label": "SEGURO",
  "explicacion": {
    "modelo_determinante": "XGBoost_calor",
    "shap": [ ... ],
    "Formula": { "detalle": "..." },
    "Bayes": {
      "diagnostico_inverso": { "temperatura": 0.42, "grasa": 0.18, "edad": 0.40 },
      "contrafactuales": [
        { "variable": "temperatura", "delta": -5, "efecto": "baja de PELIGRO a PRECAUCION" },
        { "variable": "edad", "delta": -10, "efecto": "baja de PRECAUCION a SEGURO" }
      ]
    }
  },
  "recomendaciones": [
    "Hidratación: bebe agua cada 20 minutos...",
    "Tu actividad (20:00-22:00) es fuera del pico de calor (12:00-18:00)."
  ],
  "override_fisico": null,
  "perfil_id": 42
}
```

## Archivos clave

| Archivo | Rol |
|---------|-----|
| `chat/app.py` | Endpoint + normalización de perfil + alias DB |
| `climasafeai/models/ensemble.py` | Orquestador de modelos + overrides |
| `climasafeai/models/predict_model.py` | Thresholds + clases |
| `climasafeai/features/personalizacion.py` | Factores individuales |
| `climasafeai/features/build_features.py` | Feature engineering |
| `climasafeai/data/weather_fetcher.py` | Datos meteorológicos |
| `climasafeai/models/explicabilidad.py` | Explicaciones SHAP + bayes + contrafactuales |
| `climasafeai/models/recomendaciones.py` | Recomendaciones contextuales |
| `climasafeai/models/conformal.py` | Conformal prediction |
| `climasafeai/models/bayes.py` | Diagnóstico bayesiano inverso |

Ver también: `documentacion/ml/contrafactuales.md` (generación de contrafactuales),
`documentacion/riesgo/personalizacion_individual.md` (coeficientes de factores),
`documentacion/riesgo/formulas_deterministas.md` (HI, WC, UV),
`documentacion/conformal_prediction.md` (split conformal).
