# Modelos y pesos

## Los cuatro modelos

| Modelo | Rol | Rec_riesgo (umbrales calibrados) |
|---|---|---|
| **XGBoost** | Modelo tabular de **calor** (pesos balanceados) | **0.668** |
| **RandomForest** | Modelo tabular de **frío** (`class_weight="balanced"`) | **0.612** |
| **LSTM province_hybrid** | LSTM + embedding de provincia + INE + features diarias, tarea multi-tarea calor/frío | **0.737** calor / **0.708** frío |
| **Fórmula determinista** | Índices térmicos clásicos (Heat Index, Wind Chill, WBGT) sobre datos actuales | — |

La métrica de selección es **Rec_riesgo** (recall medio de las clases de riesgo
1 y 2), **no** accuracy: es un sistema de aviso y vale más sobre-avisar que
perderse un día de riesgo.

## Cómo se combinan (ensemble conformal-weighted)

Cada modelo no vota igual: pesa según la **confianza conformal** de su
predicción.

- XGBoost y RandomForest llevan un **prediction set conformal**
  (`SplitConformalCalibrator`, alpha=0.1): el set tiene tamaño 1 (confianza
  alta), 2 (media) o 3 (baja), y el modelo pesa **`1 / set_size`** en la media.
- El LSTM y la Fórmula no tienen conformal → pesan **`1/2`** cada uno.
- La probabilidad del ensemble es la media ponderada de las probabilidades de
  riesgo de los modelos disponibles (para calor: XGBoost + LSTM + Fórmula; para
  frío: RandomForest + LSTM + Fórmula).

## Umbrales y calibración

- **Calibración isotónica post-hoc en frío** (en calor reduce sensibilidad en
  provincias frías): las probabilidades del RandomForest_frío se recalibran
  antes de umbralizar.
- **Umbrales de clase** (`CLASS_THRESHOLDS_RECOMENDADOS`): calor
  `t1=0.25, t2=0.10`; frío `t1=0.21, t2=0.20` (sobre probabilidades ya
  calibradas).
- **`PERS_THRESHOLD_PELIGRO = 0.55`**: umbral de PELIGRO sobre la probabilidad
  **personalizada** (es `P(riesgo) = P(1)+P(2)`, por eso es más exigente).
- **Overrides físicos**: si el Heat Index ≥ 39 °C o el Wind Chill ≤ −25 °C, la
  clase sube a PELIGRO con independencia del ML; hay overrides intermedios por
  vulnerabilidad y por UV.

## Factores individuales y sus pesos

La probabilidad poblacional se personaliza con factores multiplicativos
(edad, comorbilidades, medicación, actividad, aclimatación, situación social,
ocupación…). Ver la página **[Riesgo y personalización](riesgo-personalizacion.md)**:
allí están las tablas completas de factores con su peso y su fuente, la
composición en odds y el cap a ×3.0.

## Detalle técnico

La implementación vive en el repositorio:

- `climasafeai/models/ensemble.py` — ensemble conformal-weighted, overrides.
- `climasafeai/models/predict_model.py` — umbrales de clase.
- `climasafeai/models/conformal.py` — SplitConformalCalibrator.
- `climasafeai/models/calibrate.py` — isotónica en frío.
- `climasafeai/features/personalizacion.py` — factores individuales.
- `data/factores_riesgo.json` — base versionada de factores (coef, DOI, calidad).
