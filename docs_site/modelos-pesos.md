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

## Modelo bayesiano jerárquico por provincia (BAYES-001) — contraste

Como **modelo de contraste** (no entra en el ensemble) se implementó una
regresión logística **ordinal acumulativa** de 3 clases con **efectos
aleatorios por provincia** (partial pooling): cada provincia tiene su propia
curva informada por la distribución nacional, así que las provincias con pocos
episodios se encogen hacia la media en vez de sobreajustar. Se muestrea con
Metropolis-Hastings propio (numpy/scipy, sin pymc) y se compara contra el
componente tabular del ensemble reentrenado en la misma partición temporal
(test = último 20 % de fechas, 34 470 filas):

| Métrica (test temporal) | Calor: XGBoost → Jerárquico | Frío: RandomForest → Jerárquico |
|---|---|---|
| F1 macro | 0.4850 → **0.5484** (+0.063) | 0.4562 → **0.4779** (+0.022) |
| Brier | 0.0571 → **0.0454** | 0.0944 → **0.0476** |
| Accuracy | 0.8815 → 0.8616 | 0.7758 → **0.8998** |

En calor la ganancia de F1 es **mayor donde menos datos hay** (provincias con
pocos episodios: +0.0919; con muchos: +0.0549). En frío el beneficio es de
**calibración**, no de ordenación de clases.

**Decisión:** el jerárquico **NO entra en el ensemble ni lo sustituye** — no
mejora consistentemente (en frío pierde en provincias con pocos episodios,
−0.035) y su intervalo de credibilidad al 90 % no es un intervalo predictivo
calibrado de la clase (cobertura 0.37 calor / 0.29 frío). Se queda como
contraste; la vía de entrada futura (recalibrar el intervalo o añadirlo al
ensemble como miembro con peso `1/conformal_set_size`) está documentada.

> Estudio completo (implementación, comparativa y limitaciones):
> [`documentacion/modelos/bayes_jerarquico.md`](https://github.com/ANFAIA/ClimaSafe/blob/main/documentacion/modelos/bayes_jerarquico.md)

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
