# Calibración de umbrales de decisión por clase

> Sistema de **aviso** de riesgo por temperatura (calor / frío), 3 clases por
> (provincia, día): `0=seguro`, `1=precaución`, `2=peligro`. Política del
> sistema: **mejor sobre-avisar que no avisar**. Métrica de selección
> `Rec_riesgo` = recall macro de las clases de riesgo (1 y 2).

## 1. El problema: argmax deja la precisión de riesgo por los suelos

Los modelos desplegados (`XGBoost_calor`, `RandomForest_frio`) predicen con
**argmax** de `predict_proba`. Con clases muy desbalanceadas y modelos
ponderados para recall, argmax dispara muchas **falsas alarmas**: la precisión
de las clases de riesgo se queda en ~0.16–0.38 (de cada 3–6 avisos, solo 1
acierta). El recall ya es razonable; lo caro es la precisión.

La idea de esta calibración es **mover el punto de operación sin reentrenar**:
en vez de argmax, exigir a las probabilidades un umbral por clase, y elegir el
umbral que da el mejor trade-off recall/precisión sobre una **validación
temporal interna** (nunca sobre test).

## 2. La regla de decisión: cascada por severidad

En vez de un umbral independiente por clase (que puede dejar filas sin decidir),
se usa una **cascada ordinal por severidad** — la extensión natural a 3 clases
del umbral binario que ya existía en `predict_model.DECISION_THRESHOLD`:

```
p2 = P(clase 2 = peligro)          t2 = umbral de peligro
p_riesgo = P(1) + P(2)             t1 = umbral de aviso

si   p2       >= t2   ->  2  (peligro)
sino si p_riesgo >= t1  ->  1  (precaución)
sino                  ->  0  (seguro)
```

- **Peligro** exige evidencia directa de la clase 2 (`p2 >= t2`).
- **Precaución** solo exige suficiente **masa de probabilidad de riesgo total**
  (`P(1)+P(2) >= t1`), no que la clase 1 sea la más probable — coherente con
  "mejor sobre-avisar". Un día con `P = [0.55, 0.25, 0.20]` es argmax=0 (seguro)
  pero riesgo total 0.45; con `t1=0.40` pasa a precaución.

Implementada en `apply_class_thresholds(proba, t1, t2)` (predict_model.py).
Con `t1=t2=0.5` **no** coincide exactamente con argmax, por eso el argmax se
mide aparte como línea base.

## 3. Metodología (sin mirar test ni una vez)

Todo el barrido y la elección de umbrales se hace sobre validación; test se toca
**solo al final**, y solo con los 2–3 puntos ya elegidos. Script:
`tuning/calibracion_umbrales.py`.

1. **Split temporal.** Se replica la regla de `preprocess_data`
   (`build_features.py`): tras `drop_duplicates`, el último **20 %** de fechas
   distintas es test. Como `X_train_{clase}.csv` no trae fecha, se reconstruye
   desde `dataset_{clase}_labeled.parquet` y se **verifica** que las etiquetas
   reconstruidas coinciden fila a fila con `y_train_{clase}.csv` (si no, aborta).
2. **Validación = último 15 % temporal de train** (desde 2023-02-18). El resto
   (hasta 2023-02-17) es el sub-train.
3. **Clon honesto.** El modelo desplegado se entrenó con **todo** el train, así
   que sus probabilidades sobre la validación serían *in-sample* y optimistas
   (medido: `Rec_riesgo` ~0.82 in-sample vs 0.63 real en calor) y los umbrales
   elegidos ahí no transferirían. Por eso se ajusta un **clon** con los mismos
   hiperparámetros y la misma receta de pesos que `train_model.train_models`
   (XGBoost → `sample_weight` balanceado) **solo con el sub-train**, y ese clon
   da las probabilidades honestas para *elegir* los umbrales. El modelo
   desplegado, sin tocar, es el que luego se evalúa en test con esos umbrales.
4. **Barrido** de `(t1, t2)` en rejilla 0.02–0.98 (paso 0.02) sobre validación;
   se traza la **frontera de Pareto** recall/precisión de las clases de riesgo.
5. **Evaluación única en test**: solo el argmax del modelo desplegado + los
   puntos elegidos.

### Guardas anti-degeneración (imprescindibles)

El agregado `Rec_riesgo` puede subirse con canjes tramposos que **no transfieren
a test**. Se descartaron con dos guardas, aplicadas a todo punto candidato:

- **Peligro intocable**: `rec(clase 2) >= rec(clase 2) del argmax`. Es la
  prioridad del sistema y, además, impide el sobreajuste a validación: sin ella
  el barrido premiaba esquinas de la rejilla (`t1≈0.05`: casi todo pasa a
  precaución) que subían `Rec_riesgo` en validación **bajando** el recall de
  peligro — y en test se desplomaban (en calor `F1_macro` 0.56→0.49, recall de
  peligro 0.69→0.52).
- **Precaución viva**: `rec(clase 1) >= 0.5 × rec(clase 1) del argmax`. Evita el
  colapso simétrico (`t1` tan alto que el modelo deja de emitir precaución,
  `rec1≈0`, y la precisión agregada se "dispara" por 1–2 muestras que no
  transfieren: en frío `Prec_riesgo` 0.23→0.08).

## 4. La frontera recall/precisión

![Frontera recall/precisión por clase](../reports/figures/calibracion_frontera.png)

Los tres puntos de operación caen **muy cerca del argmax**: la frontera es plana
en esa zona. Es el hallazgo central — **el argmax ya está cerca del codo de la
curva**; la calibración no regala grandes saltos de precisión, sino que permite
**reasignar cobertura hacia peligro** y recortar falsas alarmas a igualdad de
recall. Frontera tabulada en `reports/calibracion_umbrales_frontera_{clase}.csv`.

### Puntos de operación elegidos (sobre validación)

- **(a) máxima cobertura** — max `Rec_riesgo` con `Prec_riesgo >= argmax`.
- **(b) máxima precisión** — max `Prec_riesgo` con `Rec_riesgo >= argmax`.
- **(c) recomendado** — max **F2** de riesgo (recall pesa doble, coherente con
  "mejor sobre-avisar") sin perder precisión agregada frente al argmax.

| clase | punto | t1 | t2 |
|---|---|---|---|
| calor | (a) max_recall | 0.56 | 0.44 |
| calor | (b) max_precision | 0.50 | 0.44 |
| **calor** | **(c) recomendado** | **0.60** | **0.42** |
| frío | (a) max_recall | 0.66 | 0.40 |
| frío | (b) max_precision | 0.66 | 0.38 |
| **frío** | **(c) recomendado** | **0.70** | **0.38** |

## 5. Resultados en test (argmax vs puntos elegidos)

Solo los puntos elegidos en validación se evalúan en test, contra el argmax
actual. `Rec_r`/`Prec_r` = recall/precisión macro de clases 1–2; `rec2` = recall
de peligro; `avisos` = % de días con aviso (coste operativo de las alarmas).

### Calor — XGBoost_calor

| punto | t1 | t2 | Rec_r | Prec_r | rec2 (peligro) | F1_macro | avisos |
|---|---|---|---|---|---|---|---|
| **argmax (actual)** | — | — | 0.6331 | 0.2949 | 0.6894 | 0.5629 | 0.269 |
| (a) max_recall | 0.56 | 0.44 | **0.6395** | **0.3016** | 0.7086 | 0.5665 | 0.273 |
| (b) max_precision | 0.50 | 0.44 | **0.6520** | 0.2974 | 0.7086 | 0.5614 | 0.289 |
| **(c) recomendado** | 0.60 | 0.42 | 0.6329 | **0.2986** | **0.7374** | **0.5679** | **0.262** |

En calor la mejora es **real y Pareto**: (a) sube a la vez recall y precisión de
riesgo; (c) mantiene el recall total, sube precisión, **eleva el recall de
peligro casi 5 pp (0.689→0.737)** y encima **avisa menos** (26.2 % vs 26.9 %) —
menos alarmas y mejor detección de lo grave a la vez.

### Frío — RandomForest_frio

| punto | t1 | t2 | Rec_r | Prec_r | rec2 (peligro) | F1_macro | avisos |
|---|---|---|---|---|---|---|---|
| **argmax (actual)** | — | — | 0.5256 | 0.2339 | 0.4962 | 0.5117 | 0.226 |
| (a) max_recall | 0.66 | 0.40 | 0.5108 | 0.2332 | 0.5509 | 0.5162 | 0.210 |
| (b) max_precision | 0.66 | 0.38 | 0.5174 | 0.2227 | 0.6469 | 0.5125 | 0.210 |
| **(c) recomendado** | 0.70 | 0.38 | 0.5025 | 0.2330 | **0.6469** | **0.5202** | **0.191** |

En frío el `Rec_riesgo` **total** no se puede mejorar (el argmax ya está en el
techo de recall), pero las probabilidades del RandomForest son más ruidosas y la
calibración sirve para **reasignar cobertura hacia peligro**: (c) sube el recall
de peligro **15 pp (0.496→0.647)** a igualdad de precisión, mejora `F1_macro`
(0.512→0.520) y recorta avisos (22.6 %→19.1 %). El coste es menos recall de
*precaución* (la clase leve). Para un sistema de mortalidad es un buen canje:
detecta más días **graves** emitiendo **menos** avisos en total.

## 6. Recomendación

**Adoptar el punto (c) en ambos modelos.** No mejora espectacularmente el
agregado (el argmax ya era buen punto de operación), pero en las dos clases
**sube el recall de peligro, mejora `F1_macro` y reduce el número de avisos** —
justo lo que pide un sistema de aviso: menos falsas alarmas sin perder (ganando,
de hecho) detección de los días graves.

Los puntos (a) y (b) quedan documentados como alternativas si en el futuro se
quiere priorizar cobertura total (a) o precisión agregada (b) sobre el equilibrio.

## 7. Integración en el código

`predict_model.py` expone la calibración con **default = comportamiento actual**:

```python
from climasafeai.models.predict_model import predict_new

# Comportamiento de siempre (argmax) — sin cambios:
predict_new("XGBoost_calor", X)

# Umbrales recomendados (punto c) para ese modelo de riesgo:
predict_new("XGBoost_calor", X, class_thresholds="calor")
predict_new("RandomForest_frio", X, class_thresholds="frio")

# Umbrales a medida:
predict_new("XGBoost_calor", X, class_thresholds={"t1": 0.60, "t2": 0.42})
```

- `apply_class_thresholds(proba, t1, t2)` — la regla en cascada (§2).
- `CLASS_THRESHOLDS_RECOMENDADOS` — los umbrales del punto (c) por clase.
- `predict_new(..., class_thresholds=None)` (por defecto) es **idéntico** a antes
  (`model.predict`, argmax). Con `class_thresholds` string/dict aplica la cascada
  sobre `predict_proba`. Requiere un modelo de 3 clases con `predict_proba`.

### Reproducir

```bash
PYTHONPATH=. python tuning/calibracion_umbrales.py \
    --data-dir data/processed --models-dir models
```

Genera: `reports/calibracion_umbrales_puntos.csv` (esta comparativa),
`reports/calibracion_umbrales_frontera_{clase}.csv` (frontera de Pareto) y
`reports/figures/calibracion_frontera.png`. Las probabilidades de validación del
clon se cachean en `reports/calibracion_umbrales_probas_{clase}.npz` (borrar para
re-ajustar el clon desde cero).
