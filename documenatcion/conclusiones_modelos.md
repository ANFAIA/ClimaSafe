# Conclusiones del modelado — ClimaSafeAI

> Sistema de **aviso** de riesgo por temperatura (calor / frío), 3 clases por
> (provincia, día): `0=seguro`, `1=precaución`, `2=peligro`. Etiquetas derivadas
> de percentiles de mortalidad atribuida de MoMo; features meteorológicas de ERA5.

## 1. La métrica: por qué NO optimizamos accuracy ni F1 ponderado

Las clases están muy desbalanceadas (`seguro` ≈ 90% en calor, ≈ 94% en frío). Con ese
desbalance, **accuracy y F1_weighted están dominados por la clase mayoritaria** y premian
a modelos que casi nunca avisan. Ejemplo real: XGBoost sin ponderar sacaba **0.91 de
accuracy** en frío... prediciendo "seguro" siempre (recall de peligro = **0.02**).

Para un sistema de aviso lo que importa es **no perderse los días de riesgo**, así que
elegimos por:
- **`Rec_riesgo`** = recall medio de las clases 1 y 2 (¿cuántos días de riesgo detecta?).
- **`F1_macro`** = media por clase (equilibra recall y precisión sin que la clase 0 mande).

Ambas métricas se añadieron a `evaluate_models` (`predict_model.py`) y son las que ordenan
el resumen y la selección de modelo en los notebooks `0-2`.

## 2. Las features: 7 → 15 → 19 (el mayor salto son los lags)

| Iteración | Features | Qué añade | Efecto |
|---|---|---|---|
| Base | 7 | t2m, rh, viento, presión, Heat Index, WBGT, Wind Chill (hora pico) | punto de partida |
| Distribución diaria | +8 = 15 | media/std/min-max y nº horas sobre/bajo umbral de las 24 h | ayuda a **calor**, neutro en frío |
| **Persistencia (lags)** | **+4 = 19** | `heat_index_c_lag1`, `wind_chill_mean_roll3/7`, `dias_consec_bajo_umbral` | **mayor mejora, ambas clases** |

**Los lags son el mayor salto** (medido sobre el mismo split, XGBoost-con-pesos):

| Clase | Métrica | 15 features | 19 (+lags) |
|---|---|---|---|
| Calor | F1_macro | 0.467 | **0.547** |
| Calor | Recall peligro | 0.54 | **0.68** |
| Frío | F1_macro | 0.441 | **0.508** |
| Frío | Accuracy | 0.763 | **0.834** |

Y `wind_chill_mean_roll7` (media de sensación térmica de los **7 días previos**) es la
feature **nº 1** en ambos modelos; `dias_consec_bajo_umbral` (racha de días fríos
consecutivos) es la **nº 2** en frío. Esto confirma la hipótesis epidemiológica: el
efecto del frío sobre la mortalidad es **acumulativo y retardado** — la *racha* importa
más que el día suelto. Las features de persistencia (`_agregar_rezagos_temporales` en
`make_dataset.py`) se calculan **estrictamente sobre el pasado** (shift positivo por
provincia), así que no hay fuga de datos aunque el split sea temporal.

## 3. Los modelos y por qué XGBoost necesita pesos

Se comparan tres, todos evaluados por recall/precisión **por clase**:

- **RandomForest** con `class_weight="balanced"` y `max_depth` **por clase** (calor=12,
  frío=8 — el frío generaliza mejor con árboles más superficiales por su señal más
  sostenida). Ya nace sensible al desbalance.
- **XGBoost por defecto**: **inservible** para aviso (recall de riesgo 0.00–0.19) — colapsa
  a la clase mayoritaria porque no tiene `class_weight`.
- **XGBoost con `sample_weight` balanceado**: el arreglo. Se le pasan pesos por muestra en
  el `fit` (equivalente a `class_weight="balanced"`). Pasa a captar el riesgo (recall de
  peligro 0.50–0.68) y suele ser el mejor en balance global.

> Conclusión clave: **XGBoost solo es válido con pesos**. Desplegar el XGBoost por defecto
> habría sido el peor error posible (el modelo que "gana" en el leaderboard pero no avisa).

## 4. Modelo elegido por clase (19 features)

> **Criterio de selección = `Rec_riesgo`** (recall medio de las clases 1 y 2), NO `F1_macro`
> ni accuracy. Es una **decisión de producto**: esto es un sistema de AVISO y hablamos de
> vidas, así que preferimos **falsos positivos** (avisar de más) antes que **falsos
> negativos** (no avisar de un día peligroso). Maximizar `Rec_riesgo` = perderse el menor
> número de días de riesgo, aceptando más falsas alarmas (menos precisión).

| Clase | Modelo elegido | Rec precaución | Rec peligro | **Rec_riesgo** | F1_macro | Acc |
|---|---|---|---|---|---|---|
| **Calor** | **XGBoost (pesos)** | 0.55 | 0.68 | **0.614** | 0.546 | 0.783 |
| **Frío** | **RandomForest** | 0.63 | 0.44 | **0.527** | 0.488 | 0.771 |

**Por qué XGBoost en calor:** gana en `Rec_riesgo` (0.614 vs 0.599 del RF) **y** en todo lo
demás (recall de peligro 0.68 vs 0.66, F1_macro, accuracy). No hay conflicto.

**Por qué RandomForest en frío:** con el criterio de recall, RF gana claramente —
`Rec_riesgo` **0.527** vs 0.460 del XGBoost, sobre todo por precaución (0.63 vs 0.46). El
XGBoost de frío tiene mejor F1_macro (0.506) y accuracy (0.834), pero eso premia no avisar;
bajo la política de "mejor sobre-avisar", RF es el correcto. (El `XGBoost_frio.joblib` queda
guardado por si algún día se prioriza balance sobre detección.)

**KNN** se descartó en ambas: recall de riesgo ~0.02–0.23 (otro modelo que "no avisa").

## 5. Hiperparámetros finales

- **RandomForest**: `n_estimators=200, max_features="sqrt", max_samples=0.8,
  class_weight="balanced"`, `max_depth` = **12 (calor) / 8 (frío)**. Los pesos de clase
  custom más agresivos NO suben el recall total, solo lo trasladan entre clases —
  `"balanced"` es el punto óptimo (ver `reports/rf_tuning_*.csv`).
- **XGBoost**: `n_estimators=300, max_depth=6, learning_rate=0.05, subsample=0.8,
  colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1.0` + **`sample_weight` balanceado**.

## 6. Techo actual y próximos pasos

- **El techo es la precisión de las clases de riesgo** (~0.08–0.27): muchos avisos son
  falsas alarmas. Es limitación de **señal/etiqueta** (etiquetas por percentil de mortalidad
  + pocas variables), no de modelo. Los lags subieron mucho el recall y el balance, pero la
  precisión sigue siendo el cuello de botella.
- **Siguientes palancas**: más histórico en los lags (rachas más largas, grados-día
  acumulados), variables de exposición/población por provincia, o revisar los cortes de
  percentil de las etiquetas.

## 7. Estado de despliegue

- Modelos guardados **namespaceados por clase** (`train_models` guarda
  `{modelo}_{clase}.joblib`, ya no se pisan calor↔frío).
- ✅ **Datos regenerados con lags y modelos reentrenados**: en `models/` están los
  6 modelos (`{RandomForest,XGBoost,KNN}_{calor,frio}.joblib`), todos con **19 features**
  (`feature_names_{clase}.joblib` lo confirma). Los `.joblib` planos antiguos (sin clase,
  15 features) se eliminaron.
- **Desplegados** (elegidos por `Rec_riesgo`, política "mejor sobre-avisar"):
  `XGBoost_calor.joblib` y `RandomForest_frio.joblib`. El `XGBoost_frio.joblib` queda
  guardado como alternativa (mejor balance/accuracy, menos falsas alarmas) por si algún día
  se cambia la política.
- Para reproducir: notebooks `0-1` (procesado, ya incluyen lags) → `0-2` (entreno), o el
  pipeline completo de `make_dataset` + `train_models(..., clase=...)`.
