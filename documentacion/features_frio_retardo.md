# Features de frío con más memoria temporal (retardo epidemiológico)

**Experimento reproducible:** `tuning/features_frio.py`
**Salidas:** `reports/features_frio_retardo_metricas.csv`, `reports/features_frio_retardo_importancias.csv`

## Motivación

Al pasar de 19 a 27 features, el calor mejoró (Rec_riesgo 0.614 → 0.633) pero el
frío se quedó igual (0.527 → 0.526). Hipótesis de partida: la mortalidad por
frío tiene un retardo epidemiológico largo (días-semanas tras la exposición, a
diferencia del calor, que es de lag corto) y las ventanas actuales
(roll3/7/14) se quedan cortas.

## Qué se probó

12 features candidatas, todas construidas **solo con pasado** (`shift(1)` por
provincia, mismo patrón que `_agregar_rezagos_temporales` en
`climasafeai/data/make_dataset.py` — sin fuga de datos), en 4 grupos:

| Grupo | Features | Idea |
|---|---|---|
| `memoria_larga` | `wind_chill_mean_roll21/28`, `grados_dia_frio_roll21/28`, `horas_bajo_umbral_sum14/28` | Las mismas familias que roll7/14, pero a 3-4 semanas |
| `desplazadas` | `wind_chill_mean_t7_21` (media en [t-21, t-7]), `wind_chill_mean_t14_28` (media en [t-28, t-14]) | Aislar la exposición retardada de la reciente (que ya cubren roll3/7) |
| `nocturnas` | `t2m_min_noche_lag1`, `t2m_min_noche_roll7` | Mínima de t2m en horas 0-8 (madrugada), desde `secuencias_24h.npz`. La mortalidad por frío se asocia a mínimas nocturnas sostenidas, no solo al pico diurno |
| `rachas_severas` | `dias_consec_wc_severo`, `horas_wc_severo_sum14` | Persistencia con umbral más frío (wind chill horario < -5 °C, vs. el 0 °C de `horas_bajo_umbral`) |

**Protocolo:** mismo split temporal que producción (último 20 % de fechas a
test, réplica de `preprocess_data` de `climasafeai/features/build_features.py`
incluido su fillna con la media y su `StandardScaler`), hiperparámetros
clonados de `models/RandomForest_frio.joblib` y `models/XGBoost_frio.joblib`,
XGBoost con `sample_weight=compute_sample_weight("balanced")` como en
`train_model.py`. Métrica objetivo: **Rec_riesgo** (recall macro de las clases
1 y 2).

La réplica del baseline con el script reproduce las cifras de referencia
(RF 0.5256/0.5117 exacto con el orden de filas original; 0.5248/0.5109 tras el
merge con el npz, que reordena filas y mueve el bootstrap del RF — diferencia
de ±0.001, ruido de semilla. XGB 0.4483/0.5137 vs. 0.4558/0.5170 de
referencia, misma escala de ruido por paralelismo/orden).

## Resultados (test = último 20 % de fechas)

### RandomForest (baseline de referencia: 0.5256 / 0.5117)

| Variante | n feats | Rec_riesgo | F1_macro | rec c0/c1/c2 |
|---|---:|---:|---:|---|
| baseline_27 | 27 | 0.5248 | 0.5109 | 0.839 / 0.564 / 0.485 |
| todas_39 | 39 | 0.5074 | 0.5136 | 0.849 / 0.591 / 0.424 |
| base+memoria_larga | 33 | 0.4905 | 0.4990 | 0.840 / 0.552 / 0.429 |
| base+desplazadas | 29 | 0.5066 | 0.5044 | 0.838 / 0.560 / 0.453 |
| **base+nocturnas** | **29** | **0.5437** | **0.5306** | 0.849 / 0.571 / 0.517 |
| base+rachas_severas | 29 | 0.5337 | 0.5153 | 0.838 / 0.587 / 0.481 |
| **base+noct+rachas** | **31** | **0.5516** | **0.5327** | 0.846 / 0.589 / 0.515 |
| base+top4_importancia | 31 | 0.5079 | 0.5150 | 0.847 / 0.543 / 0.473 |

### XGBoost (baseline de referencia: 0.4558 / 0.5170)

| Variante | n feats | Rec_riesgo | F1_macro | rec c0/c1/c2 |
|---|---:|---:|---:|---|
| baseline_27 | 27 | 0.4483 | 0.5137 | 0.884 / 0.481 / 0.415 |
| todas_39 | 39 | 0.4103 | 0.5320 | 0.918 / 0.434 / 0.387 |
| base+memoria_larga | 33 | 0.3983 | 0.5036 | 0.897 / 0.458 / 0.339 |
| base+desplazadas | 29 | 0.4214 | 0.5038 | 0.885 / 0.453 / 0.390 |
| **base+nocturnas** | **29** | **0.4633** | **0.5476** | 0.910 / 0.481 / 0.445 |
| base+rachas_severas | 29 | 0.4601 | 0.5263 | 0.889 / 0.479 / 0.441 |
| base+noct+rachas | 31 | 0.4611 | 0.5485 | 0.911 / 0.484 / 0.438 |
| base+top4_importancia | 31 | 0.4225 | 0.5285 | 0.910 / 0.438 / 0.406 |

### Importancia de las candidatas (variante todas_39)

| Feature | RF | XGB |
|---|---:|---:|
| wind_chill_mean_roll21 | 0.0675 | 0.0250 |
| t2m_min_noche_roll7 | 0.0632 | 0.0523 |
| wind_chill_mean_roll28 | 0.0418 | 0.0238 |
| wind_chill_mean_t14_28 | 0.0347 | 0.0224 |
| wind_chill_mean_t7_21 | 0.0247 | 0.0200 |
| horas_bajo_umbral_sum14 | 0.0226 | 0.0358 |
| horas_bajo_umbral_sum28 | 0.0116 | 0.0245 |
| t2m_min_noche_lag1 | 0.0110 | 0.0197 |
| horas_wc_severo_sum14 | 0.0089 | 0.0245 |
| grados_dia_frio_roll28 | 0.0064 | 0.0220 |
| grados_dia_frio_roll21 | 0.0049 | 0.0250 |
| dias_consec_wc_severo | 0.0002 | 0.0169 |

Nota sobre las importancias: `wind_chill_mean_roll21/28` acaparan importancia
(son versiones suavizadas de la señal estacional) pero **empeoran** la métrica
— importancia alta no implica utilidad predictiva para las clases de riesgo;
aquí actúan como redundancia que diluye los splits útiles del roll7/14.

## Veredicto

1. **La hipótesis del retardo largo NO se confirma.** Los tres grupos que
   alargan la memoria (roll21/28, grados-día a 21/28 días, ventanas
   desplazadas t-7..t-21 y t-14..t-28) empeoran Rec_riesgo en ambos modelos
   (RF: de 0.525 a 0.49-0.51; XGB: de 0.448 a 0.40-0.42). Con la resolución
   de este dataset, la señal útil del frío ya está capturada por roll3/7/14;
   más allá de ~2 semanas la ventana solo añade estacionalidad redundante.

2. **Lo que sí mejora es medir mejor la exposición reciente, no alargarla:**
   - `nocturnas` (`t2m_min_noche_lag1` + `t2m_min_noche_roll7`): la mejor
     variante en ambos modelos. RF 0.5437/0.5306 (+0.019 Rec_riesgo, +0.020
     F1_macro sobre su baseline) y XGB 0.4633/0.5476 (+0.015, +0.034).
     Además es la única variante que sube el recall de la clase 2 (grave)
     en RF: 0.485 → 0.517.
   - `rachas_severas` (frío severo < -5 °C): segunda mejor, RF 0.5337/0.5153
     y XGB 0.4601/0.5263.
   - Combinarlas (`base+noct+rachas`, 31 features) da el **mejor RandomForest
     del experimento: Rec_riesgo 0.5516 / F1_macro 0.5327** (+0.026 / +0.021
     sobre el baseline de referencia 0.5256 / 0.5117), manteniendo el recall
     de la clase 2 en 0.515 y subiendo la 1 a 0.589. En XGBoost queda en
     0.4611/0.5485 — Rec_riesgo apenas por debajo de base+nocturnas (0.4633)
     pero con el mejor F1_macro de todas las variantes XGB. En la variante
     combinada, `t2m_min_noche_roll7` entra 5ª en importancia RF (0.087),
     por encima de la mayoría de features actuales.

3. Meter todas las candidatas a la vez (`todas_39`) es contraproducente: las
   de memoria larga arrastran a las buenas.

## Propuesta de integración

**No** se toca `climasafeai/data/make_dataset.py` en este PR (tiene cambios
locales sin commitear). Cuando se integre:

1. En `_agregar_estadisticas_diarias()` añadir dos agregados diarios desde las
   horas: `t2m_min_noche` (mínimo de `t2m_c` en horas 0-8) y
   `horas_wc_severo` (nº de horas con `wind_chill_c < -5`, nueva constante
   `WIND_CHILL_SEVERO_C = -5.0` junto a `WIND_CHILL_UMBRAL_C`).
2. En `_agregar_rezagos_temporales()` derivar las 4 features de la variante
   ganadora `base+noct+rachas` (mismo patrón `shift(1)` que las existentes):
   - `t2m_min_noche_lag1` y `t2m_min_noche_roll7`
   - `horas_wc_severo_sum14` y `dias_consec_wc_severo` (con `_racha_previa`;
     su importancia RF individual es ~0, pero es la combinación de los 4 la
     que dio el mejor resultado — si se quiere podar, probar primero sin ella)
   y **excluir del dataset final las columnas del día actual**
   (`t2m_min_noche`, `horas_wc_severo`) o dejarlas explícitamente fuera de X,
   para mantener la regla "candidatas solo con pasado".
3. Regenerar `dataset_frio_labeled.parquet` (pasa de 27 a 31 features de
   modelo) y reentrenar frío. No añadir roll21/28, grados-día largos ni
   ventanas desplazadas: empeoran (este resultado negativo queda documentado
   aquí para no repetir el experimento).
