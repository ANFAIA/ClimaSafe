# Problemas Conocidos

## 1. Bug: `p2_frac` dividía `prob_riesgo` en `_prob_a_clase`

**Dónde**: `ensemble.py:_prob_a_clase()`

**Problema**: La función dividía `prob_riesgo` por `p2_frac` para estimar P(2):

```python
p2 = prob_riesgo * p2_frac / p2_frac  # ~ p2_frac se cancelaba pero afectaba
```

El bug real no era la cancelación sino que **`p2_frac` es minúsculo** en estratos de bajo riesgo (e.g., viejano = 0.065). Con thresholds F1-optimizados como t2=0.54, P2 necesitaba prob ≥ 8.3 para activarse, lo cual es imposible (prob máxima = 1.0). Resultado: PELIGRO no se predecía nunca.

**Solución**: `_prob_edad_a_clase()` en `ensemble.py:82` aplica thresholds directamente sobre las probabilidades del modelo, sin split por `p2_frac`. 

**Lección**: Siempre verificar que un threshold está siendo alcanzable antes de darlo por válido.

---

## 2. F1 optimization produce falsos positivos masivos para clases raras

**Dónde**: `tuning/calibrar_umbrales_f1.py`

**Problema**: Optimizar F1 (beta=1.0) para PELIGRO (clase 2, <0.1% prevalencia) produce cientos de falsos positivos. F1 da igual peso a precision y recall, pero con clases tan desbalanceadas, el grid encuentra thresholds bajos que maximizan F1 a costa de FP masivos (e.g., 4559 FP para calor).

**Solución**: Usar F0.5 (beta=0.5) que pondera precision 2× más que recall. Los thresholds F0.5 son más altos y generan menos FP, aunque sacrifican recall.

**Lección**: Para clases extremadamente raras, F1 no es la métrica adecuada para calibración de thresholds. Usar Fβ con β < 1.

---

## 3. Límite fundamental: prevalencia de PELIGRO en frío

**Dónde**: Todo el pipeline de frío

**Problema**: PELIGRO en frío tiene una prevalencia de **~0.075%** en entrenamiento (990 de 689k) y **~0.35%** en viejano (máximo estrato). Esto impone un límite estadístico a la precisión alcanzable.

Para un estrato con prevalencia p, si el modelo asigna P(2) = q a los no-PELIGRO, la precisión máxima teórica es:

```
prec_max ≈ p / (p + q)
```

Con p=0.0035 (viejano) y q=0.02 (P(2) media en no-PELIGRO):

```
prec_max ≈ 0.0035 / (0.0035 + 0.02) ≈ 0.149
```

Es decir, ~15% de precisión es el límite práctico. Para anciano (p=0.00029), el límite cae a ~1.4%.

**Solución**: 
- Aceptar que PELIGRO en frío tiene precisión intrínsecamente baja.
- Para viejano: binario XGBoost con t=0.97 da ~19.5% precisión (TP=15, FP=62).
- Para anciano: no intentar predecir PELIGRO directamente.
- El valor del sistema está en PRECAUCIÓN (clase 1), que tiene mejor relación señal/ruido.

**Lección**: No todas las clases pueden predecirse con utilidad clínica. Documentar los límites es parte del modelo.

---

## 4. Las features de persistencia frío (FRIO_EXTRA_COLS) fueron excluidas por ablación del modelo "todos", no del "edad"

**Dónde**: `build_features.py:FRIO_EXTRA_COLS`, `COLS_TO_DROP_BY_CLASE`

**Problema**: La ablación 27v19 mostró que 8 features de persistencia (wind_chill_mean_roll3/7/14, grados_dia_frio_roll7/14, etc.) empeoraban el recall de riesgo para frío en el modelo "todos" (único modelo, sin edad como feature). Pero estas features podrían ser beneficiosas para el modelo "edad" (que tiene grupo_edad como feature y puede aprender interacciones).

**Acción**: Se restauraron para el modelo edad (COLS_TO_DROP_BY_CLASE["frio"] = []).

**Lección**: Las decisiones de feature selection son específicas del modelo. Una ablación en el modelo "todos" no necesariamente se traslada al modelo "edad".

---

## 5. XGBoost multiclass vs binario para PELIGRO

**Dónde**: Modelos `XGBoost_frio_edad.joblib` (multiclass) vs `XGBoost_frio_binario.joblib` (binario)

**Problema**: El XGBoost multiclass (3 clases) reparte la probabilidad entre 0/1/2. Para PELIGRO, la P(2) media entre los casos reales de PELIGRO es solo ~0.26 (viejano), porque la masa de probabilidad se divide. El XGBoost binario (PELIGRO vs no-PELIGRO) da P(2) media de ~0.74.

**Solución**: El binario es superior para detección de PELIGRO. Usar multiclass para PRECAUCIÓN y binario para PELIGRO (o el binario directamente para decisiones de PELIGRO).

**Lección**: Para clases raras, la formulación binaria (uno-vs-resto) da mejor separación de probabilidades que la multiclass.

---

## 6. Rolling windows requieren datos históricos durante inferencia

**Dónde**: `weather_fetcher.py:_generar_features_completas()` → `_agregar_rezagos_temporales()`

**Problema**: Features como `wind_chill_mean_roll7`, `grados_dia_frio_roll7`, `dias_consec_bajo_umbral` requieren 7-14 días de datos históricos. Durante inferencia, `fetch_weather_data()` obtiene 14 días de Open-Meteo archive, pero si la API falla o los datos históricos no están disponibles, estas features serán NaN.

Si el scaler fue entrenado con estas features y en inferencia no existen (o son NaN), el pipeline falla.

**Mitigación**: `_feature_engineering()` en `build_features.py` usa condiciones (`if col in df.columns`) para crear features derivadas. Si una columna base falta, la feature derivada simplemente no se crea. Pero el scaler espera N columnas fijas.

**Solución**: Asegurar que los datos históricos estén siempre disponibles en inferencia, o rellenar con promedios.

---

## 7. `_feature_engineering` se ejecuta ANTES de `_drop_excluded_cols`

**Dónde**: `build_features.py:preprocess_data()` (líneas 180 vs 203)

**Problema**: El orden de operaciones es:
1. `_feature_engineering(df)` → añade features derivadas
2. `_drop_excluded_cols` → elimina columnas (COLS_TO_DROP + LEAKAGE + COLS_TO_DROP_BY_CLASE)

Esto significa que features derivadas de columnas que luego se eliminan pueden "colar" información. Por ejemplo, `tmin_x_edad = t2m_min_noche * grupo_edad`: si `t2m_min_noche` se elimina para calor (dropeada por COLS_TO_DROP_BY_CLASE["calor"]), `tmin_x_edad` sobrevive y contiene información de `t2m_min_noche` indirectamente.

**Impacto**: Bajo para el modelo edad (no reintroduce fuga de target), pero conceptualmente incorrecto.

**Solución**: Mover `_feature_engineering` después de `_drop_excluded_cols`, o asegurar que las features derivadas no reintroduzcan columnas eliminadas intencionalmente.

---

## 8. Los thresholds por estrato no se aplican en `_predecir_tabular_edad`

**Dónde**: `ensemble.py:_predecir_tabular_edad()` (línea 183-184)

**Problema**: `_predecir_tabular_edad()` usa `CLASS_THRESHOLDS_RECOMENDADOS` (globales) para calcular `clase_threshold`, no los thresholds por estrato de `params_estrato.joblib`. Esto significa que `resultados["XGBoost_calor"]["clase_threshold"]` NO usa los thresholds F0.5 optimizados por estrato.

Los thresholds por estrato sí se aplican más tarde en `predict_ensemble()` vía `_prob_edad_a_clase()` (líneas 523/531), reemplazando el valor de `clase_threshold`. Pero cualquier lógica que use `clase_threshold` directamente (antes de llegar a predict_ensemble) obtendría resultados subóptimos.

**Solución**: `_predecir_tabular_edad()` debería usar thresholds por estrato en lugar de globales.

---

## 9. `apply_class_thresholds` tiene handling inconsistente de shapes

**Dónde**: `predict_model.py:apply_class_thresholds()`

**Problema**: La función maneja arrays 2-D (N, 3) con `np.where` y arrays 1-D (3,) con `if/else`. Pero la versión 1-D tenía un bug: el primer `if` capturaba arrays 1-D de cualquier tamaño (e.g., (103410,)) como "shape válido" y los procesaba incorrectamente (accedía a `proba_arr[2]` en un array enorme).

**Solución**: Reescribir para que el caso 1-D reshape a (1,3) y devuelva scalar, y el caso 2-D procese normalmente. Ver `predict_model.py:apply_class_thresholds` para la implementación corregida.
