# Label sin fuga temporal: percentiles train-only

## El problema

Las etiquetas de riesgo (0=SEGURO / 1=PRECAUCIÓN / 2=PELIGRO) se asignan en
`climasafeai/features/labels.py` por percentiles de rango (p75/p95) de la
mortalidad atribuida de MoMo, por provincia. Hasta ahora esos percentiles se
calculaban sobre **todo el histórico** (2016-01-01 → 2026-06-27), pero el
split train/test del proyecto es temporal (último 20% de fechas → test, ver
`preprocess_data(split_by_date=True)`).

Consecuencia: el label de cada día de **train** dependía también de la
distribución de mortalidad del periodo de **test** (fuga temporal
train↔test). El modelo entrenaba contra una etiqueta que "ya sabía" cómo iba
a ser la mortalidad futura, y las métricas de test quedaban infladas de forma
no honesta.

## La solución

`asignar_clase_riesgo_calor()` y `asignar_clase_riesgo_frio()` aceptan ahora
un parámetro opcional `fecha_corte_percentiles` (+ `col_fecha`, por defecto
`"fecha"`):

- **Sin el parámetro** (por defecto): comportamiento clásico, percentiles
  full-history. Nada cambia para el código existente.
- **Con el parámetro** (= fecha de inicio del test): la distribución de
  referencia de los percentiles se construye SOLO con filas de
  `fecha < corte` (train) y se aplica a todo el dataset. El percentil de cada
  día de test se evalúa contra la distribución de train mediante el rango
  medio dentro de la referencia (equivalente exacto a
  `rank(pct=True, method="average")` para las filas de train, y punto medio
  de la ECDF para valores nuevos). Se mantiene así la robustez frente a
  empates/series casi constantes por la que se eligió rank en vez de
  `pd.cut` sobre bordes de quantile.

Propiedad clave (cubierta por `tests/test_labels.py`): con el corte, las
filas de train quedan etiquetadas EXACTAMENTE igual que si se etiquetara el
subconjunto de train por separado, y el label de un día de test no depende de
los valores de otros días de test.

## Experimento

Script reproducible: `experimento_label_sin_fuga.py` (raíz del repo).
Réplica exacta de la regla de split (último 20% de fechas distintas → test):
**corte = 2024-05-23**, 137.925 filas de train / 34.470 de test. Modelos
reentrenados con los MISMOS hiperparámetros de los joblib desplegados
(`sklearn.base.clone` de `XGBoost_calor.joblib` y `RandomForest_frio.joblib`;
XGBoost con `sample_weight` balanceado, RandomForest ya lleva
`class_weight='balanced'`), mismo `preprocess_data`.

### Magnitud de la fuga (cuánto cambia el label)

Comparación label clásico (full-history, código commiteado) vs label
train-only, sobre los parquet `dataset_{calor,frio}_labeled.parquet`:

| Dataset | Filas que cambian de clase | En train | En test |
|---|---|---|---|
| Calor | 1.085 (0,63 %) | 831 (0,60 %) | 254 (0,74 %) |
| Frío | 763 (0,44 %) | 512 (0,37 %) | 251 (0,73 %) |

Distribución de clases en el periodo de test (nótese que al cambiar el label
cambia también `y_test`, así que las métricas con-fuga y sin-fuga se evalúan
cada una contra su propio `y_test`):

| Dataset | y_test con fuga (0/1/2) | y_test sin fuga (0/1/2) |
|---|---|---|
| Calor | 30.446 / 1.842 / 2.182 | 30.446 / 1.634 / 2.390 |
| Frío | 31.379 / 1.030 / 2.061 | 31.379 / 849 / 2.242 |

El sesgo es sistemático y en la dirección esperada: la mortalidad del periodo
reciente es alta respecto al histórico de train, así que con percentiles
train-only más días de test suben a PELIGRO (clase 2: +208 calor, +181 frío)
desde PRECAUCIÓN. Con percentiles full-history, el periodo de test "diluía"
su propia gravedad dentro de la referencia.

### Métricas honestas (test)

| Modelo | Variante | Rec_riesgo | F1_macro | Recall clase 1 | Recall clase 2 |
|---|---|---|---|---|---|
| XGBoost calor | Referencia (label filtrado actual, con fuga) | 0.6331 | 0.5629 | — | — |
| XGBoost calor | Con fuga (label commiteado, reentrenado) | 0.6214 | 0.5567 | 0.5461 | 0.6966 |
| XGBoost calor | **Sin fuga (train-only)** | **0.6106** | **0.5517** | 0.5196 | 0.7017 |
| RF frío | Referencia (label filtrado actual, con fuga) | 0.5256 | 0.5117 | — | — |
| RF frío | Con fuga (label commiteado, reentrenado) | 0.5045 | 0.4764 | 0.6107 | 0.3984 |
| RF frío | **Sin fuga (train-only)** | **0.4898** | **0.4776** | 0.5960 | 0.3836 |

Notas de lectura:

- La comparación limpia del efecto de la fuga es **"con fuga (reentrenado)"
  vs "sin fuga"**: mismo código, mismo pipeline, misma semilla; solo cambia
  cómo se calculan los percentiles del label.
- La fila "Referencia" viene de una ejecución previa cuyo label incluye
  además un filtro local sin commitear (`min_mortalidad_peligro=2.0`, que
  degrada PELIGRO→PRECAUCIÓN si la mortalidad < 2): el label guardado en los
  parquet difiere del recomputado con el código commiteado en un 0,51 %
  (calor) / 0,89 % (frío) de las filas. Por eso "Referencia" y "Con fuga
  (reentrenado)" no coinciden exactamente; la brecha Referencia↔sin-fuga
  mezcla dos efectos (filtro + fuga).

### Conclusión

- **La fuga existe pero es moderada**: quitar la fuga cuesta ~0,011 de
  Rec_riesgo en calor (0.6214 → 0.6106) y ~0,015 en frío (0.5045 → 0.4898).
  F1_macro apenas se mueve (−0,005 calor, +0,001 frío). Las métricas
  publicadas hasta ahora estaban infladas en ese orden de magnitud.
- El grueso del cambio no es que el modelo empeore, sino que el `y_test`
  honesto es más exigente: hay más días PELIGRO reales en test cuando el
  periodo reciente se evalúa contra la distribución histórica de train.

## Recomendación

1. **Adoptar `fecha_corte_percentiles` como estándar** al generar los
   datasets etiquetados para entrenamiento/evaluación: pasar siempre la fecha
   de inicio del test que usará `preprocess_data` (último 20 % de fechas).
   Las métricas honestas (~0.61 calor / ~0.49 frío de Rec_riesgo) son las que
   deben reportarse y contra las que debe tunearse.
2. Para el **modelo desplegado final** (entrenado con todo el histórico, sin
   test), sí es correcto usar percentiles full-history: ahí ya no hay periodo
   futuro que fugar.
3. **Pendiente de reconciliar**: este trabajo parte del `labels.py`
   commiteado; hay un cambio local sin commitear (`min_mortalidad_peligro`)
   que deberá integrarse con el nuevo parámetro (el filtro por mortalidad
   absoluta es ortogonal al corte temporal y puede aplicarse después del
   percentil, pero hay que re-medir las métricas al combinarlos).
4. Si se quiere reducir la caída de recall de la clase 1, re-tunear
   hiperparámetros contra el label sin fuga (aquí se congelaron los actuales
   para aislar el efecto).

## Reproducir

```bash
python experimento_label_sin_fuga.py   # requiere dataset_*_labeled.parquet y los joblib de referencia
pytest tests/test_labels.py -q
```
