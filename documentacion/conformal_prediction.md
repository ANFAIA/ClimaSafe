# Conformal Prediction — calibración de probabilidades

## Contexto

Los modelos desplegados (RandomForest_frío, XGBoost_calor) usan `class_weight="balanced"`
y `sample_weight` para lidiar con el desbalance de clases (~91% clase 0). Esto distorsiona
las probabilidades de salida: el modelo clasifica bien (Rec_riesgo ~0.57-0.61) pero las
probabilidades no son fiables como estimaciones de riesgo real.

## Split Conformal Prediction vs Isotonic Regression

| Método | Qué hace | Cuándo usarlo |
|--------|----------|---------------|
| **Split conformal** | Prediction sets con cobertura garantizada (PELIGRO está en el set el 90% de las veces) | Auditoría de incertidumbre; útil cuando necesitas saber si el modelo "está seguro" |
| **Isotonic Regression** | Recalibra la probabilidad de PELIGRO (clase 2) con una función monótona no paramétrica | Producción: mejora Brier, thresholds recalibrados dan mejor Rec_riesgo |

## Resultados (RandomForest_frío sobre test)

| Métrica | Sin calibración | Con isotonic | Cambio |
|---------|----------------|--------------|--------|
| Brier score (PELIGRO) | 0.0464 | **0.0321** | **-30.8%** |
| Rec_riesgo (t1/t2 raw) | 0.6122 | — | umbrales 0.45/0.40 sobre raw |
| Rec_riesgo (thr calibrados) | — | **0.6334** | **+0.0212** vs raw |
| Acc (thr calibrados) | 0.7032 | 0.6370 | esperado (recall↑ precisión↓) |

\* Umbrales recalibrados sobre probas post-isotonic: t1=**0.21**, t2=**0.20**.

## Interpretación de los prediction sets

Split conformal (alpha=0.1):

- **Set size = 1**: el modelo está seguro de la clase
- **Set size = 2**: el modelo duda entre dos clases
- **Set size = 3**: máxima incertidumbre (solo pasa en RF_frío, ~2.84 de media)

El RF_frío tiene sets más grandes (²2.84) porque sus probabilidades están más
repartidas entre las 3 clases. XGBoost_calor tiene sets pequeños (²1.14) porque
sus probabilidades son más concentradas.

## Notebook de referencia

`notebooks/0-4-ConformalPrediction.ipynb` — experimento completo con reliability
diagrams, Brier scores y matrices de confusión comparativas.

## Implementación en producción

### Fitting (post-entrenamiento en main.py)

```python
from climasafeai.models.calibrate import fit_isotonic
# validation = último 15% de train (por fecha)
fit_isotonic(rf_frio, X_val, y_val, clase="frio")
# guarda → models/artifacts/iso_calib_frio.joblib
```

### Predicción (wrapper)

```python
from climasafeai.models.predict_model import predict_new

# Opción recomendada: isotonic + thr calibrados (un solo paso)
pred = predict_new("RandomForest_frio", X_new,
                   class_thresholds="frio", calibrate_isotonic=True)

# Equivalente paso a paso:
from climasafeai.models.calibrate import load_isotonic, calibrate_proba
from climasafeai.models.predict_model import apply_class_thresholds, CLASS_THRESHOLDS_RECOMENDADOS

iso = load_isotonic("frio")          # None si no existe
proba = modelo.predict_proba(X)      # shape (n, 3)
if iso:
    proba = calibrate_proba(proba, iso)  # recalibra clase 2

u = CLASS_THRESHOLDS_RECOMENDADOS["frio"]  # t1=0.21, t2=0.20
pred = apply_class_thresholds(proba, **u)
```

### Recomendación

- **RandomForest_frío**: aplicar isotonic + thresholds recalibrados (t1=**0.21**, t2=**0.20**)
  → `predict_new(..., calibrate_isotonic=True)` lo hace automáticamente
- **XGBoost_calor**: no necesita calibración (Brier 0.057, set size 1.14)

## Referencias

- Vovk et al. (2005) "Algorithmic learning in a random world"
- Angelopoulos & Bates (2021) "A gentle introduction to conformal prediction"
- `sklearn.isotonic.IsotonicRegression` — implementación usada
