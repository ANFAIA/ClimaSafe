# Random Forest — modelo desplegado (frío)

## Rol
Modelo elegido para **riesgo por frío** (`RandomForest_frio.joblib`). Segundo modelo en calor.

## Hiperparámetros
- `n_estimators=200`, `max_features="sqrt"`, `max_samples=0.8`
- `class_weight="balanced"`
- `max_depth=12` (calor) / `max_depth=8` (frío — generaliza mejor con árboles más superficiales)

## Rendimiento (19 features)

| Clase | Rec_riesgo | F1_macro | Accuracy |
|-------|-----------|----------|----------|
| Calor | 0.599 | — | — |
| **Frío** | **0.527** | 0.488 | 0.771 |

En frío gana a XGBoost por `Rec_riesgo` (0.527 vs 0.460), sobre todo por recall de precaución (0.63 vs 0.46). Bajo la política de "mejor sobre-avisar", RF es el correcto para frío.

## Fortalezas
- Maneja desbalance naturalmente con `class_weight="balanced"`
- Interpretable (feature_importances_)
- Robusto a outliers y no requiere escalado

## Debilidades frente a alternativas
- No modela dependencias temporales explícitas (lags ayudan pero son manuales)
- Menor capacidad que XGBoost/LightGBM en señal densa

## Referencias
- Breiman, L. (2001). Random Forests. *Machine Learning*, 45(1):5–32.
- Implementación: `sklearn.ensemble.RandomForestClassifier`
