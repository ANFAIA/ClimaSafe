# XGBoost — modelo desplegado (calor)

## Rol
Modelo elegido para **riesgo por calor** (`XGBoost_calor.joblib`). Segundo modelo en frío (guardado como alternativa).

## Hiperparámetros
- `n_estimators=300`, `max_depth=6`, `learning_rate=0.05`
- `subsample=0.8`, `colsample_bytree=0.8`
- `reg_alpha=0.1`, `reg_lambda=1.0`
- **`sample_weight` balanceado** (crítico — sin pesos colapsa a clase mayoritaria)

## Rendimiento (19 features)

| Clase | Rec_riesgo | F1_macro | Accuracy |
|-------|-----------|----------|----------|
| **Calor** | **0.614** | 0.546 | 0.783 |
| Frío | 0.460 | 0.506 | 0.834 |

En calor gana a RF en toda métrica (Rec_riesgo 0.614 vs 0.599). En frío tiene mejor accuracy pero peor detección de riesgo — se deja como alternativa.

## Fortalezas
- Mayor capacidad que RF: capta interacciones no lineales complejas
- Regularización incorporada (L1/L2) para evitar sobreajuste
- Eficiente y escalable

## Condición crítica
**XGBoost sin `sample_weight` balanceado produce recall de peligro 0.00–0.19** — inservible como sistema de aviso. El modelo por defecto predice "seguro" siempre.

## Referencias
- Chen, T. & Guestrin, C. (2016). XGBoost: A Scalable Tree Boosting System. *KDD*.
- Implementación: `xgboost.XGBClassifier`
