# Ablación de features: 27 vs 19 con el mismo label

**Fecha:** 2026-07-13
**Script:** `tuning/ablacion_27v19.py` (reproducible; escribe `reports/ablacion_27v19.csv`)

## Motivación

En la iteración de hoy se cambiaron **dos cosas a la vez**: el conjunto de
features (de 19 a 27, añadiendo 8 nuevas) y el label (se añadió un suelo de
mortalidad a `clase_riesgo`). Con ambos cambios simultáneos, la mejora
observada en `Rec_riesgo` no era atribuible: podía venir de las features, del
label, o de ambos.

Este experimento **fija el label** (el nuevo, con suelo de mortalidad) y
compara, para cada clase (calor, frío) y cada modelo (XGBoost, RandomForest),
el entrenamiento con las 27 features frente a solo las 19 antiguas.

## Diseño

- Mismo split temporal (`data/processed/X_{train,test}_{calor,frio}.csv`,
  labels nuevos en `y_{train,test}_{calor,frio}.csv`).
- Mismos hiperparámetros que los modelos de hoy: se clonan
  `models/XGBoost_{clase}.joblib` y `models/RandomForest_{clase}.joblib` con
  `sklearn.base.clone` (sin reutilizar sus predicciones).
- XGBoost entrenado con `sample_weight = compute_sample_weight('balanced')`;
  RandomForest con `class_weight='balanced'` en sus propios params.
- Las 8 features nuevas: `heat_index_c_roll3`, `heat_index_c_roll7`,
  `dias_consec_sobre_umbral`, `grados_dia_calor_roll7`,
  `grados_dia_calor_roll14`, `wind_chill_mean_roll14`,
  `grados_dia_frio_roll7`, `grados_dia_frio_roll14`.
- Métrica de selección: `Rec_riesgo` = recall macro de las clases
  1 (precaución) y 2 (peligro). Política del proyecto: preferir falsos
  positivos.

Verificación de reproducción: las variantes 27f reproducen las referencias de
hoy con desviaciones ≤ 0.008 (XGB calor 0.6323 vs 0.6331 de referencia; RF
calor 0.6122 vs 0.6136; RF frío 0.5257 vs 0.5256; XGB frío 0.4477 vs 0.4558,
la mayor diferencia, atribuible al no determinismo de XGBoost con
entrenamiento paralelo). Las comparaciones 27f vs 19f de este experimento se
hicieron en condiciones idénticas, así que son directamente comparables entre
sí.

## Resultados (label nuevo fijo)

| Clase | Modelo | Variante | Rec_riesgo | F1_macro | Acc_test | Prec_c1 | Rec_c1 | Prec_c2 | Rec_c2 |
|-------|--------|----------|-----------:|---------:|---------:|--------:|-------:|--------:|-------:|
| calor | XGBoost | **27f** | **0.6323** | 0.5627 | 0.7934 | 0.2040 | 0.5773 | 0.3860 | 0.6874 |
| calor | XGBoost | 19f | 0.6255 | 0.5534 | 0.7862 | 0.1981 | 0.5656 | 0.3681 | 0.6854 |
| calor | RandomForest | **27f** | **0.6122** | 0.5410 | 0.7724 | 0.1773 | 0.5421 | 0.3664 | 0.6823 |
| calor | RandomForest | 19f | 0.6102 | 0.5328 | 0.7659 | 0.1767 | 0.5426 | 0.3445 | 0.6778 |
| frío | XGBoost | 27f | 0.4477 | 0.5149 | 0.8456 | 0.1892 | 0.4839 | 0.2948 | 0.4116 |
| frío | XGBoost | **19f** | **0.4877** | 0.5226 | 0.8399 | 0.1914 | 0.5128 | 0.3004 | 0.4625 |
| frío | RandomForest | 27f | 0.5257 | 0.5109 | 0.8109 | 0.1622 | 0.5596 | 0.3020 | 0.4917 |
| frío | RandomForest | **19f** | **0.5456** | 0.5155 | 0.8056 | 0.1580 | 0.5708 | 0.3202 | 0.5204 |

Deltas de las 8 features nuevas (27f − 19f), con label fijo:

| Clase | Modelo | Δ Rec_riesgo | Δ F1_macro |
|-------|--------|-------------:|-----------:|
| calor | XGBoost | **+0.0069** | +0.0093 |
| calor | RandomForest | +0.0020 | +0.0082 |
| frío | XGBoost | **−0.0399** | −0.0077 |
| frío | RandomForest | **−0.0199** | −0.0046 |

## Interpretación

**¿Aportan las 8 features por sí mismas con el label fijo?**
Depende de la clase, y la respuesta honesta es: poco en calor y **negativo en
frío**.

- **Calor: sí, pero de forma modesta.** Con el mismo label, las 8 features
  nuevas suman +0.007 de Rec_riesgo en XGBoost y +0.002 en RandomForest
  (ganancias también en F1_macro y en precisión de la clase 2). Es una mejora
  real pero pequeña; la mayor parte del salto observado hoy en calor respecto
  a la iteración anterior (XGB 0.6138 → 0.6331) viene del **cambio de label**,
  no de las features: con 19 features y el label nuevo, XGB calor ya alcanza
  0.6255 (frente al 0.6138 del label antiguo).

- **Frío: no, las empeoran.** Con el label fijo, añadir las 8 features
  **reduce** Rec_riesgo en ambos modelos: XGB frío cae de 0.4877 (19f) a
  0.4477 (27f) y RF frío de 0.5456 a 0.5257. La caída se concentra en el
  recall de ambas clases de riesgo (RF: Rec_c2 0.5204 → 0.4917). Es llamativo
  porque varias de las nuevas features se diseñaron precisamente para frío
  (`grados_dia_frio_roll7/14`, `wind_chill_mean_roll14`): en la práctica
  añaden ruido o redundancia que degrada la señal con estos hiperparámetros.
  Además, el RF frío de hoy (27f, 0.5256) **no supera** al RF frío de la
  iteración anterior (19f + label antiguo, 0.5273); con 19f y el label nuevo
  habría llegado a 0.5456.

**Conclusión operativa:**

1. La mejora de hoy en calor es atribuible sobre todo al nuevo label; las 8
   features aportan un extra pequeño pero consistente en calor. Mantener las
   27 features para los modelos de calor es razonable.
2. Para frío, la mejor configuración conocida con el label nuevo es **19
   features**, no 27. Recomendación: usar conjuntos de features por clase
   (27f calor / 19f frío), o hacer una ablación más fina en frío para
   identificar cuál(es) de las 8 nuevas degradan (candidatas: las roll14, por
   solapamiento con las roll3/roll7 ya existentes).
3. Estos deltas se midieron con los hiperparámetros de hoy sin re-tunear; un
   re-tuning por variante podría mover los números, pero el signo de la
   diferencia en frío (consistente en dos modelos distintos) difícilmente se
   revierte solo con tuning.

## Reproducción

```bash
PYTHONPATH=. .venv/bin/python tuning/ablacion_27v19.py
# opciones: --data-dir, --models-dir, --out (por defecto reports/ablacion_27v19.csv)
```
