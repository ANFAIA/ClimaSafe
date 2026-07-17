# LSTM híbrida — contexto de ola para la LSTM multi-tarea

Este documento recoge la motivación, el diseño y los resultados de la
variante **híbrida** de la LSTM multi-tarea (`climasafeai/models/lstm_hybrid.py`),
y el veredicto frente a la LSTM base y los modelos desplegados.

---

## 1. Motivación: la LSTM pura no sabe en qué día de la ola está

La LSTM multi-tarea (`lstm_model.py`, ver `../arquitectura/diseño_modelo.md` §6) clasifica
cada día viendo SOLO su secuencia de 24 horas (t2m, rh, viento, Heat Index,
Wind Chill). Eso le da lo que los modelos tabulares no ven — el perfil
intradía completo, incluido el alivio nocturno — pero le quita lo que
aquellos sí ven: la **persistencia**. Fisiológicamente, el 5º día
consecutivo de una ola de calor mata más que el 1º con la misma secuencia
horaria (agotamiento térmico acumulado, saturación hospitalaria), y esa
información vive en las features diarias del pipeline tabular:
`dias_consec_sobre_umbral`, `grados_dia_calor_roll7/14`,
`heat_index_c_roll3/7`, `wind_chill_mean_roll3/7/14`,
`grados_dia_frio_roll7/14`, `dias_consec_bajo_umbral`.

## 2. Arquitectura

`LSTMHybridMultiTask` (checkpoint propio `models/LSTM_hybrid.pt`, no toca
`LSTM_multitask.pt`):

```
secuencia 24h (24 × 5) ──> LSTM (2 capas, hidden 64) ──> h_T (64)
                                                           │ concat
vector diario (27) ── StandardScaler(train) ───────────────┤
                                                           ▼
                                       fusión Linear(91→64) + ReLU + Dropout
                                                           │
                                        ┌──────────────────┴──────────────────┐
                                   cabeza calor (3 clases)            cabeza frío (3 clases)
```

Decisiones:

- **Vector diario = las 27 features tabulares** de
  `dataset_calor_labeled.parquet` (las mismas que ven XGBoost/RF), alineadas
  con cada secuencia por `(fecha, provincia)` (`alinear_features_diarias`,
  merge validado 172 350/172 350). Se usan las 27 y no solo las de
  persistencia porque las agregadas diarias (heat_index_mean/std/min,
  horas_sobre_umbral...) son gratis una vez hecho el merge y la capa de
  fusión puede ignorarlas si no aportan.
- **Escalado sin fugas**: StandardScaler propio ajustado SOLO con train
  (`escalar_diarias`, artefacto `scaler_diarias_lstm_hybrid.joblib`). Los
  únicos NaN (45 filas de rollings del 2016-01-01, primer día del
  histórico, todas en train) se imputan a la media de train (0 tras
  estandarizar).
- **Capa de fusión** Linear+ReLU+Dropout tras la concatenación: sin ella,
  cada cabeza solo podría combinar linealmente los dos bloques.
- **Mismo protocolo que la LSTM base**: split temporal por fecha (test 20%,
  val 10%), pesos de clase `balanced`, CE(calor)+CE(frío), Adam 1e-3,
  batch 256, early stopping por val loss (patience 5), seed 42, selección
  final por Rec_riesgo.

## 3. Resultados (test temporal, últimas ~20% de fechas)

Split: train hasta 2023-07-20 | val 2023-07-21..2024-05-21 | test desde
2024-05-22 (766 días distintos). Early stopping en época 12 (mejor
val_loss 1.5458). Métrica de selección del proyecto: **Rec_riesgo** (recall
macro de las clases de riesgo 1 y 2).

### Calor

| Modelo | F1_macro | **Rec_riesgo** | Acc_test | F1_test |
|---|---|---|---|---|
| LSTM base (24h) | 0.4085 | 0.4939 | 0.6774 | 0.7476 |
| **LSTM híbrida** | **0.5206** | **0.5971** | 0.7596 | 0.8101 |
| XGBoost (desplegado) | 0.5170 | 0.6331 | 0.8452 | 0.8733 |

### Frío

| Modelo | F1_macro | **Rec_riesgo** | Acc_test | F1_test |
|---|---|---|---|---|
| LSTM base (24h) | 0.3972 | 0.4647 | 0.6741 | 0.7597 |
| **LSTM híbrida** | **0.4706** | **0.5141** | 0.7680 | 0.8256 |
| RandomForest (desplegado) | 0.5117 | 0.5256 | 0.8106 | 0.8537 |

Reproducible con `python -m climasafeai.models.lstm_hybrid`; resumen en
`reports/resultados_lstm_hybrid.csv`, matrices de confusión
`cm_LSTM_hybrid_{calor,frio}.png` en `reports/figures/`.

## 4. Veredicto

**El contexto de ola funciona.** Concatenar las 27 features diarias sube el
Rec_riesgo frente a la LSTM base en **+0.103 en calor** (0.4939 → 0.5971) y
**+0.049 en frío** (0.4647 → 0.5141), con mejoras paralelas en F1_macro
(+0.112 calor, +0.073 frío) y sin degradar accuracy — al contrario, la
híbrida sube ~8 puntos de accuracy en ambas cabezas. Confirma la hipótesis
de `../arquitectura/diseño_modelo.md` §6: la LSTM pura pierde la persistencia que capturan
los rollings diarios, y devolvérsela cierra buena parte de la brecha.

**Frente a los modelos desplegados**, la híbrida sigue por debajo pero ya
compite: en frío queda a 0.011 de Rec_riesgo del RandomForest (0.5141 vs
0.5256) — prácticamente empate dentro del ruido del split; en calor recorta
la distancia con XGBoost a la mitad (0.5971 vs 0.6331) desde el hueco que
tenía la LSTM base. La ventaja de los árboles sobre calor persiste, pero ya
no es categórica.

**Recomendación.** La híbrida es la LSTM que merece la pena mantener: mismo
coste de inferencia (57.7k parámetros, CPU), aporta la lectura intradía que
los árboles no ven y ahora también la persistencia que les faltaba. Como
cuarta estimación del ensemble (la más restrictiva, §6) es más fiable que
la LSTM pura. No sustituye a XGBoost/RF en producción todavía; el siguiente
paso natural para intentar superarlos es la ventana de 48-72 h (§5), que se
deja documentada como propuesta por su coste de regeneración de datos.

No se re-tuneó el peso de clase: los pesos `balanced` ya baten claramente
la LSTM base y forzar el recall de riesgo (peso extra sobre clases 1/2)
solo intercambiaría precisión por recall sin cambiar el veredicto.

## 5. Propuesta futura: secuencias de 48-72 h (no implementada)

La alternativa "pura" al híbrido sería alargar la ventana de la LSTM a
48-72 horas para que la propia red vea la persistencia en crudo. **No se ha
implementado**: exige regenerar las secuencias desde los ~5.5 GB de .nc de
ERA5 (`generar_dataset_secuencias`), un coste alto, y aun con 72 h la red
solo vería 3 días hacia atrás — menos memoria que un rolling de 14 días,
con 3× más cómputo por muestra y gradientes más largos. Si se retoma:
regenerar el npz con ventana parametrizable, mantener el mismo split
temporal y comparar contra este híbrido, que es el baseline a batir.
