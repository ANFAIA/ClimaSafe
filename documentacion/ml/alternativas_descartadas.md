# Alternativas exploradas y descartadas

## ARIMA / SARIMA / Prophet

**Qué son:** Modelos estadísticos clásicos de series temporales univariantes.
ARIMA modela autocorrelación y tendencia; SARIMA añade estacionalidad;
Prophet (Meta) descompone en tendencia + estacionalidad + efectos
festivos con cambio de punto de cambio (changepoints).

**Por qué se descartaron:**

1. **Univariantes por naturaleza**: ARIMA/SARIMA/Prophet modelan una
   sola serie temporal. Nuestro problema es multivariante: necesitamos
   temperatura, humedad, viento, presión, Heat Index, WBGT, Wind Chill
   y sus lags simultáneamente. Alimentarlos con una sola variable
   (ej. solo t2m) ignora el resto de la señal.

2. **No manejan tabular + secuencia**: El problema no es puramente
   temporal — hay features estáticas (provincia, umbrales fijos),
   features derivadas (lags, rachas, grados-día) y la relación entre
   ellas. Los modelos estadísticos clásicos esperan una serie limpia
   y regular, no un vector heterogéneo de 27 features.

3. **No capturan picos extremos**: ARIMA asume normalidad en los
   errores; Prophet modela outliers como "changepoints" pero no
   aprende qué combinación de condiciones meteorológicas produce un
   día extremo. En nuestras pruebas, ambos predecían valores
   suavizados (media de la serie) y no acertaban los días de
   `peligro`.

4. **Sin interacciones no lineales**: Una ola de calor no es solo
   "temperatura alta" — es la combinación de t2m alta + humedad alta +
   viento bajo + racha de días previos calurosos. ARIMA no modela
   interacciones entre variables; Prophet solo acepta regresores
   externos lineales.

5. **Rendimiento inferior en pruebas**: Se probaron ARIMA(5,1,0),
   SARIMA(1,1,1)(1,1,1,7) y Prophet con regresores externos sobre
   datos agregados por provincia. Ninguno superó un `Rec_riesgo` >0.15
   — predecían "seguro" casi siempre. Comparado con XGBoost
   (Rec_riesgo ~0.63 en calor) o RandomForest (~0.53 en frío), la
   diferencia fue abismal.

**Conclusión:** Modelos útiles para forecasting univariante clásico
(demanda eléctrica, tráfico), pero no para clasificación de riesgo
multifactorial con features heterogéneas y no linealidades.

---

## KNN

**Problema:** Recall de riesgo ~0.02–0.26 en ambas clases. Colapsa a
la clase mayoritaria. Descartado en favor de LightGBM como baseline
(RF/XGBoost ya lo superaban ampliamente). Detalle en
`conclusiones_modelos.md`.

---

## LightGBM

**Evaluado como candidato** a tercer modelo (sustituyendo a KNN).
Rinde a la par de XGBoost en calor y peor en frío bajo la métrica
`Rec_riesgo`. Se mantiene como referencia en
`tuning/baseline_lightgbm.py` pero no desbanca a los desplegados.
Detalle en `conclusiones_modelos.md`.

---

## Multicalibración (Hebert-Johnson et al. 2018)

**Qué es:** Algoritmo post-hoc que ajusta predicciones para que estén
calibradas en cada subgrupo (edad, provincia, etc.). Disponible como
`pip install multicalibration`.

**Por qué no se implementó:** Es binario (riesgo sí/no). Habría que
binarizar el problema de 3 clases o adaptar el algoritmo. Se optó por
conformal prediction (que sí es multiclase nativo) más thresholds
manuales por edad (`t1=0.25, t2=0.40`).

---

## Transfer Learning Estructurado (CORE-Cox, Yu et al. 2026)

**Qué es:** Modelo fuente entrenado con todos los datos + adaptación
residual para cada subgrupo. Aprende estructura compartida de bajo
rango y permite especialización por estrato con pocos datos.

**Por qué no se implementó:** Requería reescribir la arquitectura
completa del modelo. El enfoque actual (modelo único + factores de
riesgo dinámicos + personalización por perfil) logra un efecto
similar sin cambiar la arquitectura.

---

## Bayes Jerárquico + Extreme Value Theory

**Qué es:** Modelo donde cada estrato tiene su propio riesgo que se
contrae hacia la media poblacional (shrinkage). EVT modela la cola
de la distribución (días extremos).

**Por qué no se implementó:** Requería migrar a PyMC/Stan y
reformular todo el pipeline probabilísticamente. El coste de
implementación no justificaba la ganancia frente al enfoque actual.

---

## Graph Neural Networks (GNN)

**Qué es:** Redes que operan sobre grafos. Los factores de riesgo
forman un grafo natural (diabetes es comorbilidad de calor y frío,
antipsicóticos se relaciona con salud mental, etc.).

**Por qué no dio resultado:** Se probaron GCN y GAT sobre el grafo
de factores. Las representaciones aprendidas no mejoraban la
clasificación frente a los features planos. El grafo de factores es
pequeño (~20 nodos) y las relaciones son mayoritariamente conocidas
a priori — una GNN no aporta valor sobre la ingeniería manual de
features. Se descartó.

---

## Temporal Fusion Transformer (TFT) / N-BEATS

**Qué son:** Modelos de forecast multi-horizonte. TFT añade atención
interpretable con variables estáticas/conocidas/futuras. N-BEATS usa
bloques fully-connected con descomposición residual.

**Por qué no se adaptaban:** Nuestro problema no es forecast
(predicti varios días ahead), sino clasificación del día siguiente
con features del día actual + lags. La arquitectura seq2seq de TFT
y N-BEATS está diseñada para horizonte múltiple con decodificación
autorregresiva, que aquí sobra. El LSTM híbrido cumple mejor al
combinar la dimensión temporal con features estáticas en una
arquitectura más simple y ajustada al problema.

---

## RL híbrido (recomendador de acciones)

**Qué es:** Un agente que aprende a recomendar acciones al usuario
("hidratarse cada 20 min", "buscar sombra") optimizando recompensa
acumulada (riesgo reducido).

**Por qué se dejó fuera:** Se planteó un enfoque híbrido (RL + reglas
fijas) para que el agente no tuviera que aprender desde cero
conductas básicas de sentido común. Sin embargo, el tiempo de
entrenamiento para converger a una política estable era demasiado
alto, y el espacio de estados/acciones requería un simulador
realista de la interacción usuario-entorno que no existía.
Queda como idea a largo plazo para cuando el sistema tenga
suficiente histórico de recomendaciones y feedback.
