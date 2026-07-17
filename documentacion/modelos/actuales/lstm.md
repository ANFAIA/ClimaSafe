# LSTM — modelo explorado (híbrido tabla+serie)

## Rol
Arquitectura híbrida que **combina features tabulares (t2m, humedad, viento...) con la secuencia temporal** de los 7 días previos. Explorada como alternativa a los modelos tabulares puros (RF, XGBoost).

## Arquitectura
- **Rama tabla**: capas densas sobre las 19 features del día objetivo
- **Rama secuencia**: LSTM(s) sobre ventana de 7 días (features meteorológicas diarias)
- **Fusión**: concatenación + capa densa final → clasificación 3 clases
- Implementación: `tensorflow.keras` con `class_weight` para balance

## Rendimiento comparativo

| Modelo | Rec_riesgo calor | Rec_riesgo frío | Accuracy |
|--------|-----------------|-----------------|----------|
| XGBoost (calor) / RF (frío) | **0.614** | **0.527** | 0.78 |
| LSTM híbrida | inferior | inferior | similar |

La LSTM no superó a los modelos tabulares con lags. La hipótesis es que la señal temporal ya queda capturada por las features de persistencia (`roll3/7/14`, lags) que son, de hecho, el mayor salto en rendimiento.

## Fortalezas (potencial)
- Modela la secuencia de forma endógena (sin ingeniería manual de lags)
- Podría capturar patrones temporales que escapan a las ventanas fijas

## Debilidades
- Mayor coste computacional de entreno e inferencia
- Menor interpretabilidad
- No superó a XGBoost/RF con lags bien diseñados
- Requiere GPU para entrenar eficientemente

## Referencias
- Hochreiter, S. & Schmidhuber, J. (1997). Long Short-Term Memory. *Neural Computation*, 9(8):1735–1780.
- Implementación: `tensorflow.keras.layers.LSTM`
- Análisis completo: `documentacion/ml/lstm_hibrida.md`
