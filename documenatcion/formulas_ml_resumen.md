# Fórmulas + ML vs. aprendizaje directo de secuencias — resumen

> Versión corta de referencia. Para el desarrollo completo con analogías,
> estudios citados y recursos en vídeo, ver el historial de conversación
> o pedir la versión extendida.

## La idea en una frase

**Feature engineering** (lo que ya se usa: Heat Index, Wind Chill): se
aplica una fórmula fija y validada, y el modelo recibe el resultado ya
calculado. **Aprendizaje directo sobre secuencias** (LSTM): se le da al
modelo la serie horaria cruda sin resumir, y el propio entrenamiento
descubre qué patrón temporal predice mejor el riesgo.

## Ejemplo concreto

```
RandomForest (actual):  1 fila/día → Heat Index de la hora de mayor riesgo
LSTM (propuesto):       1 fila/día → secuencia de 24 horas (temp/humedad/viento)
Label (igual en ambos): clase_riesgo_calor / clase_riesgo_frio (de MoMo)
```

## Por qué interesa explorarlo: el alivio nocturno

El Heat Index de la hora pico solo mira el momento más caluroso del día.
Ignora si la noche refrescó lo suficiente para que el cuerpo se recupere
— un factor de mortalidad por calor documentado de forma independiente al
pico diurno (estudios en Japón, Londres y multi-país citados en la
versión extendida). Una LSTM con las 24h completas tiene la oportunidad
de aprender ese patrón; el Heat Index de pico lo descarta por diseño.

## Modelos totales si se implementa

| | RandomForest (ya existe) | LSTM (propuesto) |
|---|---|---|
| Calor | Modelo 1 | Modelo 3 |
| Frío  | Modelo 2 | Modelo 4 |

## Input de la LSTM

`(n_días, 24, n_variables)` — recomendado `n_variables=5`
(`t2m_c`, `rh`, `wind_speed_kmh`, `heat_index_c`, `wind_chill_c` por
hora, no solo de la hora pico). El label sigue siendo 1 por día.

## Trade-off en una tabla

| | Fórmula (RandomForest) | Secuencia cruda (LSTM) |
|---|---|---|
| Datos necesarios | Menos | Muchos más |
| Interpretabilidad | Alta ("HI=42°C → peligro") | Baja (caja negra) |
| Descubre patrones no previstos | No | Sí (p.ej. alivio nocturno) |
| Sesgo heredado de la fórmula (EEUU) | Sí | No |

## ¿Son suficientes 10 años de datos?

Para RandomForest, sí. Para LSTM, con reservas: los días de "peligro" no
son eventos independientes, son ~40-60 olas de calor reales generando
filas correlacionadas entre sí (mismos días, provincias vecinas). La
diversidad real de patrones que la LSTM puede aprender es menor de lo que
sugiere el conteo bruto de filas.

## Recomendación práctica

Tratar la LSTM como un **tercer modelo experimental**, no como sustituto.
Comparar sus métricas contra RandomForest/XGBoost con **split train/test
por fechas** (no aleatorio — un split aleatorio infla artificialmente el
rendimiento aparente de modelos basados en vecinos/similitud, al mezclar
días de la misma ola de calor entre train y test).
