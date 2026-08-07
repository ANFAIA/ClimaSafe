# Resolución de la predicción (DATA-007)

La resolución del perfil horario es elegible por el usuario: API, MCP, web y bot
la piden o usan **1 h por defecto**, y el contrato está preparado para ampliarse
a resoluciones de **segundos** (p. ej. 30 s para relojes inteligentes) sin romper
la salida.

## Qué hay hoy

- `predict_ensemble(lat, lon, provincia, perfil, target_date, weather, resolucion=60)`
  acepta `resolucion` en **minutos por punto** (5, 15, 30 o 60). Con 60 la salida
  es exactamente la de siempre: un punto por hora (identidad).
- El perfil horario se construye en `perfil_horario_desde_df(df, target_date, res_min)`
  (DATA-004): con 60 usa el HI máximo de cada hora; con menos, añade los puntos
  intermedios por **interpolación lineal** entre máximos horarios consecutivos.
- Capas que exponen el parámetro:
  - **API**: `POST /api/predict` acepta `resolucion` en el body y lo pasa a
    `predict_ensemble` (default 60). El frontend no lo manda: usa 1 h por defecto.
  - **MCP**: `predict_risk_mcp` y `grafica_riesgo_horario_mcp` aceptan
    `resolucion` (minutos) y lo pasan por `predict_risk` → `_try_prediction`.
  - **Bot Telegram**: usa el default 60 (1 h). Punto de extensión en
    `climasafeai/bot/telegram_bot.py`: las llamadas a `predict_ensemble` (parte
    diario y avisos de rutinas) aceptarían `resolucion=...` sin más; la pregunta
    al usuario ("¿con qué detalle quieres la curva?") sería una opción nueva del
    flujo de chat (BOT-020 u otra), no un cambio de contrato.

## El contrato de predicción no cambia

Independientemente de la resolución, la salida de `predict_ensemble` tiene los
mismos campos y tipos:

- `clase_final`, `clase_final_label`, `perfil`, `explicacion`, `recomendaciones`,
  `modelos`, `override_fisico`
- `weather.perfil_horario`: lista de `{"hora", "HI", "temp"}` — solo cambia el
  **número de puntos**, no el esquema de cada punto.

Lo mismo vale para las capas: `riesgo_horario`, `riesgo_pico` y
`recomendacion_horario` devuelven el mismo esquema con una malla más fina.

## Cómo se ampliará a segundos (30 s para wearables)

El plan no toca el contrato, solo el paso del perfil horario:

1. **El parámetro sigue siendo un entero con unidad explícita.** Hoy es minutos
   (`resolucion=60`). Para wearables se añadiría un parámetro nuevo o un prefijo
   de unidad — p. ej. `resolucion_s=30` — sin reutilizar `resolucion` para dos
   unidades: un `resolucion=30` ambiguo (¿minutos o segundos?) rompería la
   compatibilidad. Decisión recomendada: mantener `resolucion` en minutos y
   añadir `resolucion_s` (segundos) para los valores < 60 s, con validación de
   que ambos no lleguen a la vez.
2. **`perfil_horario_desde_df` construye con ese paso.** El bucle de
   interpolación ya es genérico (paso = `res_min / 60`); con segundos el paso es
   `res_s / 3600` y el anclaje deja de estar en `:00` exacto: para 30 s habría
   que decidir el offset (p. ej. malla `:00`, `:00:30`, `:01`, ...). El campo
   `hora` ya se guarda sin redondear (float) precisamente porque la resolución
   de 5 min no es decimal exacta; segundos entra en la misma mecánica.
3. **La fuente de datos marca el límite real.** Hoy Open-Meteo solo publica
   datos **horarios**: los puntos sub-horarios son interpolación, no medición.
   Una fuente sub-horaria (API del reloj inteligente: temperatura corporal,
   frecuencia cardíaca, o microclima local) **solo cambiaría la fuente de
   datos** que alimenta `df_hora`; el contrato de salida y la lógica de
   `predict_ensemble` no se tocan. La interpolación actual seguiría siendo el
   fallback cuando no haya datos del wearable.
4. **Rendimiento.** El coste de la predicción no escala con la resolución (los
   modelos se ejecutan una vez; solo el perfil horario tiene más puntos). Las
   curvas (`riesgo_horario`, gráfica PNG) sí crecen con el nº de puntos: para
   30 s = 2880 puntos/día convendrá pintar la gráfica agregando a la malla que
   pida el usuario, no a ciegas.
