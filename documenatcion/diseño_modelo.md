# Diseño del modelo — ClimaSafe

Este documento recoge el razonamiento técnico detrás de las decisiones de
arquitectura del modelo. El README explica *qué* hace el sistema; este
documento explica *por qué* se diseñó así, para poder retomar el proyecto
más adelante sin perder el hilo de las decisiones.

---

## 1. Dos modelos separados (calor / frío), no uno único

MoMo distingue explícitamente muertes por calor (X30) y por frío (X31).
Si se entrenara un único modelo con un único label combinado, el modelo
no podría distinguir si un día en Burgos en enero con 3 muertes por frío
"significa lo mismo" que un día en Sevilla en agosto con 3 muertes por
calor — son fenómenos físicos opuestos con features relevantes distintas
(la ausencia de viento agrava el calor; la presencia de viento agrava el
frío).

Nota de alcance: al calibrarse las 3 clases sobre percentiles de exceso
de mortalidad de MoMo, el ML predice específicamente **riesgo de exceso
de mortalidad**, no morbilidad o riesgo general — un día puede ser
individualmente peligroso sin generar exceso de mortalidad a nivel
provincial (por eso la fórmula del punto 5 sigue siendo necesaria).

Por eso se entrenan dos `RandomForestClassifier` independientes:

| | Modelo Calor | Modelo Frío |
|---|---|---|
| Label | Muertes X30 (MoMo) | Muertes X31 (MoMo) |
| Features principales | Heat Index, humedad, UV, ausencia de viento | Wind Chill, temperatura, presencia de viento |
| Ventana horaria | Hora con Heat Index máximo del día | Hora con Wind Chill mínimo del día |

En producción, ambos modelos corren siempre en paralelo, junto con la
fórmula determinista (ver sección 5).

---

## 2. RiskScore artificial como selector de ventana horaria (no como feature)

ERA5 proporciona datos meteorológicos con resolución horaria, mientras que MoMo
ofrece datos agregados por día. Esto impide conocer la hora exacta en la que
se produjo el evento sanitario asociado al registro diario.
Si se usaran todas las horas del día como filas de entrenamiento, el modelo
aprendería correlaciones falsas (ej. un pico de calor a las 15h y
viento fresco a las 20h del mismo día que tuvo muertes — las features
de las 20h no deberían asociarse a ese label).

Se selecciona una única hora representativa de cada día utilizando un **RiskScore temporal**, cuyo único propósito es identificar el momento de mayor exposición al estrés térmico.

Este RiskScore se construye a partir de índices meteorológicos validados, como el **Heat Index** (estrés por calor) y el **Wind Chill** (estrés por frío), complementados con otras variables relevantes como la radiación UV. Su función no es estimar el riesgo sanitario, sino actuar como un **criterio proxy** para localizar la hora de mayor severidad meteorológica del día.

**Solución (pipeline de entrenamiento):**
1. Para cada hora del día, calcular un RiskScore provisional =
   `max(HeatIndex, WindChill, UV)`.
2. Identificar `hora_peligro = risk_hora.idxmax()`.
3. Extraer las features meteorológicas de ERA5 correspondientes
   exactamente a esa hora.
4. **Borrar el RiskScore provisional** — nunca entra como feature del
   modelo. Solo sirvió para seleccionar qué hora usar.
5. El label sigue viniendo de MoMo (muertes reales por provincia y día),
   no del RiskScore artificial.

Esto evita data leakage: el modelo aprende de variables meteorológicas
reales del momento de mayor riesgo, no de un score que ya "sabe" la
respuesta.

---

## 3. Fuentes de datos usadas para el entrenamiento del ML

| Fuente | Descripción | Datos que ofrece | API Key / Límites | Web | Documentación |
|--------|-------------|------------------|-------------------|-----|---------------|
| **ERA5 (ECMWF / Copernicus)** | Reanálisis climático y meteorológico global con datos horarios desde 1940. | Temperatura, precipitación, presión, viento, humedad, radiación solar, nubosidad y variables atmosféricas y oceánicas. | Requiere **registro gratuito** en Copernicus Climate Data Store (CDS). Sin límite de peticiones publicado. | [Copernicus CDS](https://cds.climate.copernicus.eu/) | [ERA5 docs](https://confluence.ecmwf.int/display/CKB/ERA5) |
| **AEMET OpenData** | API oficial de la Agencia Estatal de Meteorología con observaciones, predicciones y datos climatológicos de España. | Predicciones meteorológicas, observaciones de estaciones, climatología, avisos meteorológicos, radar, rayos e información oficial de España. | **Requiere API Key** gratuita. Sin límites de peticiones documentados. | [AEMET](https://opendata.aemet.es/) | [Docs AEMET](https://opendata.aemet.es/centrodedescargas/inicio) |
| **Open-Meteo API** | API gratuita para predicciones meteorológicas y datos históricos. | Predicción meteorológica, datos históricos, calidad del aire, índice UV, radiación solar, elevación y modelos climáticos. | **Sin API Key.** Hasta **10.000 llamadas/día**, **5.000/hora**, **600/minuto**. | [Open-Meteo](https://open-meteo.com/en/docs) | [Open-Meteo docs](https://open-meteo.com/en/docs) |
| **OpenUV** | API para consultar el índice UV y exposición solar. | Índice UV actual, histórico y previsto, tiempos de exposición segura e intensidad solar. | **Requiere API Key.** Plan gratuito con **50 solicitudes/día**. | [OpenUV](https://www.openuv.io/) | [OpenUV docs](https://www.openuv.io/) |
| **MoMo (ISCIII)** | Sistema de monitorización de mortalidad diaria en España. | Mortalidad observada, esperada, exceso de mortalidad, desagregación por edad, sexo y región. | **Sin API Key.** Descarga directa de datos. | [MoMo](https://momo.isciii.es/public/momo/data) | [MoMo dashboard](https://momo.isciii.es/public/momo/dashboard) |

ERA5 url api: https://cds.climate.copernicus.eu/api/v2

AEMET url api: https://opendata.aemet.es/opendata/api/prediccion/especifica/municipio/diaria/{municipio}

OpenUV url api: https://api.openuv.io/api/v1/uv?lat=:lat&lng=:lng&alt=:alt&dt=:dt


---

## 4. Justificación de la elección de Random Forest

El modelo principal seleccionado es **Random Forest**, utilizando **Logistic Regression** como modelo *baseline* para establecer una referencia de rendimiento.

La elección de Random Forest responde a las siguientes características del problema:

- **Interpretabilidad:** permite analizar la importancia de cada variable mediante métricas de *feature importance*, facilitando identificar qué factores meteorológicos influyen más en la predicción del riesgo.

- **Relaciones no lineales:** el riesgo asociado a fenómenos meteorológicos extremos no sigue relaciones lineales simples. Random Forest es capaz de modelar interacciones complejas entre variables como temperatura, humedad, velocidad del viento, radiación UV o Heat Index sin necesidad de definir dichas relaciones manualmente.

- **Robustez con datasets de tamaño moderado:** el conjunto de entrenamiento está formado por datos diarios históricos, por lo que no dispone de millones de ejemplos. Random Forest suele ofrecer un buen rendimiento en este tipo de escenarios sin requerir grandes volúmenes de datos.

- **Resistencia a variables correlacionadas:** muchas variables meteorológicas presentan correlación entre sí (por ejemplo, temperatura y Heat Index). Aunque la correlación sigue siendo un aspecto a analizar durante el preprocesado, Random Forest suele ser menos sensible a este problema que otros algoritmos.

- **Bajo ajuste de hiperparámetros:** ofrece un buen equilibrio entre rendimiento y simplicidad, reduciendo el tiempo de experimentación respecto a modelos que requieren un ajuste mucho más exhaustivo.

---

## 5. Justificación: modelo híbrido (ML + fórmula determinista)

Se opta por un modelo híbrido en lugar de un ML puro. La fórmula de riesgo
(Heat Index, Wind Chill, UV) es determinista: dado un input, el resultado
siempre es el mismo, por lo que no requeriría machine learning. Sin embargo,
una fórmula aislada evalúa cada variable por separado y no captura
interacciones peligrosas entre ellas — el caso canónico es el Pirineo en
primavera, donde frío extremo (riesgo de hipotermia) y UV alto en altitud
(riesgo de quemadura) ocurren simultáneamente sin que ninguna fórmula
aislada lo detecte como una emergencia combinada.

Dado que MoMo solo proporciona un dato agregado diario de muertes por
provincia (ver sección 3 — CMBD descartado por cobertura insuficiente),
se entrenan dos modelos de clasificación de tres clases
(Seguro=0, Precaución=1, Peligro=2) separados — uno para calor y otro
para frío. Para cada provincia y día del histórico se calcula el Heat
Index a partir de ERA5, se identifica la hora de mayor riesgo, y esa hora
se cruza con las muertes por temperaturas altas (modelo calor) o bajas
(modelo frío) de ese mismo día y provincia.
A partir de estas series horarias se selecciona **la hora de mayor riesgo del día**, entendida como aquella en la que el valor de riesgo combinado es máximo.
El riesgo combinado se define como:

- el máximo entre los índices normalizados de calor, frío y radiación UV.

Esto permite evitar inconsistencias temporales, ya que no se mezclan condiciones de distintas horas en un mismo registro.
El resultado final de cada día es un único vector representativo:

- Provincia
- Fecha
- Hora de mayor riesgo
- Variables meteorológicas en esa hora
- Índices derivados en esa hora

Este vector es el que se utiliza posteriormente para cruzarlo con los datos de mortalidad de MoMo.


**Qué cubre el ML y qué no:** el ML (MoMo) aprende el patrón poblacional
real de España — cuándo unas condiciones meteorológicas se traducen
estadísticamente en exceso de mortalidad. Es precisión *empírica*, no
teórica: si el modelo aprende que la mortalidad por calor en España
empieza a dispararse en un umbral distinto al teórico NIOSH/OIT, eso es
conocimiento nuevo (ver PDF original, sección de coeficientes SHAP).

Lo que el ML por sí solo **no puede** cubrir bien:
- Casos raros de combinación extrema (Pirineo, Himalaya: frío + UV
  simultáneos), donde apenas hay datos de entrenamiento en MoMo.
- Riesgo a nivel **individual**: MoMo es un agregado poblacional diario
  por provincia — no sabe nada del fototipo de piel de una persona
  concreta, ni de cuánto tiempo lleva expuesta. Esto es un techo
  estructural de la fuente, no un problema de que falte más histórico:
  aunque se hubiera podido usar CMBD actualizado, seguiría siendo un
  agregado de ingresos hospitalarios, no un cálculo por persona.

**Nota — asimetría muerte/ingreso:** si el ML da probabilidad alta de
PELIGRO, se infiere también riesgo de necesitar atención sanitaria (una
mortalidad extrema implica necesariamente un número aún mayor de casos
graves por debajo). Con probabilidad baja esa inferencia no es fiable
—de ahí que el sistema no dependa del ML en ese extremo, sino de la
fórmula, que sigue evaluando el riesgo individual real.

Por eso, en cada consulta se calculan siempre las tres estimaciones —
modelo ML de calor, modelo ML de frío, y la **fórmula científica
determinista** (Heat Index / Wind Chill / UV, con tiempo de exposición y
fototipo de piel — ver `formulas_riesgo_deterministico.md`) sobre los datos
meteorológicos del momento (Open-Meteo) — y se toma como resultado final
el criterio más restrictivo de las tres. La fórmula cubre exactamente el
hueco que el ML no puede cubrir por diseño (individual, universal, sin
depender de histórico de mortalidad), y actúa como red de seguridad
permanente, no como respaldo condicional: no depende de un clasificador
previo que decida cuándo confiar en el ML.

**Nota sobre precisión:** el ML y la fórmula no compiten en "cuál es más
preciso" en términos absolutos — miden cosas de naturaleza distinta. El
ML da precisión poblacional-empírica (validada contra muertes reales
españolas); la fórmula da precisión individual-teórica (validada contra
literatura clínica NWS/OMS, personalizable por fototipo y exposición).
Se combinan con la lógica del máximo precisamente porque son
complementarias, no sustitutas la una de la otra.

**Razón adicional para la fórmula, independiente de la disponibilidad de
CMBD:** un dato agregado diario de ingresos hospitalarios (como habría
sido CMBD) mezcla dos causas distintas de un pico de casos sin poder
separarlas — condiciones meteorológicas objetivamente peligrosas, o
un evento de exposición poblacional atípica ese día (un maratón, una
romería, una concentración masiva al aire libre). Un modelo ML entrenado
con ese agregado heredaría esa confusión (*confounding*): aprendería a
asociar ciertas condiciones con "riesgo alto" cuando el factor decisivo
real fue cuánta gente estaba expuesta ese día, no cuánto peor era el
clima. La fórmula no tiene este problema porque no depende de qué ocurrió
en agregado un día concreto — calcula el riesgo a partir de los inputs
reales del usuario en el momento de la consulta (actividad, MET, tiempo
de exposición, fototipo), sin ruido de eventos poblacionales ajenos a él.
Esto significa que la fórmula seguiría siendo necesaria como componente
del sistema incluso si se dispusiera de CMBD actualizado — no es solo un
sustituto temporal mientras no hay acceso a esa fuente.

Ver `formulas_riesgo_deterministico.md` para el detalle de implementación de la
fórmula (tablas Heat Index/Wind Chill del NWS, fórmula de tiempo hasta
eritema de la OMS por fototipo, y código de referencia).

---