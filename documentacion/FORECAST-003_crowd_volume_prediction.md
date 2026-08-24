# FORECAST-003 — Estudio: Predicción de volumen de gente para riesgo colectivo

**Feature:** FORECAST-003
**Fecha:** 2026-08-24
**Tipo:** Spike de investigación (no implementación)
**Estado:** Completado

---

## 1. Contexto

La feature CSV-001 ya implementa riesgo colectivo con un "factor de orgullo
colectivo" (`ORGULLO_COLECTIVO = 1.2`) que multiplica las odds del riesgo
individual cuando `tipo_actividad` es competición/deporte. El número de personas
del grupo lo aporta el usuario via CSV (`/api/riesgo-colectivo/csv`).

Este spike investiga si se puede **predecir** cuánta gente va a haber en un
lugar y momento dado, para alimentar automáticamente el riesgo colectivo sin
que el usuario tenga que introducir el número.

---

## 2. Fuentes reales de datos de volumen de gente — España

### 2.1 Datos abiertos de movilidad del MITMA

| Aspecto | Detalle |
|---------|---------|
| **Fuente** | Ministerio de Transportes y Movilidad Sostenible (MITMA) |
| **Datos** | Matrices origen-destino, número de viajes por persona, estancias nocturnas |
| **Granularidad** | Distritos, municipios, áreas urbanas mayores |
| **Temporalidad** | Diaria (desde enero 2022, continua) |
| **Cobertura España** | Nacional. ~30% cuota de mercado del operador (Orange España) |
| **Licencia** | Licencia de datos abiertos MITMA (reutilización libre con atribución) |
| **Coste** | Gratuito |
| **API / Descarga** | `movilidad-opendata.mitma.es` + paquetes `spanishoddata` (R) y `pySpainMobility` (Python) |
| **Horizonte predictivo** | Retrospectivo (D-1 a D-7). No hay forecasts oficiales |
| **Limitaciones** | Solo un operador; subrepresenta mayores y jóvenes. Granularidad zona, no punto |

**¿Sirve para predecir aforo de un evento concreto?** No directamente. Los
datos son agregados por zona y día, no por evento. No permiten decir "habrá
X personas en el estadio Santiago Bernabéu el sábado". Sí permiten estimar la
movilidad agregada de un municipio o distrito en un día dado.

### 2.2 Google Community Mobility Reports

| Aspecto | Detalle |
|---------|---------|
| **Fuente** | Google |
| **Datos** | Cambios porcentuales de movilidad vs. línea base, por categoría (retail, parques, transporte, etc.) |
| **Cobertura España** | Municipal y provincial |
| **Licencia** | Datos abiertos (CCBY) |
| **Coste** | Gratuito |
| **Estado** | **Descontinuado** — última actualización 15 octubre 2022. Histórico disponible |
| **Horizonte predictivo** | Histórico. Sin actualizaciones futuras |

**¿Sirve?** No. Está muerto. Los datos históricos sirven para calibrar
modelos, pero no para predicción en tiempo real.

### 2.3 Datos del INE (Instituto Nacional de Estadística)

| Aspecto | Detalle |
|---------|---------|
| **Fuente** | INE |
| **Datos** | Padrón, Censo, Encuesta de Población Activa, Padrón de viajeros |
| **Granularidad** | Municipal, comarcal, provincial |
| **Temporalidad** | Anual o bianual (no diaria) |
| **Cobertura España** | Nacional, censal |
| **Licencia** | CC-BY 4.0 |
| **Coste** | Gratuito |
| **API** | `ine.es/dyngs/DAB/index.htm` (JSON API) |
| **Horizonte predictivo** | Estático (población residente). No captura eventos puntuales |

**¿Sirve?** Solo como línea base de población residente. No predice aforos
de eventos.

### 2.4 Datos de movilidad por teléfonos móviles (CDR — Call Detail Records)

| Aspecto | Detalle |
|---------|---------|
| **Fuente** | Operadores móviles → INE (estadística experimental), MITMA, Facebook Data for Good |
| **Datos** | Matrices OD, estimación de población por celdas (5.000–50.000 hab) |
| **Cobertura España** | Nacional (3.214 celdas poblacionales INE) |
| **Licencia** | Datos experimentales INE (acceso restringido vía solicitud); MITMA abierto |
| **Coste** | Gratuito (vía MITMA/INE) |
| **Horizonte predictivo** | Retrospectivo (D-1). No hay forecasts |
| **Evidencia académica** | Osorio & de las Obras (2023, *Scientific Reports*): "useful source for demographic and mobility studies" con buena correlación con censo |

**¿Sirve para predicción?** Permiten estimar población "en el momento" en
una celda geográfica, pero con latencia (D-1) y sin granularidad de evento.
No predicen cuánta gente irá a un concierto futuro.

### 2.5 Sistemas de conteo en tiempo real (aforo IoT)

| Aspecto | Detalle |
|---------|---------|
| **Ejemplos** | ControlDeAforo.es (NeuralPax, acreditado CEM), IDASFEST |
| **Datos** | Conteo de personas en tiempo real por sensores IA, RFID, cámaras |
| **Precisión** | 95–99.9% (CEM homologado) |
| **Cobertura España** | Local (instalación por evento/recinto) |
| **Licencia** | Propietaria (software como servicio) |
| **Coste** | De pago (depende de num. accesos, funcionalidades) |
| **Horizonte predictivo** | Tiempo real, no predictivo |

**¿Sirve?** Son dispositivos de **medición**, no de **predicción**. Dan el
número exacto de gente dentro en cada instante. No predicen aforo a futuro.

### 2.6 Datos de eventos públicos (ticketing, ayuntamientos)

| Aspecto | Detalle |
|---------|---------|
| **Fuentes** | Ticketea, Entradas.com, portal de eventos de ayuntamientos, APIs de Ticketmaster |
| **Datos** | Capacidad del recinto, entradas vendidas |
| **Cobertura** | Parcial (eventos con venta de entradas) |
| **Licencia** | Variable (API pública Ticketmaster: gratuita con límites; otras: propietarias) |
| **Coste** | Variable |
| **Horizonte predictivo** | Anticipado (antes del evento): sabes cuántas entradas se vendieron |

**¿Sirve?** Es la fuente más directa: si un evento tiene venta de entradas,
el número de entradas vendidas = aforo esperado. Pero no cubre eventos sin
venta de entradas (manifestaciones, playas, parques,fiestas populares).

### 2.7 Smartkalea / Sensores peatonales (caso San Sebastián)

| Aspecto | Detalle |
|---------|---------|
| **Fuente** | Smartkalea (Fomento de San Sebastián) |
| **Datos** | Conteo peatonal por sensores, 5 años de histórico |
| **Modelos** | ARIMA, clustering de días similares, predicción contextual entre sensores |
| **Horizonte** | Corto plazo (15–45 min) con R²>0.7; largo plazo con clustering |
| **Coste** | Infraestructura de sensores (propia del municipio) |
| **Licencia** | Municipal (no abierta) |

**¿Sirve?** Demuestra que la predicción de flujo peatonal a corto plazo es
posible con sensores + ML. Pero requiere infraestructura dedicada y datos
históricos de años. No escalable a toda España.

---

## 3. ¿Se puede predecir el volumen de gente a futuro?

### Respuesta directa: **No de forma generalizable para España con fuentes abiertas**

#### Por qué no:

1. **No existe una fuente abierta de "aforo esperado de eventos"** con
   cobertura nacional. Los datos de movilidad (MITMA, CDR) son agregados por
   zona/día, no por evento.

2. **La granularidad necesaria** (nº de personas en un estadio/ plaza a una
   hora concreta) **no está disponible en fuentes abiertas**. Los datos
   MITMA dicen "movieronse 50.000 personas entre A y B hoy", no "20.000
   fueron al Bernabéu".

3. **Los datos de movilidad tienen latencia** (D-1 mínimo) y no son
   predictivos: son retrospectivos.

4. **Los sistemas de conteo en tiempo real** (IoT/IA) dan datos actuales,
   no futuros. Serían útiles para **validar** una predicción, no para
   generarla.

5. **Los datos de ticketing** son la excepción: sí permiten saber cuánta
   gente irá a un evento con venta de entradas. Pero no cubren:
   - Eventos gratuitos (playas, parques, fiestas populares)
   - Manifestaciones
   - Eventos sin venta de entradas

#### Horizonte temporal alcanzable (con fuentes disponibles):

| Horizonte | Factible? | Fuente requerida |
|-----------|-----------|------------------|
| D-0 (hoy, en tiempo real) | Parcialmente | Sensores IoT o estimación MITMA D-1 |
| D-1 a D-7 | Sí (zona, no evento) | MITMA open data |
| D-8 a D-30 | Difícil | Solo si hay datos de ticketing |
| D-30+ | No factible | No hay fuente de demanda futura abierta |

---

## 4. Conexión con CSV-001 sin duplicar el factor de orgullo colectivo

### Cómo funciona CSV-001 hoy

```
Usuario → sube CSV con personas → endpoint /api/riesgo-colectivo/csv
  → predict_ensemble por persona
  → estadísticas del grupo
  → ORGULLO_COLECTIVO (×1.2 en odds) si tipo_actividad = competicion/deporte
```

El `cantidad` (nº de personas) se extrae del CSV: `len(filas_csv)`. El usuario
controla explícitamente cuánta gente hay.

### Cómo se conectaría la predicción de volumen

La predicción de volumen **no reemplaza** el CSV ni el factor de orgullo.
Sería una **fuente alternativa del campo `cantidad`**:

```
Escenario A (actual):  Usuario → CSV → cantidad = len(filas)
Escenario B (nuevo):   API /api/riesgo-colectivo/evento
                       → recibe: lat, lon, fecha, tipo_actividad
                       → predice: cantidad_estimada
                       → genera CSV sintético (N perfiles demográficos
                         medios del municipio) o usa stats del grupo
                       → pasa por el mismo pipeline que CSV-001
                       → aplica ORGULLO_COLECTIVO si corresponde
```

**Puntos clave de integración:**

1. **El factor de orgullo colectivo NO cambia.** Sigue siendo ×1.2 en odds
   para competición/deporte. La predicción de volumen solo alimenta la `cantidad`.

2. **El pipeline es el mismo.** Una vez que se tiene la `cantidad`, el
   cálculo de riesgo colectivo es idéntico: predict_ensemble por perfil +
   orgullo colectivo.

3. **La predicción es un "pre-filling" del CSV.** En lugar de que el usuario
   introduzca personas manualmente, un modelo genera perfiles
   demográficos representativos del municipio/evento y los mete en el
   pipeline.

4. **Se puede validar.** Si después del evento hay datos reales de aforo
   (sensores IoT, ticketing), se puede comparar con la predicción y
   calibrar el modelo.

### Qué NO hay que hacer

- No crear un segundo factor multiplicativo de "masa de gente". El orgullo
  colectivo ya cubre el efecto fisiológico de grupo.
- No duplicar la lógica de cálculo de riesgo colectivo. Reutilizar
  `_calcular_riesgo_colectivo` o `api_riesgo_colectivo_csv`.
- No hardcodear cantidades. La predicción debe ser dinámica por fecha/lugar.

---

## 5. Decisión

### **APARCAR (shelve)** — con posibilidad de reabrir si cambian las condiciones

#### Justificación:

1. **El cuello de botella es el dato, no el modelo.** No existe una fuente
   abierta de aforo de eventos con cobertura para España que permita
   predicción a futuro. Sin dato, no hay feature.

2. **Las fuentes abiertas disponibles** (MITMA, CDR) son retrospectivas y
   agregadas por zona. No permiten predecir "cuánta gente en el estadio X
   el día Y".

3. **Los datos de ticketing** son la excepción viable, pero solo cubren
   eventos con venta de entradas (minoritarios en el contexto de riesgo
   climático: playas, parques, manifestaciones no tienen ticketing).

4. **La infraestructura IoT** (sensores de conteo) es de pago y localizada.
   No es escalable como fuente de datos para un producto generalista.

5. **El riesgo colectivo ya funciona** con el CSV manual y el factor de
   orgullo colectivo. La mejora de "predecir automáticamente el nº de
   personas" es un lujo, no una necesidad para el MVP.

#### Condiciones para reabrir:

- Si el MITMA (u otro organismo público) publica datos de movilidad con
  granularidad de evento o punto de interés.
- Si se integra una API de ticketing (Ticketmaster, etc.) que proporcione
  aforo estimado de eventos en España.
- Si un municipio concreto ofrece acceso a sus sensores de conteo para
  pilotos.
- Si la demanda de usuarios justifica el esfuerzo de modelizar
  predicciones con fuentes parciales.

#### Acción inmediata:

- Documento archivado en `documentacion/FORECAST-003_crowd_volume_prediction.md`.
- No se propone feature al backlog.
- CSV-001 sigue funcionando como está (CSV manual + orgullo colectivo).

---

## 6. Evidencia y verificación

| Criterio | Estado | Evidencia |
|----------|--------|-----------|
| Documento identifica fuentes reales de volumen con cobertura, licencia, coste | ✅ | Secciones 2.1–2.7 |
| Documento responde si se puede predecir y con qué horizonte | ✅ | Sección 3: respuesta directa + tabla de horizontes |
| Conexión con CSV-001 sin duplicar orgullo colectivo | ✅ | Sección 4 |
| Decisión explícita con justificación | ✅ | Sección 5: APARCAR |
| init.sh sigue en verde | ✅ | Ver evidencia abajo |

### init.sh

```
✔ featureslist.json      presente
✔ current.md             presente
✔ history.md             presente
✔ subagentes             lider, implementer, reviewer, explorer
✔ featureslist           137 features · 16 pending · 1 in_progress · 104 done · 16 blocked
✔ siguiente tarea        FORECAST-003 — Estudio: predecir el volumen de gente a futuro y meterlo en el riesgo colectivo [in_progress]
✔ paquete                climasafeai/ presente
✔ pyproject              presente
✔ tests                  62 ficheros de test
```

ENTORNO LISTO.
