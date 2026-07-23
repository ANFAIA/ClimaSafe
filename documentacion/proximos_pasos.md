# Próximos pasos — hoja de ruta

**Última revisión:** 2026-07-23

---

## Lo hecho ✅

### Fase 0 — Consolidación (sesiones 2026-07-22/23)

| # | Qué | Archivos |
|---|-----|----------|
| ✅ | Factores de personalización: sexo, edad, grasa relativa, entrenado, ocupación, fiesta | `climasafeai/features/personalizacion.py` |
| ✅ | Safety overrides por calor (HI con vulnerable check) y frío (WC) | `climasafeai/models/ensemble.py` |
| ✅ | Override por edad≥60 refinado: excluye si entrenado+aclimatado | `climasafeai/models/ensemble.py` |
| ✅ | Downgrade por ausencia de calor real (HI<27, WC>0, UV<6) | `climasafeai/models/ensemble.py` |
| ✅ | Perfiles guardados por alias en SQLite (find-or-create) | `chat/app.py`, `climasafeai/db/manager.py` |
| ✅ | GET /api/perfiles — lista cabeceras de todos los perfiles | `chat/app.py` |
| ✅ | Frontend: selector de perfiles, modal guardar, rellenar formulario | `chat/static/index.html` |
| ✅ | Frontend: indicadores de confianza conformal (círculos) | `chat/static/index.html` |
| ✅ | Fiesta como entrada separada (no mezclada con situacional) | `climasafeai/features/personalizacion.py` |
| ✅ | Recomendaciones contextuales (time-aware, sport-aware, fiesta-aware) | `climasafeai/models/recomendaciones.py` |
| ✅ | Diagnóstico bayesiano + contrafactuales en explicación | `climasafeai/models/explicabilidad.py`, `climasafeai/models/bayes.py` |
| ✅ | Conformal prediction (split conformal, α=0.1) en producción | `climasafeai/models/conformal.py`, `main.py` |
| ✅ | XGBoost reentrenado (1000 estimators, early stopping, balanced) | `main.py` |
| ✅ | Thresholds ajustados: calor t2=0.10, LSTM t1=0.50 | `climasafeai/models/predict_model.py` |
| ✅ | sqlite-vec RAG — embeddings semánticos sobre factores de riesgo | `climasafeai/db/rag.py`, `data/schema.sql` |
| ✅ | Tests de personalización (11 tests) | `tests/test_personalizacion.py` |
| ✅ | Documentación actualizada: pipeline, personalización, hoja de ruta | `documentacion/` |

---

## Próximo — Fase 1: Riesgo colectivo y demográfico

### 1.1 Múltiples edades en una pantalla
Mostrar **todas las curvas de riesgo por edad** simultáneamente (18–85+):
- Gráfica de líneas: eje X = hora del día, eje Y = HI / probabilidad, una línea por grupo etario
- Resaltar umbrales de PRECAUCION y PELIGRO
- Tooltip: "A las 14:00, un adulto de 70a tiene riesgo X, un joven de 25a tiene riesgo Y"

### 1.2 Riesgo colectivo por CSV
Endpoint que acepte un **CSV de personas** con sus perfiles:
```csv
nombre,edad,sexo,grasa,actividad,...
Juan,25,hombre,18,ligera,...
María,70,mujer,28,reposo,...
```
- Devuelve: riesgo individual de cada uno + estadísticas del grupo
- **Factor "orgullo colectivo"**: si es un evento deportivo, el riesgo individual
  se multiplica por un factor (la gente se exige más en grupo)

### 1.3 Predicción por volumen
Dado un volumen de N personas en una zona, estimar **cuántas podrían tener
problemas cardíacos**:
- Usar prevalencia poblacional de enfermedad cardiovascular (~10-15% en >50a)
- Combinar con las condiciones climáticas previstas para ese día
- Devolver: "De 5000 asistentes esperados, ~75 podrían requerir atención médica por calor"

### 1.4 API estructurada
Endpoint REST que permita:
- `POST /api/riesgo-colectivo` — CSV + ubicación + fecha → estadísticas
- `GET /api/riesgo-zona?lat=...&lon=...&radio=5` — riesgo por cuadrantes

---

## Fase 2: Mapa de riesgo por zona

### 2.1 Grid de riesgo por km²
Dividir el área alrededor de un punto en cuadrantes de ~1 km² y calcular:
- HI pico del día para cada cuadrante
- Clase de riesgo (SEGURO/PRECAUCION/PELIGRO) por cuadrante
- Colorear mapa con semáforo

**Fuente**: Open-Meteo devuelve datos grillados (~11 km). Habría que interpolar
o consultar múltiples puntos. Alternativa: usar la rejilla de ERA5.

**Modelos satelitales útiles aquí**:
- **Prithvi-EO-2.0** (IBM/NASA, 600M, Apache-2.0) — foundation model para
  imágenes satelitales (HLS 30m, 6 bands). Útil para clasificar superficie
  (asfalto retiene más calor que bosque) y ajustar HI por tipo de suelo.
- **AlphaEarth Foundations** (DeepMind) — embeddings satelitales 10×10m de toda
  la superficie terrestre. Disponible en Google Earth Engine. Útil para
  caracterizar el terreno sin necesidad de procesar imágenes crudas.

### 2.2 Selector de radio
Input tipo slider: "Mostrar riesgo en un radio de ___ km"
- 1 km, 5 km, 10 km, 25 km
- A más radio, menos resolución (promedio por cuadrante más grande)

### 2.3 Exportar mapa
- Captura PNG del mapa de riesgo (para LinkedIn, informes)
- Datos subyacentes en GeoJSON

---

## Fase 3: Forecasting y margen de error

### 3.1 Predicción a más días
Actualmente se puede consultar cualquier fecha. Ampliar:
- Mostrar **tendencia semanal**: "El jueves es el día más peligroso de la semana"
- Para cada día, mostrar el **margen de error** de la predicción meteorológica
- Bandas de confianza: "HI estimado: 32±3°C"

### 3.2 Modelos fundacionales para forecasting
Modelos pre-entrenados en HuggingFace que podrían complementar o reemplazar
los modelos propios (XGBoost, RF, LSTM):

| Modelo | Creador | Params | Lo hace | Licencia |
|--------|---------|--------|---------|----------|
| **TimesFM 2.5** | Google | 200M | Forecasting de series temporales (decoder-only). No es específico de clima pero funciona con cualquier serie. | Apache-2.0 |
| **Granite-TTM-R3** | IBM | 1.4M | Forecasting ligero de series, corre en CPU. Ya documentado en `evaluacion_fuentes_externas.md`. | CC-BY-NC-SA (restrictiva) |
| **WeatherNext 2** | DeepMind | — | Forecasting meteorológico global, 6h resolución. Disponible en Google Earth Engine. Ya documentado. | Propietaria (API) |
| **Prithvi-EO-2.0** | IBM/NASA | 600M | Foundation model EO (satélite). Análisis multitemporal HLS (30m, 6 bands). *No es forecasting* — clasificación suelo/vegetación. | Apache-2.0 |
| **AlphaEarth Foundations** | DeepMind | — | Embeddings satelitales 10×10m. Dataset en Earth Engine para análisis geoespacial. | Propietaria (EE) |

**Recomendación**:
- **TimesFM 2.5** es el más prometedor para reemplazar modelos propios de
  forecasting de riesgo (serie temporal de features → riesgo). Apache-2.0,
  200M params, corre en GPU consumer.
- **Granite-TTM-R3** es alternativa ligera (CPU), pero licencia CC-BY-NC-SA.
- WeatherNext y Open-Meteo son **fuentes de datos**, no modelos que sustituyan
  al nuestro.
- Prithvi-EO y AlphaEarth son para **análisis geoespacial** (mapas, Fase 2).

### 3.3 Forecast con supervisión
Cuando el forecast tiene alta incertidumbre, añadir aviso:
> "Predicción con baja confianza debido a dispersión entre modelos.
> Se recomienda supervisar manualmente las condiciones reales."

---

## Fase 4: Mensajería y notificaciones

### 4.1 Abstracción de mensajería
Interfaz común para enviar alertas, con adaptadores:

```
MessageService
  ├── TelegramAdapter    → Bot API
  ├── HermesAdapter      → Plugin para Hermes
  ├── WebhookAdapter     → URL configurable
  └── ConsoleAdapter     → Log / stdout (desarrollo)
```

Cada adaptador implementa: `send(user_id, message, image?)`.

### 4.2 Crontab en Docker
Contenedor que ejecuta evaluaciones programadas:
- `0 8 * * *` — "Hoy en tu zona: PRECAUCION a las 15:00. Toma precauciones."
- `0 20 * * *` — "Resumen del día: HI pico 34°C a las 16:00."
- Configurable por usuario (horario, canales)

### 4.3 Worker de notificaciones
- Lee de una cola (Redis / SQLite) los usuarios que deben ser notificados
- Usa el MessageService para enviar
- Escalable: N workers compitiendo por mensajes

### 4.4 Telegram
- **Chat privado**: el usuario se identifica con su `telegram_id`, se carga su
  perfil guardado, recibe predicciones personalizadas.
- **Grupo**: el bot responde a comandos como `/clima Madrid`, `/recomendaciones`
  (visibilidad pública, sin perfil individual).

---

## Fase 5: Agentes e integración MCP

### 5.1 MCP Server
Exponer el sistema como **MCP server**:
```
Herramientas:
  - predecir(provincia, perfil) → clase_riesgo
  - recomendar(provincia, perfil) → recomendaciones
  - riesgo_zona(lat, lon, radio) → grid de riesgo
  - buscar_factor(query) → factores de riesgo (RAG con sqlite-vec)
  - contrafactuales(perfil) → qué cambiar para bajar el riesgo
```

Cualquier LLM (Claude, Codex, GPT) con MCP puede consultar:
> "Voy a salir a correr por la Sierra de Madrid a las 7 de la mañana, ¿qué riesgo tengo?"

### 5.2 Agente Harness
Un agente que:
- Escucha en un puerto (MCP / WebSocket)
- Recibe queries en lenguaje natural
- Decide qué herramientas usar
- Devuelve resultado + imagen si procede

### 5.3 Plugin para otros agentes
El sistema empaquetado como **skill** de Skills.sh o plugin MCP:
- `climasafeai-mcp` publicable
- Otros agentes lo importan: "analiza el clima para mi ruta de running"
- Skills.sh ya tiene patrones para esto

### 5.4 Hermes
Investigar Hermes como orquestador:
- ¿Hermes puede leer el chat y actuar como agente?
- ¿O usarlo como gateway de mensajería (recibe → MCP server)?
- Diferencia entre agente autónomo (Hermes) vs plugin (ClimaSafeAI como skill)

---

## Fase 6: RAG + LLM local

### 6.1 Resúmenes con LLM local
Lo que devuelva el RAG (factores de riesgo, papers) resumirlo con un LLM local:
- Gemma 3 (Google) vía Unsloth
- Fine-tuning LoRA en Google Colab con `unsloth`
- Dataset: papers de factores de riesgo, documentación del proyecto

### 6.2 Enlaces de referencia
- https://unsloth.ai/docs/models/gemma-4/train — entrenamiento con GUI
- https://www.skills.sh/google-gemma/gemma-skills/gemma-trainer — skill para fine-tune
- https://github.com/mattpocock/skills/tree/main/skills/productivity/teach
- https://www.skills.sh/aradotso/trending-skills/gemma-tuner-multimodal

### 6.3 Estrategia
1. Hacer RAG sobre los papers almacenados (sqlite-vec ya implementado)
2. El LLM local (Gemma 3 2B/9B fine-tuneado) resume la respuesta del RAG
3. El usuario pregunta en lenguaje natural y recibe respuesta sintetizada
4. El fine-tune se entrena con los propios documentos del proyecto

---

## Fase 7: UX y despliegue

### 7.1 Fecha de nacimiento
En lugar de pedir "edad", pedir **fecha de nacimiento** (campo `date`):
- El frontend calcula la edad automáticamente
- Una vez enviado el perfil, la fecha se transforma en edad y se descarta
- Privacidad: no se almacena la fecha de nacimiento

### 7.2 Chat iterativo en la GUI
- Mejorar la interfaz actual con un **chat** (asistente conversacional)
- Ejemplo: usuario escribe "voy a correr mañana a las 8 en Valencia"
- El asistente interpreta, rellena el formulario y ejecuta la predicción
- Las respuestas pueden incluir gráficos inline

### 7.3 Capturas para LinkedIn
- Generar imágenes bonitas del mapa de riesgo y las gráficas
- Con marca de agua de ClimaSafeAI
- Botón "Compartir" que descarga PNG

### 7.4 Despliegue simplificado
- Dockploy / Spacebot como alternativa a Docker manual
- Skills.sh para distribuir como skill instalable
- Futuro: "agentes que te montan todo, no un Docker"

---

## Retos técnicos / Aprendizaje

Ideas para aprender conceptos nuevos que aporten valor al proyecto y sean
transferibles. Ninguno tiene prioridad fija.

### Cadenas de Markov / HMM
Modelar *trayectoria* de riesgo (dado que hoy hay precaución, probabilidad de
que mañana sea peligro). Alternativa al punto-a-punto actual.

### Redes Bayesianas
Grafo causal de factores de riesgo. Diagnóstico inverso: "dado riesgo peligro,
¿qué factor lo causa?" — ya implementado como `BayesianRiskDiagnosis`.

### Teoría causal (Pearl)
Contrafactuales formales: "¿qué pasaría si todos se aclimataran?" Siguiente
salto cualitativo sobre "solo predecir".

### Procesos Gaussianos
Incertidumbre nativa para factores con pocos datos. Complemento a XGBoost/LSTM.

### Graph Neural Networks
Embeddings de factores basados en su grafo de relaciones. Mejor que all-MiniLM
para el RAG.

### Series temporales avanzadas (TFT / N-BEATS)
Alternativa al LSTM híbrido actual. Forecast multi-día con atención interpretable.

### Aprendizaje por refuerzo
Pasar de "predecir riesgo" a "reducir riesgo": agente que recomienda acciones y
aprende de las consecuencias.

---

## Resumen visual

```
Fase 1 ── CSV, gráfica multi-edad, volumen, API   [AHORA]
Fase 2 ── Mapa de riesgo por km², colores, exportar
Fase 3 ── Tendencia semanal, TimesFM, márgenes de error
Fase 4 ── Telegram, Hermes, crontab, worker
Fase 5 ── MCP server, Harness, Skills.sh plugin
Fase 6 ── Gemma 3 + Unsloth + LoRA para RAG
Fase 7 ── Fecha nac., chat, LinkedIn, Dockploy
```

## Referencias

- `conclusion-base-conocimiento.md` — decisión técnica de base de conocimiento
- `arquitectura/agentes_ia.md` — arquitectura de agentes existente
- `arquitectura/pipeline_prediccion.md` — flujo completo de predicción
- `riesgo/personalizacion_individual.md` — coeficientes de factores
- `riesgo/formulas_deterministas.md` — HI, WC, UV
- `ml/conclusiones_modelos.md` — métricas y comparación de modelos
- `ml/contrafactuales.md` — generación de contrafactuales
- `conformal_prediction.md` — split conformal
