# Próximos pasos — hoja de ruta

**Última revisión:** 2026-07-27 (tarde)

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

### Fase 1 — Riesgo colectivo y demográfico (sesión 2026-07-24)

| # | Qué | Archivos |
|---|-----|----------|
| ✅ | Selector Individual / Grupo en el flujo (paso 1 → paso 2) | `chat/static/index.html` |
| ✅ | Modo colectivo: N personas, edad min/max, %hombres, tipo actividad (trabajo/deporte/competición), duración, aclimatación | `chat/app.py`, `chat/static/index.html` |
| ✅ | Modo por etiqueta: seleccionar tag → predecir todos los perfiles con esa tag | `chat/app.py`, `climasafeai/db/manager.py` |
| ✅ | Tags predefinidas: tabla `tags_disponibles`, CRUD, checkboxes (evita errores ortográficos) | `chat/app.py`, `climasafeai/db/manager.py`, `data/schema.sql` |
| ✅ | Página de administración de usuarios (👥 Usuarios) | `chat/static/index.html` |
| ✅ | Per-person breakdown en resultados por etiqueta (lista agrupada por riesgo) | `chat/static/index.html` |
| ✅ | Gráfica de líneas: una línea por persona en grupo por etiqueta | `chat/static/index.html` |
| ✅ | Navegación admin ↔ perfil (flag _vieneDeAdmin) | `chat/static/index.html` |
| ✅ | Tags visibles como chips en paso 2 | `chat/static/index.html` |
| ✅ | Fecha de nacimiento en lugar de edad (campo date, cálculo automático) | `climasafeai/db/manager.py`, `chat/static/index.html`, `data/schema.sql` |
| ✅ | Comorbilidades/medicación en collapsible | `chat/static/index.html` |

### 1.1 Múltiples edades en una pantalla (sesión 2026-07-27)

Todas las curvas de riesgo por edad a la vez sobre el eje horario, con los
umbrales resaltados y tooltip comparativo hora a hora.

| # | Qué | Archivos |
|---|-----|----------|
| ✅ | `POST /api/curvas-edad` — N curvas de riesgo horario por edad con UNA sola descarga de meteo y sin ejecutar los modelos (la curva solo depende del perfil horario y de la personalización) | `chat/app.py` |
| ✅ | `perfil_horario_desde_df()` extraído de `predict_ensemble`, y `PERS_THRESHOLD_PELIGRO` subido a constante de módulo para reutilizar los umbrales | `climasafeai/models/ensemble.py` |
| ✅ | Comparativa de 5 edades (antes 1-2) en una sola llamada; se descarta el tramo que solapa con la edad real para no duplicar su línea | `chat/static/index.html` |
| ✅ | Líneas horizontales de umbral PRECAUCION / PELIGRO sobre el eje de riesgo personal | `chat/static/index.html` |
| ✅ | Tooltip comparativo por hora (modo `index` de Chart.js, ya existente) | `chat/static/index.html` |
| ✅ | Fix: la comparativa ya no persiste perfiles inventados en SQLite — flag `persistir` en `/api/predict` | `chat/app.py`, `chat/static/index.html` |
| ✅ | Tests del endpoint y del flag (7 nuevos) | `tests/test_api.py` |
| ⬜ | Ajustar `EDADES_COMPARATIVA` a `(25, 55, 65, 75, 85)`: los tramos de `_factor_edad_calor` son 45/55/65/75/85, así que 25 y 40 pintan la misma curva | `chat/app.py` |

---

## Pendiente — Fase 1: Riesgo colectivo y demográfico

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

### 1.3 Predicción por volumen ✅
| # | Qué | Archivos |
|---|-----|----------|
| ✅ | `estimar_afectados()` con prevalencia ECV (4%/<50, 12%/>50), multiplicador climático por tramos HI, factor evento, rango ±30% | `climasafeai/models/volumen.py` |
| ✅ | `POST /api/riesgo-volumen` — calcula HI pico en ventana de actividad, llama estimar_afectados | `chat/app.py` |
| ✅ | Tests (26): multiplicador, escalado, ejemplo roadmap, extremos, claves presentes | `tests/test_volumen.py` |
| ✅ | Explicabilidad colectivo: 5 tarjetas (factores_detalle, contrafactuales 3 escenarios, distr. demográfica, HI overlay, resumen ejecutivo) | `chat/static/index.html`, `chat/app.py` |
| ✅ | Prevalencias internas por edad (`_prevalencia(edad)`) — elimina inputs manuales de % comorbilidades del formulario | `chat/app.py` |
| ✅ | Volumen como submodo de Grupo (junto a Colectivo y Por etiqueta) — UI más intuitiva | `chat/static/index.html` |

### 1.4 API estructurada
- `GET /api/riesgo-zona?lat=...&lon=...&radio=5` — riesgo por cuadrantes (✅ hecho en Fase 2)

---

## Fase 2: Mapa de riesgo por zona (sesión 2026-07-24)

### 2.1 Grid de riesgo por km² — ✅
- `/api/riesgo-zona` — genera grid de celdas alrededor de un punto
- Calcula HI pico y clase de riesgo (SEGURO/PRECAUCION/PELIGRO) por celda
- 4 perfiles de vulnerabilidad: vulnerable (default), mayor, adulto, joven
- Distintos umbrales HI según perfil (más restrictivo = más sensible)
- Frontend: overlay de rectángulos coloreados sobre Leaflet, slider de radio (0.5-25 km), selector de perfil

| # | Qué | Archivos |
|---|-----|----------|
| ✅ | Grid de celdas alrededor de punto (paso ~1km) | `climasafeai/data/grid_risk.py` |
| ✅ | Cálculo HI pico + clase de riesgo por celda | `climasafeai/data/grid_risk.py` |
| ✅ | 4 perfiles de vulnerabilidad con umbrales ajustables | `climasafeai/data/grid_risk.py` |
| ✅ | Endpoint GET /api/riesgo-zona | `chat/app.py` |
| ✅ | Selector de radio (slider 0.5-25 km) | `chat/static/index.html` |
| ✅ | Selector de perfil (más restrictivo / adulto mayor / adulto / joven) | `chat/static/index.html` |
| ✅ | Overlay de rectángulos coloreados en Leaflet | `chat/static/index.html` |
| ✅ | Leyenda y estadísticas en tiempo real | `chat/static/index.html` |

### 2.2 Selector de radio — ✅
Input tipo slider ya implementado (0.5-25 km, paso 0.5 km).

### 2.3 Exportar mapa — ⬜
- Captura PNG del mapa de riesgo
- Datos subyacentes en GeoJSON

---

## Fase 3: Forecasting y margen de error

### 3.1 Predicción a más días
- Mostrar **tendencia semanal**
- Bandas de confianza

### 3.2 Modelos fundacionales para forecasting
- **TimesFM 2.5** (Google, Apache-2.0) — reemplazar modelos propios
- **Granite-TTM-R3** (IBM, 1.4M, CPU, licencia restrictiva)
- **WeatherNext 2** (DeepMind) — forecasting meteorológico global

### 3.3 Forecast con supervisión
Aviso cuando el forecast tiene alta incertidumbre.

---

## Fase 4: Mensajería y notificaciones

### 4.1 Abstracción de mensajería
Interfaz común con adaptadores: Telegram, Hermes, Webhook, Console.

### 4.2 Crontab en Docker
Contenedor con evaluaciones programadas (mañana/resumen).

### 4.3 Worker de notificaciones
Cola de mensajes con N workers compitiendo.

### 4.4 Telegram
- Chat privado con perfil guardado
- Grupo con comandos `/clima`, `/recomendaciones`

---

## Fase 5: Agentes e integración MCP

| # | Qué |
|---|-----|
| ✅ | MCP Server con herramientas: predict_risk, predict_volume_risk, predict_zone_risk, predict_group_risk, predict_age_curves, system_health (puerto 8101) · `agents/tools/prediction_mcp_tool.py` |
| ⬜ | Agente Harness (WebSocket, queries en lenguaje natural) |
| ⬜ | Plugin Skills.sh / MCP publicable |
| ⬜ | Investigar Hermes como orquestador |

---

## Fase 6: RAG + LLM local

| # | Qué |
|---|-----|
| ⬜ | Resúmenes con Gemma 3 vía Unsloth |
| ⬜ | Fine-tuning LoRA con documentos del proyecto |
| ⬜ | Consultas en lenguaje natural con respuesta sintetizada |

---

## Fase 7: UX y despliegue

| # | Qué |
|---|-----|
| ✅ | Fecha de nacimiento (campo date, cálculo automático de edad) |
| ⬜ | Chat iterativo en la GUI (asistente conversacional) |
| ⬜ | Capturas para LinkedIn con marca de agua |
| ⬜ | Despliegue simplificado (Dockploy / Skills.sh) |

---

## Retos técnicos / Aprendizaje

| # | Qué |
|---|-----|
| ⬜ | Cadenas de Markov / HMM — modelar trayectoria de riesgo |
| ⬜ | Redes Bayesianas — grafo causal (diagnóstico ya implementado) |
| ⬜ | Teoría causal (Pearl) — contrafactuales formales |
| ⬜ | Procesos Gaussianos — incertidumbre nativa |
| ⬜ | Graph Neural Networks — embeddings de factores |
| ⬜ | Series temporales avanzadas (TFT / N-BEATS) |
| ⬜ | Aprendizaje por refuerzo — de predecir a reducir riesgo |

---

## Resumen visual

```
Fase 1 ── Curvas por edad ✅ · CSV, volumen ✅        [PENDIENTE — 1.2 CSV]
Fase 2 ── Mapa de riesgo por km², colores, exportar  [✅ Grid + perfiles hecho, ⬜ exportar]
Fase 3 ── Tendencia semanal, TimesFM, márgenes de error
Fase 4 ── Telegram, Hermes, crontab, worker
Fase 5 ── MCP server ✅ · Harness, Skills.sh plugin
Fase 6 ── Gemma 3 + Unsloth + LoRA para RAG
Fase 7 ── Chat, LinkedIn, Dockploy
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
