# Próximos pasos — hoja de ruta

**Última revisión:** 2026-07-30

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

### Fase 1 — Riesgo colectivo y demográfico

| # | Qué | Estado |
|---|-----|--------|
| ✅ | Selector Individual / Grupo en el flujo | Hecho |
| ✅ | Modo colectivo: N personas, edad min/max, %hombres, tipo actividad | Hecho |
| ✅ | Modo por etiqueta: tags predefinidas, CRUD, checkboxes | Hecho |
| ✅ | Página de administración de usuarios | Hecho |
| ✅ | Per-person breakdown en resultados por etiqueta | Hecho |
| ✅ | Gráfica de líneas: una línea por persona en grupo | Hecho |
| ✅ | Fecha de nacimiento en lugar de edad | Hecho |
| ✅ | Comorbilidades/medicación en collapsible | Hecho |
| ✅ | **ENS-001**: max-vote → conformal-weighted average | Hecho |
| ✅ | Curvas de riesgo por edad (comparativa 5 edades) | Hecho |
| ✅ | `POST /api/riesgo-volumen` — estimación volumétrica | Hecho |
| ⬜ | **CSV-001** — riesgo colectivo por CSV | Pendiente |
| ⬜ | **MAPA-001** — exportar mapa como PNG/GeoJSON | Pendiente |

### Fase 2 — Mapa de riesgo por zona

| # | Qué | Estado |
|---|-----|--------|
| ✅ | Grid de celdas alrededor de punto (~1km paso) | Hecho |
| ✅ | Cálculo HI pico + clase de riesgo por celda | Hecho |
| ✅ | 4 perfiles de vulnerabilidad con umbrales ajustables | Hecho |
| ✅ | Endpoint GET /api/riesgo-zona | Hecho |
| ✅ | Selector de radio (slider 0.5-25 km) | Hecho |
| ✅ | Selector de perfil de vulnerabilidad | Hecho |
| ✅ | Overlay de rectángulos coloreados en Leaflet | Hecho |

### Fase 3 — Bot de Telegram (determinista)

El anterior bot vía spacebot (con LLM y errores de tool calling) ha sido **reemplazado** por un bot determinista en `climasafeai/bot/telegram_bot.py`. No depende de LLM para la recogida de datos: todo son teclados inline nativos de Telegram. El LLM solo se usa para redactar la respuesta final (con fallback a plantilla).

| # | Qué | Archivos |
|---|-----|----------|
| ✅ | Bot determinista con 17 estados y teclados inline | `climasafeai/bot/telegram_bot.py` |
| ✅ | Flujo completo: sexo, edad, grasa, fototipo, aclimatado, actividad, duración, hora, trabajo, deporte, comorbilidades, medicación, estado previo, situación social, ubicación | `climasafeai/bot/telegram_bot.py` |
| ✅ | Toggles multiselect con toast de confirmación (sin bucle) | `climasafeai/bot/telegram_bot.py` |
| ✅ | Deporte como teclado inline con opciones predefinidas (correr, ciclismo, fútbol, tenis, pádel, senderismo, natación) + opción "Otro" para texto libre | `climasafeai/bot/telegram_bot.py` |
| ✅ | Perfiles SQLite: carga al /start si el chat_id está vinculado | `climasafeai/bot/telegram_bot.py`, `climasafeai/db/manager.py` |
| ✅ | Skip automático de preguntas personales si hay perfil cargado | `climasafeai/bot/telegram_bot.py` |
| ✅ | Guardado de perfil al final de la conversación (pregunta Si/No + alias) | `climasafeai/bot/telegram_bot.py` |
| ✅ | Geocodificación vía Nominatim (nunca LLM) para ubicaciones escritas | `climasafeai/bot/geocoding.py` |
| ✅ | Botón nativo de ubicación (request_location) | `climasafeai/bot/telegram_bot.py` |
| ✅ | 36 tests del flujo completo | `tests/test_telegram_bot.py` |
| ✅ | `make bot` — lanzar en foreground | `Makefile` |
| ✅ | `make bot-daemon` / `make bot-stop` / `make bot-logs` | `Makefile` |

### Fase 4 — MCP y herramientas para asistentes

| # | Qué | Estado |
|---|-----|--------|
| ✅ | Servidor MCP de predicción (6 tools) | `agents/tools/prediction_mcp_tool.py` |
| ✅ | Servidor MCP de factores (10 tools) | `agents/tools/factors_mcp_tool.py` |
| ✅ | `predict_risk_mcp` — todos los campos de la web | Hecho |
| ✅ | `crear_perfil_mcp` — con fototipo y situacion_social | Hecho |
| ✅ | `cargar_perfil_mcp` / `cargar_perfil_por_chat_id_mcp` | Hecho |
| ✅ | `listar_usuarios_mcp` / `vincular_chat_id_mcp` | Hecho |
| ✅ | `make mcp` / `make mcp-factors` / `make mcp-stdio` | `Makefile` |
| ✅ | Modo stdio (Claude Desktop) y SSE | Hecho |

### Fase 5 — Arnés (agentes Python)

Sistema de 26 agentes Python para orquestar el ciclo de desarrollo. Cada feature se abre, implementa, revisa y cierra con verificación automática.

| # | Qué | Archivos |
|---|-----|----------|
| ✅ | `AGENTS.md` — punto de entrada y reglas del ciclo | `AGENTS.md` |
| ✅ | `init.sh` — puerta de verificación (entorno, tests, estructura) | `init.sh` |
| ✅ | `featureslist.json` — backlog con criterios de aceptación | `featureslist.json` |
| ✅ | `progress/` — estado vivo de la feature en curso | `progress/` |
| ✅ | Agentes: lider, explorer, implementer, reviewer, harness | `agents/` |
| ✅ | `make init` / `make harness-check` / `make backlog` | `Makefile` |

---

## Pendiente

### Web UI
- Exportar mapa como PNG/GeoJSON (**MAPA-001**)
- Riesgo colectivo por CSV (**CSV-001**)

### Forecasting
- Tendencia semanal con bandas de confianza
- Modelos fundacionales: TimesFM 2.5, Granite-TTM-R3, WeatherNext 2

### Bot Telegram
- Grupos con comandos `/clima`, `/recomendaciones`
- Notificaciones programadas (crontab en Docker)

### RAG + LLM
- Fine-tuning Gemma 4 vía Unsloth
- Resúmenes con RAG sobre documentos del proyecto
- Consultas en lenguaje natural con respuesta sintetizada

### Agentes
- Publicar skill ClimaSafeAI en skills.sh registry
- Flue Framework (ejecución durable, sandboxes)

---

## Resumen visual

```
Bot Telegram   ── Bot determinista 17 estados ✅ · Perfiles SQLite ✅
MCP            ── 6+10 tools ✅ · Modo stdio/SSE ✅
Arnés          ── Ciclo agents/ ✅ · init.sh ✅ · backlog ✅
Web UI         ── Curvas edad ✅ · Volumen ✅ · Mapa riesgo ✅ · [⬜ exportar]
CSV colectivo  ── [⬜ CSV-001]
Forecasting    ── Tendencia semanal · TimesFM · márgenes error
RAG + LLM      ── Gemma 4 + Unsloth + LoRA [⬜]
```

## Referencias

- `conclusion-base-conocimiento.md` — decisión técnica de base de conocimiento
- `arquitectura/pipeline_prediccion.md` — flujo completo de predicción
- `riesgo/personalizacion_individual.md` — coeficientes de factores
- `riesgo/formulas_deterministas.md` — HI, WC, UV
- `ml/conclusiones_modelos.md` — métricas y comparación de modelos
- `ml/contrafactuales.md` — generación de contrafactuales
- `conformal_prediction.md` — split conformal
