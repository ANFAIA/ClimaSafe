# Próximos pasos — hoja de ruta

**Última revisión:** 2026-08-27

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
| ✅ | **CSV-001** — riesgo colectivo por CSV | Hecho |
| ✅ | **MAPA-001** — exportar mapa como PNG/GeoJSON | Hecho |

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

### Fase 3 — Bot de Telegram (determinista + LLM)

| # | Qué | Archivos |
|---|-----|----------|
| ✅ | Bot determinista con 17 estados y teclados inline | `climasafeai/bot/telegram_bot.py` |
| ✅ | Flujo completo: sexo, edad, grasa, fototipo, aclimatado, actividad, duración, hora, trabajo, deporte, comorbilidades, medicación, estado previo, situación social, ubicación | `climasafeai/bot/telegram_bot.py` |
| ✅ | Toggles multiselect con toast de confirmación (sin bucle) | `climasafeai/bot/telegram_bot.py` |
| ✅ | Deporte como teclado inline con opciones predefinidas | `climasafeai/bot/telegram_bot.py` |
| ✅ | Perfiles SQLite: carga al /start si el chat_id está vinculado | `climasafeai/bot/telegram_bot.py`, `climasafeai/db/manager.py` |
| ✅ | Skip automático de preguntas personales si hay perfil cargado | `climasafeai/bot/telegram_bot.py` |
| ✅ | Guardado de perfil al final de la conversación | `climasafeai/bot/telegram_bot.py` |
| ✅ | Geocodificación vía Nominatim (nunca LLM) | `climasafeai/bot/geocoding.py` |
| ✅ | Botón nativo de ubicación (request_location) | `climasafeai/bot/telegram_bot.py` |
| ✅ | **BOT-005**: Parte del bot con datos previos, intensidad y recomendación | `climasafeai/bot/telegram_bot.py`, `climasafeai/models/recomendaciones.py` |
| ✅ | **BOT-011**: Parte claro y personalizado con explicación del % | `climasafeai/models/recomendaciones.py`, `climasafeai/bot/telegram_bot.py` |
| ✅ | **BOT-012**: Horas peligrosas y horario recomendado en el parte | `climasafeai/bot/telegram_bot.py` |
| ✅ | **BOT-013**: Confianza del modelo y explicación de la clase | `climasafeai/llm/rag_qwen.py`, `climasafeai/bot/telegram_bot.py` |
| ✅ | **BOT-014**: Chat post-parte con canal dominante y factores con peso | `climasafeai/bot/telegram_bot.py` |
| ✅ | **BOT-016**: Tipo de ocupación/deporte al añadir rutina | `climasafeai/bot/telegram_bot.py` |
| ✅ | **BOT-017**: Repetir última salida al cargar perfil | `climasafeai/db/manager.py`, `climasafeai/bot/telegram_bot.py` |
| ✅ | **BOT-018**: Voz (STT Whisper + TTS gTTS) | `climasafeai/bot/voice.py` |
| ✅ | **BOT-019**: Menos pasos (deducción con perfil + rutina) | `climasafeai/bot/telegram_bot.py` |
| ✅ | **BOT-020**: Formato del parte: clasificación, %, factores, tabla, recomendaciones | `climasafeai/bot/telegram_bot.py` |
| ✅ | **BOT-021**: Memoria conversacional (entender "voy al tenis como ayer") | `climasafeai/bot/telegram_bot.py` |
| ✅ | **BOT-022**: Fix reenvío LLM y limpieza de think (Qwen3) | `climasafeai/bot/telegram_bot.py`, `climasafeai/llm/rag_qwen.py` |
| ✅ | **BOT-023**: OpenRouter como complemento a Groq (modelos free) | `climasafeai/llm/rag_qwen.py`, `climasafeai/bot/telegram_bot.py` |
| ✅ | **HOST-001**: LLM remoto (Groq/OpenRouter) + fallback determinista | `climasafeai/bot/telegram_bot.py`, `climasafeai/llm/rag_qwen.py` |
| ✅ | **TG-002**: Perfil vinculado al chat de Telegram | `climasafeai/bot/telegram_bot.py` |
| ✅ | **CHAT-003**: Fusionar /chat en /start (LLM + chat abierto) | `climasafeai/bot/telegram_bot.py` |
| ✅ | Rutinas semanales con deporte y ocupación | `climasafeai/bot/telegram_bot.py` |
| ✅ | Avisos diarios del riesgo | `climasafeai/bot/telegram_bot.py` |
| ✅ | Logging seguro (sin token, sin duplicados) | `climasafeai/bot/telegram_bot.py` |

### Fase 4 — MCP y herramientas para asistentes

| # | Qué | Estado |
|---|-----|--------|
| ✅ | Servidor MCP de predicción (6+ tools) | `agents/tools/prediction_mcp_tool.py` |
| ✅ | Servidor MCP de factores (10 tools) | `agents/tools/factors_mcp_tool.py` |
| ✅ | `predict_risk_mcp` — todos los campos de la web | Hecho |
| ✅ | `grafica_riesgo_horario_mcp` — imagen PNG de la curva | Hecho |
| ✅ | `crear_perfil_mcp` — con fototipo y situacion_social | Hecho |
| ✅ | `cargar_perfil_mcp` / `cargar_perfil_por_chat_id_mcp` | Hecho |
| ✅ | `listar_usuarios_mcp` / `vincular_chat_id_mcp` | Hecho |
| ✅ | `riesgo_rutinas_dia_mcp` — riesgo de rutinas por día | Hecho |
| ✅ | Modo stdio (Claude Desktop) y streamable HTTP | Hecho |
| ✅ | **MCP-002**: Capa de solo lectura por defecto, escritura con token | Hecho |
| ✅ | **MCP-003**: Control de acceso por identidad (IDs opacos) | Hecho |
| ✅ | **MCP-004**: Adaptación a spec MCP 2025-06-18+ (annotations) | Hecho |
| ✅ | **MCP-IMG-001**: Gráfica del riesgo por hora desde MCP | Hecho |
| ✅ | **MCP-APPS-001**: MVP de MCP Apps (UI interactiva) | Hecho |

### Fase 5 — LLM y RAG

| # | Qué | Estado |
|---|-----|--------|
| ✅ | **LLM-001**: Qwen 2.5 7B local + RAG + fallback determinista | Hecho |
| ✅ | **LLM-003**: Sistema de prompts (SYSTEM_PARTE, etc.) | Hecho |
| ✅ | **LLM-004**: Dataset LLM con input que lleva tiempo y UV | Hecho |
| ✅ | **LLM-005**: Parte del LLM explica factores y recomendaciones | Hecho |
| ✅ | **LLM-006**: Notebook Colab para fine-tuning (GPU T4 gratuita) | Hecho |
| ✅ | **LLM-007**: Estudio LoRA/aLoRA/multi-adapter | Hecho |
| ✅ | **LLM-008**: Evaluación Granite aLoRA para RAG (NO adoptar) | Hecho |
| ✅ | **LLM-012**: DLMs para fine-tuning (DESCARTADO) | Hecho |
| ✅ | **LLM-013**: Adaptar fine-tuning a Qwen3-1.7B | Hecho |
| ✅ | **LLM-014**: Bot usa qwen3:climasafe como modelo fine-tuneado | Hecho |
| ✅ | **LLM-015**: Guía de revisión + script QC del dataset | Hecho |
| ✅ | **LLM-016**: Estudio base vs instruct | Hecho |
| ✅ | **LLM-017**: Dataset regenerado con QC a 0 hallazgos | Hecho |
| ✅ | **LLM-018**: Notebook actualizado a Qwen3 instruct + dataset | Hecho |
| ✅ | **LLM-019**: Benchmark de modelos LLM gratuitos | Hecho |
| ✅ | **RAG-001**: Fine-tuning Gemma 4 + RAG local | Hecho |
| ✅ | **RAG-002**: RAG indexa coeficientes y DOI | Hecho |
| ✅ | **RAG-003**: Filtrar literatura del dominio vs ruido | Hecho |
| ✅ | **RAG-004**: Set de evaluación del RAG (recall/precision) | Hecho |
| ✅ | **RAG-005**: Reindexa cuando cambia el texto (hash sha256) | Hecho |
| ✅ | **RAG-006**: Afinar retrieval (solapamiento + comparar embeddings) | Hecho |
| ✅ | **HOST-001**: LLM en hosting gratuito (Groq/OpenRouter) | Hecho |
| ✅ | **BAYES-001**: Modelo bayesiano jerárquico por provincias | Hecho |

### Fase 6 — Arnés y agentes

| # | Qué | Archivos |
|---|-----|----------|
| ✅ | `AGENTS.md` — punto de entrada y reglas del ciclo | `AGENTS.md` |
| ✅ | `init.sh` — puerta de verificación (entorno, tests, estructura) | `init.sh` |
| ✅ | `featureslist.json` — backlog con criterios de aceptación | `featureslist.json` |
| ✅ | `progress/` — estado vivo de la feature en curso | `progress/` |
| ✅ | Agentes: lider, explorer, implementer, reviewer, harness | `agents/` |
| ✅ | **ARNES-003**: Modo debug para ver payload al LLM | `climasafeai/llm/rag_qwen.py` |
| ✅ | **ARNES-004**: Contador de tokens y coste por petición | `climasafeai/llm/costes.py` |
| ✅ | **ARNES-006**: Bucle propio en Python sobre LiteLLM | `agents/loop.py` |
| ✅ | **ARNES-007**: Subagentes y compactación | `agents/subagent.py`, `agents/compaction.py` |
| ✅ | **ARNES-009**: Comparativa de arneses (ADK, DeepAgents, manual) | `documentacion/arnes_comparativa.md` |
| ✅ | **ARNES-010**: Tope de presupuesto de tokens por petición | `climasafeai/llm/costes.py` |
| ✅ | **ARNES-011**: Security/Policy Layer (permisos, sandbox, auditoría) | `agents/security.py` |
| ✅ | **ARNES-012**: Timeout de harness (900s → 3600s) | `agents/agents/harness_agent.py` |
| ✅ | **ARNES-013**: Candado por dueño (dos asistentes en paralelo) | `agents/agents/harness_agent.py` |
| ✅ | **ARNES-014**: Commit automático al cerrar ticket (con --commit) | `agents/agents/harness_agent.py` |
| ✅ | **GIT-001**: Bump de versión + propuesta de commit sin co-autoría | `agents/agents/git_agent.py` |

### Fase 7 — Web UI

| # | Qué | Estado |
|---|-----|--------|
| ✅ | Formulario completo con mapa y selector de perfiles | Hecho |
| ✅ | Modo Individual / Grupo / Chat | Hecho |
| ✅ | Curvas de riesgo por edad | Hecho |
| ✅ | Estimación volumétrica de afectados | Hecho |
| ✅ | Mapa de riesgo por zona (grid + Leaflet) | Hecho |
| ✅ | **WEB-003**: Fix campo desconocido en perfil (500 → 200 + warning) | Hecho |
| ✅ | **WEB-004**: XSS almacenado (escapar nombre de usuario) | Hecho |
| ✅ | **WEB-005**: HTTP 200 con error en el cuerpo (→ HTTPException) | Hecho |
| ✅ | **WEB-006**: Borrar perfil no deja rutinas huérfanas | Hecho |
| ✅ | **WEB-007**: Chat como vista dedicada (no panel embebido) | Hecho |
| ✅ | **WEB-009**: Precarga de datos del perfil guardado | Hecho |
| ✅ | **WEB-011**: Conversión ONNX de modelos + artefactos a JSON | Hecho |
| ✅ | **WEB-012**: Demo "Probar ya" con ONNX en el navegador | Hecho |
| ✅ | **WEB-013**: Accesibilidad (axe-core: 0 critical/serious) | Hecho |
| ✅ | **WEB-014**: i18n (ES/EN) con detección de idioma | Hecho |
| ✅ | **WEB-015**: Perfil en localStorage de la demo | Hecho |
| ✅ | **WEB-016**: LLM en navegador (transformers.js + Granite 1B) | Hecho |
| ✅ | **WEB-019**: Shaders WebGL en landings | Hecho |
| ✅ | **WEB-020**: GSAP animaciones + Space Grotesk | Hecho |
| ✅ | **WEB-021**: Paleta de climasafe.html en probar-ya | Hecho |
| ✅ | **WEB-022**: climasafe.html textos al ancho + botones agrupados | Hecho |
| ✅ | **UX-001**: Agente conversacional en GUI (estilo SymptomAI) | Hecho |
| ✅ | **FORECAST-001**: Tendencia semanal con bandas de confianza | Hecho |
| ✅ | **FORECAST-004**: Banda y horizonte explicados en pantalla | Hecho |

### Fase 8 — Mensajería y despliegue

| # | Qué | Estado |
|---|-----|--------|
| ✅ | **MSG-001**: Abstracción de mensajería (Telegram/Hermes/Webhook) | Hecho |
| ✅ | **MSG-003**: Worker de notificaciones con cola SQLite | Hecho |
| ✅ | **MSG-004**: Evaluaciones programadas sin Docker (worker) | Hecho |
| ✅ | **DEPLOY-001**: Agente de release automatizado | Hecho |
| ✅ | **DEPLOY-002**: CI/CD + GitHub Pages | Hecho |
| ✅ | **DEPLOY-003**: Modelo de releases (sin release-please) | Hecho |

### Fase 9 — Seguridad y datos

| # | Qué | Estado |
|---|-----|--------|
| ✅ | **SEC-001**: Protección BD (permisos, logs sanitizados, backup) | Hecho |
| ✅ | **MCP-002**: Solo lectura por defecto en MCP | Hecho |
| ✅ | **MCP-003**: Control de acceso por identidad en MCP | Hecho |
| ✅ | **BUG-001**: NaN en datos meteorológicos → propagación | Hecho |
| ✅ | **BUG-002**: Riesgo colectivo por etiqueta (F821) | Hecho |
| ✅ | **BUG-003**: Test del parte (role=user) | Hecho |
| ✅ | **BUG-005**: SFTConfig pickling en fine_tune.py | Hecho |
| ✅ | **BUG-006**: PosixPath.lower() en exportar_gguf | Hecho |

### Fase 10 — Investigación y documentación

| # | Qué | Estado |
|---|-----|--------|
| ✅ | **RESEARCH-001**: HMM, Bayesianas, GPs, GNNs, TFT, RL (viabilidad) | Hecho |
| ✅ | **FORECAST-002**: Modelos fundacionales (TimesFM, Granite, WeatherNext) | Hecho |
| ✅ | **FORECAST-003**: Estudio volumen de gente (APARCAR) | Hecho |
| ✅ | **DOC-002**: Quitar densidad a documentación y README | Hecho |
| ✅ | **DOC-003**: README con formato original | Hecho |
| ✅ | **DOC-004**: PRD de ClimaSafeAI | Hecho |
| ✅ | **DOC-005**: Sitio MkDocs Material | Hecho |
| ✅ | **DOC-006**: Actualizar sitio curado (docs_site/) | Hecho |
| ✅ | **DOC-007**: Dos niveles de documentación (usuarios + técnicos) | Hecho |
| ✅ | **DOC-008**: Clarificar dualidad factor edad (ensemble vs personalización) | Hecho |
| ✅ | **META-001**: Métricas de éxito del PRD en producción | Hecho |
| ✅ | **UV-001**: Radiación UV como línea futura (v2) | Hecho |
| ✅ | **ML-001**: Arquitectura ML plug-and-play (manifiestos JSON) | Hecho |

---

## Pendiente

### Prioritario (producción)

- **LLM-002**: Fine-tuning ejecutado + skill publicada + deploy verificado (bloqueado: sin GPU NVIDIA/CUDA)
- **PACK-002**: Empaquetado para técnicos y no técnicos (pip/npm/Docker)
- **CHAT-001**: /chat usa el pipeline (cuestionario conversacional con predicción real)
- Crear PAT `PAGES_DEPLOY_TOKEN` para publicación automática a GitHub Pages
- Regenerar dataset LLM completo y re-ejecutar QC
- Reentrenar LoRA sobre Qwen3-Instruct (siguiente paso LLM-014)

### Nice to have

- Exportar mapa como PNG/GeoJSON en la web (ya funcional desde MAPA-001)
- Grupos con comandos `/clima`, `/recomendaciones`
- Notificaciones programadas (ya funcional desde MSG-004, sin Docker)
- Flue Framework (ejecución durable, sandboxes)

---

## Resumen visual

```
Bot Telegram   ── Bot determinista 17 estados ✅ · Perfiles ✅ · Rutinas ✅ · Avisos ✅ · Voz ✅
LLM            ── Qwen3 local + Groq/OpenRouter remoto + fallback determinista ✅
RAG            ── sqlite-vec + distiluse embeddings ✅ · Eval set 43 preguntas ✅
MCP            ── 16+ tools ✅ · stdio/HTTP ✅ · Control acceso ✅ · MCP Apps ✅
Arnés          ── 26 agentes ✅ · Security layer ✅ · Subagentes ✅ · Compactación ✅
Web UI         ── Curvas edad ✅ · Mapa riesgo ✅ · Demo WASM+ONNX ✅ · LLM navegador ✅
Forecasting    ── Tendencia semanal ✅ · Bandas conformal ✅ · Open-Meteo 7 días ✅
ML             ── Plug-and-play (manifiestos) ✅ · Bayesiano jerárquico ✅
CI/CD          ── GitHub Actions ✅ · Pages ✅ · Releases ✅
Seguridad      ── BD permisos ✅ · Logs sanitizados ✅ · MCP solo-lectura ✅
```

## Referencias

- `conclusion-base-conocimiento.md` — decisión técnica de base de conocimiento
- `arquitectura/pipeline_prediccion.md` — flujo completo de predicción
- `riesgo/personalizacion_individual.md` — coeficientes de factores
- `riesgo/formulas_deterministas.md` — HI, WC, UV
- `ml/conclusiones_modelos.md` — métricas y comparación de modelos
- `ml/contrafactuales.md` — generación de contrafactuales
- `conformal_prediction.md` — split conformal
- `documentacion/prd.md` — Product Requirements Document
- `documentacion/componentes.md` — guía de componentes
