# Componentes

Componentes que rodean al pipeline ML de `climasafeai/`: canales de consulta,
interfaces para asistentes, el arnés de desarrollo y los servicios de soporte.

---

## Bot de Telegram

Bot determinista que recoge datos mediante teclados inline (sin LLM para la
entrada). Tiene 17 estados que cubren desde sexo, edad y grasa corporal hasta
comorbilidades, medicación, situación social y ubicación. Al final genera una
respuesta con plantilla. Soporta perfiles persistentes en SQLite: si un chat
ya tiene perfil vinculado, salta las preguntas personales en futuras consultas.
También pregunta si quiere guardar el perfil al terminar.

**Características principales:**
- Formulario de 17 campos con botones nativos de Telegram
- Perfiles persistentes vinculados al chat_id
- Repetir última salida al cargar perfil
- Memoria conversacional (entender "voy al tenis como ayer")
- Voz: notas de audio transcritas con Whisper + parte hablado con gTTS
- Rutinas semanales con deporte/ocupación
- Avisos diarios del riesgo programados
- Logging seguro (sin token, sin duplicados)
- LLM remoto (Groq/OpenRouter) + fallback determinista
- LLM local (Ollama qwen3:climasafe) cuando está disponible

## Web UI

Interfaz web SPA con formulario completo, selector de ubicación sobre mapa,
selector de perfil guardado, mapa de riesgo por zona (grid de celdas
alrededor de un punto), curvas de riesgo comparativas por edad, estimación
de volumen de afectados para eventos, y tendencia semanal con bandas de
confianza.

**Características principales:**
- Modo Individual / Grupo / Chat
- Agente conversacional en GUI (estilo SymptomAI)
- Chat como vista dedicada (no panel embebido)
- Accesibilidad WCAG 2.1 AA (axe-core: 0 critical/serious)
- i18n (ES/EN) con detección automática de idioma
- Perfil en localStorage (demo "Probar ya")
- Exportar mapa como PNG/GeoJSON
- Tendencia semanal con banda conformal explicada

## Demo "Probar ya" (WASM + ONNX)

Demo estática que ejecuta el pipeline completo en el navegador sin backend:
XGBoost, RandomForest y LSTM convertidos a ONNX y ejecutados con
onnxruntime-web. Incluye LLM local (Granite 1B via transformers.js) para
recomendaciones contextuales.

**Archivos:** `web/probar-ya/`
**Documentación:** `documentacion/wasm/estudio_wasm.md`, `documentacion/wasm/llm_navegador.md`

## MCP (Model Context Protocol)

Dos servidores para usar ClimaSafe desde asistentes (Claude Desktop, etc.):
uno de predicción (predecir riesgo, crear, cargar y vincular perfiles, rutinas,
gráfico horario) y otro de factores (consultar la base de factores de riesgo,
buscarlos, aprobarlos y hacer búsqueda semántica sobre la base de conocimiento).

**Adaptados al spec MCP 2025-06-18+** (SDK `mcp>=1.28.1`):
- **Tool annotations**: todas llevan `title`; las de lectura llevan `readOnlyHint`
- **Transporte**: ambos soportan `--stdio` y streamable HTTP
- **Control de acceso**: solo lectura por defecto (MCP-002), escritura con token
- **Identidad opaca**: cada llamante ve solo lo suyo (MCP-003)
- **MCP Apps**: UI interactiva reutilizando la web (MCP-APPS-001)

## RAG vectorial

Los factores de riesgo y su documentación se indexan con sqlite-vec
(embeddings semánticos con distiluse-base-multilingual-cased-v2, 512d). Esto
permite responder preguntas como "qué dice la literatura sobre los
antipsicóticos en olas de calor" citando las fuentes.

**Características:**
- Reindexa cuando cambia el texto (hash sha256 por fragmento)
- Filtra literatura del dominio vs ruido ml/interno (umbral 0.50)
- Eval set con 43 preguntas etiquetadas para medir recall/precision
- Coeficientes y DOI indexados en los factores

**Documentación:** `climasafeai/db/rag.py`, `documentacion/rag_006_comparativa_embeddings.md`

## LLM y Fine-tuning

Sistema de LLM con múltiples capas de servicio:
1. **Fine-tuneado** (qwen3:climasafe via Ollama) — mejor experiencia
2. **Instruct remoto** (Groq/OpenRouter/Gemini free) — sin GPU local
3. **Determinista** (plantilla) — fallback seguro sin LLM

**Herramientas:**
- Generación de datasets sintéticos (calor + frío desde parquet reales)
- Fine-tuning LoRA/QLoRA (notebook Colab + script local)
- Benchmark de modelos gratuitos (`reports/benchmark_llm019.json`)
- Contador de tokens y coste por petición
- Tope de presupuesto configurable

**Documentación:** `documentacion/llm/`

## Mensajería y notificaciones

Abstracción de mensajería (`climasafeai/bot/messaging.py`) con interfaz común
para Telegram, Hermes y Webhook. Workers de notificaciones con cola SQLite
y evaluaciones programadas (crontab configurable sin Docker).

**Componentes:**
- `MessageAdapter` ABC con `send()` y `send_batch()`
- Adaptadores: `TelegramAdapter`, `HermesAdapter`, `WebhookAdapter`
- Cola SQLite con reserva atómica (WAL + synchronous=NORMAL)
- Workers en paralelo via asyncio.gather
- Programador de evaluaciones con crontab configurable

**Documentación:** `documentacion/bot/despliegue_llm_remoto.md`

## Arnés de desarrollo

El proyecto incluye un sistema de 26 agentes Python que orquestan el ciclo de
desarrollo: cada feature se abre, implementa, revisa y cierra con verificación
automática (init.sh + suite de tests). El backlog y el progreso viven en
featureslist.json y progress/ respectivamente.

**Características:**
- Security/Policy Layer (permisos, sandbox, auditoría en JSONL)
- Subagentes con contexto vacío y compactación
- Bucle propio en Python sobre LiteLLM
- Contador de tokens y coste por petición
- Commit automático al cerrar ticket (con --commit)
- Candado por dueño (dos asistentes en paralelo)

**Documentación:** `AGENTS.md`, `documentacion/arnes_comparativa.md`

## CI/CD y despliegue

Tres workflows de GitHub Actions (CI, release y GitHub Pages) con detalle
completo en `.github/workflows/README.md`.

**Componentes:**
- CI: test + lint en PR/push main
- Release: tag semántico v<version> + CHANGELOG + release notes
- Pages: publicación automática a GitHub Pages
- Deploy de cero verificado (sin Docker)

**Documentación:** `.github/workflows/README.md`, `documentacion/despliegue/verificacion-deploy-cero.md`

## Seguridad

- BD con permisos 600 y gitignore
- Logs sanitizados (sin token, sin datos de salud, chat_id enmascarado)
- MCP solo lectura por defecto (escritura con token)
- Control de acceso por identidad en MCP (IDs opacos)
- Security layer para tools del LLM (catalogo cerrado por agente)
- Backup y restore de la BD verificados

**Documentación:** `documentacion/seguridad_bd.md`, `documentacion/arquitectura/control_acceso_mcp.md`
