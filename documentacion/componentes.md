# Componentes

Componentes que rodean al pipeline ML de `climasafeai/`: canales de consulta,
interfaces para asistentes y el arnés de desarrollo.

---

## Bot de Telegram

Bot determinista que recoge datos mediante teclados inline (sin LLM para la
entrada). Tiene 17 estados que cubren desde sexo, edad y grasa corporal hasta
comorbilidades, medicación, situación social y ubicación. Al final genera una
respuesta con plantilla. Soporta perfiles persistentes en SQLite: si un chat
ya tiene perfil vinculado, salta las preguntas personales en futuras consultas.
También pregunta si quiere guardar el perfil al terminar.

## Web UI

Interfaz web con formulario completo, selector de ubicación sobre mapa,
selector de perfil guardado, mapa de riesgo por zona (grid de celdas
alrededor de un punto), curvas de riesgo comparativas por edad, y estimación
de volumen de afectados para eventos.

## MCP (Model Context Protocol)

Dos servidores para usar ClimaSafe desde asistentes (Claude Desktop, etc.):
uno de predicción (predecir riesgo, crear, cargar y vincular perfiles) y otro
de factores (consultar la base de factores de riesgo, buscarlos, aprobarlos
y hacer búsqueda semántica sobre la base de conocimiento).

Adaptados al spec MCP **2025-06-18+** (SDK `mcp>=1.28.1`):

- **Tool annotations**: todas las tools llevan `title`; las de lectura llevan
  `readOnlyHint` y las destructivas (`borrar_rutina_mcp`, `reject_factor_mcp`)
  llevan `destructiveHint`. Auditoría completa en
  `documentacion/arquitectura/auditoria-mcp-espec.md`.
- **Transporte**: ambos soportan `--stdio` (cliente `.mcp.json`) y **streamable
  HTTP** además de él. `make mcp-http` sirve la predicción en
  `http://localhost:8101/mcp` y `make mcp-factors` los factores en
  `http://localhost:8100/mcp`.

## RAG vectorial

Los factores de riesgo y su documentación se indexan con sqlite-vec
(embeddings semánticos). Esto permite responder preguntas como "qué dice la
literatura sobre los antipsicóticos en olas de calor" citando las fuentes.

## Arnés de desarrollo

El proyecto incluye un sistema de 26 agentes Python que orquestan el ciclo de
desarrollo: cada feature se abre, implementa, revisa y cierra con verificación
automática (init.sh + suite de tests). El backlog y el progreso viven en
featureslist.json y progress/ respectivamente.
