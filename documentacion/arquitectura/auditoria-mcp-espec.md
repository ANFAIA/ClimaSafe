# Auditoría MCP contra el spec vigente (MCP-004)

Revisión de los dos servidores MCP de ClimaSafeAI contra el protocolo MCP
**2025-06-18+**. Fecha: 2026-08-17. SDK: `mcp>=1.28.1` (resuelto a 1.28.1).

## 1. Versión del spec soportada

- SDK instalado: `mcp 1.28.1`, que implementa el spec 2025-06-18+ (tool
  annotations, streamable HTTP, versión de protocolo `2025-06-18`).
- Ambos servidores negocian `protocolVersion` a través de la biblioteca; el
  transporte streamable HTTP acepta el handshake `initialize` y responde con
  `protocolVersion = "2025-06-18"` (verificado en `tests/test_mcp_annotations.py`).

## 2. Qué usa cada servidor hoy

| Servidor | Fichero | Transporte (antes) | Transporte (ahora) | Tools |
|----------|---------|--------------------|--------------------|-------|
| Predicción | `agents/tools/prediction_mcp_tool.py` | stdio + HTTP **streamable** (`streamable_http_app`) | stdio + HTTP **streamable** (sin cambio) | 12 |
| Factores | `agents/tools/factors_mcp_tool.py` | stdio + HTTP **SSE** (`sse_app`) | stdio + HTTP **streamable** (`streamable_http_app`) | 14 |

Hallazgos de la auditoría:

- **Predicción** ya usaba streamable HTTP (`run_mcp_server` → `_mcp.streamable_http_app()`)
  y `make mcp-http` ya servía `/mcp` con Streamable HTTP. Solo faltaban annotations.
- **Factores** usaba SSE (`_mcp.sse_app()`), que quedó **obsoleto** en el spec
  2025-06-18+. Se migró a `streamable_http_app()` y `make mcp-factors` ahora
  sirve `/mcp` con Streamable HTTP.
- Ningún servidor declaraba tool annotations (`title`, `readOnlyHint`,
  `destructiveHint`) — se añadieron en esta feature.

## 3. Mapeo de annotations por tool

Annotation añadida vía `annotations=ToolAnnotations(...)` en `@_mcp.tool(...)`.
Semántica (spec 2025-06-18): `title` = título legible; `readOnlyHint=true` =
la tool no modifica el entorno; `destructiveHint=true` = puede hacer cambios
destructivos. Las tools de escritura no llevan ninguna de las dos (default).

### Servidor de predicción

| Tool | title | readOnlyHint | destructiveHint |
|------|-------|--------------|-----------------|
| `predict_risk_mcp` | Predecir riesgo cardiovascular | ✔ | — |
| `listar_usuarios_mcp` | Listar usuarios | ✔ | — |
| `cargar_perfil_mcp` | Cargar perfil | ✔ | — |
| `cargar_perfil_por_chat_id_mcp` | Cargar perfil por chat de Telegram | ✔ | — |
| `vincular_chat_id_mcp` | Vincular chat de Telegram | — | — |
| `crear_perfil_mcp` | Crear perfil | — | — |
| `listar_rutinas_mcp` | Listar rutinas | ✔ | — |
| `crear_rutina_mcp` | Crear rutina | — | — |
| `borrar_rutina_mcp` | Borrar rutina | — | ✔ |
| `configurar_hora_aviso_mcp` | Configurar hora de aviso | — | — |
| `riesgo_rutinas_dia_mcp` | Riesgo de rutinas del día | ✔ | — |
| `grafica_riesgo_horario_mcp` | Gráfica de riesgo horario | ✔ | — |

### Servidor de factores

| Tool | title | readOnlyHint | destructiveHint |
|------|-------|--------------|-----------------|
| `get_factors_mcp` | Obtener factores | ✔ | — |
| `suggest_factor_mcp` | Sugerir factor | — | — |
| `approve_factor_mcp` | Aprobar factor | — | — |
| `reject_factor_mcp` | Rechazar factor | — | ✔ |
| `update_factor_mcp` | Actualizar factor | — | — |
| `pending_factors_mcp` | Factores pendientes | ✔ | — |
| `check_acclimatization_mcp` | Comprobar aclimatación | ✔ | — |
| `auto_acclimatize_mcp` | Auto-aclimatar perfiles | — | — |
| `search_factors_mcp` | Buscar factores | ✔ | — |
| `search_documentos_mcp` | Buscar documentos | ✔ | — |
| `search_all_mcp` | Buscar todo | ✔ | — |
| `ask_rag_mcp` | Preguntar RAG | ✔ | — |
| `ask_qwen_rag_mcp` | Preguntar RAG con Qwen | ✔ | — |
| `qwen_raw_mcp` | Preguntar a Qwen | ✔ | — |

## 4. Qué falta / límites

- **Annotations**: completas. No se usaron `idempotentHint` ni `openWorldHint`
  (el spec las define pero la feature solo pedía title/readOnly/destructive).
- **Transporte**: ambos servidores soportan `--stdio` y streamable HTTP.
- **Identidad/permisos (MCP-002/MCP-003)**: intocadas — solo metadatos.
- **`.mcp.json`**: la configuración de cliente sigue siendo `--stdio` (la única
  que soportan Claude Desktop/Cursor); la alternativa streamable HTTP queda
  documentada en `documentacion/componentes.md` y en el `Makefile`
  (`make mcp-http`, `make mcp-factors`).

## 5. Verificación

- `tests/test_mcp_annotations.py`: 10 tests (title/readOnly/destructive + 2 de
  arranque streamable HTTP con TestClient sobre `streamable_http_app`).
- Arranque real verificado: `curl -X POST http://127.0.0.1:<port>/mcp` con el
  handshake `initialize` devuelve `HTTP 200` en ambos servidores.
