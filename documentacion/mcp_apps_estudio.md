# MCP Apps — Estudio y decisión de arquitectura (MCP-APPS-001)

Fecha: 2026-08-12 · Autor: implementer (arnés)

Evalúa llevar una UI a las tools MCP: en vez de devolver solo JSON, que un
host compatible (Claude Desktop, ChatGPT, Goose, VS Code Copilot...) renderice
una vista HTML en el chat. Se acota a UN MVP (`predict_risk_mcp`); las otras 11
tools quedan como recomendación.

## 1. Qué es MCP Apps

MCP Apps (SEP-1865 / extensión `io.modelcontextprotocol/ui`) es una extensión
opcional del protocolo MCP que permite a un servidor **servir HTML interactivo
como recurso** (`ui://...`) y enlazarlo a una herramienta. El host compatible:

1. pide la tool (que devuelve su resultado normal, JSON/texto);
2. ve en el `_meta.ui.resourceUri` de la tool qué recurso UI mostrar;
3. hace `resources/read` de ese `ui://` y obtiene el HTML;
4. lo renderiza en un iframe sandboxed dentro del chat;
5. se comunica con el HTML por **postMessage usando JSON-RPC 2.0** (el HTML
   actúa como un mini-cliente MCP contra el host).

El servidor no habla con la app: `ui/*` va entre el host y el iframe.

Negociación: el cliente declara la extensión en `initialize`:

```json
"capabilities": { "extensions": { "io.modelcontextprotocol/ui": {
    "mimeTypes": ["text/html;profile=mcp-app"] } } }
```

### Cómo se declara (resumen)

| Pieza | Dónde | Cómo |
|---|---|---|
| Tool con UI | `_meta.ui.resourceUri` | `{"ui": {"resourceUri": "ui://prediccion-riesgo"}}` |
| Recurso HTML | `ui://...` con MIME `text/html;profile=mcp-app` | `resources/read` lo devuelve |
| Perfil | MIME type (y meta tag `profile=mcp-app` pre-GA) | host solo renderiza ese MIME |
| Comunicación | postMessage JSON-RPC | puente mínimo en el HTML |

### Clientes que lo soportan hoy (2026)

Confirmado por la documentación oficial del MCP (modelcontextprotocol.io,
blog de MCP Apps 2026-01-26):

- **Claude / Claude Desktop** (soporte nativo, `create-mcp-app` skill)
- **VS Code GitHub Copilot**
- **Goose**
- **ChatGPT (Apps SDK)** — vía el "Apps SDK" propio de OpenAI; el protocolo
  MCP Apps como tal lo negocian los tres primeros.

Casys, Mastra y otros ecosistemas lo adoptan con el MIME
`text/html;profile=mcp-app`. MCP-UI mantiene un adaptador de compatibilidad.

## 2. ¿Existe SDK oficial de MCP Apps para Python?

**Sí, pero no en la versión instalada aquí, y con una API distinta.**

- El SDK oficial de MCP para Python (py.sdk.modelcontextprotocol.io) añadió
  la extensión **Apps** en `mcp.server.apps` (`Apps`, `APP_MIME_TYPE`,
  `client_supports_apps`) en la versión 1.29+, sobre la nueva clase servidor
  `mcp.server.mcpserver.MCPServer` con extensiones.
- Este proyecto usa **mcp==1.28.1 + FastMCP clásico** (`mcp.server.fastmcp`):
  `mcp.server.apps` NO existe en 1.28.1, y `MCPServer` tampoco.
- Verificado en el entorno:

```
mcp version: 1.28.1
mcp.server.apps -> ModuleNotFoundError
mcp.server.mcpserver -> ModuleNotFoundError
```

**Conclusión: en este proyecto el soporte MCP Apps se sirve a mano sobre
FastMCP** (tool con `meta={"ui": {"resourceUri": ...}}` + recurso `ui://` con
MIME `text/html;profile=mcp-app`). FastMCP 1.28.1 ya serializa `_meta`
correctamente en el wire y acepta recursos con ese MIME:

```python
@_mcp.tool(meta={"ui": {"resourceUri": "ui://prediccion-riesgo"}})
@_mcp.resource("ui://prediccion-riesgo", mime_type="text/html;profile=mcp-app")
```

Ambas verificadas con `list_tools`/`list_resources`/`resources/read` en tests.

## 3. El MVP: predict_risk_mcp con vista HTML

### Cómo funciona

1. `predict_risk_mcp` sigue devolviendo **el JSON de siempre**: un host sin
   soporte de apps no nota nada (degradación: recibe texto/JSON).
2. La tool se registra con `meta.ui.resourceUri = "ui://prediccion-riesgo"` y
   guarda el último resultado en `_UI_ESTADO["ultimo"]`.
3. El recurso `ui://prediccion-riesgo` (MIME `text/html;profile=mcp-app`)
   devuelve un HTML autocontenido que pinta el parte (`mostrarFinal`) y la
   gráfica de riesgo por hora (`mostrarGraficaRiesgo`).
4. Un host compatible renderiza ese HTML en el chat; el HTML responde pings
   JSON-RPC por postMessage (puente mínimo).

### Reutilización de la web, sin duplicar UI

Requisito duro del usuario: no rehacer la web, no duplicar la vista.

- La vista del recurso es una **plantilla HTML autocontenida en el backend
  Python** (`agents/tools/mcp_apps_vista.py`) con el CSS y el JS de
  visualización **inline** como constantes del módulo: `mostrarFinal`,
  `mostrarGraficaRiesgo`, `hiToNivel`, `_confianzaGlobal`... con las mismas
  firmas y comportamiento que las de `chat/static/index.html`, para que el
  parte pinte igual en el chat y en la web.
- `index.html` NO se toca: la web actual sigue funcionando sin cambios.
- Se descartó explícitamente la vía de un fichero `chat/static/js/*.js`
  compartido (primera iteración): la web no lo servía, quedaba un `.js`
  huérfano en un proyecto Python, y no compartía nada con index.html. La
  plantilla Python mantiene la vista en el backend y no ensucia el repo.

Rutas que lo demuestran:

```
chat/static/index.html          ← web actual (sin cambios)
agents/tools/mcp_apps_vista.py  ← plantilla HTML de la vista (CSS+JS inline)
agents/tools/prediction_mcp_tool.py
  └─ _html_vista_predict_risk() ← genera la vista desde mcp_apps_vista
  └─ @_mcp.resource("ui://prediccion-riesgo")
  └─ @_mcp.tool(meta={"ui": {"resourceUri": "ui://prediccion-riesgo"}})
tests/test_mcp_apps_ui.py       ← evidencia del criterio
```

## 4. Decisión de arquitectura

| Eje | Decisión | Por qué |
|---|---|---|
| Servidor | FastMCP 1.28.1 + recursos `ui://` a mano | El SDK oficial Apps exige mcp 1.29+ / MCPServer; subirlo y migrar FastMCP es una feature aparte, no el MVP de un estudio |
| Recurso | `ui://prediccion-riesgo`, `text/html;profile=mcp-app` | MIME que negocian los hosts; `resources/read` ya lo devuelve |
| Tool | `meta={"ui": {"resourceUri": ...}}` sin cambiar el retorno JSON | Degradación natural: el host sin apps recibe el mismo JSON que antes |
| UI | Plantilla Python `agents/tools/mcp_apps_vista.py` (CSS+JS inline) | La vista vive en el backend, sin `.js` suelto; mismas firmas que la web |
| Comunicación | Puente postMessage JSON-RPC mínimo (ping + initialized) | Suficiente para que un host compatible renderice; sin SDK Apps |
| Alcance | Solo `predict_risk_mcp` | El resto de tools, estudio (abajo) |

### Sobre modelos / API / cliente

- **Modelos**: no se tocan. La vista consume `result` de `predict_risk`, con
  las mismas claves que la web (`weather.perfil_horario`, `riesgo_horario`,
  `perfil_usuario`, `modelos[*].conformal_confianza`, `clase_final_label`).
- **API**: el JSON del recurso se inyecta como `window.RIESGO_DATA`; Chart.js
  se carga del mismo CDN que la web (jsdelivr). No hay endpoint nuevo.
- **Cliente**: la verificación se hace con el protocolo (list/read de
  recursos, MIME y `_meta`); el render visual se demuestra con el HTML
  generado y el puente postMessage. Un host real (Claude Desktop) lo
  renderizaría en el iframe del chat.

## 5. ¿Merece la pena migrar el resto de tools?

**No, al menos no ahora (2026-08).** Razones:

1. El SDK oficial Apps en Python acaba de salir (mcp 1.29+) y no está en el
   lock del proyecto. Migrar a `MCPServer`+extensiones es una feature de
   infraestructura (cambio de API servidor), no 11 tickets de UI.
2. Solo 2 de las 12 tools tienen salida "visual" natural:
   - `predict_risk_mcp` → parte + gráfica: **candidata** (ya es el MVP).
   - `grafica_riesgo_horario_mcp` → ya devuelve una imagen PNG; la vista HTML
     añadiría interactividad (tooltip por hora) si el usuario lo pide.
   - `predict_zone_risk` (grid) → podría pintar el mapa Leaflet, pero el mapa
     de la web es un endpoint propio (`/api/riesgo-zona`); exigiría mover
     lógica de backend al MCP.
3. El resto de tools son CRUD de perfiles/rutinas/avisos: el JSON es más
   claro que una UI para el agente, y la UI no añade valor de lectura.
4. La degradación está garantizada por diseño (JSON si el host no soporta
   apps), así que migrar es aditivo y sin riesgo — se puede hacer tool a tool
   cuando haya demanda real.

**Orden sugerido si se decide migrar:** actualizar `mcp` a 1.29+, migrar el
servidor a `MCPServer`+`Apps`, y entonces `predict_risk_mcp` (hecho),
`grafica_riesgo_horario_mcp` y `predict_zone_risk` como siguientes.

## 6. Limitaciones del MVP

- El recurso pinta el **último** resultado de `predict_risk_mcp` (patrón del
  ejemplo del reloj de la doc oficial: la tool deja el dato, el recurso lo
  renderiza). Entre llamadas concurrentes podría pisarse el estado; el patrón
  de recursos tempate/por-llamada es una mejora futura del SDK Apps.
- Chart.js se carga desde CDN; si el host bloquea la red del iframe, la
  gráfica no pinta (el parte sí, es HTML/CSS puro).
- El puente postMessage es mínimo: responde pings y notifica `initialized`.
  Llamadas tool desde la UI (visibility `app`) quedan para cuando haya SDK.