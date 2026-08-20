# Arquitectura

## El flujo de una predicción

```
Ubicación (provincia o lat/lon) + fecha (hoy/mañana)
        │
        ▼
Datos meteorológicos ── Open-Meteo (forecast + archive, sin API key)
        │  (si es hoy: observación actual; hasta 7+ días de forecast)
        ▼
Features ── índices térmicos (Heat Index, Wind Chill, WBGT) + distribución
        diaria (media/desv/mín-máx, horas sobre/bajo umbral) + persistencia
        temporal (lags, medias móviles, rachas de días fríos)
        │
        ▼
Modelos ── XGBoost (calor) + RandomForest (frío) + LSTM province_hybrid
        (LSTM + embedding de provincia + INE + features diarias)
        + Fórmula determinista (índices clásicos)
        │
        ▼
Ensemble conformal-weighted ── media ponderada por tamaño del prediction
        set conformal (1/set_size) → probabilidad de riesgo poblacional
        │
        ▼
Personalización ── factores multiplicativos en odds (edad, comorbilidades,
        medicación, actividad, aclimatación, situación social…), cap ×3.0
        │
        ▼
Clase final ── SEGURO / PRECAUCIÓN / PELIGRO, con overrides físicos
        por Heat Index / Wind Chill / UV + recomendaciones
```

## Datos

| Fuente | Uso |
|---|---|
| **ERA5 (ECMWF)** | Histórico para entrenamiento (España, ≥10 años) |
| **Open-Meteo** | Pronóstico y archivo en producción (sin API key) |
| **AEMET OpenData** | Datos oficiales de España |
| **OpenUV** | Índice UV por coordenada (opcional) |
| **MoMo (ISCIII)** | Mortalidad diaria atribuible a calor (X30) y frío (X31) — **target del modelo** |

El target del modelo son percentiles de mortalidad atribuida de MoMo,
calculados **por provincia** (para no penalizar a las provincias pequeñas); el
split es **por fecha** (no aleatorio) para no filtrar días de la misma ola entre
train y test.

## Componentes

| Componente | Qué es |
|---|---|
| **Web UI** (`chat/`) | Formulario completo (individual/grupo), mapa, perfil horario, tendencia semanal con bandas conformal, mapa de riesgo por zona, chat conversacional |
| **Demo ONNX/WASM** (`web/probar-ya/`) | El mismo pipeline portado a JavaScript y ejecutado **en el navegador** con onnxruntime-web; los modelos viajan como ONNX |
| **Bot de Telegram** | Formulario determinista con botones + avisos diarios + respuestas redactadas por un LLM local fine-tuneado (Ollama) |
| **MCP servers** | Tools para asistentes de IA: predecir riesgo, perfiles, rutinas, gráfica |
| **RAG vectorial** | Factores y documentación indexados con sqlite-vec para responder con citas |

## MCP: control de acceso y estándares

El servidor MCP de predicción (`agents/tools/prediction_mcp_tool.py`) cerró el
agujero de identidad: **ninguna tool sabía quién llamaba**, y cuatro de ellas
leían o reasignaban perfiles ajenos con solo saber el alias. Ahora toda tool
pasa por un control de acceso en un punto único (`_requiere_identidad`):

- **Identidad por proceso en stdio** (`CLIMASAFE_MCP_TOKEN` o `--identidad`),
  **bearer en HTTP** — un proceso = un llamante; el transporte se resuelve en
  un solo sitio.
- **`uid` opaco** (`usr_…`) sustituye a `alias`/`chat_id` como llave de
  acceso; la credencial (`mcp_token_hash`) nunca sale en las respuestas.
- **Minimización de campos**: un perfil ajeno no devuelve ni un campo; de un
  perfil propio nunca salen `farmacos`, `comorbilidades`, `situacion_social`,
  grasa, fototipo ni coordenadas.
- **Solo lectura por defecto (MCP-002)**: las 5 tools que escriben en la BD
  (`crear_perfil_mcp`, `crear_rutina_mcp`, `borrar_rutina_mcp`,
  `vincular_chat_id_mcp`, `configurar_hora_aviso_mcp`) exigen
  `CLIMASAFE_MCP_WRITE_TOKEN` al arrancar; sin él responden error y no tocan
  nada. El token nunca entra en la firma de una tool ni en los logs.
- **Spec 2025-06-18+ (MCP-004)**: ambos servidores (predicción y factores)
  usan streamable HTTP (el de factores migró de SSE) y declaran tool
  annotations (`title`, `readOnlyHint`, `destructiveHint`).

> Detalle: [`documentacion/arquitectura/control_acceso_mcp.md`](https://github.com/ANFAIA/ClimaSafe/blob/main/documentacion/arquitectura/control_acceso_mcp.md)
> y [`documentacion/arquitectura/auditoria-mcp-espec.md`](https://github.com/ANFAIA/ClimaSafe/blob/main/documentacion/arquitectura/auditoria-mcp-espec.md)

## Persistencia

SQLite (`data/climasafe.db`): perfiles, rutinas, consultas y RAG vectorial.

## Despliegue

- Repositorio principal: [ANFAIA/ClimaSafe](https://github.com/ANFAIA/ClimaSafe)
  (CI: tests + lint; release con tag semántico; publicación a GitHub Pages).
- Sitio web + demo: [cacelass.github.io/climasafe](https://cacelass.github.io/climasafe/),
  estático y autocontenido (la demo incluye los modelos ONNX y el runtime WASM).
