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

## Persistencia

SQLite (`data/climasafe.db`): perfiles, rutinas, consultas y RAG vectorial.

## Despliegue

- Repositorio principal: [ANFAIA/ClimaSafe](https://github.com/ANFAIA/ClimaSafe)
  (CI: tests + lint; release con tag semántico; publicación a GitHub Pages).
- Sitio web + demo: [cacelass.github.io/climasafe](https://cacelass.github.io/climasafe/),
  estático y autocontenido (la demo incluye los modelos ONNX y el runtime WASM).
