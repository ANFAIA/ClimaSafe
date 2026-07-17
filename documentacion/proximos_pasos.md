# Próximos pasos — hoja de ruta post-mentoría

**Fecha:** 2026-07-16 · **Última revisión:** 2026-07-17 · **Estado:** decisiones tomadas

> Síntesis de la mentoría del 16/07/2026 y sesión de planificación del 17/07/2026.
> Las decisiones técnicas se recogen en `conclusion-base-conocimiento.md`.
> La documentación organizada en `README.md` (índice general).

---

## 1. Base de conocimiento de usuarios

Guardar perfiles, historial de consultas y evolución del usuario para que el
sistema se adapte con el tiempo.

### Decisión

| Aspecto | Elección | Por qué |
|---------|----------|---------|
| Motor | **SQLite** + **sqlite-vec** | OLTP, transaccional, embebible, búsqueda semántica nativa |
| Interfaz LLM | **MCP server propio** (Python FastMCP) | Control total sobre esquema y validaciones |
| Contexto LLM | **Markdown generado desde SQL** | Legible para el LLM, inyectable en prompt |

Ver `conclusion-base-conocimiento.md` para la justificación completa.

### Requisitos funcionales

- **Aviso al usuario**: notificar qué datos se guardan y por qué (transparencia).
- **Adaptación temporal**: si el usuario dice "no aclimatado" y pasan ~2 semanas,
  el sistema debería inferir posible adaptación y preguntar si actualizar el perfil.
  - Evidencia: la aclimatación significativa ocurre en ~7 días y es completa
    en ~14 días (CDC/NIOSH; Tyler et al. 2024). Ver
    `papers/aclimatacion/aclimatacion-calor-tiempos.md` para fuentes completas.
- **Contexto para el LLM**: los datos guardados se inyectan como contexto al
  agente LLM para personalizar respuestas.

### Patrón de escritura

1. **LLM sugiere** cambios de perfil vía tool MCP (`suggest_profile_update`).
2. **Middleware valida** la sugerencia (tipos, rangos, consistencia).
3. **Sistema pregunta** al usuario si confirma el cambio.
4. **Usuario confirma** → se escribe en SQLite. Rechaza → se descarta o replanifica.

Esto evita escrituras inconsistentes del LLM y da control al usuario.

### Integración

- **MCP tools** para que el agente de papers añada/modifique factores de riesgo.
- **Endpoint REST** (`GET /api/factores`) para que el frontend HTML los consuma.
- SQLite como fuente de verdad única para ambas interfaces.

---

## 2. Agente de búsqueda de papers

Agente autónomo que monitorea nuevas publicaciones sobre:

- **Factores de riesgo** (aclimatación, medicación, comorbilidades) → `papers/`.
- **Modelos alternativos** (Transformers, GNN, diffusion) → `modelos/`.
- Nuevos índices biometeorológicos.
- Forecasting térmico y de mortalidad por calor/frío.
- Modelos de ML aplicados a salud ambiental.

### Funcionamiento

1. **Fuentes**: arXiv API, PubMed API, Google Scholar.
2. **Frecuencia**: **cada 2 semanas** vía GitHub Actions (cron).
3. **Output**:
   - Resumen ejecutivo del paper.
   - Clasificación automática: ¿factor de riesgo o modelo?
   - Recomendación de si vale la pena incorporarlo.
   - Enlace DOI / URL.
4. **Acción**:
   - Si es **factor de riesgo**: genera markdown en `papers/` y lo sugiere como nuevo coeficiente.
   - Si es **modelo**: guarda en `modelos/` y **notifica al usuario** para que lo explore en notebook.
   - En ambos casos, alimenta **graphify** para consultas posteriores.

### Infraestructura requerida

- API key de LLM para el agente (ver §3).
- GitHub Actions con cron semanal (o quincenal).

---

## 3. Infraestructura LLM

### Opciones de proveedor

| Opción | Descripción | URL |
|--------|-------------|-----|
| **LiteLLM** | SDK unificado para 100+ LLMs (OpenAI, Anthropic, Ollama, etc.). Permite cambiar de proveedor sin reescribir código. | https://www.litellm.ai/ |
| **Ollama** | LLM local (open-source). Sin coste, sin dependencia externa. | https://ollama.ai |
| **OpenAI API** | Remoto (GPT-4o, GPT-4-turbo). Robusto pero con coste. | https://platform.openai.com |
| **Anthropic API** | Remoto (Claude). Bueno para análisis largo. | https://console.anthropic.com |

### Decisión

| Uso | Elección | Alternativa local |
|-----|----------|-------------------|
| Chat con usuario | **Gemini 2.0 Flash** (cloud, gratis) | Ollama (privacidad total) |
| Agente de papers (batch) | **Gemini 2.0 Flash** o Llama 3.1 8B local | Según hardware disponible |
| Proxy unificador | **LiteLLM** | Misma interfaz para cualquier proveedor |

- **Por defecto**: Gemini 2.0 Flash (gratis, tool calling, suficiente para agentes).
- **Para privacidad**: el usuario puede cambiar a Ollama local vía config de LiteLLM.
- **Abstracción**: todo el código apunta a `http://localhost:4000` (LiteLLM).
  Cambiar de proveedor es cambiar variables de entorno, no código.
- **Privacidad ≠ obligación**: se documentan ambas opciones, ninguna se impone.

### LLM en la entrada del usuario

Modelo pequeño (tipo instrucción) que se ejecute en tiempo real para:

- Detectar errores de entrada del usuario (ej. "no tengo comorbilidades"
  → parsear a set vacío).
- Interpretar lenguaje natural en campos libres.
- Sugerir correcciones o aclaraciones al usuario.

---

## 4. Factores de riesgo dinámicos

Los factores de riesgo y comorbilidades se gestionan desde datos, no desde el HTML.

### Enfoque

- **Tabla SQLite `factores_riesgo`**: nombre, categoría, coeficiente, fuente (DOI), activo.
- **Endpoint REST** `GET /api/factores`: el frontend HTML consume los factores disponibles.
- **MCP tool** `suggest_new_factor`: el agente de papers propone nuevos factores.
- **Sin sobreingeniería**: la UI no se auto-modifica. Cuando añades un factor en SQLite,
  el frontend lo refleja al recargar.

### Modificación de valores en el tiempo

- Si el usuario indicó `aclimatado = false` y han pasado ≥14 días con
  temperaturas similares, el sistema pregunta: "Han pasado X semanas,
  ¿sigues sin estar aclimatado?". Fundamento: aclimatación completa
  documentada en ~14 días (ver `papers/aclimatacion/aclimatacion-calor-tiempos.md`).
- Lo mismo para peso corporal, nivel de actividad, etc.

---

## 5. Investigación de modelos alternativos

### Carpeta `documentacion/modelos/`

Cada modelo explorado tendrá su propia carpeta con:

- Paper original (resumen + DOI).
- Análisis de compatibilidad con ClimaSafeAI.
- Esfuerzo estimado de integración.
- Estado: `seleccionado`, `pendiente`, `descartado` + motivo.

### Modelos a explorar

| Modelo | Potencial aplicación |
|--------|---------------------|
| **Transformers (TimeSFormer, PatchTST)** | Forecasting de series térmicas |
| **GNN (Graph Neural Networks)** | Modelar correlación espacial entre provincias vecinas |
| **N-BEATS / N-HiTS** | Modelos puramente ML para forecasting univariante |
| **Diffusion probabilístico** | Generación de escenarios de calor extremo |

### Flujo

1. **Agente de papers** encuentra un modelo con potencial → lo clasifica en `modelos/`.
2. **Notifica al usuario**: "Este modelo podría mejorar XGBoost en Y contexto".
3. **El usuario lo explora** en un notebook (aprender, no implementar).
4. Si es prometedor → se diseña un experimento formal comparando vs XGBoost/RF/LSTM
   (ver `ml/conclusiones_modelos.md`).

---

## 6. Calibración / ajuste del sistema

### Calibración de probabilidades

- **Isotonic regression** (recomendada para el volumen de datos actual).
- Platt scaling como alternativa si hay pocos datos.
- Evaluar calibración con diagramas de fiabilidad y Brier score.

### Ajuste dinámico de umbrales

- Actualmente umbrales fijos: SEGURO < 0.35, PRECAUCION 0.35-0.65,
  PELIGRO > 0.65.
- Posible evolución: umbrales personalizados por perfil de usuario (ej.
  población mayor: umbral más bajo).

### Enfoques no técnicos

- Entrevistas con usuarios reales (prototipo).
- Tests de usabilidad de la interfaz.
- Evaluación con expertos en salud laboral / protección civil.

---

## 7. Dockerización

### Stack completo

| Servicio | Imagen | Función |
|----------|--------|---------|
| **api** | Python (FastAPI) | Web + API + modelos ML + MCP server |
| **llm-proxy** | `ghcr.io/berriai/litellm:main` | Proxy unificado para Gemini/Ollama |
| **ollama** (opcional) | `ollama/ollama` | LLM local para privacidad total |
| **SQLite** | Volumen persistente | Base de datos de perfiles y factores |

### docker-compose.yml preliminar

```yaml
services:
  api:
    build: .
    ports: ["8080:8080"]
    volumes:
      - ./data:/data  # SQLite persistente
    env_file: .env
  llm-proxy:
    image: ghcr.io/berriai/litellm:main
    ports: ["4000:4000"]
    env_file: .env
```

### Agente de papers

- **Opción A**: contenedor separado con cron (GitHub Actions no aplica si es on-prem).
- **Opción B**: GitHub Actions con schedule quincenal (más simple si el repo está en GitHub).

Ver `arquitectura/agentes_ia.md` §docker_agent para helpers existentes.

---

## 8. Acciones inmediatas

| # | Tarea | Prioridad | Depende de |
|---|-------|-----------|------------|
| 1 | ✅ Decidir tecnología de base de conocimiento (SQLite + sqlite-vec + MCP) | Hecho | — |
| 2 | ✅ Definir proveedor LLM (LiteLLM + Gemini cloud, Ollama opcional) | Hecho | — |
| 3 | **Diseñar esquema SQLite de perfiles** | Alta | — |
| 4 | **Implementar MCP server en Python** (FastMCP: get_profile, suggest_update, confirm_update) | Alta | #3 |
| 5 | **Crear endpoint REST** `GET /api/factores` para frontend | Alta | #3 |
| 6 | Organizar `documentacion/` (índice, subcarpetas papers/, modelos/) | Media | — |
| 7 | Implementar agente de papers v1 (crawl + clasificar en papers/ o modelos/) | Media | #2 |
| 8 | Adaptación automática de perfil por tiempo transcurrido | Media | #3, #4 |
| 9 | Conectar agente de papers a graphify | Media | #7 |
| 10 | Crear carpeta `modelos/` con papers de referencia | Baja | — |
| 11 | Calibración de probabilidades (isotonic regression) | Baja | — |
| 12 | Dockerizar stack completo | Baja | #3, #4 |

---

## Referencias

- `conclusion-base-conocimiento.md` — decisión técnica de base de conocimiento.
- `arquitectura/arquitectura/agentes_ia.md` — arquitectura de agentes existente en el repositorio.
- `ml/ml/conclusiones_modelos.md` — métricas y comparación de modelos actuales.
- `arquitectura/arquitectura/diseño_modelo.md` — diseño general del sistema de predicción.
- `README.md` — índice general de toda la documentación.
- `papers/aclimatacion/aclimatacion-calor-tiempos.md` — evidencia de aclimatación.
- `https://www.litellm.ai/` — LiteLLM: SDK unificado para 100+ LLMs.
- `https://github.com/berriai/litellm` — repo de LiteLLM.
- `https://python.langchain.com/docs/integrations/providers/litellm/` — integración LangChain + LiteLLM.
