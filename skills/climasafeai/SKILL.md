---
name: climasafeai
description: Predicción de riesgo térmico personalizado (calor/frío) con ensemble conformal, factores científicos, MCP tools y bot Telegram determinista. El sistema prioriza automáticamente fine-tuned → raw 7B → raw 1.5B → bot sin LLM.
---

# ClimaSafeAI — Predicción de riesgo térmico

Sistema completo de predicción de riesgo térmico personalizado. Combina un
ensemble de modelos (XGBoost, LSTM, fórmulas físicas) con 27 factores de
riesgo científicos, personalización individual y conformal prediction.

**Cobertura:** calor y frío extremo en España peninsular.

---

## Arquitectura

```
Usuario → Bot Telegram / MCP / API REST
            ↓
        [Predictor] → Ensemble (XGBoost + LSTM + Fórmula)
            ↓
        Personalización (edad, sexo, IMC, comorbilidades, medicación,
                         fototipo, situación social, aclimatación, actividad)
            ↓
        Safety Overrides (HI ≥ 39°C → EXTREMO, WC ≤ -25°C → EXTREMO)
            ↓
        Conformal Prediction (α = 0.1)
            ↓
        Respuesta + recomendaciones
```

Tres modos de conversación (auto‑detección):
1. **Qwen 2.5 fine‑tuneado** → mejor experiencia (requiere GPU)
2. **Qwen 2.5 raw + RAG** → responde preguntas de fondo citando fuentes (CPU)
3. **Bot determinista** → fallback seguro sin LLM

---

## Requisitos

| Dependencia | Versión | ¿Cuándo? |
|-------------|---------|----------|
| Python | **>= 3.12** | siempre (ver `pyproject.toml`) |
| `uv` | cualquiera reciente | siempre |
| `make` | cualquiera | opcional (targets de conveniencia) |
| Ollama | v0.32.5+ | solo LLM local (Qwen 2.5 + RAG) |
| Claves API (`.env`) | — | ver tabla abajo |

El resto de dependencias Python se instalan con `uv sync` desde
`pyproject.toml` (`numpy`, `pandas`, `scikit-learn`, `mcp>=1.28.1`,
`sqlite-vec`, `litellm`, ...). No hace falta Docker.

---

## Setup reproducible

```bash
git clone https://github.com/ANFAIA/ClimaSafe.git
cd ClimaSafe

# 1. Entorno
uv sync

# 2. Claves API — copia y rellena
cp .env.example .env
```

| Variable | ¿Obligatoria? | Uso |
|----------|---------------|-----|
| `AEMET_API_KEY` | no | Datos AEMET OpenData |
| `OpenUV_API_KEY` | no | Índice UV |
| `TELEGRAM_BOT_TOKEN` | solo bot Telegram | Token de @BotFather |
| `GEMINI_API_KEY` | solo RAG Gemini | `ask_rag_mcp` / Paper Scout |
| `GROQ_API_KEY` | no (la sustituye el LLM local) | LLM externo (legado) |
| `CLIMASAFE_MCP_TOKEN` | solo MCP con identidad | Token del llamante MCP |
| `CLIMASAFE_MCP_WRITE_TOKEN` | solo MCP con escritura | Token de escritura MCP |

```bash
# 3. Base de datos de factores + RAG
uv run python -c "from climasafeai.db.manager import DBManager; DBManager().initialize()"

# 4. MCP server (tools para agentes) — stdio, el modo recomendado
uv run python -m agents.tools.prediction_mcp_tool --stdio

# 5. Bot de Telegram (opcional)
uv run python -m climasafeai.bot.telegram_bot
```

Deploy completo de cero, perfil por perfil (A–D), en
[`DEPLOY_IA.md`](../../DEPLOY_IA.md): clonar → `uv sync` → BD → RAG → pytest →
Ollama/Qwen → MCP → bot.

### LLM local (Qwen 2.5) — opcional

```bash
# Instalar Ollama
curl -L "https://github.com/ollama/ollama/releases/download/v0.32.5/ollama-linux-amd64.tar.zst" \
  -o ollama.tar.zst
mkdir -p ~/.local/bin && tar --zstd -xf ollama.tar.zst -C ~/.local/bin/
ollama serve &

# Descargar modelo
ollama pull qwen2.5:1.5b       # ~1 GB, CPU
ollama pull qwen2.5:7b         # ~4.7 GB, GPU (opcional)
```

### Fine‑tuning (opcional, para mejor calidad)

Ver `documentacion/llm/guia-fine-tuning-qwen.md`. Requiere GPU 8 GB+.

---

## MCP Tools

### Servidor principal: predicción de riesgo

`agents.tools.prediction_mcp_tool` — FastMCP **"ClimaSafeAI Predicción de
Riesgo"**. Es el servidor que configuran `.mcp.json` y `opencode.json` del repo,
y el que expone las tools de uso diario. Se arranca con:

```bash
uv run python -m agents.tools.prediction_mcp_tool --stdio   # stdio (agentes)
uv run python -m agents.tools.prediction_mcp_tool           # HTTPS autofirmado :8101/mcp
uv run python -m agents.tools.prediction_mcp_tool --insecure  # HTTP plano :8101/mcp
# o, si instalaste el entrypoint: climasafeai-mcp
```

12 tools, más un recurso UI:

| Tool | Descripción | Acceso |
|------|-------------|--------|
| `predict_risk_mcp` | Predice riesgo cardiovascular para 1 salida (lat, lon, hora, duración, actividad, ...) | público |
| `grafica_riesgo_horario_mcp` | Curva de riesgo hora a hora como **imagen PNG** | público |
| `crear_perfil_mcp` | Crea perfil con datos demográficos, clínicos y sociales | identidad + token de escritura |
| `cargar_perfil_mcp` | Carga **tu** perfil; sin `uid` usa el del token | perfil propio |
| `cargar_perfil_por_chat_id_mcp` | Carga tu perfil por tu chat de Telegram | perfil propio |
| `listar_usuarios_mcp` | Lista los perfiles — **solo rol admin**, un usuario normal recibe error | admin |
| `vincular_chat_id_mcp` | Vincula un chat de Telegram a **tu** perfil | perfil propio + token de escritura |
| `listar_rutinas_mcp` | Lista las rutinas semanales de un perfil | perfil propio |
| `crear_rutina_mcp` | Crea una rutina semanal (días 1-7, hora inicio/fin, deporte/ocupación) | perfil propio + token de escritura |
| `borrar_rutina_mcp` | Borra una rutina propia por su id | perfil propio + token de escritura |
| `configurar_hora_aviso_mcp` | Configura/consulta la hora de aviso diario | perfil propio + token de escritura |
| `riesgo_rutinas_dia_mcp` | Riesgo de cada rutina de un perfil para un día (clase, prob, temp media) | perfil propio |

Recurso UI: `ui://prediccion-riesgo` — vista HTML del último resultado de
`predict_risk_mcp` (MCP Apps / web).

### Servidor secundario: factores, RAG y LLM (opcional)

`agents.tools.factors_mcp_tool` — FastMCP **"ClimaSafeAI Factores de Riesgo"**.
Gestión del catálogo científico y RAG. No es el servidor por defecto; se arranca
solo si necesitas sus tools (`make mcp-factors` o
`uv run python -m agents.tools.factors_mcp_tool`):

| Tool | Descripción |
|------|-------------|
| `get_factors_mcp` | Lista factores, filtro por tipo (calor/frío) |
| `suggest_factor_mcp` | Propone nuevo factor científico |
| `approve_factor_mcp` | Activa factor candidato |
| `reject_factor_mcp` | Rechaza factor candidato |
| `update_factor_mcp` | Edita campos de un factor |
| `pending_factors_mcp` | Factores pendientes de revisión |
| `check_acclimatization_mcp` | Detecta perfiles listos para aclimatar |
| `auto_acclimatize_mcp` | Aclimata perfiles automáticamente |
| `search_factors_mcp` | Búsqueda semántica sobre factores |
| `search_documentos_mcp` | Búsqueda sobre documentación del proyecto |
| `search_all_mcp` | Búsqueda combinada (factores + docs) |
| `ask_rag_mcp` | RAG con Gemini (requiere API key) |
| `ask_qwen_rag_mcp` | RAG con Qwen 2.5 local + citas de fuentes |
| `qwen_raw_mcp` | Qwen raw sin RAG |

---

## Identidad y permisos del MCP (MCP-003)

Las tools de perfil exigen **identidad del llamante** (MCP-003). El servidor se
arranca con `CLIMASAFE_MCP_TOKEN=<token>` (o `--identidad <token>`) en stdio, o
el cliente manda `Authorization: Bearer <token>` en HTTP. El token se emite con
`make mcp-token ALIAS=<alias>` (o `--emitir-token <alias>`) y se enseña una sola
vez. Sin él, ninguna tool de perfil devuelve datos.

Desde MCP-002 el servidor arranca en **solo lectura**: las tools que escriben
(`crear_perfil_mcp`, `crear_rutina_mcp`, `borrar_rutina_mcp`,
`vincular_chat_id_mcp`, `configurar_hora_aviso_mcp`) devuelven error hasta que
el proceso se arranca con `CLIMASAFE_MCP_WRITE_TOKEN=<secreto>` (o
`--token-escritura <secreto>`). El host que no lo lleve puede leer y predecir
igual que siempre; el que lo lleve, además puede escribir.

Cada llamante ve **solo su propio perfil**. El identificador público es el `uid`
opaco (`usr_…`): alias, `id` y `chat_id` ya no sirven como llave de acceso, y
pedir el `uid` de otra persona devuelve error, no una versión recortada.

---

## Uso desde cualquier agente IA

### Configurar el MCP server

En `opencode.json` o `claude_desktop_config.json` (misma config que el repo):

```json
{
  "mcpServers": {
    "climasafeai": {
      "command": "uv",
      "args": ["run", "python", "-m", "agents.tools.prediction_mcp_tool", "--stdio"],
      "env": {
        "CLIMASAFE_MCP_TOKEN": "<tu-token>",
        "CLIMASAFE_MCP_WRITE_TOKEN": "<secreto-escritura>"
      }
    }
  }
}
```

### Ejemplo de consulta

```
Usuario: ¿Qué riesgo tengo para una hora de padel a las 14h en Sevilla?
Agente: (usa predict_risk_mcp con lat=37.38, lon=-5.99, duracion=1h,
        hora_inicio=14, nivel_actividad=intensa)
        → Riesgo MUY ALTO. HI=41°C, factor cardiovascular ×2.3.
          Recomendaciones: evitar horas centrales, hidratación cada 15 min.
```

---

## Publicación en skills.sh

Este skill vive en `skills/climasafeai/SKILL.md` (ubicación estándar de
skills.sh). Para instalarlo desde cualquier máquina:

```bash
npx skills add ANFAIA/ClimaSafe
```

---

## Modelo de datos

### Factores de riesgo (27 implementados)

Cada factor tiene: tipo (calor/frío), categoría, clave, nombre, coeficiente,
DOI de la fuente, calidad (alta/media/baja), población de referencia.

Ejemplos: `edad>65`, `obesidad`, `diabetes`, `antipsicoticos`,
`vive_solo`, `fototipo_II`, `falta_sueno`, `humedad`, `contaminacion`.

### Perfil de usuario

Campos: alias, edad, sexo, grasa%, aclimatado, comorbilidades, medicación,
nivel_actividad, fototipo, situación social, chat_id Telegram.

---

## Thresholds de seguridad

| Condición | Acción |
|-----------|--------|
| HI ≥ 39°C | Safety override → EXTREMO |
| HI ≥ 41°C | Safety override → EXTREMO + recomendación letal |
| WC ≤ -25°C | Safety override → EXTREMO |
| IMC > 40 | Cap ×2.5 en calor |
| Factor_total > 5 | Cap en 5 |
| Índice personalizado > 1 | Satura en 1 |

---

## Tests

```bash
uv run pytest tests/ -q
```

---

## Referencias

- `DEPLOY_IA.md` — deploy manifest sin Docker (perfiles A–D)
- Documentación completa en `documentacion/`
- Factores científicos con DOI en SQLite (`data/climasafe.db`)
- Guía de fine‑tuning: `documentacion/llm/guia-fine-tuning-qwen.md`
- Roadmap: `documentacion/proximos_pasos.md`
