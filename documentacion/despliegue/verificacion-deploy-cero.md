# Verificación del despliegue de cero — DEPLOY-001

> **Fecha:** 2026-08-18
> **Qué se verificó:** el flujo de `DEPLOY_IA.md` ejecutado **de cero**, en un
> directorio limpio, pegando los logs reales (no el manifiesto).
> **Dónde:** `/home/cacelas/Documentos/anfaia/deploy-cero/ClimaSafeAI`
> (clon local del repo en HEAD `23f3304`).
> **Por qué no en `/tmp`:** `/tmp` tiene 2,7 GB y el venv real del proyecto
> ocupa ~7 GB; el despliegue necesitaba un venv completo de verdad.

## Resultado

**El despliegue de cero llegó al 100 %**: base de datos + RAG, pytest, Ollama
con LLM local, RAG+Qwen respondiendo, MCP server y API REST sirviendo, y el
health check del manifiesto pasando. No hizo falta ningún servicio externo
que no existiera (Ollama ya tenía los modelos descargados en la máquina).

Se detectaron **4 huecos reales en `DEPLOY_IA.md`** (pasos que, ejecutados tal
cual, fallan o apuntan a código que ya no existe). Están listados abajo con
su corrección. No se tocó `DEPLOY_IA.md` (decisión: la corrección del
manifiesto es otra feature; aquí solo se documenta lo que se ejecutó).

---

## Log real por pasos

### Paso 1 — Requisitos y hardware

```
$ python3 --version
Python 3.13.5
$ uv --version
uv 0.9.12
$ git --version
git version 2.47.3

$ python3 -c "..."
{"gpu": "", "ram_mb": 32029, "python": "3.13.5 (main, Jul 15 2026, 20:25:40) [GCC 14.2.0]"}
```

Sin GPU NVIDIA, 32 GB RAM → perfil **B** (Qwen 1.5b) según la regla del
manifiesto. Al final se usó mejor que B: la máquina ya tenía el modelo
fine-tuned `qwen3:climasafe`, que el sistema detecta y prioriza.

### Paso 2 — Clonar

```
$ git clone /home/cacelas/Documentos/anfaia/ClimaSafeAI ClimaSafeAI
$ cd ClimaSafeAI && git log --oneline -1
23f3304 feat: cierra MCP-004
```

Clon limpio: sin `.venv`, sin `data/` ni `models/` (gitignored).

### Paso 3 — Entorno Python

El manifiesto dice `uv sync` a secas. **Ejecutado tal cual**:

```
$ uv sync
   (resuelve e instala las dependencias base; OK)
$ uv run python -c "import pytest"
   (OK — pytest llega como dependencia transitiva)
$ uv run python -c "import fastapi"
ModuleNotFoundError: No module named 'fastapi'
```

**Hueco 1 (extra `api`):** `uv sync` del manifiesto no instala fastapi/uvicorn,
necesarios para el paso 9.3 (API). **Hueco 2 (extra `rag`):** igual para
`sentence-transformers`, necesario para el paso 4 (RAG).

Corrección usada (la misma que `make setup`):

```
$ uv sync --extra dev --extra supervisado --extra no_supervisado \
    --extra redes_neuronales --extra mlflow_tracking --extra optuna \
    --extra api --extra monitoring --extra rag
   (OK — fastapi y sentence_transformers importables)
```

### Paso 4 — Base de datos + RAG

El manifiesto dice `db.init_rag()` a secas. **Ejecutado tal cual**:

```
$ uv run python -c "
from climasafeai.db.manager import DBManager
db = DBManager()
stats = db.init_rag()
print('DB lista:', stats)"
sqlite3.OperationalError: no such table: factores_riesgo
```

**Hueco 3 (secuencia de inicialización):** `init_rag()` asume que las tablas
ya existen y que los factores ya están migrados. La secuencia real (la que
usan los tests `test_rag_factores.py`/`test_rag_reindex.py`) es:

```
$ rm -f data/climasafe.db   # BD vacía creada por el intento fallido
$ uv run python -c "
from climasafeai.db.manager import DBManager
db = DBManager()
db.initialize()              # crea tablas desde data/schema.sql
db.migrar_desde_json()       # vierte los 28 factores de riesgo desde JSON
stats = db.init_rag()        # indexa vectores (sqlite-vec) + documentación
print('DB lista:', stats)"
DB lista: {'success': True, 'stats': {'factores': {'embedded': 28, 'total': 28, 'pending': 0}, 'documentos': {'fragmentos': 573, 'palabras': 94995}}}
```

Nota: el manifiesto dice "27 factores"; el JSON actual tiene 28.

### Paso 5 — Verificar base

```
$ uv run pytest tests/test_rag_qwen.py tests/test_proba.py -q --no-header 2>&1 | tail -3
-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
======================= 62 passed, 3 warnings in 15.02s ========================
```

El manifiesto promete "29 passed" — la suite ha crecido, hoy son 62.

### Paso 6 — Ollama (LLM local)

```
$ ollama serve &
$ curl -s http://localhost:11434/api/tags
{"models":[{"name":"qwen3:climasafe", ...}, {"name":"qwen2.5:climasafe", ...},
 {"name":"gemma3:4b", ...}, {"name":"gemma3:1b", ...}, {"name":"qwen3:1.7b", ...},
 {"name":"qwen2.5:1.5b", ...}]}
```

Ollama arrancó sin descargar nada: los modelos ya estaban en la máquina
(incluidos los fine-tuned `qwen3:climasafe` y `qwen2.5:climasafe`).

### Paso 6.3 — Verificar que el LLM responde

```
$ uv run python -c "
from climasafeai.llm.rag_qwen import check_ollama
st = check_ollama()
print('Ollama:', 'OK' if st['available'] else 'NO DISPONIBLE')
print('Modelos:', st['models'])
print('Mejor modelo:', st['best_model'])"
Ollama: OK
Modelos: ['qwen3:climasafe', 'qwen2.5:climasafe', 'gemma3:4b', 'gemma3:1b', 'qwen3:1.7b', 'qwen2.5:1.5b']
Mejor modelo: ollama/qwen3:climasafe
```

El sistema detecta el fine-tuned `qwen3:climasafe` como mejor modelo → la
interacción principal usa perfil D (fine-tuned + RAG), no B.

### Paso 7 — Probar RAG + Qwen

**Hueco 4 (API renombrada):** el manifiesto importa `QwenConfig`, que ya no
existe; hoy es `LLMConfig` (`climasafeai/llm/rag_qwen.py`). Ejecutado con el
nombre actual:

```
$ uv run python -c "
from climasafeai.llm.rag_qwen import ask_with_rag, LLMConfig
cfg = LLMConfig()
res = ask_with_rag('¿Qué riesgo tengo con 72 años, diabetes, y 35°C?',
                   k_factores=4, k_docs=3, config=cfg)
print('Respuesta:', res.get('answer','')[:400])
print('Fuentes:', len(res.get('sources_factores',[])), 'factores,',
      len(res.get('sources_docs',[])), 'documentos')"
Basándome en el contexto proporcionado, tienes un alto riesgo térmico debido a la combinación de tus factores de riesgo. [...]
Fuentes: 1 factores, 2 documentos
```

### Paso 9.1 — MCP server

```
$ uv run python -m agents.tools.factors_mcp_tool --port 8100 &
INFO:     Uvicorn running on http://0.0.0.0:8100 (Press CTRL+C to quit)
```

El check del manifiesto (`curl http://localhost:8100/sse`) da **404**: el
servidor usa el transporte streamable HTTP moderno en `/mcp`, no SSE. Con el
endpoint real responde el handshake:

```
$ curl -X POST http://localhost:8100/mcp -H "Content-Type: application/json" \
    -H "Accept: application/json, text/event-stream" \
    -d '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2025-03-26","capabilities":{},"clientInfo":{"name":"deploy-check","version":"0.1"}}}'
event: message
data: {"jsonrpc":"2.0","id":1,"result":{"protocolVersion":"2025-03-26","capabilities":{"...":"...","tools":{"listChanged":false}},"serverInfo":{"name":"ClimaSafeAI Factores de Riesgo","version":"1.28.1"}}}
```

### Paso 9.3 — API REST

**Hueco 5 (ruta del módulo):** el manifiesto lanza `climasafeai.api.main:app`,
módulo que **no existe** en el repo (no hay `climasafeai/api/`). La API real
es `chat.app:app` (la sirve `chat/entrypoint.sh` y el target `make web`):

```
$ uv run uvicorn chat.app:app --host 0.0.0.0 --port 8000 &
INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)

$ curl -s http://localhost:8000/api/status | python3 -m json.tool
{
    "project": "ClimaSafeAI",
    "ml_type": "supervisado",
    "task_type": "clasificacion",
    "version": "0.0.75",
    ...
}
```

`version: 0.0.75` es la del HEAD clonado — la versión sale de la fuente única
(pyproject.toml), no hay hardcodes (DEPLOY-001, criterio 2).

### Paso 11 — Health check final

```
$ echo "=== ClimaSafeAI — Diagnóstico ==="
=== ClimaSafeAI — Diagnóstico ===
$ uv run python -c "..."    # script completo del manifiesto
{
  "base_de_datos": {
    "factores": 28,
    "fragmentos_docs": 573
  },
  "ollama": {
    "disponible": true,
    "modelos": ["qwen3:climasafe", "qwen2.5:climasafe", "gemma3:4b", "gemma3:1b", "qwen3:1.7b", "qwen2.5:1.5b"],
    "mejor": "ollama/qwen3:climasafe"
  },
  "rag_qwen": {
    "funciona": true,
    "respuesta_ejemplo": "ClimaSafeAI es un sistema diseñado para mejorar la seguridad y salud del usuario alrededor de su entorno climático, especialmente en términos de riesgo térmico."
  }
}
```

---

## Huecos encontrados en DEPLOY_IA.md (sin corregir — solo documentados)

| # | Paso | Qué falla tal cual | Corrección usada en esta verificación |
|---|------|--------------------|----------------------------------------|
| 1 | 3 | `uv sync` no instala el extra `api` (fastapi) → el paso 9.3 no puede arrancar | `uv sync --extra api ...` (los extras de `make setup`) |
| 2 | 3/4 | `uv sync` no instala el extra `rag` (sentence-transformers) → `init_rag()` muere con `ModuleNotFoundError` | ídem |
| 3 | 4 | `db.init_rag()` solo no basta en BD limpia: falla `no such table: factores_riesgo` | `db.initialize()` → `db.migrar_desde_json()` → `db.init_rag()` |
| 4 | 7 | `QwenConfig` ya no existe (renombrado a `LLMConfig`) | importar `LLMConfig` |
| 5 | 9.1 | `curl /sse` da 404: el MCP usa streamable HTTP en `/mcp` | handshake `POST /mcp` con `initialize` |
| 6 | 9.3 | `climasafeai.api.main` no existe; la API real es `chat.app:app` | `uv run uvicorn chat.app:app --port 8000` |
| — | 5 | "Deben salir 29 passed" está desactualizado (hoy 62) | — |
| — | 4 | "27 factores" está desactualizado (el JSON tiene 28) | — |

## Qué NO se ejecutó (y por qué)

| Paso | Motivo |
|------|--------|
| 9.2 Bot Telegram | Requiere token de @BotFather; no es el modo principal (dice el propio manifiesto) |
| 10 Fine-tuning | Requiere GPU CUDA + Unsloth (perfil D) — la máquina no tiene GPU. Además los modelos fine-tuned ya están en Ollama |
| `git push` / GitHub Release | No procede en un despliegue local; ver nota de release abajo |

## Nota sobre `make release` y el push (criterios 1 y 4)

La verificación del release se hizo en un clon aislado
(`/home/cacelas/Documentos/anfaia/deploy-cero/` no; en
`/tmp/opencode/release-check/ClimaSafeAI`, luego borrado) para no tocar el
repo real: `make release` crea el commit de release, el tag local y **no hace
push**. Evidencia: tras ejecutarlo, `git ls-remote origin` seguía mostrando
`refs/heads/main` en el commit anterior y solo el tag `v0.0.71`; el commit
`chore(release): 0.0.77` y el tag `0.0.77` solo existían localmente.

A nivel de código, `agents/tools/git_tool.py` no tiene ningún método `push`
(solo `add`, `commit`, `create_tag`, `tag_exists`), así que el agente git no
puede pushear por su cuenta. Los únicos flujos que pushean en el repo son
CI: `scripts/pages_deploy.sh` (dry-run por defecto, `PUSH=no`),
`scripts/release_ci.sh` (solo CI) y `paper_scout.yml` (bot de CI).
