# Deploy manifest — ClimaSafeAI

> ⚡ Instrucciones para que un **agente IA** monte ClimaSafeAI desde cero.
> **Sin Docker.** Sin sudo. Lee todo antes de ejecutar: hay puntos de decisión
> donde debes adaptarte al hardware del usuario y a lo que pide.

---

## 0. Antes de empezar — decide qué montar

Este manifiesto tiene tres perfiles de deploy. Elige según el hardware y el
uso que el usuario quiere:

| Perfil | Requisitos | Lo que obtienes |
|--------|------------|-----------------|
| **A — Solo RAG** | Cualquier CPU | Bot determinista + MCP tools + RAG sobre documentación. Sin LLM. |
| **B — LLM local (rápido)** | CPU, 2 GB RAM libres | Perfil A + Qwen 2.5 1.5B (~1 GB) + RAG+LLM para preguntas |
| **C — LLM local (máxima calidad)** | GPU 4 GB+, 8 GB RAM | Perfil B + Qwen 2.5 7B (~4.7 GB) — respuestas más precisas |
| **D — Fine‑tuned** | GPU 8 GB VRAM+, 16 GB RAM | Perfil C + LoRA fine‑tuneado para voz ClimaSafe perfecta |

**Regla:** Si el usuario no dice nada, empieza con B (1.5B es funcional en
cualquier CPU). Si tiene GPU, sube a C. El fine‑tuning (D) solo si lo pide
explícitamente o si ves que tiene GPU y tiempo.

---

## 1. Requisitos previos

```bash
# Indispensables
python3 --version          # debe ser ≥ 3.13
uv --version               # debe estar instalado
git --version

# Detecta hardware (útil para decidir perfil)
python3 -c "
import subprocess
import json

# GPU NVIDIA?
try:
    nvidia = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total',
        '--format=csv,noheader'], capture_output=True, text=True, timeout=5)
    gpu = nvidia.stdout.strip() if nvidia.returncode == 0 else ''
except: gpu = ''

# RAM total
mem = None
try:
    with open('/proc/meminfo') as f:
        for line in f:
            if line.startswith('MemTotal:'):
                mem = int(line.split()[1]) // 1024  # MB
                break
except: pass

print(json.dumps({
    'gpu': gpu,
    'ram_mb': mem,
    'python': __import__('sys').version,
}))
"

# Tu decides el perfil según esto:
# - ¿gpu contiene RTX/GTX/Tesla/A10/etc? → perfil C o D
# - ¿ram_mb > 6000? → perfil B o C
# - ¿solo CPU y poca RAM? → perfil A
```

---

## 2. Clonar

```bash
git clone https://github.com/ANFAIA/ClimaSafe.git
cd ClimaSafe
```

---

## 3. Entorno Python

```bash
uv sync
uv run python -c "import climasafeai; print('OK')"
```

---

## 4. Base de datos + RAG

```bash
# Esto crea la BD, migra los 27 factores de riesgo desde JSON a SQLite,
# e indexa los vectores (sqlite-vec) + los 509 fragmentos de documentación.
uv run python -c "
from climasafeai.db.manager import DBManager
db = DBManager()
stats = db.init_rag()
print('DB lista:', stats)
"
```

---

## 5. Verificar base

```bash
uv run pytest tests/test_rag_qwen.py tests/test_proba.py -q --no-header 2>&1
# Deben salir 29 passed
```

El sistema ya funciona en **modo determinista** desde aquí.

---

## 6. LLM local — Qwen 2.5

### 6.1 Instalar Ollama

```bash
# Descargar el binario estático (no necesita instalación)
curl -L "https://github.com/ollama/ollama/releases/download/v0.32.5/ollama-linux-amd64.tar.zst" \
  -o /tmp/ollama.tar.zst
mkdir -p ~/.local/bin
tar --zstd -xf /tmp/ollama.tar.zst -C ~/.local/bin/
export PATH="$HOME/.local/bin:$PATH"

# Arrancar en segundo plano
ollama serve &
# Esperar a que esté listo
for i in $(seq 1 10); do
  if curl -s http://localhost:11434/api/tags >/dev/null 2>&1; then break; fi
  sleep 1
done
```

### 6.2 Elegir modelo según hardware

```bash
# REGLA para el agente:
# - Si el usuario tiene GPU NVIDIA con ≥4 GB VRAM → usar 7b
# - Si no → usar 1.5b
# - Si el usuario tiene un Mac con Apple Silicon → ambos funcionan bien
# - Preguntar antes de descargar 7b si el usuario tiene poco espacio (~4.7 GB)

# Opción A: CPU / RAM limitada (~1 GB, funcional)
ollama pull qwen2.5:1.5b

# Opción B: GPU o Apple Silicon (~4.7 GB, mucha más calidad)
ollama pull qwen2.5:7b
```

### 6.3 Verificar que el LLM responde

```bash
uv run python -c "
from climasafeai.llm.rag_qwen import check_ollama
st = check_ollama()
print('Ollama:', 'OK' if st['available'] else 'NO DISPONIBLE')
print('Modelos:', st['models'])
print('Mejor modelo:', st['best_model'])
print('Sistema usará:',
    'fine-tuned + RAG (si existe)' if 'climasafe' in str(st['models'])
    else 'raw + RAG (Qwen + documentos)'
    if st['available']
    else 'modo determinista (sin LLM)')
"
```

---

## 7. Probar RAG + Qwen (interacción principal)

Este es el **modo principal** del sistema: el usuario hace preguntas en lenguaje
natural y el sistema responde usando RAG (factores + documentación) sintetizado
por Qwen.

```bash
# Pregunta con RAG (busca en factores + documentación, responde citando fuentes)
uv run python -c "
from climasafeai.llm.rag_qwen import ask_with_rag, QwenConfig
cfg = QwenConfig()

res = ask_with_rag('¿Qué riesgo tengo con 72 años, diabetes, y 35°C?',
                   k_factores=4, k_docs=3, config=cfg)
print('Respuesta:', res.get('answer','')[:400])
print('Fuentes:', len(res.get('sources_factores',[])), 'factores,',
      len(res.get('sources_docs',[])), 'documentos')
"
```

Si el usuario quiere respuestas más profundas o citas de documentos:

```bash
# MCP tool para agents (expone la misma capacidad via MCP)
# En opencode.json o claude_desktop_config.json:
# {
#   "mcpServers": {
#     "climasafeai": {
#       "command": "uv",
#       "args": ["run", "python", "-m", "agents.tools.factors_mcp_tool"]
#     }
#   }
# }
```

---

## 8. RAG sobre documentación (reindexar si cambian los .md)

```bash
# Solo si se añadieron/quitaron archivos en documentacion/
uv run python -c "
from climasafeai.db.manager import DBManager
db = DBManager()
db.init_rag()
nuevos = db.sync_documentos()
print(f'Fragmentos nuevos indexados: {nuevos}')
"
```

---

## 9. Servicios complementarios (elige según lo que pida el usuario)

### 9.1 MCP server (tools para que OTROS agentes usen ClimaSafeAI)

```bash
uv run python -m agents.tools.factors_mcp_tool --port 8100 &
curl -s http://localhost:8100/sse | head -3
```

Esto expone 16 tools MCP (factores, perfiles, predicción, RAG) para que
Claude Code, opencode, Cline, Flue y otros agentes las usen.

### 9.2 Bot de Telegram (solo si el usuario lo pide expresamente)

Requiere crear un bot con @BotFather y obtener un token. No es el modo
principal — el LLM local + MCP es más flexible.

```bash
export TELEGRAM_BOT_TOKEN="token_de_BotFather"
uv run python -m climasafeai.bot.telegram_bot &
```

### 9.3 API REST (para integración web)

```bash
uv run uvicorn climasafeai.api.main:app --host 0.0.0.0 --port 8000 &
curl -s http://localhost:8000/status | python -m json.tool
```

---

## 10. Fine‑tuning (solo si el usuario pide mejor calidad)

Activa el perfil D. Requiere GPU 8 GB+, Python 3.10 (entorno separado) y
Unsloth. El resultado es un modelo `qwen2.5:climasafe` que Ollama sirve y el
sistema detecta automáticamente como mejor opción.

Ver guía completa en `documentacion/llm/guia-fine-tuning-qwen.md`.

```bash
# 1. Generar dataset sintético desde el pipeline real
uv run python climasafeai/llm/generar_dataset.py \
  --num-ejemplos 200 --output data/llm/train.jsonl

# 2. Los siguientes pasos requieren un entorno con Unsloth (conda, py3.10)
#    y una GPU con CUDA. Pregunta al usuario antes de continuar.
#
# conda create -n unsloth python=3.10 -y
# conda activate unsloth
# pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
# pip install "unsloth[cu121] @ git+https://github.com/unslothai/unsloth.git"
#
# python climasafeai/llm/fine_tune.py \
#   --model qwen2.5-7b \
#   --train-file data/llm/train.jsonl \
#   --output-dir models/llm/qwen-climasafe-lora \
#   --batch-size 2

# 3. Exportar a GGUF + Ollama
# python climasafeai/llm/fine_tune.py --export-gguf \
#   --lora-path models/llm/qwen-climasafe-lora \
#   --gguf-path models/llm/qwen-climasafe.gguf

# 4. El sistema lo detecta solo:
#    check_ollama()['best_model'] → 'qwen2.5:climasafe'
```

---

## 11. Health check final

```bash
echo "=== ClimaSafeAI — Diagnóstico ==="
uv run python -c "
from climasafeai.db.manager import DBManager
from climasafeai.llm.rag_qwen import ask_with_rag, check_ollama, QwenConfig
import json

db = DBManager()
db.init_rag()
stats = db.rag.stats()
ollama = check_ollama()

health = {
    'base_de_datos': {
        'factores': stats['factores']['total'],
        'fragmentos_docs': stats['documentos']['fragmentos'],
    },
    'ollama': {
        'disponible': ollama['available'],
        'modelos': ollama['models'],
        'mejor': ollama['best_model'],
    },
}

# Probar RAG+LLM
if ollama['available']:
    res = ask_with_rag('Resume en una frase qué es ClimaSafeAI',
                       k_factores=2, k_docs=2, config=QwenConfig())
    health['rag_qwen'] = {
        'funciona': res.get('answer') is not None,
        'respuesta_ejemplo': (res.get('answer', '') or '')[:200],
    }
else:
    health['modo'] = 'determinista (sin LLM)'

print(json.dumps(health, indent=2))
"
```

---

## Resumen para el agente

```
PASOS SIEMPRE:
  1-5:  clonar, uv sync, BD, RAG, pytest

SI HAY CPU (perfil B):
  6:    Ollama + qwen2.5:1.5b
  7:    probar RAG+Qwen ← INTERACCIÓN PRINCIPAL

SI HAY GPU (perfil C):
  6:    Ollama + qwen2.5:7b
  7:    probar RAG+Qwen

SI EL USUARIO QUIERE BOT:
  9.1:  MCP server
  9.2:  Telegram Bot (pedir token)

SI EL USUARIO QUIERE WEB:
  9.3:  API REST

SI EL USUARIO PIDE MÁS CALIDAD Y TIENE GPU:
  10:   Fine‑tuning (guía en documentacion/llm/)
```

---

## Puertos y variables

| Puerto | Servicio | Para qué |
|--------|----------|----------|
| 11434 | Ollama | LLM local (Qwen) |
| 8100 | MCP server | Tools para agentes IA |
| 8000 | API REST | Web + predicción |

| Variable | ¿Obligatoria? | Uso |
|----------|---------------|-----|
| `TELEGRAM_BOT_TOKEN` | Solo si usas bot | Token de @BotFather |
| `CLIMASAFE_DB_PATH` | No (default: `data/climasafe.db`) | Ruta a SQLite |
| `OLLAMA_HOST` | No (default: `http://localhost:11434`) | Host de Ollama |
