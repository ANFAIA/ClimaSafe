# Despliegue del bot con LLM remoto (sin depender de Ollama)

> **HOST-001** · 2026-08-18 · Reproduce cómo arrancar el bot de Telegram con
> el LLM remoto gratuito (Groq) y sin Ollama. Ejecutado de cero una vez el
> mismo día (ver §4).

## 1. Qué hace falta

- El repo con `climasafeai/` (el bot usa LiteLLM, que no necesita GPU ni
  modelos locales).
- `.env` con `GROQ_API_KEY` (y opcional `GEMINI_API_KEY` como alternativa) y
  `TELEGRAM_BOT_TOKEN`. **No** se necesita Ollama ni ningún modelo descargado.
- Python con las dependencias del proyecto (`uv run --no-sync` o el `.venv`).

## 2. Qué cambió el código para que esto funcione (HOST-001)

| Cambio | Fichero | Efecto |
|---|---|---|
| `MODELO_API` → `groq/openai/gpt-oss-20b` | `climasafeai/bot/telegram_bot.py` | El modo «API externa» (/api) usa un modelo que existe en el free tier de Groq (llama-3.3-70b-versatile daba 404). |
| `MODELO_API_GEMINI` → `gemini/gemini-3.6-flash` | ídem | Alternativa automática si solo hay `GEMINI_API_KEY`. |
| `_modelo_por_defecto()` | ídem | Sin Ollama: si hay `GROQ_API_KEY` usa `MODELO_API`; si solo hay `GEMINI_API_KEY` usa `MODELO_API_GEMINI`; sin claves → determinista (como antes). |
| Chat libre (`_preguntar_al_rag`) | ídem | Si el LLM no contesta (caído o sin cuota), responde `CHAT_LIBRE_SIN_LLM` (plantilla determinista) en vez del error visible. |
| `_OcultarToken` | ídem | Tapa también las claves `*_API_KEY` del entorno en los logs (un error de litellm de Gemini puede traer `?key=...`). |
| `MODELO_API_DEFECTO` | `climasafeai/llm/rag_qwen.py` | Mismo modelo nuevo, por coherencia. |

La degradación del **parte** a plantilla cuando el LLM no contesta ya existía
(«Respuesta redactada por …» / «… no contestó; se responde con la plantilla»);
no se tocó.

## 3. Reproducir el arranque de cero

```bash
# 1. Entorno sano (opcional pero recomendado)
./init.sh --quick

# 2. Arrancar el bot (con autoreinicio, igual que siempre)
./scripts/run_bot.sh            # foreground, log en logs/bot.log
# o en background:
./scripts/run_bot.sh --daemon

# 3. Verificar en el log que arrancó sin Ollama
tail -f logs/bot.log
```

El bot elige modelo **por conversación** al primer mensaje: con Ollama parado
y `GROQ_API_KEY` presente, `_modelo_por_defecto()` devuelve
`groq/openai/gpt-oss-20b` y el parte lo redacta Groq. Se puede forzar por
chat con `/api` (Groq), `/determinista` o escribiendo cualquier modelo
LiteLLM.

### Demo de una conversación completa sin Ollama

```bash
.venv/bin/python scripts/demo_llm_remoto.py
```

Hace 3 llamadas reales con la cuota gratuita: (1) parte por Groq, (2) pregunta
libre con RAG, (3) fallo provocado con el modelo retirado → plantilla.

## 4. Ejecutado una vez de cero (evidencia del 18-08-2026)

1. `check_ollama()` con Ollama parado → `available=False`.
2. `.venv/bin/python scripts/demo_llm_remoto.py` → parte y pregunta libre
   respondidos por `groq/openai/gpt-oss-20b` (1.849 y 2.938 tokens), y
   degradación a plantilla con el modelo retirado (404 real).
3. Arranque del proceso del bot sin Ollama → el proceso levanta y empieza a
   hacer polling de Telegram (log de arranque en `logs/bot.log`), sin
   depender de `ollama serve`.

## 5. Límites y avisos (ver la comparativa completa en `hosting_llm_gratis.md`)

- **Cuota Groq free:** 8K TPM / 200K TPD / 30 RPM / 1.000 RPD. Una
  conversación completa ≈ 4.800 tokens → ~40 conversaciones/día. Superarla
  devuelve 429 sin coste; el bot cae a plantilla mientras tanto.
- **Latencia:** cada intento al LLM caído espera el timeout de LiteLLM antes
  de degradar a plantilla; es el comportamiento actual, no un cambio nuevo.
- **Hosting del bot:** si se mueve a un host gratis (Railway/Render/…), la
  SQLite (`data/climasafe.db`, perfiles + RAG) vive en un filesystem efímero
  y se pierde al reiniciar; habrá que persistirla (volumen de pago) o
  reconstruirla. Ningún host gratis la persiste hoy (ver comparativa).
- **Claves:** el log filtra `TELEGRAM_BOT_TOKEN` y todas las `*_API_KEY`
  (BOT-004 + HOST-001); no pegar claves en el `.env` versionado (`.env` está
  fuera del repo; `.env.example` solo documenta nombres).
