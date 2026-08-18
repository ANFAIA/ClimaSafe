# Hosting gratuito del LLM — comparativa con cuota real

> **HOST-001** · Fecha del estudio: 2026-08-18 · Fuentes: páginas oficiales
> consultadas ese día (URL en cada sección) + llamadas reales con las keys del
> `.env` (ver §6).
>
> Regla de oro del ticket: «gratuito» no se decide por la palabra *free* de la
> página, sino por **cuántas peticiones y tokens permite por minuto y por día**
> y **qué cuesta superarlos**.

## 0. Cuánto consume el bot por conversación (medido, 18-08-2026)

Medición real con `scripts/demo_llm_remoto.py` (Groq, free tier, modelo
`groq/openai/gpt-oss-20b`, vía las mismas funciones que usa el bot):

| Paso | Tokens (prompt + completion) |
|---|---|
| Parte redactado (`ask_con_perfil`) | 858 + 991 = **1.849** |
| Pregunta libre del chat (`ask_with_rag`, con RAG y contexto) | 890 + 2048 = **2.938** |
| Conversación completa (parte + 1 pregunta) | ≈ **4.800** |

Esto encaja con la lección previa del proyecto (spacebot): ~7.900 tokens por
mensaje con el prompt antiguo → 1 mensaje/min en Groq. Con el prompt actual
una conversación completa son ~4.800 tokens.

## 1. Groq — free plan

Fuente: <https://console.groq.com/docs/rate-limits> (18-08-2026). Los límites
son por organización; se ven los exactos en
<https://console.groq.com/settings/limits>. Cabeceras `x-ratelimit-*` en cada
respuesta.

| Modelo (free plan) | RPM | RPD | TPM | TPD |
|---|---|---|---|---|
| `openai/gpt-oss-20b` · `openai/gpt-oss-120b` | 30 | 1.000 | 8.000 | 200.000 |
| `qwen/qwen3.6-27b` | 30 | 1.000 | 8.000 | 200.000 |
| `meta-llama/llama-prompt-guard-2-*` (clasificadores) | 30 | 14.400 | 15.000 | 500.000 |

**Con los números del bot (8K TPM):** ~1 conversación completa por minuto
(4.800 tokens) y ~40 conversaciones/día (200K TPD). Con la cuota medida para
solo el parte (1.849 tokens): ~4 partes/minuto, ~108 partes/día.

**Coste de superarla:** HTTP 429 con `retry-after`; no hay cargo extra en el
plan free — simplemente se deja de responder hasta que se repone la ventana.
Para más cuota hay que pasar al plan Developer (de pago).

**Hallazgo real (18-08-2026):** el modelo que usaba el bot
(`groq/llama-3.3-70b-versatile`) **ya no existe** en el free tier. La llamada
real devuelve 404 `model_not_found`. Se sustituyó por
`groq/openai/gpt-oss-20b` (HOST-001, verificado respondiendo).

## 2. Google Gemini — free tier

Fuentes: <https://ai.google.dev/gemini-api/docs/rate-limits> y
<https://ai.google.dev/gemini-api/docs/pricing> (18-08-2026).

- El free tier ofrece **tokens de entrada y salida gratis** en los modelos de
  la familia (la página de pricing marca «Free of charge» en 2.5 Pro, 2.5
  Flash, 2.5 Flash-Lite y la familia 3.x).
- Los límites se miden en **RPM, TPM (entrada) y RPD**, se aplican **por
  proyecto** (no por key) y se resetean a medianoche (hora del Pacífico).
- Los **números exactos por modelo** solo se ven en la consola de AI Studio
  (por proyecto), no en una página estática; los valores publicados
  históricamente para 2.5-flash free eran ~15 RPM / 250K TPM / 1.000 RPD,
  pero **hay que confirmarlos en la consola** — no los doy por verificados.
- El tier Free no tiene límite de gasto (`spend rate limit: N/A`).

**Hallazgo real (18-08-2026, con la GEMINI_API_KEY del `.env`):**
- `gemini/gemini-2.5-flash` → 404 «no longer available to new users»
  (confirma la lección previa del proyecto: los 2.x ya no están en esa cuenta).
- `gemini/gemini-3.6-flash` → **responde** (la doc pide migrar a 3.x).
- La advertencia histórica de la reunión (thought_signature de Gemini 3.x con
  tools de spacebot) afecta a llamadas *con herramientas*; el bot de Telegram
  solo hace completion de texto, así que no aplica. Aun así, no se eligió
  Gemini como primario porque Groq da cuota explícita publicada y ya hay key.

**Coste de superarla:** 429; subir de tier (vincular cuenta de facturación)
para más cuota.

## 3. OpenRouter — modelos `:free`

Fuente: <https://openrouter.ai/docs/limits> (18-08-2026).

| Condición (créditos comprados en total) | RPM | RPD |
|---|---|---|
| < 10 $ | 20 | 50 |
| ≥ 10 $ | 20 | 1.000 |

- OpenRouter **no publica límites de tokens** para modelos free a nivel de
  plataforma: lo que ata son los 20 RPM / 50 RPD (sin créditos). Los límites
  de tokens son los del proveedor aguas arriba.
- **Confirmado lo de «pide saldo»:** si la cuenta tiene saldo negativo, los
  modelos free devuelven **402 Payment Required** — hace falta mantener el
  saldo ≥ 0 aunque no se gaste nada por petición.

**Con los números del bot:** 50 RPD ≈ **50 conversaciones/día** como tope
práctico (más que Groq), pero solo 20 RPM y con el riesgo de 402 si la cuenta
se queda a cero. No hay key en el `.env`.

## 4. Hosting gratuito para el BOT (Telegram)

El objetivo del ticket es que el LLM no viva en el portátil; el bot también
podría moverse a un host. Comparativa real:

| Plataforma | Plan gratis | Límites | Problema para este bot |
|---|---|---|---|
| **Render** ([docs/free](https://render.com/docs/free), 18-08) | Web service free: 512 MB RAM / 0.1 CPU | 750 h de instancia/mes; **spin-down a los 15 min sin tráfico entrante**; filesystem efímero (se pierde `data/climasafe.db`); Postgres free 1 GB caduca a los 30 días | Un bot con long-polling no recibe tráfico HTTP entrante → se duerme y **no** se despierta solo. Sirve con webhook + keepalive, no con polling. |
| **Railway** ([pricing](https://railway.com/pricing), 18-08) | 0 $/mes + **1 $ de crédito de uso** | 1 vCPU / 0.5 GB por servicio, 1 réplica; CPU 0,00000772 $/vCPU/s + RAM 0,00000386 $/GB/s | El bot 24/7 ≈ 0,50 $/día → **el crédito de 1 $ dura ~2 días/mes**; después se suspende o se paga (~5 $/mes en Hobby). Filesystem efímero igualmente. |
| **Fly.io** ([pricing](https://fly.io/docs/about/pricing/), 18-08) | Sin free tier de uso | Tarjeta obligatoria; máquina mínima ~2 $/mes | No es gratis. |
| **HF Spaces** ([spaces-overview](https://huggingface.co/docs/hub/spaces-overview), 18-08) | Static gratis; **crear Spaces de compute (Docker/Gradio) requiere plan de pago**; free solo ZeroGPU (2 Gradio) | CPU Basic sin coste/hora, pero no se puede crear con cuenta free | Un bot Docker 24/7 no entra en el free de Spaces. |

**Conclusión de hosting:** ninguno aloja este bot 24/7 *gratis de verdad* con
la SQLite de perfiles + RAG. La opción menos mala es **Railway** (el crédito
de 1 $/mes da ~2 días de 24/7; 5 $/mes lo deja estable) o **Render con
webhook**. El despliegue documentado (ver `despliegue_llm_remoto.md`) se
centra en lo que el ticket pide: **el LLM remoto funciona sin el portátil**;
el bot puede seguir arrancando desde el portátil (que ya no necesita Ollama ni
GPU) o desde un host cuando se decida pagar.

## 5. Decisión (HOST-001)

**Proveedor elegido: Groq free tier, modelo `groq/openai/gpt-oss-20b`.**

- Cuota publicada y verificada por llamada real: 8K TPM / 200K TPD / 30 RPM /
  1.000 RPD.
- Sin riesgo de cargo: el 429 no cuesta dinero.
- Ya hay key en el `.env`.
- Fallback: si Groq se cae o agota cuota, el bot **ya** degrada a plantilla
  determinista (verificado, §6.3). Con `GEMINI_API_KEY` sola, el bot usa
  `gemini/gemini-3.6-flash` como alternativa automática.

## 6. Verificación real (mismo día, con las keys del `.env`)

Salida resumida de `.venv/bin/python scripts/demo_llm_remoto.py` (Ollama
parado; no se imprimen claves):

1. **Parte por Groq** (`ask_con_perfil`, 1.849 tokens): responde con el parte
   completo, frases obligatorias copiadas y cierre contextualizado.
2. **Pregunta libre** (`ask_with_rag` con RAG, 2.938 tokens): responde sobre
   SPF citando 2 documentos; el corte por `max_tokens` añade el aviso de
   BOT-022 (comportamiento esperado).
3. **Fallo provocado** (`groq/llama-3.3-70b-versatile`): 404 real →
   `ask_con_perfil` devuelve `None` → el parte cae a la plantilla
   determinista (exactamente la degradación que vería el usuario con el
   servicio caído o sin cuota).

Evidencia completa de la ejecución en el registro del arnés (HOST-001) y en
`progress/current.md`.
