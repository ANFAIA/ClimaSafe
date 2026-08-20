# LLM

## Qué papel juega el LLM

ClimaSafe es **determinista por diseño**: la predicción, los factores y las
recomendaciones salen de modelos ML e índices, no de un LLM. El LLM se usa
solo para **redactar** respuestas en lenguaje natural (bot de Telegram y chat
web): recibe el resultado real del pipeline y lo explica con tus palabras.

## Modelos soportados: cualquiera, vía LiteLLM

La capa de LLM usa **LiteLLM** (`litellm.completion`), que habla con
prácticamente cualquier proveedor con una sola API. El modelo se configura con
el formato LiteLLM:

| Proveedor | Ejemplo |
|---|---|
| Ollama local | `ollama/qwen3:climasafe` · `ollama/qwen2.5:1.5b` |
| Groq | `groq/openai/gpt-oss-20b` |
| OpenAI | `gpt-4o` |
| Gemini | `gemini/gemini-3.6-flash` |

`LLMConfig` acepta cualquiera de esos formatos, así que **el sistema soporta
cualquier modelo** de cualquier proveedor — solo hay que indicarlo.

### Selección automática

En Ollama, `LLMConfig.mejor_disponible()` elige el mejor modelo instalado en
este orden:

1. **`qwen3:climasafe`** — el fine-tuneado (ver abajo), preferido si existe.
2. `qwen2.5:7b` — calidad máxima local (GPU).
3. `qwen3:1.7b` — mejor relación calidad/peso medida en el benchmark (LLM-003):
   clase 38 %, formato 100 %, inventa cifras 13 %, error de índice 0.297.
4. `qwen2.5:1.5b` — mínimo viable en CPU (el benchmark lo descartó para
   producción: inventa cifras en el 100 % de las respuestas).

Si no hay ningún LLM disponible, el sistema cae a **modo determinista** (sin
redacción LLM): la predicción y los factores siguen funcionando igual.

## Modelo fine-tuneado: qwen3:climasafe

Sobre Qwen3 se aplica un **LoRA** (fine-tuning de bajo rango) con un dataset
sintético de preguntas y respuestas ClimaSafe (pares generados con el propio
pipeline de predicción, para que las respuestas sean factuales). El resultado
se cuantiza a GGUF (q4_k_m) y se sirve con Ollama. Es el **preferido por
defecto**: si está instalado, `mejor_disponible()` lo prioriza.

## Modelo base vs modelo de instrucciones — conclusiones

> Estudio completo: [`documentacion/llm/base-vs-instruct.md`](https://github.com/ANFAIA/ClimaSafe/blob/main/documentacion/llm/base-vs-instruct.md)
> — incluye la comparación real de salidas y la decisión para el próximo LoRA.

Hay dos variantes del mismo modelo:

- **Modelo base (Qwen3 base):** entrenado con billones de textos para predecir
  la siguiente palabra. Sabe "escribir", pero no "responder": ante una pregunta
  continúa el texto en vez de dar una respuesta útil.
- **Modelo de instrucciones (Qwen3 instruct):** el mismo modelo, ajustado
  después con datos de "sigue instrucciones" (chat template + preferencias
  humanas, SFT + RLHF/DPO). Sabe comportarse como asistente: formato, ayuda,
  "no lo sé".

**Conclusión de este proyecto:** el primer fine-tuning se hizo sobre el modelo
**base** con un dataset de **instrucciones**. Ese desajuste (pedirle al base
que aprenda a la vez a comportarse como asistente y el formato ClimaSafe) es la
receta clásica de alucinaciones: el modelo da el formato correcto solo ~20 % de
las veces e inventa el resto.

**Qué hacer en el próximo LoRA:**

- Partir de **Qwen3 instruct**, no del base: ya sabe seguir instrucciones y
  solo tiene que aprender el dominio ClimaSafe → menos ejemplos necesarios y
  menos alucinaciones.
- El benchmark apoya la misma idea en los modelos sin fine-tunear: el
  `qwen3:1.7b` (mejor instruido) supera claramente al `qwen2.5:1.5b` (clase
  38 % vs 32 %, pero sobre todo inventa cifras 13 % vs 100 %).

## LLM remoto gratuito (HOST-001)

El LLM puede vivir fuera del portátil sin coste: **Groq free tier**, modelo
`groq/openai/gpt-oss-20b` — cuota publicada y verificada por llamada real
(30 RPM / 1.000 RPD / 8K TPM / 200K TPD). Una conversación completa del bot
son ~4.800 tokens medidos, así que la cuota da ~1 conversación/minuto y ~40
al día; el 429 no cuesta dinero y el bot degrada a plantilla determinista si
el servicio cae. El modelo anterior (`groq/llama-3.3-70b-versatile`) **ya no
existe** en el free tier (404 real, 18-08-2026); con `GEMINI_API_KEY` sola la
alternativa automática es `gemini/gemini-3.6-flash`.

> Estudio completo (cuotas de Groq, Gemini y OpenRouter, y la decisión):
> [`documentacion/bot/hosting_llm_gratis.md`](https://github.com/ANFAIA/ClimaSafe/blob/main/documentacion/bot/hosting_llm_gratis.md)

## RAG

Además del fine-tuning, los factores de riesgo y la documentación se indexan
con **sqlite-vec** (embeddings semánticos). Ante una pregunta, el sistema
recupera los fragmentos relevantes y el LLM responde citando las fuentes
(RAG sobre `documentacion/`).

### Modelo de embeddings: distiluse (RAG-006)

El retrieval se cambió **por los números**: el modelo por defecto es ahora
`distiluse-base-multilingual-cased-v2` (512 dims), que mejora el recall@5 en
los dos canales frente a la línea base (`all-MiniLM-L6-v2`, 384):

| Config | factores recall@5 | documentos recall@5 |
|---|---|---|
| Línea base (all-MiniLM-L6-v2) | 0.780 | 0.325 |
| **distiluse-base-multilingual-cased-v2** | **0.940** | **0.611** |

El solapamiento de chunks (200 caracteres) se probó y se **revertió**:
empeora el recall de documentos en ambos modelos (0.325→0.286 con MiniLM;
0.611→0.526 con distiluse), porque el chunking es por secciones y el solape
diluye el embedding. `CHUNK_OVERLAP = 0` queda como defecto.

> Comparativa completa:
> [`documentacion/rag_006_comparativa_embeddings.md`](https://github.com/ANFAIA/ClimaSafe/blob/main/documentacion/rag_006_comparativa_embeddings.md)

### QC del dataset de fine-tuning (LLM-017)

El dataset de entrenamiento se regeneró y pasó el QC con **0 hallazgos**
(`data/llm/train.jsonl` 300 pares + `val.jsonl` 100 pares): 0 críticas, 0
duplicados y 0 inputs incompletos en ambos splits (50 verificados por split;
los incoherentes menores de clase son ruido del detector, no del dataset).
Se compuso en bloques calor/frío con dedupe global por clave normalizada y
Jaccard de tokens > 0.9, cuotas por clase (las tres ≥ 10 %) y split
estratificado.

> Resumen del QC:
> [`documentacion/llm/qc-llm-017.md`](https://github.com/ANFAIA/ClimaSafe/blob/main/documentacion/llm/qc-llm-017.md)

## Estado actual y pipeline

- `climasafeai/llm/rag_qwen.py` — selección de modelo (LiteLLM), RAG y
  redacción (`_chat_litellm`).
- `climasafeai/llm/fine_tune.py` — entrenamiento LoRA y exportación a GGUF.
- `climasafeai/llm/generar_dataset.py` — dataset sintético desde el pipeline
  real de predicción.
- El entrenamiento se lanza desde el notebook de Colab
  (`notebooks/llm-fine-tuning-colab.ipynb`) con GPU T4 gratuita.
