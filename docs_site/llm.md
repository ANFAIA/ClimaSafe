# LLM

## Qué papel juega el LLM

ClimaSafe es **determinista por diseño**: la predicción, los factores y las
recomendaciones salen de modelos ML e índices, no de un LLM. El LLM se usa
solo para **redactar** respuestas en lenguaje natural (bot de Telegram y chat
web): recibe el resultado real del pipeline y lo explica con tus palabras.

## Modelo fine-tuneado: qwen3:climasafe

Sobre Qwen3 se aplica un **LoRA** (fine-tuning de bajo rango) con un dataset
sintético de preguntas y respuestas ClimaSafe (pares generados con el propio
pipeline de predicción, para que las respuestas sean factuales). El resultado
se cuantiza a GGUF (q4_k_m) y se sirve con Ollama.

El bot lo prioriza automáticamente: `check_ollama()` detecta `qwen3:climasafe`
y lo usa como modelo preferente sobre el Qwen3 base.

## Modelo base vs modelo de instrucciones

Hay dos variantes del mismo modelo:

- **Modelo base (Qwen3 base):** entrenado con billones de textos para predecir
  la siguiente palabra. Sabe "escribir", pero no "responder": ante una pregunta
  continúa el texto en vez de dar una respuesta útil.
- **Modelo de instrucciones (Qwen3 instruct):** el mismo modelo, ajustado
  después con datos de "sigue instrucciones" (chat template + preferencias
  humanas, SFT + RLHF/DPO). Sabe comportarse como asistente: formato, ayuda,
  "no lo sé".

**Por qué importa en este proyecto:** el primer fine-tuning se hizo sobre el
modelo **base** con un dataset de **instrucciones**. Ese desajuste (pedirle al
base que aprenda a la vez a comportarse como asistente y el formato ClimaSafe)
es la receta clásica de alucinaciones: el modelo da el formato correcto solo
~20 % de las veces e inventa el resto. La conclusión práctica: para el próximo
LoRA, partir de **Qwen3 instruct** — ya sabe seguir instrucciones y solo tiene
que aprender el dominio ClimaSafe, con menos ejemplos y menos alucinaciones.

## RAG

Además del fine-tuning, los factores de riesgo y la documentación se indexan
con **sqlite-vec** (embeddings semánticos). Ante una pregunta, el sistema
recupera los fragmentos relevantes y el LLM responde citando las fuentes
(RAG sobre `documentacion/`).

## Estado actual y pipeline

- `climasafeai/llm/fine_tune.py` — entrenamiento LoRA y exportación a GGUF.
- `climasafeai/llm/generar_dataset.py` — dataset sintético desde el pipeline
  real de predicción.
- `climasafeai/llm/rag_qwen.py` — selección de modelo, RAG y redacción.
- El entrenamiento se lanza desde el notebook de Colab
  (`notebooks/llm-fine-tuning-colab.ipynb`) con GPU T4 gratuita.
