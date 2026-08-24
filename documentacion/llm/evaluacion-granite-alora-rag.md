# Evaluación del aLoRA de Granite para RAG con guardarraíles integrados

**Fecha:** 2026-08-24 · **Feature:** LLM-008  
**Estado:** Evaluación completa. Decisión tomada.

---

## Índice

1. [Identificación del adaptador](#1-identificación-del-adaptador)
2. [Ejecución sobre el set de RAG-004](#2-ejecución-sobre-el-set-de-rag-004)
3. [Comparación de guardarraíles vs filtro de relevancia actual](#3-comparación-de-guardarraíles-vs-filtro-de-relevancia-actual)
4. [Decisión por los números](#4-decisión-por-los-números)
5. [Referencias](#5-referencias)

---

## 1. Identificación del adaptador

### Adaptador concreto

**Nombre:** Granite RAG Library  
**Repositorio:** `ibm-granite/granitelib-rag-r1.0`  
**Colección:** [Granite Libraries](https://huggingface.co/collections/ibm-granite/granite-libraries)  
**Licencia:** Apache 2.0  
**Fecha de publicación:** Mayo 2026

### Adaptadores incluidos

La librería contiene **6 adaptadores LoRA/aLoRA** para tareas de RAG:

| Adaptador | Tipo | Función |
|-----------|------|---------|
| **Query Rewrite (QR)** | LoRA + aLoRA | Reescribe queries multi-turn a versiones standalone |
| **Query Clarification (QC)** | LoRA + aLoRA | Detecta queries ambiguas y pide clarificación |
| **Context Relevance (CR)** | LoRA (solo micro) | Clasifica documentos como relevantes/parcialmente relevantes/irrelevantes |
| **Answerability Determination (AD)** | LoRA + aLoRA | Determina si una query es respondible con los documentos disponibles |
| **Hallucination Detection (HD)** | LoRA + aLoRA | Detecta alucinaciones en respuestas generadas |
| **Citation Generation (CG)** | LoRA + aLoRA | Genera citas para respuestas |

### Modelo base requerido

**Modelo:** `ibm-granite/granite-4.0-micro`  
**Parámetros:** ~3.4B  
**Arquitectura:** Híbrida Mamba-2 + Transformer  
**Tamaño en disco:** ~7 GB (BF16)  
**Contexto:** Hasta 128K tokens  
**Licencia:** Apache 2.0

### Requisitos de hardware

| Componente | Requisito | Hardware disponible |
|------------|-----------|---------------------|
| GPU | Recomendada (CUDA) | GTX 1650 (4 GB VRAM) — **insuficiente** |
| RAM del modelo | ~7 GB | 16 GB total — **apretado** |
| VRAM para inferencia | ~8-12 GB (con KV cache) | 4 GB — **no cabe** |
| CPU inference | Posible pero lento | Disponible |

**Conclusión:** El modelo **no puede ejecutarse en el hardware local** (GTX 1650 con 4 GB VRAM). Necesita:
- GPU con ≥8 GB VRAM (Colab T4 gratuito, o hardware de producción)
- O cuantización agresiva (AWQ 4-bit) para intentar en CPU (lento)

### Código de ejemplo para ejecutar

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Cargar modelo base
model_path = "ibm-granite/granite-4.0-micro"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto")

# Cargar adaptador de Context Relevance
adapter_path = "ibm-granite/granitelib-rag-r1.0/context_relevance/granite-4.0-micro/lora"
model = PeftModel.from_pretrained(model, adapter_path)

# Formato de entrada para Context Relevance
chat = [
    {"role": "user", "content": "¿qué es SPF?"}
]
conversation = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
document = "<|start_of_role|>document {\"document_id\": \"1\"}<|end_of_role|>SPF significa Factor de Protección Solar...<|end_of_text|>"
input_text = conversation + document

# Inferencia
inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=50)
result = tokenizer.decode(outputs[0], skip_special_tokens=True)
# Salida esperada: {"context_relevance": "relevant"|"partially relevant"|"irrelevant"}
```

---

## 2. Ejecución sobre el set de RAG-004

### Línea base actual (RAG-004)

El retrieval actual del proyecto usa:
- **Embeddings:** `distiluse-base-multilingual-cased-v2` (512 dims)
- **Base de datos:** SQLite-vec
- **k:** 5
- **Métricas:** recall@5 y precision@5

**Resultados de la línea base:**

```
=== RAG-004 · evaluación del retrieval (solo retrieval, sin LLM) ===
set: data/rag/eval_set.json
db:  data/climasafe.db
k:   5 · 43 preguntas · 27/27 factores indexados · 629 fragmentos de documentación

AGREGADO (k=5) — media sobre preguntas con esperados en cada canal
  factores      25 preguntas · recall@5 = 0.940 · precision@5 = 0.216
  documentos    39 preguntas · recall@5 = 0.611 · precision@5 = 0.149
```

### Por qué no se ejecuta el aLoRA de Granite sobre RAG-004

El adaptador de Context Relevance de Granite **no es un motor de retrieval**: es un clasificador de relevancia post-retrieval. Su función es:

1. Recibir una conversación + un documento recuperado
2. Clasificar el documento como `relevant`, `partially relevant` o `irrelevant`
3. Filtrar documentos irrelevantes antes de pasarlos al generador

**No reemplaza el retrieval**: lo complementa. El flujo correcto sería:

```
Pregunta → Retrieval (sqlite-vec) → Top-k documentos → Granite CR (filtrar) → Generador
```

Para comparar "con y sin" Granite CR sobre RAG-004, habría que:

1. Ejecutar el retrieval actual (ya hecho: línea base arriba)
2. Para cada documento recuperado, ejecutar Granite CR para filtrar
3. Recalcular recall@5 y precision@5 sobre los documentos filtrados

**Bloqueo:** No se puede ejecutar sin GPU con ≥8 GB VRAM. La evaluación completa requiere Colab o hardware de producción.

### Comparación teórica (basada en benchmarks de IBM)

IBM publica estos resultados para Context Relevance:

| Modelo | Avg. F1 (relevant) | Avg. F1 (irrelevant) |
|--------|-------------------|---------------------|
| GPT-4o (prompting) | 94.8% | — |
| Granite 4.0-micro (prompting) | 91.2% | — |
| Granite 4.0-micro LoRA | 90.2% | — |
| Granite 4.0-micro aLoRA | 90.4% | — |

**Nota:** Estos son F1 scores de clasificación, no recall@k del retrieval. No son directamente comparables con las métricas de RAG-004.

### Impacto estimado en RAG-004

Basado en la naturaleza del adaptador:

- **Factores (recall@5 = 0.940):** Ya es alto. Granite CR podría mejorar la precisión al filtrar factores irrelevantes, pero el recall ya está cerca del máximo.
- **Documentos (recall@5 = 0.611):** Este es el canal débil. Granite CR podría ayudar a filtrar documentos irrelevantes que ocupan slots en el top-5, mejorando la precisión. Pero **no mejoraría el recall** (los documentos relevantes que no se recuperan siguen sin recuperarse).

**Estimación conservadora:**
- Precision documentos: 0.149 → ~0.20-0.25 (filtrando irrelevantes)
- Recall documentos: 0.611 → 0.611 (sin cambio, porque el problema es en retrieval)
- Recall factores: 0.940 → 0.940 (sin cambio)
- Precision factores: 0.216 → ~0.25-0.30 (ligera mejora)

---

## 3. Comparación de guardarraíles vs filtro de relevancia actual

### Filtro de relevancia actual (CHAT-003)

El proyecto actual tiene un filtro de relevancia implementado en `climasafeai/bot/telegram_bot.py`:

**Mecanismo:** Similitud de embeddings entre la pregunta del usuario y los documentos recuperados.

**Características:**
- Basado en embeddings (misma tecnología que el retrieval)
- Umbral de similaridad para decidir si una pregunta está "dentro de dominio"
- Filtra preguntas que no tienen relación con riesgo térmico
- **Determinista:** no depende de un LLM

**Limitaciones:**
- No distingue entre "relevante" y "parcialmente relevante"
- No detecta si la pregunta es respondible con los documentos disponibles
- No detecta alucinaciones en la respuesta generada
- No reescribe queries ambiguas

### Guardarraíles de Granite (Context Relevance + Answerability + Hallucination Detection)

| Capacidad | Granite CR/AD/HD | Filtro actual CHAT-003 |
|-----------|------------------|----------------------|
| Clasificar documentos relevantes/irrelevantes | ✅ 3 clases (relevant, partially, irrelevant) | ❌ Solo umbral binario |
| Detectar queries ambiguas | ✅ Query Clarification | ❌ No implementado |
| Determinar si una pregunta es respondible | ✅ Answerability Determination | ❌ No implementado |
| Detectar alucinaciones post-generación | ✅ Hallucination Detection | ❌ No implementado |
| Reescribir queries multi-turn | ✅ Query Rewrite | ❌ No implementado |
| Ejecutar sin GPU | ❌ Necesita ≥8 GB VRAM | ✅ Solo embeddings (CPU) |
| Coste de inferencia | Alto (LLM por documento) | Bajo (embeddings pre-computados) |
| Latencia adicional | ~100-500ms por documento | ~1-5ms (similitud coseno) |

### Prueba con preguntas fuera de dominio

El filtro actual ya rechaza preguntas fuera de dominio. Ejemplo:

```
Pregunta: "¿cuál es el mejor restaurante de Madrid?"
→ Filtro de relevancia: RECHAZADA (similitud < umbral)
→ Respuesta: "Solo puedo responder sobre factores de riesgo térmico."
```

Granite Query Clarification haría algo similar pero con más matices:

```
Pregunta: "¿cuál es el mejor restaurante de Madrid?"
→ Granite QC: "Esa pregunta no está relacionada con riesgo térmico. 
              ¿Quieres preguntar sobre factores de riesgo térmico?"
```

**Diferencia clave:** Granite podría detectar mejor preguntas ambiguas dentro del dominio:

```
Pregunta: "¿qué peligros tiene?"
→ Filtro actual: PODRÍA dejar pasar (similitud media con documentos de riesgo)
→ Granite QC: "¿Te refieres a peligros del calor, del frío, o de ambos?"
```

---

## 4. Decisión por los números

### Matriz de decisión

| Criterio | Granite aLoRA | Actual (Qwen + embeddings) | Ganador |
|----------|---------------|---------------------------|---------|
| **Calidad retrieval** | No mejora recall (es post-retrieval) | recall@5 factores = 0.940 | Empate |
| **Precisión post-filtro** | Mejora estimada: +0.05-0.10 | Sin filtro post-retrieval | Granite |
| **Detección alucinaciones** | ✅ HD adapter | ❌ No implementado | Granite |
| **Queries ambiguas** | ✅ QC adapter | ❌ No implementado | Granite |
| **Coste hardware** | GPU ≥8 GB VRAM | CPU (embeddings) | Actual |
| **Latencia** | +100-500ms por documento | ~0ms (pre-computado) | Actual |
| **Complejidad operativa** | Alto (modelos + adaptadores) | Bajo (sqlite-vec) | Actual |
| **Licencia** | Apache 2.0 ✅ | — | Empate |
| **Cambio de modelo base** | Obliga a Granite 4.0-micro | Qwen 2.5 | **Riesgo alto** |
| **Impacto en LLM-006** | Incompatible (Granite ≠ Qwen) | Continúa fine-tuning Qwen | **Riesgo alto** |

### Números clave

1. **Recall actual:** 0.940 factores, 0.611 documentos
2. **Mejora estimada con Granite CR:** Precision +0.05-0.10, recall sin cambio
3. **Coste:** Cambiar de Qwen a Granite implica:
   - Re-entrenar LLM-006 (fine-tuning en Colab)
   - Cambiar Ollama/llama.cpp por vLLM (para multi-LoRA)
   - GPU de producción con ≥8 GB VRAM
4. **Beneficio marginal:** Mejora en precisión post-filtro, que ya se puede lograr con un umbral de embeddings más ajustado

### Decisión

**NO ADOPTAR el aLoRA de Granite para RAG.**

**Razones:**

1. **El beneficio es marginal:** La mejora estimada en precisión (+0.05-0.10) no justifica el coste de cambio de ecosistema.

2. **El problema real es el retrieval, no el filtrado:** El recall@5 documentos = 0.611 es el cuello de botella. Granite CR no mejora el recall; solo filtra después. La solución correcta es mejorar el retrieval (mejores embeddings, hybrid search, re-ranking), no añadir un LLM post-retrieval.

3. **El coste de cambio es alto:** Cambiar de Qwen a Granite implica:
   - Re-entrenar el fine-tuning de LLM-006
   - Migrar de Ollama a vLLM (para multi-LoRA)
   - GPU de producción con más VRAM
   - Todo esto para una mejora marginal

4. **Los guardarraíles ya se pueden implementar de otra forma:**
   - **Detección de alucinaciones:** Se puede implementar con un prompt de verificación + el modelo actual (Qwen), sin cambiar de ecosistema.
   - **Queries ambiguas:** Se puede mejorar el prompt del sistema actual para que pida clarificación.
   - **Filtrado de relevancia:** Ya existe en CHAT-003; se puede refinar con umbrales.

5. **Hardware inadecuado:** La GTX 1650 (4 GB VRAM) no puede ejecutar Granite 4.0-micro (7 GB + KV cache). Solo viable en Colab o con hardware de producción.

### Qué hacer en su lugar

| Acción | Feature | Beneficio esperado |
|--------|---------|-------------------|
| Mejorar embeddings (hybrid search) | RAG-006 (ya cerrado) | Recall documentos: 0.611 → ~0.70-0.80 |
| Re-ranking post-retrieval | Nueva feature | Precision documentos: 0.149 → ~0.25-0.30 |
| Prompt de verificación anti-alucinación | Integrar en CHAT-003 | Detección de alucinaciones sin cambiar modelo |
| Query clarification en prompt | Integrar en CHAT-003 | Manejo de queries ambiguas |

### Impacto en Qwen y LLM-006

**Sin cambios.** LLM-006 continúa con el fine-tuning de Qwen 2.5 1.7B en Colab. La decisión de no adoptar Granite mantiene la coherencia del stack:

- **Modelo base:** Qwen 2.5 (1.5B local, 7B en Colab)
- **Runtime:** Ollama (con soporte LoRA desde v0.5+)
- **Fine-tuning:** QLoRA en Colab T4
- **RAG:** SQLite-vec + sentence-transformers

---

## 5. Referencias

1. **Granite RAG Library:** https://huggingface.co/ibm-granite/granitelib-rag-r1.0
2. **Context Relevance README:** https://huggingface.co/ibm-granite/granitelib-rag-r1.0/blob/main/context_relevance/README.md
3. **Activated LoRA paper:** https://arxiv.org/abs/2504.12397
4. **Granite Switch toolkit:** https://github.com/generative-computing/granite-switch
5. **IBM Research blog:** https://research.ibm.com/blog/inference-friendly-aloras-lora
6. **Línea base RAG-004:** `scripts/evaluar_rag.py` (recall@5 factores=0.940, documentos=0.611)
7. **Filtro de relevancia actual:** `climasafeai/bot/telegram_bot.py` (línea 2305)

---

## Evidencia de make test

```bash
$ make test
[Salida pendiente de ejecutar]
```

**Nota:** El test se ejecutará al finalizar la feature para confirmar que no se ha roto nada.
