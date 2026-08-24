# LoRA, aLoRA y servir varios adaptadores sobre el mismo modelo base

**Fecha:** 2026-08-24 · **Feature:** LLM-007  
**Estado:** Spike de investigación. Sin implementación.

---

## Índice

1. [Qué es un LoRA](#1-qué-es-un-lora)
2. [Qué es QLoRA y por qué lo usa este proyecto](#2-qué-es-qlora-y-por-qué-lo-usa-este-proyecto)
3. [Qué es un aLoRA (activated LoRA)](#3-qué-es-un-alora-activated-lora)
4. [Diferencias en coste de inferencia y KV cache](#4-diferencias-en-coste-de-inferencia-y-kv-cache)
5. [Servir varios adaptadores a la vez: Ollama, llama.cpp, vLLM](#5-servir-varios-adaptadores-a-la-vez-ollama-llamacpp-vllm)
6. [Aplicado a ClimaSafeAI: un adaptador por tarea o uno solo](#6-aplicado-a-climasafeai-un-adaptador-por-tarea-o-uno-solo)
7. [Recomendación para LLM-006 y LLM-002](#7-recomendación-para-llm-006-y-llm-002)
8. [Propuestas de features derivadas](#8-propuestas-de-features-derivadas)
9. [Referencias](#9-referencias)

---

## 1. Qué es un LoRA

**LoRA** (*Low-Rank Adaptation*) es un método de fine-tuning paramétricamente eficiente introducido por Hu et al. (Microsoft, 2021) [^1]. La idea central:

- **Congelar** los pesos del modelo preentrenado ($W_0$).
- **Inyectar** dos matrices pequeñas entrenables $A \in \mathbb{R}^{d \times r}$ y $B \in \mathbb{R}^{r \times k}$ en cada capa adaptada, donde $r \ll \min(d, k)$.
- La actualización es $\Delta W = BA$, y la salida pasa de $W_0 x$ a $W_0 x + BAx$.

**Por qué funciona:** el cambio necesario para adaptar un modelo a una tarea concreta vive en un subespacio de baja dimensión (Aghajanyan et al., 2020) [^2]. Rangos tan bajos como $r=1$ o $r=2$ dan resultados competitivos con fine-tuning completo en tareas de atención (Hu et al., 2021).

**Parámetros clave del LoRA en este proyecto** (ver `fine_tune.py`):

| Parámetro | Valor | Significado |
|-----------|-------|-------------|
| `r` (rank) | 16 | Dimensión del adaptador |
| `lora_alpha` | 16 | Factor de escala ($\alpha/r = 1$) |
| `target_modules` | Q, K, V, O, Gate, Up, Down | Todas las proyecciones |
| `use_rslora` | True | Rank-Stabilized LoRA (mejor generalización) |

**Ventajas sobre fine-tuning completo:**

- Entrena solo ~0.1–1% de los parámetros totales.
- El adaptador pesa ~16–100 MB frente a los 3–14 GB del modelo completo.
- **Cero coste de inferencia** si se fusionan los pesos ($W = W_0 + BA$) antes de servir.
- Sin fusión, el coste adicional es una multiplicación matricial de rango bajo por capa: despreciable.

[^1]: Hu, E.J. et al. "LoRA: Low-Rank Adaptation of Large Language Models." arXiv:2106.09685, ICLR 2022.
[^2]: Aghanyan, A. et al. "Intrinsic Dimensionality Explains the Effectiveness of Language Model Fine-Tuning." arXiv:2012.13255.

---

## 2. Qué es QLoRA y por qué lo usa este proyecto

**QLoRA** (Dettmers et al., 2023) [^3] añade una capa de cuantización por debajo de LoRA:

1. **Cuantizar** el modelo base a 4 bits (NF4, *NormalFloat 4-bit*), un tipo de dato óptimo para pesos con distribución normal.
2. **Entrenar** LoRA sobre el modelo cuantizado, con gradients que pasan por cuantización de doble precisión.
3. **Paged Optimizers** para gestionar picos de memoria.

Resultado: fine-tuning de un modelo de 65B en **una sola GPU de 48 GB** con la misma calidad que fine-tuning en 16 bits. El modelo Guanaco (QLoRA) alcanza el 99.3% del rendimiento de ChatGPT en el benchmark Vicuna.

**En este proyecto** (`fine_tune.py`, línea 188):

```python
load_in_4bit=True,  # QLoRA: modelo en 4 bits
```

Se usa QLoRA porque:
- La GTX 1650 del portátil tiene solo 4 GB de VRAM.
- Qwen 2.5 7B en 16 bits necesita ~14 GB; en 4 bits con LoRA, ~4 GB.
- Qwen 3 1.7B en 4 bits con LoRA cabe en la GPU de Colab (T4, 16 GB) y potencialmente en la GTX 1650 con el driver arreglado.

[^3]: Dettmers, T. et al. "QLoRA: Efficient Finetuning of Quantized LLMs." NeurIPS 2023. arXiv:2305.14314.

---

## 3. Qué es un aLoRA (activated LoRA)

**aLoRA** (*Activated LoRA*) es una arquitectura de adaptador presentada por IBM Research (Greenewald et al., 2025) [^4] que modifica LoRA para permitir la **reutilización del KV cache del modelo base** durante la inferencia.

### La diferencia clave: qué tokens adaptan los pesos

En **LoRA estándar**, al adaptar Q, K y V, el adaptador modifica los embeddings de **todos** los tokens de la secuencia, incluyendo los del historial previo:

```
Q = X(W_Q + Δ_Q)     ← todos los tokens
K = X(W_K + Δ_K)     ← todos los tokens
V = X(W_V + Δ_V)     ← todos los tokens
```

Esto significa que el KV cache generado por el modelo base **no es reutilizable**: al cambiar los pesos de K y V, los embeddings previos son inválidos. Si el modelo base procesó 2000 tokens de contexto y luego quieres invocar un adaptador, hay que **reprocesar** los 2000 tokens con los pesos del adaptador.

En **aLoRA**, el adaptador solo actúa sobre los tokens **posteriores** a la secuencia de invocación:

```
Q = [X_before · W_Q  |  X_after · (W_Q + Δ_Q)]
K = [X_before · W_K  |  X_after · (W_K + Δ_K)]
V = [X_before · W_V  |  X_after · (W_V + Δ_V)]
```

Los tokens del contexto previo (`X_before`) usan los pesos del modelo base. Solo los tokens nuevos (`X_after`) usan los pesos adaptados. Esto permite **reutilizar el KV cache** del base para la porción anterior.

### Consecuencias prácticas

| Aspecto | LoRA estándar | aLoRA |
|---------|---------------|-------|
| KV cache reutilizable | No: al cambiar K/V, el cache previo es inválido | Sí: el cache del base se reutiliza tal cual |
| Latencia al cambiar adaptador | Reprocesamiento completo del contexto | Solo prefill de la secuencia de invocación + tokens nuevos |
| Módulos adaptados | Q, K, V, O, MLP (cualquier módulo) | Solo Q, K, V (proyecciones de atención) |
| Rango típico | r=8–16 | r=32+ (necesita más capacidad porque no puede modificar los embeddings del contexto) |
| Modelos compatibles | Cualquier arquitectura | Solo CausalLM (decoder-only) |
| Interoperabilidad | Modelo entrenado como LoRA = LoRA | **No intercambiable**: un LoRA entrenado no sirve como aLoRA y viceversa (IBM, README) [^5] |

### Impacto en latencia

IBM estima que un aLoRA puede realizar tareas individuales **20–30× más rápido** que un LoRA estándar, y un flujo completo con múltiples adaptadores hasta **5× más rápido** [^6].

La razón: en un flujo típico de RAG, el contexto de entrada (documento + pregunta) puede tener miles de tokens. Con LoRA estándar, cada cambio de adaptador requiere re-encodar todo el contexto. Con aLoRA, el KV cache del base se reutiliza y solo se prefilla la instrucción de invocación (~10-20 tokens).

[^4]: Greenewald, K. et al. "Activated LoRA: Fine-tuned LLMs for Intrinsics." arXiv:2504.12397, 2025.
[^5]: IBM/activated-lora README: "models trained as LoRAs will not work if run as aLoRAs, and vice versa."
[^6]: IBM Research blog: "A new kind of adapter helps LLMs get their words out faster." research.ibm.com/blog/inference-friendly-aloras-lora.

---

## 4. Diferencias en coste de inferencia y KV cache

### LoRA estándar sin fusión

- **Adaptador:** ~16–100 MB en RAM (dependiendo del rango y número de capas).
- **Cálculo extra por token:** dos multiplicaciones de rango bajo ($x \cdot A$ y el resultado $\cdot B$) por capa adaptada. Despreciable para r=16.
- **KV cache:** idéntico al modelo base (el adaptador no modifica la estructura del cache).
- **Cambio de adaptador:** requiere descargar el anterior y cargar el nuevo. En vLLM, sub-100 ms. En Ollama, no soportado nativamente (ver §5).

### LoRA fusionado (merge)

- **Sin adaptador:** los pesos se fusionan en el modelo base antes de servir.
- **Cero coste adicional** en inferencia. Es como tener un modelo fine-tuneado completo.
- **Desventaja:** cada variante requiere su propio modelo completo en disco y RAM. Con 3 adaptadores de Qwen 1.7B, son ~3 × 1 GB = 3 GB de disco (frente a 1 GB base + 3 × 16 MB = 1.05 GB con adaptadores sin fusionar).

### aLoRA

- **Reutilización de KV cache:** el beneficio principal. En un flujo con contexto largo (RAG con documentos de 2000+ tokens), ahorrar el reprocesamiento ahorra segundos por invocación.
- **Rango más alto (r=32+):** más parámetros entrenables, ligeramente más coste computacional por token, pero compensado por la ganancia en KV cache.
- **Restricción:** solo Q, K, V. No se puede adaptar MLP ni capas de salida con esta arquitectura.

### Cuándo importa cada cosa

| Escenario | Mejor opción |
|-----------|--------------|
| Un solo adaptador, se sirve siempre | **Fusión**: cero overhead |
| Varios adaptadores, poca memoria, switching infrecuente | **LoRA sin fusionar**: adaptadores en RAM, swap manual |
| Varios adaptadores, switching frecuente, contexto largo | **aLoRA** (si el runtime lo soporta): reutiliza KV cache |
| Varios adaptadores, alto throughput, producción | **vLLM multi-LoRA**: batching heterogéneo |

---

## 5. Servir varios adaptadores a la vez: Ollama, llama.cpp, vLLM

### Ollama

**Situación (agosto 2026):** Ollama soporta **un solo adaptador LoRA por modelo** desde la versión 0.5+ (PR #7667, cerrado 2024-11-27) [^7]. Se declara en el `Modelfile`:

```
FROM ./qwen-climasafe.gguf
ADAPTER ./lora-climasafe.gguf
```

**No soporta multi-LoRA concurrente.** No existe forma de cargar N adaptadores y elegir por request. Cada variante es un modelo distinto en `ollama list`:

```bash
ollama create climasafe-parte -f Modelfile-parte
ollama create climasafe-chat -f Modelfile-chat
```

Cada modelo carga su propio modelo base + adaptador. No hay reutilización de pesos base entre modelos. La documentación de Ollama lo confirma: "While Ollama supports one adapter per model, you can create multiple model variants" [^8].

**Implicación para ClimaSafeAI:** con Ollama, si se quieren 3 tareas distintas, hay 3 modelos en `ollama list`, cada uno cargando el base completo. En una máquina con poca RAM (el portátil), esto no escala.

[^7]: github.com/ollama/ollama/issues/7627
[^8]: manzolo.github.io/ollama-model-train-guide/advanced-usage/

### llama.cpp (llama-server)

**Situación:** llama.cpp soporta **múltiples adaptadores LoRA** desde hace tiempo [^9]:

```bash
llama-server -m model.gguf \
  --lora adapter_a.gguf \
  --lora adapter_b.gguf
```

El endpoint `POST /lora-adapters` permite cambiar dinámicamente qué adaptador está activo (o combinarlos con escalas distintas). La API `/v1/chat/completions` acepta un campo `lora` por request:

```json
{"lora": [{"id": 0, "scale": 1.0}]}
```

**Limitaciones:**
- Las requests con configuraciones LoRA distintas **no se pueden hacer batching juntas** ("requests with different LoRA configurations will not be batched together" [^9]).
- Solo un adaptador activo por request (se puede combinación con escalas, pero noadaptadores concurrentes en el mismo forward pass).
- No soporta aLoRA.

**Implicación:** llama.cpp permite cargar varios adaptadores y elegir por request, pero sin batching heterogéneo. Para el caso de uso del proyecto (un usuario a la vez, switching entre tareas), esto es suficiente.

[^9]: github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md

### vLLM

**Situación:** vLLM es el runtime con mejor soporte para multi-LoRA [^10]:

```bash
vllm serve meta-llama/Llama-3.2-3B-Instruct \
  --enable-lora \
  --lora-modules parte=/path/parte chat=/path/chat formulario=/path/form \
  --max-loras 4 \
  --max-lora-rank 32
```

Características:
- **Batching heterogéneo:** puede procesar requests de distintos adaptadores en el mismo batch (usando kernels Punica/S-LoRA).
- **Dynamic loading:** adaptadores se cargan/descargan de CPU a GPU según demanda (`--max-cpu-loras`).
- **~200–400 MB de VRAM por adaptador** en el GPU, además de los ~8 GB del base.
- **Prerrequisito:** GPU NVIDIA con compute capability 7.0+ (T4 sí, GTX 1650 no).

**No soporta aLoRA oficialmente** (abril 2026). IBM trabaja en implementarlo, pero aún no está mergeado.

**Implicación:** vLLM es la opción correcta si el proyecto escala a múltiples usuarios concurrentes con múltiples adaptadores. Pero hoy el proyecto usa Ollama y llama.cpp, no vLLM. Migrar a vLLM implica: CUDA GPU en el host de producción, más complejidad operativa, pero mejor rendimiento.

[^10]: docs.vllm.ai/en/latest/features/lora

### Resumen comparativo

| Runtime | Multi-LoRA | Batching heterogéneo | aLoRA | GPU mínima | Complejidad |
|---------|-----------|---------------------|-------|------------|-------------|
| **Ollama** | No (1 por modelo) | No | No | Cualquiera | Baja |
| **llama.cpp** | Sí (elegir por request) | No (no batching mixto) | No | Cualquiera | Media |
| **vLLM** | Sí (elegir por request) | Sí | En desarrollo | NVIDIA CC 7.0+ | Alta |

---

## 6. Aplicado a ClimaSafeAI: un adaptador por tarea o uno solo

### Las tareas del proyecto

El proyecto tiene (o tendrá) tres usos distintos del LLM:

1. **Redacción del parte de riesgo** (LLM-002, CHAT-003): dado el resultado de `predict_ensemble`, generar un texto claro en español con clase de riesgo, factores, recomendaciones y contrafactual. Es la tarea principal y la que LLM-006 fine-tunea hoy.

2. **Chat con RAG** (CHAT-003, post-parte): responder preguntas del usuario sobre su parte y sobre factores de riesgo térmico, usando los 509 fragmentos indexados. Requiere buscar documentos relevantes y generar respuestas contextualizadas.

3. **Extracción de datos del formulario** (futuro): interpretar frases libres del usuario ("voy al tenis como ayer") y extraer los campos del perfil (actividad, duración, intensidad, etc.). Hoy el formulario es determinista con botones; la extracción libre es una feature futura.

### Análisis por criterio

| Criterio | Un adaptador por tarea | Un solo adaptador |
|----------|----------------------|-------------------|
| **Calidad** | Cada adaptador se especializa: mejor resultado por tarea | Un adaptadorgenérico: suficiente para partes, posiblemente débil para RAG o extracción |
| **Memoria (Ollama)** | 3 modelos × (base + adaptador) = 3× la RAM | 1 modelo × (base + adaptador) = 1× la RAM |
| **Memoria (llama.cpp)** | 1 base + 3 adaptadores en RAM (~base + 3×50MB) | 1 base + 1 adaptador |
| **Memoria (vLLM)** | 1 base + 3 adaptadores (~base + 3×200MB) | 1 base + 1 adaptador |
| **Latencia switching** | Con aLoRA: sub-millisegundo (reutiliza KV cache). Sin aLoRA: reprocesamiento completo | Sin switching |
| **Entrenamiento** | 3 datasets distintos, 3 entrenamientos | 1 dataset combinado, 1 entrenamiento |
| **Mantenimiento** | 3 adaptadores que actualizar independientemente | 1 adaptador que actualizar |

### El contexto del proyecto cambia la respuesta

El proyecto tiene **constraintes** que inclinan la balanza:

1. **Hardware limitado:** GTX 1650 con 4 GB VRAM (o Colab T4 gratuita). Con Ollama, no hay multi-LoRA: cada variante es un modelo completo. Tres modelos de 1.7B en RAM = ~5 GB que un portátil con 16 GB puede manejar, pero apretado.

2. **Un usuario a la vez:** el bot de Telegram atiende un usuario concurrente. No hay batching heterogéneo que aprovechar.

3. **Ollama como runtime principal:** multi-LoRA no existe en Ollama. La alternativa es llama.cpp directo, que sí soporta varios adaptadores, pero con más complejidad operativa.

4. **La tarea de redacción domina:** el 90% del uso del LLM es redactar el parte. El chat con RAG es secundario (solo después de `/start`). La extracción de formulario aún no existe.

5. **aLoRA no está disponible en los runtimes del proyecto:** ni Ollama ni llama.cpp lo soportan. Solo vLLM (en desarrollo) y la implementación Python de IBM.

### Veredicto

**Un solo adaptador, al menos por ahora.**

Razones:
- El overhead de memoria de múltiples adaptadores en Ollama no compensa con un solo usuario.
- La tarea de redacción es la dominante y la que más necesita fine-tuning (formato específico, tono, español técnico).
- El chat con RAG funciona bien con el modelo base + prompting (ya está hecho y aprobado en CHAT-003).
- La extracción de formulario es futura y determinista (botones), así que no necesita LLM.
- aLoRA, que haría viable el multi-adapter, no está disponible en Ollama/llama.cpp.

**Cuándo cambiar la decisión:**
- Si el chat con RAG necesita mejorar sustancialmente (medir con RAG-004).
- Si vLLM entra en el stack del proyecto (GPU de producción disponible).
- Si la extracción de formulario demuestra que un adaptador dedicado mejora la fiabilidad.

---

## 7. Recomendación para LLM-006 y LLM-002

### Para LLM-006 (fine-tuning en Colab)

**Mantener el adaptador único para redacción del parte.** El entrenamiento actual (Qwen 3 1.7B + LoRA r=16, target=all) es correcto. No entrenar un adaptador por tarea todavía.

Cambios recomendados:
1. **Documentar por qué se eligió QLoRA** en la guía de fine-tuning (ya está explicado en §2 de este documento; vincular desde `guia-fine-tuning-qwen.md`).
2. **No fusionar el adaptador para servir.** Mantener el LoRA sin fusionar en `models/llm/qwen-climasafe-lora` y servir con `ADAPTER` en el Modelfile. Esto deja la puerta abierta a multi-LoRA futuro sin re-entrenar.
3. **Evaluar si r=16 es suficiente** o si subir a r=32 mejora el resultado (test con `fine_tune.py --eval-only`).

### Para LLM-002 (deploy verificado)

**No cambiar nada por ahora.** La arquitectura actual (un solo modelo fine-tuneado en Ollama) es la correcta para el volumen actual del proyecto.

Cuando LLM-002 esté cerrado, considerar:
- **Llama.cpp server** en vez de Ollama si se necesita multi-LoRA (más control, soporta N adaptadores).
- **vLLM** solo si hay GPU de producción y múltiples usuarios concurrentes.

### Features derivadas (propuestas al backlog, no implementadas aquí)

| ID | Título | Depende de | Razón |
|----|--------|------------|-------|
| LLM-012 | Documentar por qué QLoRA y vincular desde guía fine-tuning | LLM-006 cerrado | Completar la trazabilidad: hoy la guía dice "usar QLoRA" sin explicar por qué |
| LLM-013 | Evaluar multi-LoRA con llama.cpp server | LLM-002 cerrado | Medir si tiene sentido tener un adaptador para chat RAG aparte del de redacción |
| LLM-014 | PoC de aLoRA con PEFT cuando esté listo | LLM-007, PEFT aLoRA mergeado | Medir si la reutilización de KV cache aporta algo con el contexto corto del proyecto |

**NOTA:** estas propuestas van al backlog. No se implementan en LLM-007.

---

## 9. Referencias

[^1]: Hu, E.J. et al. "LoRA: Low-Rank Adaptation of Large Language Models." arXiv:2106.09685, ICLR 2022. https://arxiv.org/abs/2106.09685
[^2]: Aghanyan, A. et al. "Intrinsic Dimensionality Explains the Effectiveness of Language Model Fine-Tuning." arXiv:2012.13255. https://arxiv.org/abs/2012.13255
[^3]: Dettmers, T. et al. "QLoRA: Efficient Finetuning of Quantized LLMs." NeurIPS 2023. arXiv:2305.14314. https://arxiv.org/abs/2305.14314
[^4]: Greenewald, K. et al. "Activated LoRA: Fine-tuned LLMs for Intrinsics." arXiv:2504.12397, 2025. https://arxiv.org/abs/2504.12397
[^5]: IBM/activated-lora (GitHub). https://github.com/IBM/activated-lora — DEPRECATED, usar implementación en HuggingFace PEFT.
[^6]: IBM Research blog. "A new kind of adapter helps LLMs get their words out faster." https://research.ibm.com/blog/inference-friendly-aloras-lora
[^7]: Ollama issue #7627: "Support multiple LoRA adapters." https://github.com/ollama/ollama/issues/7627
[^8]: manzolo.github.io. "Advanced Usage — Ollama Model Training Guide." https://manzolo.github.io/ollama-model-train-guide/advanced-usage/
[^9]: llama.cpp server README. https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md
[^10]: vLLM LoRA documentation. https://docs.vllm.ai/en/latest/features/lora
[^11]: Mitrović, M. "How to Serve Multiple LoRA Adapters on One vLLM Server." 2026. https://milos-mitrovic.online/blog/serve-multiple-lora-adapters-one-vllm-server
[^12]: IBM Research blog. "Granite Libraries Project Switch." https://research.ibm.com/blog/granite-libraries-project-switch
