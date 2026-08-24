# Difusión aplicada a modelos de lenguaje: investigación vs LoRA/QLoRA

**Fecha:** 2026-08-24
**Feature:** LLM-012
**Estado:** Spike de investigación. Sin implementación.

---

## Índice

1. [Qué es un modelo de lenguaje por difusión](#1-qué-es-un-modelo-de-lenguaje-por-difusión)
2. [Diferencia frente al entrenamiento autorregresivo](#2-diferencia-frente-al-entrenamiento-autorregresivo)
3. **⚠️ NO confundir con difusión para series temporales** (#3-no-confundir-con-difusión-para-series-temporales)
4. [Modelos representativos: MDLM, SEDD, Plaid, LLaDA, Dream](#4-modelos-representativos-mdlm-seddy-plaid-llada-dream)
5. [Requisitos de hardware y viabilidad en este proyecto](#5-requisitos-de-hardware-y-viabilidad-en-este-proyecto)
6. [Comparación con LoRA/QLoRA (ruta de LLM-006)](#6-comparación-con-loraqlora-ruta-de-llm-006)
7. [Recomendación](#7-recomendación)
8. [Referencias](#8-referencias)

---

## 1. Qué es un modelo de lenguaje por difusión

Un **modelo de lenguaje por difusión** (*Diffusion Language Model*, DLM) adapta la idea de los modelos de difusión del dominio continuo (imágenes, audio) al dominio discreto del texto. En lugar de generar tokens de izquierda a derecha, el DLM:

1. **Proceso de ruido (forward):** parte de una secuencia limpia y va enmascarando o corrompiendo tokens progresivamente hasta llegar a ruido puro (típicamente una distribución uniforme o máscara completa).
2. **Proceso de denoising (reverse):** entrena un modelo Transformer para recuperar la secuencia original paso a paso, empezando desde el ruido y refinando iterativamente.
3. **Generación (inferencia):** parte de tokens aleatorios y aplica múltiples pasos de denoising en paralelo hasta producir texto coherente.

La ventaja teórica es la **decodificación paralela**: como no se genera token por token secuencialmente, la inferencia puede resolver múltiples posiciones a la vez, potencialmente con menor latencia secuencial que un modelo autorregresivo.

### Variantes principales

| Tipo | Ejemplo | Cómo trabaja |
|------|---------|--------------|
| **Discreto con máscara (MDM)** | MDLM, LLaDA, Dream | Enmascara tokens, entrena a recuperarlos; similar a BERT pero iterativo |
| **Discreto con puntuación** | SEDD | Aprende una función de puntuación del espacio discreto |
| **Continuo** | Plaid, RePlaid | Proyecta tokens a espacio continuo, aplica difusión gaussiana, luego discretiza |
| **Flujo discreto** | DFM | Aprende un campo de velocidad para transformar una distribución en otra |

[^survey]: Li et al. "A Survey on Diffusion Language Models." arXiv:2508.10875, 2026.

---

## 2. Diferencia frente al entrenamiento autorregresivo

| Aspecto | Autorregresivo (GPT, Qwen, LLaMA) | Difusión (MDLM, LLaDA, Plaid) |
|---------|------------------------------------|--------------------------------|
| **Dirección de generación** | Izquierda → derecha, un token a la vez | Todos los tokens en paralelo, refinando iterativamente |
| **Dependencia** | Cada token depende de los anteriores (máscara causal) | Cada token depende de todos los demás (atención bidireccional) |
| **Objetivo de entrenamiento** | Cross-entropy next-token (máxima verosimilitud) | ELBO / cross-entropy ponderada / score matching |
| **Velocidad de inferencia** | Serial: N tokens → N pasos | Paralelo: N tokens → T pasos (T < N típicamente) |
| **Coherencia a largo plazo** | Depende de la ventana de contexto | La naturaleza bidireccional puede capturar dependencias globales |
| **Madurez de ecosistema** | Extremadamente maduro (HuggingFace, Unsloth, vLLM) | Incipiente; hay herramientas pero falta infraestructura de producción |
| **Fine-tuning** | LoRA, QLoRA, full FT, SFT — bien establecido | SFT funcional pero menos explorado; RL (GRPO adaptado) incipiente |

### La diferencia clave

En un modelo autorregresivo, la probabilidad de la secuencia se descompone como:

```
P(x₁, x₂, ..., xₙ) = P(x₁) · P(x₂|x₁) · P(x₃|x₁,x₂) · ...
```

En un modelo de difusión, la secuencia se modela conjuntamente, y la generación es un proceso iterativo de refinamiento. Esto permite que el modelo "reconsidere" tokens generados previamente (autocorrección), algo que un modelo autorregresivo no puede hacer sin técnicas externas como beam search o refusión.

---

## 3. ⚠️ NO confundir con difusión para series temporales

**Este documento trata sobre modelos de lenguaje por difusión.** NO es lo mismo que los modelos de difusión para series temporales que ya están documentados en `documentacion/modelos/diffusion/`.

| | Difusión para series temporales | Difusión para modelos de lenguaje |
|---|---|---|
| **Dominio** | Valores numéricos continuos (temperatura, precipitación, precios) | Tokens discretos (texto) |
| **Objetivo** | Forecasting probabilístico, generación de trayectorias | Generación de texto coherente |
| **Arquitectura típica** | U-Net, Transformer temporal con atención causal | Transformer con atención bidireccional (DiT) |
| **Método** | Difusión gaussiana sobre series numéricas | Difusión discreta (máscara) o continua sobre embeddings de tokens |
| **Ejemplos** | GluonTS, Lag-Llama, pysteps | LLaDA, MDLM, Plaid, Dream, SEDD |
| **Repositorio en este proyecto** | `documentacion/modelos/diffusion/` | **Este documento** (`documentacion/llm/`) |

Los papers de `documentacion/modelos/diffusion/` (Predict-Refine-Synthetize, Survey on Diffusion for Time Series, pysteps, Lag-Llama, etc.) son sobre forecasting meteorológico y probabilístico — la difusión como herramienta para generar distribuciones de probabilidad sobre series numéricas. Nada que ver con generar texto.

---

## 4. Modelos representativos: MDLM, SEDD, Plaid, LLaDA, Dream

### MDLM (Masked Diffusion Language Model)

- **Autor:** Sahoo et al. (2024)
- **Idea:** Simplifica la difusión discreta a un promedio ponderado de pérdidas de modelado de lenguaje enmascarado (similar a BERT, pero iterativo).
- **Resultado:** Escalable hasta ~65M parámetros con leyes de escala comparables. Gap de compute ~14× respecto a autorregresivo para alcanzar la misma loss.
- **Estado:** Es el baseline discrete DLM más fuerte en la actualidad.

### SEDD (Score Entropy Discrete Diffusion)

- **Autor:** Lou et al. (2023)
- **Idea:** Aprende la función de puntuación (*score*) del espacio discreto usando una pérdida de entropía de puntuación.
- **Resultado:** Perplexity competitiva, pero más complejo de entrenar que MDLM.

### Plaid / RePlaid

- **Autor:** Sahoo et al. (2023), revisado por Yang et al. (2026)
- **Idea:** Difusión continua: proyecta tokens a espacio continuo, aplica difusión gaussiana, luego discretiza.
- **RePlaid (2026):** Revisado para alinearse con la arquitectura de MDLM. Logra PPL de 22.1 en OpenWebText (el mejor entre DLMs continuos), y supera a Duo (discreto) en el régimen over-trained.
- **Gap de compute:** ~20× respecto a autorregresivo (con self-conditioning).
- **Resultado:** Primer scaling law unificado que muestra que difusión continua escala de forma competitiva con discreta.

### LLaDA (Large Language Diffusion Architecture)

- **Autor:** Nie et al. (2025)
- **Modelo:** LLaDA-8B, inicializado desde un transformer preentrenado y ajustado con difusión discreta.
- **Resultado:** Empata o supera a LLaMA-3 8B en MMLU, ARC-C y otros benchmarks de reasoning. Es el DLM más grande y competitivo hasta la fecha.
- **Fine-tuning:** Soporta SFT (máscara sobre la respuesta) y alineamiento por preferencias (VRPO, diffu-GRPO).

### Dream-7B

- **Autor:** Ye et al. (2025)
- **Modelo:** Inicializado desde Qwen2.5 7B y entrenado con 580B tokens de difusión.
- **Resultado:** Supera a todos los DLMs anteriores y empata con modelos AR de primer nivel. Demostración de que un modelo AR existente puede convertirse en DLM.

### Difusión como fine-tuning de un AR existente

- **Cetin et al. (2025)** en ICML proponen un método de fine-tuning que añade la capacidad de difusión a un LLM preentrenado **sin modificar sus pesos originales**. Permite escalar compute en test-time aumentando pasos de difusión, con mejora monótona en accuracy.

---

## 5. Requisitos de hardware y viabilidad en este proyecto

### Lo que hay en este proyecto

| Recurso | Disponibilidad |
|---------|---------------|
| **Local** | CPU sin GPU (driver NVIDIA roto, GTX 1650 con 4 GB sin CUDA) |
| **Colab** | GPU T4 gratuita (16 GB VRAM, sin bf16, solo fp16) |
| **Datos** | ~300 ejemplos de entrenamiento, ~178k tokens |

### Requisitos de memoria para DLMs

Los modelos de difusión de lenguaje que están a la vanguardia son significativamente más pesados en inferencia que los autorregresivos equivalentes, porque:

1. **Múltiples pasos de denoising:** cada paso requiere una pasada completa del Transformer con atención bidireccional (no se puede usar KV cache como en AR).
2. **Atención bidireccional:** el Transformer completo procesa todos los tokens en cada paso, sin la optimización causal.
3. **No hay cuantización nativa:** a diferencia de QLoRA para AR, la comunidad no tiene herramientas maduras de cuantización 4-bit para DLMs.

| Modelo | Parámetros | VRAM inferencia (estimada) | VRAM fine-tuning (estimada) |
|--------|-----------|---------------------------|----------------------------|
| MDLM (OWT) | ~65M | ~1-2 GB | ~4-8 GB |
| LLaDA-8B | 8B | ~16-20 GB (FP16) | ~32-40 GB (sin QLoRA disponible) |
| Dream-7B | 7B | ~14 GB (FP16) | ~28-32 GB |
| LLaDA-1.5 | 8B | ~16-20 GB (FP16) | ~32-40 GB |

**Nota:** Estas son estimaciones conservadoras basadas en la infraestructura DLM actual. A diferencia de LoRA/QLoRA para AR, no hay bitsandbytes ni Unsloth para DLMs. El fine-tuning típico de LLaDA se hace con DeepSpeed ZeRO-2 o FSDP, que requieren multi-GPU o al menos una GPU de 24+ GB.

### Viabilidad en T4 (16 GB)

- **Para MDLM pequeño (~65M):** viable en T4. El problema es que un modelo de 65M parámetros es demasiado pequeño para ser útil como modelo de lenguaje general — tiene capacidad limitada.
- **Para LLaDA-8B o Dream-7B:** **NO viable en T4**. Necesitarían ~32+ GB para fine-tuning (sin QLoRA), y la T4 solo tiene 16 GB. Incluso para inferencia, 8B en FP16 necesita ~16 GB, que es justo el tope de la T4 sin margen para los pasos de difusión.
- **Sin cuantización 4-bit disponible:** QLoRA es lo que hace factible fine-tuning de 7B en T4 para AR. Para DLMs no existe una implementación equivalente (aunque hay trabajo en dirección, no está listo para uso general).

### Tiempo estimado

Los DLMs requieren múltiples pasos de denoising tanto en entrenamiento como en inferencia. Un fine-tuning de LLaDA-8B en un clúster de 8× A100 tarda horas. En una T4, incluso si cupiera en memoria, sería 10-50× más lento (estimación conservadora), lo que lo haría impráctico.

### Veredicto de hardware

**Los modelos de difusión de lenguaje competitivos (8B+) no son viables con el hardware disponible.** Solo los modelos muy pequeños (<100M) caben en T4, pero su capacidad es insuficiente para el caso de uso de este proyecto (generar recomendaciones de riesgo a partir de perfiles de usuario).

---

## 6. Comparación con LoRA/QLoRA (ruta de LLM-006)

### Lo que ya hace LLM-006

| Aspecto | LLM-006 (QLoRA sobre Qwen 2.5) |
|---------|--------------------------------|
| **Modelo** | Qwen 2.5 7B (o 1.7B) |
| **Método** | QLoRA: modelo en 4 bits + LoRA r=16 |
| **VRAM** | ~4-6 GB en T4 |
| **Tiempo** | ~5-15 min en T4 (300 ejemplos, 3 epochs) |
| **Ecosistema** | Unsloth, bitsandbytes, Ollama, vLLM — maduro |
| **Exportación** | GGUF → Ollama (ya funcional) |
| **Calidad** | 99%+ de Guanaco sobre Vicuna benchmark |

### Qué aportaría la difusión (si fuera viable)

| Ventaja teórica de DLMs | Relevancia para ClimaSafeAI |
|--------------------------|----------------------------|
| **Decodificación paralela** (menor latencia) | Mínima: el bot genera ~100-200 tokens por respuesta; la latencia secuencial no es el cuello de botella |
| **Autocorrección** (revisar tokens previos) | Potencialmente útil: el modelo podría corregir recomendaciones si detecta contradicciones, pero esto se logra mejor con RAG y verificación post-generación |
| **Atención bidireccional** (contexto completo) | Irrelevante en generación: el modelo siempre ve todo el prompt; la máscara causal no impide eso |
| **Generación de alta calidad en reasoning** | Los benchmarks muestran ventaja en math/code; para recomendaciones textuales la ventaja es marginal |

### Qué costaría

| Coste | DLM (LLaDA-8B) | QLoRA (Qwen 2.5 7B) |
|-------|----------------|----------------------|
| **Hardware mínimo** | 1× A100 80GB o 4× RTX 4090 | 1× T4 16GB (Colab gratis) |
| **Coste Colab** | ~$1-3/hora (A100) | Gratis (T4) |
| **Tiempo de fine-tuning** | Horas a días | Minutos |
| **Ecosistema** | Incipiente, sin cuantización 4-bit | Maduro, con Unsloth y Ollama |
| **Exportación a producción** | Sin camino claro a Ollama/llama.cpp | GGUF → Ollama (funcional) |
| **Calidad estimada** | 95-99% AR (según benchmarks) | 99%+ AR |
| **Mantenimiento** | Riesgo alto: el campo cambia cada mes | Riesgo bajo: LoRA es estándar |

### ¿Sustituye o convive?

**Conviven, pero en el estado actual la difusión NO sustituye a LoRA/QLoRA:**

- **LoRA/QLoRA es el camino pragmático:** funciona, es barato, tiene ecosistema, produce resultados medibles.
- **La difusión es una investigación prometedora pero no madura para producción:** los modelos más fuertes (LLaDA-8B, Dream-7B) requieren hardware que este proyecto no tiene, y el ecosistema de fine-tuning no está listo para workflows caseros.
- **Si la difusión madura** (cuantización 4-bit para DLMs, herramientas como Unsloth para difusión, modelos más pequeños competitivos), podría considerarse como alternativa futura.

---

## 7. Recomendación

### **Descartar la difusión para fine-tuning en este proyecto, al menos por ahora.**

### Justificación

1. **Hardware insuficiente:** Los DLMs competitivos (8B+) necesitan 32+ GB VRAM para fine-tuning. La T4 de Colab tiene 16 GB y no hay cuantización 4-bit disponible para DLMs.

2. **Ecosistema inmaduro:** No existe Unsloth, bitsandbytes ni Ollama para difusión. El camino de exportación a producción no está claro.

3. **Beneficio marginal para el caso de uso:** La principal ventaja de los DLMs (decodificación paralela con menor latencia) no es relevante para un bot que genera ~200 tokens por respuesta. La autocorrección teórica se logra mejor con RAG y verificación post-generación.

4. **Coste de oportunidad:** Invertir tiempo en investigar y adaptar un paradigma incipiente distrae de perfeccionar lo que ya funciona (QLoRA + Ollama + RAG).

5. **La investigación está activa:** Los DLMs están evolucionando rápidamente (RePlaid, LLaDA-1.5, TRIMS, VRPO...). Lo que hoy requiere un clúster de A100 puede ser viable en T4 en 12-18 meses si aparece cuantización nativa para DLMs.

### Qué monitorsiar

Si se quiere mantener la puerta abierta para el futuro:

- **Cuando Unsloth o bitsandbytes soporten DLMs:** reevaluar. La cuantización 4-bit es el multiplicador que hizo factible LoRA en hardware limitado.
- **Cuando haya un DLM competitivo <3B parámetros:** reevaluar. Un modelo pequeño podría caber en T4.
- **Cuando LLaDA o Dream tengan pipelines de fine-tuning documentados para una GPU:** reevaluar.

### Features propuestas (NO implementar aquí)

Si en el futuro se decide explorar, estas features irían al backlog:

1. **LLM-013:** Benchmark comparativo AR vs DLM en el dataset de ClimaSafeAI (requiere hardware adecuado)
2. **LLM-014:** Adaptación de fine-tuning de LLaDA/Dream para una T4 (requiere cuantización 4-bit para DLMs)
3. **LLM-015:** Evaluación de latencia real de DLM vs AR para generación de recomendaciones (~200 tokens)

---

## 8. Referencias

### Papers principales

1. Sahoo et al. "Simple and Effective Masked Diffusion Language Models." NeurIPS 2024. (MDLM)
2. Lou et al. "Score Entropy Discrete Diffusion." 2023. (SEDD)
3. Sahoo et al. "Plaid: Efficient Scaling of Language Diffusion Transformers." 2023.
4. Yang et al. "Continuous Diffusion Scales Competitively with Discrete Diffusion for Language." arXiv:2605.18530, 2026. (RePlaid)
5. Nie et al. "Masked Diffusion Language Models." 2025. (LLaDA)
6. Ye et al. "Dream-7B." 2025.
7. Cetin et al. "Large Language Models to Diffusion Finetuning." ICML 2025.
8. Li et al. "A Survey on Diffusion Language Models." arXiv:2508.10875, 2026.

### Fine-tuning y alineamiento de DLMs

9. Kim et al. "Fine-Tuning Masked Diffusion for Provable Self-Correction." ICML 2026. (PRISM)
10. Zhao et al. "LLaDA 1.5: Variance-Reduced Preference Optimization." ACL 2026. (VRPO)
11. Tang et al. "TRIMS: Trajectory-Ranked Instruction Masked Supervision." 2026.
12. Deng et al. "Beyond Fully Random Masking: Attention-Guided Denoising and Optimization for Diffusion Language Models." ACL 2026. (AGDO)
13. "PADRE: Pseudo-Likelihood Training for Reasoning Diffusion Language Models." EACL 2026.

### Hardware y fine-tuning general

14. "LLM Fine-Tuning Hardware Requirements 2026." llmhardware.io, 2026.
15. "Local LLM Fine-Tuning Hardware Requirements 2026." Presenc AI, 2026.

### Contexto del proyecto

- `documentacion/llm/lora-alora-multi-adapter.md` — LoRA, aLoRA y servir varios adaptadores
- `documentacion/llm/colab-fine-tuning.md` — Flujo de fine-tuning en Colab con T4
- `documentacion/modelos/diffusion/` — Papers de difusión para SERIES TEMPORALES (otro dominio)
