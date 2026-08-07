# Recursos externos: adaptadores, modelos de 1 bit y cuantización

**Fecha:** 2026-08-06 · **Origen:** enlaces compartidos en la charla/reunión del 06/08/2026
(los pasó Ismael Faro) · **Features:** LLM-007, LLM-008, LLM-009, LLM-010, LLM-011

> **Son referencias a evaluar, no decisiones tomadas.** Todo lo que hay aquí viene de
> terceros: blogs de producto, colecciones de Hugging Face y un paper. Ninguno se ha
> probado en este proyecto. Lo que dice un vendedor de su propio modelo se anota como
> *afirmación del fabricante*, y no cuenta como evidencia hasta que alguien lo mida con
> el banco de pruebas de LLM-003. Los enlaces quedan aquí para no volver a buscarlos.

---

## Índice

| Recurso | Qué aporta | Ticket que lo usa |
|---------|------------|-------------------|
| IBM Granite Libraries | Adaptadores LoRA/aLoRA ya publicados para RAG, guardarraíles y comprobaciones | LLM-008, LLM-007 |
| Project Granite Switch | Componer y activar varios adaptadores sobre el mismo modelo base | LLM-007 |
| BitNet b1.58 (arXiv 2402.17764) | El paper de referencia de los pesos ternarios | LLM-009, LLM-010 |
| Bonsai (prism-ml) | Modelos entrenados nativamente en 1 bit, con GGUF | LLM-009 |
| Crítica de Bonsai (r/LocalLLaMA) | Contrapunto: alguien dice que no rinde lo que promete | LLM-009 |
| ResABit (Ismael Faro) | Resultado **negativo** medido sobre cuantización de 1 bit | LLM-009, LLM-010, LLM-011 |

---

## IBM Granite Libraries

- Blog: <https://research.ibm.com/blog/granite-libraries-project-switch>
- Colección de adaptadores: <https://huggingface.co/collections/ibm-granite/granite-libraries>

Tres librerías de adaptadores **LoRA y aLoRA** (activated LoRA), Apache 2.0, para los
modelos **Granite 4.1**:

| Librería | Tareas que cubre |
|----------|------------------|
| **RAG Library** | Reescritura de la consulta, evaluación de si la pregunta es respondible, detección de alucinaciones, generación de citas |
| **Core Library** | Comprobación de requisitos, puntuación de certeza, atribución contextual |
| **Guardian Library** | Comprobaciones de seguridad, factualidad y política *en línea*, sin un modelo de guardarraíl aparte |

**Qué ticket lo usa:** **LLM-008** (evaluar el aLoRA de Granite para RAG con
guardarraíles). Esto confirma que LLM-008 apunta a algo real y publicado, no a un rumor
de la reunión: los adaptadores existen, tienen licencia usable y sus tareas coinciden
con lo que el proyecto ya hace a mano.

**Lo que NO confirma, y por eso LLM-008 sigue siendo una evaluación y no una adopción:**

1. Exigen **Granite 4.1 como modelo base**. Hoy el proyecto va con Qwen 2.5 y el
   fine-tuning de LLM-006 es sobre Qwen. Adoptar los adaptadores significa cambiar de
   familia de modelo, no añadir una pieza.
2. Nadie ha medido si su detección de alucinaciones y su evaluación de *answerability*
   mejoran lo que ya hacen el filtro de relevancia del chat (CHAT-003) y el RAG
   (RAG-001, RAG-002). Ese es justo el criterio de LLM-008: compararlos contra el set de
   evaluación de RAG-004, no contra la impresión de que "los de IBM lo tendrán mejor".
3. Están pensados para flujos de empresa. Que sirvan para un parte de riesgo térmico en
   español hay que verlo.

Nota lateral: la **Guardian Library** hace comprobaciones de política dentro del modelo.
Es lo contrario del enfoque de **ARNES-011** y **MCP-002**, donde la frontera de permisos
vive en código determinista precisamente para no depender de la buena voluntad del
modelo. No se sustituyen: un guardarraíl dentro del modelo puede fallar o ser rodeado por
un *prompt*; un gateway en Python, no. Si alguien propone la Guardian Library como
sustituto del gateway, esta es la razón por la que no lo es.

## Project Granite Switch

Mismo blog. Es un *toolkit* experimental que actúa de capa de conmutación: gestiona
adaptadores de forma dinámica y los activa en inferencia, componiéndolos sobre el modelo
base sin modificarlo.

**Qué ticket lo usa:** **LLM-007**, que pregunta literalmente si se pueden servir varios
adaptadores a la vez sobre el mismo modelo base ("agrupar loras" en las notas de la
reunión). Granite Switch es una respuesta concreta a esa pregunta —y una pista de que la
respuesta general no es trivial, o IBM no habría necesitado un toolkit para ello. El
documento de LLM-007 debe decir si algo equivalente existe para los runtimes que este
proyecto usa de verdad (Ollama, llama.cpp) o si es exclusivo del ecosistema Granite.

## BitNet b1.58 — el paper de los ternarios

- PDF: <https://arxiv.org/pdf/2402.17764> · Abstract: <https://arxiv.org/abs/2402.17764>

**Verificado el 06/08/2026:** el enlace es *The Era of 1-bit LLMs: All Large Language
Models are in 1.58 Bits* (Ma, Wang, Ma, Wang, Wang, Huang, Dong, Wang, Xue, Wei;
27/02/2024). Introduce **BitNet b1.58**, donde cada peso es ternario **{-1, 0, 1}**, y
afirma rendimiento comparable al de un Transformer en precisión completa con menos
latencia, memoria, energía y más *throughput*. El "1.58" sale de log₂(3): tres valores
posibles por peso.

**Qué ticket lo usa:** **LLM-009** (PoC de modelos nativos de 1 bit o ternarios) como
referencia teórica de por qué esto debería funcionar, y **LLM-010** (cuantizar el modelo
propio) para no confundir dos cosas distintas: BitNet se **entrena** en ternario desde el
principio; cuantizar a posteriori un modelo entrenado en FP16 es otro problema y no
hereda estos resultados.

## Bonsai — modelos entrenados nativamente en 1 bit

- Colección: <https://huggingface.co/collections/prism-ml/bonsai>
- Fork de llama.cpp: <https://github.com/PrismML-Eng/llama.cpp>

Modelos de prism-ml en 1 bit, con pesos +1/-1, publicados en GGUF y MLX en tamaños 1.7B,
4B y 8B. Hay kernels de 1 bit en forks de MLX, mlx-swift y llama.cpp (CUDA y Metal).

**Afirmaciones del fabricante, sin verificar por nadie de este proyecto:** Apache 2.0,
entrenado nativamente a 1 bit de punta a punta (embeddings, atención, MLP y cabeza),
"10,8× más densidad de inteligencia" que Qwen 3 8B en precisión completa, y capaz de
correr en hardware de consumo. La cifra de densidad es material de marketing y no
significa nada operativo hasta que se traduzca a las métricas de LLM-003.

**Qué ticket lo usa:** **LLM-009**. Es el candidato más obvio para el criterio de "al
menos un modelo nativo de 1 bit o ternario", y el GGUF encaja con el stack actual
(Ollama/llama.cpp) — con la salvedad importante de que hace falta **su fork** de
llama.cpp para los kernels de 1 bit, no el llama.cpp de siempre. Eso es coste de
integración que hay que contar antes de empezar.

### Contrapunto: la crítica

- Hilo: <https://www.reddit.com/r/LocalLLaMA/comments/1snvv64/bonsai_models_are_pure_hype_bonsai8b_is_much/>
  — *"Bonsai models are pure hype: Bonsai-8B is MUCH dumber than Gemma-4-E2B"*

**No he podido abrir el hilo desde aquí** (Reddit bloquea la petición), así que el título
y la tesis quedan tal cual los pasó el humano, sin verificar el contenido ni los
contraargumentos de los comentarios. Lo que sí está en las notas de producto del propio
fabricante y apunta en la misma dirección: **58,1 en HumanEval+ frente a 80,1 de Qwen 3
8B**, 22 puntos de diferencia, además de problemas con salida estructurada anidada y más
alucinación de detalles factuales. (Ese dato viene de un resumen promocional de terceros
—AlphaSignal—, tampoco de una medición nuestra.)

**Por qué está anotado aquí y no descartado:** este contrapunto es exactamente lo que
convierte LLM-009 en una prueba de concepto con números en vez de en una semana quemada.
Un modelo puede ser mucho peor programando y aun así valer para lo que este proyecto le
pide —redactar un parte de riesgo a partir de un resultado ya calculado por el
pipeline—, porque el cálculo nunca lo hace el LLM. Pero "más alucinación de detalles
factuales" sí es una alarma directa contra el criterio de LLM-003 de *no introducir
cifras que no estén en el resultado del pipeline*. Las dos hipótesis se resuelven
midiendo, y el orden importa: si Bonsai suspende ese criterio, se descarta y no se gasta
más tiempo en la integración del fork.

## ResABit

- Repositorio: <https://github.com/ismaelfaro/ResABit> — de Ismael Faro, quien compartió
  los enlaces.

Revisado leyendo el repositorio (sin clonar ni ejecutar nada). Es un estudio experimental
controlado, Apache 2.0, en Python: pregunta si unos residuos de atención entre capas
pueden reparar el daño que causa cuantizar los pesos a 1 bit. Ablación 2×2 sobre
Qwen1.5-0.5B-Chat, 15 ejecuciones con cinco semillas emparejadas, doble backend
(MLX y PyTorch) y resultados congelados en un fichero de registro.

**El resultado es negativo, y por eso vale:** la interacción medida entre las dos técnicas
es estadísticamente indistinguible de cero — el residuo no repara preferentemente el daño
de la binarización.

**Qué ticket lo usa:** **LLM-009** y **LLM-010** como aviso de que la cuantización
agresiva hace daño y que los arreglos elegantes no siempre lo reparan, y **LLM-011**
(pruning, marcado "ni fu ni fa") como modelo metodológico: semillas emparejadas,
resultados congelados y la disposición a publicar que el efecto es cero. Es lo contrario
de la anécdota que LLM-003 quiere desterrar.

---

## Qué hacer con esto

Nada de aquí cambia el trabajo en curso. **LLM-006** (fine-tuning de Qwen en Colab) sigue
siendo la tarea abierta y estos enlaces no la afectan. El orden razonable sigue siendo el
que ya marca el backlog:

1. **LLM-007** responde por escrito qué es un aLoRA y si se pueden agrupar adaptadores
   (Granite Switch entra aquí).
2. **LLM-008** evalúa entonces los adaptadores de Granite con números, sabiendo ya lo que
   son y lo que cuesta cambiar de modelo base.
3. **LLM-009** mide Bonsai y compañía con el banco de pruebas de LLM-003 — con la crítica
   de r/LocalLLaMA delante, no después.
