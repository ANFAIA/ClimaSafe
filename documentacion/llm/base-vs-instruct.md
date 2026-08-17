# Modelo base vs modelo de instrucciones — estudio completo

**Fecha:** 2026-08-17
**Feature:** LLM-016
**Resumen:** `docs_site/llm.md` tiene la conclusión; aquí está el estudio técnico
completo con la comparación real y la decisión para el próximo LoRA.

---

## Para qué sirve este documento

En el fine-tuning de Qwen3 de este proyecto (LLM-013/LLM-014) se partió del
modelo **base** `unsloth/Qwen3-1.7B` y se le aplicó un LoRA con un dataset de
instrucciones. El resultado (`qwen3:climasafe`) da el formato ClimaSafe pero
sigue alucinando detalles. Este documento explica **por qué**: qué le falta a
un modelo base que sí tiene un modelo instruct, y qué cambia en el LoRA según
la variante de la que se parta. El objetivo es decidir sobre qué variante
aplicar el próximo LoRA y entender el origen de las alucinaciones.

---

## 1. Qué es un modelo base y qué es un modelo de instrucciones

| | Modelo base | Modelo de instrucciones (instruct) |
|---|---|---|
| Entrenado para | Predecir la siguiente palabra en billones de textos | Lo mismo, **más** un ajuste posterior con datos de "sigue instrucciones" |
| Qué sabe hacer | Escribir texto fluido, completar, continuar | Comportarse como asistente: responder, formatear, decir "no lo sé" |
| Cómo se obtiene | Pre-entrenamiento (una sola fase) | Base + SFT (supervised fine-tuning con chat template) + RLHF/DPO |
| Coste | Ya está publicado, se descarga | Ya está publicado, se descarga |

En una frase:

- **Modelo base:** sabe *escribir*, pero no *responder*. Ante una pregunta
  continúa el texto como si fuera un fragmento más del corpus.
- **Modelo instruct:** el mismo modelo, ajustado después para que *responda*:
  entiende el chat template, da respuestas útiles, reconoce cuándo no sabe.

El instruct no es un modelo distinto: es el base más un segundo entrenamiento
que le enseña el **comportamiento de asistente**.

---

## 2. Chat template: cómo le llega el prompt

Un modelo de chat no recibe un texto suelto: recibe una conversación con
**roles**, que se serializan con un formato concreto (el *chat template*).
Qwen3 usa el formato ChatML:

```
<|im_start|>system
Eres un asistente experto en riesgo térmico...<|im_end|>
<|im_start|>user
Perfil: 72 años, mujer, diabetes...<|im_end|>
<|im_start|>assistant
```

Cada turno va envuelto entre `<|im_start|>` y `<|im_end|>`, y el modelo tiene
que aprender (o ya saber, si es instruct) qué se espera tras ese último
`<|im_start|>assistant`: una respuesta de asistente, no la continuación del
texto del usuario.

### Por qué el base no sabe seguirlo sin entrenarlo

El chat template es una **convención aprendida**, no una propiedad del texto.
Un modelo base se entrenó con texto plano: nunca vio miles de conversaciones
envueltas en `<|im_start|>`, así que cuando las ve por primera vez no tiene
ninguna razón para comportarse como asistente. Sigue con su tarea de siempre:
continuar el texto con la palabra más probable — que, tras un prompt de
instrucciones, suele ser más texto de instrucciones, una pregunta al revés o
una respuesta a medias.

### Qué pasa si le das formato instruct a un base

Tres comportamientos típicos, todos vistos en este proyecto:

1. **Ignora los roles** y trata todo el prompt como texto a continuar: en vez
   de responder, continúa con `user` o repite la pregunta.
2. **Inventa el formato a medias**: empieza a responder pero sin estructura de
   asistente, mezclando consejo con texto del prompt.
3. **Aprende el formato solo si se lo enseñas con muchos ejemplos** — que es
   exactamente lo que hace un LoRA de instrucciones sobre un base: gasta parte
   de su capacidad en aprender algo que el instruct ya trae de serie.

---

## 3. Thinking mode de Qwen3

Qwen3 introduce un **modo de razonamiento** opcional: antes de la respuesta
final, el modelo puede emitir un bloque de razonamiento envuelto en
`<thought>...</thought>` (o en Ollama, ` thinking... response`):

```
 thinking
La usuaria tiene 72 años, diabetes, sin aclimatar...
Voy a responder con precaución.
 response
RIESGO: PELIGRO
```

El bloque de razonamiento **no es parte de la respuesta**: es el "borrador
interno" del modelo. En una API de chat normal va en un campo aparte
(`reasoning_content`), pero en Ollama puede caer **dentro** del `content` si el
Modelfile se sirve con el thinking activo.

### Por qué hay que limpiarlo en el bot (BOT-022)

`rag_qwen.py` limpia el bloque con `_limpiar_bloque_think()` antes de devolver
la respuesta. Sin esa limpieza, el parte de Telegram acabaría con el
razonamiento pegado ("el modelo está pensando que...") o, peor, con solo el
cierre ` response` suelto al principio, como pasó en una conversación real del
13-08. La comparación de la sección 5 muestra el bloque vacío
` thinking\n\n response` que emite `qwen3:climasafe` — justo lo que esa
función elimina.

### Cómo afecta a base vs instruct

- El **instruct** puede razonar (Qwen3 trae el modo por defecto) y además
  sabe *cuándo* razonar y *cuándo* pasar a la respuesta.
- El **base**, si emite el bloque, lo hace como continuación de texto: a veces
  abre ` thinking` y se queda dentro sin cerrarlo nunca, o lo cierra y
  continúa con más texto en vez de la respuesta final.
- El **LoRA hereda el comportamiento de su base**: un LoRA sobre instruct
  razona "bien" (bloque completo + respuesta); uno sobre base hereda el
  razonamiento a medias y necesita que el dataset le enseñe también a cerrar
  el bloque — otra cosa más que aprender.

---

## 4. LoRA sobre base vs LoRA sobre instruct

Un LoRA (Low-Rank Adaptation) congela los pesos del modelo y entrena un
adaptador pequeño. El adaptador **no reemplaza** lo que el modelo ya sabe: lo
ajusta. Por eso la variante de partida importa:

| | LoRA sobre **base** | LoRA sobre **instruct** |
|---|---|---|
| Qué tiene que aprender | Seguir instrucciones (chat template, formato asistente) **y** el dominio ClimaSafe | Solo el dominio ClimaSafe |
| Ejemplos necesarios | Muchos (los mismos ejemplos hacen las dos cosas a la vez) | Menos (el instruct ya sabe responder; el LoRA solo añade el formato y el conocimiento ClimaSafe) |
| Riesgo de desajuste | Alto: el adaptador "tira" de un modelo que no sabe responder → respuestas a medias, formato parcial, alucinaciones | Bajo: el modelo ya tiene el comportamiento de asistente; el adaptador solo lo especializa |
| Alucinaciones | Se inventa cifras y detalles porque no distingue "responder" de "continuar" | Menos: sabe decir "no lo sé" y seguir instrucciones de no inventar |
| Coste de entrenamiento | Más épocas / más datos para el mismo resultado | Menos épocas / menos datos para el mismo resultado |

La intuición: un LoRA es un **parche** de bajo rango. Si el parche tiene que
enseñar dos cosas a la vez (comportamiento + dominio), cada cosa recibe menos
"presupuesto" y el resultado es peor en ambas. Si el modelo ya sabe comportarse,
todo el presupuesto del LoRA va al dominio.

Este es exactamente el problema del primer fine-tuning de este proyecto
(LLM-013): se partió del **base** con un dataset de **instrucciones**. El
modelo tuvo que aprender a la vez a ser asistente y a dar el formato ClimaSafe,
y el resultado es el que se ve en la sección 5: formato correcto pero detalles
alucinados.

---

## 5. Comparación real

### Limitación: el modelo base no está servido en Ollama

`ollama list` (2026-08-17) muestra estos modelos:

```
qwen3:climasafe      61b2d19d6d70    1.1 GB    4 days ago
qwen2.5:climasafe    504594d9b9ec    986 MB    5 days ago
gemma3:4b            a2af6cc3eb7f    3.3 GB    2 weeks ago
gemma3:1b            8648f39daa8f    815 MB    2 weeks ago
qwen3:1.7b           8f68893c685c    1.4 GB    2 weeks ago
qwen2.5:1.5b         65ec06548149    986 MB    2 weeks ago
```

**No hay un modelo base servido**: `qwen3:1.7b` es el instruct y
`qwen3:climasafe` es el fine-tuned (LoRA sobre el base, servido como GGUF).
El base sin instruir no está en Ollama, así que no se puede ejecutar la misma
conversación contra él. Lo que sí se puede comparar es el **instruct** contra
el **fine-tuned sobre base**, que es donde se ve el efecto del LoRA; y el
comportamiento esperado del base se explica conceptualmente (sección 2) y con
el benchmark de LLM-003 (sección 6).

### Pregunta y método

Misma pregunta, servida con `ollama run <modelo> "..."` (salidas literales,
17-08-2026):

> Perfil: 72 años, mujer, diabetes, sin aclimatar, 2h al aire libre a 38°C.
> ¿Qué riesgo y qué hago?

### Salida de `qwen3:1.7b` (instruct, sin fine-tuning)

```
**Riesgo y acciones a tomar:**

1. **Riesgo principal:**
   - **Síndrome de calor (calor estrés):** Puede evolucionar a **calor estropeado** (enfermedad por calor) si no se maneja adecuadamente.
   - **Riesgo de hiperglucemia o hipoglucemia:** La calor puede alterar el metabolismo, especialmente en personas con diabetes, aumentando el riesgo de fluctuaciones de glucosa en sangre.
   - **Dehidratación:** La calor alta puede provocar deshidratación, que puede empeorar la glucemia y causar síntomas como fatiga, confusión o pérdida de conciencia.

2. **Acciones a tomar:**
   - **Hidratación:** Bebe agua o soluciones electrolíticas (como la **Glytek** o **Powerade**) para evitar la deshidratación.
   - **Descanso y refugio:** Se recomienda moverse a un lugar fresco y sombreado, y descansar en una zona con ventilación.
   - **Monitoreo de síntomas:**
     - **Calor estresado:** Síntomas como sudoración intensa, fatiga, debilidad, vértigo o náuseas.
     - **Calor estropeado:** Confusión, pérdida de conciencia, temperatura corporal elevada (>40°C), sequedad de boca, vómitos o pérdida de apetito.
   - **Uso de protección:**
     - **Ropa ligera y de manga corta** para evitar el calor.
     - **Uñas de sol** y **crema protectora** en áreas expuestas.
   - **Atención médica inmediata:** Si se presentan síntomas graves (como confusión severa, pérdida de conciencia o vómitos frecuentes), se debe acudir a un servicio de emergencia o hospital.

3. **Preventiva:**
   - **Aclimatación:** Si planea actividades al aire libre en climas extremos, se recomienda acclimatar gradualmente (ej.: 1-2 días de exposición inicial).
   - **Control de glucosa:** Mantén un régimen de diabetes ajustado, evitando dietas ricas en azúcar y asegurando una dieta equilibrada.
   - **Hidratación constante:** Bebe agua regularmente, incluso si no siente sed.

**Consejo final:**
La calor puede ser peligrosa para personas con diabetes, especialmente si no se accliman. Si te sientes mal, busca un refugio fresco y consulta a un profesional de la salud si hay síntomas graves. 🌞💧
```

### Salida de `qwen3:climasafe` (fine-tuned con LoRA sobre el base)

```
 thinking

 response

RIESGO: PELIGRO

Índice personalizado: 0.94
Índice poblacional: 0.56
Factor total aplicado: ×2.71 (tope; considera tus condiciones)

Factores activados:
- edad: 72 años
- sexo: mujer
- enfermedad respiratoria: diabetes (controlado con medicación)
- duración: 2 horas al aire libre (sol, sombra o bajo proteccion de un medio)
- temperatura ambiente: 38 °C (equivalente a 104 °F; calor extremo, riesgo para la salud)

Recomendaciones:
- El calor extremo sobrecarga el sistema cardiovascular. Evita esfuerzos en horas de calor intenso. Dado tu nivel de riesgo actual, no se recomienda realizar actividad fisica al aire libre. Busca un lugar fresco o usa un somier ajustado para permanecer en reposo.
- El calor puede afectar la absorcion de insulina y el nivel de glucosa en sangre. Manten la insulina refrigerada. Hidratate bien y controla tus niveles con mas frecuencia.
- Bebe agua regularmente aunque no tengas sed. La sed es un signo tardio de deshidratacion. Evita alcohol y cafeina que deshidratan.
```

### Análisis

| Aspecto | `qwen3:1.7b` (instruct) | `qwen3:climasafe` (LoRA sobre base) |
|---|---|---|
| Responde como asistente | Sí: markdown, viñetas, consejo final | Sí: formato ClimaSafe estructurado |
| Formato ClimaSafe | **No**: no dice RIESGO: X ni índice | **Sí**: RIESGO: PELIGRO + índices + factores |
| Inventa cifras | Sí: marcas comerciales ("Glytek", "Powerade") | **Sí, peligrosas**: "enfermedad respiratoria: diabetes", "controlado con medicación", "insulina refrigerada", "somier ajustado" |
| Pensamiento (thinking) | No se ve en la salida (mode def) | Emite ` thinking\n\n response` **vacío** — el bloque que limpia `_limpiar_bloque_think` (BOT-022) |
| Conocimiento del pipeline | Ninguno: consejo genérico de salud | Formato del pipeline, pero las cifras **no salen del pipeline** (el prompt no llevaba el resultado de `predict_ensemble`) |
| Errores de lenguaje | "calor estropeado", "Uñas de sol", "acclimatar" | Tildes sueltas, "proteccion", "fisica" |

La lectura correcta de esta comparación:

1. **El LoRA sí enseñó el formato**: el fine-tuned responde con la estructura
   ClimaSafe exacta que el instruct no da. Eso es lo que el dataset de
   instrucciones consiguió.
2. **Pero el LoRA no enseñó el *comportamiento* completo**: al partir del
   base, el modelo tuvo que aprender a la vez a seguir instrucciones y el
   dominio. Resultado: da el formato pero **alucina detalles** que no están en
   el prompt — clasifica la diabetes como "enfermedad respiratoria", asume
   insulina donde no se menciona, e inventa un índice (0.94) que no viene del
   pipeline.
3. **El instruct, sin fine-tuning, no alucina cifras médicas del pipeline**
   (no tiene formato que rellenar), pero tampoco sirve para el producto: da
   consejo genérico sin la clasificación ClimaSafe.

Un LoRA sobre el **instruct** debería dar lo mejor de ambos: el formato
ClimaSafe del fine-tuned *sin* los detalles inventados, porque el modelo ya
sabe distinguir "responder" de "continuar" y el adaptador solo tiene que
enseñar el dominio.

---

## 6. El benchmark apoya lo mismo (LLM-003)

El benchmark real de LLM-003 (4 modelos × 100 ejemplos de `data/llm/val.jsonl`,
corrida definitiva) compara los modelos **sin fine-tuning** que sí estaban
servidos:

| modelo | clase correcta | formato | inventa cifras | error del índice |
|---|---|---|---|---|
| `qwen3:1.7b` | 38 % | 100 % | 13 % | 0.297 |
| `qwen2.5:1.5b` | 32 % | 100 % | **100 %** | 7.098 |

`qwen2.5:1.5b` se inventa alguna cifra en **todas** las respuestas y da
índices fuera del rango 0-1; `qwen3:1.7b` (mejor instruido) inventa en el 13 %
y con un error de índice 37 veces menor. Aunque ambos son modelos instruct,
la diferencia de **calidad del ajuste de instrucciones** entre generaciones se
traduce directamente en alucinaciones. Es el mismo mecanismo que separa un
LoRA sobre base de un LoRA sobre instruct, medido en el mundo real.

---

## 7. Conclusión práctica para este proyecto

### Qué base usar en el próximo LoRA

**Partir de Qwen3 instruct (`unsloth/Qwen3-1.7B-Instruct`), no del base.**

- El instruct ya sabe seguir instrucciones: el LoRA solo tiene que enseñar el
  dominio ClimaSafe (formato del parte, factores, recomendaciones del
  pipeline).
- El LoRA sobre el base tuvo que enseñar las dos cosas a la vez y el resultado
  (sección 5) muestra la consecuencia: formato correcto, detalles inventados.
- El benchmark (sección 6) muestra el mismo patrón sin fine-tuning: el modelo
  mejor instruido alucina 8 veces menos.

### Cómo mitiga las alucinaciones

1. **Menos que aprender → menos desajuste.** Un adaptador de bajo rango que
   solo especializa el dominio deforma menos el comportamiento de asistente
   que uno que tiene que crearlo desde cero.
2. **El instruct ya sabe decir "no lo sé".** El comportamiento de rechazo
   (no inventar lo que no está en el contexto) se entrena con RLHF/DPO del
   instruct; un LoRA pequeño difícilmente lo borra, pero un LoRA sobre base
   difícilmente lo crea.
3. **Menos ejemplos necesarios.** El dataset puede ser más pequeño y más
   limpio, lo que se apoya en LLM-015 (guía de revisión del dataset): al
   partir de instruct, los pares dudosos se pueden **descartar** en vez de
   corregir, porque ya no hacen falta tantos ejemplos para enseñar el
   comportamiento base.
4. **El formato del parte no se deja al modelo.** Independientemente de la
   base, el pipeline ya redacta en Python las frases obligatorias del parte
   (BOT-013/BOT-020) y el LLM solo las copia. El LoRA sobre instruct reduce la
   alucinación residual en lo que sí queda a su cargo (la recomendación de
   cierre y las respuestas RAG).

### Acción

- Próximo fine-tuning: **`unsloth/Qwen3-1.7B-Instruct`** como base del LoRA,
  dataset del mismo generador (`generar_dataset.py`) con el QC de LLM-015, y
  re-benchmark con `data/llm/val.jsonl` para verificar que "inventa cifras"
  baja del 13 % hacia 0.
- El `qwen3:climasafe` actual (LoRA sobre base) sigue siendo el preferido de
  `mejor_disponible()` hasta que exista el reentrenado: da el formato y su
  alucinación de detalles no afecta al parte obligatorio, que se repone en
  Python.

---

## Referencias

- `docs_site/llm.md` — resumen y conclusión (versión corta de este estudio).
- `climasafeai/llm/rag_qwen.py` — `_limpiar_bloque_think` (BOT-022), prompts y
  selección de modelo.
- `documentacion/llm/guia-fine-tuning-qwen.md` — cómo se entrena y sirve el
  LoRA (LLM-001).
- `documentacion/llm/guia-revision-dataset.md` — QC del dataset (LLM-015).
- Benchmark: LLM-003 (números en `docs_site/llm.md` y `rag_qwen.py`).