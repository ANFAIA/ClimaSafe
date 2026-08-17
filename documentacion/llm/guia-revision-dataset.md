# Guía de revisión del dataset sintético

**Versión:** 1.0
**Fecha:** 2026-08-17
**Feature:** LLM-015

---

## Para qué sirve esta guía

El fine-tuneado aprende lo que ve repetido: un par con la clase equivocada, un
input sin el parte meteorológico o una respuesta que inventa una cifra se
convierte en una alucinación del modelo en producción. Antes de empaquetar el
zip de Colab hay que revisar los pares de `data/llm/train.jsonl` y
`data/llm/val.jsonl`.

Esta guía dice qué comprobar en cada par y cómo decidir si se queda, se corrige
o se descarta. El control se puede hacer a mano sobre una muestra, o con el
script de QC que automatiza la mayoría de las comprobaciones:

```bash
uv run python climasafeai/llm/revisar_dataset.py \
    --train data/llm/train.jsonl --val data/llm/val.jsonl \
    --out /tmp/informe_qc.json --muestra 50
```

El script señala los pares sospechosos; esta guía explica el **porqué** de cada
criterio y qué hacer con cada hallazgo.

Contexto de dónde salen los datos: `climasafeai/llm/generar_dataset.py` (pares
de calor) y `climasafeai/llm/generar_dataset_frio.py` (pares de frío desde el
parquet). El papel del LLM en el producto está en `docs_site/llm.md`.

---

## Anatomía de un par

Cada línea del JSONL es un par Alpaca:

- **`instruction`**: siempre la misma ("Predice el riesgo térmico para este
  perfil y da recomendaciones.").
- **`input`**: el perfil (edad, sexo, grasa, aclimatado, comorbilidades,
  medicación, actividad, duración, hora, ubicación...) + el parte meteorológico
  de la ventana de actividad (**"Tiempo en esa franja"**).
- **`output`**: la respuesta ideal, generada por el pipeline real
  (`predict_ensemble`): clase (`RIESGO: X`), índices, factores y
  recomendaciones.

La regla de oro: **todo lo que dice el output tiene que salir del pipeline. Si
no puede salir de `predict_ensemble`, es una alucinación.**

---

## 1. Campos obligatorios del input

El input tiene que permitir reproducir la respuesta. Comprueba que:

- **"Tiempo en esa franja" está presente.** Sin el parte, el mismo perfil tiene
  respuestas distintas según el día de generación y el modelo no puede
  aprender nada (solo memoriza). Es la marca que verifica
  `empaquetar_dataset_colab.py`.
- **El parte lleva los 5 campos**: temperatura **media**, temperatura **máxima**
  (siempre, aunque coincida con la media), **humedad**, **viento** y **UV**.
  Un parte sin máx o sin UV es un input con menos información de la que el
  modelo verá en producción.

Si falta alguno → el par **se descarta o se regenera** (el generador ya los
descarta; un par así en el dataset es señal de que se generó con una versión
vieja).

## 2. Coherencia de la clase con el pipeline

El output empieza con `RIESGO: X` (SEGURO / PRECAUCIÓN / PELIGRO). Esa clase
tiene que ser la que daría `predict_ensemble` con ese perfil y ese parte.

Para comprobarlo: ejecuta el pipeline sobre el par (el QC lo hace con una
muestra, con weather sintético reproducible y sin red) y compara:

| Respuesta afirma | Pipeline da | Diagnóstico |
|---|---|---|
| SEGURO | PELIGRO | **Crítico**: el modelo aprenderá a decir SEGURO con 39+ °C. Corregir o descartar. |
| PRECAUCIÓN | PELIGRO | Revisar: si el parte dice máx ≥ 39 °C, la respuesta debería ser PELIGRO (override físico). |
| PELIGRO | SEGURO | Revisar: puede ser una respuesta de un día distinto al del parte. |
| Coinciden | — | OK. |

Reglas físicas del pipeline que no se negocian (override del ensemble):

- **HI ≥ 39 °C** (índice de calor de la ventana) → **PELIGRO** siempre.
- **HI ≥ 27 °C con perfil vulnerable** (edad ≥ 60 sin entrenar y aclimatado, o
  comorbilidades, o fármacos, o no aclimatado, o factor > 1.8, o capado) y
  (HI ≥ 32 o UV > 3) → **PRECAUCIÓN**.
- **Wind chill ≤ -25 °C** → **PELIGRO** (riesgo de congelación).
- Sin calor real (HI < 27 y WC > 0 y UV < 6) la clase personalizada baja a
  PRECAUCIÓN como mucho.

Un par cuya clase contradice estas reglas se corrige o se descarta, nunca se
deja.

> Nota sobre el QC: la re-ejecución del pipeline reconstruye el weather desde
> el parte del input (features de persistencia y secuencia LSTM aproximadas),
> así que una discrepancia de **un nivel** (SEGURO↔PRECAUCIÓN o
> PRECAUCIÓN↔PELIGRO) puede ser margen de la reconstrucción y hay que mirar el
> par a mano. Una discrepancia de **dos niveles** es un hallazgo sólido.

## 3. Alucinación: cifras que no salen del pipeline

Una alucinación en este dataset es cualquier **cifra** (índice, factor, clase,
recomendación) que no pueda salir de `predict_ensemble` con ese input:

- Índice personalizado / poblacional que no coincide con el rango de la clase
  que afirma (p. ej. `RIESGO: PELIGRO` con índice 0.05).
- Un factor de riesgo con un nombre o coeficiente que no está en
  `data/factores_riesgo.json` (p. ej. "cardiopatía ×2.5" cuando el coeficiente
  real es ×1.4).
- Recomendaciones inventadas ("ve al hospital") que el pipeline no genera.
- Una clase que contradice los factores listados (factores que apuntan a riesgo
  alto y clase SEGURO).

El generador escribe el output directamente desde el resultado del pipeline,
así que un par generado por `generar_dataset.py` no alucina por construcción.
Los pares editados a mano o de versiones antiguas sí pueden hacerlo. Si lo
ves → **descarta** (mejor regenerar el par que parchearlo a mano).

## 4. Perfiles imposibles o contradictorios

El input describe a una persona; lo que describe tiene que ser posible y el
pipeline tiene que poder leerlo:

| Campo | Rango / valores válidos |
|---|---|
| Edad | 0-120 |
| Grasa corporal | 3-70 % (fuera de ese rango el factor de personalización no tiene sentido) |
| Sexo | `hombre`, `mujer` |
| Comorbilidades | claves de `factores_riesgo.json` (`cardiovascular`, `diabetes`, `mental`, `respiratoria`) |
| Medicación | claves de `factores_riesgo.json` (`antipsicoticos`, `diureticos_asa`) |
| Situación social | `vive_solo`, `sin_aire_acondicionado`, `no_sale`, `encamado`, `vivienda_fria`, `alcohol` |
| Ocupación | `reparto`, `construccion`, `campo` (definidas en `personalizacion.py`) |
| Fototipo | `II`, `III`, `IV` |

Dos errores típicos de perfiles **contradictorios** (no los detecta el QC
porque cada campo por separado es válido):

- "Aclimatado: sí" con "Entrenado: no" y 2 semanas en Sevilla en julio — la
  aclimatación no aparece de la nada.
- Edad 25 con comorbilidades graves y sin medicación — posible, pero revisa
  que los factores del output lo reflejen.
- "Situación social: no_sale" con "Actividad: intensa" en la calle — no sale
  de casa pero hace actividad intensa fuera.

Un perfil imposible o contradictorio → **corregir el perfil** (no la
respuesta) o descartar el par.

## 5. La respuesta no contradice los factores del perfil

El output tiene que ser coherente con el input que lo genera:

- Si el input dice "Comorbilidades: cardiovascular" y el output no activa el
  factor cardiovascular → hueco de aprendizaje.
- Si el input NO menciona nada (perfil limpio) y el output lista "obesidad /
  grasa alta" → el perfil y la respuesta no se corresponden.
- Si el input dice "Aclimatado: no" en un día de 40 °C y la clase es SEGURO →
  incoherente (ver reglas físicas de la sección 2).

## 6. Duplicados y desequilibrio (nivel dataset)

Estos no se ven par a par, pero el QC los reporta en el agregado:

- **Duplicados casi idénticos**: dos inputs con el mismo perfil y el parte casi
  igual no aportan información nueva y enseñan al modelo a repetirse. El QC
  normaliza (minúsculas, espacios, números) y marca pares con similitud de
  Jaccard > 0.9. Si un perfil se repite mucho, busca variar hora/duración/
  localización en la regeneración.
- **Desequilibrio de clases**: si una clase (SEGURO/PRECAUCIÓN/PELIGRO) cae por
  debajo del 10 % del conjunto, el modelo aprende a no decirla nunca. El
  generador equilibra por cupo; si el QC lo avisa, regenera con más escenarios
  de esa clase (fríos → SEGURO/PRECAUCIÓN).

## Cómo decidir: se queda, se corrige o se descarta

| Hallazgo | Decisión |
|---|---|
| Todo coherente, campos completos, sin alucinaciones | **Se queda** |
| Falta un campo del parte / la clase no cuadra por un error claro del par | **Se corrige** regenerando el par (nunca a mano) o editando el perfil |
| Alucinación (cifra inventada), perfil imposible, clase que contradice la física (SEGURO con HI ≥ 39) | **Se descarta** |
| Duplicado casi idéntico | **Se descarta uno** (o se varía el perfil al regenerar) |

Un par corregido a mano es una fuente nueva de alucinaciones: si hay que
corregir algo, regenera con el generador y vuelve a pasar el QC.

---

## Flujo recomendado antes de empaquetar

```bash
# 1. QC sobre el dataset actual
uv run python climasafeai/llm/revisar_dataset.py --train data/llm/train.jsonl \
    --val data/llm/val.jsonl --out /tmp/informe_qc.json --muestra 50

# 2. Revisar a mano los hallazgos del informe (críticos primero)

# 3. Regenerar si hace falta (el generador ya aplica las correcciones del QC)
uv run python climasafeai/llm/generar_dataset.py --output data/llm/train.jsonl \
    --num-ejemplos 300

# 4. Volver a pasar el QC hasta que los hallazgos críticos sean cero

# 5. Empaquetar (verifica 300/100 y la marca en todos los inputs)
uv run python climasafeai/llm/empaquetar_dataset_colab.py
```
