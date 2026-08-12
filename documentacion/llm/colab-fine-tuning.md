# Fine‑tuning de Qwen en Google Colab (GPU T4 gratuita)

**Versión:** 1.0
**Fecha:** 2026-08-06
**Feature:** LLM-006

---

## Resumen del flujo

Esta máquina no puede entrenar: el bloqueo de la feature LLM-002 sigue vigente
(driver NVIDIA roto, sin `/proc/driver/nvidia`). El fine‑tuning se hace en
Google Colab con una GPU T4 gratuita, en tres pasos:

1. **En local:** empaquetar el dataset con verificación de versión
   (`empaquetar_dataset_colab.py`) → `data/llm/colab_dataset.zip`.
2. **En Colab:** subir el zip y `fine_tune.py` al notebook
   (`notebooks/llm-fine-tuning-colab.ipynb`) y ejecutar las celdas en orden.
   Los artefactos quedan en `MyDrive/climasafeai/`.
3. **En local:** bajar LoRA y GGUF a `models/llm/` y servirlos con Ollama.

---

## 1. Empaquetar el dataset en local

```bash
uv run python climasafeai/llm/empaquetar_dataset_colab.py
```

Antes de empaquetar verifica que el dataset es la **versión buena**, no la fake:

- `train.jsonl` con **300** líneas y `val.jsonl` con **100**.
- El campo **"Tiempo en esa franja"** presente en **todos** los inputs
  (la versión vieja del `_predecir_fake` no lo tiene).
- Si no coincide, sale con error y **no** genera el zip — así nunca subes la
  versión fake a Colab.

Si el zip ya existe, se regenera. Los `.bak` de `data/llm/` no se empaquetan.

Salida esperada (los sha256 cambian si el dataset cambia):

```text
Empaquetado: data/llm/colab_dataset.zip (431 KB)
  train.jsonl: 12208ccfb8ba506350a154e1fb4b3915103a93276b893bfd43e1a59e06221cb2  (300 líneas)
  val.jsonl: cda323eeb2870ee5090372cd8e0276bce22744e5a38ff210384877acf18feb36  (100 líneas)
Guarda estos sha256: la Celda 4 del notebook los compara.
```

Anota el sha256: la Celda 4 del notebook imprime el de la copia descomprimida
en Colab y es con lo que verificas que es la misma versión.

---

## 2. Abrir el notebook en Colab y activar la T4

1. Ve a <https://colab.research.google.com>.
2. **Archivo → Subir notebook → Subir** y selecciona
   `notebooks/llm-fine-tuning-colab.ipynb`.
3. **Entorno de ejecución → Cambiar tipo de entorno de ejecución → T4 GPU →
   Guardar**.
4. Ejecuta las celdas en orden (`Entorno de ejecución → Ejecutar todas`, o una
   a una con Shift+Enter).

> La Celda 1 aborta con un mensaje claro si el runtime no tiene GPU: sin T4,
> Unsloth y el QLoRA no funcionan.

---

## 3. Subir el zip al notebook y comprobación de versión

La **Celda 4** busca el zip primero en `MyDrive/climasafeai/colab_dataset.zip`
y, si no está, muestra un botón para subirlo desde el navegador:

```text
Sube el fichero colab_dataset.zip (el del empaquetado local):
```

Elige `data/llm/colab_dataset.zip` de tu máquina. La celda:

1. Lo descomprime en `/content/data/` (quedan `train.jsonl` y `val.jsonl`).
2. Verifica la versión: **300** líneas en `train`, **100** en `val` y
   `"Tiempo en esa franja"` en **todos** los inputs. Si algo no coincide,
   imprime exactamente qué está mal y aborta.
3. Imprime el **sha256** de cada fichero descomprimido: compáralo con el del
   paso 1. Si coinciden, es la misma versión.

> Si prefieres no subir el zip por el navegador cada vez: súbelo una vez a
> `MyDrive/climasafeai/colab_dataset.zip` y la celda lo cogerá de ahí.

---

## 4. Qué hace cada celda

| Celda | Qué hace |
|-------|----------|
| 1 | Comprueba que el runtime tiene GPU (`nvidia-smi` / `torch.cuda`) y aborta con mensaje claro si no |
| 2 | Desinstala el stack viejo y instala `unsloth[colab-new]` — la receta oficial de Unsloth para Colab. NO la de `entrenar.sh`: forzar `torch==2.5.1` rompe la importación de unsloth (el torchao de Colab 2026 usa `torch.int1`, que solo existe desde torch 2.6). También desinstala `timm` y `fastai`: son preinstalados de Colab que dependen de `torchvision` y no se usan en este flujo (cero menciones en `climasafeai/`), así que se quitan para que no arrastren una versión vieja de torch |
| 3 | Monta Google Drive (`/content/drive`) para persistir los artefactos |
| 4 | Sube/descomprime `colab_dataset.zip` y verifica la versión (300/100 + marca + sha256) |
| 5 | Copia `fine_tune.py` a `/content/` (desde Drive o subiéndolo) y entrena QLoRA con los hiperparámetros de `fine_tune.py`: rank 16, seq 1024, batch 2 × accum 4, 3 épocas, lr 2e-4 |
| 6 | Exporta el LoRA a GGUF **q4_k_m** en Drive (el paso que más VRAM pide: correlo en su propia celda) |
| 7 | Lista los artefactos persistidos en `MyDrive/climasafeai/` |

Los hiperparámetros son los de `climasafeai/llm/fine_tune.py` (rank 16, max-seq
1024, batch 2, accum 4, epochs 3, lr 2e-4). Con 300 ejemplos (~178k tokens) en
la T4 el entrenamiento es del orden de minutos.

---

## 5. Dónde quedan los artefactos en Drive

Todo se guarda en `MyDrive/climasafeai/`:

| Artefacto | Ruta en Drive | Tamaño aprox. |
|-----------|---------------|---------------|
| Adaptador LoRA | `MyDrive/climasafeai/qwen-climasafe-lora/` | decenas de MB |
| Modelo cuantizado | `MyDrive/climasafeai/qwen-climasafe-q4_k_m.gguf` | ~0.9–1.0 GB |

El LoRA se escribe directamente en Drive durante el entrenamiento (Celda 5) y
el GGUF durante la exportación (Celda 6). No hay que copiar nada a mano.

---

## 6. Bajar artefactos y servir con Ollama

### 6.1 Bajar LoRA y GGUF

En Colab, panel izquierdo → icono de archivos → `drive/MyDrive/climasafeai/`,
clic derecho sobre cada fichero → **Descargar**. O en local:

```bash
mkdir -p models/llm
# desde Drive, descarga a:
#   models/llm/qwen-climasafe-lora/          (el LoRA, la carpeta entera)
#   models/llm/qwen-climasafe-q4_k_m.gguf    (el GGUF)
```

### 6.2 Crear el modelo en Ollama

El `FROM` del `climasafeai/llm/Modelfile` es **relativo** al propio Modelfile
(`FROM ./qwen-climasafe-q4_k_m.gguf`): apunta a
`climasafeai/llm/qwen-climasafe-q4_k_m.gguf`. Como el GGUF descargado vive en
`models/llm/`, lo más limpio es copiar el Modelfile junto al GGUF — el `FROM`
relativo entonces resuelve a `models/llm/qwen-climasafe-q4_k_m.gguf`:

```bash
cp climasafeai/llm/Modelfile models/llm/Modelfile
ollama create climasafe -f models/llm/Modelfile
```

Alternativa si prefieres mantener el comando de `estado-fine-tuning.md`:

```bash
cp models/llm/qwen-climasafe-q4_k_m.gguf climasafeai/llm/
ollama create climasafe -f climasafeai/llm/Modelfile
```

O editar el `FROM` del Modelfile a la ruta absoluta del GGUF descargado
(el propio Modelfile lo documenta: `FROM /path/to/qwen-climasafe.gguf`).

---

## 7. Verificación final

```bash
ollama list | grep climasafe
#    climasafe        ...     qwen-climasafe-q4_k_m.gguf ...
```

Esto demuestra el criterio 3 de LLM-002: la misma conversación del bot con el
modelo fine‑tuneado y con `qwen2.5:7b` raw, pegando las dos salidas:

```bash
ollama run climasafe "Perfil: edad 72, mujer, diabetes, sin aclimatar, 2h al aire libre a 38°C. ¿Qué riesgo y qué hago?"
ollama run qwen2.5:7b "Perfil: edad 72, mujer, diabetes, sin aclimatar, 2h al aire libre a 38°C. ¿Qué riesgo y qué hago?"
```

El fine‑tuneado debe dar el riesgo con los factores y recomendaciones del
pipeline real; el raw responde genérico. El bot (`check_ollama()` /
`best_available_model()`) ya prioriza `climasafe` automáticamente, sin
configuración adicional.

---

## Notas

- El notebook NO depende de ningún path de esta máquina: solo usa `/content`
  y `/content/drive`. Los únicos ficheros que hay que subir a mano son
  `colab_dataset.zip` y `fine_tune.py` (o dejarlos en `MyDrive/climasafeai/`).
- T4 no soporta bf16, así que el entrenamiento usará fp16
  (`is_bfloat16_supported()=False` en `fine_tune.py`): es normal, no un error.
- Si algo falla en Colab, mira el mensaje de la celda: la 1 avisa de GPU, la 4
  de dataset, la 5/6 de VRAM. Los pasos son los mismos que documenta
  `guia-fine-tuning-qwen.md`, pero con la T4 en vez de la GPU local.
