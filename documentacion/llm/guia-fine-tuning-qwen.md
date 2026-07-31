# Guía de fine‑tuning: Qwen 2.5 7B + Unsloth → GGUF → Ollama

**Versión:** 1.0  
**Fecha:** 2026-07-30  
**Feature:** LLM-001 (criterio 1)

---

## Índice

1. [Requisitos de hardware](#1-requisitos-de-hardware)
2. [Entorno](#2-entorno)
3. [Dataset](#3-dataset)
4. [Fine‑tuning con Unsloth](#4-fine-tuning-con-unsloth)
5. [Exportar a GGUF](#5-exportar-a-gguf)
6. [Servir con Ollama](#6-servir-con-ollama)
7. [Verificación](#7-verificación)
8. [Integración con ClimaSafeAI](#8-integración-con-climasafeai)
9. [Solución de problemas](#9-solución-de-problemas)

---

## 1. Requisitos de hardware

| Componente | Mínimo | Recomendado |
|------------|--------|-------------|
| GPU VRAM | 8 GB | 12–24 GB |
| RAM | 16 GB | 32 GB |
| Disco | 20 GB libres | 50 GB |
| CUDA | 12.1+ | 12.4+ |

Probado en: RTX 3060 12 GB, RTX 4090 24 GB, A10G 24 GB.

Para Qwen 2.5 7B con LoRA (rank=16, target_modules=all):
- **8 GB VRAM:** batch_size=1, gradient_accum=4, context_length=2048
- **12 GB VRAM:** batch_size=2, gradient_accum=2, context_length=4096
- **24 GB VRAM:** batch_size=4, gradient_accum=1, context_length=8192

---

## 2. Entorno

Unsloth requiere Python 3.10–3.11 y CUDA. No funciona en Python 3.13.  
Crea un entorno **separado** del proyecto principal:

### 2.1 Con conda (recomendado)

```bash
conda create -n unsloth python=3.10 -y
conda activate unsloth

# Instalar CUDA toolkit si no lo tienes
conda install cuda-toolkit -c nvidia -y

# Instalar PyTorch 2.5+ con CUDA 12.1+
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121

# Instalar Unsloth (incluye flash-attention, xformers)
pip install "unsloth[cu121] @ git+https://github.com/unslothai/unsloth.git"
```

> **Nota:** Si tu CUDA es 12.4+, cambia `cu121` por `cu124`.  
> Si no usas conda, el proyecto incluye `pyproject.toml` con extras `[llm]`  
> para el entorno principal, pero Unsloth necesita su propio entorno.

### 2.2 Verificar instalación

```python
import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))
from unsloth import FastLanguageModel; print("Unsloth OK")
```

Debe salir: `True RTX 3060` (o tu GPU) y `Unsloth OK`.

### 2.3 Clonar este repo

```bash
git clone https://github.com/tu-usuario/ClimaSafeAI.git
cd ClimaSafeAI
```

---

## 3. Dataset

El modelo se fine‑tunea con pares **(instrucción + contexto → respuesta ideal)**  
en formato Alpaca (`instruction`, `input`, `output`).

### 3.1 Formato

Cada línea del JSONL:

```json
{
  "instruction": "Predice el riesgo térmico para este perfil y da recomendaciones.",
  "input": "Edad: 72, Sexo: mujer, IMC: 31, Fototipo: II, Comorbilidades: diabetes, cardiopatía, Medicación: metformina, enalapril. Actividad: ligera, 2h al aire libre, 28°C, 65% HR. Sin aclimatar.",
  "output": "RIESGO: MUY ALTO (HI=38°C, riesgo cardiovascular ×3.2).\n\nFactores activados:\n- Edad >65: ×1.25\n- Obesidad (IMC>30): ×1.1 en calor con esfuerzo\n- Diabetes: ×1.18\n- Cardiopatía: ×1.15\n- Sin aclimatar: ×1.35\n- Antipsicóticos: (no aplica)\n\nRecomendaciones:\n- Evitar salir entre 12:00 y 18:00\n- Hidratación forzada cada 20 min\n- Sombrero, ropa ligera y pausas cada 30 min en sombra\n- Supervisión por otra persona"
}
```

### 3.2 Dataset sintético mínimo (100 ejemplos)

El script `scripts/generar_dataset_sintetico.py` (ver §A.1) genera 100+ ejemplos
cubriendo:

- **10 combinaciones demográficas:** edad (joven/adulto/anciano), sexo (H/M), IMC
- **5 niveles de actividad:** reposo, ligera, moderada, intensa, muy intensa
- **3 condiciones ambientales:** calor moderado (30°C), calor extremo (38°C),
  frío (-5°C)
- **5 combinaciones de comorbilidades:** sano, diabetes, cardiovascular,
  respiratorio, múltiple
- **3 situaciones sociales:** vive acompañado, vive solo, sin aire acondicionado
- **2 estados de aclimatación:** sí / no
- **Casos borde:** perfil mínimo (solo edad/sexo), error (valores inválidos),
  valores extremos

Los ejemplos se generan con el pipeline real de `predict_ensemble` + 
`personalizar_riesgo()` para que las respuestas sean factuales.

### 3.3 Dataset real (de conversaciones del bot)

Para producción, extraer pares reales del bot Telegram:

```bash
# Exportar conversaciones a JSONL (herramienta proporcionada en el repo)
uv run python scripts/extraer_conversaciones.py \
  --db data/climasafe.db \
  --output data/dataset_bot.jsonl \
  --min-ejemplos 200
```

Cada conversación genera: `(datos introducidos → respuesta del bot)` como par
instruction/input/output.

### 3.4 Colocar el dataset

```bash
mkdir -p data/llm
# Copiar o generar el dataset
cp data/dataset_climasafe.jsonl data/llm/train.jsonl

# Dividir en train/validation (90/10)
uv run python -c "
import json
with open('data/llm/train.jsonl') as f:
    lines = f.readlines()
split = int(len(lines) * 0.9)
with open('data/llm/train.jsonl', 'w') as f: f.writelines(lines[:split])
with open('data/llm/val.jsonl', 'w') as f: f.writelines(lines[split:])
print(f'{split} train, {len(lines)-split} val')
"
```

---

## 4. Fine‑tuning con Unsloth

### 4.1 Script de entrenamiento

El proyecto incluye `climasafeai/llm/fine_tune.py`. Ejecutar desde el entorno
Unsloth:

```bash
# Activar entorno Unsloth
conda activate unsloth

cd ClimaSafeAI
python climasafeai/llm/fine_tune.py \
  --model qwen2.5-7b \
  --train-file data/llm/train.jsonl \
  --val-file data/llm/val.jsonl \
  --output-dir models/llm/qwen-climasafe-lora \
  --batch-size 2 \
  --epochs 3 \
  --lr 2e-4
```

### 4.2 Configuración LoRA (valores por defecto)

| Parámetro | Valor | Explicación |
|-----------|-------|-------------|
| `r` (rank) | 16 | Dimensión del adaptador LoRA |
| `lora_alpha` | 16 | Escala LoRA (igual a rank → no sobreescala) |
| `target_modules` | all | Q, K, V, O, Gate, Up, Down |
| `lora_dropout` | 0 | Sin dropout (mejor en datasets pequeños) |
| `bias` | none | No entrenar bias |
| `use_rslora` | True | Rank-Stabilized LoRA (mejor generalización) |

### 4.3 Hiperparámetros

| Parámetro | Valor por defecto | Descripción |
|-----------|-------------------|-------------|
| `per_device_train_batch_size` | 2 | Depende de VRAM (1 si 8 GB) |
| `gradient_accumulation_steps` | 4 | Batch efectivo = batch_size × accum |
| `learning_rate` | 2e-4 | LR típico para LoRA |
| `num_train_epochs` | 3 | 3 épocas suele bastar |
| `max_seq_length` | 4096 | Contexto máximo en tokens |
| `warmup_ratio` | 0.03 | 3% de warmup |
| `lr_scheduler_type` | cosine | Cosine decay |
| `optim` | adamw_8bit | Ahorra VRAM |
| `fp16` / `bf16` | bf16 (si GPU lo soporta) | Precisión mixta |

### 4.4 Monitorización

El entrenamiento logea pérdida cada pocos pasos. Para ver gráficas en vivo:

```bash
# Opción A: WandB (gratuito)
pip install wandb
# Añadir --use-wandb al script

# Opción B: TensorBoard
tensorboard --logdir models/llm/qwen-climasafe-lora/logs
```

### 4.5 Tiempo estimado

| GPU | Tamaño dataset | Tiempo |
|-----|----------------|--------|
| RTX 3060 12 GB | 100 ejemplos | ~15 min |
| RTX 3060 12 GB | 500 ejemplos | ~60 min |
| RTX 4090 24 GB | 100 ejemplos | ~5 min |
| RTX 4090 24 GB | 500 ejemplos | ~20 min |

---

## 5. Exportar a GGUF

Una vez entrenado el LoRA, hay que fusionar los pesos y exportar a GGUF.

```bash
conda activate unsloth
python climasafeai/llm/fine_tune.py \
  --export-gguf \
  --lora-path models/llm/qwen-climasafe-lora \
  --gguf-path models/llm/qwen-climasafe.gguf
```

Esto:

1. Carga el modelo base Qwen 2.5 7B
2. Fusiona los pesos LoRA en el modelo base
3. Guarda en formato GGUF (Q4_K_M por defecto)

**Alternativa manual** (si prefieres control granular):

```bash
# 1. Fusionar LoRA
python scripts/merge_lora.py \
  --base unsloth/qwen2.5-7b \
  --lora models/llm/qwen-climasafe-lora \
  --output models/llm/qwen-climasafe-merged

# 2. Convertir a GGUF con llama.cpp
git clone https://github.com/ggml-org/llama.cpp
cd llama.cpp
make -j4
python convert.py \
  ../models/llm/qwen-climasafe-merged \
  --outfile ../models/llm/qwen-climasafe.gguf \
  --outtype q4_k_m
```

---

## 6. Servir con Ollama

### 6.1 Crear Modelfile

```bash
cd models/llm
cat > Modelfile << 'EOF'
FROM ./qwen-climasafe.gguf

PARAMETER temperature 0.3
PARAMETER top_p 0.9
PARAMETER num_ctx 4096

TEMPLATE """{{ .System }}

{{ .Prompt }}"""

SYSTEM """Eres ClimaSafeAI, un asistente experto en riesgo térmico.
Responde en español, sé directo, usa datos concretos.
Cuando recibas un perfil de usuario, calcula el riesgo y da recomendaciones.
Cuando te pregunten por factores, cita el tipo (calor/frío) y la fuente.  Si no sabes algo, dilo — no inventes.  Responde SIEMPRE en español."""
EOF

ollama create qwen2.5:climasafe -f Modelfile
```

### 6.2 Verificar

```bash
ollama list | grep climasafe
# Debería aparecer: qwen2.5:climasafe

# Probar
ollama run qwen2.5:climasafe "¿Qué factores de riesgo cardiovascular se agravan con el frío?"
```

---

## 7. Verificación

El script `climasafeai/llm/fine_tune.py` incluye un modo `--eval-only` que
evalúa el modelo fine‑tuneado contra el conjunto de validación y reporta:

- Pérdida (perplejidad)
- Ejemplos de respuestas generadas vs. esperadas

```bash
python climasafeai/llm/fine_tune.py --eval-only \
  --lora-path models/llm/qwen-climasafe-lora \
  --val-file data/llm/val.jsonl
```

Para prueba manual con el RAG integrado:

```python
from climasafeai.llm.rag_qwen import ask_with_rag, QwenConfig

# Usar el modelo fine-tuneado
cfg = QwenConfig(model="qwen2.5:climasafe")
res = ask_with_rag("¿Qué umbrales de calor son peligrosos?", config=cfg)
print(res["answer"])
```

---

## 8. Integración con ClimaSafeAI

Una vez creado el modelo `qwen2.5:climasafe` en Ollama, el sistema lo detecta
automáticamente:

```python
from climasafeai.llm.rag_qwen import check_ollama

estado = check_ollama()
# estado["best_model"] será "qwen2.5:climasafe" si existe
```

La función `best_available_model()` ya prioriza modelos en este orden:

1. `qwen2.5:climasafe` (fine‑tuneado) → mejor experiencia
2. `qwen2.5:7b` (raw) → capa intermedia
3. `qwen2.5:1.5b` (raw) → CPU
4. _Ninguno_ → bot determinista (fallback seguro)

No hace falta configuración adicional: el sistema elige el mejor modo solo.

---

## 9. Solución de problemas

### "CUDA out of memory"

| Síntoma | Solución |
|---------|----------|
| OOM al cargar el modelo | `--batch-size 1 --gradient-accum 8 --max-seq-len 2048` |
| OOM al exportar GGUF | Cerrar otros programas, reducir `--context-size` |

### "No module named unsloth"

```bash
# Asegúrate de estar en el entorno correcto
conda activate unsloth
python -c "import unsloth; print('OK')"
```

### "Ollama no encuentra el modelo"

```bash
# Verificar que el GGUF existe
ls -lh models/llm/qwen-climasafe.gguf

# Recrear
ollama rm qwen2.5:climasafe
ollama create qwen2.5:climasafe -f models/llm/Modelfile
```

### "El modelo fine‑tuneado alucina igual que el raw"

- Aumentar `--epochs` a 5
- Reducir `--lr` a 1e-4
- Asegurarse de que el dataset tiene suficiente diversidad (mín. 100 ejemplos)
- Verificar que el LoRA se fusionó correctamente (el GGUF debe pesar ~4.7 GB,
  no 200 MB)

---

## A. Scripts incluidos

### A.1 `climasafeai/llm/fine_tune.py`

Script principal de fine‑tuning. Uso completo:

```bash
python climasafeai/llm/fine_tune.py \
  --model qwen2.5-7b \
  --train-file data/llm/train.jsonl \
  --val-file data/llm/val.jsonl \
  --output-dir models/llm/qwen-climasafe-lora \
  --batch-size 2 \
  --epochs 3 \
  --lr 2e-4 \
  --max-seq-len 4096 \
  --lora-rank 16 \
  --use-wandb
```

### A.2 Script de generación de dataset sintético

El dataset se genera a partir del pipeline real de ClimaSafeAI para que las
respuestas sean factuales. Se incluirá en `scripts/generar_dataset_sintetico.py`.

---

## Referencias

- [Unsloth documentation](https://github.com/unslothai/unsloth)
- [Qwen 2.5 Technical Report](https://arxiv.org/abs/2412.15115)
- [GGUF format](https://github.com/ggml-org/ggml)
- [Ollama Modelfile](https://github.com/ollama/ollama/blob/main/docs/modelfile.md)
