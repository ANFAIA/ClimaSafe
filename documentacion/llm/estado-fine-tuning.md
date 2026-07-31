# Estado del fine‑tuning (LLM‑002)

Qué está listo, qué falta y cómo lanzarlo. La guía del proceso está en
[`guia-fine-tuning-qwen.md`](guia-fine-tuning-qwen.md); esto es el parte de situación,
comprobado el 30 de julio de 2026 en la máquina de desarrollo.

## Resumen

El entrenamiento **sigue sin ejecutarse**. Queda **un solo bloqueo**: el driver de
NVIDIA, que necesita root y un reinicio. Todo lo demás —dataset, script y entorno de
Unsloth con sus 6,3 GB de dependencias— está montado y verificado.

```
$ ./climasafeai/llm/entrenar.sh --check
▶  Entrenando en el entorno 'unsloth'  ·  --model qwen2.5-1.5b --check
No se puede entrenar todavía:
  1. CUDA no disponible para torch (no hay driver NVIDIA cargado: falta
     /proc/driver/nvidia). Unsloth y la cuantización en 4 bits NO funcionan en CPU:
     sin GPU no hay entrenamiento, ni lento ni rápido.
```

Antes de esta tanda el mismo `--check` daba **dos** problemas; el de los paquetes ya
está resuelto.

## El único bloqueo: el driver

La GPU está físicamente presente pero la lleva **nouveau**, el driver libre, que no
soporta CUDA:

```
$ lspci -k -s 01:00.0
01:00.0 VGA compatible controller: NVIDIA Corporation TU117 [GeForce GTX 1650]
	Kernel driver in use: nouveau
	Kernel modules: nouveau

$ dpkg -l | grep -i nvidia
ii  firmware-nvidia-graphics  20250410-2      ← solo el firmware, no el driver
```

Y el driver propietario no se puede instalar tal cual, porque **los repos de trixie
solo tienen `main non-free-firmware`**:

```
$ grep -h ^deb /etc/apt/sources.list
deb http://deb.debian.org/debian/ trixie main non-free-firmware
deb http://deb.debian.org/debian/ bookworm-backports main contrib non-free

$ apt-cache policy nvidia-driver
  Instalados: (ninguno)
  Candidato:  535.216.03-2~bpo12+1
     100 http://deb.debian.org/debian bookworm-backports/non-free amd64 Packages
```

El único candidato viene de **bookworm‑backports**, o sea de Debian 12 sobre un
sistema Debian 13. Instalar eso puede funcionar, pero no es la forma correcta: falta
añadir `contrib non-free` a los repos de trixie.

Lo bueno: **Secure Boot está desactivado**, así que no hace falta firmar el módulo.

### Lo que hay que ejecutar (requiere root y reiniciar)

```bash
# 1. Añadir contrib y non-free a los repos de trixie
sudo sed -i 's|trixie main non-free-firmware|trixie main contrib non-free non-free-firmware|' \
    /etc/apt/sources.list
sudo apt update

# 2. Instalar el driver (arrastra los headers y compila el módulo con DKMS)
sudo apt install nvidia-driver firmware-misc-nonfree

# 3. Reiniciar: nouveau se pone en la lista negra y se carga nvidia
sudo reboot

# 4. Comprobar
nvidia-smi                       # tiene que listar la GTX 1650
./climasafeai/llm/entrenar.sh --check
```

No lo he ejecutado: son acciones de sistema con root, y un reinicio.

## VRAM: 4 GB manda sobre el modelo

La GTX 1650 tiene **4 GB**. La guía pide 8–10 GB para el 7B en QLoRA, así que en esta
máquina el 7B **no cabe ni con el driver puesto**. Por eso `entrenar.sh` usa
`qwen2.5-1.5b` por defecto. Para el 7B hace falta otra máquina, o Colab/Kaggle.

`--check` distingue los dos casos: pide ~8 GB para los 7B y ~4 GB para los 1.5B, y lo
dice con el nombre de la GPU que encuentre.

## Lo que sí está montado y verificado

### El entorno de Unsloth

Creado sin root con micromamba, en `~/.micromamba/envs/unsloth` (6,3 GB):

```
$ micromamba run -n unsloth pip list | grep -iE "unsloth|peft|trl|datasets|bitsandbytes|accelerate|transformers|torch|xformers"
accelerate               1.14.0
bitsandbytes             0.50.0
datasets                 4.3.0
peft                     0.20.0
torch                    2.5.1+cu121
transformers             5.5.0
trl                      0.24.0
unsloth                  2026.7.6
xformers                 0.0.29.post1
```

`import unsloth` llega hasta el final y falla en lo único que falta:

```
NotImplementedError: Unsloth currently only works on NVIDIA, AMD and Intel GPUs.
```

Cómo se creó, por si hay que rehacerlo:

```bash
# micromamba, sin root, un solo binario
curl -sL https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -xj bin/micromamba
install -m755 bin/micromamba ~/.local/bin/micromamba

export MAMBA_ROOT_PREFIX="$HOME/.micromamba"
micromamba create -y -n unsloth -c conda-forge python=3.11 pip
micromamba run -n unsloth pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
micromamba run -n unsloth pip install "unsloth[cu121-torch251]"
```

**Ojo con `/tmp`:** son 2,7 GB en esta máquina y pip revienta a mitad de instalar
torch con `OSError: [Errno 28] No space left on device`. Hay que apuntar `TMPDIR` a
`/home`, que tiene 700 GB libres. `entrenar.sh` ya lo hace.

### El dataset

```
$ wc -l data/llm/*.jsonl
  135 data/llm/train.jsonl
   15 data/llm/val.jsonl
```

Carga y se formatea a mensajes de chat sin errores. Longitudes medidas con el
tokenizer de Qwen2.5 sobre los 150 ejemplos:

```
tokens  min / mediana / max :  252 / 412 / 562
p95                         :  509
ejemplos que pasan de 4096  :  0
tokens totales              :  58.534
```

Por eso `DEFAULT_MAX_SEQ_LEN` bajó de **4096 a 1024**: el ejemplo más largo son 562
tokens, así que 4096 solo reservaba VRAM de más — y la VRAM es el recurso escaso.

**Importante:** este dataset solo sirve desde el 30/07/2026. Las versiones anteriores
las generaba un `_predecir_fake` que calculaba el riesgo únicamente con la edad — 150
ejemplos con 5 valores distintos de índice y etiquetas que ni eran las del sistema.
Entrenar con una copia vieja es tirar la GPU a la basura. Ante la duda, regenerar:

```bash
uv run python climasafeai/llm/generar_dataset.py -n 150 --val-split 0.1
```

## Plan de ejecución, cuando haya driver

```bash
# 0. Confirmar que el bloqueo ha caído
nvidia-smi
./climasafeai/llm/entrenar.sh --check            # tiene que decir "Entorno listo"

# 1. Entrenar (1.5B: es lo que cabe en 4 GB)
./climasafeai/llm/entrenar.sh --epochs 3
#    salida esperada: parámetros entrenables, loss por epoch y
#    el adaptador en models/llm/qwen-climasafe-lora

# 2. Evaluar contra el val-set
./climasafeai/llm/entrenar.sh --eval-only \
    --lora-path models/llm/qwen-climasafe-lora \
    --val-file data/llm/val.jsonl

# 3. Exportar a GGUF
./climasafeai/llm/entrenar.sh --export-gguf \
    --lora-path models/llm/qwen-climasafe-lora \
    --gguf-path models/llm/qwen-climasafe-q4_k_m.gguf

# 4. Servir en Ollama
ollama create climasafe -f climasafeai/llm/Modelfile
ollama list                                       # tiene que aparecer `climasafe`

# 5. Comparar con el raw, que es lo que pide el criterio 3 de LLM-002
#    misma conversación con `climasafe` y con `qwen2.5:7b`, y pegar las dos salidas
```

Con 58.534 tokens y 3 epochs son ~175.000 tokens de entrenamiento: en una GPU con
driver, minutos.

## Qué NO demuestra este documento

Los criterios de LLM‑002 piden la salida real del entrenamiento (epochs, loss, ruta
del adaptador), el modelo servido en Ollama y la comparación contra el raw. **Nada de
eso existe todavía** y este parte no lo sustituye. La feature no está hecha.
