#!/usr/bin/env bash
# Lanza el fine-tuning en el entorno de Unsloth, no en el .venv del proyecto.
#
# Unsloth pide Python 3.10-3.11 y su propio stack de CUDA, así que vive en un
# entorno aparte (micromamba). Meterlo en el .venv arriesga el entorno del resto del
# proyecto, que ya se rompió una vez.
#
# Uso:
#   ./climasafeai/llm/entrenar.sh --check              ¿se puede entrenar ya?
#   ./climasafeai/llm/entrenar.sh                      entrena con qwen2.5-1.5b
#   ./climasafeai/llm/entrenar.sh --model qwen2.5-7b   ...o con el 7B, si hay VRAM
#
# Cualquier argumento se pasa tal cual a fine_tune.py.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
ENTORNO="${UNSLOTH_ENV:-unsloth}"
MICROMAMBA="${MICROMAMBA_BIN:-$HOME/.local/bin/micromamba}"

export MAMBA_ROOT_PREFIX="${MAMBA_ROOT_PREFIX:-$HOME/.micromamba}"
# /tmp son 2,7 GB en esta máquina y pip revienta a mitad de instalar torch.
export TMPDIR="${TMPDIR:-$HOME/.cache/tmp-pip}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-$HOME/.cache/pip}"
mkdir -p "$TMPDIR" "$PIP_CACHE_DIR"

if [ ! -x "$MICROMAMBA" ]; then
  echo "No encuentro micromamba en $MICROMAMBA."
  echo "Instálalo sin root con:"
  echo "  curl -sL https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -xj bin/micromamba"
  echo "  install -m755 bin/micromamba ~/.local/bin/micromamba"
  exit 1
fi

if ! "$MICROMAMBA" env list | grep -qE "^\s*${ENTORNO}\s"; then
  echo "No existe el entorno '${ENTORNO}'. Créalo con:"
  echo "  micromamba create -y -n ${ENTORNO} -c conda-forge python=3.11 pip"
  echo "  micromamba run -n ${ENTORNO} pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121"
  echo "  micromamba run -n ${ENTORNO} pip install 'unsloth[cu121-torch251]'"
  exit 1
fi

cd "$ROOT"
# Por defecto el 1.5B: es lo que cabe en 4 GB de VRAM. El 7B pide 8-10 GB.
if [[ ! " $* " =~ " --model " ]]; then
  set -- --model qwen2.5-1.5b "$@"
fi

echo "▶  Entrenando en el entorno '${ENTORNO}'  ·  $*"
exec "$MICROMAMBA" run -n "$ENTORNO" python climasafeai/llm/fine_tune.py "$@"
