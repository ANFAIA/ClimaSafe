#! /usr/bin/env python
"""
ClimaSafeAI — Empaqueta data/llm para subir a Google Colab (fine‑tuning).

Genera data/llm/colab_dataset.zip con train.jsonl y val.jsonl (los .bak
NUNCA se empaquetan), verificando antes que el dataset es la versión buena:
300 líneas de train, 100 de val y el campo "Tiempo en esa franja" presente
en todos los inputs (la versión fake del _predecir_fake no lo tiene).

Uso:
    uv run python climasafeai/llm/empaquetar_dataset_colab.py

Imprime el sha256 de ambos ficheros: es lo que compara la Celda 4 del
notebook de Colab con lo que se descomprime allí.
"""

from __future__ import annotations

import hashlib
import json
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TRAIN = ROOT / "data/llm/train.jsonl"
VAL = ROOT / "data/llm/val.jsonl"
ZIP = ROOT / "data/llm/colab_dataset.zip"

EXPECT_TRAIN = 300
EXPECT_VAL = 100
MARCA = "Tiempo en esa franja"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def _cargar(path: Path) -> list[dict]:
    ejemplos = []
    with open(path) as f:
        for linea in f:
            if linea.strip():
                ejemplos.append(json.loads(linea))
    return ejemplos


def verificar() -> list[str]:
    """Devuelve los problemas que impiden empaquetar. Vacía = se puede."""
    errores: list[str] = []
    for path, esperados, etiqueta in (
        (TRAIN, EXPECT_TRAIN, "train"),
        (VAL, EXPECT_VAL, "val"),
    ):
        if not path.exists():
            errores.append(f"No existe {path}")
            continue
        ejemplos = _cargar(path)
        if len(ejemplos) != esperados:
            errores.append(
                f"{etiqueta}: {len(ejemplos)} líneas (esperadas {esperados})"
            )
        con_marca = sum(1 for e in ejemplos if MARCA in (e.get("input") or ""))
        if con_marca != len(ejemplos):
            errores.append(
                f"{etiqueta}: solo {con_marca}/{len(ejemplos)} inputs contienen "
                f"'{MARCA}' — parece la versión fake"
            )
    return errores


def main() -> None:
    errores = verificar()
    if errores:
        print("No se empaqueta: el dataset NO es la versión buena (300/100).",
              file=sys.stderr)
        for e in errores:
            print(f"  - {e}", file=sys.stderr)
        print("Revisa data/llm/ (los .bak no se empaquetan).", file=sys.stderr)
        sys.exit(1)

    if ZIP.exists():
        ZIP.unlink()  # si ya existe, se regenera
    with zipfile.ZipFile(ZIP, "w", zipfile.ZIP_DEFLATED) as z:
        z.write(TRAIN, TRAIN.name)
        z.write(VAL, VAL.name)

    print(f"Empaquetado: {ZIP} ({ZIP.stat().st_size / 1024:.0f} KB)")
    print(f"  {TRAIN.name}: {_sha256(TRAIN)}  "
          f"({len(_cargar(TRAIN))} líneas)")
    print(f"  {VAL.name}: {_sha256(VAL)}  "
          f"({len(_cargar(VAL))} líneas)")
    print("Guarda estos sha256: la Celda 4 del notebook los compara.")


if __name__ == "__main__":
    main()
