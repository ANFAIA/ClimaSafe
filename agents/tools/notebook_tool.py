"""
agents.tools.notebook_tool — Extrae salidas de un .ipynb e inserta celdas
markdown nuevas. Lógica idéntica a la que ya probé y validé como skill de
Claude independiente (`notebook-insights-commentator`) — aquí vive la misma
implementación, reutilizada como herramienta del sistema de agentes.

Formato nbformat v4 verificado contra la documentación oficial
(nbformat.readthedocs.io, schema en github.com/jupyter/nbformat):
- notebook: {"nbformat": 4, "nbformat_minor": N, "metadata": {...}, "cells": [...]}
- celda de código: {"cell_type": "code", "source": str|list[str], "outputs": [...]}
- tipos de output: "execute_result" (data + execution_count), "display_data"
  (data, sin execution_count), "stream" (name: stdout/stderr, text),
  "error" (ename, evalue, traceback)
- "data" es un mime-bundle {"text/plain": ..., "image/png": ..., ...} cuyo
  valor puede ser un string o una lista de strings (a unir con "").
- los ids de celda son obligatorios desde nbformat 4.5 ("These ids must be
  unique to any given Notebook") — ver `insert_comments` para cómo se maneja.

Límite honesto e importante: esta herramienta NO interpreta nada. Extrae
imágenes/texto para que algo con comprensión real (una persona, o un LLM
orquestando este sistema) escriba la interpretación. `NotebookAgent` es, a
propósito, el primer agente de este sistema cuya utilidad depende de que
quien lo invoca aporte juicio que el propio agente no tiene — está
documentado así en `agents/agents/notebook_agent.py` y en
`agents/prompts/notebook_agent.md`, no se pretende lo contrario en ningún sitio.
"""

from __future__ import annotations

import base64
import json
import uuid
from pathlib import Path
from typing import Any

from agents.tools.registry import register_tool

MAX_TEXT_CHARS = 2000


def _join_source(value: Any) -> str:
    """'source' (y los campos de texto dentro de 'data') pueden ser un string
    o una lista de strings — la documentación de nbformat exige soportar ambos."""
    if isinstance(value, list):
        return "".join(value)
    return value or ""


@register_tool("notebook")
class NotebookTool:
    @staticmethod
    def extract(notebook_path: Path, workdir: Path) -> dict:
        """Extrae salidas de texto e imágenes a `workdir/` y devuelve el manifest."""
        notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
        workdir.mkdir(parents=True, exist_ok=True)

        manifest: list[dict] = []
        n_images = 0

        for cell_index, cell in enumerate(notebook.get("cells", [])):
            if cell.get("cell_type") != "code":
                continue
            outputs = cell.get("outputs", [])
            if not outputs:
                continue

            entry: dict[str, Any] = {
                "cell_index": cell_index,
                "source": _join_source(cell.get("source"))[:MAX_TEXT_CHARS],
                "text_outputs": [],
                "image_paths": [],
                "errored": False,
            }

            for output_index, output in enumerate(outputs):
                output_type = output.get("output_type")

                if output_type == "error":
                    entry["errored"] = True
                    entry["text_outputs"].append(
                        f"[ERROR] {output.get('ename', '')}: {output.get('evalue', '')}"
                    )
                    continue

                if output_type == "stream":
                    text = _join_source(output.get("text"))
                    stream_name = output.get("name", "stdout")
                    entry["text_outputs"].append(f"[{stream_name}] {text[:MAX_TEXT_CHARS]}")
                    continue

                if output_type in ("execute_result", "display_data"):
                    data = output.get("data", {})

                    image_b64 = data.get("image/png")
                    if image_b64:
                        image_bytes = base64.b64decode(_join_source(image_b64))
                        image_path = workdir / f"cell{cell_index:03d}_out{output_index}.png"
                        image_path.write_bytes(image_bytes)
                        entry["image_paths"].append(str(image_path))
                        n_images += 1

                    text_plain = data.get("text/plain")
                    if text_plain:
                        entry["text_outputs"].append(_join_source(text_plain)[:MAX_TEXT_CHARS])

                    if "text/html" in data and not text_plain:
                        entry["text_outputs"].append(
                            "[salida HTML disponible (probablemente una tabla) — "
                            "no se incluye el HTML completo aquí]"
                        )

            if entry["text_outputs"] or entry["image_paths"] or entry["errored"]:
                manifest.append(entry)

        manifest_path = workdir / "manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

        return {
            "manifest_path": str(manifest_path),
            "manifest": manifest,
            "n_cells_with_output": len(manifest),
            "n_images": n_images,
        }

    @staticmethod
    def insert_comments(notebook_path: Path, insertions: list[dict], *, output_path: Path | None = None) -> dict:
        """
        Inserta una celda markdown nueva justo debajo de cada
        `{"cell_index": int, "markdown": str}` de `insertions`. Procesa de
        mayor a menor `cell_index` para que los índices originales sigan
        siendo válidos mientras inserta.
        """
        notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
        cells = notebook.get("cells", [])
        # Los ids de celda son obligatorios desde nbformat 4.5 — se generan
        # solo si el notebook ya declara esa versión, para no introducir un
        # campo que versiones más antiguas del formato no esperan.
        include_id = notebook.get("nbformat") == 4 and notebook.get("nbformat_minor", 0) >= 5

        skipped = []
        ordered = sorted(insertions, key=lambda item: item["cell_index"], reverse=True)
        n_inserted = 0
        for item in ordered:
            idx = item["cell_index"]
            if idx < 0 or idx >= len(cells):
                skipped.append(idx)
                continue
            new_cell: dict[str, Any] = {
                "cell_type": "markdown",
                "metadata": {"tags": ["auto-comentario"]},
                "source": [item["markdown"]],
            }
            if include_id:
                new_cell["id"] = uuid.uuid4().hex[:8]
            cells.insert(idx + 1, new_cell)
            n_inserted += 1

        notebook["cells"] = cells
        target = output_path or notebook_path
        target.write_text(json.dumps(notebook, indent=1, ensure_ascii=False), encoding="utf-8")

        return {"output_path": str(target), "n_inserted": n_inserted, "skipped_indices": skipped}
