from __future__ import annotations

import base64
import io
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from agents.agents.notebook_agent import NotebookAgent
from agents.tools.notebook_tool import NotebookTool


def _make_notebook_with_image(path: Path) -> None:
    fig, ax = plt.subplots()
    ax.plot([1, 2, 3], [1, 4, 9])
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    plt.close(fig)
    png_b64 = base64.b64encode(buf.getvalue()).decode("ascii")

    notebook = {
        "nbformat": 4, "nbformat_minor": 5, "metadata": {},
        "cells": [
            {"cell_type": "code", "id": "c1", "metadata": {}, "execution_count": 1,
             "source": ["plt.plot([1,2,3],[1,4,9])"],
             "outputs": [{"output_type": "display_data", "data": {"image/png": [png_b64]}, "metadata": {}}]},
            {"cell_type": "code", "id": "c2", "metadata": {}, "execution_count": 2,
             "source": ["print('ok')"],
             "outputs": [{"output_type": "stream", "name": "stdout", "text": ["ok\n"]}]},
        ],
    }
    path.write_text(json.dumps(notebook))


def test_extract_outputs_finds_image_and_text(context):
    notebook_path = context.root / "nb.ipynb"
    _make_notebook_with_image(notebook_path)

    agent = NotebookAgent(context=context)
    result = agent.extract_outputs(notebook_path="nb.ipynb")

    assert result.success
    assert result.data["n_images"] == 1
    assert result.data["n_cells_with_output"] == 2
    workdir = context.agent_workspace("notebook")
    assert (workdir / "manifest.json").exists()


def test_extract_outputs_missing_notebook_fails(context):
    agent = NotebookAgent(context=context)
    result = agent.extract_outputs(notebook_path="no_existe.ipynb")
    assert not result.success


def test_insert_comments_default_does_not_touch_original(context):
    notebook_path = context.root / "nb.ipynb"
    _make_notebook_with_image(notebook_path)
    original_cells = json.loads(notebook_path.read_text())["cells"]

    agent = NotebookAgent(context=context)
    result = agent.insert_comments(
        notebook_path="nb.ipynb",
        insertions=[{"cell_index": 0, "markdown": "Interpretación de prueba."}],
    )

    assert result.success
    # el original no cambió
    assert json.loads(notebook_path.read_text())["cells"] == original_cells
    # la copia sí tiene la celda nueva
    commented = json.loads(Path(result.data["output_path"]).read_text())
    assert len(commented["cells"]) == 3
    assert commented["cells"][1]["cell_type"] == "markdown"


def test_insert_comments_in_place_overwrites_original(context):
    notebook_path = context.root / "nb.ipynb"
    _make_notebook_with_image(notebook_path)

    agent = NotebookAgent(context=context)
    result = agent.insert_comments(
        notebook_path="nb.ipynb",
        insertions=[{"cell_index": 1, "markdown": "Comentario in situ."}],
        in_place=True,
    )

    assert result.success
    updated = json.loads(notebook_path.read_text())
    assert len(updated["cells"]) == 3
    assert any("Comentario in situ." in "".join(c.get("source", [])) for c in updated["cells"])


def test_notebook_tool_skips_out_of_range_cell_index(tmp_path: Path):
    notebook_path = tmp_path / "nb.ipynb"
    _make_notebook_with_image(notebook_path)
    result = NotebookTool.insert_comments(notebook_path, [{"cell_index": 99, "markdown": "x"}])
    assert result["n_inserted"] == 0
    assert result["skipped_indices"] == [99]
