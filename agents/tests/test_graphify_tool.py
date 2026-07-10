from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

from agents.tools.graphify_tool import GraphifyTool


def _sample_graph() -> dict:
    # Un nodo padre "info" con tres papers hijos; p1 y p2 comparten vecinos
    # (correlación alta) y ref1 es una referencia; iso queda aislado.
    return {
        "nodes": [
            {"id": "info", "label": "informacion", "type": "topic", "community": 0},
            {"id": "p1", "label": "Paper A", "type": "paper", "community": 0},
            {"id": "p2", "label": "Paper B", "type": "paper", "community": 0},
            {"id": "p3", "label": "Paper C", "type": "paper", "community": 1},
            {"id": "ref1", "label": "http://x", "type": "reference"},
            {"id": "iso", "label": "aislado", "type": "doc"},
        ],
        "edges": [
            {"source": "info", "target": "p1"},
            {"source": "info", "target": "p2"},
            {"source": "info", "target": "p3"},
            {"source": "p1", "target": "p2"},
            {"source": "p1", "target": "ref1"},
            {"source": "p2", "target": "ref1"},
        ],
    }


def test_parent_summaries_groups_children_with_correlation():
    summaries = GraphifyTool.parent_summaries(_sample_graph(), min_children=3, top=5)
    # 'info' es el padre de mayor grado (3 hijos)
    top = summaries[0]
    assert top["id"] == "info"
    assert top["n_children"] == 3
    assert top["child_types"]["paper"] == 3
    assert top["dominant_community"] == "0"
    # Paper A y Paper B están correlacionados (comparten info y ref1)
    pair = top["correlated_children"][0]
    assert {pair["a"], pair["b"]} == {"Paper A", "Paper B"}
    assert pair["shared_neighbors"] >= 2
    assert "agrupa 3" in top["summary"]


def test_parent_summaries_respects_min_children():
    # Con min_children alto, ningún nodo califica
    assert GraphifyTool.parent_summaries(_sample_graph(), min_children=10) == []


def test_prune_by_type_and_isolated():
    pruned, stats = GraphifyTool.prune(
        _sample_graph(), node_types=["reference"], drop_isolated=True,
    )
    remaining_ids = {n["id"] for n in pruned["nodes"]}
    assert "ref1" not in remaining_ids  # quitado por tipo
    assert "iso" not in remaining_ids   # quitado por aislado
    assert stats["nodes_removed"] == 2
    assert stats["edges_removed"] == 2
    assert stats["nodes_remaining"] == 4


def test_prune_by_id():
    pruned, stats = GraphifyTool.prune(_sample_graph(), node_ids=["p3"])
    assert "p3" not in {n["id"] for n in pruned["nodes"]}
    assert stats["nodes_removed"] == 1


def test_save_graph_writes_backup(tmp_path: Path):
    out = tmp_path / "graphify-out"
    out.mkdir()
    original = _sample_graph()
    (out / "graph.json").write_text(json.dumps(original), encoding="utf-8")

    pruned, _ = GraphifyTool.prune(original, node_types=["reference"])
    GraphifyTool.save_graph(tmp_path, pruned, backup=True)

    assert (out / "graph.json.bak").exists()  # backup del original
    saved = json.loads((out / "graph.json").read_text(encoding="utf-8"))
    assert "ref1" not in {n["id"] for n in saved["nodes"]}


def test_obsidian_note_has_frontmatter_and_body():
    note = GraphifyTool.obsidian_note(
        "Mi Nota", tags=["knowledge", "knowledge/papers"],
        body="> [!info] Hola\n> cuerpo", aliases=["Alias"], cssclasses=["c1"],
    )
    assert note.startswith("---\ntitle: Mi Nota\n")
    assert "tags:\n  - knowledge\n  - knowledge/papers" in note
    assert "aliases:\n  - Alias" in note
    assert "cssclasses:\n  - c1" in note
    assert "> [!info] Hola" in note


def test_knowledge_base_is_valid_yaml():
    base = GraphifyTool.knowledge_base()
    yaml = __import__("yaml") if _has_yaml() else None
    if yaml is not None:
        parsed = yaml.safe_load(base)
        assert parsed["views"][0]["type"] == "table"
        assert 'file.hasTag("knowledge")' in parsed["filters"]["and"]
    else:
        assert "views:" in base and 'file.hasTag("knowledge")' in base


def _has_yaml() -> bool:
    try:
        import yaml  # noqa: F401
        return True
    except ImportError:
        return False


def test_command_prefix_uses_marker_python_over_binary(tmp_path: Path):
    # Con .graphify_python presente, el prefijo es `python -m graphify`, no el
    # binario del PATH — invocarlo como `graphify -m graphify` sería inválido.
    (tmp_path / "graphify-out").mkdir()
    (tmp_path / "graphify-out" / ".graphify_python").write_text(sys.executable)
    assert GraphifyTool.command_prefix(tmp_path) == [sys.executable, "-m", "graphify"]
    assert GraphifyTool.is_available(tmp_path) is True


def test_command_prefix_none_without_graphify(tmp_path: Path):
    # Sin marcador y sin binario en el PATH, no está disponible.
    if shutil.which("graphify") is not None:
        return  # graphify instalado en este entorno — el caso None no aplica
    assert GraphifyTool.command_prefix(tmp_path) is None
    assert GraphifyTool.is_available(tmp_path) is False


def test_prune_isolated_only_keeps_connected(tmp_path: Path):
    graph = {
        "nodes": [{"id": "a"}, {"id": "b"}, {"id": "lonely"}],
        "edges": [{"source": "a", "target": "b"}],
    }
    pruned, stats = GraphifyTool.prune(graph, drop_isolated=True)
    ids = {n["id"] for n in pruned["nodes"]}
    assert "lonely" not in ids           # sin aristas → fuera
    assert {"a", "b"} <= ids             # conectados → se quedan
    assert stats["isolated_removed"] == 1


def test_detect_obsidian_vaults(tmp_path: Path):
    (tmp_path / "knowledge" / ".obsidian").mkdir(parents=True)
    (tmp_path / ".venv" / "junk" / ".obsidian").mkdir(parents=True)  # debe ignorarse
    vaults = GraphifyTool.detect_obsidian_vaults(tmp_path)
    assert tmp_path / "knowledge" in vaults
    assert all(".venv" not in str(v) for v in vaults)
