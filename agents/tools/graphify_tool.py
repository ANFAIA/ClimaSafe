"""
agents.tools.graphify_tool — Puente entre el template y graphify.

graphify (github.com/anomalyco/graphify) convierte cualquier carpeta de código,
docs, papers, imágenes o vídeo en un grafo de conocimiento navegable
(`graphify-out/graph.json`). Esta herramienta centraliza TODA la interacción
con él para que los agentes (`knowledge`, `docsearch`, `git`) no reimplementen
cada uno la misma lógica de "¿dónde está el intérprete?", "¿existe el grafo?",
"¿cómo lanzo un --update?".

Límite honesto (igual que `vision_tool`): esta herramienta NO entiende
semánticamente el contenido de los nodos. Los "resúmenes de nodo padre" y las
"correlaciones" que calcula son **estructurales** — se derivan de la topología
del grafo (grado, vecinos compartidos, comunidad dominante), no de leer el
texto. Un resumen aquí dice "este nodo agrupa 12 hijos, los más conectados
entre sí son X e Y", no "este nodo trata sobre redes de atención".

No añade dependencias: graphify vive en su propio intérprete (`.graphify_python`)
y aquí solo se lee `graph.json` con la stdlib (json) y se lanzan subprocesos.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

from agents.tools.registry import register_tool

# Subcarpeta de caché dentro de graphify-out/. Va al .gitignore del template:
# es contenido derivado y voluminoso, no debe versionarse.
CACHE_SUBDIR = "cache"


@register_tool("graphify")
class GraphifyTool:
    # -- localización ---------------------------------------------------------
    @staticmethod
    def out_dir(root: Path) -> Path:
        return root / "graphify-out"

    @staticmethod
    def graph_json(root: Path) -> Path:
        return GraphifyTool.out_dir(root) / "graph.json"

    @staticmethod
    def cache_dir(root: Path) -> Path:
        return GraphifyTool.out_dir(root) / CACHE_SUBDIR

    @staticmethod
    def graph_exists(root: Path) -> bool:
        return GraphifyTool.graph_json(root).exists()

    @staticmethod
    def resolve_python(root: Path) -> str | None:
        """
        Devuelve el intérprete de Python que graphify dejó anotado en
        ``graphify-out/.graphify_python`` (escrito por el skill /graphify), o
        None si el marcador no existe o apunta a algo inexistente. NO cae al
        binario ``graphify``: eso no es un intérprete y no sirve para
        ``python -m graphify`` (ver ``command_prefix``).
        """
        marker = GraphifyTool.out_dir(root) / ".graphify_python"
        if marker.exists():
            candidate = marker.read_text(encoding="utf-8").strip()
            if candidate and Path(candidate).exists():
                return candidate
        return None

    @staticmethod
    def command_prefix(root: Path) -> list[str] | None:
        """
        Prefijo correcto para invocar graphify:
          - si hay intérprete anotado → ``[python, "-m", "graphify"]``
          - si no, pero hay binario ``graphify`` en el PATH → ``[graphify]``
          - si no hay ninguno → None.

        Distinguir ambos importa: ``graphify -m graphify ...`` (binario con
        ``-m``) es un comando inválido — ese era el bug de mezclar los dos.
        """
        python_bin = GraphifyTool.resolve_python(root)
        if python_bin is not None:
            return [python_bin, "-m", "graphify"]
        binary = shutil.which("graphify")
        if binary:
            return [binary]
        return None

    @staticmethod
    def is_available(root: Path) -> bool:
        return GraphifyTool.command_prefix(root) is not None

    # -- Obsidian -------------------------------------------------------------
    @staticmethod
    def detect_obsidian_vaults(root: Path, *, max_depth: int = 4) -> list[Path]:
        """
        Busca bóvedas de Obsidian bajo ``root``: cualquier carpeta que
        contenga un subdirectorio ``.obsidian/`` es la raíz de una bóveda.
        Devuelve las raíces de bóveda encontradas (no los ``.obsidian``).

        No desciende dentro de ``.git``, ``.venv``, ``node_modules`` ni
        ``graphify-out``: se podan del recorrido (con ``os.walk``, no
        ``rglob``, que sí entraría en ellos), porque esto corre en cada commit
        y esos árboles pueden ser enormes.
        """
        import os

        skip = {".git", ".venv", "venv", "node_modules", "graphify-out",
                "__pycache__", ".mypy_cache", ".ruff_cache"}
        vaults: list[Path] = []
        root = root.resolve()
        for dirpath, dirnames, _ in os.walk(root):
            if ".obsidian" in dirnames:
                vaults.append(Path(dirpath))
            rel = Path(dirpath).relative_to(root)
            depth = 0 if rel == Path(".") else len(rel.parts)
            # Poda in situ: no descender a skip, a .obsidian, ni más allá de max_depth.
            dirnames[:] = [
                d for d in dirnames
                if d not in skip and d != ".obsidian" and (depth + 1) <= max_depth
            ]
        return sorted(set(vaults))

    # -- lectura del grafo ----------------------------------------------------
    @staticmethod
    def _links(graph: dict[str, Any]) -> list[dict[str, Any]]:
        """Normaliza links/edges a una lista única (graphify usa 'links')."""
        return graph.get("links") or graph.get("edges") or []

    @staticmethod
    def load_graph(root: Path) -> dict[str, Any]:
        """
        Lee ``graphify-out/graph.json``. Devuelve el dict crudo con al menos
        ``nodes`` y la lista de enlaces accesible vía ``_links()``.
        Lanza FileNotFoundError si no existe —
        deja que el agente lo convierta en un AgentResult con mensaje claro.
        """
        path = GraphifyTool.graph_json(root)
        if not path.exists():
            raise FileNotFoundError(f"No existe {path}. Ejecuta graphify primero.")
        data = json.loads(path.read_text(encoding="utf-8"))
        data.setdefault("nodes", [])
        data.setdefault("links", [])
        return data

    @staticmethod
    def _adjacency(graph: dict[str, Any]) -> dict[str, set[str]]:
        """Construye adyacencia no dirigida {id_nodo: set(vecinos)} desde links/edges."""
        adj: dict[str, set[str]] = defaultdict(set)
        for link in GraphifyTool._links(graph):
            src = str(link.get("source", link.get("from", "")))
            tgt = str(link.get("target", link.get("to", "")))
            if not src or not tgt or src == tgt:
                continue
            adj[src].add(tgt)
            adj[tgt].add(src)
        return adj

    @staticmethod
    def _node_index(graph: dict[str, Any]) -> dict[str, dict[str, Any]]:
        return {str(n.get("id")): n for n in graph.get("nodes", []) if n.get("id") is not None}

    # -- agrupación por tópicos (extracción de temas desde labels) ------------
    @staticmethod
    def _topic_clusters(
        children: Iterable[str],
        nodes: dict[str, dict[str, Any]],
        *,
        min_term_freq: int = 2,
    ) -> list[dict[str, Any]]:
        """
        Agrupa los hijos por temas compartidos. Extrae palabras significativas
        de los labels de los hijos (excluye stopwords comunes) y forma clusters
        de términos que aparecen en múltiples hijos.

        Ejemplo: hijos con labels "paper_attention.pdf", "attention_is_all_you_need"
          → cluster "attention" con 2 hijos.
        """
        import re
        _STOPWORDS = {
            "the", "a", "an", "of", "in", "to", "for", "and", "or", "is", "are",
            "was", "were", "be", "been", "has", "have", "had", "not", "no",
            "with", "on", "at", "by", "from", "as", "it", "its", "this", "that",
            "de", "la", "el", "en", "un", "una", "del", "con", "por", "para",
            "que", "es", "lo", "se", "su", "al", "las", "los", "todo", "mas",
            "pero", "como", "cuando", "donde", "entre", "sobre", "tras", "sin",
        }

        term_children: dict[str, set[str]] = defaultdict(set)
        for child_id in children:
            node = nodes.get(child_id, {})
            label = str(node.get("label", node.get("norm_label", child_id)))
            # Extraer palabras alfanuméricas con al menos 3 caracteres
            words = set(
                w.lower() for w in re.findall(r"[a-zA-ZáéíóúñüÁÉÍÓÚÑÜ][a-zA-ZáéíóúñüÁÉÍÓÚÑÜ]{2,}", label)
                if w.lower() not in _STOPWORDS and not w.isdigit()
            )
            for word in words:
                term_children[word].add(child_id)

        clusters = [
            {"topic": term, "children": sorted(children_ids), "size": len(children_ids)}
            for term, children_ids in term_children.items()
            if len(children_ids) >= min_term_freq
        ]
        clusters.sort(key=lambda c: -c["size"])
        return clusters

    @staticmethod
    def _child_type_groups(
        children: Iterable[str],
        nodes: dict[str, dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Agrupa hijos por su file_type (code/document/image/...) con conteo y labels."""
        groups: dict[str, list[str]] = defaultdict(list)
        for child_id in children:
            node = nodes.get(child_id, {})
            ftype = str(node.get("file_type", node.get("type", "unknown")))
            groups[ftype].append(node.get("label", child_id))
        return [
            {"type": t, "count": len(labels), "labels": sorted(labels)}
            for t, labels in sorted(groups.items(), key=lambda kv: -len(kv[1]))
        ]

    # -- resúmenes de nodo padre (semánticos + estructurales) -----------------
    @staticmethod
    def parent_summaries(
        graph: dict[str, Any],
        *,
        min_children: int = 3,
        top: int = 10,
    ) -> list[dict[str, Any]]:
        """
        Para los nodos "padre" del grafo (los de mayor grado — hubs/god nodes),
        produce un resumen que combina información ESTRUCTURAL + SEMÁNTICA:

          - cuántos hijos tiene y agrupados por tipo (code, document, image...)
          - **tópicos detectados**: palabras clave que se repiten entre los
            labels de los hijos (ej: varios hijos con "attention" en el label
            indican un cluster temático)
          - la comunidad dominante entre los hijos
          - los pares de hijos más correlacionados, con explicación de POR QUÉ
            (misma comunidad, mismo tipo, vecinos compartidos)

        Devuelve como mucho ``top`` resúmenes, ordenados por número de hijos.
        Solo incluye nodos con al menos ``min_children`` hijos.
        """
        adj = GraphifyTool._adjacency(graph)
        nodes = GraphifyTool._node_index(graph)

        parents = sorted(adj.items(), key=lambda kv: len(kv[1]), reverse=True)
        summaries: list[dict[str, Any]] = []
        for parent_id, children in parents:
            if len(children) < min_children:
                continue
            parent_node = nodes.get(parent_id, {})

            # Tipo y comunidad de cada hijo
            type_counts: dict[str, int] = defaultdict(int)
            comm_counts: dict[str, int] = defaultdict(int)
            child_comm_names: dict[str, str] = {}
            for child_id in children:
                cnode = nodes.get(child_id, {})
                ftype = str(cnode.get("file_type", cnode.get("type", "desconocido")))
                type_counts[ftype] += 1
                comm = cnode.get("community")
                if comm is not None:
                    comm_counts[str(comm)] += 1
                cname = cnode.get("community_name")
                if cname:
                    child_comm_names[str(comm)] = cname

            dominant_community_id = (
                max(comm_counts.items(), key=lambda kv: kv[1])[0] if comm_counts else None
            )
            dominant_community_name = child_comm_names.get(dominant_community_id, dominant_community_id)

            # Tópicos detectados en los labels de los hijos
            topics = GraphifyTool._topic_clusters(children, nodes)

            # Grupos por tipo
            type_groups = GraphifyTool._child_type_groups(children, nodes)

            # Correlación entre hijos (con explicación)
            correlated = GraphifyTool._correlated_pairs(
                children, adj, nodes, graph=graph, top=5,
            )

            summaries.append({
                "id": parent_id,
                "label": parent_node.get("label", parent_id),
                "type": parent_node.get("file_type", parent_node.get("type", "desconocido")),
                "n_children": len(children),
                "child_types": dict(type_counts),
                "child_type_groups": type_groups,
                "dominant_community": dominant_community_id,
                "dominant_community_name": dominant_community_name,
                "topics": topics[:8] if topics else [],
                "correlated_children": correlated,
                "summary": GraphifyTool._render_summary(
                    parent_node.get("label", parent_id),
                    len(children), type_groups, topics, dominant_community_name, correlated,
                ),
            })
            if len(summaries) >= top:
                break
        return summaries

    @staticmethod
    def _correlated_pairs(
        children: Iterable[str],
        adj: dict[str, set[str]],
        nodes: dict[str, dict[str, Any]],
        *,
        graph: dict[str, Any] | None = None,
        top: int = 5,
        max_children: int = 200,
    ) -> list[dict[str, Any]]:
        """
        Pares de hijos más correlacionados. Además del score de Jaccard,
        explica POR QUÉ se relacionan: misma comunidad, mismo tipo de archivo,
        o vecinos compartidos.
        """
        children_list = sorted(children, key=lambda c: (-len(adj.get(c, set())), c))[:max_children]
        pairs: list[tuple[float, str, str]] = []
        for i in range(len(children_list)):
            for j in range(i + 1, len(children_list)):
                a, b = children_list[i], children_list[j]
                na, nb = adj.get(a, set()), adj.get(b, set())
                union = na | nb
                if not union:
                    continue
                jaccard = len(na & nb) / len(union)
                direct = b in na
                score = jaccard + (0.5 if direct else 0.0)
                if score > 0:
                    pairs.append((score, a, b))
        pairs.sort(reverse=True)

        result = []
        for score, a, b in pairs[:top]:
            na_node = nodes.get(a, {})
            nb_node = nodes.get(b, {})
            reasons = []

            # Razón 1: misma comunidad
            ca = na_node.get("community")
            cb = nb_node.get("community")
            if ca is not None and cb is not None and ca == cb:
                cname = na_node.get("community_name", f"comunidad {ca}")
                reasons.append(f"misma {cname}")

            # Razón 2: mismo tipo de archivo
            ta = na_node.get("file_type", na_node.get("type"))
            tb = nb_node.get("file_type", nb_node.get("type"))
            if ta is not None and tb is not None and ta == tb:
                reasons.append(f"ambos {ta}")

            # Razón 3: vecinos compartidos
            shared = len(adj.get(a, set()) & adj.get(b, set()))
            if shared > 2:
                reasons.append(f"{shared} vecinos en común")

            result.append({
                "a": na_node.get("label", a),
                "b": nb_node.get("label", b),
                "score": round(score, 3),
                "shared_neighbors": shared,
                "reasons": reasons,
            })
        return result

    @staticmethod
    def _render_summary(
        label: str,
        n_children: int,
        type_groups: list[dict[str, Any]],
        topics: list[dict[str, Any]],
        dominant_community: str | None,
        correlated: list[dict[str, Any]],
    ) -> str:
        parts = [f"'{label}' agrupa {n_children} hijo(s)"]

        # Desglose por tipo
        type_desc = ", ".join(
            f"{g['count']} {g['type']}" for g in type_groups[:4]
        )
        if type_desc:
            parts.append(f"tipos: {type_desc}")

        # Tópicos destacados
        if topics:
            topic_desc = ", ".join(
                f"'{t['topic']}' ({t['size']})" for t in topics[:5]
            )
            parts.append(f"tópicos: {topic_desc}")

        if dominant_community:
            parts.append(f"mayoría en {dominant_community}")

        if correlated:
            top_pair = correlated[0]
            reason_str = f" — {'; '.join(top_pair['reasons'])}" if top_pair.get("reasons") else ""
            parts.append(
                f"más relacionados: '{top_pair['a']}' ↔ '{top_pair['b']}'"
                f"{reason_str}"
            )

        return "; ".join(parts) + "."

    # -- poda de nodos --------------------------------------------------------
    @staticmethod
    def prune(
        graph: dict[str, Any],
        *,
        node_types: Iterable[str] | None = None,
        node_ids: Iterable[str] | None = None,
        drop_isolated: bool = False,
    ) -> tuple[dict[str, Any], dict[str, int]]:
        """
        Devuelve una COPIA del grafo sin los nodos indicados (y sin sus
        enlaces). No escribe a disco — el agente decide si persistir
        con ``save_graph``.

          - node_types : elimina todo nodo cuyo ``file_type`` esté en el conjunto
          - node_ids   : elimina nodos concretos por id.

        Devuelve (grafo_podado, stats).
        """
        drop_types = {str(t) for t in (node_types or [])}
        drop_ids = {str(i) for i in (node_ids or [])}

        kept_nodes = []
        removed_ids: set[str] = set()
        for n in graph.get("nodes", []):
            nid = str(n.get("id"))
            nodetype = str(n.get("file_type", n.get("type", "")))
            if nid in drop_ids or nodetype in drop_types:
                removed_ids.add(nid)
            else:
                kept_nodes.append(n)

        kept_links = []
        removed_links = 0
        for link in GraphifyTool._links(graph):
            src = str(link.get("source", link.get("from", "")))
            tgt = str(link.get("target", link.get("to", "")))
            if src in removed_ids or tgt in removed_ids:
                removed_links += 1
            else:
                kept_links.append(link)

        if drop_isolated:
            connected: set[str] = set()
            for link in kept_links:
                connected.add(str(link.get("source", link.get("from", ""))))
                connected.add(str(link.get("target", link.get("to", ""))))
            isolated_ids = {str(n.get("id")) for n in kept_nodes
                            if str(n.get("id")) not in connected}
            kept_nodes = [n for n in kept_nodes if str(n.get("id")) not in isolated_ids]
            removed_ids |= isolated_ids
            isolated_removed = len(isolated_ids)
        else:
            isolated_removed = 0

        key = "links" if graph.get("links") is not None else "edges"
        pruned = dict(graph)
        pruned["nodes"] = kept_nodes
        pruned[key] = kept_links
        stats = {
            "nodes_removed": len(removed_ids),
            "edges_removed": removed_links,
            "isolated_removed": isolated_removed,
            "nodes_remaining": len(kept_nodes),
            "edges_remaining": len(kept_links),
        }
        return pruned, stats

    @staticmethod
    def save_graph(root: Path, graph: dict[str, Any], *, backup: bool = True) -> Path:
        """Escribe el grafo a ``graph.json``, dejando antes un ``graph.json.bak``."""
        path = GraphifyTool.graph_json(root)
        if backup and path.exists():
            shutil.copy2(path, path.with_suffix(".json.bak"))
        path.write_text(json.dumps(graph, indent=2, ensure_ascii=False), encoding="utf-8")
        return path

    # -- Obsidian Flavored Markdown (convenciones kepano/obsidian-skills) ------
    # Estas notas siguen la spec de github.com/kepano/obsidian-skills para que
    # la bóveda sea óptima y editable por cualquier agente que tenga instaladas
    # esas skills (Claude Code, Codex, opencode). Frontmatter con properties,
    # wikilinks [[...]], callouts > [!type] y tags anidados.
    @staticmethod
    def obsidian_frontmatter(
        title: str,
        tags: list[str],
        *,
        aliases: list[str] | None = None,
        cssclasses: list[str] | None = None,
    ) -> str:
        """Bloque de properties (YAML frontmatter) de una nota de Obsidian."""
        lines = ["---", f"title: {title}"]
        if tags:
            lines.append("tags:")
            lines += [f"  - {t}" for t in tags]
        if aliases:
            lines.append("aliases:")
            lines += [f"  - {a}" for a in aliases]
        if cssclasses:
            lines.append("cssclasses:")
            lines += [f"  - {c}" for c in cssclasses]
        lines.append("---")
        return "\n".join(lines)

    @staticmethod
    def obsidian_note(
        title: str,
        tags: list[str],
        body: str,
        *,
        aliases: list[str] | None = None,
        cssclasses: list[str] | None = None,
    ) -> str:
        """Nota completa: frontmatter + cuerpo en Obsidian Flavored Markdown."""
        front = GraphifyTool.obsidian_frontmatter(
            title, tags, aliases=aliases, cssclasses=cssclasses
        )
        return f"{front}\n\n{body.rstrip()}\n"

    @staticmethod
    def knowledge_base(*, name: str = "Nodos del grafo", tag: str = "knowledge") -> str:
        """
        Devuelve un archivo Obsidian Bases (`.base`, YAML) que muestra las notas
        de la bóveda etiquetadas con ``tag`` como tabla y como tarjetas. Sigue
        la spec obsidian-bases de kepano/obsidian-skills.
        """
        return (
            "filters:\n"
            "  and:\n"
            f"    - 'file.hasTag(\"{tag}\")'\n"
            "properties:\n"
            "  file.name:\n"
            "    displayName: Nota\n"
            "  tags:\n"
            "    displayName: Etiquetas\n"
            "views:\n"
            "  - type: table\n"
            f"    name: \"{name}\"\n"
            "    order:\n"
            "      - file.name\n"
            "      - tags\n"
            "  - type: cards\n"
            "    name: \"Tarjetas\"\n"
            "    order:\n"
            "      - file.name\n"
        )

    # -- ejecución de graphify (subprocesos) ----------------------------------
    @staticmethod
    def run_cli(root: Path, args: list[str], *, timeout: int = 180) -> subprocess.CompletedProcess:
        """
        Lanza graphify con el prefijo correcto (intérprete + ``-m graphify`` o
        el binario del PATH) en la raíz del proyecto. El que llama inspecciona
        ``returncode``, ``stdout`` y ``stderr``.
        """
        prefix = GraphifyTool.command_prefix(root)
        if prefix is None:
            raise FileNotFoundError(
                "graphify no está disponible (ni .graphify_python ni binario en PATH). "
                "Ejecuta el skill /graphify una vez para instalarlo."
            )
        cmd = [*prefix, *args]
        return subprocess.run(cmd, cwd=str(root), capture_output=True, text=True, timeout=timeout)

    @staticmethod
    def build(root: Path, *, timeout: int = 300) -> subprocess.CompletedProcess:
        """
        Construye/actualiza el grafo. Si ya existe ``graph.json`` usa
        ``--update`` (solo archivos nuevos/cambiados); si no, hace un build
        completo — ``--update`` sin grafo previo no tiene manifest del que
        partir.
        """
        args = [str(root), "--update"] if GraphifyTool.graph_exists(root) else [str(root)]
        return GraphifyTool.run_cli(root, args, timeout=timeout)

    @staticmethod
    def update(root: Path, *, timeout: int = 180) -> subprocess.CompletedProcess:
        """Re-extrae solo los archivos nuevos o cambiados (``graphify . --update``)."""
        return GraphifyTool.run_cli(root, [str(root), "--update"], timeout=timeout)

    @staticmethod
    def export_obsidian(root: Path, vault_dir: Path, *, timeout: int = 180) -> subprocess.CompletedProcess:
        """Exporta el grafo como bóveda de Obsidian (``graphify export obsidian --dir``)."""
        return GraphifyTool.run_cli(
            root, ["export", "obsidian", "--dir", str(vault_dir)], timeout=timeout
        )

    @staticmethod
    def query(root: Path, question: str, *, budget: int | None = None,
              timeout: int = 120) -> subprocess.CompletedProcess:
        """Consulta el grafo en lenguaje natural (``graphify query``)."""
        args = ["query", question]
        if budget:
            args += ["--budget", str(budget)]
        return GraphifyTool.run_cli(root, args, timeout=timeout)
