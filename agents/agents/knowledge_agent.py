"""
agents.agents.knowledge_agent — Grafo de conocimiento + Obsidian, cacheado.

Este es el agente que "cachea lo máximo posible": envuelve graphify
(`graphify-out/graph.json`) y una bóveda de Obsidian, y mantiene ambos en
sincronía. Su razón de ser:

  1. Detecta si el proyecto ya tiene una bóveda de Obsidian; si no, ofrece
     crear una con una estructura de árbol adaptada al grafo (carpetas por
     tipo de nodo: papers, code, docs, references, media). Las notas siguen
     las convenciones de github.com/kepano/obsidian-skills (Obsidian Flavored
     Markdown: properties, wikilinks, callouts + vistas Obsidian Bases), para
     que la bóveda sea óptima y editable desde Claude Code, Codex u opencode.
  2. Construye/actualiza el grafo con graphify y lo exporta a esa bóveda,
     fusionando ambos mundos.
  3. Resume cada NODO PADRE (hub) con un resumen de sus nodos hijo, incluyendo
     la correlación estructural entre ellos — para que, por ejemplo, un nodo
     "información" con muchos papers hijos tenga un resumen de qué agrupan y
     cómo se relacionan.
  4. Cachea en `graphify-out/cache/` (ignorado por git) todo lo caro:
     los resúmenes de nodo padre y las consultas.

Toda la interacción con graphify vive en `agents.tools.graphify_tool`; aquí
solo está la orquestación y el formateo a `AgentResult`.

Límite honesto: los resúmenes y correlaciones son ESTRUCTURALES (topología del
grafo), no una lectura semántica del contenido de los nodos. Ver el docstring
de `graphify_tool` para el detalle.
"""

from __future__ import annotations

from pathlib import Path

from agents.core.base_agent import AgentResult, BaseAgent
from agents.core.registry import register_agent
from agents.tools.cache_tool import CacheTool
from agents.tools.graphify_tool import GraphifyTool

# Árbol adaptado que se crea dentro de una bóveda nueva. Cada carpeta agrupa un
# tipo de nodo del grafo; graphify luego rellena una nota por nodo.
_VAULT_TREE = {
    "00-index": "Mapas de contenido (MOC) y punto de entrada de la bóveda.",
    "papers": "Artículos, PDFs y sus extracciones.",
    "code": "Módulos y símbolos extraídos por AST.",
    "docs": "Documentación, notas y markdown del proyecto.",
    "references": "Referencias externas, enlaces y citas.",
    "media": "Imágenes y transcripciones de audio/vídeo.",
}


@register_agent
class KnowledgeAgent(BaseAgent):
    name = "knowledge"
    description = (
        "Grafo de conocimiento (graphify) + Obsidian, cacheado al máximo. "
        "Crea/sincroniza una bóveda con estructura de árbol, resume cada nodo "
        "padre con la correlación entre sus hijos, y mantiene el grafo al día."
    )
    # `graphify-out/` tiene UN dueño: este agente. Las palabras de búsqueda
    # ("busca", "vecinos", "referencias") son de `doc`, que solo lee.
    capabilities = [
        "conocimiento", "knowledge", "grafo", "graph", "graphify", "obsidian",
        "boveda", "vault", "indexar", "nodo padre", "resumen del grafo",
        "sincroniza el grafo",
        # Absorbidas de `cache`, que era el tercer agente tocando graphify-out/.
        "cache", "caché", "cachear", "warmup", "precarga", "precalentar",
        "limpiar cache", "estado de cache", "podar el grafo",
    ]

    def action_aliases(self) -> dict:
        return {
            "setup_vault": ["obsidian", "boveda", "vault", "crea la boveda", "estructura"],
            "build": ["construye", "genera el grafo", "indexa", "graphify"],
            "summarize_parents": ["resume", "nodo padre", "padres", "resumen", "correlacion"],
            "preprocess": ["preprocesa", "enriquece", "cachea resumenes"],
            "clean": ["limpia", "elimina ruido", "rationale", "purga", "noise"],
            "prune": ["poda", "podar", "quita", "elimina nodos", "descarta"],
            "sync": ["sincroniza", "actualiza", "update"],
            "status": ["estado", "situacion"],
            "cache_warmup": ["warmup", "precarga", "precalienta", "prepara cache"],
            "cache_status": ["estado de cache", "tamano de cache", "cuantas entradas"],
            "cache_clear": ["limpia la cache", "borra la cache", "vacia", "resetea"],
            "graph_stats": ["estadisticas del grafo", "nodos", "aristas", "comunidades"],
        }

    def actions(self) -> dict:
        return {
            "status": self.status,
            "setup_vault": self.setup_vault,
            "build": self.build,
            "summarize_parents": self.summarize_parents,
            "preprocess": self.preprocess,
            "clean": self.clean,
            "prune": self.prune,
            "sync": self.sync,
            "cache_warmup": self.cache_warmup,
            "cache_status": self.cache_status,
            "cache_clear": self.cache_clear,
            "graph_stats": self.graph_stats,
        }

    # -------------------------------------------------------------------------
    def _cache_dir(self) -> Path:
        """Fija (y crea) el directorio de caché en graphify-out/cache/."""
        cache_dir = GraphifyTool.cache_dir(self.ctx.root)
        CacheTool.set_cache_dir(cache_dir)
        return cache_dir

    def _require_graph(self, action: str) -> AgentResult | None:
        if not GraphifyTool.graph_exists(self.ctx.root):
            return AgentResult(
                False, self.name, action,
                "No hay grafo (graphify-out/graph.json). Ejecuta 'knowledge build' primero.",
            )
        return None

    def status(self) -> AgentResult:
        """Resume el estado del grafo, la caché y las bóvedas de Obsidian."""
        root = self.ctx.root
        available = GraphifyTool.is_available(root)
        graph_exists = GraphifyTool.graph_exists(root)
        vaults = GraphifyTool.detect_obsidian_vaults(root)
        cache_dir = GraphifyTool.cache_dir(root)
        n_cache = len(list(cache_dir.glob("*.joblib"))) if cache_dir.exists() else 0

        n_nodes = n_edges = 0
        if graph_exists:
            try:
                graph = GraphifyTool.load_graph(root)
                n_nodes, n_edges = len(graph["nodes"]), len(graph["edges"])
            except Exception:  # noqa: BLE001 — grafo corrupto, no debe tumbar el status
                pass

        warnings = []
        if not available:
            warnings.append(
                "graphify no está instalado. Ejecuta el skill /graphify una vez "
                "para instalarlo y construir el grafo inicial."
            )
        if not vaults:
            warnings.append(
                "No se detectó ninguna bóveda de Obsidian. Ejecuta "
                "'knowledge setup_vault' para crear una con estructura de árbol."
            )

        return AgentResult(
            available or graph_exists, self.name, "status",
            f"graphify {'disponible' if available else 'no instalado'}; "
            f"grafo con {n_nodes} nodo(s)/{n_edges} arista(s); "
            f"{len(vaults)} bóveda(s) Obsidian; {n_cache} entrada(s) en caché.",
            data={
                "graphify_available": available,
                "graph_exists": graph_exists,
                "nodes": n_nodes,
                "edges": n_edges,
                "obsidian_vaults": [str(v.relative_to(root)) if v.is_relative_to(root) else str(v) for v in vaults],
                "cache_entries": n_cache,
                "cache_dir": str(cache_dir.relative_to(root)) if cache_dir.is_relative_to(root) else str(cache_dir),
            },
            warnings=warnings,
        )

    def setup_vault(self, *, vault_dir: str | None = None, create_if_missing: bool = True) -> AgentResult:
        """
        Detecta la bóveda de Obsidian del proyecto. Si no hay ninguna y
        ``create_if_missing`` es True, crea una en ``vault_dir`` (por defecto
        ``knowledge/``) con la estructura de árbol adaptada al grafo.

        Es la respuesta determinista a "¿existe alguna carpeta de obsidian?":
        si existe, la reutiliza; si no, la construye.
        """
        root = self.ctx.root
        existing = GraphifyTool.detect_obsidian_vaults(root)
        if existing and vault_dir is None:
            return AgentResult(
                True, self.name, "setup_vault",
                f"Ya existe {len(existing)} bóveda(s) de Obsidian: "
                f"{', '.join(str(v.relative_to(root)) for v in existing)}. "
                f"Se reutilizará(n) — no hace falta crear otra.",
                data={"vaults": [str(v) for v in existing], "created": False},
            )

        target = (root / (vault_dir or "knowledge")).resolve()
        if not create_if_missing and not (target / ".obsidian").exists():
            return AgentResult(
                False, self.name, "setup_vault",
                f"No hay bóveda en {target} y create_if_missing=False.",
            )

        # Estructura de árbol: .obsidian (marca de bóveda) + carpetas por tipo.
        # Cada nota sigue Obsidian Flavored Markdown (kepano/obsidian-skills):
        # properties en el frontmatter, callouts y wikilinks.
        (target / ".obsidian").mkdir(parents=True, exist_ok=True)
        created_dirs = []
        for folder, purpose in _VAULT_TREE.items():
            path = target / folder
            path.mkdir(parents=True, exist_ok=True)
            note = path / "README.md"
            if not note.exists():
                body = (
                    f"> [!info] {folder}\n> {purpose}\n\n"
                    f"Las notas de esta carpeta las genera `knowledge build` a "
                    f"partir del grafo de graphify. Vuelve al índice: [[MOC]]."
                )
                note.write_text(
                    GraphifyTool.obsidian_note(
                        folder, tags=["knowledge", f"knowledge/{folder}"], body=body,
                    ),
                    encoding="utf-8",
                )
            created_dirs.append(folder)

        # MOC raíz: índice de la bóveda desde el que se navega el árbol.
        moc = target / "00-index" / "MOC.md"
        if not moc.exists():
            tree_links = "\n".join(
                f"- [[{f}/README|{f}]] — {p}" for f, p in _VAULT_TREE.items()
            )
            body = (
                "> [!abstract] Mapa de contenido\n"
                "> Bóveda generada por el `knowledge` agent de dskit. El grafo de "
                "graphify se exporta aquí y se organiza en este árbol.\n\n"
                f"{tree_links}\n\n"
                "> [!tip] Poblar la bóveda\n"
                "> Ejecuta `knowledge build` para volcar el grafo actual en estas "
                "carpetas. Consulta [[Nodos del grafo]] para una vista de tabla.\n\n"
                "Convención de notas: [[obsidian-skills]]."
            )
            moc.write_text(
                GraphifyTool.obsidian_note(
                    "Mapa de contenido", tags=["knowledge", "moc"],
                    body=body, aliases=["MOC", "Índice"], cssclasses=["knowledge-moc"],
                ),
                encoding="utf-8",
            )

        # Vista de base de datos (.base) del grafo, y nota de convención.
        base_file = target / "00-index" / "Nodos del grafo.base"
        if not base_file.exists():
            base_file.write_text(GraphifyTool.knowledge_base(), encoding="utf-8")
        skills_note = target / "references" / "obsidian-skills.md"
        if not skills_note.exists():
            skills_note.write_text(
                GraphifyTool.obsidian_note(
                    "obsidian-skills", tags=["knowledge", "reference"],
                    body=(
                        "> [!quote] Convención de esta bóveda\n"
                        "> Las notas siguen [Obsidian Flavored Markdown](https://github.com/kepano/obsidian-skills) "
                        "(properties, wikilinks, embeds, callouts) y las vistas usan Obsidian Bases (`.base`).\n\n"
                        "Para editar la bóveda con un agente (Claude Code, Codex, opencode) "
                        "instala las skills:\n\n"
                        "```bash\n"
                        "npx skills add https://github.com/kepano/obsidian-skills\n"
                        "```\n"
                    ),
                ),
                encoding="utf-8",
            )

        return AgentResult(
            True, self.name, "setup_vault",
            f"Bóveda creada en {target.relative_to(root)} con {len(created_dirs)} "
            f"carpeta(s) de árbol: {', '.join(created_dirs)}.",
            data={"vault": str(target), "tree": created_dirs, "created": True},
        )

    def build(self, *, vault_dir: str | None = None, export_obsidian: bool = True) -> AgentResult:
        """
        Actualiza el grafo con graphify y, si hay bóveda, lo exporta a Obsidian.
        Fija la caché en ``graphify-out/cache/`` antes de empezar para que todo
        lo caro (resúmenes, consultas posteriores) se reutilice.
        """
        root = self.ctx.root
        if not GraphifyTool.is_available(root):
            return AgentResult(
                False, self.name, "build",
                "graphify no está instalado. Ejecuta el skill /graphify una vez primero.",
            )

        self._cache_dir()
        warnings: list[str] = []

        try:
            built = GraphifyTool.build(root)
        except FileNotFoundError as exc:
            return AgentResult(False, self.name, "build", str(exc))
        if built.returncode != 0:
            warnings.append(f"graphify build devolvió error: {built.stderr.strip()[:200]}")

        exported_to = None
        if export_obsidian:
            vaults = GraphifyTool.detect_obsidian_vaults(root)
            if vault_dir:
                vault = (root / vault_dir).resolve()
            elif vaults:
                vault = vaults[0]
            else:
                vault = None
            if vault is not None:
                try:
                    exp = GraphifyTool.export_obsidian(root, vault)
                    if exp.returncode == 0:
                        exported_to = str(vault.relative_to(root)) if vault.is_relative_to(root) else str(vault)
                    else:
                        warnings.append(f"export obsidian falló: {exp.stderr.strip()[:200]}")
                except FileNotFoundError as exc:
                    warnings.append(str(exc))
            else:
                warnings.append(
                    "No hay bóveda de Obsidian — se omitió la exportación. "
                    "Ejecuta 'knowledge setup_vault' para crear una."
                )

        graph_stats = {}
        if GraphifyTool.graph_exists(root):
            try:
                g = GraphifyTool.load_graph(root)
                graph_stats = {"nodes": len(g["nodes"]), "edges": len(g["edges"])}
            except Exception:  # noqa: BLE001
                pass

        msg = f"Grafo actualizado ({graph_stats.get('nodes', '?')} nodos)."
        if exported_to:
            msg += f" Exportado a Obsidian en {exported_to}."
        return AgentResult(
            True, self.name, "build", msg,
            data={"graph": graph_stats, "obsidian": exported_to}, warnings=warnings,
        )

    def summarize_parents(self, *, min_children: int = 3, top: int = 10, no_cache: bool = False) -> AgentResult:
        """
        Genera un resumen por cada nodo padre (hub) con:
          - cuántos hijos y de qué tipos
          - **tópicos detectados** en los labels de los hijos (palabras que
            se repiten, ej: varios hijos con "datos" o "preprocess" en su
            label indican un cluster temático)
          - la correlación entre hijos con explicación de por qué se relacionan
            (misma comunidad, mismo tipo, vecinos compartidos)

        Cacheado en graphify-out/cache/ — la clave depende del mtime de
        graph.json para que nunca sirva resúmenes obsoletos.
        """
        root = self.ctx.root
        if not GraphifyTool.graph_exists(root):
            return AgentResult(
                False, self.name, "summarize_parents",
                "No hay grafo (graphify-out/graph.json). Ejecuta 'knowledge build' primero.",
            )
        self._cache_dir()

        try:
            graph = GraphifyTool.load_graph(root)
        except Exception as exc:  # noqa: BLE001
            return AgentResult(False, self.name, "summarize_parents", f"No se pudo leer el grafo: {exc}")

        def _compute() -> list:
            return GraphifyTool.parent_summaries(graph, min_children=min_children, top=top)

        if no_cache:
            summaries = _compute()
        else:
            mtime = int(GraphifyTool.graph_json(root).stat().st_mtime)
            cache_key = f"parents_{mtime}_{min_children}_{top}"
            summaries = CacheTool.disk_cache(name=cache_key)(_compute)()

        if not summaries:
            return AgentResult(
                True, self.name, "summarize_parents",
                f"Ningún nodo tiene ≥{min_children} hijos — el grafo es plano o pequeño.",
                data=[],
            )

        lines = []
        for s in summaries[:top]:
            line = f"  • {s['label']}"
            if s.get("topics"):
                topics_str = ", ".join(f"'{t['topic']}' ({t['size']} hijos)" for t in s["topics"][:4])
                line += f"\n      tópicos: {topics_str}"
            if s.get("correlated_children"):
                top_c = s["correlated_children"][0]
                reasons = " — " + "; ".join(top_c["reasons"]) if top_c.get("reasons") else ""
                line += f"\n      correlación: '{top_c['a']}' ↔ '{top_c['b']}'{reasons}"
            lines.append(line)

        return AgentResult(
            True, self.name, "summarize_parents",
            f"{len(summaries)} nodo(s) padre resumido(s):\n" + "\n".join(lines),
            data=summaries,
        )

    def preprocess(self, *, force: bool = False) -> AgentResult:
        """
        Preprocesa el grafo enriqueciendo cada nodo con metadatos calculados:
          - ``degree``: número de conexiones del nodo
          - ``child_types``: desglose de tipos de sus vecinos directos
          - ``parent_summary``: resumen del nodo si es un hub (≥3 hijos)

        Además, precarga la caché de resúmenes de nodo padre para que las
        consultas posteriores sean instantáneas.

        Cacheado: el resultado se guarda en graphify-out/cache/ y se reusa
        mientras el mtime de graph.json no cambie.
        """
        root = self.ctx.root
        if not GraphifyTool.graph_exists(root):
            return AgentResult(
                False, self.name, "preprocess",
                "No hay grafo. Ejecuta 'knowledge build' primero.",
            )

        self._cache_dir()
        import time

        try:
            graph = GraphifyTool.load_graph(root)
        except Exception as exc:
            return AgentResult(False, self.name, "preprocess", f"No se pudo leer el grafo: {exc}")

        def _run() -> dict:
            adj = GraphifyTool._adjacency(graph)
            nodes = GraphifyTool._node_index(graph)
            enriched = 0

            for node in graph["nodes"]:
                nid = str(node.get("id"))
                neighbors = adj.get(nid, set())
                node["degree"] = len(neighbors)

                if neighbors:
                    type_counts: dict[str, int] = {}
                    for nb_id in neighbors:
                        nb = nodes.get(nb_id, {})
                        ft = str(nb.get("file_type", nb.get("type", "unknown")))
                        type_counts[ft] = type_counts.get(ft, 0) + 1
                    node["child_types"] = type_counts

                enriched += 1

            # Precargar caché de resúmenes
            _ = GraphifyTool.parent_summaries(graph)

            return {
                "nodes_enriched": enriched,
                "total_nodes": len(graph["nodes"]),
                "total_edges": len(GraphifyTool._links(graph)),
            }

        mtime = int(GraphifyTool.graph_json(root).stat().st_mtime)
        cache_key = f"preprocess_{mtime}"
        if force:
            CacheTool.clear(cache_key)
        stats = CacheTool.disk_cache(name=cache_key)(_run)()

        return AgentResult(
            True, self.name, "preprocess",
            f"Grafo preprocesado: {stats['nodes_enriched']} nodos enriquecidos "
            f"(degree, child_types). Caché de resúmenes cargada.",
            data=stats,
        )

    def clean(self, *, drop_rationale: bool = True, drop_isolated: bool = True,
              re_cluster: bool = True) -> AgentResult:
        """
        Limpia el grafo eliminando nodos que aportan ruido en vez de
        información:
          - ``rationale`` (docstrings): 441 nodos en el grafo actual que son
            fragmentos de documentación interna, no conceptos del dominio
          - aislados (grado 0): nodos sin ninguna conexión, no aportan
            estructura

        Hace backup antes de modificar y, por defecto, re-ejecuta el
        clustering tras la poda para que las comunidades reflejen solo los
        nodos significativos.

        Uso:
            knowledge clean  → poda rationale + aislados + re-cluster
            knowledge clean --drop-rationale false  → solo aislados
        """
        root = self.ctx.root
        if not GraphifyTool.graph_exists(root):
            return AgentResult(
                False, self.name, "clean",
                "No hay grafo — ejecuta 'knowledge build' primero.",
            )

        graph = GraphifyTool.load_graph(root)
        before_nodes = len(graph["nodes"])
        before_links = len(GraphifyTool._links(graph))

        node_types_to_prune: list[str] = []
        if drop_rationale:
            node_types_to_prune.append("rationale")

        pruned, stats = GraphifyTool.prune(
            graph,
            node_types=node_types_to_prune,
            drop_isolated=drop_isolated,
        )

        removed = stats["nodes_removed"]
        if removed == 0:
            return AgentResult(
                True, self.name, "clean",
                "No se encontraron nodos que podar — el grafo ya está limpio.",
                data={"removed": 0},
            )

        GraphifyTool.save_graph(root, pruned, backup=True)

        lines = [
            f"Nodos eliminados: {removed} ({stats['edges_removed']} aristas)",
        ]
        if drop_rationale:
            n_rationale = sum(1 for n in graph["nodes"]
                              if n.get("file_type", n.get("type")) == "rationale")
            lines.append(f"  rationale/docstrings: {n_rationale} eliminados")
        if stats.get("isolated_removed", 0):
            lines.append(f"  aislados (grado 0): {stats['isolated_removed']} eliminados")
        lines.append(
            f"  restantes: {stats['nodes_remaining']} nodos, {stats['edges_remaining']} aristas"
        )

        cluster_msg = ""
        if re_cluster and stats["nodes_remaining"] > 0:
            try:
                proc = GraphifyTool.run_cli(root, [str(root), "--cluster-only"])
                if proc.returncode == 0:
                    cluster_msg = " Re-clustering completado."
                else:
                    cluster_msg = f" Re-clustering falló: {proc.stderr.strip()[:100]}"
            except Exception as exc:
                cluster_msg = f" Re-clustering error: {exc}"

        # Limpiar caché porque el grafo cambió
        cache_dir = GraphifyTool.cache_dir(root)
        if cache_dir.exists():
            for f in cache_dir.glob("*.joblib"):
                f.unlink()

        message = (
            f"Grafo limpiado: {removed} nodos ruidosos eliminados "
            f"({before_nodes} → {stats['nodes_remaining']})."
            + cluster_msg
            + " Caché invalidada. Ejecuta 'knowledge cache_warmup' para recargar."
        )

        return AgentResult(
            True, self.name, "clean", message,
            data={
                "before": {"nodes": before_nodes, "links": before_links},
                "after": {"nodes": stats["nodes_remaining"], "links": stats["edges_remaining"]},
                "removed": removed,
                "backup": str(GraphifyTool.graph_json(root).with_suffix(".json.bak")),
                "re_clustered": bool(cluster_msg),
            },
        )

    def prune(
        self,
        *,
        node_types: list[str] | None = None,
        node_ids: list[str] | None = None,
        drop_isolated: bool = False,
        dry_run: bool = True,
    ) -> AgentResult:
        """
        Poda nodos del grafo por tipo (p. ej. ``references``) o por id, y
        opcionalmente los nodos que queden aislados. Por seguridad es
        ``dry_run=True`` por defecto: informa de qué quitaría sin escribir.
        Pasa ``dry_run=False`` para persistir (deja un ``graph.json.bak``).

        Se diferencia de ``clean`` en que ``clean`` es la poda con criterio
        fijo (ruido: rationale + aislados, y re-clustering); ésta es la poda
        arbitraria, para cuando sabes exactamente qué quitar.
        """
        guard = self._require_graph("prune")
        if guard:
            return guard
        if not node_types and not node_ids and not drop_isolated:
            return AgentResult(
                False, self.name, "prune",
                "Indica qué podar: node_types=['reference'], node_ids=[...] o drop_isolated=True.",
            )
        try:
            graph = GraphifyTool.load_graph(self.ctx.root)
        except Exception as exc:  # noqa: BLE001
            return AgentResult(False, self.name, "prune", f"No se pudo leer el grafo: {exc}")

        pruned, stats = GraphifyTool.prune(
            graph, node_types=node_types, node_ids=node_ids, drop_isolated=drop_isolated,
        )

        if dry_run:
            return AgentResult(
                True, self.name, "prune",
                f"[dry-run] Se quitarían {stats['nodes_removed']} nodo(s) y "
                f"{stats['edges_removed']} arista(s) "
                f"(quedarían {stats['nodes_remaining']} nodos). "
                f"Pasa dry_run=False para aplicarlo.",
                data=stats,
                warnings=["Nada escrito — esto es una simulación."],
            )

        GraphifyTool.save_graph(self.ctx.root, pruned, backup=True)
        return AgentResult(
            True, self.name, "prune",
            f"Grafo podado: -{stats['nodes_removed']} nodo(s), "
            f"-{stats['edges_removed']} arista(s). Quedan {stats['nodes_remaining']} nodos. "
            f"Backup en graph.json.bak.",
            data=stats,
        )

    def sync(self, *, vault_dir: str | None = None) -> AgentResult:
        """
        Punto de entrada único para "pon el grafo y Obsidian al día". Lo usa el
        `git` agent antes de un commit. Equivale a ``build`` pero con un mensaje
        pensado para el flujo de commit y sin fallar si no hay grafo todavía.
        """
        root = self.ctx.root
        if not GraphifyTool.graph_exists(root) and not GraphifyTool.is_available(root):
            return AgentResult(
                True, self.name, "sync",
                "No hay grafo ni graphify instalado — nada que sincronizar.",
                data={"skipped": True},
            )
        return self.build(vault_dir=vault_dir, export_obsidian=True)

    # -- caché del grafo ------------------------------------------------------
    # Absorbido del agente `cache`: la caché vive en graphify-out/cache/ y este
    # agente es el dueño de graphify-out/. Tenerla en otro agente obligaba a
    # coordinar dos dueños para el mismo directorio.
    def cache_status(self) -> AgentResult:
        """Estado de la caché: número de entradas, tamaño y antigüedad."""
        cache_dir = self._cache_dir()
        if not cache_dir.exists():
            return AgentResult(True, self.name, "cache_status",
                               "La caché está vacía (graphify-out/cache/ no existe).",
                               data={"entries": 0, "size_bytes": 0})

        import time

        entries = list(cache_dir.glob("*.joblib"))
        total_bytes = sum(f.stat().st_size for f in entries)
        now = time.time()
        oldest = min((now - f.stat().st_mtime) for f in entries) if entries else 0

        return AgentResult(
            True, self.name, "cache_status",
            f"{len(entries)} entrada(s) en caché, {_human_size(total_bytes)}, "
            f"la más antigua hace {oldest / 60:.0f} min.",
            data={
                "entries": len(entries),
                "size_bytes": total_bytes,
                "size_human": _human_size(total_bytes),
                "oldest_minutes": round(oldest / 60, 1),
                "cache_dir": str(cache_dir),
            },
        )

    def cache_clear(self, *, name: str | None = None) -> AgentResult:
        """
        Limpia la caché. ``name`` opcional: solo borra entradas de una función
        concreta (p. ej. ``parents_*``, ``preprocess_*``).
        """
        self._cache_dir()
        removed = CacheTool.clear(name=name)
        return AgentResult(
            True, self.name, "cache_clear",
            f"Entradas eliminadas: {removed}{' (todas)' if name is None else f' (filtro: {name})'}.",
            data={"removed": removed, "filter": name},
        )

    def cache_warmup(self) -> AgentResult:
        """
        Precarga todo lo cacheable:
          1. Resúmenes de nodo padre (top 20, min 2 hijos)
          2. Resúmenes de los padres grandes (≥5 hijos)
          3. Exportación a Obsidian si hay bóveda
        """
        root = self.ctx.root
        self._cache_dir()

        guard = self._require_graph("cache_warmup")
        if guard:
            return guard

        graph = GraphifyTool.load_graph(root)
        graph_stats = {"nodes": len(graph["nodes"]), "edges": len(GraphifyTool._links(graph))}
        results: list[str] = []

        try:
            summaries = GraphifyTool.parent_summaries(graph, min_children=2, top=20)
            results.append(f"{len(summaries)} resúmenes de padres cacheados")
        except Exception as exc:  # noqa: BLE001
            results.append(f"resúmenes de padres: error ({exc})")

        try:
            big_summaries = GraphifyTool.parent_summaries(graph, min_children=5, top=5)
            results.append(f"{len(big_summaries)} resúmenes grandes (≥5 hijos) precargados")
        except Exception as exc:  # noqa: BLE001
            results.append(f"resúmenes grandes: error ({exc})")

        try:
            vaults = GraphifyTool.detect_obsidian_vaults(root)
            if vaults:
                exp = GraphifyTool.export_obsidian(root, vaults[0])
                if exp.returncode == 0:
                    results.append(f"exportado a Obsidian ({vaults[0].name})")
        except Exception:  # noqa: BLE001
            pass

        cache_dir = GraphifyTool.cache_dir(root)
        n_entries = len(list(cache_dir.glob("*.joblib"))) if cache_dir.exists() else 0

        return AgentResult(
            True, self.name, "cache_warmup",
            f"Warmup completado. {n_entries} entrada(s) en caché tras:\n  "
            + "\n  ".join(results),
            data={"graph": graph_stats, "cache_entries": n_entries},
        )

    def graph_stats(self) -> AgentResult:
        """Estadísticas rápidas del grafo (desde caché si está preprocesado)."""
        root = self.ctx.root
        guard = self._require_graph("graph_stats")
        if guard:
            return guard

        graph = GraphifyTool.load_graph(root)
        adj = GraphifyTool._adjacency(graph)
        nodes = GraphifyTool._node_index(graph)

        degrees = [len(adj.get(nid, set())) for nid in nodes]
        avg_degree = sum(degrees) / len(degrees) if degrees else 0
        max_degree = max(degrees) if degrees else 0

        type_counts: dict[str, int] = {}
        for n in graph["nodes"]:
            ft = str(n.get("file_type", n.get("type", "unknown")))
            type_counts[ft] = type_counts.get(ft, 0) + 1

        communities = len({n.get("community") for n in graph["nodes"] if n.get("community") is not None})
        orphans = sum(1 for d in degrees if d == 0)

        return AgentResult(
            True, self.name, "graph_stats",
            f"{len(nodes)} nodos, {len(GraphifyTool._links(graph))} aristas, "
            f"{communities} comunidades, {orphans} aislados. "
            f"Grado medio: {avg_degree:.1f}, máximo: {max_degree}.",
            data={
                "nodes": len(nodes),
                "edges": len(GraphifyTool._links(graph)),
                "communities": communities,
                "orphans": orphans,
                "avg_degree": round(avg_degree, 1),
                "max_degree": max_degree,
                "types": type_counts,
            },
        )


def _human_size(bytes_: int) -> str:
    for unit in ("B", "KB", "MB"):
        if bytes_ < 1024:
            return f"{bytes_:.1f} {unit}"
        bytes_ /= 1024
    return f"{bytes_:.1f} GB"
