"""
agents.agents.docsearch_agent — Búsqueda y navegación por el grafo de docs.

Complementa al `knowledge` agent: mientras aquél construye y resume el grafo,
éste lo NAVEGA. Permite:

  - buscar en la documentación en lenguaje natural (delegando en
    ``graphify query``, con caché),
  - listar los vecinos de un nodo (navegación por el árbol de conocimiento),
  - listar y podar referencias/nodos innecesarios del grafo — "quitar
    información, referencias y lo necesario" para mantenerlo limpio.

La poda escribe el grafo de vuelta dejando un backup ``graph.json.bak``; nunca
toca los archivos fuente del proyecto, solo la representación en el grafo.
"""

from __future__ import annotations

import hashlib

from agents.core.base_agent import AgentResult, BaseAgent
from agents.core.registry import register_agent
from agents.tools.cache_tool import CacheTool
from agents.tools.graphify_tool import GraphifyTool


@register_agent
class DocSearchAgent(BaseAgent):
    name = "docsearch"
    description = (
        "Busca y navega la documentación a través del grafo de conocimiento: "
        "consultas en lenguaje natural, vecinos de un nodo, y poda de "
        "referencias o nodos innecesarios."
    )
    capabilities = [
        "buscar", "busca", "search", "navegar",
        "referencias", "podar", "prune", "limpiar", "consulta", "query", "vecinos",
    ]

    def action_aliases(self) -> dict:
        return {
            "search": ["busca", "buscar", "consulta", "query", "encuentra"],
            "neighbors": ["vecinos", "navega", "conecta", "relacionado"],
            "list_references": ["referencias", "citas", "enlaces"],
            "prune": ["poda", "podar", "quita", "elimina", "limpia", "borra"],
        }

    def actions(self) -> dict:
        return {
            "search": self.search,
            "neighbors": self.neighbors,
            "list_references": self.list_references,
            "prune": self.prune,
        }

    def _require_graph(self, action: str) -> AgentResult | None:
        if not GraphifyTool.graph_exists(self.ctx.root):
            return AgentResult(
                False, self.name, action,
                "No hay grafo (graphify-out/graph.json). Ejecuta 'knowledge build' primero.",
            )
        return None

    # -------------------------------------------------------------------------
    def search(self, *, question: str, budget: int | None = None, no_cache: bool = False) -> AgentResult:
        """Consulta la documentación en lenguaje natural vía ``graphify query`` (cacheado)."""
        guard = self._require_graph("search")
        if guard:
            return guard
        if not GraphifyTool.is_available(self.ctx.root):
            return AgentResult(
                False, self.name, "search",
                "graphify no está instalado — no se puede consultar. Ejecuta el skill /graphify.",
            )
        CacheTool.set_cache_dir(GraphifyTool.cache_dir(self.ctx.root))

        def _run() -> str:
            proc = GraphifyTool.query(self.ctx.root, question, budget=budget)
            if proc.returncode != 0:
                # Lanza para NO cachear el fallo (un error transitorio de la
                # consulta no debe quedar cacheado para siempre). El caller lo
                # convierte en un AgentResult(success=False).
                raise RuntimeError(proc.stderr.strip()[:200] or "graphify query devolvió error")
            return proc.stdout.strip()

        try:
            if no_cache:
                answer = _run()
            else:
                # Clave estable con hashlib: hash() lleva PYTHONHASHSEED y cambia
                # entre procesos, así que la caché en disco nunca acertaría entre
                # invocaciones de la CLI.
                digest = hashlib.md5(f"{question}|{budget}".encode()).hexdigest()[:16]
                answer = CacheTool.disk_cache(name=f"query_{digest}")(_run)()
        except RuntimeError as exc:
            return AgentResult(False, self.name, "search", f"graphify query falló: {exc}")

        return AgentResult(
            True, self.name, "search",
            answer or "graphify query no devolvió texto.",
            data={"question": question, "answer": answer},
        )

    def neighbors(self, *, node: str, limit: int = 20) -> AgentResult:
        """
        Lista los nodos vecinos de ``node`` (por id o por label, insensible a
        mayúsculas) — un paso de navegación por el árbol de conocimiento.
        """
        guard = self._require_graph("neighbors")
        if guard:
            return guard
        try:
            graph = GraphifyTool.load_graph(self.ctx.root)
        except Exception as exc:  # noqa: BLE001
            return AgentResult(False, self.name, "neighbors", f"No se pudo leer el grafo: {exc}")

        nodes = GraphifyTool._node_index(graph)
        target_id = node if node in nodes else None
        if target_id is None:
            wanted = node.lower()
            for nid, n in nodes.items():
                if str(n.get("label", "")).lower() == wanted:
                    target_id = nid
                    break
        if target_id is None:
            return AgentResult(
                False, self.name, "neighbors",
                f"No hay ningún nodo con id o label '{node}'.",
            )

        adj = GraphifyTool._adjacency(graph)
        neighbor_ids = sorted(adj.get(target_id, set()))
        neighbors = [
            {"id": nid, "label": nodes.get(nid, {}).get("label", nid),
             "type": nodes.get(nid, {}).get("type", "desconocido")}
            for nid in neighbor_ids[:limit]
        ]
        return AgentResult(
            True, self.name, "neighbors",
            f"'{nodes.get(target_id, {}).get('label', target_id)}' tiene "
            f"{len(neighbor_ids)} vecino(s)"
            + (f" (mostrando {limit})" if len(neighbor_ids) > limit else "") + ".",
            data={"node": target_id, "neighbors": neighbors, "total": len(neighbor_ids)},
        )

    def list_references(self) -> AgentResult:
        """Lista los nodos de tipo 'reference' (referencias/citas externas)."""
        guard = self._require_graph("list_references")
        if guard:
            return guard
        try:
            graph = GraphifyTool.load_graph(self.ctx.root)
        except Exception as exc:  # noqa: BLE001
            return AgentResult(False, self.name, "list_references", f"No se pudo leer el grafo: {exc}")

        refs = [
            {"id": str(n.get("id")), "label": n.get("label", n.get("id"))}
            for n in graph["nodes"]
            if str(n.get("type", "")).lower() in {"reference", "citation", "link", "url"}
        ]
        return AgentResult(
            True, self.name, "list_references",
            f"{len(refs)} referencia(s) en el grafo.",
            data=refs,
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
