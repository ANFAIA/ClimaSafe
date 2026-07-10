"""
agents.agents.cache_agent — Agente que cachea al máximo las operaciones del grafo.

Su propósito es mantener todo lo caro (resúmenes de nodo padre, consultas al
grafo, exportaciones a Obsidian) precargado en ``graphify-out/cache/`` para
que las respuestas sean instantáneas.

Flujo recomendado tras un build/update del grafo:
    1. cache warmup   → precarga resúmenes de padres + preprocesado
    2. cache status   → ver qué hay en caché
    3. cache clear    → si algo cambió manualmente en el grafo
"""

from __future__ import annotations

from pathlib import Path

from agents import audit
from agents.core.base_agent import AgentResult, BaseAgent
from agents.core.registry import register_agent
from agents.tools.cache_tool import CacheTool
from agents.tools.graphify_tool import GraphifyTool


@register_agent
class CacheAgent(BaseAgent):
    name = "cache"
    description = (
        "Gestiona la caché del grafo de conocimiento: precarga resúmenes, "
        "consulta el estado de la caché, limpia entradas obsoletas."
    )
    capabilities = [
        "cache", "caché", "cachear", "precarga", "warmup", "precalentar",
        "limpiar cache", "estado de cache",
    ]

    def action_aliases(self) -> dict:
        return {
            "warmup": ["precarga", "precalienta", "cachea todo", "prepara cache"],
            "status": ["estado", "situacion", "tamano", "cuantas entradas"],
            "clear": ["limpia", "borra", "vacia", "purga", "resetea"],
            "graph_stats": ["grafo", "estadisticas del grafo", "nodos", "aristas"],
        }

    def actions(self) -> dict:
        return {
            "warmup": self.warmup,
            "status": self.status,
            "clear": self.clear,
            "graph_stats": self.graph_stats,
        }

    def _cache_dir(self) -> Path:
        cache_dir = GraphifyTool.cache_dir(self.ctx.root)
        CacheTool.set_cache_dir(cache_dir)
        return cache_dir

    # -------------------------------------------------------------------------

    def status(self) -> AgentResult:
        """Muestra el estado de la caché: número de entradas, tamaño, antigüedad."""
        cache_dir = self._cache_dir()
        if not cache_dir.exists():
            return AgentResult(True, self.name, "status",
                               "La caché está vacía (graphify-out/cache/ no existe).",
                               data={"entries": 0, "size_bytes": 0})

        entries = list(cache_dir.glob("*.joblib"))
        total_bytes = sum(f.stat().st_size for f in entries)
        import time
        now = time.time()
        oldest = min((now - f.stat().st_mtime) for f in entries) if entries else 0

        return AgentResult(
            True, self.name, "status",
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

    def clear(self, *, name: str | None = None) -> AgentResult:
        """
        Limpia la caché. ``name`` opcional: solo borra entradas de una función
        concreta (p. ej. ``parents_*``, ``preprocess_*``).
        """
        removed = CacheTool.clear(name=name)
        return AgentResult(
            True, self.name, "clear",
            f"Entradas eliminadas: {removed}{' (todas)' if name is None else f' (filtro: {name})'}.",
            data={"removed": removed, "filter": name},
        )

    def warmup(self) -> AgentResult:
        """
        Precarga todo lo cacheable:
          1. Resúmenes de nodo padre (top 20, min 2 hijos)
          2. Preprocesado del grafo (degree, child_types)
          3. Exportación a Obsidian si hay bóveda
        """
        root = self.ctx.root
        self._cache_dir()

        if not GraphifyTool.graph_exists(root):
            return AgentResult(
                False, self.name, "warmup",
                "No hay grafo. Ejecuta 'knowledge build' primero.",
            )

        audit.record(self.ctx, agent=self.name, action="warmup", success=True,
                     duration_ms=0, message="Iniciando warmup de caché...")

        graph = GraphifyTool.load_graph(root)
        graph_stats = {"nodes": len(graph["nodes"]), "edges": len(GraphifyTool._links(graph))}
        results: list[str] = []

        # 1. Resúmenes de padres (top 20, min 2 hijos)
        try:
            summaries = GraphifyTool.parent_summaries(graph, min_children=2, top=20)
            results.append(f"{len(summaries)} resúmenes de padres cacheados")
        except Exception as exc:
            results.append(f"resúmenes de padres: error ({exc})")

        # 2. Precarga de tópicos y correlaciones para padres grandes
        try:
            big_summaries = GraphifyTool.parent_summaries(graph, min_children=5, top=5)
            results.append(f"{len(big_summaries)} resúmenes grandes (≥5 hijos) precargados")
        except Exception as exc:
            results.append(f"resúmenes grandes: error ({exc})")

        # 3. Exportar a Obsidian si hay bóveda
        try:
            vaults = GraphifyTool.detect_obsidian_vaults(root)
            if vaults:
                exp = GraphifyTool.export_obsidian(root, vaults[0])
                if exp.returncode == 0:
                    results.append(f"exportado a Obsidian ({vaults[0].name})")
        except Exception:
            pass

        # Contar entradas finales
        cache_dir = GraphifyTool.cache_dir(root)
        n_entries = len(list(cache_dir.glob("*.joblib"))) if cache_dir.exists() else 0

        audit.record(self.ctx, agent=self.name, action="warmup", success=True,
                     duration_ms=0,
                     message=f"Warmup completado: {n_entries} entradas en caché",
                     kwarg_names=[])

        return AgentResult(
            True, self.name, "warmup",
            f"Warmup completado. {n_entries} entrada(s) en caché tras:\n  "
            + "\n  ".join(results),
            data={"graph": graph_stats, "cache_entries": n_entries},
        )

    def graph_stats(self) -> AgentResult:
        """Estadísticas rápidas del grafo (todo desde caché si está preprocesado)."""
        root = self.ctx.root
        if not GraphifyTool.graph_exists(root):
            return AgentResult(
                False, self.name, "graph_stats",
                "No hay grafo — ejecuta 'knowledge build' primero.",
            )

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
