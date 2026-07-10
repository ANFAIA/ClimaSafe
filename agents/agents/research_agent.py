"""
agents.agents.research_agent — Busca papers académicos relacionados con el
proyecto.

Deriva las palabras clave del proyecto (README, descripción de pyproject.toml
y, si existe, las etiquetas de los nodos más centrales del grafo de graphify) y
busca papers relevantes en fuentes abiertas (arXiv, OpenAlex) vía
`research_tool`. Todo cacheado en ``graphify-out/cache/``.

Este agente es un "worker" pensado también para competir bajo el
`supervisor` agent: cada backend (arxiv / openalex) es una estrategia distinta
que puede proponer su propia lista de papers.

Límite honesto: la relevancia es léxica (solapamiento de keywords con
título+abstract), no una lectura semántica del paper. Requiere conexión a
internet; sin ella cada búsqueda falla de forma controlada, no rompe el agente.
"""

from __future__ import annotations

from agents.core.base_agent import AgentResult, BaseAgent
from agents.core.registry import register_agent
from agents.exceptions import ToolExecutionError
from agents.tools.cache_tool import CacheTool
from agents.tools.graphify_tool import GraphifyTool
from agents.tools.research_tool import ResearchTool

_BACKENDS = {
    "arxiv": ResearchTool.search_arxiv,
    "openalex": ResearchTool.search_openalex,
}


@register_agent
class ResearchAgent(BaseAgent):
    name = "research"
    description = (
        "Busca papers académicos relacionados con el proyecto en fuentes "
        "abiertas (arXiv, OpenAlex): deriva las palabras clave del proyecto y "
        "devuelve una lista de papers rankeada por relevancia."
    )
    capabilities = [
        "research", "papers", "paper", "articulos", "artículos", "bibliografia",
        "bibliografía", "investigacion", "investigación", "arxiv", "openalex",
        "estado del arte", "literatura", "citas",
    ]

    def action_aliases(self) -> dict:
        return {
            "project_keywords": ["keywords", "palabras clave", "temas", "de que trata"],
            "find_papers": ["papers", "busca papers", "relacionados", "estado del arte", "encuentra"],
            "search": ["busca", "buscar", "consulta"],
        }

    def actions(self) -> dict:
        return {
            "project_keywords": self.project_keywords,
            "find_papers": self.find_papers,
            "search": self.search,
        }

    # -------------------------------------------------------------------------
    def _cache_dir(self) -> None:
        CacheTool.set_cache_dir(GraphifyTool.cache_dir(self.ctx.root))

    def _gather_project_text(self) -> str:
        """Junta el texto del que se extraen las keywords: README + pyproject + grafo."""
        parts: list[str] = []
        for path in (self.ctx.readme_file, self.ctx.pyproject_file):
            if path.exists():
                try:
                    parts.append(path.read_text(encoding="utf-8", errors="ignore"))
                except OSError:
                    pass
        # Etiquetas de los nodos más centrales del grafo, si graphify ya corrió.
        if GraphifyTool.graph_exists(self.ctx.root):
            try:
                graph = GraphifyTool.load_graph(self.ctx.root)
                adj = GraphifyTool._adjacency(graph)
                nodes = GraphifyTool._node_index(graph)
                top = sorted(adj, key=lambda n: len(adj[n]), reverse=True)[:20]
                parts.extend(str(nodes.get(nid, {}).get("label", "")) for nid in top)
            except Exception:  # noqa: BLE001 — grafo ausente/corrupto no debe romper esto
                pass
        return "\n".join(parts)

    def project_keywords(self, *, top: int = 12) -> AgentResult:
        """Extrae las palabras clave que describen el proyecto."""
        text = self._gather_project_text()
        if not text.strip():
            return AgentResult(
                False, self.name, "project_keywords",
                "No encontré README ni pyproject.toml de los que extraer keywords.",
            )
        keywords = ResearchTool.extract_keywords(text, top=top)
        return AgentResult(
            True, self.name, "project_keywords",
            f"{len(keywords)} palabra(s) clave del proyecto: {', '.join(keywords)}",
            data={"keywords": keywords},
        )

    def search(self, *, query: str, backend: str = "openalex", max_results: int = 10,
               no_cache: bool = False) -> AgentResult:
        """Busca papers para una consulta concreta en un backend (arxiv|openalex)."""
        if backend not in _BACKENDS:
            return AgentResult(
                False, self.name, "search",
                f"Backend '{backend}' desconocido. Disponibles: {sorted(_BACKENDS)}.",
            )
        self._cache_dir()
        search_fn = _BACKENDS[backend]

        def _run() -> list:
            return search_fn(query, max_results=max_results)

        try:
            if no_cache:
                papers = _run()
            else:
                papers = CacheTool.disk_cache(name=f"research_{backend}_{query}_{max_results}")(_run)()
        except ToolExecutionError as exc:
            return AgentResult(False, self.name, "search", f"Búsqueda en {backend} falló (¿sin red?): {exc}")

        keywords = ResearchTool.extract_keywords(query, top=8) or query.split()
        ranked = ResearchTool.rank(ResearchTool.dedupe(papers), keywords)
        return AgentResult(
            True, self.name, "search",
            f"{len(ranked)} paper(s) en {backend} para '{query}'.",
            data={"backend": backend, "query": query, "papers": ranked},
        )

    def find_papers(self, *, backend: str = "openalex", max_results: int = 10,
                    top_keywords: int = 8) -> AgentResult:
        """
        Busca papers relacionados con el PROYECTO: deriva sus keywords y consulta
        el backend con ellas. Es la acción principal del agente.
        """
        kw_result = self.project_keywords(top=top_keywords)
        if not kw_result.success:
            return kw_result
        keywords = kw_result.data["keywords"]
        query = " ".join(keywords)

        search_result = self.search(query=query, backend=backend, max_results=max_results)
        if not search_result.success:
            return search_result

        papers = ResearchTool.rank(search_result.data["papers"], keywords)
        return AgentResult(
            True, self.name, "find_papers",
            f"{len(papers)} paper(s) relacionados con el proyecto (backend {backend}).",
            data={"backend": backend, "keywords": keywords, "papers": papers},
        )
