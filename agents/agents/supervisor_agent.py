"""
agents.agents.supervisor_agent — Coordina "workers" que compiten.

Implementa el patrón que pediste: varios agentes trabajan POR SEPARADO sobre la
misma tarea, cada uno entrega una propuesta, el supervisor las PRUEBA (las
puntúa con una métrica), ELIGE la mejor y la PULE.

Dos acciones:

  - ``research``: competición concreta de búsqueda de papers. Lanza el
    `research` agent con cada backend (arXiv, OpenAlex) en paralelo — cada
    backend es un worker independiente —, puntúa cada propuesta por relevancia,
    cobertura de keywords y volumen, elige la ganadora y la pule fusionando lo
    mejor de ambas (dedupe + re-ranking).

  - ``compete``: versión genérica. Recibe varios candidatos
    ``{agent, action, kwargs}``, los ejecuta por separado, puntúa cada
    resultado con una heurística y devuelve el ranking + el ganador.

Coherente con la filosofía del sistema (ver `agents/README.md`): el arbitraje
es DETERMINISTA (una métrica explícita), no un juez LLM — así no se ata a
ningún proveedor de IA. La métrica es un punto de extensión claro si mañana
quieres un juez más sofisticado.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

from agents.core.base_agent import AgentResult, BaseAgent
from agents.core.registry import register_agent
from agents.tools.research_tool import ResearchTool

_RESEARCH_BACKENDS = ("arxiv", "openalex")


@register_agent
class SupervisorAgent(BaseAgent):
    name = "supervisor"
    description = (
        "Coordina workers que compiten: los hace trabajar por separado, prueba "
        "cada propuesta con una métrica, elige la mejor y la pule. Especializado "
        "en competición de búsqueda de papers (research) y genérico (compete)."
    )
    capabilities = [
        "supervisor", "supervisa", "coordina", "competir", "competición",
        "competicion", "propuesta", "arbitro", "árbitro", "elige", "compara",
        "evalua", "evalúa", "mejor",
    ]

    def action_aliases(self) -> dict:
        return {
            "research": ["papers", "investiga", "estado del arte", "busca papers"],
            "compete": ["compite", "compara", "enfrenta", "propuestas"],
        }

    def actions(self) -> dict:
        return {
            "research": self.research,
            "compete": self.compete,
        }

    # -- competición de research ---------------------------------------------
    def research(self, *, max_results: int = 10, top_keywords: int = 8,
                 backends: list[str] | None = None) -> AgentResult:
        """
        Cada backend de `research` es un worker. Corren en paralelo, cada uno
        propone su lista de papers, el supervisor las puntúa, elige la mejor y
        pule fusionando ambas.
        """
        from agents.agents.research_agent import ResearchAgent

        chosen = [b for b in (backends or _RESEARCH_BACKENDS) if b in _RESEARCH_BACKENDS]
        if not chosen:
            return AgentResult(
                False, self.name, "research",
                f"Ningún backend válido. Disponibles: {list(_RESEARCH_BACKENDS)}.",
            )

        research_agent = ResearchAgent(context=self.ctx)
        kw_result = research_agent.project_keywords(top=top_keywords)
        if not kw_result.success:
            return kw_result
        keywords = kw_result.data["keywords"]
        query = " ".join(keywords)

        # Workers separados, en paralelo (I/O de red → hilos).
        def _worker(backend: str) -> dict:
            res = research_agent.search(query=query, backend=backend, max_results=max_results)
            papers = res.data.get("papers", []) if res.success else []
            return {
                "backend": backend,
                "success": res.success,
                "message": res.message,
                "papers": papers,
                "score": self._score_research(papers, keywords, max_results),
            }

        with ThreadPoolExecutor(max_workers=len(chosen)) as pool:
            proposals = list(pool.map(_worker, chosen))

        proposals.sort(key=lambda p: -p["score"])
        ok = [p for p in proposals if p["success"]]
        if not ok:
            return AgentResult(
                False, self.name, "research",
                "Ningún worker devolvió resultados (¿sin red?). "
                + "; ".join(f"{p['backend']}: {p['message']}" for p in proposals),
                data={"proposals": proposals},
            )

        winner = ok[0]
        # Pulido: fusiona TODAS las propuestas, dedupe y re-rank por relevancia.
        merged = [p for prop in ok for p in prop["papers"]]
        polished = ResearchTool.rank(ResearchTool.dedupe(merged), keywords)[:max_results]

        scoreboard = ", ".join(f"{p['backend']}={p['score']}" for p in proposals)
        return AgentResult(
            True, self.name, "research",
            f"Ganador: '{winner['backend']}' ({scoreboard}). "
            f"Pulido: {len(polished)} paper(s) fusionando {len(ok)} propuesta(s).",
            data={
                "winner": winner["backend"],
                "keywords": keywords,
                "scoreboard": {p["backend"]: p["score"] for p in proposals},
                "polished": polished,
            },
        )

    @staticmethod
    def _score_research(papers: list[dict], keywords: list[str], max_results: int) -> float:
        """
        Métrica de una propuesta de papers: 0.5·relevancia media + 0.4·cobertura
        de keywords + 0.1·volumen. Determinista y explícita.
        """
        if not papers:
            return 0.0
        rels = [ResearchTool.relevance(p, keywords) for p in papers]
        mean_rel = sum(rels) / len(rels)
        covered = set()
        for p in papers:
            text = f"{p.get('title', '')} {p.get('abstract', '')}".lower()
            covered |= {kw.lower() for kw in keywords if kw.lower() in text}
        coverage = len(covered) / len(keywords) if keywords else 0.0
        volume = min(1.0, len(papers) / max(1, max_results))
        return round(0.5 * mean_rel + 0.4 * coverage + 0.1 * volume, 4)

    # -- competición genérica -------------------------------------------------
    def compete(self, *, candidates: list[dict], parallel: bool = False) -> AgentResult:
        """
        Enfrenta candidatos arbitrarios. Cada candidato es un dict
        ``{"agent": str, "action": str, "kwargs": dict, "label": str?}``. Los
        ejecuta por separado, puntúa cada AgentResult con una heurística
        (`_default_score`) y devuelve el ranking + el ganador.

        `parallel=True` los corre en hilos (útil si son tareas de I/O; no todas
        las acciones son thread-safe, por eso el defecto es secuencial).
        """
        if not candidates:
            return AgentResult(False, self.name, "compete", "No se pasaron candidatos.")

        from agents.orchestrator import Orchestrator
        orch = Orchestrator(context=self.ctx)

        def _run(cand: dict) -> dict:
            label = cand.get("label") or f"{cand.get('agent')}.{cand.get('action')}"
            res = orch.run(cand["agent"], cand["action"], **cand.get("kwargs", {}))
            return {"label": label, "success": res.success, "message": res.message,
                    "score": self._default_score(res), "result": res}

        if parallel:
            with ThreadPoolExecutor(max_workers=min(len(candidates), 8)) as pool:
                proposals = list(pool.map(_run, candidates))
        else:
            proposals = [_run(c) for c in candidates]

        proposals.sort(key=lambda p: -p["score"])
        winner = proposals[0]
        board = ", ".join(f"{p['label']}={p['score']}" for p in proposals)
        return AgentResult(
            winner["success"], self.name, "compete",
            f"Ganador: '{winner['label']}' ({board}).",
            data={
                "winner": winner["label"],
                "scoreboard": {p["label"]: p["score"] for p in proposals},
                "proposals": [{k: v for k, v in p.items() if k != "result"} for p in proposals],
            },
        )

    @staticmethod
    def _default_score(res: AgentResult) -> float:
        """
        Heurística genérica para puntuar un AgentResult: premia el éxito, la
        riqueza de `data` y penaliza los warnings. Es deliberadamente simple —
        el punto de extensión para una métrica por dominio.
        """
        if not res.success:
            return 0.0
        data = res.data
        size = len(data) if isinstance(data, (list, dict, str)) else 1
        return round(1.0 + min(1.0, size / 50.0) - 0.1 * len(res.warnings), 4)
