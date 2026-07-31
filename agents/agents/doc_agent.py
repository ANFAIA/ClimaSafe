"""
agents.agents.doc_agent — Documentación unificada del proyecto.

Combina tres fuentes de conocimiento:
1. Graphify → grafo estructural (dependencias, nodos, relaciones)
2. RAG → búsqueda semántica vectorial (ChromaDB)
3. Vault → búsqueda textual directa en markdown de Obsidian
"""

from __future__ import annotations


import hashlib

from agents.core.base_agent import AgentResult, BaseAgent
from agents.core.registry import register_agent
from agents.tools.cache_tool import CacheTool

try:
    from agents.tools.graphify_tool import GraphifyTool
    HAS_GRAPHIFY = True
except ImportError:
    HAS_GRAPHIFY = False

try:
    from agents.tools.rag_tool import RagTool
    HAS_RAG = True
except ImportError:
    HAS_RAG = False


@register_agent
class DocAgent(BaseAgent):
    name = "doc"
    description = (
        "Documentación unificada del proyecto: busca en el grafo graphify "
        "(estructura), el índice RAG (semántica) y el vault Obsidian (notas)."
    )
    # Es el punto de entrada UNIFICADO: sus palabras son las que expresan
    # "búscalo donde sea", no los nombres de las fuentes concretas — esos
    # pertenecen a `knowledge`, dueño del grafo (un keyword, un dueño).
    capabilities = [
        "doc", "todas las fuentes", "busqueda unificada", "búsqueda unificada",
        "documentacion unificada", "donde esta documentado",
        "dónde está documentado", "que hace", "qué hace",
        "como funciona", "cómo funciona", "explica", "informacion", "información",
        # Absorbidas de `docsearch`: navegacion del grafo.
        "navegar", "navega el grafo", "busca en el grafo", "vecinos",
        "referencias", "busca en el grafo de conocimiento",
    ]

    def action_aliases(self) -> dict:
        return {
            "search": ["busca", "consulta", "encuentra", "informacion", "documentacion"],
            "graph_query": ["grafo", "graph", "estructura", "dependencias", "nodos"],
            "rag_search": ["semantico", "semantic", "vectorial", "embedding"],
            "vault_grep": ["vault", "obsidian", "grep", "texto en vault"],
            "index": ["indexar", "construye indice", "reindexa todo"],
            "status": ["estado", "fuentes", "disponible"],
        }

    def actions(self) -> dict:
        return {
            "search": self.search,
            "graph_query": self.graph_query,
            "rag_search": self.rag_search,
            "vault_grep": self.vault_grep,
            "neighbors": self.neighbors,
            "list_references": self.list_references,
            "index": self.index,
            "status": self.status,
        }

    def _buscar_grafo(self, query: str, avisos: list) -> list[dict]:
        """Nodos del grafo cuyo label o id contienen la consulta."""
        if not (HAS_GRAPHIFY and GraphifyTool.graph_exists(self.ctx.root)):
            return []
        try:
            grafo = GraphifyTool.load_graph(self.ctx.root)
        except Exception:  # noqa: BLE001
            avisos.append("error al leer el grafo graphify")
            return []
        q = query.lower()
        encontrados = [
            n for n in grafo.get("nodes", [])
            if q in n.get("label", "").lower() or q in n.get("id", "").lower()
        ]
        return [
            {
                # Ojo: la clave es `source_type`, no `source`. Antes se
                # escribia `source` y el formateo de abajo comprueba
                # `source_type`, asi que los resultados del grafo se contaban
                # pero no se imprimian nunca.
                "source_type": "graphify",
                "label": n.get("label", n.get("id", "?")),
                "type": n.get("type", "node"),
            }
            for n in encontrados[:10]
        ]

    def _buscar_rag(self, query: str, avisos: list) -> list[dict]:
        """Coincidencias semanticas del indice vectorial."""
        if not (HAS_RAG and RagTool.available()):
            return []
        try:
            resultados = RagTool.search(self.ctx.root, query, top_k=5)
        except Exception:  # noqa: BLE001
            avisos.append("error al consultar RAG")
            return []
        for r in resultados:
            r["source_type"] = "rag"
        return resultados

    def _buscar_vault(self, query: str, avisos: list) -> list[dict]:
        """Grep literal sobre las notas del vault."""
        vault = self.ctx.root / "vault"
        if not vault.exists():
            return []
        q = query.lower()
        encontrados = []
        try:
            for md in vault.rglob("*.md"):
                texto = md.read_text(encoding="utf-8", errors="replace")
                if q not in texto.lower():
                    continue
                for numero, linea in enumerate(texto.split("\n"), 1):
                    if q in linea.lower():
                        encontrados.append({
                            "source_type": "vault",
                            "file": str(md.relative_to(self.ctx.root)),
                            "line": numero,
                            "text": linea.strip()[:200],
                        })
        except Exception:  # noqa: BLE001
            avisos.append("error al leer vault")
        return encontrados

    @staticmethod
    def _formatear(resultado: dict) -> str:
        tipo = resultado.get("source_type")
        if tipo == "graphify":
            return f"  [graphify] {resultado['label']} ({resultado['type']})"
        if tipo == "rag":
            return (f"  [rag] {resultado.get('source', '?')}:{resultado.get('line', '?')}"
                    f" — {resultado.get('text', '')[:100]}")
        return f"  [vault] {resultado['file']}:{resultado['line']} — {resultado['text']}"

    def search(self, *, query: str, sources: str = "all") -> AgentResult:
        """Busca en todas las fuentes disponibles y funde los resultados."""
        if not query.strip():
            return AgentResult(False, self.name, "search", "Proporciona una consulta.")

        avisos: list[str] = []
        buscadores = {
            "graph": self._buscar_grafo,
            "rag": self._buscar_rag,
            "vault": self._buscar_vault,
        }
        combinados: list[dict] = []
        for nombre, buscador in buscadores.items():
            if sources in ("all", nombre):
                combinados.extend(buscador(query, avisos))

        if not combinados:
            return AgentResult(
                True, self.name, "search",
                "No se encontraron resultados en ninguna fuente.",
                data=[], warnings=avisos,
            )

        fuentes = len({r.get("source_type", "") for r in combinados})
        lineas = [self._formatear(r) for r in combinados[:10]]
        return AgentResult(
            True, self.name, "search",
            f"{len(combinados)} resultado(s) de {fuentes} fuente(s).\n" + "\n".join(lineas),
            data=combinados, warnings=avisos,
        )

    def graph_query(self, *, question: str, budget: int | None = None,
                    no_cache: bool = False) -> AgentResult:
        """
        Consulta el grafo graphify en lenguaje natural (cacheado).

        Implementacion traida de `docsearch` al absorberlo: la que habia aqui
        cacheaba tambien los fallos, asi que un error transitorio de graphify
        quedaba servido desde cache para siempre.
        """
        guard = self._require_graph("graph_query")
        if guard:
            return guard
        if not GraphifyTool.is_available(self.ctx.root):
            return AgentResult(
                False, self.name, "graph_query",
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
            return AgentResult(False, self.name, "graph_query", f"graphify query falló: {exc}")

        return AgentResult(
            True, self.name, "graph_query",
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

    def _require_graph(self, action: str) -> AgentResult | None:
        if not GraphifyTool.graph_exists(self.ctx.root):
            return AgentResult(
                False, self.name, action,
                "No hay grafo (graphify-out/graph.json). Ejecuta 'knowledge build' primero.",
            )
        return None

    # -------------------------------------------------------------------------
    def rag_search(self, *, query: str, top_k: int = 10) -> AgentResult:
        """Búsqueda semántica pura vía RAG."""
        if not HAS_RAG:
            return AgentResult(
                False, self.name, "rag_search",
                "RAG no disponible (chromadb no instalado).",
            )
        if not RagTool.available():
            return AgentResult(
                False, self.name, "rag_search",
                "chromadb no instalado. Ejecuta: uv sync --extra rag",
            )
        results = RagTool.search(self.ctx.root, query, top_k=top_k)
        if not results:
            return AgentResult(
                True, self.name, "rag_search",
                "No hay resultados. Ejecuta 'rag index' primero.",
                data=[],
            )
        if "error" in results[0]:
            return AgentResult(False, self.name, "rag_search", results[0]["error"])
        lines = []
        for r in results[:5]:
            lines.append(f"  [{r['score']}] {r['source']}:{r['line']} — {r['text'][:120]}")
        return AgentResult(
            True, self.name, "rag_search",
            f"{len(results)} resultado(s). Top:\n" + "\n".join(lines),
            data=results,
        )

    def vault_grep(self, *, pattern: str) -> AgentResult:
        """Busca texto directamente en el vault Obsidian."""
        vault_path = self.ctx.root / "vault"
        if not vault_path.exists():
            return AgentResult(
                False, self.name, "vault_grep",
                "No hay directorio vault/ en el proyecto.",
            )
        matches = []
        for md_file in vault_path.rglob("*.md"):
            try:
                text = md_file.read_text(encoding="utf-8", errors="replace")
                for lineno, line in enumerate(text.split("\n"), 1):
                    if pattern.lower() in line.lower():
                        matches.append({
                            "file": str(md_file.relative_to(self.ctx.root)),
                            "line": lineno,
                            "text": line.strip()[:200],
                        })
            except Exception:
                pass
        if not matches:
            return AgentResult(
                True, self.name, "vault_grep",
                f"No se encontró '{pattern}' en vault/.",
                data=[],
            )
        lines = [f"  {m['file']}:{m['line']} — {m['text']}" for m in matches[:20]]
        return AgentResult(
            True, self.name, "vault_grep",
            f"{len(matches)} coincidencia(s) en vault/:\n" + "\n".join(lines),
            data=matches[:20],
        )

    def index(self) -> AgentResult:
        """Construye el grafo graphify + indexa RAG en un solo paso."""
        results = {}
        warnings = []

        if HAS_GRAPHIFY and GraphifyTool.is_available(self.ctx.root):
            try:
                built = GraphifyTool.build(self.ctx.root)
                results["graphify"] = {"success": built.returncode == 0}
                if built.returncode == 0:
                    g = GraphifyTool.load_graph(self.ctx.root)
                    results["graphify"]["nodes"] = len(g.get("nodes", []))
                    results["graphify"]["edges"] = len(g.get("edges", []))
            except Exception as exc:
                warnings.append(f"graphify build falló: {exc}")
                results["graphify"] = {"success": False}
        else:
            results["graphify"] = {"skipped": True}

        if HAS_RAG and RagTool.available():
            try:
                rag_result = RagTool.index_project(self.ctx.root)
                results["rag"] = rag_result
            except Exception as exc:
                warnings.append(f"rag index falló: {exc}")
                results["rag"] = {"error": str(exc)}
        else:
            results["rag"] = {"skipped": True}

        lines = []
        for src, res in results.items():
            if res.get("skipped"):
                lines.append(f"  {src}: omitido (no disponible)")
            elif res.get("success") or "total_chunks" in res:
                lines.append(f"  {src}: OK")
            else:
                lines.append(f"  {src}: error")

        return AgentResult(
            True, self.name, "index",
            "Indexación completa:\n" + "\n".join(lines),
            data=results, warnings=warnings,
        )

    def _estado_grafo(self) -> tuple[dict, bool, str]:
        """(datos, disponible, detalle) del grafo graphify."""
        existe = HAS_GRAPHIFY and GraphifyTool.graph_exists(self.ctx.root)
        datos = {
            "available": HAS_GRAPHIFY and GraphifyTool.is_available(self.ctx.root),
            "graph_exists": existe,
        }
        if not existe:
            return datos, False, "no disponible"
        try:
            grafo = GraphifyTool.load_graph(self.ctx.root)
            datos["nodes"] = len(grafo.get("nodes", []))
            datos["edges"] = len(grafo.get("edges", []))
        except Exception:  # noqa: BLE001
            return datos, False, "grafo ilegible"
        return datos, True, f"{datos['nodes']} nodos / {datos['edges']} aristas"

    def _estado_rag(self) -> tuple[dict, bool, str]:
        disponible = HAS_RAG and RagTool.available()
        datos = {"available": disponible}
        if not disponible:
            return datos, False, "chromadb no instalado"
        try:
            datos["chunks"] = RagTool.status(self.ctx.root).get("total_chunks", 0)
        except Exception:  # noqa: BLE001
            return datos, True, "indice ilegible"
        return datos, True, f"{datos['chunks']} fragmentos"

    def _estado_vault(self) -> tuple[dict, bool, str]:
        ruta = self.ctx.root / "vault"
        existe = ruta.exists()
        datos = {
            "exists": existe,
            "files": len(list(ruta.rglob("*.md"))) if existe else 0,
        }
        return datos, existe, f"{datos['files']} archivos" if existe else "no existe"

    def status(self) -> AgentResult:
        """Estado de cada fuente de documentación."""
        comprobaciones = (
            ("graphify", self._estado_grafo),
            ("RAG", self._estado_rag),
            ("vault", self._estado_vault),
        )
        fuentes, lineas = {}, []
        for etiqueta, comprobar in comprobaciones:
            datos, ok, detalle = comprobar()
            fuentes[etiqueta.lower()] = datos
            lineas.append(f"  {'✓' if ok else '✗'} {etiqueta} — {detalle}")

        return AgentResult(
            True, self.name, "status",
            "Fuentes de documentación:\n" + "\n".join(lineas),
            data=fuentes,
        )
