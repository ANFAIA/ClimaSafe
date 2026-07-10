"""
agents.tools.research_tool — Búsqueda de papers académicos relacionados con el
proyecto, contra APIs públicas y gratuitas (sin API key).

Como `dependency_tool`, ESTA herramienta sí hace red — a dos fuentes que no
requieren clave:

  - **arXiv**   (``export.arxiv.org/api/query``): Atom XML, preprints.
  - **OpenAlex** (``api.openalex.org/works``): JSON, catálogo abierto con
    recuento de citas.

Todo lo que no es red es determinista y offline (extracción de keywords,
relevancia, deduplicación, ranking), para que se pueda testear sin conexión.
Sin dependencias nuevas: HTTP con `RestTool` (urllib), XML con `xml.etree` de
la stdlib.

Límite honesto: la "relevancia" es léxica (solapamiento de palabras clave del
proyecto con título+abstract), no una comprensión semántica del paper.
"""

from __future__ import annotations

import re
import urllib.parse
import xml.etree.ElementTree as ET
from typing import Any

from agents.tools.registry import register_tool
from agents.tools.rest_tool import RestTool

ARXIV_API = "http://export.arxiv.org/api/query"
OPENALEX_API = "https://api.openalex.org/works"

# Stopwords mínimas ES+EN — suficiente para que la extracción de keywords no
# devuelva "the", "de", "and"... No pretende ser exhaustiva.
_STOPWORDS = {
    # inglés
    "the", "and", "for", "with", "that", "this", "from", "are", "was", "were",
    "has", "have", "not", "but", "our", "using", "used", "use", "can", "will",
    "which", "based", "into", "such", "these", "than", "then", "also", "may",
    # español
    "los", "las", "una", "unos", "unas", "del", "que", "con", "por", "para",
    "como", "más", "este", "esta", "estos", "estas", "sus", "sobre", "entre",
    "cada", "según", "desde", "muy", "sin", "son", "fue", "ser", "hay",
}


@register_tool("research")
class ResearchTool:
    # -- extracción de keywords (offline) -------------------------------------
    @staticmethod
    def extract_keywords(text: str, *, top: int = 12, min_len: int = 4) -> list[str]:
        """
        Palabras clave candidatas de un texto, por frecuencia. Determinista:
        tokeniza, descarta stopwords y palabras cortas, ordena por frecuencia
        (desempate alfabético para estabilidad).
        """
        words = re.findall(r"[a-zA-Záéíóúñü]+", text.lower())
        freq: dict[str, int] = {}
        for w in words:
            if len(w) < min_len or w in _STOPWORDS:
                continue
            freq[w] = freq.get(w, 0) + 1
        ordered = sorted(freq.items(), key=lambda kv: (-kv[1], kv[0]))
        return [w for w, _ in ordered[:top]]

    # -- búsqueda: arXiv ------------------------------------------------------
    @staticmethod
    def search_arxiv(query: str, *, max_results: int = 10, timeout: int = 15) -> list[dict[str, Any]]:
        """Busca en arXiv y devuelve papers normalizados. Lanza si la red falla."""
        params = urllib.parse.urlencode({
            "search_query": f"all:{query}",
            "start": 0,
            "max_results": max_results,
            "sortBy": "relevance",
            "sortOrder": "descending",
        })
        resp = RestTool.get(f"{ARXIV_API}?{params}", timeout=timeout)
        return ResearchTool._parse_arxiv(resp.text)

    @staticmethod
    def _parse_arxiv(atom_xml: str) -> list[dict[str, Any]]:
        ns = {"a": "http://www.w3.org/2005/Atom"}
        papers: list[dict[str, Any]] = []
        try:
            root = ET.fromstring(atom_xml)
        except ET.ParseError:
            return papers
        for entry in root.findall("a:entry", ns):
            title = (entry.findtext("a:title", default="", namespaces=ns) or "").strip()
            summary = (entry.findtext("a:summary", default="", namespaces=ns) or "").strip()
            published = entry.findtext("a:published", default="", namespaces=ns) or ""
            url = (entry.findtext("a:id", default="", namespaces=ns) or "").strip()
            authors = [
                (a.findtext("a:name", default="", namespaces=ns) or "").strip()
                for a in entry.findall("a:author", ns)
            ]
            year = int(published[:4]) if published[:4].isdigit() else None
            papers.append(ResearchTool._normalize(
                title=title, abstract=summary, authors=authors, year=year,
                url=url, doi=None, citations=None, source="arxiv",
            ))
        return papers

    # -- búsqueda: OpenAlex ---------------------------------------------------
    @staticmethod
    def search_openalex(query: str, *, max_results: int = 10, timeout: int = 15) -> list[dict[str, Any]]:
        """Busca en OpenAlex y devuelve papers normalizados. Lanza si la red falla."""
        params = urllib.parse.urlencode({"search": query, "per_page": max_results})
        resp = RestTool.get(f"{OPENALEX_API}?{params}", timeout=timeout)
        try:
            data = resp.json()
        except ValueError:
            return []
        return ResearchTool._parse_openalex(data)

    @staticmethod
    def _parse_openalex(data: dict[str, Any]) -> list[dict[str, Any]]:
        papers: list[dict[str, Any]] = []
        for work in data.get("results", []):
            authors = [
                (a.get("author", {}) or {}).get("display_name", "")
                for a in work.get("authorships", [])
            ]
            doi = work.get("doi")
            url = (work.get("primary_location", {}) or {}).get("landing_page_url") or work.get("id")
            papers.append(ResearchTool._normalize(
                title=work.get("display_name") or "",
                abstract=ResearchTool._abstract_from_inverted(work.get("abstract_inverted_index")),
                authors=[a for a in authors if a],
                year=work.get("publication_year"),
                url=url, doi=doi, citations=work.get("cited_by_count"),
                source="openalex",
            ))
        return papers

    @staticmethod
    def _abstract_from_inverted(inverted: dict[str, list[int]] | None) -> str:
        """OpenAlex da el abstract como índice invertido {palabra: [posiciones]}."""
        if not inverted:
            return ""
        positions: list[tuple[int, str]] = []
        for word, idxs in inverted.items():
            for i in idxs:
                positions.append((i, word))
        positions.sort()
        return " ".join(w for _, w in positions)

    # -- normalización, relevancia, dedupe, ranking (offline) -----------------
    @staticmethod
    def _normalize(**kw: Any) -> dict[str, Any]:
        """Registro de paper con las mismas claves salga de la fuente que salga."""
        return {
            "title": kw.get("title", "").strip(),
            "abstract": kw.get("abstract", "").strip(),
            "authors": kw.get("authors", []) or [],
            "year": kw.get("year"),
            "url": kw.get("url", ""),
            "doi": (kw.get("doi") or "").replace("https://doi.org/", "").lower() or None,
            "citations": kw.get("citations"),
            "source": kw.get("source", ""),
        }

    @staticmethod
    def relevance(paper: dict[str, Any], keywords: list[str]) -> float:
        """Fracción de keywords del proyecto que aparecen en título+abstract (0..1)."""
        if not keywords:
            return 0.0
        text = f"{paper.get('title', '')} {paper.get('abstract', '')}".lower()
        hits = sum(1 for kw in keywords if kw.lower() in text)
        return round(hits / len(keywords), 4)

    @staticmethod
    def _dedupe_key(paper: dict[str, Any]) -> str:
        if paper.get("doi"):
            return f"doi:{paper['doi']}"
        return "title:" + re.sub(r"[^a-z0-9]+", "", paper.get("title", "").lower())

    @staticmethod
    def dedupe(papers: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Quita duplicados por DOI (o título normalizado), conservando el de más citas."""
        best: dict[str, dict[str, Any]] = {}
        for p in papers:
            key = ResearchTool._dedupe_key(p)
            cur = best.get(key)
            if cur is None or (p.get("citations") or -1) > (cur.get("citations") or -1):
                best[key] = p
        return list(best.values())

    @staticmethod
    def rank(papers: list[dict[str, Any]], keywords: list[str]) -> list[dict[str, Any]]:
        """Ordena por relevancia (desc) y, a igualdad, por citas (desc). Anota 'relevance'."""
        scored = []
        for p in papers:
            p = dict(p)
            p["relevance"] = ResearchTool.relevance(p, keywords)
            scored.append(p)
        scored.sort(key=lambda p: (-p["relevance"], -(p.get("citations") or 0)))
        return scored
