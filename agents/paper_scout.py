"""
agents.paper_scout — Búsqueda programada de papers con clasificación LLM.

Flujo:
  1. Para cada temática del proyecto, busca papers en arXiv y OpenAlex.
  2. Clasifica cada paper con LLM (LiteLLM + Gemini): factor de riesgo,
     modelo alternativo, o no relevante.
  3. Genera resumen en markdown y lo guarda en documentacion/papers/<categoria>/
     o documentacion/modelos/<modelo>/.
  4. Retorna resumen de lo encontrado.

Uso:
    uv run python -m agents scout
    uv run python -m agents scout --dry-run
"""

from __future__ import annotations

import json
import os
import re
import textwrap
import time
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

load_dotenv()

from agents.tools.research_tool import ResearchTool  # noqa: E402

try:
    from openai import OpenAI, RateLimitError as _RateLimitError
    RateLimitError = _RateLimitError
except ImportError:
    OpenAI = None  # type: ignore
    RateLimitError = type("RateLimitError", (Exception,), {})  # fallback no-op

# ── Configuración ─────────────────────────────────────────────────────────────

BASE_DIR = Path(__file__).resolve().parent.parent
DOCS_DIR = BASE_DIR / "documentacion"
PAPERS_DIR = DOCS_DIR / "papers"
MODELOS_DIR = DOCS_DIR / "modelos"

GEMINI_BASE_URL = os.getenv("GEMINI_BASE_URL", "https://generativelanguage.googleapis.com/v1beta/openai/")
DEFAULT_LLM_MODEL = os.getenv("PAPER_SCOUT_MODEL", "gemini-2.0-flash")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

MAX_RESULTS_PER_QUERY = 15
RELEVANCE_THRESHOLD = 0.20

# Filtros de calidad: palabras clave que deben aparecer (por categoría)
# para que un paper se considere relevante al dominio del proyecto.
_POSITIVE_DOMAIN_KEYWORDS: dict[str, list[str]] = {
    "factor-riesgo": [
        "human", "health", "mortality", "risk", "patient", "population",
        "clinical", "epidemiolog", "death", "hospital", "exposure",
        "cardiovascular", "temperature", "heat", "cold", "physiological",
        "comorbidity", "medication", "vulnerability", "protective",
    ],
    "modelo": [
        "forecasting", "time series", "prediction", "neural", "deep learning",
        "weather", "climate", "temperature", "model", "network",
        "probabilistic", "uncertainty", "regression", "classification",
    ],
}

# Palabras clave requeridas en el TÍTULO (no solo abstract) por categoría.
# Garantiza que papers de factor-riesgo mencionen explícitamente el dominio climático.
_TITLE_REQUIRED_KEYWORDS: dict[str, list[str]] = {
    "factor-riesgo": [
        "heat", "cold", "temperature", "thermal", "climate", "climatic",
        "weather", "uv", "ultraviolet", "solar", "heatwave", "heat wave",
        "acclimatization", "acclimation", "acclimatised", "acclimatised",
        "thermoregulation", "hyperthermia", "hypothermia", "heat stress",
        "cold stress", "wind chill", "heat index", "wbgt", "utci",
        "humidex", "biometeorolog", "heat-related", "cold-related",
        "heat health", "heat warning", "heat action", "heat exposure",
        "cold exposure", "warm spell", "cooling", "heating degree",
        "seasonal", "ambient temperature", "tropospheric",
        "summer", "winter",
    ],
    "modelo": [],  # no restricción extra — los positivos del abstract bastan
}

# Filtros negativos: si el título contiene alguna de estas cadenas, se descarta.
# Incluye dominios ajenos (física, matemáticas, finanzas, cosmética, etc.) y
# términos que producen falsos positivos con "heat" (heat equation, heat operator).
_NEGATIVE_TITLE_FILTERS = [
    # Física / fluidos
    "reynolds number", "flow pattern", "turbulent", "plasma", "magnetized",
    "circular cylinder", "heat transfer", "heat convection", "fluid dynamics",
    # Matemáticas
    "heat equation", "heat operator", "asymptotic", "semigroup", "pde",
    "functional analysis", "solution of the heat", "laplacian", "manifold",
    "long-time asymptotic",
    # Finanzas / actuarial / pensiones
    "portfolio", "hedging", "annuity", "asset pricing", "sharpe ratio",
    "financial risk", "life insurance", "longevity risk", "longevity hedge",
    "pension scheme", "pension fund", "force of mortality", "mortality rate",
    "mortality improvement", "stochastic process toolkit",
    "agricultural insurance", "indemnity payment",
    # Cosméticos / dermatología no climática
    "cosmetic", "cosmeceutical", "nanotechnology in cosmetics",
    "osteoporosis", "probiotic", "atopic dermatitis",
    "skin-on-a-chip", "skin disease", "dermatology", "dermatologist",
    # Ruido ocupacional (no térmico)
    "noise exposure", "hearing loss", "noise-induced",
    # Agricultura / aves / ganadería
    "egg production", "commercial hens", "poultry",
    "food insecurity", "food security",
    # Astrofísica
    "supernova", "neutrino", "gravitational wave",
    # Psicología / cognitivo (no térmico)
    "cognitive load", "response time", "drift diffusion",
    "annotating", "cat is a cat",
    # Sociología
    "social origin", "diffusion of innovation",
    "social network",
    # COVID / enfermedades infecciosas no climáticas
    "covid-19", "coronavirus", "sars-cov", "dengue",
    # Resonancia magnética / neuroimagen
    "diffusion-weighted imaging", "tractography", "mri",
    "temporal lobe",
    # Texto / imagen generativa no climática
    "text-to-image", "image generation",
    # Metáforas / usos figurados de "heat"
    "depression fans the flames", "feasts on the heat",
    # General / demasiado genéricos para proyectos de salud
    "digital transformation in health",
    "personal health device", "diabetes management",
    "global burden", "gbd 201", "gbd 2016", "gbd 2017",
    "328 disease", "282 cause",
    # Comentarios / réplicas editoriales no sustantivas
    "comment on ", "reply to comment",
    "stochastic processes toolkit", "stochastic process toolkit",
    # Ocupacional no térmico específico
    "logdoctor",
    # Procesamiento de imágenes / texto no climáticas
    "inference-time augmentation", "physiological time series",
    "event causality", "blogs and films",
    # Gestión de riesgos genérica (no climática)
    "stochastic processes toolkit", "stochastic modelling",
    # Riesgo de mortalidad genérico (no climático)
    "quantifying mortality risk in",
]


# ── Queries de búsqueda ──────────────────────────────────────────────────────

SEARCH_QUERIES: list[dict[str, Any]] = [
    {
        "query": "heat risk factors medication comorbidity mortality",
        "category": "factor-riesgo",
        "target": "factores-riesgo",
        "description": "Medicación y comorbilidades como factor de riesgo en calor extremo",
    },
    {
        "query": "heat acclimatization time course physiological adaptation",
        "category": "factor-riesgo",
        "target": "aclimatacion",
        "description": "Tiempos de aclimatación al calor y desadaptación",
    },
    {
        "query": "heat stress occupational exposure workers health",
        "category": "factor-riesgo",
        "target": "ocupacional",
        "description": "Estrés térmico ocupacional y salud laboral",
    },
    {
        "query": "biometeorological index heat cold wind chill WBGT UTCI",
        "category": "factor-riesgo",
        "target": "indices-biometeorologicos",
        "description": "Índices biometeorológicos (Heat Index, Wind Chill, WBGT, UTCI)",
    },
    {
        "query": "heat health action plan early warning system",
        "category": "factor-riesgo",
        "target": "planes-accion",
        "description": "Planes de acción frente al calor y sistemas de alerta temprana",
    },
    {
        "query": "cold exposure cardiovascular mortality risk factors",
        "category": "factor-riesgo",
        "target": "factores-riesgo",
        "description": "Exposición al frío y riesgo cardiovascular",
    },
    {
        "query": "UV exposure risk factors skin health personalized warning",
        "category": "factor-riesgo",
        "target": "factores-riesgo",
        "description": "Radiación UV y factores de riesgo personalizados",
    },
    {
        "query": "time series forecasting weather temperature deep learning attention",
        "category": "modelo",
        "target": "transformers",
        "description": "Transformers / atención para forecasting de series temporales meteorológicas",
    },
    {
        "query": "graph neural network spatiotemporal forecasting weather temperature",
        "category": "modelo",
        "target": "gnn",
        "description": "Redes neuronales de grafo para forecasting espacio-temporal",
    },
    {
        "query": "neural basis expansion time series forecasting N-BEATS N-HiTS",
        "category": "modelo",
        "target": "nbeats",
        "description": "N-BEATS / N-HiTS para forecasting univariante",
    },
    {
        "query": "diffusion probabilistic model time series generation extreme events",
        "category": "modelo",
        "target": "diffusion",
        "description": "Modelos de difusión probabilística para generación de escenarios extremos",
    },
    {
        "query": "probabilistic forecast uncertainty quantification temperature heatwave",
        "category": "modelo",
        "target": "transformers",
        "description": "Forecasting probabilístico con cuantificación de incertidumbre",
    },
]


# ── Estructuras ───────────────────────────────────────────────────────────────

@dataclass
class ScoutPaper:
    title: str
    authors: list[str]
    year: int | None
    url: str
    doi: str | None
    source: str
    abstract: str
    relevance: float
    citations: int | None
    query_info: dict[str, Any]

    classification: str = ""
    llm_summary: str = ""
    llm_reasoning: str = ""
    llm_data_points: list[str] = field(default_factory=list)


@dataclass
class ScoutResult:
    total_found: int = 0
    new_papers: list[ScoutPaper] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    by_category: dict[str, int] = field(default_factory=dict)
    saved_files: list[Path] = field(default_factory=list)
    llm_calls: int = 0


# ── LLM helper (batch) ────────────────────────────────────────────────────────

def _classify_batch(papers: list[ScoutPaper]) -> list[ScoutPaper]:
    """Clasifica un lote de papers con una sola llamada LLM."""
    if not GEMINI_API_KEY:
        for p in papers:
            p.classification = "factor-riesgo"
            p.llm_reasoning = "Sin GEMINI_API_KEY — clasificación por defecto"
        return papers

    if OpenAI is None:
        for p in papers:
            p.classification = "factor-riesgo"
            p.llm_reasoning = "openai no instalado — clasificación por defecto"
        return papers

    if not papers:
        return papers

    # Construir lista de papers para el prompt
    papers_block = "\n\n".join(
        f"ID: {i}\nTÍTULO: {p.title}\nABSTRACT: {p.abstract[:800]}"
        for i, p in enumerate(papers)
    )

    prompt = textwrap.dedent(f"""\
        Eres un asistente que clasifica papers para ClimaSafeAI
        (predicción de riesgo climático personalizado: calor, frío, UV).

        Papers a clasificar ({len(papers)}):
        {papers_block}

        Por cada paper, decide si describe:
        1. "factor-riesgo": factor de riesgo cuantificable (medicación, comorbilidad,
           exposición, índice biometeorológico, plan de acción).
        2. "modelo": modelo ML alternativo (arquitectura, forecasting, generación).
        3. "irrelevante": no aporta al proyecto.

        Responde ÚNICAMENTE con un JSON array, un objeto por paper:
        [
          {{
            "id": 0,
            "classification": "factor-riesgo" | "modelo" | "irrelevante",
            "reasoning": "explicación breve en español (máx 2 líneas)",
            "summary": "resumen de lo que aporta (máx 3 líneas)",
            "data_points": ["RR=1.37", ...]  // datos numéricos relevantes o []
          }},
          ...
        ]
    """)

    client = OpenAI(api_key=GEMINI_API_KEY, base_url=GEMINI_BASE_URL)
    for attempt in range(2):
        try:
            resp = client.chat.completions.create(
                model=DEFAULT_LLM_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=2048,
            )
            content = resp.choices[0].message.content.strip()
            break
        except RateLimitError:
            if attempt == 0:
                time.sleep(60)
                continue
            for p in papers:
                p.classification = "factor-riesgo"
                p.llm_reasoning = "Tasa limitada (429) incluso tras reintento"
            return papers
        except Exception as exc:
            for p in papers:
                p.classification = "factor-riesgo"
                p.llm_reasoning = f"Error LLM: {exc}"
            return papers

    # Extraer JSON
    if "```json" in content:
        content = content.split("```json")[1].split("```")[0].strip()
    elif "```" in content:
        content = content.split("```")[1].split("```")[0].strip()
    try:
        classifications = json.loads(content)
    except json.JSONDecodeError:
        for p in papers:
            p.classification = "factor-riesgo"
            p.llm_reasoning = "Error parseando respuesta LLM"
        return papers

    if not isinstance(classifications, list):
        for p in papers:
            p.classification = "factor-riesgo"
            p.llm_reasoning = "Respuesta LLM inesperada"
        return papers

    id_map = {c.get("id"): c for c in classifications if isinstance(c, dict)}
    for p in papers:
        idx = papers.index(p)
        c = id_map.get(idx, {})
        p.classification = c.get("classification", "factor-riesgo")
        p.llm_reasoning = c.get("reasoning", "")
        p.llm_summary = c.get("summary", "")
        p.llm_data_points = c.get("data_points", [])

    return papers


# ── Markdown generation ──────────────────────────────────────────────────────

def _slugify(text: str, max_len: int = 80) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9]+", "-", text)
    text = text.strip("-")
    return text[:max_len].strip("-")


def _generate_markdown(paper: ScoutPaper) -> str:
    lines: list[str] = []

    lines.append(f"# {paper.title}")
    lines.append("")

    lines.append(f"> **Fuente:** {paper.source}  ")
    lines.append(f"> **DOI:** {paper.doi or 'N/A'}  ")
    lines.append(f"> **URL:** {paper.url}  ")
    if paper.year:
        lines.append(f"> **Año:** {paper.year}")
    lines.append("")

    authors_str = ", ".join(paper.authors[:5])
    if len(paper.authors) > 5:
        authors_str += " et al."
    lines.append(f"**Autores:** {authors_str}")
    lines.append("")

    lines.append("## Abstract")
    lines.append("")
    lines.append(paper.abstract)
    lines.append("")

    if paper.llm_summary:
        lines.append("## Relevancia para ClimaSafeAI")
        lines.append("")
        lines.append(paper.llm_summary)
        lines.append("")

    if paper.llm_data_points:
        lines.append("## Datos numéricos extraídos")
        lines.append("")
        for dp in paper.llm_data_points:
            lines.append(f"- {dp}")
        lines.append("")

    lines.append("---")
    lines.append("")
    lines.append(
        f"_Clasificado automáticamente como **{paper.classification}**"
        f" el {date.today().isoformat()}. Razón: {paper.llm_reasoning}_"
    )
    lines.append("")

    return "\n".join(lines)


def _papers_index_md(category: str, target_dir: str, papers: list[ScoutPaper]) -> str:
    lines = [f"# {target_dir.replace('-', ' ').title()}", ""]
    lines.append("Papers encontrados automáticamente por el agente scout.")
    lines.append("")
    lines.append("| Título | Año | Fuente | Relevancia | Clasificación |")
    lines.append("|--------|-----|--------|------------|---------------|")
    for p in papers:
        slug = _slugify(p.title)
        short_title = p.title[:60] + "..." if len(p.title) > 60 else p.title
        lines.append(f"| [{short_title}]({slug}.md) | {p.year or '—'} | {p.source} | {p.relevance:.2f} | {p.classification} |")
    lines.append("")
    lines.append(f"_Última actualización: {datetime.now().strftime('%Y-%m-%d %H:%M')}_")
    lines.append("")
    return "\n".join(lines)


# ── Búsqueda ─────────────────────────────────────────────────────────────────

def _quality_filter(p: dict[str, Any], qinfo: dict[str, Any]) -> bool:
    """Filtro de calidad: descarta papers fuera del dominio del proyecto."""
    title = (p.get("title") or "").lower()
    abstract = (p.get("abstract") or "").lower()
    combined = f"{title} {abstract}"

    # Filtro negativo: palabras clave en título que indican dominio ajeno
    for neg in _NEGATIVE_TITLE_FILTERS:
        if neg.lower() in title:
            return False

    cat = qinfo.get("category", "factor-riesgo")

    # Filtro de título requerido: al menos una keyword del dominio climático
    required = _TITLE_REQUIRED_KEYWORDS.get(cat, [])
    if required and not any(kw.lower() in title for kw in required):
        return False

    # Filtro positivo (abstract+título): al menos una keyword del dominio del proyecto
    positive = _POSITIVE_DOMAIN_KEYWORDS.get(cat, _POSITIVE_DOMAIN_KEYWORDS["factor-riesgo"])
    if not any(kw.lower() in combined for kw in positive):
        return False

    return True


def _search_query(qinfo: dict[str, Any]) -> list[ScoutPaper]:
    query = qinfo["query"]
    papers: list[ScoutPaper] = []
    keywords = ResearchTool.extract_keywords(query, top=8)

    for backend_name in ("arxiv", "openalex"):
        search_fn = ResearchTool.search_arxiv if backend_name == "arxiv" else ResearchTool.search_openalex
        try:
            raw = search_fn(query, max_results=MAX_RESULTS_PER_QUERY)
        except Exception:
            continue

        for p in raw:
            if not p.get("title"):
                continue
            relevance = ResearchTool.relevance(p, keywords)
            if relevance < RELEVANCE_THRESHOLD:
                continue
            if not _quality_filter(p, qinfo):
                continue
            papers.append(ScoutPaper(
                title=p["title"],
                authors=p["authors"],
                year=p["year"],
                url=p.get("url", ""),
                doi=p.get("doi"),
                source=p.get("source", backend_name),
                abstract=p.get("abstract", ""),
                relevance=relevance,
                citations=p.get("citations"),
                query_info=dict(qinfo),
            ))

    # Dedupe
    seen: set[str] = set()
    deduped: list[ScoutPaper] = []
    for p in papers:
        key = p.doi or ResearchTool._dedupe_key({"doi": p.doi, "title": p.title})
        if key not in seen:
            seen.add(key)
            deduped.append(p)

    deduped.sort(key=lambda p: (-p.relevance, -(p.citations or 0)))
    return deduped[:MAX_RESULTS_PER_QUERY]


# ── Save ─────────────────────────────────────────────────────────────────────

def _save_paper(paper: ScoutPaper) -> Path:
    category = paper.query_info["category"]
    target = paper.query_info["target"]

    if category == "modelo":
        base_dir = MODELOS_DIR / target
    else:
        base_dir = PAPERS_DIR / target

    base_dir.mkdir(parents=True, exist_ok=True)

    filename = f"{_slugify(paper.title)}.md"
    filepath = base_dir / filename

    md_content = _generate_markdown(paper)
    filepath.write_text(md_content, encoding="utf-8")

    return filepath


def _update_indices(all_papers: list[ScoutPaper]) -> list[Path]:
    updated: list[Path] = []
    targets = {p.query_info["target"] for p in all_papers}
    for target in targets:
        same_target = [p for p in all_papers if p.query_info["target"] == target]
        if not same_target:
            continue
        category = same_target[0].query_info["category"]
        if category == "modelo":
            base_dir = MODELOS_DIR / target
        else:
            base_dir = PAPERS_DIR / target
        index_path = base_dir / "README.md"
        md = _papers_index_md(category, target, same_target)
        index_path.write_text(md, encoding="utf-8")
        updated.append(index_path)
    return updated


# ── CLI principal ─────────────────────────────────────────────────────────────

def scout_run(*, dry_run: bool = False, queries: list[str] | None = None) -> ScoutResult:
    """
    Ejecuta la ronda de búsqueda de papers.

    Args:
        dry_run: si True, no guarda archivos.
        queries: filtra por substring en query o target (por defecto todas).
    """
    result = ScoutResult()

    active_queries = [
        q for q in SEARCH_QUERIES
        if not queries or any(sub.lower() in q["query"].lower() or sub.lower() in q["target"].lower() for sub in queries)
    ]

    all_relevant: list[ScoutPaper] = []

    for qinfo in active_queries:
        try:
            papers = _search_query(qinfo)
        except Exception as exc:
            result.errors.append(f"Error en query '{qinfo['query']}': {exc}")
            continue

        if not papers:
            continue

        result.total_found += len(papers)

        # Clasificar lote
        classified = _classify_batch(papers)
        result.llm_calls += 1

        relevant = [p for p in classified if p.classification != "irrelevante"]

        result.by_category[qinfo["category"]] = (
            result.by_category.get(qinfo["category"], 0) + len(relevant)
        )
        result.new_papers.extend(relevant)
        all_relevant.extend(relevant)

        if not dry_run:
            for p in relevant:
                path = _save_paper(p)
                result.saved_files.append(path)

    if not dry_run and all_relevant:
        _update_indices(all_relevant)

    return result


def scout_summary(result: ScoutResult) -> str:
    lines: list[str] = []
    lines.append("── Paper Scout ──────────────────────────────────")
    lines.append(f"  Total papers encontrados: {result.total_found}")
    lines.append(f"  Papers relevantes:       {len(result.new_papers)}")
    lines.append(f"  Archivos guardados:      {len(result.saved_files)}")
    lines.append(f"  LLM calls:               {result.llm_calls}")
    if result.by_category:
        lines.append("  Por categoría:")
        for cat, count in sorted(result.by_category.items()):
            lines.append(f"    • {cat}: {count}")
    if result.errors:
        lines.append(f"  Errores: {len(result.errors)}")
        for err in result.errors[:3]:
            lines.append(f"    ⚠ {err}")
    if result.new_papers:
        lines.append("")
        lines.append("Papers encontrados:")
        for p in sorted(result.new_papers, key=lambda x: -x.relevance):
            tag = str(p.classification)
            lines.append(f"  [{tag:14s}] {p.relevance:.2f}  {p.title[:70]}")
    lines.append("─────────────────────────────────────────────────")
    return "\n".join(lines)


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Paper Scout — búsqueda programada de papers")
    parser.add_argument("--dry-run", action="store_true", help="Solo mostrar resultados, no guardar")
    parser.add_argument("--query", action="append", help="Filtrar por substring en query o target (puede repetirse)")
    args = parser.parse_args()

    result = scout_run(dry_run=args.dry_run, queries=args.query)
    print(scout_summary(result))
    return 0 if not result.errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
