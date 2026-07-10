from __future__ import annotations

from agents.tools.research_tool import ResearchTool

# Respuesta Atom real (recortada) del formato que devuelve export.arxiv.org.
_ARXIV_XML = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <entry>
    <id>http://arxiv.org/abs/1234.5678v1</id>
    <published>2021-03-01T00:00:00Z</published>
    <title>Graph Neural Networks for Molecules</title>
    <summary>We study graph neural networks applied to molecular property prediction.</summary>
    <author><name>Ada Lovelace</name></author>
    <author><name>Alan Turing</name></author>
  </entry>
</feed>"""

_OPENALEX_JSON = {
    "results": [
        {
            "display_name": "Attention Is All You Need",
            "publication_year": 2017,
            "doi": "https://doi.org/10.5555/ATTN",
            "cited_by_count": 90000,
            "authorships": [{"author": {"display_name": "A. Vaswani"}}],
            "abstract_inverted_index": {"The": [0], "transformer": [1], "attention": [2]},
            "primary_location": {"landing_page_url": "https://example.com/attn"},
        }
    ]
}


def test_extract_keywords_orders_by_frequency():
    kws = ResearchTool.extract_keywords(
        "networks networks networks attention attention transformer the the",
        top=5,
    )
    assert "networks" in kws and "attention" in kws
    assert "the" not in kws  # stopword
    # 'networks' (3) va antes que 'attention' (2) — orden por frecuencia
    assert kws.index("networks") < kws.index("attention")


def test_extract_keywords_drops_short_words():
    # 'api' (3 letras) < min_len por defecto (4); 'data' se conserva.
    kws = ResearchTool.extract_keywords("api api api data data", top=5)
    assert "data" in kws and "api" not in kws


def test_parse_arxiv():
    papers = ResearchTool._parse_arxiv(_ARXIV_XML)
    assert len(papers) == 1
    p = papers[0]
    assert p["title"] == "Graph Neural Networks for Molecules"
    assert p["year"] == 2021
    assert p["authors"] == ["Ada Lovelace", "Alan Turing"]
    assert p["source"] == "arxiv"


def test_parse_openalex_reconstructs_abstract_and_doi():
    papers = ResearchTool._parse_openalex(_OPENALEX_JSON)
    assert len(papers) == 1
    p = papers[0]
    assert p["title"] == "Attention Is All You Need"
    assert p["abstract"] == "The transformer attention"  # índice invertido reconstruido
    assert p["doi"] == "10.5555/attn"                     # normalizado, sin el prefijo URL
    assert p["citations"] == 90000


def test_relevance_is_fraction_of_keywords_present():
    paper = {"title": "graph neural networks", "abstract": "for molecules"}
    assert ResearchTool.relevance(paper, ["graph", "neural", "quantum"]) == round(2 / 3, 4)
    assert ResearchTool.relevance(paper, []) == 0.0


def test_dedupe_keeps_most_cited():
    papers = [
        {"title": "Same Paper", "doi": "10.1/x", "citations": 5},
        {"title": "Same Paper", "doi": "10.1/x", "citations": 50},
        {"title": "Other", "doi": None, "citations": 1},
    ]
    deduped = ResearchTool.dedupe(papers)
    assert len(deduped) == 2
    same = next(p for p in deduped if p["doi"] == "10.1/x")
    assert same["citations"] == 50


def test_rank_orders_by_relevance_then_citations():
    keywords = ["transformer", "attention"]
    papers = [
        {"title": "Unrelated", "abstract": "", "citations": 1000},
        {"title": "transformer attention", "abstract": "", "citations": 1},
        {"title": "transformer only", "abstract": "", "citations": 5},
    ]
    ranked = ResearchTool.rank(papers, keywords)
    assert ranked[0]["title"] == "transformer attention"  # relevancia 1.0
    assert ranked[-1]["title"] == "Unrelated"             # relevancia 0.0
    assert ranked[0]["relevance"] == 1.0
