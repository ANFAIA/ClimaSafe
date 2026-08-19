"""Tests de RAG-006: el esquema admite varios modelos de embeddings a la vez.

sqlite-vec fija la dimensión en el schema de la columna virtual, así que un
modelo con dimensión distinta necesita su propia tabla. Cada modelo alternativo
vive en ``docs_vec_<slug>`` / ``factores_vec_<slug>`` (con su ``_src``) mientras
que el índice activo (por defecto) se mantiene en las tablas sin sufijo, para
poder comparar sin perder el índice anterior.

Se usan dos embedders falsos de dimensión distinta (384 y 512) para comprobar
que ambos índices conviven y que cada búsqueda consulta el suyo.
"""

from __future__ import annotations

import struct

import pytest

import climasafeai.db.rag as rag_mod
from climasafeai.db.manager import DBManager

_ALT = rag_mod.RAG_EMBEDDER_ALT


class _EmbedderA:
    """Embedder del índice activo (384 dims), sensible al contenido."""

    def encode(self, texto: str):
        if "alpha" in texto:
            return [1.0] + [0.0] * (rag_mod.EMBEDDING_DIM - 1)
        return [0.0, 1.0] + [0.0] * (rag_mod.EMBEDDING_DIM - 2)


class _EmbedderB:
    """Embedder del modelo alternativo (512 dims): mismo contenido, otra dimensión."""

    def encode(self, texto: str):
        if "alpha" in texto:
            return [1.0] + [0.0] * (rag_mod.EMBEDDING_DIM_ALT - 1)
        return [0.0, 1.0] + [0.0] * (rag_mod.EMBEDDING_DIM_ALT - 2)


def _get_embedder_falso(modelo=None):
    if modelo and modelo != rag_mod.RAG_EMBEDDER_DEFAULT:
        return _EmbedderB()
    return _EmbedderA()


@pytest.fixture(autouse=True)
def embedder_falso(monkeypatch):
    monkeypatch.setattr(rag_mod, "_get_embedder", _get_embedder_falso)


@pytest.fixture
def rag_db(tmp_path):
    db = DBManager(tmp_path / "test.db")
    db.initialize()
    db.migrar_desde_json()
    original = db.rag.sync_documentos
    db.rag.sync_documentos = lambda *a, **k: 0
    db.rag.initialize()
    db.rag.sync_documentos = original
    return db


def _docs(tmp_path):
    """Dos documentos de papers (colección recuperable), uno por índice."""
    docs = tmp_path / "documentacion" / "papers"
    docs.mkdir(parents=True)
    a = docs / "doc_a.md"
    a.write_text(
        "# Doc A\n\n## Seccion\n\n"
        "documento alpha sobre manzanas con palabras suficientes para indexar "
        "en el indice vectorial del proyecto.\n",
        encoding="utf-8",
    )
    b = docs / "doc_b.md"
    b.write_text(
        "# Doc B\n\n## Seccion\n\n"
        "documento beta sobre peras con palabras suficientes para indexar en "
        "el indice vectorial del proyecto.\n",
        encoding="utf-8",
    )
    return tmp_path, (a, b)


# ── Esquema: tablas por modelo ───────────────────────────────────────────


class TestEsquema:
    def test_initialize_crea_las_tablas_de_ambos_modelos(self, rag_db):
        with rag_db.rag._conn() as conn:
            tablas = {r["name"] for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' OR type='virtual_table'"
            )}
        # Índice activo (sin sufijo) + alternativo.
        for t in (
            "docs_vec",
            "docs_vec_src",
            "docs_vec_minilm",
            "docs_vec_minilm_src",
            "factores_vec",
            "factores_vec_src",
            "factores_vec_minilm",
            "factores_vec_minilm_src",
        ):
            assert t in tablas, f"falta {t}"

    def test_cada_indice_tiene_su_dimension(self, rag_db, tmp_path):
        root, _ = _docs(tmp_path)
        rag = rag_db.rag
        rag.sync_documentos(project_root=root, modelo=rag_mod.RAG_EMBEDDER_DEFAULT)
        rag.sync_documentos(project_root=root, modelo=_ALT)
        with rag._conn() as conn:
            dim_activa = len(conn.execute("SELECT embedding FROM docs_vec LIMIT 1").fetchone()["embedding"]) // 4
            dim_alt = len(conn.execute("SELECT embedding FROM docs_vec_minilm LIMIT 1").fetchone()["embedding"]) // 4
        assert dim_activa == rag_mod.EMBEDDING_DIM
        assert dim_alt == rag_mod.EMBEDDING_DIM_ALT
        assert dim_activa != dim_alt


# ── Convivencia: cada búsqueda consulta su índice ────────────────────────


class TestConvivencia:
    def test_sync_por_modelo_llena_cada_tabla_sin_mezclar(self, rag_db, tmp_path):
        root, _ = _docs(tmp_path)
        rag = rag_db.rag
        # Activo con el modelo por defecto; alternativo con el modelo B.
        assert rag.sync_documentos(project_root=root, modelo=rag_mod.RAG_EMBEDDER_DEFAULT) == 2
        assert rag.resync_documentos_modelo(_ALT, project_root=root) == 2

        with rag._conn() as conn:
            activo = conn.execute(
                "SELECT modelo, COUNT(*) n FROM docs_vec_src GROUP BY modelo"
            ).fetchall()
            alt = conn.execute(
                "SELECT modelo, COUNT(*) n FROM docs_vec_minilm_src GROUP BY modelo"
            ).fetchall()
        assert [(r["modelo"], r["n"]) for r in activo] == [
            (rag_mod.RAG_EMBEDDER_DEFAULT, 2)
        ]
        assert [(r["modelo"], r["n"]) for r in alt] == [(_ALT, 2)]

    def test_buscar_en_un_indice_no_mezcla_el_otro(self, rag_db, tmp_path):
        root, (a, b) = _docs(tmp_path)
        rag = rag_db.rag
        rag.sync_documentos(project_root=root, modelo=rag_mod.RAG_EMBEDDER_DEFAULT)
        rag.resync_documentos_modelo(_ALT, project_root=root)

        # Quita doc_a SOLO del índice alternativo: así se ve qué índice consulta
        # cada búsqueda.
        with rag._conn() as conn:
            filas = conn.execute(
                "SELECT vec_rowid FROM docs_vec_minilm_src WHERE ruta LIKE '%doc_a.md'"
            ).fetchall()
            for f in filas:
                conn.execute("DELETE FROM docs_vec_minilm WHERE rowid = ?", (f["vec_rowid"],))
                conn.execute(
                    "DELETE FROM docs_vec_minilm_src WHERE vec_rowid = ?", (f["vec_rowid"],)
                )

        # search_documentos (activo) sigue viendo doc_a: consulta docs_vec.
        res_activo = rag.search_documentos("alpha", k=5)
        assert any("doc_a.md" in r["ruta"] for r in res_activo)

        # search_documentos_modelo (alternativo) NO ve doc_a: consulta docs_vec_minilm.
        res_alt = rag.search_documentos_modelo("alpha", _ALT, k=5)
        assert res_alt  # el índice alternativo sigue teniendo doc_b
        assert all("doc_a.md" not in r["ruta"] for r in res_alt)


# ── Detección de cambio de modelo: reindexa cuando el modelo difiere ─────


class TestCambioDeModelo:
    def test_sin_cambios_no_reindexa(self, rag_db, tmp_path):
        root, _ = _docs(tmp_path)
        rag = rag_db.rag
        rag.sync_documentos(project_root=root)
        assert rag.sync_documentos(project_root=root) == 0

    def test_cambiar_el_modelo_de_las_filas_reindexa(self, rag_db, tmp_path):
        root, _ = _docs(tmp_path)
        rag = rag_db.rag
        rag.sync_documentos(project_root=root)
        # Simula filas embebidas con otro modelo (p. ej. tras cambiar el default):
        # el sync las detecta por la columna modelo y las reembebe.
        with rag._conn() as conn:
            conn.execute("UPDATE docs_vec_src SET modelo = 'otro-modelo'")
        assert rag.sync_documentos(project_root=root) == 2
        with rag._conn() as conn:
            modelos = {r["modelo"] for r in conn.execute("SELECT modelo FROM docs_vec_src")}
        assert modelos == {rag_mod.RAG_EMBEDDER_DEFAULT}
