"""Tests de RAG-005: el RAG reindexa cuando cambia el CONTENIDO, no solo la clave.

Antes la detección de cambios miraba únicamente la identidad del fragmento:
`sync_factores` filtraba con `WHERE s.factor_id IS NULL` (solo factores nunca
indexados) y `_documentos_nuevos` comparaba la clave `ruta::seccion`. Editar el
cuerpo de una sección sin tocar su título, o cambiar el coeficiente de un factor
ya indexado, dejaba el embedding viejo para siempre.

Ahora se guarda un hash del texto por fragmento y la comparación usa ese hash.

El embedder real se sustituye por uno determinista: aquí se prueba la lógica de
detección de cambios, no la calidad de los vectores.
"""

from __future__ import annotations

import pytest

import climasafeai.db.rag as rag_mod
from climasafeai.db.manager import DBManager


class _FakeEmbedder:
    """Vector determinista derivado del texto. No hace falta el modelo real."""

    def encode(self, texto: str):
        h = abs(hash(texto)) % 1000
        return [float(h) / 1000.0] * rag_mod.EMBEDDING_DIM


@pytest.fixture(autouse=True)
def embedder_falso(monkeypatch):
    monkeypatch.setattr(rag_mod, "_get_embedder", lambda: _FakeEmbedder())


@pytest.fixture
def rag_db(tmp_path):
    """BD temporal con los factores cargados, sin indexar documentacion/.

    `initialize()` llama a `sync_documentos()` sin argumentos y ahí indexaría
    la documentacion/ real del repo; se silencia solo durante esa llamada y se
    restaura después, porque los tests de documentos sí lo invocan de verdad.
    """
    db = DBManager(tmp_path / "test.db")
    db.initialize()
    db.migrar_desde_json()
    original = db.rag.sync_documentos
    db.rag.sync_documentos = lambda *a, **k: 0
    db.rag.initialize()
    db.rag.sync_documentos = original
    return db


def _docs(tmp_path, cuerpo: str) -> tuple:
    """Crea un documentacion/ mínimo con una sección y devuelve (root, fichero)."""
    docs = tmp_path / "documentacion"
    docs.mkdir(exist_ok=True)
    f = docs / "guia.md"
    f.write_text(f"# Guía\n\n## Sección A\n\n{cuerpo}\n", encoding="utf-8")
    return tmp_path, f


_CUERPO = "Este es el cuerpo original de la seccion con palabras suficientes para pasar el filtro."
_CUERPO_2 = "Cuerpo REESCRITO por completo, con contenido distinto y palabras suficientes para indexar."


# ── Documentos: cambia el cuerpo, no el título ──────────────────────────


class TestDocumentos:

    def test_editar_el_cuerpo_sin_tocar_el_titulo_reindexa(self, tmp_path, rag_db):
        root, fichero = _docs(tmp_path, _CUERPO)
        rag = rag_db.rag

        assert rag.sync_documentos(project_root=root) == 1
        # Sin cambios no se reindexa nada.
        assert rag.sync_documentos(project_root=root) == 0

        # Mismo título de sección, cuerpo distinto: ANTES daba 0.
        fichero.write_text(f"# Guía\n\n## Sección A\n\n{_CUERPO_2}\n", encoding="utf-8")
        assert rag.sync_documentos(project_root=root) == 1

        with rag._conn() as conn:
            filas = conn.execute("SELECT texto FROM docs_vec_src").fetchall()
        assert len(filas) == 1, "reindexar no debe duplicar la fila"
        assert "REESCRITO" in filas[0]["texto"]

    def test_stats_no_duplica_fragmentos_al_reindexar(self, tmp_path, rag_db):
        root, fichero = _docs(tmp_path, _CUERPO)
        rag = rag_db.rag
        rag.sync_documentos(project_root=root)
        antes = rag.stats()["documentos"]["fragmentos"]

        fichero.write_text(f"# Guía\n\n## Sección A\n\n{_CUERPO_2}\n", encoding="utf-8")
        rag.sync_documentos(project_root=root)
        despues = rag.stats()["documentos"]["fragmentos"]

        assert antes == despues == 1

    def test_vec_y_src_no_se_desalinean(self, tmp_path, rag_db):
        """Al reindexar se borra el embedding viejo, no solo la fila de texto."""
        root, fichero = _docs(tmp_path, _CUERPO)
        rag = rag_db.rag
        rag.sync_documentos(project_root=root)
        fichero.write_text(f"# Guía\n\n## Sección A\n\n{_CUERPO_2}\n", encoding="utf-8")
        rag.sync_documentos(project_root=root)

        with rag._conn() as conn:
            n_src = conn.execute("SELECT COUNT(*) n FROM docs_vec_src").fetchone()["n"]
            n_vec = conn.execute("SELECT COUNT(*) n FROM docs_vec").fetchone()["n"]
        assert n_src == n_vec == 1


# ── Factores: cambia el nombre o el coeficiente ────────────────────────


class TestFactores:

    def _un_factor(self, rag):
        with rag._conn() as conn:
            return dict(conn.execute(
                "SELECT factor_id, texto, hash FROM factores_vec_src ORDER BY factor_id LIMIT 1"
            ).fetchone())


    def test_cambiar_el_coeficiente_reembebe_el_factor(self, rag_db):
        rag = rag_db.rag
        assert rag.sync_factores() == 0, "recién inicializado no hay nada que hacer"

        antes = self._un_factor(rag)
        with rag._conn() as conn:
            conn.execute(
                "UPDATE factores_riesgo SET coef = coef + 0.77 WHERE id = ?",
                (antes["factor_id"],),
            )

        assert rag.sync_factores() == 1

        despues = self._un_factor(rag)
        assert despues["texto"] != antes["texto"]
        assert despues["hash"] != antes["hash"]
        assert "coeficiente:" in despues["texto"]

    def test_cambiar_el_nombre_reembebe_el_factor(self, rag_db):
        rag = rag_db.rag
        antes = self._un_factor(rag)
        with rag._conn() as conn:
            conn.execute(
                "UPDATE factores_riesgo SET nombre = ? WHERE id = ?",
                ("nombre completamente nuevo", antes["factor_id"]),
            )

        assert rag.sync_factores() == 1
        assert "nombre completamente nuevo" in self._un_factor(rag)["texto"]

    def test_sin_cambios_no_reindexa(self, rag_db):
        assert rag_db.rag.sync_factores() == 0
        assert rag_db.rag.sync_factores() == 0

    def test_stats_no_duplica_factores_al_reindexar(self, rag_db):
        rag = rag_db.rag
        antes = rag.stats()["factores"]
        with rag._conn() as conn:
            conn.execute("UPDATE factores_riesgo SET coef = coef + 0.5")
        rag.sync_factores()
        despues = rag.stats()["factores"]

        assert antes["embedded"] == despues["embedded"]
        assert despues["pending"] == 0
