"""Tests de RAG-002: el RAG indexa el coeficiente y el DOI de cada factor.

El texto que se embebe (columna `texto` de `factores_vec_src`) debe llevar
`coeficiente: X` y `doi: ...` cuando existen, para que el LLM pueda citar
números y fuentes en vez de generalidades. Se prueba contra un SQLite
temporal: el RAG usa sqlite-vec sobre la misma BD.
"""

from __future__ import annotations

import re
from unittest.mock import MagicMock

import pytest

from climasafeai.db.manager import DBManager


# ── Fixtures ────────────────────────────────────────────────────────────


@pytest.fixture
def rag_db(tmp_path, monkeypatch):
    """BD temporal con los factores cargados y las tablas RAG de factores.

    Se silencia `sync_documentos` (parte de `initialize()`) para no pagar el
    indexado de documentacion/ en los tests: aquí solo interesan los factores.
    """
    db = DBManager(tmp_path / "test.db")
    db.initialize()
    db.migrar_desde_json()
    rag = db.rag
    monkeypatch.setattr(rag, "sync_documentos", lambda: 0)
    rag.initialize()
    return db


# ── Coeficiente y DOI en el texto indexado ─────────────────────────────


class TestTextoIndexado:

    def test_diureticos_asa_lleva_coef_y_doi(self, rag_db):
        """El factor del ejemplo de la feature trae 'coeficiente: 1.3' y su doi."""
        results = rag_db.rag.search_factores("diureticos de asa riesgo", k=10)
        diu = next(r for r in results if "diuréticos de asa" in r["texto"])
        assert "coeficiente: 1.3" in diu["texto"]
        assert "doi: 10.1371/journal.pone.0233617" in diu["texto"]

    def test_todos_los_factores_llevan_coeficiente(self, rag_db):
        """coef es NOT NULL en la BD: todo texto indexado lo incluye."""
        with rag_db.rag._conn() as conn:
            filas = conn.execute(
                "SELECT texto FROM factores_vec_src"
            ).fetchall()
        assert len(filas) > 0
        for f in filas:
            assert "coeficiente:" in f["texto"]

    def test_doi_solo_cuando_existe(self, rag_db):
        """Factores sin doi (p. ej. no_aclimatado) no llevan la etiqueta 'doi'."""
        with rag_db.rag._conn() as conn:
            filas = conn.execute(
                """SELECT s.texto FROM factores_vec_src s
                   JOIN factores_riesgo f ON s.factor_id = f.id
                   WHERE f.doi IS NULL"""
            ).fetchall()
        assert len(filas) > 0
        for f in filas:
            assert "doi:" not in f["texto"]


# ── Reindexado ─────────────────────────────────────────────────────────


class TestReindexado:

    def test_resync_no_duplica(self, rag_db):
        """resync_factores() borra y reindexa: mismo total, cero pendientes."""
        rag = rag_db.rag
        antes = rag.stats()["factores"]["embedded"]
        assert antes > 0

        rag.resync_factores()

        stats = rag.stats()["factores"]
        assert stats["embedded"] == antes
        assert stats["embedded"] == stats["total"]
        assert stats["pending"] == 0


# ── Distinción calor/frío ──────────────────────────────────────────────


class TestCalorFrio:

    def test_el_texto_indexado_conserva_el_tipo(self, rag_db):
        """Cada texto embebido lleva 'tipo: calor' o 'tipo: frio'."""
        with rag_db.rag._conn() as conn:
            filas = conn.execute(
                "SELECT texto FROM factores_vec_src"
            ).fetchall()
        for f in filas:
            assert ("tipo: calor" in f["texto"]) or ("tipo: frio" in f["texto"])

    def test_busqueda_de_calor_no_devuelve_frio_en_top(self, rag_db):
        """Una pregunta de calor recupera factores de calor, no de frío."""
        results = rag_db.rag.search_factores("ola de calor alta temperatura verano", k=5)
        assert len(results) > 0
        tipos_top = {r["tipo"] for r in results}
        assert tipos_top == {"calor"}
        # Y los dos tipos existen indexados (el test no pasa por tener solo calor)
        with rag_db.rag._conn() as conn:
            tipos = {r["tipo"] for r in conn.execute(
                "SELECT DISTINCT tipo FROM factores_vec_src").fetchall()}
        assert tipos == {"calor", "frio"}


# ── Respuesta con número y fuente (LLM simulado sobre contexto real) ────


class TestRespuestaConNumeroYFuente:
    """Sin LLM configurado (cuota GEMINI agotada en el entorno), se simula un
    LLM determinista que lee el número y el DOI del contexto recuperado: si el
    contexto no los llevara, el test falla al construir la respuesta."""

    def _fake_client(self, captured):
        """Cliente OpenAI simulado: extrae coef/doi del prompt de usuario."""
        client = MagicMock()

        def create(*args, **kwargs):
            user = [m["content"] for m in kwargs["messages"] if m["role"] == "user"][0]
            captured["user"] = user
            # Los datos salen del contexto, no de aquí: si el contexto no
            # trae 'coeficiente:' ni 'doi:', el regex falla y el test falla.
            coef = re.search(
                r"diuréticos de asa\. tipo: calor.*?coeficiente: ([\d.]+)", user
            ).group(1)
            doi = re.search(r"doi: (10\.\S+)", user).group(1)
            resp = MagicMock()
            resp.choices[0].message.content = (
                f"Los diuréticos de asa multiplican el riesgo por {coef} "
                f"(DOI: {doi}). Es un factor de calor; los factores de frío "
                "del contexto no aplican a esta pregunta."
            )
            return resp

        client.chat.completions.create.side_effect = create
        return client

    def test_respuesta_cita_numero_y_fuente(self, rag_db, monkeypatch):
        """'¿cuanto sube el riesgo con diureticos?' se responde con 1.3 y su doi."""
        captured: dict = {}
        monkeypatch.setattr(
            "climasafeai.db.rag._get_llm_client", lambda: self._fake_client(captured)
        )
        res = rag_db.rag.ask("¿cuanto sube el riesgo con diureticos?", k=5)
        assert res.get("error") is None
        assert "1.3" in res["answer"]
        assert "10.1371/journal.pone.0233617" in res["answer"]
        # El prompt que ve el LLM lleva el número y la fuente
        assert "coeficiente: 1.3" in captured["user"]
        assert "doi: 10.1371/journal.pone.0233617" in captured["user"]

    def test_respuesta_no_cita_factores_de_frio(self, rag_db, monkeypatch):
        """El contexto distingue tipo calor/frío y la respuesta de calor no
        cita los factores de frío recuperados (vive solo, encamado...)."""
        captured: dict = {}
        monkeypatch.setattr(
            "climasafeai.db.rag._get_llm_client", lambda: self._fake_client(captured)
        )
        res = rag_db.rag.ask("¿cuanto sube el riesgo con diureticos?", k=5)
        assert res.get("error") is None
        # Los tipos viajan en el contexto para que el LLM los distinga
        assert "tipo: calor" in captured["user"]
        assert "tipo: frio" in captured["user"]
        # Y la respuesta cita solo el factor de calor de la pregunta
        assert "vive solo" not in res["answer"]
        assert "encamado" not in res["answer"]
