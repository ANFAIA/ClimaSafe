"""Tests de RAG-006: solapamiento en el chunking por secciones.

Antes, ``_chunks_desde_md`` dividía el .md por secciones (##) sin solapamiento:
cada sección era exactamente su propio texto, así que una pregunta que caía en
el borde entre dos secciones solo recuperaba una de las dos. Con solapamiento
(configurable en caracteres), cada sección lleva como prefijo la cola de la
sección anterior, de modo que la pregunta de la transición recupera ambas.

El número de fragmentos NO cambia (sigue siendo uno por sección): lo que cambia
es el contenido de cada fragmento (más palabras, y con él el hash, que hace que
sync_documentos reindexe). Por eso el conteo antes/después es 1:1 en fragmentos
pero las palabras suben — se comprueba abajo.

El embedder real se sustituye por uno determinista basado en palabras: se prueba
la lógica de solapamiento y de recuperación, no la calidad de los vectores.
"""

from __future__ import annotations

import pytest

import climasafeai.db.rag as rag_mod
from climasafeai.db.manager import DBManager

# Palabras del vocabulario del test y la posición que ocupan en el vector.
# One-hot por palabra: dos textos comparten similitud si comparten palabras.
_VOCAB = {
    "borde": 0,
    "huerto": 1,
    "manzana": 2,
    "pera": 3,
    "cosecha": 4,
    "maduran": 5,
    "verde": 6,
    "roja": 7,
    "otono": 8,
    "sol": 9,
    "sombra": 10,
}

_DOC = (
    "# Huerta\n\n"
    "## Manzanos\n\n"
    "En el huerto maduran las manzanas verdes y rojas junto al borde del huerto "
    "al sol del mediodia cada verano.\n\n"
    "## Perales\n\n"
    "Las peras se cosechan en otono en la sombra del lado norte de la tierra humeda.\n"
)


class _EmbedderVocabulario:
    """Vector one-hot por palabra del vocabulario: 1.0 si la palabra aparece."""

    def encode(self, texto: str):
        v = [0.0] * rag_mod.EMBEDDING_DIM
        for token in texto.lower().split():
            token = token.strip(".,;:()")
            if token in _VOCAB:
                v[_VOCAB[token]] = 1.0
        return v


@pytest.fixture(autouse=True)
def embedder_falso(monkeypatch):
    monkeypatch.setattr(rag_mod, "_get_embedder", lambda: _EmbedderVocabulario())


@pytest.fixture
def rag_db(tmp_path):
    """BD temporal con los factores cargados y sin indexar documentacion/."""
    db = DBManager(tmp_path / "test.db")
    db.initialize()
    db.migrar_desde_json()
    original = db.rag.sync_documentos
    db.rag.sync_documentos = lambda *a, **k: 0
    db.rag.initialize()
    db.rag.sync_documentos = original
    return db


def _doc(tmp_path) -> tuple:
    """Crea documentacion/papers/huerta.md y devuelve (root, fichero)."""
    docs = tmp_path / "documentacion" / "papers"
    docs.mkdir(parents=True)
    f = docs / "huerta.md"
    f.write_text(_DOC, encoding="utf-8")
    return tmp_path, f


# ── Chunking: la sección siguiente incluye la cola de la anterior ─────────


class TestChunking:
    def test_sin_overlap_cada_seccion_es_solo_su_texto(self, tmp_path, rag_db):
        _, f = _doc(tmp_path)
        chunks = rag_db.rag._chunks_desde_md(f, overlap=0)
        secciones = {c["seccion"]: c for c in chunks}
        assert set(secciones) == {"Manzanos", "Perales"}
        # Sin solapamiento, "Perales" NO contiene el borde que está en "Manzanos".
        assert "borde del huerto" not in secciones["Perales"]["texto"]

    def test_con_overlap_la_siguiente_incluye_la_cola(self, tmp_path, rag_db):
        _, f = _doc(tmp_path)
        chunks = rag_db.rag._chunks_desde_md(f, overlap=200)
        secciones = {c["seccion"]: c for c in chunks}
        assert set(secciones) == {"Manzanos", "Perales"}
        # Con solapamiento, "Perales" lleva como prefijo la cola de "Manzanos".
        assert "borde del huerto" in secciones["Perales"]["texto"]

    def test_el_conteo_de_fragmentos_no_cambia_pero_las_palabras_suben(
        self, tmp_path, rag_db
    ):
        _, f = _doc(tmp_path)
        rag = rag_db.rag
        sin = rag._chunks_desde_md(f, overlap=0)
        con = rag._chunks_desde_md(f, overlap=200)
        # 1 fragmento por sección en ambos casos (la clave de dedup no cambia).
        assert len(sin) == len(con) == 2
        # El texto de "Perales" crece porque arrastra la cola de "Manzanos".
        palabras_sin = {c["seccion"]: c["palabras"] for c in sin}
        palabras_con = {c["seccion"]: c["palabras"] for c in con}
        assert palabras_con["Perales"] > palabras_sin["Perales"]

    def test_overlap_default_revertido_a_cero(self, tmp_path, rag_db):
        """RAG-006: el solapamiento se midió y NO mejora el recall, así que el
        default quedó en 0 (revertido). El código queda disponible pasándolo
        explícitamente, pero por defecto no se aplica."""
        assert rag_mod.CHUNK_OVERLAP == 0
        _, f = _doc(tmp_path)
        chunks = rag_db.rag._chunks_desde_md(f)
        secciones = {c["seccion"]: c for c in chunks}
        assert "borde del huerto" not in secciones["Perales"]["texto"]

    def test_la_clave_de_dedup_no_cambia(self, tmp_path, rag_db):
        """ruta::seccion se mantiene con y sin solapamiento: sync no se rompe."""
        root, _ = _doc(tmp_path)
        rag = rag_db.rag
        # Primera sincronización con solapamiento: 2 fragmentos nuevos.
        assert rag.sync_documentos(project_root=root) == 2
        # Sin cambios, no se reindexa nada (misma clave, mismo hash).
        assert rag.sync_documentos(project_root=root) == 0
        with rag._conn() as conn:
            filas = conn.execute("SELECT ruta, seccion FROM docs_vec_src").fetchall()
        assert len(filas) == 2


# ── Integración: la pregunta de la transición recupera las dos secciones ──


class TestBusqueda:
    def test_una_pregunta_entre_dos_secciones_recupera_las_dos(self, rag_db, tmp_path):
        root, _ = _doc(tmp_path)
        rag = rag_db.rag
        # Con solapamiento por defecto, ambas secciones recuperables para la
        # query del borde: "Manzanos" por su propio texto y "Perales" porque
        # lleva la cola de "Manzanos" como prefijo.
        assert rag.sync_documentos(project_root=root) == 2
        res = rag.search_documentos("borde del huerto", k=2)
        secciones = {r["seccion"] for r in res}
        assert secciones == {"Manzanos", "Perales"}
