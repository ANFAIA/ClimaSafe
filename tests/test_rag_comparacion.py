"""Tests de RAG-006: la comparación de modelos/overlap contra la línea base.

El script scripts/comparar_modelos_rag.py mide el recall@k de distintos estados
del índice (línea base, +solapamiento, +modelo alternativo) reutilizando las
métricas de evaluar_rag. Estos tests verifican que esa maquinaria de medición
funciona sobre índices con dos modelos de dimensión distinta — no que un modelo
real sea mejor que otro (eso es el experimento, documentado aparte).

Se usa el embedder determinista por contenido de test_rag_eval: la query
"diabetes" solo empata con los textos que contienen la palabra, así que el
recall es controlable.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

import climasafeai.db.rag as rag_mod
from climasafeai.db.manager import DBManager

_RAIZ = Path(__file__).resolve().parents[1]
_SCRIPT = _RAIZ / "scripts" / "comparar_modelos_rag.py"
_ALT = rag_mod.RAG_EMBEDDER_ALT

# El script importa evaluar_rag del mismo directorio (como en ejecución normal).
sys.path.insert(0, str(_SCRIPT.parent))

_spec = importlib.util.spec_from_file_location("comparar_modelos_rag", _SCRIPT)
comparar = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(comparar)


class _FakeEmbedder:
    """Igual que en test_rag_eval: "diabetes" → [1,0,...], resto → [0,1,0,...]."""

    def encode(self, texto: str):
        if "diabetes" in texto:
            return [1.0] + [0.0] * (rag_mod.EMBEDDING_DIM - 1)
        return [0.0, 1.0] + [0.0] * (rag_mod.EMBEDDING_DIM - 2)


class _FakeEmbedderAlt:
    """Idéntico en contenido pero de 512 dims (índice alternativo)."""

    def encode(self, texto: str):
        if "diabetes" in texto:
            return [1.0] + [0.0] * (rag_mod.EMBEDDING_DIM_ALT - 1)
        return [0.0, 1.0] + [0.0] * (rag_mod.EMBEDDING_DIM_ALT - 2)


def _get_embedder_falso(modelo=None):
    if modelo and modelo != rag_mod.RAG_EMBEDDER_DEFAULT:
        return _FakeEmbedderAlt()
    return _FakeEmbedder()


@pytest.fixture(autouse=True)
def embedder_falso(monkeypatch):
    monkeypatch.setattr(rag_mod, "_get_embedder", _get_embedder_falso)


@pytest.fixture
def rag_db(tmp_path):
    """BD temporal con un doc de papers que contiene 'diabetes'."""
    db = DBManager(tmp_path / "test.db")
    db.initialize()
    db.migrar_desde_json()
    original = db.rag.sync_documentos
    db.rag.sync_documentos = lambda *a, **k: 0
    db.rag.initialize()
    db.rag.sync_documentos = original

    docs = tmp_path / "documentacion" / "papers"
    docs.mkdir(parents=True)
    (docs / "doc.md").write_text(
        "# Paper\n\n## Resumen\n\n"
        "Este es un paper sobre diabetes y temperatura con palabras suficientes "
        "para pasar el filtro de indexado.\n",
        encoding="utf-8",
    )
    # Activo (default) y alternativo, cada uno con su índice.
    assert db.rag.sync_documentos(project_root=tmp_path, modelo=rag_mod.RAG_EMBEDDER_DEFAULT) == 1
    assert db.rag.resync_documentos_modelo(_ALT, project_root=tmp_path) == 1
    return db


_PREGUNTAS = [
    {
        "id": "q1",
        "pregunta": "diabetes y temperatura",
        "factores_esperados": ["calor/diabetes"],
        "documentos_esperados": ["documentacion/papers/doc.md"],
    },
    {
        "id": "q2",
        "pregunta": "algo totalmente ajeno",
        "factores_esperados": [],
        "documentos_esperados": ["documentacion/papers/no-existe.md"],
    },
]


class TestMedicion:
    def test_medir_canal_devuelve_estructura_de_metrica(self, rag_db):
        m = comparar.medir_canal(
            rag_db.rag.search_documentos, _PREGUNTAS, "documentos", k=5
        )
        assert set(m) == {"n_preguntas", "recall", "precision"}
        assert m["n_preguntas"] == 2
        assert 0.0 <= m["recall"] <= 1.0
        assert 0.0 <= m["precision"] <= 1.0

    def test_medir_canal_activo_y_alternativo(self, rag_db):
        """La misma maquinaria mide el índice activo y el del modelo alternativo."""
        activo = comparar.medir_canal(
            rag_db.rag.search_documentos, _PREGUNTAS, "documentos", k=5
        )
        alt = comparar.medir_canal(
            lambda q, k=5: rag_db.rag.search_documentos_modelo(q, _ALT, k),
            _PREGUNTAS,
            "documentos",
            k=5,
        )
        # Ambos índices contienen el doc de diabetes: la pregunta q1 lo recupera
        # (recall 1/1) y q2 no (0), así que el recall agregado es 0.5 en ambos.
        assert activo["recall"] == pytest.approx(0.5)
        assert alt["recall"] == pytest.approx(0.5)

    def test_medir_canal_distingue_canal_sin_esperados(self, rag_db):
        """Una pregunta sin esperados en un canal no cuenta en el agregado."""
        m = comparar.medir_canal(
            rag_db.rag.search_factores, _PREGUNTAS, "factores", k=5
        )
        # q2 no tiene factores_esperados; q1 sí (calor/diabetes). Con el fake,
        # la query "diabetes" recupera el factor diabetes → recall 1.0.
        assert m["n_preguntas"] == 1
        assert m["recall"] == pytest.approx(1.0)

    def test_formatear_tabla_incluye_las_tres_configuraciones(self):
        filas = [
            {"nombre": "a", "metricas": {"factores": {"recall": 0.7}, "documentos": {"recall": 0.3}}},
            {"nombre": "b", "metricas": {"factores": {"recall": 0.8}, "documentos": {"recall": 0.4}}},
            {"nombre": "c", "metricas": {"factores": {"recall": 0.9}, "documentos": {"recall": 0.5}}},
        ]
        tabla = comparar.formatear_tabla(filas, k=5)
        assert "0.700" in tabla and "0.500" in tabla
