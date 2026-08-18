"""Tests de RAG-004: el set de evaluación y las métricas del script.

El script vive en scripts/evaluar_rag.py (fuera del paquete); se carga con
importlib para probar sus funciones sin ejecutar la CLI. Las definiciones de
recall@k y precision@k están documentadas en la docstring del propio script;
estos tests verifican que el código implementa exactamente esas definiciones:

  recall@k    = |recuperados ∩ esperados| / |esperados|  (solo si hay esperados)
  precision@k = |recuperados ∩ esperados| / k           (solo si hay esperados)

La recuperación end-to-end se prueba con un embedder falso sobre una BD
temporal (mismo patrón que test_rag_reindex.py): aquí se verifica el
plumbing (extracción de claves tipo/clave y rutas normalizadas), no la
calidad de los vectores.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

import climasafeai.db.rag as rag_mod
from climasafeai.db.manager import DBManager

_RAIZ = Path(__file__).resolve().parents[1]
_SCRIPT = _RAIZ / "scripts" / "evaluar_rag.py"
_EVAL_SET = _RAIZ / "data" / "rag" / "eval_set.json"

_spec = importlib.util.spec_from_file_location("evaluar_rag", _SCRIPT)
evaluar_rag = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(evaluar_rag)


# ── Fixtures ──────────────────────────────────────────────────────────────


class _FakeEmbedder:
    """Determinista y controlable: los textos con "diabetes" dan el vector
    [1,0,...] y el resto [0,1,0,...]. Así una query con "diabetes" tiene
    distancia 0 SOLO con los textos que la contienen y 1 con el resto: el
    top-k es determinista sin depender del modelo real."""

    def encode(self, texto: str):
        if "diabetes" in texto:
            return [1.0] + [0.0] * (rag_mod.EMBEDDING_DIM - 1)
        return [0.0, 1.0] + [0.0] * (rag_mod.EMBEDDING_DIM - 2)


@pytest.fixture(autouse=True)
def embedder_falso(monkeypatch):
    monkeypatch.setattr(rag_mod, "_get_embedder", lambda: _FakeEmbedder())


@pytest.fixture
def rag_db(tmp_path):
    """BD temporal con los factores cargados y un documento de papers."""
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
    assert db.rag.sync_documentos(project_root=tmp_path) == 1
    return db


# ── El set: 30+ entradas bien formadas y etiquetadas contra la KB ─────────


class TestSet:

    def test_tiene_30_mas_entradas(self):
        preguntas = evaluar_rag.cargar_set(_EVAL_SET)
        assert len(preguntas) >= 30

    def test_preguntas_bien_formadas(self):
        preguntas = evaluar_rag.cargar_set(_EVAL_SET)
        ids = [p["id"] for p in preguntas]
        assert len(ids) == len(set(ids)), "los ids deben ser únicos"
        for p in preguntas:
            assert p["id"], "id no vacío"
            assert isinstance(p["pregunta"], str) and p["pregunta"].strip()
            for canal in ("factores_esperados", "documentos_esperados"):
                assert canal in p
                assert isinstance(p[canal], list)
                assert all(isinstance(e, str) and e for e in p[canal])

    def test_etiquetas_validas_contra_la_base_de_conocimiento(self):
        """Cada etiqueta existe de verdad: factores en factores_riesgo.json y
        documentos en documentacion/ (rutas relativas al repo)."""
        factores_kb = {}
        with (_RAIZ / "data" / "factores_riesgo.json").open(encoding="utf-8") as fh:
            kb = json.load(fh)
        for tipo in ("calor", "frio"):
            for cat, factores in kb[tipo].items():
                for clave in factores:
                    factores_kb[f"{tipo}/{clave}"] = True

        preguntas = evaluar_rag.cargar_set(_EVAL_SET)
        for p in preguntas:
            for f in p["factores_esperados"]:
                assert f in factores_kb, f"{p['id']}: factor '{f}' no existe en la KB"
            for d in p["documentos_esperados"]:
                ruta = _RAIZ / d
                assert ruta.exists(), f"{p['id']}: documento '{d}' no existe"
                assert d.startswith("documentacion/"), f"{p['id']}: ruta '{d}' no es relativa"


# ── Métricas: las definiciones documentadas, con casos conocidos ──────────


class TestMetricas:

    def test_recall_y_precision_caso_conocido(self):
        esperados = ["calor/diabetes", "calor/diureticos_asa"]
        recuperados = [
            {"tipo": "calor", "clave": "diabetes"},
            {"tipo": "calor", "clave": "antipsicoticos"},
            {"tipo": "frio", "clave": "encamado"},
            {"tipo": "frio", "clave": "vive_solo"},
            {"tipo": "frio", "clave": "no_sale"},
        ]
        m = evaluar_rag.calcular_metricas(esperados, recuperados, k=5, canal="factores")
        # recall = 1/2 (solo diabetes está); precision = 1/5 (un slot de 5)
        assert m["recall"] == pytest.approx(0.5)
        assert m["precision"] == pytest.approx(0.2)
        assert m["esperados"] == 2
        assert m["encontrados"] == 1

    def test_recall_total_y_precision_maxima(self):
        esperados = ["calor/diabetes"]
        recuperados = [
            {"tipo": "calor", "clave": "diabetes"},
            {"tipo": "frio", "clave": "vive_solo"},
            {"tipo": "frio", "clave": "encamado"},
            {"tipo": "frio", "clave": "no_sale"},
            {"tipo": "frio", "clave": "vivienda_fria"},
        ]
        m = evaluar_rag.calcular_metricas(esperados, recuperados, k=5, canal="factores")
        assert m["recall"] == pytest.approx(1.0)
        assert m["precision"] == pytest.approx(0.2)  # 1 acierto / k=5 slots

    def test_sin_aciertos_cero(self):
        esperados = ["calor/diabetes"]
        recuperados = [{"tipo": "frio", "clave": "vive_solo"}] * 5
        m = evaluar_rag.calcular_metricas(esperados, recuperados, k=5, canal="factores")
        assert m["recall"] == pytest.approx(0.0)
        assert m["precision"] == pytest.approx(0.0)

    def test_sin_esperados_no_hay_metrica(self):
        """Sin etiqueta en un canal no se calcula nada (celda '-', no cuenta
        en el agregado) — así una pregunta solo de documentos no penaliza el
        recall/precision de factores."""
        m = evaluar_rag.calcular_metricas([], [], k=5, canal="factores")
        assert m is None

    def test_el_tipo_distingue_factores_iguales(self):
        """El mismo clave en calor y frío NO son el mismo ítem: 'frio/
        cardiovascular' no satisface un esperado 'calor/cardiovascular'."""
        esperados = ["calor/cardiovascular"]
        recuperados = [{"tipo": "frio", "clave": "cardiovascular"}] * 5
        m = evaluar_rag.calcular_metricas(esperados, recuperados, k=5, canal="factores")
        assert m["recall"] == pytest.approx(0.0)

    def test_documentos_se_normalizan_a_ruta_relativa(self):
        """La BD guarda rutas absolutas; la etiqueta es relativa al repo."""
        esperados = ["documentacion/papers/foo.md", "documentacion/papers/bar.md"]
        recuperados = [
            {"ruta": "/home/usuario/repo/documentacion/papers/foo.md"},
            {"ruta": "/home/usuario/repo/documentacion/papers/otro.md"},
        ]
        m = evaluar_rag.calcular_metricas(esperados, recuperados, k=5, canal="documentos")
        assert m["recall"] == pytest.approx(0.5)  # 1 de 2 esperados
        assert m["precision"] == pytest.approx(0.2)  # 1 acierto / k=5 slots

    def test_agregado_media_sobre_preguntas_con_esperados(self):
        resultados = [
            {
                "id": "a",
                "pregunta": "q1",
                "metricas": {
                    "factores": {"recall": 1.0, "precision": 0.2, "esperados": 1, "encontrados": 1},
                    "documentos": None,
                },
                "faltan": {"factores": [], "documentos": []},
            },
            {
                "id": "b",
                "pregunta": "q2",
                "metricas": {
                    "factores": {"recall": 0.0, "precision": 0.0, "esperados": 1, "encontrados": 0},
                    "documentos": {"recall": 0.5, "precision": 0.1, "esperados": 2, "encontrados": 1},
                },
                "faltan": {"factores": ["calor/x"], "documentos": ["doc/y.md"]},
            },
        ]
        agg = evaluar_rag.agregar(resultados)
        assert agg["factores"]["n_preguntas"] == 2
        assert agg["factores"]["recall"] == pytest.approx(0.5)  # (1.0 + 0.0)/2
        assert agg["documentos"]["n_preguntas"] == 1  # la 'a' no tiene esperados
        assert agg["documentos"]["recall"] == pytest.approx(0.5)


# ── Integración: evaluar_pregunta contra el RAG real (embedder falso) ─────


class TestEvaluacionIntegracion:

    def test_clave_de_factor_usa_tipo_clave(self):
        res = {"tipo": "calor", "clave": "diabetes", "texto": "diabetes", "distance": 0.1}
        assert evaluar_rag.clave_recuperado(res, "factores") == "calor/diabetes"

    def test_normalizar_ruta(self):
        assert (
            evaluar_rag.normalizar_ruta("/home/x/documentacion/papers/foo.md")
            == "documentacion/papers/foo.md"
        )
        assert evaluar_rag.normalizar_ruta("documentacion/riesgo/a.md") == "documentacion/riesgo/a.md"

    def test_evaluar_pregunta_devuelve_metricas_de_ambos_canales(self, rag_db):
        pregunta = {
            "id": "test-001",
            "pregunta": "diabetes y temperatura",
            "factores_esperados": ["calor/diabetes"],
            "documentos_esperados": ["documentacion/papers/doc.md"],
        }
        r = evaluar_rag.evaluar_pregunta(rag_db.rag, pregunta, k=5)
        assert r["id"] == "test-001"
        # Con el embedder falso, la query con "diabetes" solo empata con el
        # factor diabetes y con el único doc papers (ambos contienen la
        # palabra): entran en el top-k de forma determinista. Se verifica el
        # plumbing (extracción de claves), no los vectores.
        assert r["metricas"]["factores"] is not None
        assert r["metricas"]["documentos"] is not None
        for canal in ("factores", "documentos"):
            m = r["metricas"][canal]
            assert 0.0 <= m["recall"] <= 1.0
            assert 0.0 <= m["precision"] <= 1.0
            assert m["encontrados"] == m["esperados"]

    def test_evaluar_pregunta_faltan_lista_lo_no_recuperado(self, rag_db):
        pregunta = {
            "id": "test-002",
            "pregunta": "diabetes y algo que no existe",
            "factores_esperados": ["calor/diabetes"],
            "documentos_esperados": ["documentacion/papers/no-existe.md"],
        }
        r = evaluar_rag.evaluar_pregunta(rag_db.rag, pregunta, k=5)
        assert r["faltan"]["factores"] == []  # el factor diabetes entra (dist. 0)
        assert r["faltan"]["documentos"] == ["documentacion/papers/no-existe.md"]
