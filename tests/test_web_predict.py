"""Tests del endpoint /api/predict de la web (WEB-003).

El fallo que blindan: el endpoint escribía en SQLite todo lo que le mandaba el
frontend, así que un campo que no fuera columna de `perfiles` —`peso`, por ejemplo,
que no lo es— tumbaba la petición entera con un 500 mudo, aunque la predicción
hubiera salido bien.
"""

import pathlib
import tempfile

import pytest

from climasafeai.db.manager import CampoDesconocidoError, DBManager


@pytest.fixture
def db():
    """Una BD limpia con el esquema real del proyecto."""
    gestor = DBManager(tempfile.mktemp(suffix=".db"))
    esquema = pathlib.Path("data/schema.sql").read_text(encoding="utf-8")
    with gestor.conn() as c:
        c.executescript(esquema)
    return gestor


def test_columnas_perfiles_se_leen_del_esquema(db):
    cols = db.columnas_perfiles()
    assert {"alias", "edad", "sexo", "porcentaje_grasa", "aclimatado"} <= cols
    # Los dos campos que el modelo no usa y que la tabla nunca tuvo
    assert "peso" not in cols
    assert "altura" not in cols


def test_perfil_valido_se_guarda(db):
    pid = db.crear_perfil({"alias": "alex", "edad": 21, "sexo": "hombre"})
    assert db.obtener_perfil(pid)["edad"] == 21


def test_campo_desconocido_se_rechaza_con_su_nombre(db):
    """El error dice QUÉ campo falla; antes salía un OperationalError de sqlite."""
    with pytest.raises(CampoDesconocidoError) as exc:
        db.crear_perfil({"alias": "b", "edad": 21, "peso": 86})
    assert "peso" in str(exc.value)
    assert exc.value.campos == ["peso"]


def test_varios_campos_desconocidos_se_listan(db):
    with pytest.raises(CampoDesconocidoError) as exc:
        db.crear_perfil({"alias": "c", "peso": 86, "altura": 180})
    assert exc.value.campos == ["altura", "peso"]


def test_actualizar_perfil_valida_igual(db):
    pid = db.crear_perfil({"alias": "d", "edad": 30})
    with pytest.raises(CampoDesconocidoError):
        db.actualizar_perfil(pid, {"peso": 70})


def test_arrays_no_cuentan_como_columnas(db):
    """comorbilidades y farmacos van a tablas aparte, no son columnas de perfiles."""
    pid = db.crear_perfil({
        "alias": "e", "edad": 40,
        "comorbilidades": ["cardiovascular"],
        "farmacos": ["diureticos_asa"],
        "situacion_social": ["vive_solo"],
    })
    perfil = db.obtener_perfil(pid)
    assert perfil["comorbilidades"] == ["cardiovascular"]
    assert perfil["farmacos"] == ["diureticos_asa"]


def test_endpoint_perfil_tambien_valida(db, monkeypatch):
    """WEB-003: /api/perfil tenia el mismo 500 mudo que /api/predict.

    Desde WEB-005 el campo desconocido sale como HTTPException 400 en vez de un
    dict con 'error' y HTTP 200. Este test llama a la corutina directamente, sin
    TestClient, así que ve la excepción en crudo; el cuerpo que recibe el cliente
    lo cubre tests/test_web_rutinas.py.
    """
    import chat.app as web
    monkeypatch.setattr(web, "_db", db)
    import asyncio

    from fastapi import HTTPException

    ok = asyncio.run(web.api_save_perfil({"alias": "z", "edad": 30}))
    assert "perfil_id" in ok and "error" not in ok

    with pytest.raises(HTTPException) as exc:
        asyncio.run(web.api_save_perfil({"alias": "y", "peso": 80}))
    assert exc.value.status_code == 400
    assert "peso" in exc.value.detail


def test_api_predict_semanal_devuelve_la_serie(monkeypatch):
    """FORECAST-001: el endpoint /api/predict/semanal devuelve la serie con banda.

    Se mockea `climasafeai.models.ensemble.prediccion_semanal` (el endpoint la
    importa dentro de la función): lo que se prueba es el contrato del endpoint,
    no el cálculo (que cubren tests/test_ensemble.py).
    """
    import asyncio

    import chat.app as web
    import climasafeai.models.ensemble as ens

    def _fake(lat=None, lon=None, provincia="Madrid", perfil=None, resolucion=60):
        assert lat == 40.4 and lon == -3.7
        return {
            "horizonte_dias": 7,
            "completo": True,
            "forecast_hasta": "2026-08-16",
            "dias": [
                {"fecha": "2026-08-10", "prob": 0.54, "clase": "PRECAUCION",
                 "confianza_conformal": "alta", "set_size_conformal": 1,
                 "banda": [0.49, 0.59]},
            ],
            "banda_origen": "conformal",
        }

    monkeypatch.setattr(ens, "prediccion_semanal", _fake)
    out = asyncio.run(web.api_predict_semanal(
        {"lat": 40.4, "lon": -3.7, "provincia": "Madrid", "perfil": {"edad": 57}}
    ))
    assert out["horizonte_dias"] == 7
    assert out["completo"] is True
    assert out["dias"][0]["banda"] == [0.49, 0.59]
    # FORECAST-004: el endpoint pasa `banda_origen` cuando el cálculo la da.
    assert out["banda_origen"] == "conformal"


def test_api_predict_semanal_incompleto_avisa_hasta_donde(monkeypatch):
    """FORECAST-004: el endpoint semanal reporta `completo=False` y
    `forecast_hasta` cuando el forecast no cubre los 7 días: la UI puede
    avisar explícitamente sin extrapolar en silencio."""
    import asyncio

    import chat.app as web
    import climasafeai.models.ensemble as ens

    def _fake(lat=None, lon=None, provincia="Madrid", perfil=None, resolucion=60):
        return {
            "horizonte_dias": 7,
            "completo": False,
            "forecast_hasta": "2026-08-12",
            "dias": [
                {"fecha": "2026-08-10", "prob": 0.54, "clase": "PRECAUCION",
                 "confianza_conformal": "alta", "set_size_conformal": 1,
                 "banda": [0.49, 0.59]},
                {"fecha": "2026-08-11", "prob": 0.61, "clase": "PRECAUCION",
                 "confianza_conformal": "media", "set_size_conformal": 2,
                 "banda": [0.46, 0.76]},
                {"fecha": "2026-08-12", "prob": 0.40, "clase": "SEGURO",
                 "confianza_conformal": "alta", "set_size_conformal": 1,
                 "banda": [0.35, 0.45]},
            ],
            "banda_origen": "conformal",
        }

    monkeypatch.setattr(ens, "prediccion_semanal", _fake)
    out = asyncio.run(web.api_predict_semanal(
        {"lat": 40.4, "lon": -3.7, "provincia": "Madrid", "perfil": {"edad": 57}}
    ))
    assert out["completo"] is False
    assert out["forecast_hasta"] == "2026-08-12"
    assert len(out["dias"]) == 3
    assert out["horizonte_dias"] == 7
    # cada día predicho conserva su banda; el resto no está en la serie
    assert all(d["banda"] is not None for d in out["dias"])
    assert out["banda_origen"] == "conformal"
