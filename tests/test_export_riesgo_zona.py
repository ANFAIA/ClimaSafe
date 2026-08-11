"""Tests de MAPA-001: exportación del mapa de riesgo por zona.

Cubre los dos endpoints nuevos (PNG del overlay y GeoJSON de las celdas) sin
tocar red: `riesgo_zona_grid` se sustituye por un doble que devuelve un grid
fijo, igual que hacen los tests de la web con predict_ensemble.
"""

import json

import pytest
from fastapi.testclient import TestClient

from chat.app import app
from climasafeai.data.grid_risk import celdas_a_featurecollection, render_riesgo_png

PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


def _celdas_fake() -> list[dict]:
    """Grid 3x3 con paso ~1 km (0.009°) sobre Madrid."""
    celdas = []
    riesgos = [(0, "SEGURO"), (1, "PRECAUCION"), (2, "PELIGRO")]
    for i, lat in enumerate([40.41, 40.419, 40.428]):
        for j, lon in enumerate([-3.71, -3.701, -3.692]):
            riesgo, label = riesgos[(i + j) % 3]
            celdas.append({
                "lat": lat,
                "lon": lon,
                "hi": 33.5,
                "riesgo": riesgo,
                "riesgo_label": label,
            })
    return celdas


def _resultado_fake() -> dict:
    return {
        "center": {"lat": 40.419, "lon": -3.7005},
        "perfil_usado": "personalizado",
        "perfil_label": "Personalizado",
        "stats": {
            "total_celdas": 9,
            "seguro": 3,
            "precaucion": 3,
            "peligro": 3,
            "pct_peligro": 33.3,
        },
        "celdas": _celdas_fake(),
        "resumen_horario": {"hi_peak": 33.5, "hora_pico": 16},
        "target_date": "2026-08-11",
    }


@pytest.fixture(autouse=True)
def fake_riesgo_zona(monkeypatch):
    """Sustituye el grid real (que descarga meteo) por un resultado fijo."""
    import climasafeai.data.grid_risk as grid_risk_module

    monkeypatch.setattr(grid_risk_module, "riesgo_zona_grid", lambda **kw: _resultado_fake())


@pytest.fixture
def client():
    return TestClient(app)


# ── Builder puro: GeoJSON ──────────────────────────────────────────────────


def test_featurecollection_celdas_con_clase_y_hi():
    fc = celdas_a_featurecollection(_celdas_fake())

    assert fc["type"] == "FeatureCollection"
    assert len(fc["features"]) == 9
    for f in fc["features"]:
        assert f["type"] == "Feature"
        # Geometría: Polygon cerrado con 5 posiciones [lon, lat]
        coords = f["geometry"]["coordinates"][0]
        assert f["geometry"]["type"] == "Polygon"
        assert len(coords) == 5
        assert coords[0] == coords[-1]  # anillo cerrado
        for lon, lat in coords:
            assert -180 <= lon <= 180
            assert -90 <= lat <= 90
        # Properties: clase de riesgo + HI pico
        assert f["properties"]["riesgo"] in (0, 1, 2)
        assert f["properties"]["riesgo_label"] in ("SEGURO", "PRECAUCION", "PELIGRO")
        assert f["properties"]["hi_pico"] == 33.5


def test_featurecollection_es_json_valido_y_serializable():
    """El GeoJSON debe poder re-serializarse y parsearse sin errores."""
    fc = celdas_a_featurecollection(_celdas_fake())
    texto = json.dumps(fc)
    recargado = json.loads(texto)
    assert recargado == fc


def test_featurecollection_celda_unica_usa_rectangulo_default():
    celda = [{"lat": 40.41, "lon": -3.71, "hi": 30.0, "riesgo": 0, "riesgo_label": "SEGURO"}]
    fc = celdas_a_featurecollection(celda)
    coords = fc["features"][0]["geometry"]["coordinates"][0]
    # Una celda sola no tiene vecinos para deducir el paso: usa 0.01° (1.1 km)
    assert round(coords[0][0], 6) == round(-3.71 - 0.005, 6)
    assert round(coords[2][0], 6) == round(-3.71 + 0.005, 6)


# ── Render PNG ─────────────────────────────────────────────────────────────


def test_render_png_devuelve_bytes_png():
    png = render_riesgo_png(
        _celdas_fake(),
        stats=_resultado_fake()["stats"],
        center=_resultado_fake()["center"],
        resumen=_resultado_fake()["resumen_horario"],
        perfil_label="Personalizado",
    )
    assert isinstance(png, bytes)
    assert png.startswith(PNG_MAGIC)


# ── Endpoints ──────────────────────────────────────────────────────────────


def test_endpoint_geojson_descarga_celdas(client):
    r = client.post("/api/riesgo-zona/export/geojson", json={"lat": 40.41, "lon": -3.71})
    assert r.status_code == 200
    assert "attachment" in r.headers.get("content-disposition", "")
    body = r.json()
    assert body["type"] == "FeatureCollection"
    assert len(body["features"]) == 9
    props = body["features"][0]["properties"]
    assert {"riesgo", "riesgo_label", "hi_pico"} <= set(props)


def test_endpoint_geojson_usa_perfil_personalizado(client):
    """Con perfil en el body se manda a riesgo_zona_grid (como el POST base)."""
    r = client.post("/api/riesgo-zona/export/geojson", json={
        "lat": 40.41, "lon": -3.71, "perfil_id": "adulto",
        "perfil": {"edad": 40, "hora_inicio": 10, "duracion_actividad_h": 2},
    })
    assert r.status_code == 200
    assert len(r.json()["features"]) == 9


def test_endpoint_png_descarga_image_png(client):
    r = client.post("/api/riesgo-zona/export/png", json={"lat": 40.41, "lon": -3.71})
    assert r.status_code == 200
    assert r.headers["content-type"] == "image/png"
    assert "attachment" in r.headers.get("content-disposition", "")
    assert r.content.startswith(PNG_MAGIC)


def test_endpoint_error_sin_coordenadas_es_400(client):
    for ep in ("/api/riesgo-zona/export/geojson", "/api/riesgo-zona/export/png"):
        r = client.post(ep, json={})
        assert r.status_code == 400
        assert "lat y lon requeridos" in r.json()["error"]
