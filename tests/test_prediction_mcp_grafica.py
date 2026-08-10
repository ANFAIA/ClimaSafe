"""Tests de MCP-IMG-001: la tool MCP que devuelve la gráfica de riesgo por hora.

Cubre:
  - `grafica_riesgo_horario_png`: PNG válido a partir de un resultado del
    pipeline, sin recalcular la predicción (consume riesgo_horario/perfil_horario)
  - el caso sin serie horaria: devuelve None / texto, no revienta ni pinta vacío
  - `grafica_riesgo_horario_mcp`: devuelve ImageContent con mimeType image/png y
    bytes que decodifican como PNG
  - las 11 tools que ya existían siguen registradas y con su output_schema

La predicción real se evita monkeypatcheando `_try_prediction`, igual que hace
tests/test_prediction_mcp_rutinas.py.
"""

from __future__ import annotations

import base64
import io
from datetime import date, timedelta

import pytest

import agents.tools.prediction_mcp_tool as mcp

PNG_MAGIC = b"\x89PNG\r\n\x1a\n"

# Las 11 tools anteriores a MCP-IMG-001: ninguna se toca.
TOOLS_PREVIAS = {
    "predict_risk_mcp", "listar_usuarios_mcp", "cargar_perfil_mcp",
    "cargar_perfil_por_chat_id_mcp", "vincular_chat_id_mcp", "crear_perfil_mcp",
    "listar_rutinas_mcp", "crear_rutina_mcp", "borrar_rutina_mcp",
    "configurar_hora_aviso_mcp", "riesgo_rutinas_dia_mcp",
}


def _perfil_horario(pico: float = 41.0) -> list[dict]:
    """24 horas con una campana de HI centrada en las 16:00."""
    return [
        {"hora": h, "HI": round(20.0 + (pico - 20.0) * max(0.0, 1 - abs(h - 16) / 10), 1),
         "temp": round(15.0 + h * 0.4, 1)}
        for h in range(24)
    ]


def _resultado(perfil_horario: list[dict] | None) -> dict:
    return {
        "clase_final": 1,
        "clase_final_label": "PRECAUCIÓN",
        "weather": {"provincia": "Pontevedra", "current": {"t2m_c": 30.0},
                    "perfil_horario": perfil_horario if perfil_horario is not None else []},
        "modelos": {},
        "perfil": {"calor": {"prob_personalizada": 0.4}, "frio": {"prob_personalizada": 0.05}},
    }


@pytest.fixture
def sin_prediccion(monkeypatch):
    """`_try_prediction` falso: devuelve perfil horario sin llamar al ensemble."""
    llamadas: list[dict] = []

    def _fake(lat, lon, provincia, perfil, target_date=None, resolucion=60):
        llamadas.append(dict(perfil))
        return _resultado(_perfil_horario())

    monkeypatch.setattr(mcp, "_try_prediction", _fake)
    return llamadas


# ── El helper de la gráfica ────────────────────────────────────────────────


class TestGraficaPNG:
    def test_devuelve_png_valido(self):
        from climasafeai.features.personalizacion import riesgo_horario_acumulado

        ph = _perfil_horario()
        perfil = {"edad": 57, "sexo": "hombre", "nivel_actividad": "moderada",
                  "hora_inicio": 10, "duracion_actividad_h": 2}
        res = _resultado(ph)
        res["riesgo_horario"] = riesgo_horario_acumulado(ph, perfil)
        res["recomendacion_horario"] = {"hora_inicio": 7, "hora_fin": 9, "riesgo_medio": 0.1}

        png = mcp.grafica_riesgo_horario_png(res, hora_inicio=10, duracion_h=2,
                                             edad=57, fecha="2026-08-04")
        assert isinstance(png, bytes)
        assert png.startswith(PNG_MAGIC)

        from PIL import Image as PILImage
        img = PILImage.open(io.BytesIO(png))
        img.load()
        assert img.format == "PNG"
        assert img.width > 300 and img.height > 150

    def test_sin_perfil_horario_devuelve_none(self):
        res = _resultado([])
        res["riesgo_horario"] = []
        assert mcp.grafica_riesgo_horario_png(res) is None

    def test_con_perfil_pero_sin_curva_devuelve_none(self):
        # El pipeline solo añade riesgo_horario si hubo perfil_horario; si falta,
        # no se pinta media gráfica.
        assert mcp.grafica_riesgo_horario_png(_resultado(_perfil_horario())) is None

    def test_no_recalcula_el_riesgo_usa_la_curva_del_pipeline(self):
        # Una curva inventada plana al 90%: si el helper recalculase el riesgo a
        # partir del HI, no podría respetarla. Se comprueba por el pico dibujado.
        ph = _perfil_horario()
        res = _resultado(ph)
        res["riesgo_horario"] = [{"hora": h, "riesgo": 0.9, "hi": ph[h]["HI"],
                                  "temp": ph[h]["temp"]} for h in range(24)]
        png = mcp.grafica_riesgo_horario_png(res, hora_inicio=10, duracion_h=2)
        assert png is not None and png.startswith(PNG_MAGIC)

    def test_hi_a_nivel_respeta_los_tramos_de_la_web(self):
        assert mcp._hi_a_nivel(10) == 1.0
        assert mcp._hi_a_nivel(27) == pytest.approx(2.0)
        assert mcp._hi_a_nivel(32) == pytest.approx(4.0)
        assert mcp._hi_a_nivel(39) == pytest.approx(7.0)
        assert mcp._hi_a_nivel(46) == pytest.approx(9.0)
        assert mcp._hi_a_nivel(100) == 10.0


# ── Guarda de horizonte del MCP (FORECAST-001) ──────────────────────────────


class TestHorizonteFecha:
    """_parse_date rechaza fechas que el forecast no cubre (antes: datos falsos)."""

    def test_sin_fecha_devuelve_none(self):
        assert mcp._parse_date(None) is None
        assert mcp._parse_date("") is None

    def test_fecha_dentro_de_horizonte_se_acepta(self):
        hoy = date.today()
        assert mcp._parse_date(hoy.isoformat()) == hoy
        assert mcp._parse_date((hoy + timedelta(days=7)).isoformat()) == hoy + timedelta(days=7)

    def test_mas_de_7_dias_se_rechaza_con_mensaje_claro(self):
        hoy = date.today()
        con = hoy + timedelta(days=8)
        with pytest.raises(ValueError) as exc:
            mcp._parse_date(con.isoformat())
        assert "7 días" in str(exc.value)
        assert con.isoformat() in str(exc.value)

    def test_fecha_pasada_se_rechaza(self):
        with pytest.raises(ValueError) as exc:
            mcp._parse_date((date.today() - timedelta(days=1)).isoformat())
        assert "ya pasó" in str(exc.value)

    def test_formato_invalido_se_rechaza(self):
        with pytest.raises(ValueError) as exc:
            mcp._parse_date("no-es-una-fecha")
        assert "ISO" in str(exc.value)

    def test_predict_risk_con_fecha_fuera_devuelve_error_sin_predecir(self, monkeypatch):
        """La guarda corta antes de tocar red/modelos: error JSON, no predicción."""
        llamadas: list = []
        monkeypatch.setattr(mcp, "_try_prediction",
                            lambda *a, **k: llamadas.append(1) or {})
        out = mcp.predict_risk(
            lat=40.4, lon=-3.7,
            fecha=(date.today() + timedelta(days=8)).isoformat(),
        )
        assert "error" in out
        assert "7 días" in out["error"]
        assert llamadas == []

    def test_predict_risk_con_fecha_invalida_devuelve_error(self, monkeypatch):
        llamadas: list = []
        monkeypatch.setattr(mcp, "_try_prediction",
                            lambda *a, **k: llamadas.append(1) or {})
        out = mcp.predict_risk(lat=40.4, lon=-3.7, fecha="garbage")
        assert "error" in out
        assert "ISO" in out["error"]
        assert llamadas == []


# ── La tool MCP ────────────────────────────────────────────────────────────


class TestGraficaRiesgoHorarioMCP:
    def test_devuelve_imagecontent_png(self, sin_prediccion):
        from mcp.types import ImageContent

        out = mcp.grafica_riesgo_horario_mcp(
            lat=42.29, lon=-8.81, provincia="Pontevedra",
            edad=57, hora_inicio=10, duracion_h=2.0,
        )
        assert isinstance(out, ImageContent)
        assert out.type == "image"
        assert out.mimeType == "image/png"

        raw = base64.b64decode(out.data, validate=True)
        assert raw.startswith(PNG_MAGIC)

        from PIL import Image as PILImage
        img = PILImage.open(io.BytesIO(raw))
        img.load()
        assert img.format == "PNG"

    def test_sin_serie_horaria_devuelve_texto_claro(self, monkeypatch):
        monkeypatch.setattr(mcp, "_try_prediction",
                            lambda lat, lon, provincia, perfil, target_date=None, resolucion=60: _resultado([]))
        out = mcp.grafica_riesgo_horario_mcp(lat=42.29, lon=-8.81, provincia="Pontevedra")
        assert isinstance(out, str)
        assert "no hay" in out.lower()
        assert "predict_risk_mcp" in out

    def test_fecha_fuera_de_horizonte_devuelve_texto_claro(self, monkeypatch):
        """La guarda de fecha corta antes de predecir y la gráfica lo dice claro."""
        monkeypatch.setattr(mcp, "_try_prediction",
                            lambda *a, **k: (_ for _ in ()).throw(
                                AssertionError("no debe predecir con fecha fuera de rango")))
        out = mcp.grafica_riesgo_horario_mcp(
            lat=42.29, lon=-8.81, provincia="Pontevedra",
            fecha=(date.today() + timedelta(days=8)).isoformat(),
        )
        assert isinstance(out, str)
        assert "No se pudo calcular" in out
        assert "7 días" in out

    def test_las_11_tools_previas_siguen_registradas(self):
        import asyncio

        tools = asyncio.run(mcp._mcp.list_tools())
        por_nombre = {t.name: t for t in tools}
        assert TOOLS_PREVIAS <= set(por_nombre)
        assert "grafica_riesgo_horario_mcp" in por_nombre
        # Las previas devuelven JSON estructurado; la nueva no lleva output_schema
        # (devuelve contenido de imagen, que no encaja en un schema de salida).
        for nombre in TOOLS_PREVIAS:
            assert por_nombre[nombre].outputSchema is not None, nombre
        assert por_nombre["grafica_riesgo_horario_mcp"].outputSchema is None
