"""Tests de MCP-APPS-001: predict_risk_mcp sirve una vista MCP App (HTML) y
sigue sirviendo JSON para hosts sin soporte de apps.

Cubre:
  - `_html_vista_predict_risk`: HTML autocontenido servido desde el backend
    Python (agents/tools/mcp_apps_vista.py) — el CSS y el JS de visualización
    viven inline en la plantilla, no hay fichero .js suelto en el repo —,
    lleva `profile=mcp-app` y el puente postMessage JSON-RPC
  - la tool `predict_risk_mcp` declara `_meta.ui.resourceUri` y SIGUE
    devolviendo JSON (degradación para clientes sin soporte de apps)
  - el recurso `ui://prediccion-riesgo` está registrado con el MIME type de
    MCP Apps y resources/read devuelve el HTML del último resultado
  - las 11 tools previas siguen registradas y ninguna otra tool lleva meta ui
    (no se migra nada más allá del estudio)
  - el estudio de arquitectura existe en documentacion/mcp_apps_estudio.md
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

import agents.tools.mcp_apps_vista as vista
import agents.tools.prediction_mcp_tool as mcp

REPO = Path(__file__).resolve().parents[1]

# Las 11 tools previas a este ticket: ninguna se toca (criterio 6).
TOOLS_PREVIAS = {
    "listar_usuarios_mcp",
    "cargar_perfil_mcp",
    "cargar_perfil_por_chat_id_mcp",
    "vincular_chat_id_mcp",
    "crear_perfil_mcp",
    "listar_rutinas_mcp",
    "crear_rutina_mcp",
    "borrar_rutina_mcp",
    "configurar_hora_aviso_mcp",
    "riesgo_rutinas_dia_mcp",
    "grafica_riesgo_horario_mcp",
}

pytestmark = pytest.mark.skipif(not mcp._HAS_MCP, reason="mcp no instalado")


def _perfil_horario() -> list[dict]:
    return [
        {
            "hora": h,
            "HI": round(20.0 + 0.7 * max(0.0, 1 - abs(h - 16) / 10), 1),
            "temp": round(15.0 + h * 0.4, 1),
        }
        for h in range(24)
    ]


def _resultado_ui() -> dict:
    return {
        "clase_final": 1,
        "clase_final_label": "PRECAUCIÓN",
        "explicacion": {"modelo_determinante": "XGBoost_calor"},
        "modelos": {"XGBoost_calor": {"conformal_confianza": "media"}},
        "peor_hora": {"hora": 16, "prob": 0.42},
        "weather": {"provincia": "Pontevedra", "perfil_horario": _perfil_horario()},
        "riesgo_horario": [
            {
                "hora": h,
                "riesgo": 0.4,
                "hi": _perfil_horario()[h]["HI"],
                "temp": _perfil_horario()[h]["temp"],
            }
            for h in range(24)
        ],
        "perfil_usuario": {"edad": 57, "hora_inicio": 10, "duracion_actividad_h": 2},
    }


@pytest.fixture
def sin_prediccion(monkeypatch):
    """_try_prediction falso: devuelve el resultado sin tocar el ensemble."""

    def _fake(lat, lon, provincia, perfil, target_date=None, resolucion=60):
        return _resultado_ui()

    monkeypatch.setattr(mcp, "_try_prediction", _fake)


# ── El HTML del recurso ui:// ──────────────────────────────────────────────


class TestHtmlVistaPredictRisk:
    def test_la_vista_vive_en_el_backend_python(self):
        """La plantilla es un módulo Python: el CSS y el JS de la vista son
        constantes del backend, no un fichero .js suelto en la web."""
        assert hasattr(vista, "html_vista_predict_risk")
        assert hasattr(vista, "_VISTA_JS") and hasattr(vista, "_VISTA_CSS")
        # No queda ningún .js huérfano de esta feature en la web.
        assert not (REPO / "chat" / "static" / "js").exists()

    def test_lleva_profile_mcp_app_y_puente_postmessage(self):
        html = mcp._html_vista_predict_risk(_resultado_ui())
        assert 'content="mcp-app"' in html
        assert "postMessage" in html
        assert "jsonrpc" in html and "notifications/initialized" in html

    def test_inyecta_los_datos_y_llama_a_las_funciones_de_la_web(self):
        html = mcp._html_vista_predict_risk(_resultado_ui())
        assert "window.RIESGO_DATA" in html
        assert '"clase_final_label"' in html
        assert "mostrarFinal(RIESGO_DATA)" in html
        assert "mostrarGraficaRiesgo(RIESGO_DATA)" in html
        # Las funciones de la vista vienen del JS inline del backend Python.
        assert "function mostrarFinal" in html
        assert "function mostrarGraficaRiesgo" in html

    def test_el_js_de_la_vista_tiene_las_funciones_de_la_web(self):
        """El JS inline de la plantilla reproduce las firmas de index.html."""
        js = vista._VISTA_JS
        assert "function mostrarFinal" in js
        assert "function mostrarGraficaRiesgo" in js
        assert "hiToNivel" in js
        assert "nivelLabel" in js

    def test_sin_resultado_devuelve_mensaje(self, monkeypatch):
        monkeypatch.setitem(mcp._UI_ESTADO, "ultimo", None)
        low = mcp._mcp._mcp_server
        from mcp.types import ReadResourceRequest

        res = asyncio.run(_read_resource(low, mcp._UI_RESOURCE_URI))
        assert "Aún no hay predicción" in res.text


# ── La tool predict_risk_mcp ────────────────────────────────────────────────


class TestPredictRiskMCP:
    def test_sigue_devolviendo_json_sin_soporte_de_apps(self, sin_prediccion):
        """Degradación (criterio 5): un host sin apps recibe el JSON de siempre."""
        out = mcp.predict_risk_mcp(lat=42.29, lon=-8.81, provincia="Pontevedra")
        assert isinstance(out, str)
        data = json.loads(out)
        assert data["clase_final_label"] == "PRECAUCIÓN"

    def test_guarda_el_resultado_para_el_recurso_ui(self, sin_prediccion):
        mcp.predict_risk_mcp(lat=42.29, lon=-8.81, provincia="Pontevedra")
        assert mcp._UI_ESTADO["ultimo"] is not None
        assert mcp._UI_ESTADO["ultimo"]["clase_final_label"] == "PRECAUCIÓN"

    def test_tool_declara_meta_ui_resourceuri(self):
        low = mcp._mcp._mcp_server
        from mcp.types import ListToolsRequest

        listed = asyncio.run(_list_tools(low, ListToolsRequest))
        por_nombre = {t.name: t for t in listed}
        meta = por_nombre["predict_risk_mcp"].meta or {}
        assert meta.get("ui", {}).get("resourceUri") == mcp._UI_RESOURCE_URI
        # Ninguna otra tool se migró: solo predict_risk_mcp lleva meta ui.
        for nombre, t in por_nombre.items():
            if nombre == "predict_risk_mcp":
                continue
            assert not (t.meta or {}).get("ui"), nombre

    def test_recurso_ui_registrado_con_mime_de_mcp_apps(self):
        low = mcp._mcp._mcp_server
        from mcp.types import ListResourcesRequest

        listed = asyncio.run(_list_resources(low, ListResourcesRequest))
        recursos = {str(r.uri): r for r in listed}
        assert mcp._UI_RESOURCE_URI in recursos
        assert recursos[mcp._UI_RESOURCE_URI].mimeType == vista.UI_MIME_TYPE

    def test_resources_read_devuelve_el_html_del_ultimo_resultado(self, sin_prediccion):
        mcp.predict_risk_mcp(lat=42.29, lon=-8.81, provincia="Pontevedra")
        low = mcp._mcp._mcp_server
        from mcp.types import ReadResourceRequest

        res = asyncio.run(_read_resource(low, mcp._UI_RESOURCE_URI))
        assert res.mimeType == vista.UI_MIME_TYPE
        assert 'content="mcp-app"' in res.text
        assert "PRECAUCIÓN" in res.text

    def test_las_11_tools_previas_siguen_registradas(self):
        from mcp.types import ListToolsRequest

        listed = asyncio.run(_list_tools(mcp._mcp._mcp_server, ListToolsRequest))
        por_nombre = {t.name: t for t in listed}
        assert TOOLS_PREVIAS <= set(por_nombre)
        # Sin fugas: sigue habiendo 12 tools (11 previas + predict_risk_mcp).
        assert len(por_nombre) == len(TOOLS_PREVIAS) + 1


# ── Módulo compartido y estudio ─────────────────────────────────────────────


class TestCompartidoYEstudio:
    def test_estudio_de_arquitectura_existe(self):
        estudio = REPO / "documentacion" / "mcp_apps_estudio.md"
        assert estudio.exists(), "el estudio de arquitectura (criterio 4) debe existir"
        texto = estudio.read_text(encoding="utf-8")
        assert "resourceUri" in texto and "ui://" in texto
        assert "postMessage" in texto
        assert "Python" in texto


# ── Helpers para hablar con el servidor lowlevel ─────────────────────────────


async def _list_tools(low, request_type):
    handler = low.request_handlers[request_type]
    result = handler(request_type())
    if asyncio.iscoroutine(result):
        result = await result
    return result.root.tools


async def _list_resources(low, request_type):
    handler = low.request_handlers[request_type]
    result = handler(request_type())
    if asyncio.iscoroutine(result):
        result = await result
    return result.root.resources


async def _read_resource(low, uri):
    from mcp.types import ReadResourceRequest

    req = ReadResourceRequest(params={"uri": uri})
    handler = low.request_handlers[ReadResourceRequest]
    result = handler(req)
    if asyncio.iscoroutine(result):
        result = await result
    return result.root.contents[0]
