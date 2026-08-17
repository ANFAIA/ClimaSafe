"""MCP-004 — Los servidores MCP exponen annotations del spec 2025-06-18+.

Comprueba que todas las tools registradas por ambos servidores llevan `title`
y que las de lectura llevan `readOnlyHint` y las destructivas `destructiveHint`.

Método: inspección en runtime de las tools registradas en el `_tool_manager`
de FastMCP (mcp>=1.28.1), donde cada tool expone `.annotations` con los campos
`title`, `readOnlyHint` y `destructiveHint`.
"""

import json

import pytest

from agents.tools import prediction_mcp_tool as pred
from agents.tools import factors_mcp_tool as factors
def _tools(module):
    return {t.name: t for t in module._mcp._tool_manager.list_tools()}


# Listas explícitas por nombre (fuente de verdad de qué tool es de lectura).
PRED_READ = {
    "predict_risk_mcp",
    "listar_usuarios_mcp",
    "cargar_perfil_mcp",
    "cargar_perfil_por_chat_id_mcp",
    "listar_rutinas_mcp",
    "riesgo_rutinas_dia_mcp",
    "grafica_riesgo_horario_mcp",
}
PRED_DESTRUCTIVE = {"borrar_rutina_mcp"}

FACTORS_READ = {
    "get_factors_mcp",
    "pending_factors_mcp",
    "check_acclimatization_mcp",
    "search_factors_mcp",
    "search_documentos_mcp",
    "search_all_mcp",
    "ask_rag_mcp",
    "ask_qwen_rag_mcp",
    "qwen_raw_mcp",
}
FACTORS_DESTRUCTIVE = {"reject_factor_mcp"}


@pytest.mark.parametrize(
    "module",
    [pred, factors],
    ids=["prediction", "factors"],
)
def test_todas_las_tools_tienen_title(module):
    tools = _tools(module)
    assert tools, f"{module.__name__} no registra ninguna tool"
    sin_title = [n for n, t in tools.items() if not (t.annotations and t.annotations.title)]
    assert not sin_title, f"tools sin title: {sin_title}"


@pytest.mark.parametrize(
    "module,nombres",
    [(pred, PRED_READ), (factors, FACTORS_READ)],
    ids=["prediction-read", "factors-read"],
)
def test_lectura_llevan_readonlyhint(module, nombres):
    tools = _tools(module)
    faltan = [
        n
        for n in nombres
        if not (tools[n].annotations and tools[n].annotations.readOnlyHint)
    ]
    assert not faltan, f"tools de lectura sin readOnlyHint: {faltan}"


@pytest.mark.parametrize(
    "module,nombres",
    [(pred, PRED_DESTRUCTIVE), (factors, FACTORS_DESTRUCTIVE)],
    ids=["prediction-destructive", "factors-destructive"],
)
def test_destructivas_llevan_destructivehint(module, nombres):
    tools = _tools(module)
    faltan = [
        n
        for n in nombres
        if not (tools[n].annotations and tools[n].annotations.destructiveHint)
    ]
    assert not faltan, f"tools destructivas sin destructiveHint: {faltan}"


@pytest.mark.parametrize(
    "module,nombres",
    [(pred, PRED_READ), (factors, FACTORS_READ)],
    ids=["prediction-read", "factors-read"],
)
def test_todas_las_nombradas_existen(module, nombres):
    tools = _tools(module)
    ausentes = [n for n in nombres if n not in tools]
    assert not ausentes, f"tools esperadas no registradas: {ausentes}"


@pytest.mark.parametrize(
    "module",
    [pred, factors],
    ids=["prediction", "factors"],
)
def test_transporte_streamable_http_responde(module):
    """El transporte streamable HTTP responde al handshake initialize.

    Verifica el transporte además de --stdio (criterio 4 de MCP-004) sin
    levantar un puerto: construye la app Streamable HTTP del servidor y la
    prueba con el TestClient de Starlette.
    """
    from starlette.testclient import TestClient

    app = module._mcp.streamable_http_app()
    payload = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "initialize",
        "params": {
            "protocolVersion": "2025-06-18",
            "capabilities": {},
            "clientInfo": {"name": "test", "version": "0"},
        },
    }
    with TestClient(app) as client:
        resp = client.post(
            "/mcp",
            json=payload,
            headers={
                "Host": "127.0.0.1:8101",
                "Accept": "application/json, text/event-stream",
            },
        )
        assert resp.status_code == 200, resp.text
        # La respuesta streamable HTTP llega como text/event-stream con una
        # línea `data: <jsonrpc>`. La extraemos y comprobamos el handshake.
        data = None
        for line in resp.text.splitlines():
            if line.startswith("data:"):
                data = json.loads(line[len("data:"):].strip())
                break
        assert data is not None, f"sin evento data en la respuesta: {resp.text!r}"
        assert data.get("jsonrpc") == "2.0"
        assert "result" in data, data
        assert data["result"]["protocolVersion"] == "2025-06-18"

