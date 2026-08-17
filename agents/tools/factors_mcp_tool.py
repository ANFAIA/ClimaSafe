"""
agents.tools.factors_mcp_tool — MCP server para gestionar factores de riesgo.

Proporciona tools MCP para que el LLM (o un agente) pueda leer y modificar
la tabla ``factores_riesgo`` en SQLite de forma controlada.

La migración desde JSON: los datos se leen de la BBDD SQLite.
Ejecuta ``climasafeai.db.manager.DBManager.migrar_desde_json()`` para
volcar el JSON existente en SQLite.

Uso standalone:
    uv run python -m agents.tools.factors_mcp_tool

Uso como tool registrada:
    from agents.tools.factors_mcp_tool import FactorsMCPTool
"""

from __future__ import annotations

import asyncio
import json
import sys
from typing import Any

from agents.tools.registry import register_tool
from climasafeai.db.manager import DBManager


def _db() -> DBManager:
    return DBManager()


# ── Funciones standalone (compatibilidad con agentes) ───────────────


def get_factors(tipo: str | None = None, solo_implementados: bool = True) -> dict:
    return _db().obtener_factores(solo_implementados=solo_implementados, tipo=tipo)


def suggest_factor(
    tipo: str,
    categoria: str,
    clave: str,
    nombre: str,
    coef: float,
    doi: str | None = None,
    calidad: str = "baja",
    poblacion: str | None = None,
) -> dict:
    if tipo not in ("calor", "frio"):
        return {"success": False, "error": f"tipo debe ser 'calor' o 'frio', no {tipo!r}"}
    if calidad not in ("alta", "media", "baja"):
        return {"success": False, "error": f"calidad debe ser alta/media/baja, no {calidad!r}"}
    if coef <= 0 or coef > 100:
        return {"success": False, "error": f"coeficiente debe estar en (0, 100], no {coef}"}
    return _db().sugerir_factor(tipo, categoria, clave, nombre, coef, doi, calidad, poblacion)


def approve_factor(clave: str, tipo: str, categoria: str) -> dict:
    return _db().aprobar_factor(tipo, categoria, clave)


def reject_factor(clave: str, tipo: str, categoria: str) -> dict:
    return _db().rechazar_factor(tipo, categoria, clave)


def update_factor(clave: str, tipo: str, categoria: str, **kwargs: Any) -> dict:
    return _db().actualizar_factor(tipo, categoria, clave, **kwargs)


def get_pending_factors() -> list[dict]:
    return _db().factores_pendientes()


# ── MCP Server ──────────────────────────────────────────────────────

try:
    from mcp.server.fastmcp import FastMCP
    from mcp.server.fastmcp.tools.base import ToolAnnotations

    _mcp = FastMCP("ClimaSafeAI Factores de Riesgo")

    @_mcp.tool(annotations=ToolAnnotations(title="Obtener factores", readOnlyHint=True))
    def get_factors_mcp(tipo: str | None = None, solo_implementados: bool = True) -> str:
        """Devuelve factores de riesgo. Filtra por tipo ('calor'/'frio') y opcionalmente solo implementados."""
        return json.dumps(get_factors(tipo=tipo, solo_implementados=solo_implementados), indent=2)

    @_mcp.tool(annotations=ToolAnnotations(title="Sugerir factor"))
    def suggest_factor_mcp(tipo: str, categoria: str, clave: str, nombre: str, coef: float, doi: str | None = None, calidad: str = "baja", poblacion: str | None = None) -> str:
        """Añade un nuevo factor candidato (implementado=false). Requiere tipo, categoria, clave, nombre y coef."""
        return json.dumps(suggest_factor(tipo, categoria, clave, nombre, coef, doi, calidad, poblacion), indent=2)

    @_mcp.tool(annotations=ToolAnnotations(title="Aprobar factor"))
    def approve_factor_mcp(clave: str, tipo: str, categoria: str) -> str:
        """Activa un factor candidato (implementado=true)."""
        return json.dumps(approve_factor(clave, tipo, categoria), indent=2)

    @_mcp.tool(annotations=ToolAnnotations(title="Rechazar factor", destructiveHint=True))
    def reject_factor_mcp(clave: str, tipo: str, categoria: str) -> str:
        """Elimina un factor candidato de la BBDD."""
        return json.dumps(reject_factor(clave, tipo, categoria), indent=2)

    @_mcp.tool(annotations=ToolAnnotations(title="Actualizar factor"))
    def update_factor_mcp(clave: str, tipo: str, categoria: str, coef: float | None = None, nombre: str | None = None, doi: str | None = None, calidad: str | None = None, poblacion: str | None = None) -> str:
        """Actualiza campos de un factor existente (solo los que se pasen no-None)."""
        kwargs = {}
        if coef is not None:
            kwargs["coef"] = coef
        if nombre is not None:
            kwargs["nombre"] = nombre
        if doi is not None:
            kwargs["doi"] = doi
        if calidad is not None:
            kwargs["calidad"] = calidad
        if poblacion is not None:
            kwargs["poblacion"] = poblacion
        return json.dumps(update_factor(clave, tipo, categoria, **kwargs), indent=2)

    @_mcp.tool(annotations=ToolAnnotations(title="Factores pendientes", readOnlyHint=True))
    def pending_factors_mcp() -> str:
        """Lista factores pendientes de revisión (implementado=false)."""
        return json.dumps(get_pending_factors(), indent=2)

    @_mcp.tool(annotations=ToolAnnotations(title="Comprobar aclimatación", readOnlyHint=True))
    def check_acclimatization_mcp(dias: int | None = None) -> str:
        """Busca perfiles no aclimatados que ya deberían estarlo según tiempo transcurrido. La evidencia (Karlsen 2015, DOI: 10.1111/sms.12449) indica aclimatación completa en 14 días. Devuelve lista de candidatos."""
        return json.dumps(_db().perfiles_para_aclimatar(dias=dias), indent=2, default=str)

    @_mcp.tool(annotations=ToolAnnotations(title="Auto-aclimatar perfiles"))
    def auto_acclimatize_mcp(perfil_id: int | None = None, dias: int | None = None) -> str:
        """Marca como aclimatados los perfiles que cumplan el criterio temporal. Si perfil_id se omite, actualiza todos los que cumplan. Devuelve resumen de cuántos se aclimataron."""
        return json.dumps(_db().auto_aclimatar(perfil_id=perfil_id, dias=dias), indent=2, default=str)

    @_mcp.tool(annotations=ToolAnnotations(title="Buscar factores", readOnlyHint=True))
    def search_factors_mcp(query: str, k: int = 5) -> str:
        """Búsqueda semántica sobre factores de riesgo usando sqlite-vec. Devuelve los k factores más relevantes para la consulta con su distancia coseno."""
        return json.dumps(_db().search_factores(query, k=k), indent=2, default=str)

    @_mcp.tool(annotations=ToolAnnotations(title="Buscar documentos", readOnlyHint=True))
    def search_documentos_mcp(query: str, k: int = 5) -> str:
        """Búsqueda semántica sobre la documentación del proyecto (documentacion/). Devuelve fragmentos de .md con distancia coseno."""
        return json.dumps(_db().search_documentos(query, k=k), indent=2, default=str)

    @_mcp.tool(annotations=ToolAnnotations(title="Buscar todo", readOnlyHint=True))
    def search_all_mcp(query: str, k: int = 5) -> str:
        """Búsqueda combinada: factores de riesgo + documentación del proyecto."""
        return json.dumps(_db().search_all(query, k=k), indent=2, default=str)

    @_mcp.tool(annotations=ToolAnnotations(title="Preguntar RAG", readOnlyHint=True))
    def ask_rag_mcp(query: str, k: int = 5) -> str:
        """RAG completo: responde una pregunta en lenguaje natural sobre factores de riesgo térmico. Busca factores relevantes y genera respuesta con IA."""
        return json.dumps(_db().ask_rag(query, k=k), indent=2, default=str)

    @_mcp.tool(annotations=ToolAnnotations(title="Preguntar RAG con Qwen", readOnlyHint=True))
    def ask_qwen_rag_mcp(query: str, k_factores: int = 5, k_docs: int = 5, model: str = "ollama/qwen2.5:1.5b") -> str:
        """RAG completo con LLM local o API vía LiteLLM. Busca en factores de riesgo y documentación del proyecto, responde citando fuentes. Modelos: ollama/qwen2.5:1.5b (CPU), ollama/qwen2.5:7b (GPU), groq/llama-3.3-70b-versatile (API)."""
        from climasafeai.llm.rag_qwen import ask_with_rag, LLMConfig
        config = LLMConfig(model=model)
        return json.dumps(ask_with_rag(query, k_factores=k_factores, k_docs=k_docs, config=config), indent=2, default=str)

    @_mcp.tool(annotations=ToolAnnotations(title="Preguntar a Qwen", readOnlyHint=True))
    def qwen_raw_mcp(query: str, model: str = "ollama/qwen2.5:1.5b") -> str:
        """LLM raw sin RAG: responde preguntas generales usando solo el conocimiento del modelo."""
        from climasafeai.llm.rag_qwen import ask_raw, LLMConfig
        config = LLMConfig(model=model)
        return json.dumps(ask_raw(query, config=config), indent=2, default=str)

    _HAS_MCP = True
except ImportError:
    _HAS_MCP = False


def search_factors(query: str, k: int = 5) -> list[dict]:
    return _db().search_factores(query, k=k)


def ask_rag(query: str, k: int = 5) -> dict:
    """RAG completo: retrieve + generate."""
    return _db().ask_rag(query, k=k)


def run_mcp_server(
    host: str = "0.0.0.0",
    port: int = 8100,
    ssl_keyfile: str | None = None,
    ssl_certfile: str | None = None,
) -> None:
    """Arranca el servidor MCP en modo Streamable HTTP (spec 2025-06-18+)."""
    if not _HAS_MCP:
        print("Error: mcp no está instalado. Ejecuta: uv add mcp", file=sys.stderr)
        return

    import uvicorn

    starlette_app = _mcp.streamable_http_app()
    proto = "https" if ssl_certfile else "http"
    print(f"MCP Server — ClimaSafeAI Factores de Riesgo", file=sys.stderr)
    print(f"   Escuchando en {proto}://{host}:{port}/mcp (Streamable HTTP)", file=sys.stderr)

    uvicorn_config: dict[str, Any] = {
        "app": starlette_app,
        "host": host,
        "port": port,
        "log_level": "info",
    }
    if ssl_keyfile and ssl_certfile:
        uvicorn_config["ssl_keyfile"] = ssl_keyfile
        uvicorn_config["ssl_certfile"] = ssl_certfile

    try:
        uvicorn.run(**uvicorn_config)
    except (KeyboardInterrupt, asyncio.CancelledError):
        print("\n   Servidor detenido.", file=sys.stderr, flush=True)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="ClimaSafeAI MCP Factors Server")
    parser.add_argument("--host", default="0.0.0.0", help="Host (default 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8100, help="Puerto (default 8100)")
    parser.add_argument("--ssl-keyfile", help="Ruta a clave privada SSL")
    parser.add_argument("--ssl-certfile", help="Ruta a certificado SSL")
    args = parser.parse_args()
    run_mcp_server(
        host=args.host, port=args.port,
        ssl_keyfile=args.ssl_keyfile, ssl_certfile=args.ssl_certfile,
    )
