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

import json
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

    _mcp = FastMCP("ClimaSafeAI Factores de Riesgo")

    @_mcp.tool()
    def get_factors_mcp(tipo: str | None = None, solo_implementados: bool = True) -> str:
        """Devuelve factores de riesgo. Filtra por tipo ('calor'/'frio') y opcionalmente solo implementados."""
        return json.dumps(get_factors(tipo=tipo, solo_implementados=solo_implementados), indent=2)

    @_mcp.tool()
    def suggest_factor_mcp(tipo: str, categoria: str, clave: str, nombre: str, coef: float, doi: str | None = None, calidad: str = "baja", poblacion: str | None = None) -> str:
        """Añade un nuevo factor candidato (implementado=false). Requiere tipo, categoria, clave, nombre y coef."""
        return json.dumps(suggest_factor(tipo, categoria, clave, nombre, coef, doi, calidad, poblacion), indent=2)

    @_mcp.tool()
    def approve_factor_mcp(clave: str, tipo: str, categoria: str) -> str:
        """Activa un factor candidato (implementado=true)."""
        return json.dumps(approve_factor(clave, tipo, categoria), indent=2)

    @_mcp.tool()
    def reject_factor_mcp(clave: str, tipo: str, categoria: str) -> str:
        """Elimina un factor candidato de la BBDD."""
        return json.dumps(reject_factor(clave, tipo, categoria), indent=2)

    @_mcp.tool()
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

    @_mcp.tool()
    def pending_factors_mcp() -> str:
        """Lista factores pendientes de revisión (implementado=false)."""
        return json.dumps(get_pending_factors(), indent=2)

    @_mcp.tool()
    def check_acclimatization_mcp(dias: int | None = None) -> str:
        """Busca perfiles no aclimatados que ya deberían estarlo según tiempo transcurrido. La evidencia (Karlsen 2015, DOI: 10.1111/sms.12449) indica aclimatación completa en 14 días. Devuelve lista de candidatos."""
        return json.dumps(_db().perfiles_para_aclimatar(dias=dias), indent=2, default=str)

    @_mcp.tool()
    def auto_acclimatize_mcp(perfil_id: int | None = None, dias: int | None = None) -> str:
        """Marca como aclimatados los perfiles que cumplan el criterio temporal. Si perfil_id se omite, actualiza todos los que cumplan. Devuelve resumen de cuántos se aclimataron."""
        return json.dumps(_db().auto_aclimatar(perfil_id=perfil_id, dias=dias), indent=2, default=str)

    _HAS_MCP = True
except ImportError:
    _HAS_MCP = False


def run_mcp_server(host: str = "0.0.0.0", port: int = 8100) -> None:
    """Arranca el servidor MCP en modo SSE."""
    if not _HAS_MCP:
        print("Error: mcp no está instalado. Ejecuta: uv add mcp")
        return
    print(f"MCP Server — ClimaSafeAI Factores de Riesgo")
    print(f"   Escuchando en http://{host}:{port}/sse")
    _mcp.run(host=host, port=port)


if __name__ == "__main__":
    run_mcp_server()
