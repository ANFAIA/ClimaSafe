"""
agents.tools.factors_mcp_tool — MCP server para gestionar factores de riesgo.

Proporciona tools MCP para que el LLM (o un agente) pueda leer y modificar
`data/factores_riesgo.json` de forma controlada.

Uso standalone:
    uv run python -m agents.tools.factors_mcp_tool

Uso como tool registrada:
    from agents.tools.factors_mcp_tool import FactorsMCPTool
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from agents.tools.registry import register_tool

_FACTORES_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "factores_riesgo.json"


def _cargar() -> dict:
    if not _FACTORES_PATH.exists():
        return {"version": 1, "cap_factores": 3.0}
    return json.loads(_FACTORES_PATH.read_text(encoding="utf-8"))


def _guardar(data: dict) -> None:
    _FACTORES_PATH.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def get_factors(tipo: str | None = None, solo_implementados: bool = True) -> dict:
    """Devuelve factores de riesgo, filtrados opcionalmente por tipo y estado."""
    data = _cargar()
    tipos = ["calor", "frio"] if tipo is None else [tipo]
    result = {}
    for t in tipos:
        seccion = data.get(t, {})
        t_result = {}
        for categoria, factores in seccion.items():
            if not isinstance(factores, dict):
                continue
            items = []
            for clave, info in factores.items():
                if not isinstance(info, dict):
                    continue
                if solo_implementados and not info.get("implementado"):
                    continue
                items.append({
                    "clave": clave,
                    "nombre": info.get("nombre", clave),
                    "coeficiente": info.get("coef"),
                    "doi": info.get("doi"),
                    "calidad": info.get("calidad", "baja"),
                    "implementado": info.get("implementado", False),
                })
            if items:
                t_result[categoria] = items
        result[t] = t_result
    return result


def suggest_factor(
    tipo: str,
    categoria: str,
    clave: str,
    nombre: str,
    coef: float,
    doi: str | None = None,
    calidad: str = "baja",
) -> dict:
    """Añade un factor candidato con implementado=false. No sobreescribe si ya existe y está activo."""
    if tipo not in ("calor", "frio"):
        return {"success": False, "error": f"tipo debe ser 'calor' o 'frio', no {tipo!r}"}
    if calidad not in ("alta", "media", "baja"):
        return {"success": False, "error": f"calidad debe ser alta/media/baja, no {calidad!r}"}
    if coef <= 0 or coef > 100:
        return {"success": False, "error": f"coeficiente debe estar en (0, 100], no {coef}"}

    data = _cargar()
    seccion = data.setdefault(tipo, {})
    sub = seccion.setdefault(categoria, {})
    if clave in sub and sub[clave].get("implementado"):
        return {"success": False, "error": f"'{clave}' ya existe y está implementado, no se sobreescribe"}

    sub[clave] = {
        "coef": coef,
        "nombre": nombre,
        "doi": doi,
        "calidad": calidad,
        "implementado": False,
    }
    _guardar(data)
    return {"success": True, "factor": sub[clave]}


def approve_factor(clave: str, tipo: str, categoria: str) -> dict:
    """Marca un factor como implementado=true."""
    data = _cargar()
    factor = data.get(tipo, {}).get(categoria, {}).get(clave)
    if factor is None:
        return {"success": False, "error": f"factor '{clave}' no encontrado en {tipo}/{categoria}"}
    factor["implementado"] = True
    _guardar(data)
    return {"success": True, "factor": factor}


def reject_factor(clave: str, tipo: str, categoria: str) -> dict:
    """Elimina un factor del JSON."""
    data = _cargar()
    seccion = data.get(tipo, {}).get(categoria, {})
    if clave not in seccion:
        return {"success": False, "error": f"factor '{clave}' no encontrado en {tipo}/{categoria}"}
    del seccion[clave]
    _guardar(data)
    return {"success": True}


def update_factor(clave: str, tipo: str, categoria: str, **kwargs: Any) -> dict:
    """Actualiza campos de un factor existente (coef, nombre, doi, calidad)."""
    data = _cargar()
    factor = data.get(tipo, {}).get(categoria, {}).get(clave)
    if factor is None:
        return {"success": False, "error": f"factor '{clave}' no encontrado en {tipo}/{categoria}"}
    for k in ("coef", "nombre", "doi", "calidad"):
        if k in kwargs:
            factor[k] = kwargs[k]
    _guardar(data)
    return {"success": True, "factor": factor}


def get_pending_factors() -> list[dict]:
    """Devuelve todos los factores con implementado=false (pendientes de revisión)."""
    data = _cargar()
    pendientes = []
    for tipo in ("calor", "frio"):
        for categoria, factores in data.get(tipo, {}).items():
            if not isinstance(factores, dict):
                continue
            for clave, info in factores.items():
                if isinstance(info, dict) and not info.get("implementado"):
                    pendientes.append({
                        "clave": clave,
                        "tipo": tipo,
                        "categoria": categoria,
                        "nombre": info.get("nombre", clave),
                        "coeficiente": info.get("coef"),
                        "doi": info.get("doi"),
                        "calidad": info.get("calidad", "baja"),
                    })
    return pendientes


# ── MCP Server ────────────────────────────────────────────────────────────────

try:
    from mcp.server.fastmcp import FastMCP

    _mcp = FastMCP("ClimaSafeAI Factores de Riesgo")

    @_mcp.tool()
    def get_factors_mcp(tipo: str | None = None, solo_implementados: bool = True) -> str:
        """Devuelve factores de riesgo. Filtra por tipo ('calor'/'frio') y opcionalmente solo implementados."""
        import json
        return json.dumps(get_factors(tipo=tipo, solo_implementados=solo_implementados), indent=2)

    @_mcp.tool()
    def suggest_factor_mcp(tipo: str, categoria: str, clave: str, nombre: str, coef: float, doi: str | None = None, calidad: str = "baja") -> str:
        """Añade un nuevo factor candidato (implementado=false). Requiere tipo, categoria, clave, nombre y coef."""
        import json
        return json.dumps(suggest_factor(tipo, categoria, clave, nombre, coef, doi, calidad), indent=2)

    @_mcp.tool()
    def approve_factor_mcp(clave: str, tipo: str, categoria: str) -> str:
        """Activa un factor candidato (implementado=true)."""
        import json
        return json.dumps(approve_factor(clave, tipo, categoria), indent=2)

    @_mcp.tool()
    def reject_factor_mcp(clave: str, tipo: str, categoria: str) -> str:
        """Elimina un factor candidato del JSON."""
        import json
        return json.dumps(reject_factor(clave, tipo, categoria), indent=2)

    @_mcp.tool()
    def update_factor_mcp(clave: str, tipo: str, categoria: str, coef: float | None = None, nombre: str | None = None, doi: str | None = None, calidad: str | None = None) -> str:
        """Actualiza campos de un factor existente (solo los que se pasen no-None)."""
        import json
        kwargs = {}
        if coef is not None:
            kwargs["coef"] = coef
        if nombre is not None:
            kwargs["nombre"] = nombre
        if doi is not None:
            kwargs["doi"] = doi
        if calidad is not None:
            kwargs["calidad"] = calidad
        return json.dumps(update_factor(clave, tipo, categoria, **kwargs), indent=2)

    @_mcp.tool()
    def pending_factors_mcp() -> str:
        """Lista factores pendientes de revisión (implementado=false)."""
        import json
        return json.dumps(get_pending_factors(), indent=2)

    _HAS_MCP = True
except ImportError:
    _HAS_MCP = False


def run_mcp_server(host: str = "0.0.0.0", port: int = 8100) -> None:
    """Arranca el servidor MCP en modo SSE."""
    if not _HAS_MCP:
        print("Error: mcp no está instalado. Ejecuta: uv add mcp")
        return
    print(f"🧠 MCP Server — ClimaSafeAI Factores de Riesgo")
    print(f"   Escuchando en http://{host}:{port}/sse")
    _mcp.run(host=host, port=port)


if __name__ == "__main__":
    run_mcp_server()
