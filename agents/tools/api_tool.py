"""
agents.tools.api_tool — Extrae y valida los endpoints de `api/main.py`.

Grounding: los endpoints reales de este template son
`@app.get("/health")`, `@app.get("/info")`, `@app.post("/predict")`
(comprobado leyendo `api/main.py`, no asumido) — el extractor de aquí es
genérico (cualquier `@app.<metodo>("<ruta>")`), no hardcodea esos tres.
`httpx` ya es una dependencia declarada del extra `use_api` precisamente
para tests de la API — `fastapi.testclient.TestClient` la usa.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from agents.tools.registry import register_tool

_ROUTE_RE = re.compile(r'@app\.(get|post|put|delete|patch)\(\s*["\']([^"\']+)["\']')
_DOCSTRING_ENDPOINT_RE = re.compile(r"^\s*(GET|POST|PUT|DELETE|PATCH)\s+(\S+)", re.MULTILINE)


@dataclass
class RouteInfo:
    method: str
    path: str


@register_tool("api")
class APITool:
    @staticmethod
    def extract_declared_routes(main_py_path: Path) -> list[RouteInfo]:
        if not main_py_path.exists():
            return []
        text = main_py_path.read_text(encoding="utf-8")
        return [RouteInfo(method=m.upper(), path=p) for m, p in _ROUTE_RE.findall(text)]

    @staticmethod
    def extract_documented_routes(main_py_path: Path) -> list[RouteInfo]:
        """Busca un bloque tipo 'GET /health → ...' en el docstring del módulo."""
        if not main_py_path.exists():
            return []
        text = main_py_path.read_text(encoding="utf-8")
        return [RouteInfo(method=m.upper(), path=p) for m, p in _DOCSTRING_ENDPOINT_RE.findall(text)]
