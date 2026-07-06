"""
agents.agents.api_agent — Valida la API REST del proyecto (`api/main.py`).

Solo aplica si el proyecto se generó con `use_api=true`. `smoke_test`
importa `api.main:app` de verdad y lo ejercita con
`fastapi.testclient.TestClient` — no es una comprobación estática, arranca
la app (sin un servidor real escuchando en un puerto) y le pega una
petición HTTP de verdad a `/health`.
"""

from __future__ import annotations

from agents.core.base_agent import AgentResult, BaseAgent
from agents.core.registry import register_agent
from agents.tools.api_tool import APITool


@register_agent
class APIAgent(BaseAgent):
    name = "api"
    description = "Valida la API REST del proyecto: cruza endpoints declarados vs. documentados, smoke test real con TestClient."
    capabilities = ["api", "fastapi", "endpoint", "rest", "smoke test api", "smoke test"]

    def actions(self) -> dict:
        return {
            "check_endpoints_documented": self.check_endpoints_documented,
            "smoke_test": self.smoke_test,
        }

    def _main_py_path(self):
        return self.ctx.api_dir / "main.py"

    def check_endpoints_documented(self) -> AgentResult:
        path = self._main_py_path()
        if not path.exists():
            return AgentResult(False, self.name, "check_endpoints_documented", "No existe api/main.py (¿use_api=false?).")

        declared = {(r.method, r.path) for r in APITool.extract_declared_routes(path)}
        documented = {(r.method, r.path) for r in APITool.extract_documented_routes(path)}

        undocumented = declared - documented
        stale_docs = documented - declared
        warnings = [f"Endpoint {m} {p} declarado con @app pero no aparece en el docstring del módulo." for m, p in undocumented]
        warnings += [f"El docstring menciona {m} {p} pero no hay ningún @app.{m.lower()}(\"{p}\") real." for m, p in stale_docs]

        return AgentResult(
            not undocumented and not stale_docs, self.name, "check_endpoints_documented",
            f"{len(declared)} endpoint(s) declarado(s), {len(undocumented)} sin documentar, {len(stale_docs)} documentado(s) que ya no existen.",
            data={"declared": list(declared), "documented": list(documented)}, warnings=warnings,
        )

    def smoke_test(self, *, endpoint: str = "/health") -> AgentResult:
        try:
            from fastapi.testclient import TestClient
        except ImportError:
            return AgentResult(
                False, self.name, "smoke_test",
                "fastapi (o httpx, que TestClient necesita) no está instalado — instala el extra use_api.",
            )

        path = self._main_py_path()
        if not path.exists():
            return AgentResult(False, self.name, "smoke_test", "No existe api/main.py (¿use_api=false?).")

        try:
            import importlib
            module = importlib.import_module("api.main")
        except Exception as exc:  # noqa: BLE001 — cualquier fallo de import de la app es el propio resultado a reportar
            return AgentResult(False, self.name, "smoke_test", f"No se pudo importar api.main: {exc}")

        app = getattr(module, "app", None)
        if app is None:
            return AgentResult(False, self.name, "smoke_test", "api.main no expone una variable 'app'.")

        with TestClient(app) as client:
            response = client.get(endpoint)

        success = response.status_code == 200
        return AgentResult(
            success, self.name, "smoke_test",
            f"GET {endpoint} -> {response.status_code}.",
            data={"status_code": response.status_code, "body": response.json() if success else response.text},
        )
