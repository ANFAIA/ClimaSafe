"""
agents.agents.doctor_agent — Diagnóstico integral del proyecto.

Revisa el estado del proyecto en todas las dimensiones: entorno, git,
estructura de datos, código, tests, dependencias y configuración.
"""

from __future__ import annotations

import sys
from pathlib import Path

from agents.core.base_agent import AgentResult, BaseAgent
from agents.core.registry import register_agent
from agents.tools.process_tool import run_command


@register_agent
class DoctorAgent(BaseAgent):
    name = "doctor"
    description = "Diagnóstico integral: entorno, git, datos, código, tests, dependencias, config."
    capabilities = ["diagnóstico", "health", "healthcheck", "check", "doctor", "estado", "status"]

    def actions(self) -> dict:
        return {
            "checkup": self.checkup,
            "disk_usage": self.disk_usage,
            "summary": self.summary,
        }

    def checkup(self) -> AgentResult:
        """Ejecuta todas las verificaciones y devuelve un dict con el estado."""
        checks = {
            "python": self._check_python(),
            "git": self._check_git(),
            "project_config": self._check_project_config(),
            "structure": self._check_structure(),
            "tests": self._check_tests(),
            "lock": self._check_lock_sync(),
            "data": self._check_data(),
        }
        ok = sum(1 for c in checks.values() if c.get("ok"))
        total = len(checks)
        all_ok = ok == total
        return AgentResult(
            all_ok, self.name, "checkup",
            f"{ok}/{total} verificaciones superadas",
            data=checks,
            warnings=[
                f"{k}: {v['message']}"
                for k, v in checks.items()
                if not v.get("ok")
            ],
        )

    def disk_usage(self) -> AgentResult:
        """Muestra el tamaño de los directorios principales del proyecto."""
        dirs = {
            "data": self.ctx.data_dir,
            "models": self.ctx.models_dir,
            "reports": self.ctx.reports_dir,
            "notebooks": self.ctx.notebooks_dir,
            "agents/workspace": self.ctx.workspace_dir,
        }
        sizes = {}
        for label, path in dirs.items():
            if path.exists():
                total_bytes = sum(
                    f.stat().st_size for f in path.rglob("*") if f.is_file()
                )
                sizes[label] = self._human_size(total_bytes)
            else:
                sizes[label] = "no existe"
        return AgentResult(True, self.name, "disk_usage", "Uso de disco por directorio.", data=sizes)

    def summary(self) -> AgentResult:
        """Resumen ejecutivo del proyecto."""
        pyproject = self._load_pyproject()
        git_result = run_command(
            ["git", "log", "--oneline", "-5"], cwd=self.ctx.root
        )
        git_log = git_result.stdout.strip() if git_result.ok else "no disponible"
        project_name = (
            pyproject.get("project", {}).get("name", "desconocido")
            if pyproject else "desconocido"
        )
        python_v = f"{sys.version_info.major}.{sys.version_info.minor}"
        test_dirs = list(self.ctx.tests_dir.glob("test_*.py")) if self.ctx.tests_dir.exists() else []
        data_files = list(self.ctx.raw_data_dir.glob("*")) if self.ctx.raw_data_dir.exists() else []

        data = {
            "project": project_name,
            "ml_type": self.ctx.config.ml_type,
            "python": python_v,
            "data_files": len(data_files),
            "test_files": len(test_dirs),
            "git_log": git_log,
        }
        return AgentResult(True, self.name, "summary", f"{project_name} ({self.ctx.config.ml_type})", data=data)

    # ---- helpers internos ----

    def _load_pyproject(self) -> dict | None:
        import tomllib
        try:
            with open(self.ctx.pyproject_file, "rb") as f:
                return tomllib.load(f)
        except Exception:
            return None

    def _check_python(self) -> dict:
        pyproject = self._load_pyproject()
        if pyproject is None:
            return {"ok": False, "message": "pyproject.toml no encontrado"}
        requires = pyproject.get("project", {}).get("requires-python", "")
        current = f"{sys.version_info.major}.{sys.version_info.minor}"
        ok = not requires or current in requires
        return {"ok": ok, "message": f"Python {current} (requiere {requires})"}

    def _check_git(self) -> dict:
        try:
            result = run_command(["git", "status", "--porcelain"], cwd=self.ctx.root)
        except Exception as e:
            return {"ok": False, "message": f"git no disponible: {e}"}
        if not result.ok:
            return {"ok": False, "message": "no es un repositorio git"}
        changes = result.stdout.strip()
        n_changes = len([l for l in changes.split("\n") if l.strip()]) if changes else 0
        branch_result = run_command(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=self.ctx.root
        )
        branch = branch_result.stdout.strip() if branch_result.ok else "?"
        if n_changes == 0:
            return {"ok": True, "message": f"Working directory clean ({branch})"}
        return {"ok": True, "message": f"{n_changes} archivo(s) sin commit ({branch})"}

    def _check_project_config(self) -> dict:
        config = self.ctx.config
        if not config.project_slug:
            return {"ok": False, "message": "configuración del proyecto no encontrada"}
        return {
            "ok": True,
            "message": f"{config.project_slug} (ml_type={config.ml_type}, mlflow={config.use_mlflow})",
        }

    def _check_structure(self) -> dict:
        required = [
            self.ctx.package_dir / "__init__.py",
            self.ctx.pyproject_file,
        ]
        missing = [str(p) for p in required if not p.exists()]
        if missing:
            return {"ok": False, "message": f"faltan: {', '.join(missing)}"}
        return {"ok": True, "message": "estructura del proyecto correcta"}

    def _check_tests(self) -> dict:
        tests_dir = self.ctx.tests_dir
        if not tests_dir.exists():
            return {"ok": True, "message": "directorio tests/ no existe (puede no ser necesario)"}
        test_files = list(tests_dir.glob("test_*.py"))
        n = len(test_files)
        if n == 0:
            return {"ok": False, "message": "tests/ existe pero no contiene tests"}
        return {"ok": True, "message": f"{n} archivo(s) de test"}

    def _check_lock_sync(self) -> dict:
        result = run_command(["uv", "lock", "--check"], cwd=self.ctx.root)
        if result.ok:
            return {"ok": True, "message": "uv.lock sincronizado"}
        return {"ok": False, "message": "uv.lock desincronizado (corre 'uv lock')"}

    def _check_data(self) -> dict:
        raw = self.ctx.raw_data_dir
        if not raw.exists():
            return {"ok": True, "message": "data/raw/ no existe (proyecto nuevo)"}
        files = list(raw.glob("*"))
        return {"ok": True, "message": f"{len(files)} archivo(s) en data/raw/"}

    @staticmethod
    def _human_size(bytes_: int) -> str:
        for unit in ("B", "KB", "MB", "GB"):
            if bytes_ < 1024:
                return f"{bytes_:.1f} {unit}"
            bytes_ /= 1024
        return f"{bytes_:.1f} TB"
