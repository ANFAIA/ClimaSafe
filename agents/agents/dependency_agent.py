"""
agents.agents.dependency_agent — Analiza pyproject.toml/uv.lock contra PyPI.

Único agente del sistema (junto con `installer` en su parte de clonado) que
depende de acceso a internet — todos los demás trabajan solo con lo que ya
hay en el proyecto. Si no hay red, cada paquete falla individualmente (ver
`PackageStatus.error`) y el resto del chequeo continúa igual.
"""

from __future__ import annotations

from agents.core.base_agent import AgentResult, BaseAgent
from agents.core.registry import register_agent
from agents.tools.dependency_tool import DependencyTool, parse_dependency_name, parse_pyproject_dependencies, parse_uv_lock_versions


@register_agent
class DependencyAgent(BaseAgent):
    name = "dependency"
    description = (
        "Detecta versiones de paquetes desactualizadas respecto a PyPI, vulnerabilidades conocidas "
        "(OSV) y estima cadencia de releases. Necesita acceso a internet."
    )
    capabilities = [
        "dependencias", "paquetes obsoletos", "versiones desactualizadas", "vulnerabilidad",
        "pypi", "actualizar dependencias", "uv lock",
    ]

    def action_aliases(self) -> dict:
        return {
            "check_outdated": ["revisa", "desactualizadas", "obsoletas", "actualizar"],
            "check_vulnerabilities": ["vulnerabilidad", "vulnerabilidades", "seguridad"],
            "check_lock_sync": ["sincronizado", "sincronia", "lock"],
        }

    def actions(self) -> dict:
        return {
            "check_outdated": self.check_outdated,
            "check_vulnerabilities": self.check_vulnerabilities,
            "check_lock_sync": self.check_lock_sync,
        }

    def _declared_dependencies(self) -> list[str] | None:
        if not self.ctx.pyproject_file.exists():
            return None
        return parse_pyproject_dependencies(self.ctx.pyproject_file.read_text(encoding="utf-8"))

    def _locked_versions(self) -> dict[str, str]:
        lock_path = self.ctx.root / "uv.lock"
        if not lock_path.exists():
            return {}
        return parse_uv_lock_versions(lock_path.read_text(encoding="utf-8"))

    def check_outdated(self, *, include_cadence: bool = False) -> AgentResult:
        """
        Compara la versión bloqueada en `uv.lock` (o, si no existe, la
        última de PyPI como referencia — en ese caso no se puede saber si
        está desactualizada de verdad, solo se informa la última disponible)
        contra la última versión publicada en PyPI.

        `include_cadence=True` añade una estimación de cada cuánto libera
        versiones el paquete — es un promedio histórico, no una predicción,
        ver `DependencyTool.estimate_release_cadence_days`.
        """
        specs = self._declared_dependencies()
        if specs is None:
            return AgentResult(False, self.name, "check_outdated", "No existe pyproject.toml en la raíz del proyecto.")
        if not specs:
            return AgentResult(True, self.name, "check_outdated", "No hay dependencias declaradas en [project.dependencies].", data=[])

        locked = self._locked_versions()
        results = []
        warnings = []
        for spec in specs:
            status = DependencyTool.check_package(
                spec, locked_version=locked.get(parse_dependency_name(spec)),
                include_vulnerabilities=False, include_cadence=include_cadence,
            )
            results.append(status.__dict__)
            if status.error:
                warnings.append(f"{status.name}: {status.error}")

        if not locked:
            warnings.append(
                "No se encontró uv.lock — sin él, 'is_outdated' compara la versión declarada en pyproject.toml "
                "de forma poco fiable (a menudo sin pin exacto). Ejecuta 'uv lock' para un chequeo preciso."
            )

        n_outdated = sum(1 for r in results if r["is_outdated"] is True)
        return AgentResult(
            True, self.name, "check_outdated",
            f"{len(results)} dependencia(s) revisada(s), {n_outdated} desactualizada(s) (según lo que se pudo determinar).",
            data=results, warnings=warnings,
        )

    def check_vulnerabilities(self) -> AgentResult:
        """Consulta advisorios OSV (vía la API de PyPI) para la versión bloqueada de cada dependencia."""
        specs = self._declared_dependencies()
        if specs is None:
            return AgentResult(False, self.name, "check_vulnerabilities", "No existe pyproject.toml en la raíz del proyecto.")

        locked = self._locked_versions()
        if not locked:
            return AgentResult(
                False, self.name, "check_vulnerabilities",
                "No se encontró uv.lock — sin la versión exacta instalada no se puede consultar "
                "vulnerabilidades de forma fiable. Ejecuta 'uv lock' primero.",
            )

        findings = []
        warnings = []
        for spec in specs:
            name = parse_dependency_name(spec)
            version = locked.get(name)
            if not version:
                warnings.append(f"'{name}' no aparece en uv.lock con una versión exacta — omitido.")
                continue
            vulns = DependencyTool.fetch_vulnerabilities(name, version)
            if vulns:
                findings.append({"name": name, "version": version, "vulnerabilities": vulns})

        return AgentResult(
            True, self.name, "check_vulnerabilities",
            f"{len(findings)} paquete(s) con vulnerabilidades conocidas de {len(specs)} revisado(s).",
            data=findings, warnings=warnings,
        )

    def check_lock_sync(self) -> AgentResult:
        """
        Ejecuta `uv lock --check` (comando real de uv, verificado en su
        documentación oficial) para confirmar que uv.lock sigue reflejando
        pyproject.toml — no una comparación de fechas de archivo, la
        comprobación real que hace uv.
        """
        from agents.exceptions import MissingDependencyError, ToolExecutionError
        from agents.tools.process_tool import run_command

        try:
            result = run_command(["uv", "lock", "--check"], cwd=self.ctx.root, timeout=60)
        except MissingDependencyError as exc:
            return AgentResult(False, self.name, "check_lock_sync", str(exc))
        except ToolExecutionError as exc:
            return AgentResult(False, self.name, "check_lock_sync", str(exc))

        if result.ok:
            return AgentResult(True, self.name, "check_lock_sync", "uv.lock está sincronizado con pyproject.toml.")
        return AgentResult(
            False, self.name, "check_lock_sync",
            "uv.lock está desactualizado respecto a pyproject.toml — ejecuta 'uv lock'.",
            data={"stderr": result.stderr},
        )
