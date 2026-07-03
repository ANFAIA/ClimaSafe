"""
agents.agents.review_agent — Revisión de código Python del proyecto.

Complementa a `ruff` (`make lint`), no lo sustituye: `ruff` cubre estilo y
errores sintácticos; este agente busca señales que necesitan contexto de
varias funciones o archivos a la vez (funciones largas, duplicación
estructural entre archivos), que un linter por archivo no ve.
"""

from __future__ import annotations

from agents.core.base_agent import AgentResult, BaseAgent
from agents.core.registry import register_agent
from agents.tools.code_analysis_tool import CodeAnalysisTool


@register_agent
class ReviewAgent(BaseAgent):
    name = "review"
    description = "Revisa código Python: funciones largas, demasiados argumentos, except desnudos, duplicación."
    capabilities = ["revisar", "review", "code smell", "duplicacion", "refactor", "calidad de codigo", "bug"]

    def actions(self) -> dict:
        return {
            "review_package": self.review_package,
            "review_file": self.review_file,
        }

    def review_file(self, *, relative_path: str) -> AgentResult:
        path = self.ctx.root / relative_path
        if not path.exists() or path.suffix != ".py":
            return AgentResult(False, self.name, "review_file", f"'{relative_path}' no existe o no es un archivo .py.")

        smells, _functions = CodeAnalysisTool.analyze_file(path)
        return AgentResult(
            True, self.name, "review_file", f"{len(smells)} hallazgo(s) en '{relative_path}'.",
            data=[s.__dict__ for s in smells],
        )

    def review_package(self, *, within: str | None = None) -> AgentResult:
        """
        Revisa `{{ project_slug }}/` por defecto (el paquete principal del
        proyecto). Pasa `within` para revisar otra carpeta, p. ej. "tests".
        """
        target = within or self.ctx.config.project_slug
        if not target:
            return AgentResult(
                False, self.name, "review_package",
                "No se pudo determinar qué carpeta revisar: project_slug está vacío "
                "y no se pasó 'within' explícitamente. Pasa within='mi_paquete' o "
                "revisa .copier-answers.yml.",
            )
        base = self.ctx.root / target
        if not base.exists():
            return AgentResult(False, self.name, "review_package", f"La carpeta '{base}' no existe.")

        py_files = [
            p for p in base.rglob("*.py")
            if "__pycache__" not in p.parts
        ]
        if not py_files:
            return AgentResult(True, self.name, "review_package", f"No hay archivos .py en '{base}'.", data=[])

        all_smells = []
        all_functions = []
        for path in py_files:
            smells, functions = CodeAnalysisTool.analyze_file(path)
            all_smells.extend(smells)
            all_functions.extend(functions)

        duplicate_groups = CodeAnalysisTool.find_duplicates(all_functions)

        report = {
            "n_files_analyzed": len(py_files),
            "smells": [s.__dict__ for s in all_smells],
            "duplicate_function_groups": [
                [{"file": f.file, "name": f.name, "line": f.line} for f in group]
                for group in duplicate_groups
            ],
        }
        warnings = []
        if duplicate_groups:
            warnings.append(
                f"{len(duplicate_groups)} grupo(s) de funciones con estructura AST idéntica "
                f"— posible duplicación (revisa manualmente, un falso positivo es posible)."
            )

        return AgentResult(
            True, self.name, "review_package",
            f"{len(py_files)} archivo(s) analizados, {len(all_smells)} hallazgo(s), "
            f"{len(duplicate_groups)} grupo(s) de posible duplicación.",
            data=report, warnings=warnings,
        )
