"""
agents.agents.cicd_agent — Genera y valida `.github/workflows/*.yml`.

Escribe directamente en `.github/workflows/` (no en `agents/workspace/`):
a diferencia de lo que generan otros agentes (manifests, staging de
clonados), un workflow de CI solo funciona si GitHub Actions lo encuentra en
esa ruta exacta — es la misma excepción que ya aplican
`DocumentationAgent.bump_version` (pyproject.toml, README.md) y
`GitAgent`/`update_changelog` (CHANGELOG.md): archivos cuyo propósito
completo depende de vivir en una ubicación fija del proyecto.
"""

from __future__ import annotations

import re

from agents.core.base_agent import AgentResult, BaseAgent
from agents.core.registry import register_agent
from agents.tools.cicd_tool import WORKFLOWS_DIR, CICDTool


@register_agent
class CICDAgent(BaseAgent):
    name = "cicd"
    description = "Genera y valida workflows de GitHub Actions, cruzando referencias contra el Makefile real del proyecto."
    capabilities = ["ci", "cd", "cicd", "github actions", "workflow", "pipeline de integracion continua"]

    def action_aliases(self) -> dict:
        return {
            "validate_workflow": ["valida", "revisa el workflow", "comprueba"],
            "generate_workflow": ["genera", "crea el workflow", "añade ci"],
        }

    def actions(self) -> dict:
        return {
            "validate_workflow": self.validate_workflow,
            "generate_workflow": self.generate_workflow,
            "list_workflows": self.list_workflows,
        }

    def list_workflows(self) -> AgentResult:
        workflows_dir = self.ctx.root / WORKFLOWS_DIR
        if not workflows_dir.exists():
            return AgentResult(True, self.name, "list_workflows", "No existe .github/workflows/ todavía.", data=[])
        files = [p.name for p in workflows_dir.glob("*.yml")] + [p.name for p in workflows_dir.glob("*.yaml")]
        return AgentResult(True, self.name, "list_workflows", f"{len(files)} workflow(s) encontrado(s).", data=files)

    def validate_workflow(self, *, filename: str = "ci.yml") -> AgentResult:
        path = self.ctx.root / WORKFLOWS_DIR / filename
        problems = CICDTool.validate(path)

        warnings = list(problems)
        if not path.exists():
            return AgentResult(False, self.name, "validate_workflow", problems[0])

        referenced_targets = CICDTool.referenced_make_targets(path)
        if referenced_targets:
            makefile_path = self.ctx.root / "Makefile"
            if makefile_path.exists():
                makefile_text = makefile_path.read_text(encoding="utf-8")
                real_targets = set(re.findall(r"^([a-zA-Z_-]+):", makefile_text, re.MULTILINE))
                missing = [t for t in referenced_targets if t not in real_targets]
                if missing:
                    warnings.append(
                        f"El workflow invoca make {', '.join(missing)} pero ese/esos target(s) no existen en el Makefile actual."
                    )

        success = not problems and not any("no existen en el Makefile" in w for w in warnings)
        return AgentResult(
            success, self.name, "validate_workflow",
            f"{len(problems)} problema(s) encontrado(s) en '{filename}'." if problems else f"'{filename}' parece correcto.",
            data={"problems": problems, "referenced_make_targets": referenced_targets}, warnings=warnings,
        )

    def generate_workflow(self, *, filename: str = "ci.yml", python_version: str | None = None, overwrite: bool = False) -> AgentResult:
        destination = self.ctx.root / WORKFLOWS_DIR / filename
        if destination.exists() and not overwrite:
            return AgentResult(
                False, self.name, "generate_workflow",
                f"Ya existe '{destination.relative_to(self.ctx.root)}'. Usa overwrite=True para sobreescribirlo.",
            )

        version = python_version or self.ctx.config.python_version
        module = self.ctx.config.project_slug
        if not module:
            return AgentResult(False, self.name, "generate_workflow", "project_slug está vacío — revisa .copier-answers.yml.")

        yaml_content = CICDTool.generate(module=module, python_version=version)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(yaml_content, encoding="utf-8")

        problems = CICDTool.validate(destination)
        return AgentResult(
            True, self.name, "generate_workflow",
            f"Workflow generado en '{destination.relative_to(self.ctx.root)}'.",
            data={"path": str(destination), "content": yaml_content},
            warnings=problems,
        )
