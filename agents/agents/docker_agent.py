"""
agents.agents.docker_agent — Revisión de la configuración Docker del proyecto.

Conoce que este template genera `Dockerfile` y `docker-compose.yml` en la
raíz solo cuando `use_docker=true` (interfaz de chat, ver `chat/app.py`) —
si no existen, lo dice explícitamente en vez de fallar con un traceback.
"""

from __future__ import annotations

from agents.core.base_agent import AgentResult, BaseAgent
from agents.core.registry import register_agent
from agents.exceptions import MissingDependencyError, ToolExecutionError
from agents.tools.docker_tool import DockerTool


@register_agent
class DockerAgent(BaseAgent):
    name = "docker"
    description = "Revisa Dockerfile y docker-compose.yml: malas prácticas, tamaño de imagen, seguridad."
    capabilities = ["docker", "dockerfile", "contenedor", "compose", "imagen"]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.docker = DockerTool(project_root=self.ctx.root)

    def actions(self) -> dict:
        return {
            "lint_dockerfile": self.lint_dockerfile,
            "validate_compose": self.validate_compose,
            "ps": self.ps,
        }

    def lint_dockerfile(self) -> AgentResult:
        if not self.ctx.dockerfile.exists():
            return AgentResult(
                False, self.name, "lint_dockerfile",
                "No existe Dockerfile en la raíz del proyecto (¿generaste el proyecto con use_docker=false?).",
            )
        findings = self.docker.lint_dockerfile(self.ctx.dockerfile)
        warnings = [f"L{f.line_number}: {f.message}" for f in findings if f.severity == "warning"]
        return AgentResult(
            True, self.name, "lint_dockerfile",
            f"{len(findings)} hallazgo(s) en Dockerfile.",
            data=[f.__dict__ for f in findings], warnings=warnings,
        )

    def validate_compose(self) -> AgentResult:
        if not self.ctx.docker_compose_file.exists():
            return AgentResult(
                False, self.name, "validate_compose", "No existe docker-compose.yml en la raíz del proyecto."
            )
        try:
            result = self.docker.compose_config()
        except MissingDependencyError as exc:
            return AgentResult(False, self.name, "validate_compose", str(exc))

        if not result.ok:
            return AgentResult(
                False, self.name, "validate_compose",
                "docker-compose.yml no es válido.", data=result.stderr,
            )
        return AgentResult(True, self.name, "validate_compose", "docker-compose.yml es válido.", data=result.stdout)

    def ps(self) -> AgentResult:
        try:
            result = self.docker.ps()
        except MissingDependencyError as exc:
            return AgentResult(False, self.name, "ps", str(exc))
        except ToolExecutionError as exc:
            return AgentResult(False, self.name, "ps", str(exc))
        return AgentResult(True, self.name, "ps", "Contenedores listados.", data=result.stdout)
