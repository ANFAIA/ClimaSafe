"""
agents.tools.docker_tool — Envoltorio sobre el CLI `docker` / `docker compose`
más un analizador estático de `Dockerfile` (sin dependencias, parseo de texto).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from agents.tools.process_tool import ProcessResult, run_command
from agents.tools.registry import register_tool


@dataclass
class DockerfileFinding:
    line_number: int
    severity: str  # "warning" | "info"
    message: str


@register_tool("docker")
@dataclass
class DockerTool:
    project_root: Path

    # -- CLI --------------------------------------------------------------------
    def ps(self) -> ProcessResult:
        return run_command(["docker", "ps"])

    def images(self) -> ProcessResult:
        return run_command(["docker", "images"])

    def compose_config(self) -> ProcessResult:
        """Valida docker-compose.yml (`docker compose config`) sin arrancar nada."""
        return run_command(["docker", "compose", "config"], cwd=self.project_root)

    def build(self, *, tag: str, dockerfile: Path | None = None) -> ProcessResult:
        args = ["docker", "build", "-t", tag]
        if dockerfile:
            args += ["-f", str(dockerfile)]
        args.append(str(self.project_root))
        return run_command(args, cwd=self.project_root, timeout=600)

    # -- análisis estático de Dockerfile -----------------------------------------
    @staticmethod
    def lint_dockerfile(dockerfile_path: Path) -> list[DockerfileFinding]:
        """
        Analiza un Dockerfile en busca de malas prácticas conocidas y bien
        documentadas (no es un sustituto de hadolint, pero cubre las más
        comunes sin añadir esa dependencia).
        """
        if not dockerfile_path.exists():
            return [DockerfileFinding(0, "warning", f"No existe {dockerfile_path}")]

        findings: list[DockerfileFinding] = []
        lines = dockerfile_path.read_text(encoding="utf-8").splitlines()

        saw_from = False
        saw_user = False

        for i, raw_line in enumerate(lines, start=1):
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            upper = line.upper()

            if upper.startswith("FROM"):
                saw_from = True
                image_ref = line.split()[1] if len(line.split()) > 1 else ""
                if ":" not in image_ref or image_ref.endswith(":latest"):
                    findings.append(
                        DockerfileFinding(
                            i, "warning",
                            f"Imagen base sin versión fijada ('{image_ref}'): "
                            f"usa un tag concreto (ej. python:3.12-slim) para builds reproducibles."
                        )
                    )

            if upper.startswith("USER"):
                saw_user = True

            if upper.startswith("ADD") and not line.split()[-1].startswith(("http://", "https://")):
                findings.append(
                    DockerfileFinding(
                        i, "info",
                        "ADD usado para copiar archivos locales: prefiere COPY, "
                        "que es más explícito (ADD tiene comportamiento especial con "
                        "URLs y archivos comprimidos)."
                    )
                )

            if upper.startswith("RUN") and " apt-get install" in upper.replace("-Y", "") and "--no-install-recommends" not in line:
                findings.append(
                    DockerfileFinding(
                        i, "info",
                        "apt-get install sin --no-install-recommends: la imagen puede "
                        "quedar más pesada de lo necesario."
                    )
                )

            if upper.startswith("RUN") and "apt-get update" in upper and "&&" not in line:
                findings.append(
                    DockerfileFinding(
                        i, "warning",
                        "apt-get update en un RUN separado de apt-get install: "
                        "invalida el cache de capas de forma inconsistente. "
                        "Combínalos en el mismo RUN con &&."
                    )
                )

        if saw_from and not saw_user:
            findings.append(
                DockerfileFinding(
                    len(lines), "warning",
                    "No se declara USER: el contenedor corre como root por defecto. "
                    "Considera crear un usuario sin privilegios para producción."
                )
            )

        return findings
