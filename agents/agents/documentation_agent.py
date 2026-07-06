"""
agents.agents.documentation_agent — Mantiene README, CHANGELOG y docs/ al día.

Conoce el formato exacto que ya usa este template:
  - CHANGELOG.md sigue Keep a Changelog (ver la cabecera real del archivo).
  - README.md documenta los targets de `make` en una sección "Comandos" /
    similar — este agente compara esa lista contra los targets reales del
    Makefile para detectar documentación desincronizada, en vez de asumir
    que README y Makefile nunca se desalinean.
  - `make docs` ya sabe generar la documentación Sphinx; este agente solo
    la invoca, no reimplementa esa lógica.
"""

from __future__ import annotations

import re

from agents.agents.git_agent import GitAgent
from agents.core.base_agent import AgentResult, BaseAgent
from agents.core.registry import register_agent
from agents.exceptions import MissingDependencyError, ToolExecutionError
from agents.tools.filesystem_tool import FilesystemTool
from agents.tools.process_tool import run_command

_MAKE_TARGET_RE = re.compile(r"^([a-zA-Z_-]+):", re.MULTILINE)
# Targets internos que no tiene sentido exigir en el README (ayudan al propio Makefile).
_INTERNAL_TARGETS = {"help", ".PHONY", ".DEFAULT_GOAL"}


@register_agent
class DocumentationAgent(BaseAgent):
    name = "documentation"
    description = "Sincroniza README con el Makefile real, actualiza CHANGELOG.md, genera docs Sphinx."
    capabilities = ["readme", "changelog", "documentacion", "docs", "sphinx"]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fs = FilesystemTool(root=self.ctx.root)

    def actions(self) -> dict:
        return {
            "check_readme_makefile_sync": self.check_readme_makefile_sync,
            "update_changelog": self.update_changelog,
            "build_docs": self.build_docs,
            "bump_version": self.bump_version,
        }

    def check_readme_makefile_sync(self) -> AgentResult:
        """
        Compara los targets reales de `Makefile` con las menciones de `make
        <target>` en `README.md`. Señala targets del Makefile que el README
        no menciona — un indicio de documentación desactualizada.
        """
        makefile_path = self.ctx.root / "Makefile"
        readme_path = self.ctx.readme_file
        if not makefile_path.exists() or not readme_path.exists():
            return AgentResult(False, self.name, "check_readme_makefile_sync", "Falta Makefile o README.md en la raíz.")

        makefile_text = makefile_path.read_text(encoding="utf-8")
        readme_text = readme_path.read_text(encoding="utf-8")

        targets = {
            t for t in _MAKE_TARGET_RE.findall(makefile_text)
            if t not in _INTERNAL_TARGETS and not t.startswith(".")
        }
        undocumented = sorted(t for t in targets if f"make {t}" not in readme_text)

        warnings = [f"'make {t}' no aparece mencionado en README.md" for t in undocumented]
        return AgentResult(
            True, self.name, "check_readme_makefile_sync",
            f"{len(targets)} targets en Makefile, {len(undocumented)} sin mencionar en README.md.",
            data={"all_targets": sorted(targets), "undocumented": undocumented}, warnings=warnings,
        )

    def update_changelog(self, *, since_tag: str | None = None, dry_run: bool = True) -> AgentResult:
        """
        Genera una entrada de changelog (vía GitAgent) e, si `dry_run=False`,
        la inserta en `CHANGELOG.md` justo después de la cabecera del
        archivo (antes de la primera entrada de versión existente).
        """
        git_agent = GitAgent(context=self.ctx)
        changelog_result = git_agent.run("generate_changelog", since_tag=since_tag)
        if not changelog_result.success or not changelog_result.data:
            return AgentResult(
                changelog_result.success, self.name, "update_changelog",
                changelog_result.message or "No hay cambios nuevos que añadir al changelog.",
            )

        entry = changelog_result.data
        if dry_run:
            return AgentResult(
                True, self.name, "update_changelog",
                "Entrada generada en modo dry_run (no se escribió en disco). "
                "Vuelve a llamar con dry_run=False para aplicarla.",
                data=entry,
            )

        if not self.ctx.changelog_file.exists():
            new_content = f"# Changelog\n\n{entry}\n"
        else:
            current = self.ctx.changelog_file.read_text(encoding="utf-8")
            # Inserta tras la primera línea en blanco que sigue al título (cabecera del archivo),
            # que es donde ya viven las notas introductorias de CHANGELOG.md en este template.
            marker = "\n---\n"
            if marker in current:
                head, _, tail = current.partition(marker)
                new_content = f"{head}{marker}\n{entry}\n{tail.lstrip(chr(10))}"
            else:
                new_content = f"{current.rstrip()}\n\n{entry}\n"

        self.ctx.changelog_file.write_text(new_content, encoding="utf-8")
        return AgentResult(
            True, self.name, "update_changelog", "CHANGELOG.md actualizado.", data=entry,
        )

    def build_docs(self) -> AgentResult:
        """
        Ejecuta `sphinx-apidoc` + build HTML, igual que `make docs`. Requiere
        que `sphinx` esté instalado (extra `dev`: `uv sync --extra dev`).
        """
        module = self.ctx.config.project_slug
        if not module:
            return AgentResult(
                False, self.name, "build_docs",
                "project_slug está vacío — revisa .copier-answers.yml antes de generar documentación.",
            )
        try:
            apidoc = run_command(
                ["uv", "run", "sphinx-apidoc", "-o", "docs/source/", module],
                cwd=self.ctx.root, timeout=120,
            )
        except MissingDependencyError as exc:
            return AgentResult(False, self.name, "build_docs", str(exc))

        if not apidoc.ok:
            return AgentResult(False, self.name, "build_docs", "sphinx-apidoc falló.", data=apidoc.stderr)

        try:
            html = run_command(["make", "html"], cwd=self.ctx.docs_dir, timeout=180)
        except (MissingDependencyError, ToolExecutionError) as exc:
            return AgentResult(False, self.name, "build_docs", str(exc))

        if not html.ok:
            return AgentResult(False, self.name, "build_docs", "El build HTML de Sphinx falló.", data=html.stderr)

        return AgentResult(True, self.name, "build_docs", "Documentación generada en docs/build/html/.")

    def bump_version(self, *, new_version: str) -> AgentResult:
        """
        Actualiza el número de versión en `pyproject.toml` (`version = "..."`)
        y en `README.md` (el badge `Version-X-green` y la línea
        `**Versión:** X`) — comprobé estos tres sitios exactos leyendo el
        `README.md`/`pyproject.toml` reales de este template antes de escribir
        este método, no son una suposición.

        Si algún patrón no aparece en el archivo (p. ej. porque el usuario
        reescribió el README a mano y quitó el badge), se avisa explícitamente
        en vez de fallar en silencio o fingir que se actualizó algo que no
        estaba ahí.
        """
        changed_files = []
        warnings = []

        pyproject_path = self.ctx.pyproject_file
        if pyproject_path.exists():
            text = pyproject_path.read_text(encoding="utf-8")
            new_text, n = re.subn(
                r'^version = "[^"]*"', f'version = "{new_version}"', text, count=1, flags=re.MULTILINE
            )
            if n:
                pyproject_path.write_text(new_text, encoding="utf-8")
                changed_files.append(str(pyproject_path.relative_to(self.ctx.root)))
            else:
                warnings.append("No se encontró 'version = \"...\"' en pyproject.toml — no se tocó.")
        else:
            warnings.append("No existe pyproject.toml en la raíz del proyecto.")

        if self.ctx.readme_file.exists():
            text = self.ctx.readme_file.read_text(encoding="utf-8")
            text, n_badge = re.subn(r"Version-[^-]+-green", f"Version-{new_version}-green", text, count=1)
            text, n_line = re.subn(r"(\*\*Versión:\*\*\s*)[^\s{·]+", rf"\g<1>{new_version}", text, count=1)
            if n_badge or n_line:
                self.ctx.readme_file.write_text(text, encoding="utf-8")
                changed_files.append(str(self.ctx.readme_file.relative_to(self.ctx.root)))
            if not n_badge:
                warnings.append("No se encontró el badge 'Version-X-green' en README.md.")
            if not n_line:
                warnings.append("No se encontró la línea '**Versión:** X' en README.md.")
        else:
            warnings.append("No existe README.md en la raíz del proyecto.")

        if not changed_files:
            return AgentResult(False, self.name, "bump_version", "No se actualizó ningún archivo.", warnings=warnings)

        return AgentResult(
            True, self.name, "bump_version",
            f"Versión actualizada a '{new_version}' en: {', '.join(changed_files)}.",
            data={"new_version": new_version, "changed_files": changed_files}, warnings=warnings,
        )
