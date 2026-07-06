"""
agents.agents.installer_agent — Instala agentes externos en `agents/external/`.

Lee el aviso de seguridad en `agents/tools/agent_installer_tool.py` antes de
usar este agente: instalar código de un origen externo significa
ejecutarlo al importarlo. Este agente valida la FORMA (estructura AST) del
código candidato, no su contenido — no sustituye una revisión humana antes
de usar un agente de un origen que no sea de confianza.

Usa `agents/workspace/installer/` para el staging de repos clonados —
nunca clona directamente dentro de `agents/external/`, primero valida en
el workspace y solo copia el archivo concreto si pasa la validación.
"""

from __future__ import annotations

import shutil

from agents.core.base_agent import AgentResult, BaseAgent
from agents.core.registry import agent_registry, register_agent
from agents.exceptions import MissingDependencyError, ToolExecutionError
from agents.tools.agent_installer_tool import AgentInstallerTool


@register_agent
class InstallerAgent(BaseAgent):
    name = "installer"
    description = (
        "Instala agentes externos (por URL de git o ruta local) en agents/external/, "
        "valida su estructura y confirma que quedan registrados y funcionando."
    )
    capabilities = ["instalar agente", "clonar agente", "agente externo", "añadir agente", "installer"]

    def actions(self) -> dict:
        return {
            "install_from_git": self.install_from_git,
            "install_from_path": self.install_from_path,
            "list_installed": self.list_installed,
            "verify": self.verify,
        }

    # -------------------------------------------------------------------------
    def _stage_and_pick_candidate(self, source_root, *, subpath: str | None):
        """Busca candidatos en `source_root`; si hay más de uno y no se dio
        `subpath`, no adivina — devuelve None y dejar que el llamador falle explícito."""
        if subpath:
            candidate_path = source_root / subpath
            if not candidate_path.exists():
                return None, f"'{subpath}' no existe dentro de lo clonado/copiado."
            candidates = [c for c in AgentInstallerTool.find_agent_candidates(source_root) if c.path == candidate_path]
            if not candidates:
                return None, f"'{subpath}' no contiene ninguna clase con @register_agent."
            return candidates[0], None

        candidates = AgentInstallerTool.find_agent_candidates(source_root)
        if not candidates:
            return None, "No se encontró ninguna clase decorada con @register_agent en el origen."
        if len(candidates) > 1:
            paths = [str(c.path.relative_to(source_root)) for c in candidates]
            return None, f"Se encontraron {len(candidates)} agentes candidatos, especifica 'subpath': {paths}"
        return candidates[0], None

    def _finish_install(self, candidate, *, force: bool) -> AgentResult:
        destination_dir = self.ctx.root / "agents" / "external"
        try:
            installed_path = AgentInstallerTool.install_file(candidate.path, destination_dir, force=force)
        except FileExistsError as exc:
            return AgentResult(False, self.name, "install", str(exc))

        agent_registry.discover(force=True)
        registered = candidate.declared_name in agent_registry.all() if candidate.declared_name else False

        warnings = list(candidate.warnings)
        warnings.append(
            "SEGURIDAD: este código se ejecutó al importarse para verificar el registro. "
            "Revísalo tú mismo si el origen no es de completa confianza."
        )
        if not registered:
            warnings.append(
                "No se pudo confirmar el registro automáticamente (no se detectó 'name = \"...\"' "
                "como literal de clase, o el import falló en silencio) — comprueba con "
                "'python -m agents list' o el agente 'verify'."
            )

        return AgentResult(
            True, self.name, "install",
            f"Agente instalado en '{installed_path}'"
            + (f" y registrado como '{candidate.declared_name}'." if registered else " (registro sin confirmar, ver warnings)."),
            data={"installed_path": str(installed_path), "declared_name": candidate.declared_name, "registered": registered},
            warnings=warnings,
        )

    def install_from_git(self, *, repo_url: str, subpath: str | None = None, force: bool = False) -> AgentResult:
        resolved_url = AgentInstallerTool.normalize_github_shorthand(repo_url)
        workdir = self.ctx.agent_workspace("installer")
        repo_dir_name = resolved_url.rstrip("/").rsplit("/", 1)[-1].removesuffix(".git")
        clone_dir = workdir / repo_dir_name

        if clone_dir.exists():
            shutil.rmtree(clone_dir)
        try:
            AgentInstallerTool.clone_git_repo(resolved_url, clone_dir)
        except MissingDependencyError as exc:
            return AgentResult(False, self.name, "install_from_git", str(exc))
        except ToolExecutionError as exc:
            return AgentResult(False, self.name, "install_from_git", str(exc))

        candidate, error = self._stage_and_pick_candidate(clone_dir, subpath=subpath)
        if candidate is None:
            return AgentResult(False, self.name, "install_from_git", error)

        return self._finish_install(candidate, force=force)

    def install_from_path(self, *, local_path: str, subpath: str | None = None, force: bool = False) -> AgentResult:
        source_root = (self.ctx.root / local_path).resolve()
        if not source_root.exists():
            return AgentResult(False, self.name, "install_from_path", f"No existe '{local_path}'.")

        search_root = source_root if source_root.is_dir() else source_root.parent
        effective_subpath = subpath or (source_root.name if source_root.is_file() else None)

        candidate, error = self._stage_and_pick_candidate(search_root, subpath=effective_subpath)
        if candidate is None:
            return AgentResult(False, self.name, "install_from_path", error)

        return self._finish_install(candidate, force=force)

    def list_installed(self) -> AgentResult:
        external_dir = self.ctx.root / "agents" / "external"
        files = [
            p.name for p in external_dir.glob("*.py")
            if p.name not in ("__init__.py",) and not p.name.startswith("_")
        ]
        return AgentResult(True, self.name, "list_installed", f"{len(files)} agente(s) externo(s) instalado(s).", data=files)

    def verify(self, *, agent_name: str) -> AgentResult:
        agent_registry.discover(force=True)
        if agent_name not in agent_registry.all():
            return AgentResult(
                False, self.name, "verify",
                f"'{agent_name}' no está registrado. Revisa que el archivo esté en agents/external/ o agents/agents/ "
                f"y que la clase tenga @register_agent con name='{agent_name}'.",
            )
        agent_cls = agent_registry.get(agent_name)
        instance = agent_cls(context=self.ctx)
        return AgentResult(True, self.name, "verify", f"'{agent_name}' registrado y operativo.", data=instance.describe())
