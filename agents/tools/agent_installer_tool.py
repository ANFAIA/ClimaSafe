"""
agents.tools.agent_installer_tool — Clona/copia un agente externo, lo valida
estructuralmente y lo dispone para que `agents/core/registry.py` lo descubra.

Aviso de seguridad — léelo antes de usar esta herramienta, no lo saltees:
instalar un agente externo significa clonar código de un origen que no
controlas y luego IMPORTARLO, lo que lo EJECUTA (los decoradores y el
cuerpo de la clase corren al importar el módulo). Esto es, literalmente,
ejecución de código arbitrario. Esta herramienta no tiene ninguna forma de
verificar que el código sea seguro — solo verifica que tenga la FORMA de un
agente válido (AST: hereda de algo, usa el decorador correcto), no que su
contenido sea benigno. Revisa el código tú mismo antes de instalar algo de
un origen que no sea de completa confianza.
"""

from __future__ import annotations

import ast
import re
import shutil
from dataclasses import dataclass, field
from pathlib import Path

from agents.exceptions import MissingDependencyError, ToolExecutionError
from agents.tools.process_tool import run_command
from agents.tools.registry import register_tool

# Nombres de carpeta típicos de este template — si aparecen como string
# literal en el código de un agente candidato, es una señal de que puede
# no estar usando `self.ctx` (que resuelve rutas contra la raíz real del
# proyecto donde se instale) y en su lugar asume una ruta fija. No es una
# prueba concluyente (un agente externo podría, legítimamente, referirse a
# estas carpetas por otro motivo), así que se reporta como aviso, no como
# fallo — decide tú si es un problema real revisando el código señalado.
_SUSPICIOUS_PATH_RE = re.compile(
    r"^(\.?/)?(data|models|reports|tests|docs|notebooks|api|monitoring|tuning)(/|$)"
)


@dataclass
class AgentCandidate:
    path: Path
    declared_name: str | None
    class_name: str | None
    has_register_decorator: bool
    warnings: list[str] = field(default_factory=list)


def _decorator_name(decorator: ast.expr) -> str | None:
    if isinstance(decorator, ast.Name):
        return decorator.id
    if isinstance(decorator, ast.Attribute):
        return decorator.attr
    return None


def _uses_shared_context(node: ast.ClassDef) -> bool:
    """True si en algún sitio del cuerpo de la clase se referencia `self.ctx`."""
    for sub in ast.walk(node):
        if (
            isinstance(sub, ast.Attribute) and sub.attr == "ctx"
            and isinstance(sub.value, ast.Name) and sub.value.id == "self"
        ):
            return True
    return False


def _find_suspicious_literal_paths(node: ast.ClassDef) -> list[str]:
    """Strings literales que parecen rutas fijas a carpetas del proyecto, en vez de resueltas vía self.ctx."""
    found = []
    for sub in ast.walk(node):
        if isinstance(sub, ast.Constant) and isinstance(sub.value, str):
            if _SUSPICIOUS_PATH_RE.match(sub.value):
                found.append(sub.value)
    return sorted(set(found))


def _inspect_agent_class(node: ast.ClassDef) -> tuple[str | None, list[str]]:
    """Extrae `name = "..."` del cuerpo de la clase (best-effort) y avisa si faltan atributos esperados."""
    declared_name = None
    seen_attrs = set()
    for stmt in node.body:
        if isinstance(stmt, ast.Assign) and len(stmt.targets) == 1 and isinstance(stmt.targets[0], ast.Name):
            attr = stmt.targets[0].id
            seen_attrs.add(attr)
            if attr == "name" and isinstance(stmt.value, ast.Constant) and isinstance(stmt.value.value, str):
                declared_name = stmt.value.value

    warnings = []
    for expected in ("name", "description", "capabilities"):
        if expected not in seen_attrs:
            warnings.append(f"La clase no define '{expected}' como atributo de clase (se esperaba, como en BaseAgent).")
    method_names = {n.name for n in node.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}
    if "actions" not in method_names:
        warnings.append("La clase no define un método 'actions()' — no se podrá despachar ninguna acción por CLI/Orchestrator.")

    if not _uses_shared_context(node):
        warnings.append(
            "No se detectó ningún uso de 'self.ctx' en la clase — si este agente necesita rutas del "
            "proyecto (data/, models/, etc.), probablemente no se adapte solo al proyecto donde lo "
            "instales. Revisa si usa rutas fijas en su lugar (ver también el siguiente aviso, si aplica)."
        )

    suspicious_paths = _find_suspicious_literal_paths(node)
    if suspicious_paths:
        warnings.append(
            f"Rutas literales sospechosas en el código: {suspicious_paths} — si son fijas en vez de "
            f"resueltas vía 'self.ctx', este agente puede no adaptarse a la estructura real del "
            f"proyecto donde lo instales. No es concluyente (podría ser intencional), revísalo."
        )

    return declared_name, warnings


@register_tool("agent_installer")
class AgentInstallerTool:
    @staticmethod
    def normalize_github_shorthand(repo_url: str) -> str:
        """
        Convierte un atajo `usuario/repo` en una URL de git clonable
        (`https://github.com/usuario/repo.git`). Si `repo_url` ya parece una
        URL completa (contiene '://' o '@', típico de SSH `git@host:...`) o
        es una ruta local existente, se devuelve tal cual — esta función solo
        actúa sobre el caso concreto "una barra, sin esquema".
        """
        looks_like_url = "://" in repo_url or repo_url.startswith("git@")
        looks_like_local_path = Path(repo_url).exists()
        if looks_like_url or looks_like_local_path:
            return repo_url
        parts = repo_url.strip("/").split("/")
        if len(parts) == 2 and all(parts):
            return f"https://github.com/{parts[0]}/{parts[1]}.git"
        return repo_url

    @staticmethod
    def clone_git_repo(repo_url: str, destination: Path, *, depth: int = 1) -> None:
        try:
            result = run_command(
                ["git", "clone", "--depth", str(depth), repo_url, str(destination)],
                timeout=120, check=True,
            )
        except MissingDependencyError:
            raise
        except ToolExecutionError as exc:
            raise ToolExecutionError(f"No se pudo clonar '{repo_url}': {exc}") from exc
        del result  # el resultado no aporta nada más allá de "no lanzó" — check=True ya validó el éxito

    @staticmethod
    def find_agent_candidates(root: Path) -> list[AgentCandidate]:
        """
        Busca en `root` (recursivo) archivos .py que definan una clase
        decorada con `@register_agent`. Puramente estructural (AST) — no
        ejecuta nada del código encontrado.
        """
        candidates = []
        for path in root.rglob("*.py"):
            if "__pycache__" in path.parts:
                continue
            try:
                tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            except (SyntaxError, UnicodeDecodeError):
                continue

            for node in ast.walk(tree):
                if not isinstance(node, ast.ClassDef):
                    continue
                decorator_names = [_decorator_name(d) for d in node.decorator_list]
                if "register_agent" not in decorator_names:
                    continue
                declared_name, warnings = _inspect_agent_class(node)
                candidates.append(AgentCandidate(
                    path=path, declared_name=declared_name, class_name=node.name,
                    has_register_decorator=True, warnings=warnings,
                ))
        return candidates

    @staticmethod
    def install_file(source: Path, destination_dir: Path, *, force: bool = False) -> Path:
        destination_dir.mkdir(parents=True, exist_ok=True)
        destination = destination_dir / source.name
        if destination.exists() and not force:
            raise FileExistsError(f"Ya existe '{destination}'. Usa force=True para sobreescribirlo.")
        shutil.copy2(source, destination)
        return destination
