"""
agents.context — Contexto compartido entre todos los agentes.

Todos los agentes reciben la misma instancia de `SharedContext`, calculada
una única vez por proceso. Esto evita que cada agente reimplemente su propia
lógica de "¿dónde está `data/raw/`?" y asegura que si mañana cambia la
estructura del template, solo hay que tocar este archivo.

Nota de diseño: este módulo NO importa `{{ project_slug }}.utils.paths` a
propósito. Los agentes deben poder listar `data/raw/`, leer el `Dockerfile`
o correr `git log` incluso si el paquete del proyecto todavía no se ha
instalado (`uv sync` no se ha ejecutado) o tiene un error de import. Por eso
las rutas se derivan directamente del sistema de archivos, con la misma
convención (`parents[N]` sobre `__file__`) que ya usa
`{{ project_slug }}/utils/paths.py`, para que ambos módulos coincidan si
algún día divergen y haya que depurarlo.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from agents.config import ProjectConfig, load_project_config

# agents/context.py -> parents[1] es la raíz del proyecto generado
# (agents/ vive directamente bajo la raíz, igual que data/, models/, tests/...)
_AGENTS_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _AGENTS_DIR.parent


@dataclass(frozen=True)
class SharedContext:
    """Rutas y configuración compartidas. Inmutable: si algo cambia, se crea otra instancia."""

    root: Path
    config: ProjectConfig

    # -- paquete del proyecto -------------------------------------------------
    @property
    def package_dir(self) -> Path:
        return self.root / self.config.project_slug

    # -- datos ------------------------------------------------------------------
    @property
    def data_dir(self) -> Path:
        return self.root / "data"

    @property
    def raw_data_dir(self) -> Path:
        return self.data_dir / "raw"

    @property
    def interim_data_dir(self) -> Path:
        return self.data_dir / "interim"

    @property
    def processed_data_dir(self) -> Path:
        return self.data_dir / "processed"

    # -- modelos ------------------------------------------------------------------
    @property
    def models_dir(self) -> Path:
        return self.root / "models"

    @property
    def artifacts_dir(self) -> Path:
        return self.models_dir / "artifacts"

    # -- reportes / figuras ------------------------------------------------------
    @property
    def reports_dir(self) -> Path:
        return self.root / "reports"

    @property
    def figures_dir(self) -> Path:
        return self.reports_dir / "figures"

    @property
    def monitoring_dir(self) -> Path:
        return self.reports_dir / "monitoring"

    # -- resto del proyecto --------------------------------------------------
    @property
    def tests_dir(self) -> Path:
        return self.root / "tests"

    @property
    def notebooks_dir(self) -> Path:
        return self.root / "notebooks"

    @property
    def docs_dir(self) -> Path:
        return self.root / "docs"

    @property
    def api_dir(self) -> Path:
        return self.root / "api"

    @property
    def monitoring_module_dir(self) -> Path:
        return self.root / "monitoring"

    @property
    def tuning_dir(self) -> Path:
        return self.root / "tuning"

    @property
    def dockerfile(self) -> Path:
        return self.root / "Dockerfile"

    @property
    def docker_compose_file(self) -> Path:
        return self.root / "docker-compose.yml"

    @property
    def changelog_file(self) -> Path:
        return self.root / "CHANGELOG.md"

    @property
    def readme_file(self) -> Path:
        return self.root / "README.md"

    @property
    def pyproject_file(self) -> Path:
        return self.root / "pyproject.toml"

    def runs_dir(self) -> Path | None:
        """Solo existe cuando ml_type == 'redes_neuronales' (logs de TensorBoard)."""
        if self.config.ml_type == "redes_neuronales":
            return self.root / "runs"
        return None

    # -- workspace de los propios agentes -----------------------------------
    @property
    def workspace_dir(self) -> Path:
        """
        Raíz donde los agentes guardan lo que generan (manifests, imágenes
        extraídas, staging de clonados...). Nunca en la raíz del proyecto —
        eso es una decisión explícita del usuario de este template, no un
        detalle de implementación: cada agente tiene su propia subcarpeta
        dentro de `agents/workspace/`.
        """
        return self.root / "agents" / "workspace"

    def agent_workspace(self, agent_name: str) -> Path:
        """Crea (si hace falta) y devuelve `agents/workspace/<agent_name>/`."""
        path = self.workspace_dir / agent_name
        path.mkdir(parents=True, exist_ok=True)
        return path


_cached_context: SharedContext | None = None


def get_context(root: Path | None = None, *, force_reload: bool = False) -> SharedContext:
    """
    Devuelve el `SharedContext` del proceso (singleton perezoso).

    Parameters
    ----------
    root          : raíz del proyecto. Por defecto se autodetecta como el
                    padre de `agents/`. Solo se pasa explícitamente en tests.
    force_reload  : ignora la caché y vuelve a leer `.copier-answers.yml`
                    (útil si un agente acaba de escribir esa configuración).
    """
    global _cached_context
    if _cached_context is not None and not force_reload and root is None:
        return _cached_context

    project_root = root or _PROJECT_ROOT
    context = SharedContext(root=project_root, config=load_project_config(project_root))

    if root is None:
        _cached_context = context
    return context
