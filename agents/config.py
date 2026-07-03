"""
agents.config — Configuración centralizada del sistema de agentes.

Este módulo NO depende de PyYAML (no es una dependencia del proyecto
generado). `.copier-answers.yml` es, en la práctica, una lista plana de
`clave: valor` sin anidamiento (así es como Copier la escribe), así que un
parser línea a línea cubre el caso real sin añadir una dependencia nueva.

Importante — esto NO es un parser YAML general. Si en el futuro
`.copier-answers.yml` incluyera listas, bloques multilínea o anidamiento,
este parser lo ignorará silenciosamente para esas claves. Si eso llega a
pasar, la opción honesta es añadir PyYAML como dependencia de `dev` y
sustituir `_parse_flat_yaml` por `yaml.safe_load`.

Fuente de verdad: si `.copier-answers.yml` no existe (p. ej. agents/ se está
probando fuera de un proyecto generado por copier), se usan los `default:` de
`copier.yml` tal y como estaban en el momento de escribir este módulo. Esos
valores por defecto pueden haber cambiado desde entonces — no lo des por
hecho, revisa `copier.yml` en la raíz del template si te importa la
precisión exacta.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

ANSWERS_FILENAME = ".copier-answers.yml"

_TRUE_STRINGS = {"true", "yes", "1", "on"}
_FALSE_STRINGS = {"false", "no", "0", "off"}


def _coerce(raw: str) -> Any:
    """Convierte el string crudo de una línea `key: value` a bool/int/str."""
    value = raw.strip()
    if value.startswith(("'", '"')) and value.endswith(("'", '"')) and len(value) >= 2:
        value = value[1:-1]
    low = value.lower()
    if low in _TRUE_STRINGS:
        return True
    if low in _FALSE_STRINGS:
        return False
    if value.lstrip("-").isdigit():
        return int(value)
    return value


def _parse_flat_yaml(text: str) -> dict[str, Any]:
    """Parser mínimo para pares `clave: valor` de una sola línea. Ver docstring del módulo."""
    data: dict[str, Any] = {}
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or stripped == "---":
            continue
        if ":" not in stripped:
            continue
        key, _, value = stripped.partition(":")
        key = key.strip()
        if not key or key.startswith("_"):
            # _src_path, _commit, etc. — metadatos de copier, no configuración de proyecto
            continue
        data[key] = _coerce(value)
    return data


@dataclass(frozen=True)
class ProjectConfig:
    """
    Respuestas de `copier.yml` relevantes para los agentes, con defaults
    seguros si `.copier-answers.yml` no existe o no define la clave.
    """

    project_name: str = "Proyecto"
    project_slug: str = ""
    project_author_name: str = "Nombre del autor"
    project_description: str = ""
    project_version: str = "0.1.0"
    python_version: str = "3.12"

    ml_type: str = "supervisado"
    task_type: str = "clasificacion"
    model_type: str = "todos"
    cluster_model: str = "todos"
    nn_model: str = "MLP"

    use_mlflow: bool = False
    use_monitoring: bool = False
    use_optuna: bool = False
    use_duckdb: bool = False
    use_api: bool = False
    use_docker: bool = False
    use_shap: bool = False
    use_xgboost: bool = False
    use_lightgbm: bool = False
    use_catboost: bool = False

    # Cualquier clave no reconocida cae aquí en vez de perderse.
    extra: dict[str, Any] = field(default_factory=dict, repr=False, compare=False)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ProjectConfig":
        known = {f.name for f in fields(cls) if f.name != "extra"}
        kwargs = {k: v for k, v in data.items() if k in known}
        extra = {k: v for k, v in data.items() if k not in known}
        return cls(**kwargs, extra=extra)

    def as_dict(self) -> dict[str, Any]:
        out = {f.name: getattr(self, f.name) for f in fields(self) if f.name != "extra"}
        out.update(self.extra)
        return out


def load_project_config(root: Path) -> ProjectConfig:
    """
    Carga la configuración del proyecto desde `<root>/.copier-answers.yml`.

    No lanza excepción si el archivo falta: devuelve `ProjectConfig()` con los
    defaults documentados arriba, para que `agents/` siga siendo usable en un
    checkout parcial o en desarrollo del propio template.
    """
    answers_path = root / ANSWERS_FILENAME
    if not answers_path.exists():
        return ProjectConfig()
    try:
        text = answers_path.read_text(encoding="utf-8")
    except OSError:
        return ProjectConfig()
    return ProjectConfig.from_dict(_parse_flat_yaml(text))


def env_override(key: str, default: str | None = None) -> str | None:
    """Permite sobreescribir cualquier config vía variable de entorno `DSKIT_AGENTS_<KEY>`."""
    return os.environ.get(f"DSKIT_AGENTS_{key.upper()}", default)
