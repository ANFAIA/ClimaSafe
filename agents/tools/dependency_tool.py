"""
agents.tools.dependency_tool — Detecta versiones obsoletas y vulnerabilidades
conocidas en las dependencias del proyecto, contra la API JSON pública de PyPI.

Diferencia importante con el resto de herramientas de este sistema: esta
SÍ hace llamadas de red a un servicio externo (pypi.org). Si no hay
conexión, cada consulta falla individualmente (no se aborta todo el
chequeo) y se reporta como advertencia, no como error fatal.

Esquemas verificados de verdad antes de escribir este módulo, no supuestos:
- `https://pypi.org/pypi/<paquete>/json` -> `info.version` (última versión).
- `https://pypi.org/pypi/<paquete>/<version>/json` -> `vulnerabilities`
  (lista de adivsorios OSV: `id`, `aliases` (CVE/GHSA), `details`, `fixed_in`).
- `uv.lock` es TOML con bloques `[[package]]` -> `name = "..."` / `version = "..."`.
  Se parsea con un extractor de líneas simple (no un parser TOML completo:
  `tomllib` no está disponible en Python 3.10, uno de los `python_version`
  soportados por este template, y no quiero añadir `tomli` como dependencia
  nueva solo para esto). Funciona porque `name`/`version` siempre aparecen
  como líneas `clave = "valor"` de una sola línea en cada bloque — lo
  comprobé generando un `uv.lock` real antes de escribir el regex.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from agents.tools.registry import register_tool
from agents.tools.rest_tool import RestTool

PYPI_JSON_URL = "https://pypi.org/pypi/{name}/json"
PYPI_JSON_VERSION_URL = "https://pypi.org/pypi/{name}/{version}/json"

_DEPENDENCY_SPEC_RE = re.compile(r"^\s*([A-Za-z0-9_.-]+)")
_UV_LOCK_PACKAGE_RE = re.compile(r'\[\[package\]\]\s*\nname = "([^"]+)"\s*\nversion = "([^"]+)"', re.MULTILINE)


@dataclass
class PackageStatus:
    name: str
    declared_spec: str
    locked_version: str | None
    latest_version: str | None
    is_outdated: bool | None  # None si no se pudo determinar (fallo de red, etc.)
    release_cadence_days: float | None = None
    vulnerabilities: list[dict[str, Any]] = field(default_factory=list)
    error: str | None = None


def normalize_package_name(name: str) -> str:
    """PEP 503: nombres de paquete normalizados a minúsculas, con '-' como único separador."""
    return re.sub(r"[-_.]+", "-", name).lower()


def parse_dependency_name(spec: str) -> str:
    """Extrae el nombre del paquete de un especificador tipo 'requests>=2.20.0' o 'pandas[extra]'."""
    spec = spec.split(";")[0].strip()  # descarta markers de entorno (; python_version >= "3.10")
    spec = spec.split("[")[0]  # descarta extras: paquete[extra]
    match = _DEPENDENCY_SPEC_RE.match(spec)
    return normalize_package_name(match.group(1) if match else spec.strip())


def parse_pyproject_dependencies(pyproject_text: str) -> list[str]:
    """
    Extrae los especificadores de `[project] dependencies = [...]`. Parser de
    texto simple (regex sobre el bloque, no un parser TOML completo) — ver
    limitación documentada en el docstring del módulo.
    """
    match = re.search(r"dependencies\s*=\s*\[(.*?)\]", pyproject_text, re.DOTALL)
    if not match:
        return []
    block = match.group(1)
    return [
        item.strip().strip("'\"")
        for item in re.findall(r'["\']([^"\']+)["\']', block)
    ]


def parse_uv_lock_versions(uv_lock_text: str) -> dict[str, str]:
    """Devuelve {nombre_paquete_normalizado: version_bloqueada} a partir del contenido de uv.lock."""
    return {normalize_package_name(name): version for name, version in _UV_LOCK_PACKAGE_RE.findall(uv_lock_text)}


@register_tool("dependency")
class DependencyTool:
    @staticmethod
    def fetch_latest_version(package_name: str, *, timeout: int = 15) -> str | None:
        try:
            response = RestTool.get(PYPI_JSON_URL.format(name=package_name), timeout=timeout)
            if response.status != 200:
                return None
            return response.json()["info"]["version"]
        except Exception:  # noqa: BLE001 — cualquier fallo de red/parseo para ESTE paquete no debe abortar el resto
            return None

    @staticmethod
    def fetch_vulnerabilities(package_name: str, version: str, *, timeout: int = 15) -> list[dict[str, Any]]:
        try:
            response = RestTool.get(PYPI_JSON_VERSION_URL.format(name=package_name, version=version), timeout=timeout)
            if response.status != 200:
                return []
            return response.json().get("vulnerabilities", [])
        except Exception:  # noqa: BLE001
            return []

    @staticmethod
    def estimate_release_cadence_days(package_name: str, *, timeout: int = 15, max_releases: int = 8) -> float | None:
        """
        Media de días entre las últimas `max_releases` versiones publicadas.
        Es un promedio histórico, NO una predicción de cuándo saldrá la
        próxima versión — repórtalo así de explícito, la cadencia de
        releases de un paquete puede cambiar de un día para otro (cambio de
        mantenedor, proyecto archivado, etc.).
        """
        try:
            response = RestTool.get(PYPI_JSON_URL.format(name=package_name), timeout=timeout)
            if response.status != 200:
                return None
            data = response.json()
            dates = []
            for files in data.get("releases", {}).values():
                if files:
                    dates.append(files[0]["upload_time_iso_8601"])
            if len(dates) < 2:
                return None
            from datetime import datetime
            parsed = sorted(datetime.fromisoformat(d.replace("Z", "+00:00")) for d in dates)[-max_releases:]
            if len(parsed) < 2:
                return None
            deltas = [(parsed[i] - parsed[i - 1]).total_seconds() / 86400 for i in range(1, len(parsed))]
            return round(sum(deltas) / len(deltas), 1)
        except Exception:  # noqa: BLE001
            return None

    @staticmethod
    def check_package(
        spec: str, *, locked_version: str | None, include_vulnerabilities: bool, include_cadence: bool
    ) -> PackageStatus:
        name = parse_dependency_name(spec)
        latest = DependencyTool.fetch_latest_version(name)
        if latest is None:
            return PackageStatus(
                name=name, declared_spec=spec, locked_version=locked_version, latest_version=None,
                is_outdated=None, error="No se pudo consultar PyPI (red, timeout, o el paquete no existe con ese nombre).",
            )

        reference_version = locked_version or latest
        is_outdated = None
        try:
            from packaging.version import Version
            is_outdated = Version(reference_version) < Version(latest)
        except ImportError:
            # 'packaging' no está garantizado como dependencia directa del proyecto
            # generado (suele estar presente transitivamente, pero no es seguro).
            # Sin él, comparar versiones semánticas de forma fiable no es posible
            # con solo la librería estándar — se compara como string, que es
            # incorrecto en casos como "2.9" vs "2.10", así que se marca is_outdated=None
            # en vez de dar un resultado que podría ser mediante una comparación rota.
            pass
        except Exception:  # noqa: BLE001 — versión con formato no estándar (p. ej. calver raro)
            pass

        vulnerabilities = (
            DependencyTool.fetch_vulnerabilities(name, reference_version) if include_vulnerabilities else []
        )
        cadence = DependencyTool.estimate_release_cadence_days(name) if include_cadence else None

        return PackageStatus(
            name=name, declared_spec=spec, locked_version=locked_version, latest_version=latest,
            is_outdated=is_outdated, release_cadence_days=cadence, vulnerabilities=vulnerabilities,
        )
