"""
agents.tools.cicd_tool — Genera y valida workflows de GitHub Actions.

Grounding explícito (no inventado): las versiones de actions usadas aquí
(`actions/checkout@v4`, `astral-sh/setup-uv@v8.1.0`) se verificaron contra
la documentación oficial de Astral (docs.astral.sh/uv/guides/integration/github)
y el repo astral-sh/setup-uv en el momento de escribir este módulo. Las
actions de GitHub cambian versión con frecuencia — si ha pasado tiempo desde
entonces, vale la pena comprobar si hay una versión más reciente antes de
asumir que estas siguen siendo las recomendadas.

Validación: la comprobación de claves obligatorias (`on`, `jobs`) se hace
con regex, sin depender de un parser YAML — funciona para el caso simple de
comprobar presencia de claves de primer nivel. Para una validación
estructural más profunda (¿cada job tiene `runs-on` y `steps`?) se usa
PyYAML si está disponible, con import perezoso — no es una dependencia
nueva del proyecto generado, es opcional y ya la trae `copier` en el
entorno de desarrollo de muchos usuarios, pero no se asume instalada.
"""

from __future__ import annotations

import re
from pathlib import Path

from agents.tools.registry import register_tool

WORKFLOWS_DIR = ".github/workflows"

# Verificado contra docs.astral.sh/uv/guides/integration/github/ — revisar
# si ha pasado mucho tiempo, estas versiones se quedan desactualizadas rápido.
_CHECKOUT_ACTION = "actions/checkout@v4"
_SETUP_UV_ACTION = "astral-sh/setup-uv@v8.1.0"


def generate_ci_yaml(*, module: str, python_version: str, lint_target: str = "lint", test_target: str = "test") -> str:
    """Genera un workflow mínimo: lint + test en push/PR, usando los targets reales del Makefile."""
    return f"""\
# Generado por agents/agents/cicd_agent.py — revisa las versiones de las
# actions (checkout, setup-uv) periódicamente, cambian con frecuencia.
name: CI

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  lint-and-test:
    name: Lint + Test
    runs-on: ubuntu-latest
    steps:
      - uses: {_CHECKOUT_ACTION}

      - name: Instalar uv
        uses: {_SETUP_UV_ACTION}
        with:
          enable-cache: true

      - name: Instalar Python {python_version}
        run: uv python install {python_version}

      - name: Instalar dependencias
        run: uv sync --extra dev

      - name: Lint
        run: make {lint_target} MODULE={module}

      - name: Test
        run: make {test_target}
"""


@register_tool("cicd")
class CICDTool:
    @staticmethod
    def generate(*, module: str, python_version: str) -> str:
        return generate_ci_yaml(module=module, python_version=python_version)

    @staticmethod
    def validate(path: Path) -> list[str]:
        """
        Devuelve una lista de problemas encontrados (vacía si no hay
        ninguno). Combina comprobaciones regex (siempre) con comprobaciones
        más profundas vía PyYAML si está disponible.
        """
        if not path.exists():
            return [f"No existe '{path}'."]

        text = path.read_text(encoding="utf-8")
        problems = []

        if not re.search(r"^on:", text, re.MULTILINE):
            problems.append("No se encontró la clave de primer nivel 'on:' (disparadores del workflow).")
        if not re.search(r"^jobs:", text, re.MULTILINE):
            problems.append("No se encontró la clave de primer nivel 'jobs:'.")

        try:
            import yaml  # PyYAML — opcional, import perezoso
        except ImportError:
            problems.append(
                "PyYAML no está instalado — solo se hicieron comprobaciones superficiales (regex). "
                "Instala pyyaml para una validación estructural más profunda (cada job con 'runs-on' y 'steps')."
            )
            return problems

        try:
            parsed = yaml.safe_load(text)
        except yaml.YAMLError as exc:
            problems.append(f"El archivo no es YAML válido: {exc}")
            return problems

        if not isinstance(parsed, dict):
            problems.append("El contenido no es un mapeo YAML de primer nivel (¿archivo vacío o mal formado?).")
            return problems

        jobs = parsed.get("jobs", {})
        if not jobs:
            problems.append("'jobs:' está vacío o no existe tras parsear el YAML.")
        for job_name, job_def in (jobs or {}).items():
            if not isinstance(job_def, dict):
                problems.append(f"El job '{job_name}' no es un mapeo válido.")
                continue
            if "runs-on" not in job_def:
                problems.append(f"El job '{job_name}' no define 'runs-on'.")
            if "steps" not in job_def or not job_def["steps"]:
                problems.append(f"El job '{job_name}' no define 'steps' (o está vacío).")

        return problems

    @staticmethod
    def referenced_make_targets(path: Path) -> list[str]:
        """Extrae los targets 'make X' invocados en pasos 'run:' del workflow — para cruzarlos con el Makefile real."""
        if not path.exists():
            return []
        text = path.read_text(encoding="utf-8")
        return re.findall(r"make\s+([a-zA-Z_-]+)", text)
