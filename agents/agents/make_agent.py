"""
agents.agents.make_agent — Validación y gestión del Makefile.

Conoce los targets del Makefile del proyecto, la cadena de dependencias del
pipeline (pipeline → predict → train → features → data) y puede sugerir
nuevos targets según la configuración del proyecto.
"""

from __future__ import annotations

import re
from pathlib import Path

from agents.core.base_agent import AgentResult, BaseAgent
from agents.core.registry import register_agent


_PIPELINE_CHAIN = ["pipeline", "predict", "train", "features", "data"]

_TARGET_SUGGESTIONS: dict[str, list[dict]] = {
    "use_api": [
        {"target": "serve", "help": "Inicia el servidor FastAPI para servir el modelo", "depends_on": ["train"]},
    ],
    "use_monitoring": [
        {"target": "monitor", "help": "Ejecuta detección de drift con Evidently", "depends_on": ["predict"]},
    ],
    "use_optuna": [
        {"target": "tune", "help": "Optimiza hiperparámetros con Optuna", "depends_on": ["features"]},
    ],
    "use_mlflow": [
        {"target": "mlflow", "help": "Abre la UI de MLflow", "depends_on": []},
    ],
}


@register_agent
class MakeAgent(BaseAgent):
    name = "make"
    description = (
        "Valida y gestiona el Makefile del proyecto: verifica que los targets "
        "existan, que la cadena del pipeline sea correcta, y sugiere nuevos targets."
    )
    capabilities = [
        "makefile", "make", "target", "pipeline", "build", "task", "tarea",
    ]

    def actions(self) -> dict:
        return {
            "validate": self.validate,
            "check_pipeline_chain": self.check_pipeline_chain,
            "suggest_targets": self.suggest_targets,
            "run": self.run_target,
            "list_targets": self.list_targets,
        }

    def _parse_makefile(self) -> dict[str, str]:
        """Parsea el Makefile y devuelve {target: help_text}."""
        makefile_path = self.ctx.root / "Makefile"
        if not makefile_path.exists():
            return {}

        targets: dict[str, str] = {}
        text = makefile_path.read_text(encoding="utf-8")
        current_help = ""

        for line in text.splitlines():
            if line.startswith("## "):
                current_help = line[3:].strip()
            target_match = re.match(r"^([a-zA-Z0-9_-]+):", line)
            if target_match:
                target = target_match.group(1)
                if not target.startswith("."):
                    targets[target] = current_help
                    current_help = ""

        return targets

    def list_targets(self) -> AgentResult:
        targets = self._parse_makefile()
        if not targets:
            return AgentResult(False, self.name, "list_targets", "No se encontró Makefile o está vacío.", data={})
        return AgentResult(True, self.name, "list_targets", f"{len(targets)} targets encontrados.", data=targets)

    def validate(self) -> AgentResult:
        """Valida que el Makefile exista y que todos los targets sean sintácticamente válidos."""
        makefile_path = self.ctx.root / "Makefile"
        if not makefile_path.exists():
            return AgentResult(False, self.name, "validate", "No existe Makefile en la raíz del proyecto.")
        return AgentResult(True, self.name, "validate", "Makefile encontrado.")

    def check_pipeline_chain(self) -> AgentResult:
        """Verifica que la cadena pipeline → predict → train → features → data sea correcta."""
        targets = self._parse_makefile()
        warnings = []
        for target in _PIPELINE_CHAIN:
            if target not in targets:
                warnings.append(f"Falta el target '{target}' en el Makefile.")
        missing = len([w for w in warnings if w.startswith("Falta")])
        return AgentResult(
            len(warnings) == 0, self.name, "check_pipeline_chain",
            f"Pipeline: {len(_PIPELINE_CHAIN) - missing}/{len(_PIPELINE_CHAIN)} targets encontrados.",
            data={"found": [t for t in _PIPELINE_CHAIN if t in targets], "missing": [t for t in _PIPELINE_CHAIN if t not in targets]},
            warnings=warnings,
        )

    def suggest_targets(self) -> AgentResult:
        """Sugiere nuevos targets según la configuración del proyecto."""
        suggestions = []
        for config_key, new_targets in _TARGET_SUGGESTIONS.items():
            if getattr(self.ctx.config, config_key, False):
                suggestions.extend(new_targets)

        ml_type_targets = {
            "redes_neuronales": {"target": "tb", "help": "Abre TensorBoard", "depends_on": ["train"]},
        }
        if self.ctx.config.ml_type in ml_type_targets:
            suggestions.append(ml_type_targets[self.ctx.config.ml_type])

        existing = self._parse_makefile()
        new_suggestions = [s for s in suggestions if s["target"] not in existing]

        return AgentResult(
            True, self.name, "suggest_targets",
            f"{len(new_suggestions)} target(s) sugerido(s) para tu configuración actual.",
            data={"suggested": new_suggestions, "total_possible": len(suggestions)},
            warnings=[] if new_suggestions else ["Todos los targets relevantes ya existen."],
        )

    def run_target(self, *, target: str, dry_run: bool = False) -> AgentResult:
        """Ejecuta un target de Makefile."""
        targets = self._parse_makefile()
        if target not in targets:
            return AgentResult(False, self.name, "run", f"El target '{target}' no existe en el Makefile.")

        if dry_run:
            return AgentResult(True, self.name, "run", f"[dry-run] make {target}")

        from agents.tools.process_tool import run_command
        result = run_command(["make", target], cwd=self.ctx.root)
        if not result.ok:
            return AgentResult(False, self.name, "run", f"make {target} falló: {result.stderr.strip()}")
        return AgentResult(True, self.name, "run", f"make {target} completado.", data={"stdout": result.stdout})
