"""
agents.agents.mlflow_agent — Analiza experimentos MLflow del proyecto.

Solo aplica si el proyecto se generó con `use_mlflow=true`. El nombre de
experimento por defecto es `project_slug`, igual que
`{{ project_slug }}/models/train_model.py` (ver `agents/tools/mlflow_tool.py`
para el detalle verificado, incluido un aviso sobre un cambio reciente en
el backend de tracking por defecto de mlflow).
"""

from __future__ import annotations

from agents.core.base_agent import AgentResult, BaseAgent
from agents.core.registry import register_agent
from agents.exceptions import MissingDependencyError
from agents.tools.mlflow_tool import MLflowTool


@register_agent
class MLflowAgent(BaseAgent):
    name = "mlflow"
    description = "Lista runs de MLflow, encuentra el mejor por una métrica, y avisa si el run más reciente empeoró respecto al anterior."
    capabilities = ["mlflow", "experimentos", "runs", "tracking de modelos"]

    def actions(self) -> dict:
        return {
            "list_runs": self.list_runs,
            "best_run": self.best_run,
            "compare_latest": self.compare_latest,
        }

    def _experiment_name(self, experiment_name: str | None) -> str | None:
        return experiment_name or self.ctx.config.project_slug or None

    def list_runs(self, *, experiment_name: str | None = None, max_results: int = 20) -> AgentResult:
        name = self._experiment_name(experiment_name)
        if not name:
            return AgentResult(False, self.name, "list_runs", "No se pudo determinar el nombre del experimento (project_slug vacío).")

        try:
            runs = MLflowTool.list_runs(name, max_results=max_results)
        except MissingDependencyError as exc:
            return AgentResult(False, self.name, "list_runs", str(exc))

        if runs is None:
            return AgentResult(
                True, self.name, "list_runs",
                f"No existe el experimento '{name}' todavía (¿se ha ejecutado 'make train' con use_mlflow=true?).",
                data=[],
            )
        return AgentResult(
            True, self.name, "list_runs", f"{len(runs)} run(s) encontrado(s) en el experimento '{name}'.",
            data=[r.__dict__ for r in runs],
        )

    def best_run(self, *, metric: str, experiment_name: str | None = None, higher_is_better: bool = True) -> AgentResult:
        name = self._experiment_name(experiment_name)
        if not name:
            return AgentResult(False, self.name, "best_run", "No se pudo determinar el nombre del experimento (project_slug vacío).")

        try:
            runs = MLflowTool.list_runs(name)
        except MissingDependencyError as exc:
            return AgentResult(False, self.name, "best_run", str(exc))
        if not runs:
            return AgentResult(True, self.name, "best_run", f"No hay runs en '{name}'.", data=None)

        best = MLflowTool.best_run(runs, metric, higher_is_better=higher_is_better)
        if best is None:
            return AgentResult(
                False, self.name, "best_run",
                f"Ningún run de '{name}' registra la métrica '{metric}' — revisa el nombre exacto.",
            )
        return AgentResult(True, self.name, "best_run", f"Mejor run por '{metric}': {best.run_id[:8]}.", data=best.__dict__)

    def compare_latest(self, *, metric: str, experiment_name: str | None = None, higher_is_better: bool = True) -> AgentResult:
        """Compara el run más reciente contra el inmediatamente anterior — no contra el histórico completo."""
        name = self._experiment_name(experiment_name)
        if not name:
            return AgentResult(False, self.name, "compare_latest", "No se pudo determinar el nombre del experimento (project_slug vacío).")

        try:
            runs = MLflowTool.list_runs(name, max_results=2)
        except MissingDependencyError as exc:
            return AgentResult(False, self.name, "compare_latest", str(exc))

        if not runs or len(runs) < 2:
            return AgentResult(
                True, self.name, "compare_latest",
                f"Se necesitan al menos 2 runs para comparar — '{name}' tiene {len(runs) if runs else 0}.",
                data=None,
            )

        latest, previous = runs[0], runs[1]
        if metric not in latest.metrics or metric not in previous.metrics:
            return AgentResult(
                False, self.name, "compare_latest",
                f"'{metric}' no está presente en ambos runs a comparar (últimos: {latest.run_id[:8]}, {previous.run_id[:8]}).",
            )

        delta = latest.metrics[metric] - previous.metrics[metric]
        regressed = (delta < 0) if higher_is_better else (delta > 0)
        warnings = [f"El run más reciente empeora '{metric}' en {abs(delta):.4g} respecto al anterior."] if regressed else []

        return AgentResult(
            True, self.name, "compare_latest",
            f"'{metric}': {previous.metrics[metric]:.4g} -> {latest.metrics[metric]:.4g} ({'peor' if regressed else 'igual o mejor'}).",
            data={"latest": latest.__dict__, "previous": previous.__dict__, "delta": delta, "regressed": regressed},
            warnings=warnings,
        )
