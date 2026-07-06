"""
agents.tools.mlflow_tool — Compara runs de MLflow.

Grounding: `{{ project_slug }}/models/train_model.py` de este mismo template
usa `mlflow.set_experiment("{{ project_slug }}")` — así que el nombre de
experimento por defecto de este agente es exactamente ese, no una
suposición genérica.

Aviso importante y muy reciente (lo encontré verificando esto, no lo sabía
de antes): en la versión de mlflow que se instala ahora mismo sin fijar
versión (igual que pandas en este template), el backend de archivos
`./mlruns` está en modo mantenimiento y el tracking URI por defecto de
`mlflow.get_tracking_uri()` resuelve a un `sqlite:///.../mlflow.db` en vez
del clásico directorio `mlruns/`. No asumas cuál de los dos usa tu proyecto
— este módulo no fuerza ningún tracking_uri, deja que mlflow resuelva el
suyo (o usa la variable de entorno MLFLOW_TRACKING_URI si la has fijado tú).
Verifica la versión de mlflow instalada si esto te importa.
"""

from __future__ import annotations

from dataclasses import dataclass

from agents.exceptions import MissingDependencyError
from agents.tools.registry import register_tool


@dataclass
class RunSummary:
    run_id: str
    status: str
    start_time: int | None
    metrics: dict
    params: dict


@register_tool("mlflow")
class MLflowTool:
    @staticmethod
    def _client():
        try:
            from mlflow.tracking import MlflowClient
        except ImportError as exc:
            raise MissingDependencyError(
                "mlflow no está instalado. Si tu proyecto se generó con use_mlflow=true, "
                "instálalo con: uv sync --extra mlflow_tracking"
            ) from exc
        return MlflowClient()

    @staticmethod
    def list_runs(experiment_name: str, *, max_results: int = 20) -> list[RunSummary] | None:
        """Devuelve None si el experimento no existe (no es un error — puede que aún no se haya entrenado nada)."""
        client = MLflowTool._client()
        experiment = client.get_experiment_by_name(experiment_name)
        if experiment is None:
            return None

        runs = client.search_runs(
            experiment_ids=[experiment.experiment_id], order_by=["start_time DESC"], max_results=max_results
        )
        return [
            RunSummary(
                run_id=run.info.run_id, status=run.info.status, start_time=run.info.start_time,
                metrics=dict(run.data.metrics), params=dict(run.data.params),
            )
            for run in runs
        ]

    @staticmethod
    def best_run(runs: list[RunSummary], metric: str, *, higher_is_better: bool = True) -> RunSummary | None:
        candidates = [r for r in runs if metric in r.metrics]
        if not candidates:
            return None
        return max(candidates, key=lambda r: r.metrics[metric]) if higher_is_better else min(
            candidates, key=lambda r: r.metrics[metric]
        )
