"""
agents.agents.ml_agent — Análisis de modelos entrenados para este template.

Conoce que los modelos se guardan en `models/*.joblib` (ver
`{{ project_slug }}/models/train_model.py`) y los encoders/scalers en
`models/artifacts/`. No entrena modelos nuevos — eso es responsabilidad de
`make train`; este agente analiza lo que ya existe en disco.
"""

from __future__ import annotations

from agents.core.base_agent import AgentResult, BaseAgent
from agents.core.registry import register_agent
from agents.tools.sklearn_tool import SklearnTool


@register_agent
class MLAgent(BaseAgent):
    name = "ml"
    description = (
        "Analiza modelos entrenados (.joblib): overfitting/underfitting a partir de "
        "métricas dadas, importancia de variables, inspección de hiperparámetros."
    )
    capabilities = [
        "modelo", "overfitting", "underfitting", "hiperparametros", "importancia",
        "metricas", "pipeline", "algoritmo", "entrenamiento",
    ]

    def actions(self) -> dict:
        return {
            "list_models": self.list_models,
            "inspect_model": self.inspect_model,
            "feature_importance": self.feature_importance,
            "check_overfitting": self.check_overfitting,
        }

    def list_models(self) -> AgentResult:
        models = SklearnTool.list_models(self.ctx.models_dir)
        if not models:
            return AgentResult(
                True, self.name, "list_models",
                "No hay modelos en models/ todavía (ejecuta 'make train' primero).",
                data=[],
            )
        return AgentResult(
            True, self.name, "list_models", f"{len(models)} modelo(s) encontrado(s).",
            data=[p.name for p in models],
        )

    def inspect_model(self, *, model_name: str) -> AgentResult:
        path = self._resolve_model_path(model_name)
        if path is None:
            return AgentResult(False, self.name, "inspect_model", f"No se encontró el modelo '{model_name}' en models/.")
        try:
            estimator = SklearnTool.load(path)
        except Exception as exc:  # noqa: BLE001 — cualquier fallo de deserialización es un error de datos, no de lógica
            return AgentResult(False, self.name, "inspect_model", f"No se pudo cargar '{model_name}': {exc}")

        info = SklearnTool.inspect(estimator)
        return AgentResult(True, self.name, "inspect_model", f"'{model_name}' inspeccionado.", data=info)

    def feature_importance(self, *, model_name: str, feature_names: list[str] | None = None) -> AgentResult:
        path = self._resolve_model_path(model_name)
        if path is None:
            return AgentResult(False, self.name, "feature_importance", f"No se encontró el modelo '{model_name}' en models/.")
        estimator = SklearnTool.load(path)
        importances = SklearnTool.feature_importances(estimator, feature_names)
        if importances is None:
            return AgentResult(
                True, self.name, "feature_importance",
                f"'{model_name}' no expone feature_importances_ ni coef_ (¿es KNN u otro modelo sin esa propiedad?).",
                data=None,
            )
        top = dict(list(importances.items())[:15])
        return AgentResult(
            True, self.name, "feature_importance",
            f"Top {len(top)} variables por importancia en '{model_name}'.", data=top,
        )

    def check_overfitting(self, *, train_score: float, test_score: float, gap_threshold: float = 0.1) -> AgentResult:
        verdict = SklearnTool.detect_overfitting(train_score, test_score, gap_threshold=gap_threshold)
        warnings = [] if verdict["verdict"] == "ok" else [verdict["note"]]
        return AgentResult(
            True, self.name, "check_overfitting", verdict["note"], data=verdict, warnings=warnings,
        )

    def _resolve_model_path(self, model_name: str):
        candidate = self.ctx.models_dir / model_name
        if not candidate.suffix:
            candidate = candidate.with_suffix(".joblib")
        return candidate if candidate.exists() else None
