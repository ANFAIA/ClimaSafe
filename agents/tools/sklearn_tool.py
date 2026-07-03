"""
agents.tools.sklearn_tool — Carga e inspección de modelos scikit-learn (.joblib).

joblib y scikit-learn están en las dependencias base del template, así que
se importan directamente.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np

from agents.tools.registry import register_tool


@register_tool("sklearn")
class SklearnTool:
    @staticmethod
    def load(path: Path) -> Any:
        return joblib.load(path)

    @staticmethod
    def list_models(models_dir: Path) -> list[Path]:
        return sorted(models_dir.glob("*.joblib"))

    @staticmethod
    def inspect(estimator: Any) -> dict[str, Any]:
        """Resumen genérico de un estimador ya entrenado: tipo, params, tamaño en memoria."""
        info: dict[str, Any] = {
            "class": type(estimator).__name__,
            "module": type(estimator).__module__,
        }
        if hasattr(estimator, "get_params"):
            info["params"] = estimator.get_params()
        if hasattr(estimator, "n_features_in_"):
            info["n_features_in"] = int(estimator.n_features_in_)
        if hasattr(estimator, "classes_"):
            info["classes"] = list(np.asarray(estimator.classes_).tolist())
        return info

    @staticmethod
    def feature_importances(estimator: Any, feature_names: list[str] | None = None) -> dict[str, float] | None:
        """
        Devuelve {feature: importancia} ordenado descendente, o None si el
        estimador no expone `feature_importances_` ni `coef_` (p. ej. KNN).
        Soporta Pipelines de sklearn.pipeline.Pipeline buscando el último step.
        """
        target = estimator
        if hasattr(estimator, "steps"):  # sklearn.pipeline.Pipeline
            target = estimator.steps[-1][1]

        if hasattr(target, "feature_importances_"):
            values = np.asarray(target.feature_importances_)
        elif hasattr(target, "coef_"):
            coef = np.asarray(target.coef_)
            values = np.abs(coef[0]) if coef.ndim > 1 else np.abs(coef)
        else:
            return None

        names = feature_names or [f"feature_{i}" for i in range(len(values))]
        if len(names) != len(values):
            names = [f"feature_{i}" for i in range(len(values))]
        pairs = sorted(zip(names, values.tolist()), key=lambda p: -abs(p[1]))
        return dict(pairs)

    @staticmethod
    def detect_overfitting(train_score: float, test_score: float, *, gap_threshold: float = 0.1) -> dict[str, Any]:
        """
        Heurística simple: si train_score - test_score supera `gap_threshold`
        (en la misma métrica, más alto = mejor), hay indicio de overfitting.
        Si test_score - train_score es notablemente positivo, revisa el split
        (posible fuga de datos entre train y test).
        """
        gap = train_score - test_score
        if gap > gap_threshold:
            verdict = "overfitting"
            note = f"train supera a test por {gap:.3f} (umbral {gap_threshold}) — el modelo memoriza en vez de generalizar."
        elif gap < -gap_threshold:
            verdict = "sospechoso"
            note = f"test supera a train por {-gap:.3f} — revisa el split, no es lo esperable."
        else:
            verdict = "ok"
            note = f"diferencia train/test de {gap:.3f}, dentro de lo razonable."
        return {"verdict": verdict, "train_score": train_score, "test_score": test_score, "gap": gap, "note": note}
