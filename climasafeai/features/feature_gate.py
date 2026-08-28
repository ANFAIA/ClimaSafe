"""
feature_gate.py — Feature Gate anti-leakage inspirado en el PPE de Google.

Evalúa cada covariable contra 4 criterios antes de que entre al modelo:
  1. No es sub-componente matemático del target.
  2. No viene de la misma encuesta que el target.
  3. No es efecto causal downstream del target.
  4. No usa datos temporales futuros.

Si una feature falla algún criterio se filtra y se registra el motivo.
El gate produce un informe (report) que se pega en la traza de entrenamiento.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Sequence

import pandas as pd


# ---------------------------------------------------------------------------
# Reglas concretas para ClimaSafeAI
# ---------------------------------------------------------------------------

class LeakReason(str, Enum):
    """Motivo por el que una feature se filtra."""
    MATH_SUB_COMPONENT = "math_sub_component"
    SAME_SURVEY_SOURCE = "same_survey_source"
    DOWNSTREAM_CAUSAL = "downstream_causal"
    FUTURE_TEMPORAL = "future_temporal"


# Columnas que son sub-componente matemático directo del target.
# El target sale de percentiles sobre estas columnas → si entran como
# feature, el modelo ve el resultado que intenta predecir.
_MATH_SUB_COMPONENT_COLUMNS: set[str] = {
    "defunciones_atrib_exc_temp",   # de aquí sale clase_riesgo_calor
    "defunciones_atrib_def_temp",   # de aquí sale clase_riesgo_frio
}

# Columnas derivadas de MoMo (misma encuesta que el target).
# La encuesta de MoMo provee mortalidad Y las etiquetas de riesgo;
# las features meteorológicas de ERA5 son independientes (otra fuente).
_SAME_SURVEY_COLUMNS: set[str] = {
    "defunciones_atrib_exc_temp",
    "defunciones_atrib_def_temp",
    "clase_riesgo_calor",
    "clase_riesgo_frio",
    "clase_riesgo_calor_label",
    "clase_riesgo_frio_label",
}

# Columnas que serían efecto causal downstream del target (muertes).
# Hoy no existen en el pipeline, pero el gate las lista para que si
# aparecen se filtren automáticamente (prevención).
_DOWNSTREAM_CAUSAL_COLUMNS: set[str] = {
    "ingresos_hospitalarios",
    "llamadas_112",
    "urgencias_calor",
    "urgencias_frio",
}

# Columnas que usan datos temporales futuros (pre-monitoring).
# Hoy se excluyen por el split temporal; el gate las lista como guardia.
_FUTURE_TEMPORAL_COLUMNS: set[str] = set()


# ---------------------------------------------------------------------------
# Resultado del gate
# ---------------------------------------------------------------------------

@dataclass
class FeatureVerdict:
    """Resultado del chequeo anti-leakage de una feature."""
    feature: str
    passed: bool
    failed_criteria: list[LeakReason] = field(default_factory=list)


@dataclass
class GateReport:
    """Informe completo del feature gate."""
    clase: str
    all_features: list[str]
    verdicts: list[FeatureVerdict]

    @property
    def filtered(self) -> list[FeatureVerdict]:
        return [v for v in self.verdicts if not v.passed]

    @property
    def passed_features(self) -> list[str]:
        return [v.feature for v in self.verdicts if v.passed]

    @property
    def filtered_features(self) -> list[dict[str, str]]:
        return [
            {"feature": v.feature, "reasons": [r.value for r in v.failed_criteria]}
            for v in self.filtered
        ]

    def summary(self) -> str:
        """Resumen legible para pegar en la traza de entrenamiento."""
        lines = [
            f"=== Feature Gate ({self.clase}) ===",
            f"  Total features evaluadas: {len(self.all_features)}",
            f"  Pasan el gate:           {len(self.passed_features)}",
            f"  Filtradas:               {len(self.filtered)}",
        ]
        if self.filtered:
            lines.append("")
            lines.append("  Filtradas:")
            for v in self.filtered:
                reasons = ", ".join(r.value for r in v.failed_criteria)
                lines.append(f"    - {v.feature}: {reasons}")
        lines.append("")
        lines.append(f"  Features que pasan: {self.passed_features}")
        lines.append("==========================")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Chequeos individuales
# ---------------------------------------------------------------------------

def _check_math_sub_component(feature: str) -> bool:
    """¿Es esta feature un sub-componente matemático del target?"""
    return feature in _MATH_SUB_COMPONENT_COLUMNS


def _check_same_survey(feature: str) -> bool:
    """¿Viene de la misma encuesta que el target (MoMo)?"""
    return feature in _SAME_SURVEY_COLUMNS


def _check_downstream_causal(feature: str) -> bool:
    """¿Es efecto causal downstream del target?"""
    return feature in _DOWNSTREAM_CAUSAL_COLUMNS


def _check_future_temporal(feature: str) -> bool:
    """¿Usa datos temporales futuros?"""
    return feature in _FUTURE_TEMPORAL_COLUMNS


_CHECKS: list[tuple[LeakReason, callable]] = [
    (LeakReason.MATH_SUB_COMPONENT, _check_math_sub_component),
    (LeakReason.SAME_SURVEY_SOURCE, _check_same_survey),
    (LeakReason.DOWNSTREAM_CAUSAL, _check_downstream_causal),
    (LeakReason.FUTURE_TEMPORAL, _check_future_temporal),
]


# ---------------------------------------------------------------------------
# API pública
# ---------------------------------------------------------------------------

def evaluate_feature(
    feature: str,
    checks: Sequence[tuple[LeakReason, callable]] | None = None,
) -> FeatureVerdict:
    """
    Evalúa una feature contra los 4 criterios anti-leakage.

    Returns
    -------
    FeatureVerdict con passed=True si pasa todos los checks, o passed=False
    con la lista de criterios que falló.
    """
    if checks is None:
        checks = _CHECKS

    failed: list[LeakReason] = []
    for reason, check_fn in checks:
        if check_fn(feature):
            failed.append(reason)

    return FeatureVerdict(
        feature=feature,
        passed=len(failed) == 0,
        failed_criteria=failed,
    )


def run_feature_gate(
    df: pd.DataFrame,
    clase: str,
    target_col: str | None = None,
    exclude_cols: Sequence[str] | None = None,
    extra_blocked: Sequence[str] | None = None,
) -> GateReport:
    """
    Ejecuta el feature gate sobre un DataFrame.

    Evalúa TODAS las columnas del DataFrame excepto las que se excluyen
    explícitamente (identificadores, target). El gate detecta leakage
    que las listas estáticas de build_features podrían no cubrir.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset completo (antes del split X/y).
    clase : "calor" | "frio"
        Modelo que se va a entrenar.
    target_col : str | None
        Columna objetivo. Si se pasa, se excluye del chequeo.
    exclude_cols : Sequence[str] | None
        Columnas a excluir del chequeo (identificadores como provincia,
        fecha, datetime).
    extra_blocked : Sequence[str] | None
        Columnas adicionales a bloquear que no están en las listas
        hardcodeadas (para configuración ad-hoc).

    Returns
    -------
    GateReport con el informe completo.
    """
    from climasafeai.features.build_features import COLS_TO_DROP

    # Columnas que el pipeline excluye como identificadores o por diseño
    # (no son features候选atas — el gate no las evalúa).
    # NOTA: NO excluimos LEAKAGE_COLS_BY_CLASE aquí porque justamente
    # el gate es quien debe detectar esas columnas como filtración.
    already_excluded = set(COLS_TO_DROP)

    if target_col:
        already_excluded.add(target_col)

    if exclude_cols:
        already_excluded.update(exclude_cols)

    # Columnas adicionales a bloquear (configuración ad-hoc): se pasan como
    # un set temporal sin mutar el module-level _DOWNSTREAM_CAUSAL_COLUMNS.
    blocked_extra = set(extra_blocked) if extra_blocked else set()

    # Features a evaluar: las que están en el DataFrame y no están excluidas
    features_to_check = [
        c for c in df.columns if c not in already_excluded
    ]

    verdicts = []
    for f in features_to_check:
        v = evaluate_feature(f)
        # Si la feature está en extra_blocked y no fue filtrada ya, añadir DOWNSTREAM
        if f in blocked_extra and v.passed:
            v.passed = False
            v.failed_criteria.append(LeakReason.DOWNSTREAM_CAUSAL)
        verdicts.append(v)

    return GateReport(
        clase=clase,
        all_features=features_to_check,
        verdicts=verdicts,
    )


def apply_feature_gate(
    df: pd.DataFrame,
    clase: str,
    target_col: str | None = None,
    exclude_cols: Sequence[str] | None = None,
    extra_blocked: Sequence[str] | None = None,
) -> tuple[pd.DataFrame, GateReport]:
    """
    Ejecuta el feature gate y devuelve el DataFrame sin las features filtradas.

    Útil para integración directa en el pipeline: ejecuta el gate, imprime
    el reporte y elimina las columnas que no pasan.

    Returns
    -------
    (df_filtered, report)
        df_filtered: DataFrame sin las features que fallaron el gate.
        report: informe completo del gate.
    """
    report = run_feature_gate(
        df, clase, target_col, exclude_cols, extra_blocked,
    )

    filtered_cols = {v.feature for v in report.filtered}
    if filtered_cols:
        df = df.drop(columns=[c for c in filtered_cols if c in df.columns])

    return df, report
