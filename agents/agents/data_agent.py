"""
agents.agents.data_agent — Análisis de calidad de datos para este template.

Conoce dónde vive cada etapa del pipeline de datos:
`data/raw/` (crudo) -> `data/processed/` (`{{ project_slug }}/data/make_dataset.py`)
-> `data/interim/` (`{{ project_slug }}/features/build_features.py`). Por
defecto analiza `data/raw/`, que es donde tiene más sentido detectar
problemas antes de que se propaguen al resto del pipeline.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from agents.core.base_agent import AgentResult, BaseAgent
from agents.core.registry import register_agent
from agents.tools.data_io_tool import DataIOTool
from agents.tools.dataframe_analysis_tool import DataFrameAnalysisTool


@register_agent
class DataAgent(BaseAgent):
    name = "data"
    description = (
        "Analiza datasets: columnas constantes, cardinalidad, outliers, fuga de "
        "información, correlaciones, y recomienda limpieza / feature engineering."
    )
    capabilities = [
        "dataset", "datos", "eda", "outlier", "outliers", "cardinalidad",
        "fuga de informacion", "leakage", "correlacion", "csv", "parquet",
        "limpieza", "features",
    ]

    def actions(self) -> dict:
        return {
            "list_datasets": self.list_datasets,
            "eda_report": self.eda_report,
            "detect_leakage": self.detect_leakage,
        }

    # -------------------------------------------------------------------------
    def _load(self, path: Path) -> pd.DataFrame:
        reader = DataIOTool.infer_reader(path)
        result = reader(path)
        if not isinstance(result, pd.DataFrame):
            raise TypeError(
                f"'{path.name}' no se pudo cargar como tabla (¿es un .json con estructura no tabular?)."
            )
        return result

    def list_datasets(self) -> AgentResult:
        candidates = []
        for stage_dir in (self.ctx.raw_data_dir, self.ctx.interim_data_dir, self.ctx.processed_data_dir):
            if stage_dir.exists():
                candidates.extend(
                    p for p in stage_dir.iterdir()
                    if p.is_file() and p.suffix.lower() in (".csv", ".parquet", ".json")
                )
        return AgentResult(
            True, self.name, "list_datasets",
            f"{len(candidates)} archivo(s) de datos encontrado(s).",
            data=[str(p.relative_to(self.ctx.root)) for p in sorted(candidates)],
        )

    def eda_report(self, *, filename: str, target_col: str | None = None) -> AgentResult:
        """
        Genera un informe EDA sobre `data/raw/<filename>` (o una ruta relativa
        a la raíz del proyecto si `filename` no está en `data/raw/`).
        """
        path = self.ctx.raw_data_dir / filename
        if not path.exists():
            path = self.ctx.root / filename
        if not path.exists():
            return AgentResult(False, self.name, "eda_report", f"No se encontró el archivo '{filename}'.")

        try:
            df = self._load(path)
        except (ValueError, TypeError) as exc:
            return AgentResult(False, self.name, "eda_report", str(exc))

        tool = DataFrameAnalysisTool
        report = {
            "summary": tool.summary(df),
            "constant_columns": [f.__dict__ for f in tool.constant_columns(df)],
            "high_cardinality_columns": [f.__dict__ for f in tool.high_cardinality_columns(df)],
            "high_missing_columns": [f.__dict__ for f in tool.high_missing_columns(df)],
            "outliers": [f.__dict__ for f in tool.outliers_iqr(df)],
            "highly_correlated_pairs": tool.highly_correlated_pairs(df),
        }
        warnings = []
        if target_col:
            leakage = tool.leakage_suspects(df, target_col)
            report["leakage_suspects"] = [f.__dict__ for f in leakage]
            if leakage:
                warnings.append(f"{len(leakage)} columna(s) sospechosa(s) de fuga de información con '{target_col}'.")

        n_issues = sum(
            len(report[k]) for k in
            ("constant_columns", "high_cardinality_columns", "high_missing_columns", "outliers")
        )
        return AgentResult(
            True, self.name, "eda_report",
            f"EDA de '{filename}' completo: {df.shape[0]} filas × {df.shape[1]} columnas, {n_issues} hallazgo(s).",
            data=report, warnings=warnings,
        )

    def detect_leakage(self, *, filename: str, target_col: str, correlation_threshold: float = 0.95) -> AgentResult:
        path = self.ctx.raw_data_dir / filename
        if not path.exists():
            path = self.ctx.root / filename
        if not path.exists():
            return AgentResult(False, self.name, "detect_leakage", f"No se encontró el archivo '{filename}'.")

        try:
            df = self._load(path)
        except (ValueError, TypeError) as exc:
            return AgentResult(False, self.name, "detect_leakage", str(exc))

        suspects = DataFrameAnalysisTool.leakage_suspects(df, target_col, correlation_threshold=correlation_threshold)
        return AgentResult(
            True, self.name, "detect_leakage",
            f"{len(suspects)} columna(s) sospechosa(s) de fuga de información.",
            data=[f.__dict__ for f in suspects],
            warnings=["Correlación alta no es prueba de fuga — es una señal para revisar manualmente."] if suspects else [],
        )
