"""
agents.tools.dataframe_analysis_tool — Heurísticas de EDA reutilizables.

Todo aquí es determinista y basado en pandas/numpy/scipy (dependencias base
del template). No hay heurística mágica: cada función documenta exactamente
qué criterio usa, para que puedas ajustar el umbral si tu dataset lo necesita.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from agents.tools.registry import register_tool


@dataclass
class ColumnFinding:
    column: str
    kind: str  # "constant" | "high_cardinality" | "outliers" | "leakage_suspect" | "high_missing"
    detail: str
    severity: str = "warning"  # "warning" | "info"


@register_tool("dataframe_analysis")
class DataFrameAnalysisTool:
    @staticmethod
    def constant_columns(df: pd.DataFrame) -> list[ColumnFinding]:
        """Columnas con un único valor distinto (ignorando NaN) — no aportan información."""
        findings = []
        for col in df.columns:
            nunique = df[col].nunique(dropna=True)
            if nunique <= 1:
                findings.append(
                    ColumnFinding(col, "constant", f"{nunique} valor(es) único(s) — descártala.")
                )
        return findings

    @staticmethod
    def high_cardinality_columns(
        df: pd.DataFrame, *, categorical_only: bool = True, threshold_ratio: float = 0.5
    ) -> list[ColumnFinding]:
        """
        Columnas categóricas donde nunique/nrows supera `threshold_ratio`
        (por defecto 0.5): sugiere un identificador o un campo de texto libre,
        mal candidato para one-hot encoding directo.
        """
        findings = []
        n = len(df)
        if n == 0:
            return findings
        # Nota: en pandas 3.x esto genera un FutureWarning sugiriendo incluir
        # "str" explícitamente junto a "object"/"category". No lo añado aquí:
        # no pude verificar si "str" como token de select_dtypes se comporta
        # igual en pandas 2.x (que algún proyecto ya generado podría tener
        # fijado en su lockfile), y una incompatibilidad real es peor que un
        # warning cosmético. Si te molesta el warning y sabes que tu proyecto
        # usa pandas 3.x, añade "str" tú mismo a esta lista.
        cols = df.select_dtypes(include=["object", "category"]).columns if categorical_only else df.columns
        for col in cols:
            ratio = df[col].nunique(dropna=True) / n
            if ratio > threshold_ratio:
                findings.append(
                    ColumnFinding(
                        col, "high_cardinality",
                        f"{df[col].nunique()} valores únicos de {n} filas ({ratio:.0%}). "
                        f"Considera target/frequency encoding en vez de one-hot, "
                        f"o comprueba si es un ID que deberías excluir."
                    )
                )
        return findings

    @staticmethod
    def high_missing_columns(df: pd.DataFrame, *, threshold_ratio: float = 0.3) -> list[ColumnFinding]:
        findings = []
        n = len(df)
        if n == 0:
            return findings
        missing_ratio = df.isna().mean()
        for col, ratio in missing_ratio.items():
            if ratio > threshold_ratio:
                findings.append(
                    ColumnFinding(col, "high_missing", f"{ratio:.0%} de valores nulos.")
                )
        return findings

    @staticmethod
    def outliers_iqr(df: pd.DataFrame, *, iqr_factor: float = 1.5) -> list[ColumnFinding]:
        """Detecta outliers por rango intercuartílico (Tukey) en columnas numéricas."""
        findings = []
        numeric = df.select_dtypes(include=[np.number])
        for col in numeric.columns:
            series = numeric[col].dropna()
            if series.empty:
                continue
            q1, q3 = series.quantile(0.25), series.quantile(0.75)
            iqr = q3 - q1
            if iqr == 0:
                continue
            lower, upper = q1 - iqr_factor * iqr, q3 + iqr_factor * iqr
            n_outliers = ((series < lower) | (series > upper)).sum()
            if n_outliers > 0:
                pct = n_outliers / len(series)
                findings.append(
                    ColumnFinding(
                        col, "outliers",
                        f"{n_outliers} outliers ({pct:.1%}) fuera de [{lower:.3g}, {upper:.3g}] "
                        f"(IQR × {iqr_factor})."
                    )
                )
        return findings

    @staticmethod
    def leakage_suspects(
        df: pd.DataFrame, target_col: str, *, correlation_threshold: float = 0.95
    ) -> list[ColumnFinding]:
        """
        Señala columnas numéricas con correlación de Pearson absoluta muy alta
        con el target. Una correlación así de alta suele significar que la
        columna se calculó A PARTIR del target (fuga de información) — pero
        también puede ser una relación real y fuerte, así que esto es una
        señal para revisar manualmente, no una certeza.
        """
        if target_col not in df.columns:
            return []
        numeric = df.select_dtypes(include=[np.number])
        if target_col not in numeric.columns:
            return []
        findings = []
        correlations = numeric.corr(numeric_only=True)[target_col].drop(target_col, errors="ignore")
        for col, corr in correlations.items():
            if pd.notna(corr) and abs(corr) >= correlation_threshold:
                findings.append(
                    ColumnFinding(
                        col, "leakage_suspect",
                        f"correlación {corr:.3f} con '{target_col}' — revisa si '{col}' "
                        f"se calculó usando el target (posible fuga de información).",
                        severity="warning",
                    )
                )
        return findings

    @staticmethod
    def correlation_matrix(df: pd.DataFrame) -> pd.DataFrame:
        return df.select_dtypes(include=[np.number]).corr(numeric_only=True)

    @staticmethod
    def highly_correlated_pairs(df: pd.DataFrame, *, threshold: float = 0.9) -> list[tuple[str, str, float]]:
        """Pares de features numéricas muy correlacionadas entre sí (candidatas a redundantes)."""
        corr = DataFrameAnalysisTool.correlation_matrix(df)
        pairs = []
        cols = corr.columns
        for i, col_a in enumerate(cols):
            for col_b in cols[i + 1:]:
                value = corr.loc[col_a, col_b]
                if pd.notna(value) and abs(value) >= threshold:
                    pairs.append((col_a, col_b, float(value)))
        return sorted(pairs, key=lambda t: -abs(t[2]))

    @staticmethod
    def summary(df: pd.DataFrame) -> dict:
        return {
            "shape": df.shape,
            "columns": list(df.columns),
            "dtypes": {c: str(t) for c, t in df.dtypes.items()},
            "memory_usage_mb": round(df.memory_usage(deep=True).sum() / 1_048_576, 3),
            "n_duplicated_rows": int(df.duplicated().sum()),
        }
