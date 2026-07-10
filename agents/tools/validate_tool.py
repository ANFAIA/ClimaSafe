"""
agents.tools.validate_tool — Validación y perfilado de datos.

Todo determinista, sobre pandas/numpy/scipy (dependencias base del template).
Sin heurísticas mágicas: cada función documenta exactamente qué calcula.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy import stats as _stats

from agents.tools.registry import register_tool


@register_tool("validate")
class ValidateTool:
    @staticmethod
    def profile(df: pd.DataFrame) -> dict[str, Any]:
        """Perfil completo del dataset: métricas por columna + resumen."""
        rows, cols = df.shape
        dups = int(df.duplicated().sum())
        profile_data = {}
        for col in df.columns:
            info: dict[str, Any] = {}
            info["dtype"] = str(df[col].dtype)
            info["nulls"] = int(df[col].isnull().sum())
            info["null_pct"] = round(float(df[col].isnull().mean()), 4)
            info["unique"] = int(df[col].nunique())
            if df[col].dtype.kind in "ifc":
                info["mean"] = float(df[col].mean()) if not df[col].isnull().all() else None
                info["std"] = float(df[col].std()) if not df[col].isnull().all() else None
                info["min"] = float(df[col].min()) if not df[col].isnull().all() else None
                info["max"] = float(df[col].max()) if not df[col].isnull().all() else None
            profile_data[col] = info
        return {
            "rows": rows, "cols": cols, "duplicates": dups,
            "duplicate_pct": round(dups / rows, 4) if rows else 0,
            "columns": profile_data,
        }

    @staticmethod
    def check_schema(df: pd.DataFrame, expected: dict[str, str]) -> list[dict[str, Any]]:
        """
        Valida que el DataFrame tenga las columnas y tipos esperados.

        expected : {nombre_columna: dtype_string}, e.g. {"age": "int64", "name": "object"}
        Devuelve lista de errores (vacía si todo ok).
        """
        errors = []
        for col, dtype in expected.items():
            if col not in df.columns:
                errors.append({"column": col, "error": "missing", "expected": dtype})
                continue
            actual = str(df[col].dtype)
            if actual != dtype:
                errors.append({"column": col, "error": "type_mismatch", "expected": dtype, "actual": actual})
        return errors

    @staticmethod
    def detect_drift(reference: pd.DataFrame, current: pd.DataFrame, threshold: float = 0.05) -> dict[str, Any]:
        """
        Compara distribuciones entre dos datasets (referencia vs actual).

        Para numéricas: KS test (H₀: mismas distribución).
        Para categóricas: χ² test (H₀: mismas proporciones).
        threshold: p-valor mínimo para considerar drift significativo.
        """
        report: dict[str, Any] = {"drifted_columns": [], "total_columns": 0, "details": {}}
        common = [c for c in reference.columns if c in current.columns]
        report["total_columns"] = len(common)
        for col in common:
            ref = reference[col].dropna()
            cur = current[col].dropna()
            if len(ref) < 5 or len(cur) < 5:
                continue
            entry: dict[str, Any] = {"ref_n": len(ref), "cur_n": len(cur)}
            if ref.dtype.kind in "ifc":
                stat, p = _stats.ks_2samp(ref, cur)
                entry["test"] = "ks"
                entry["statistic"] = float(stat)
                entry["pvalue"] = float(p)
            else:
                ref_counts = ref.value_counts(normalize=True)
                cur_counts = cur.value_counts(normalize=True)
                all_cats = list(set(ref_counts.index) | set(cur_counts.index))
                ref_props = [ref_counts.get(c, 0) for c in all_cats]
                cur_props = [cur_counts.get(c, 0) for c in all_cats]
                if len(all_cats) < 2:
                    continue
                _, p = _stats.chisquare(cur_props, f_exp=ref_props)
                entry["test"] = "chisquare"
                entry["pvalue"] = float(p)
            entry["drift"] = entry["pvalue"] < threshold
            report["details"][col] = entry
            if entry["drift"]:
                report["drifted_columns"].append(col)
        return report

    @staticmethod
    def find_outliers(data, method: str = "iqr", factor: float = 1.5) -> dict[str, Any]:
        """
        Detecta outliers en datos numéricos.

        method : 'iqr'    → Q1 - factor*IQR / Q3 + factor*IQR
                 'zscore' → |z-score| > factor (factor actúa como umbral z)
        """
        data = np.asarray(data, dtype=float)
        mask = ~np.isnan(data)
        clean = data[mask]
        n_total = len(data)
        if method == "iqr":
            q1, q3 = np.percentile(clean, [25, 75])
            iqr = q3 - q1
            lower = q1 - factor * iqr
            upper = q3 + factor * iqr
            outlier_mask = (clean < lower) | (clean > upper)
        elif method == "zscore":
            mean, std = np.mean(clean), np.std(clean)
            if std == 0:
                return {"count": 0, "fraction": 0.0, "indices": [], "method": method}
            z = np.abs((clean - mean) / std)
            outlier_mask = z > factor
        else:
            raise ValueError(f"method debe ser 'iqr' o 'zscore', no '{method}'")
        outlier_indices = np.where(mask)[0][outlier_mask].tolist()
        return {
            "count": int(outlier_mask.sum()),
            "fraction": round(float(outlier_mask.sum() / n_total), 4),
            "indices": outlier_indices,
            "method": method,
        }

    @staticmethod
    def check_data_quality(df: pd.DataFrame) -> dict[str, Any]:
        """Reporte de calidad: nulos, duplicados, constantes, cardinalidad."""
        report: dict[str, Any] = {}
        for col in df.columns:
            info: dict[str, Any] = {}
            info["null_pct"] = round(float(df[col].isnull().mean()), 4)
            info["dtype"] = str(df[col].dtype)
            n_unique = int(df[col].nunique())
            info["unique"] = n_unique
            info["cardinality_pct"] = round(n_unique / len(df), 4) if len(df) else 0
            info["constant"] = n_unique <= 1
            if df[col].dtype.kind in "ifc":
                info["zeros_pct"] = round(float((df[col] == 0).mean()), 4)
                info["negatives_pct"] = round(float((df[col] < 0).mean()), 4)
            else:
                info["zeros_pct"] = None
                info["negatives_pct"] = None
            report[col] = info
        dup_pct = round(float(df.duplicated().mean()), 4)
        return {"columns": report, "duplicate_rows_pct": dup_pct}
