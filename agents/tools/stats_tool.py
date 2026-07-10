"""
agents.tools.stats_tool — Tests estadísticos sobre arrays/DataFrames.

scipy está en las dependencias base del template, se importa directamente.
Cada función documenta exactamente qué hipótesis nula contrasta.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy import stats as _stats

from agents.tools.registry import register_tool


@register_tool("stats")
class StatsTool:
    @staticmethod
    def normal_test(data, method: str = "dagostino") -> dict[str, Any]:
        """
        Contrasta H₀: la muestra proviene de una distribución normal.

        method : 'dagostino' (D'Agostino-Pearson, potente para n>20)
                 'shapiro'   (Shapiro-Wilk, más preciso para n<50)
        """
        data = np.asarray(data, dtype=float)
        if method == "shapiro":
            stat, p = _stats.shapiro(data)
        else:
            stat, p = _stats.normaltest(data)
        return {"statistic": float(stat), "pvalue": float(p), "normal": p > 0.05, "method": method}

    @staticmethod
    def ttest_ind(a, b, equal_var: bool = True) -> dict[str, Any]:
        """H₀: medias de dos muestras independientes iguales (t de Student)."""
        a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
        stat, p = _stats.ttest_ind(a, b, equal_var=equal_var)
        return {"statistic": float(stat), "pvalue": float(p), "significant": p < 0.05}

    @staticmethod
    def ttest_paired(a, b) -> dict[str, Any]:
        """H₀: medias de dos muestras apareadas iguales."""
        a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
        stat, p = _stats.ttest_rel(a, b)
        return {"statistic": float(stat), "pvalue": float(p), "significant": p < 0.05}

    @staticmethod
    def mannwhitney(a, b) -> dict[str, Any]:
        """H₀: distribuciones de dos muestras iguales (U de Mann-Whitney, no paramétrico)."""
        a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
        stat, p = _stats.mannwhitneyu(a, b, alternative="two-sided")
        return {"statistic": float(stat), "pvalue": float(p), "significant": p < 0.05}

    @staticmethod
    def chisquare(observed, expected=None) -> dict[str, Any]:
        """H₀: frecuencias observadas = frecuencias esperadas (χ² bondad de ajuste)."""
        observed = np.asarray(observed, dtype=float)
        if expected is not None:
            expected = np.asarray(expected, dtype=float)
        stat, p = _stats.chisquare(observed, f_exp=expected)
        return {"statistic": float(stat), "pvalue": float(p), "significant": p < 0.05}

    @staticmethod
    def chisquare_independence(contingency_table) -> dict[str, Any]:
        """H₀: dos variables categóricas son independientes (χ² sobre tabla de contingencia)."""
        table = np.asarray(contingency_table, dtype=float)
        stat, p, dof, expected = _stats.chi2_contingency(table)
        return {"statistic": float(stat), "pvalue": float(p), "dof": int(dof), "significant": p < 0.05}

    @staticmethod
    def anova(*groups) -> dict[str, Any]:
        """H₀: medias de k grupos iguales (ANOVA unidireccional)."""
        groups = [np.asarray(g, dtype=float) for g in groups]
        stat, p = _stats.f_oneway(*groups)
        return {"statistic": float(stat), "pvalue": float(p), "significant": p < 0.05}

    @staticmethod
    def kruskal(*groups) -> dict[str, Any]:
        """H₀: distribuciones de k grupos iguales (Kruskal-Wallis, no paramétrico)."""
        groups = [np.asarray(g, dtype=float) for g in groups]
        stat, p = _stats.kruskal(*groups)
        return {"statistic": float(stat), "pvalue": float(p), "significant": p < 0.05}

    @staticmethod
    def correlation(x, y, method: str = "pearson") -> dict[str, Any]:
        """
        Correlación entre dos variables con p-valor.

        method : 'pearson'  (relación lineal, supone normalidad)
                 'spearman' (monótona, no paramétrica)
                 'kendall'  (tau de Kendall, robusta a outliers)
        """
        x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
        valid = ~(np.isnan(x) | np.isnan(y))
        x, y = x[valid], y[valid]
        if method == "pearson":
            stat, p = _stats.pearsonr(x, y)
        elif method == "spearman":
            stat, p = _stats.spearmanr(x, y)
        elif method == "kendall":
            stat, p = _stats.kendalltau(x, y)
        else:
            raise ValueError(f"method debe ser 'pearson', 'spearman' o 'kendall', no '{method}'")
        return {"statistic": float(stat), "pvalue": float(p), "method": method, "n": len(x)}

    @staticmethod
    def effect_size(a, b) -> dict[str, Any]:
        """Cohen's d: diferencia estandarizada entre dos muestras independientes."""
        a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
        n1, n2 = len(a), len(b)
        s1, s2 = np.var(a, ddof=1), np.var(b, ddof=1)
        pooled = np.sqrt(((n1 - 1) * s1 + (n2 - 1) * s2) / (n1 + n2 - 2))
        d = (np.mean(a) - np.mean(b)) / pooled if pooled > 0 else 0.0
        interpretation = "pequeño" if abs(d) < 0.2 else "medio" if abs(d) < 0.5 else "grande" if abs(d) < 0.8 else "muy grande"
        return {"d": float(d), "interpretation": interpretation}
