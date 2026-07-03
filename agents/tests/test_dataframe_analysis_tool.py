from __future__ import annotations

import numpy as np
import pandas as pd

from agents.tools.dataframe_analysis_tool import DataFrameAnalysisTool


def test_constant_columns_detected():
    df = pd.DataFrame({"a": [1, 1, 1], "b": [1, 2, 3]})
    findings = DataFrameAnalysisTool.constant_columns(df)
    assert [f.column for f in findings] == ["a"]


def test_high_cardinality_columns_detected():
    df = pd.DataFrame({"id": [f"user_{i}" for i in range(100)], "cat": ["A", "B"] * 50})
    findings = DataFrameAnalysisTool.high_cardinality_columns(df, threshold_ratio=0.5)
    assert "id" in [f.column for f in findings]
    assert "cat" not in [f.column for f in findings]


def test_outliers_iqr_detected():
    import numpy as np
    rng = np.random.default_rng(0)
    values = list(rng.normal(loc=10, scale=1, size=50)) + [1000]  # un outlier evidente
    df = pd.DataFrame({"x": values})
    findings = DataFrameAnalysisTool.outliers_iqr(df)
    assert len(findings) == 1
    assert findings[0].column == "x"


def test_leakage_suspects_high_correlation():
    rng = np.random.default_rng(42)
    target = rng.normal(size=200)
    df = pd.DataFrame({
        "target": target,
        "leaky": target * 2 + 0.0001,  # prácticamente el target reescalado
        "normal_feature": rng.normal(size=200),
    })
    findings = DataFrameAnalysisTool.leakage_suspects(df, "target", correlation_threshold=0.95)
    assert [f.column for f in findings] == ["leaky"]


def test_leakage_suspects_missing_target_returns_empty():
    df = pd.DataFrame({"a": [1, 2, 3]})
    assert DataFrameAnalysisTool.leakage_suspects(df, "no_existe") == []
