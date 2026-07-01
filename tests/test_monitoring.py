
"""
test_monitoring.py — Tests del modulo de monitorizacion.
"""
import json
import numpy as np
import pandas as pd
import pytest


def _make_ref_curr(n=200, n_feat=4, with_drift=False):
    np.random.seed(42)
    ref = pd.DataFrame(
        np.random.randn(n, n_feat),
        columns=[f"feat_{i}" for i in range(n_feat)],
    )
    if with_drift:
        # Shift la media de feat_0 para simular drift real
        curr = ref.copy()
        curr["feat_0"] = curr["feat_0"] + 5.0
    else:
        curr = pd.DataFrame(
            np.random.randn(n // 2, n_feat),
            columns=[f"feat_{i}" for i in range(n_feat)],
        )
    return ref, curr


# ---------------------------------------------------------------------------
# Tests de check_drift
# ---------------------------------------------------------------------------
def test_check_drift_sin_drift(patch_paths):
    """Distribuciones similares no deben mostrar drift."""
    from monitoring.monitor import check_drift
    ref, curr = _make_ref_curr(with_drift=False)
    result = check_drift(ref, curr)
    assert isinstance(result, pd.DataFrame)
    assert "feature" in result.columns
    assert "drift_detected" in result.columns
    assert len(result) == 4


def test_check_drift_con_drift(patch_paths):
    """Distribución con shift debe detectar drift en feat_0."""
    from monitoring.monitor import check_drift
    ref, curr = _make_ref_curr(with_drift=True)
    result = check_drift(ref, curr)
    drifted = result[result["drift_detected"]]
    assert "feat_0" in drifted["feature"].values


def test_check_drift_devuelve_columnas_correctas(patch_paths):
    """El DataFrame de drift debe tener todas las columnas esperadas."""
    from monitoring.monitor import check_drift
    ref, curr = _make_ref_curr()
    result = check_drift(ref, curr)
    for col in ["feature", "test", "statistic", "p_value", "drift_detected"]:
        assert col in result.columns


def test_check_drift_threshold_bajo_detecta_mas(patch_paths):
    """Threshold bajo debe detectar más drift que threshold alto."""
    from monitoring.monitor import check_drift
    ref, curr = _make_ref_curr(with_drift=False)
    strict = check_drift(ref, curr, threshold=0.5)
    lenient = check_drift(ref, curr, threshold=0.001)
    assert strict["drift_detected"].sum() >= lenient["drift_detected"].sum()


def test_check_drift_categorica(patch_paths):
    """check_drift debe usar chi2 para features categóricas."""
    from monitoring.monitor import check_drift
    np.random.seed(0)
    ref  = pd.DataFrame({"cat": np.random.choice(["A", "B", "C"], 100)})
    curr = pd.DataFrame({"cat": np.random.choice(["A", "B", "C"], 50)})
    result = check_drift(ref, curr)
    assert result.iloc[0]["test"] == "chi2"


# ---------------------------------------------------------------------------
# Tests de run_monitoring
# ---------------------------------------------------------------------------
def test_run_monitoring_guarda_csv(patch_paths, tmp_path):
    """run_monitoring debe guardar drift_report.csv."""
    from monitoring.monitor import run_monitoring
    ref, curr = _make_ref_curr()
    mon_dir = tmp_path / "monitoring"
    run_monitoring(reference=ref, current=curr, monitoring_dir=mon_dir)
    assert (mon_dir / "drift_report.csv").exists()


def test_run_monitoring_guarda_html(patch_paths, tmp_path):
    """run_monitoring debe guardar drift_report.html."""
    from monitoring.monitor import run_monitoring
    ref, curr = _make_ref_curr()
    mon_dir = tmp_path / "monitoring"
    run_monitoring(reference=ref, current=curr, monitoring_dir=mon_dir)
    html_path = mon_dir / "drift_report.html"
    assert html_path.exists()
    content = html_path.read_text()
    assert "ClimaSafeAI" in content
    assert "drift" in content.lower()


def test_run_monitoring_csv_columnas_correctas(patch_paths, tmp_path):
    """El CSV de drift debe tener las columnas esperadas."""
    from monitoring.monitor import run_monitoring
    ref, curr = _make_ref_curr()
    mon_dir = tmp_path / "monitoring"
    run_monitoring(reference=ref, current=curr, monitoring_dir=mon_dir)
    df = pd.read_csv(mon_dir / "drift_report.csv")
    for col in ["feature", "test", "statistic", "p_value", "drift_detected"]:
        assert col in df.columns



# ---------------------------------------------------------------------------
# Tests de check_performance
# ---------------------------------------------------------------------------
def test_check_performance_crea_baseline(patch_paths, tmp_path):
    """Sin baseline existente, check_performance debe crear uno."""
    from monitoring.monitor import check_performance
    np.random.seed(42)

    y_true = np.array([0, 1, 0, 1, 0, 1, 1, 0])
    y_pred = np.array([0, 1, 0, 1, 1, 1, 0, 0])

    baseline_path = tmp_path / "baseline_metrics.json"
    result = check_performance(y_true, y_pred, baseline_path=baseline_path)
    assert baseline_path.exists()
    assert "current" in result
    assert result["baseline"] is None


def test_check_performance_compara_con_baseline(patch_paths, tmp_path):
    """Con baseline existente, check_performance debe comparar métricas."""
    from monitoring.monitor import check_performance

    y_true = np.array([0, 1, 0, 1, 0, 1, 1, 0])
    y_pred = np.array([0, 1, 0, 1, 1, 1, 0, 0])
    baseline = {"accuracy": 0.75, "f1": 0.74, "precision": 0.75, "recall": 0.75}

    baseline_path = tmp_path / "baseline_metrics.json"
    baseline_path.write_text(json.dumps(baseline))
    result = check_performance(y_true, y_pred, baseline_path=baseline_path)
    assert result["baseline"] is not None
    assert result["delta"] is not None
    assert set(result["current"].keys()) == set(baseline.keys())


def test_run_monitoring_con_predicciones(patch_paths, tmp_path):
    """run_monitoring con y_true e y_pred guarda performance.csv."""
    from monitoring.monitor import run_monitoring
    ref, curr = _make_ref_curr()

    y_true = np.random.randint(0, 2, 50)
    y_pred = np.random.randint(0, 2, 50)

    mon_dir = tmp_path / "monitoring"
    run_monitoring(
        reference=ref, current=curr,
        y_true=y_true, y_pred=y_pred,
        monitoring_dir=mon_dir,
    )
    assert (mon_dir / "performance.csv").exists()


