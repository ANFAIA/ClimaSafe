
"""
monitoring/monitor.py — Monitorización de drift y rendimiento del modelo.

Ejecutar con:
    make monitor

O directamente:
    uv run python -m monitoring.monitor

Qué detecta:
    - Data drift: cambio en la distribución de features entre referencia y producción
      - Kolmogorov-Smirnov para features numéricas
      - Chi-cuadrado para features categóricas
    - Performance drift: degradación de métricas respecto al baseline guardado

Genera:
    reports/monitoring/drift_report.csv    → resultados por feature
    reports/monitoring/drift_report.html   → informe visual navegable
    reports/monitoring/performance.csv     → métricas actuales vs baseline
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from climasafeai.utils.paths import (
    PROCESSED_DATA_DIR,
    ARTIFACTS_DIR,
    REPORTS_DIR,
)

MONITORING_DIR = REPORTS_DIR / "monitoring"

_METRICS = {
    "accuracy":  lambda y, p: accuracy_score(y, p),
    "f1":        lambda y, p: f1_score(y, p, average="weighted", zero_division=0),
    "precision": lambda y, p: precision_score(y, p, average="weighted", zero_division=0),
    "recall":    lambda y, p: recall_score(y, p, average="weighted", zero_division=0),
}

# p-value threshold para considerar drift significativo
DRIFT_THRESHOLD = 0.05

# ---------------------------------------------------------------------------
# Drift detection
# ---------------------------------------------------------------------------
def check_drift(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    threshold: float = DRIFT_THRESHOLD,
) -> pd.DataFrame:
    """
    Compara distribuciones entre datos de referencia (entrenamiento) y producción.

    Usa KS-test para numéricas y chi-cuadrado para categóricas.

    Parameters
    ----------
    reference  : DataFrame de referencia (X_train procesado)
    current    : DataFrame de producción (nuevos datos)
    threshold  : p-value por debajo del cual se declara drift (default: 0.05)

    Returns
    -------
    DataFrame con columnas: feature, test, statistic, p_value, drift_detected
    """
    results = []
    common_cols = [c for c in reference.columns if c in current.columns]

    for col in common_cols:
        ref_col = reference[col].dropna()
        cur_col = current[col].dropna()

        if len(ref_col) == 0 or len(cur_col) == 0:
            continue

        if pd.api.types.is_numeric_dtype(ref_col):
            # Kolmogorov-Smirnov: sensible a diferencias en forma y localización
            stat, pval = stats.ks_2samp(ref_col.values, cur_col.values)
            test_name = "KS"
        else:
            # Chi-cuadrado: compara frecuencias de categorías
            all_cats  = set(ref_col.unique()) | set(cur_col.unique())
            ref_freq  = ref_col.value_counts().reindex(all_cats, fill_value=0)
            cur_freq  = cur_col.value_counts().reindex(all_cats, fill_value=0)
            # Evitar categorías con frecuencia esperada 0
            mask = ref_freq > 0
            ref_freq, cur_freq = ref_freq[mask], cur_freq[mask]
            if len(ref_freq) < 2:
                continue
            stat, pval = stats.chisquare(cur_freq.values,
                                          f_exp=ref_freq.values / ref_freq.sum() * cur_freq.sum())
            test_name = "chi2"

        results.append({
            "feature":        col,
            "test":           test_name,
            "statistic":      round(float(stat), 4),
            "p_value":        round(float(pval), 4),
            "drift_detected": pval < threshold,
        })

    return pd.DataFrame(results)

# ---------------------------------------------------------------------------
# Performance monitoring
# ---------------------------------------------------------------------------

def check_performance(
    y_true,
    y_pred,
    baseline_path: Path = None,
) -> dict:
    """
    Calcula métricas actuales y las compara con el baseline guardado.

    Si no existe baseline, lo crea. En ejecuciones posteriores compara
    y emite alertas si alguna métrica empeora más de un 5%.

    Parameters
    ----------
    y_true        : etiquetas reales
    y_pred        : predicciones del modelo
    baseline_path : ruta al JSON de baseline (default: artifacts/baseline_metrics.json)

    Returns
    -------
    dict con métricas actuales y delta respecto al baseline
    """
    baseline_path = baseline_path or (ARTIFACTS_DIR / "baseline_metrics.json")
    current = {name: round(fn(y_true, y_pred), 4) for name, fn in _METRICS.items()}

    if baseline_path.exists():
        baseline = json.loads(baseline_path.read_text())
        delta = {
            name: round(current[name] - baseline.get(name, current[name]), 4)
            for name in current
        }
        print(f"\n  {'Métrica':<15} {'Actual':>10} {'Baseline':>10} {'Delta':>10}")
        print(f"  {'-'*47}")
        for name in current:
            d = delta[name]

            flag = " ⚠" if d < -0.05 else ""

            print(f"  {name:<15} {current[name]:>10.4f} {baseline.get(name, 'N/A'):>10} {d:>+10.4f}{flag}")
        result = {"current": current, "baseline": baseline, "delta": delta}
    else:
        print("\n  Baseline no encontrado. Guardando métricas actuales como baseline...")
        baseline_path.write_text(json.dumps(current, indent=2))
        print(f"  Baseline guardado en {baseline_path.name}")
        result = {"current": current, "baseline": None, "delta": None}

    return result

# ---------------------------------------------------------------------------
# Generación de informe HTML
# ---------------------------------------------------------------------------
def _build_html(drift_df: pd.DataFrame, perf: dict = None) -> str:
    """Genera un HTML minimalista con los resultados de monitorización."""
    drift_html = drift_df.to_html(index=False, border=0, classes="table")
    n_drift    = int(drift_df["drift_detected"].sum()) if "drift_detected" in drift_df else 0
    n_total    = len(drift_df)
    status     = "Sin drift significativo" if n_drift == 0 else f"Drift detectado en {n_drift}/{n_total} features"
    bg_color   = "#d5f5e3" if n_drift == 0 else "#fadbd8"


    if perf and perf.get("baseline"):
        rows = "".join(
            "<tr><td>" + k + "</td><td>" + str(round(perf["current"][k], 4)) + "</td>"
            "<td>" + str(perf["baseline"].get(k, "N/A")) + "</td>"
            "<td>" + str(round(perf["delta"][k], 4)) + "</td></tr>"
            for k in perf["current"]
        )
        perf_section = (
            "<h2>Rendimiento del modelo</h2>"
            "<table class=\"table\"><thead><tr>"
            "<th>Metrica</th><th>Actual</th><th>Baseline</th><th>Delta</th>"
            "</tr></thead><tbody>" + rows + "</tbody></table>"
        )
    else:
        perf_section = "<p>Sin baseline de rendimiento disponible.</p>"


    css = (
        "body{font-family:sans-serif;max-width:1000px;margin:40px auto;padding:0 20px}"
        "h1{color:#2c3e50}h2{color:#34495e;margin-top:30px}"
        ".status{font-size:1.2em;padding:10px 16px;border-radius:6px;"
        "display:inline-block;margin:10px 0;background:" + bg_color + "}"
        ".table{border-collapse:collapse;width:100%;margin-top:12px}"
        ".table td,.table th{border:1px solid #ddd;padding:8px 12px;text-align:left}"
        ".table th{background:#f5f6fa}"
        ".table tr:nth-child(even){background:#fafafa}"
    )

    return (
        "<!DOCTYPE html><html lang=\"es\"><head>"
        "<meta charset=\"UTF-8\">"
        "<title>Monitoring - ClimaSafeAI</title>"
        "<style>" + css + "</style>"
        "</head><body>"
        "<h1>Informe de monitorizacion</h1>"
        "<p><strong>Proyecto:</strong> ClimaSafeAI</p>"
        "<div class=\"status\">" + status + "</div>"
        "<h2>Drift por feature</h2>"
        + drift_html
        + perf_section
        + "</body></html>"
    )

# ---------------------------------------------------------------------------
# Punto de entrada principal
# ---------------------------------------------------------------------------
def run_monitoring(
    reference: pd.DataFrame = None,
    current: pd.DataFrame = None,

    y_true=None,
    y_pred=None,

    threshold: float = DRIFT_THRESHOLD,
    monitoring_dir: Path = None,
) -> None:
    """
    Ejecuta la monitorización completa y genera el informe.

    Si no se pasan DataFrames, carga automáticamente X_train (referencia)
    y X_test (producción simulada) desde data/processed/.

    Parameters
    ----------
    reference     : DataFrame de referencia (X_train)
    current       : DataFrame de producción (X_test o nuevos datos)
    threshold     : p-value para drift (default: 0.05)
    monitoring_dir: directorio de salida (default: reports/monitoring/)
    """
    _mon_dir = monitoring_dir or MONITORING_DIR
    _mon_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  Monitoring — ClimaSafeAI")
    print(f"{'='*60}")

    # Carga automática si no se pasan DataFrames
    if reference is None:
        ref_path = PROCESSED_DATA_DIR / "X_train.csv"
        if not ref_path.exists():
            print("  X_train.csv no encontrado. Ejecuta 'make features' primero.")
            return
        reference = pd.read_csv(ref_path)
        print(f"  Referencia: X_train.csv  {reference.shape}")

    if current is None:
        cur_path = PROCESSED_DATA_DIR / "X_test.csv"
        if not cur_path.exists():
            print("  X_test.csv no encontrado. Ejecuta 'make features' primero.")
            return
        current = pd.read_csv(cur_path)
        print(f"  Producción: X_test.csv   {current.shape}")

    # Drift detection
    print(f"\n  Analizando drift (threshold p < {threshold})...")
    drift_df = check_drift(reference, current, threshold=threshold)
    n_drift  = int(drift_df["drift_detected"].sum())
    print(f"  Features analizadas: {len(drift_df)}")
    print(f"  Features con drift:  {n_drift}")

    if n_drift > 0:
        print("\n  Features con drift detectado:")
        drifted = drift_df[drift_df["drift_detected"]]
        for _, row in drifted.iterrows():
            print(f"    ⚠  {row['feature']} ({row['test']}: p={row['p_value']:.4f})")

    drift_df.to_csv(_mon_dir / "drift_report.csv", index=False)
    print(f"\n  drift_report.csv guardado → {_mon_dir / 'drift_report.csv'}")

    # Performance monitoring
    perf = None
    if y_true is not None and y_pred is not None:
        print("\n  Comprobando rendimiento del modelo...")
        perf = check_performance(y_true, y_pred)
        perf_rows = [{"metric": k, "value": v} for k, v in perf["current"].items()]
        pd.DataFrame(perf_rows).to_csv(_mon_dir / "performance.csv", index=False)

    # HTML report
    html = _build_html(drift_df, perf)
    html_path = _mon_dir / "drift_report.html"
    html_path.write_text(html, encoding="utf-8")
    print(f"  drift_report.html guardado → {html_path}")
    print(f"\n{'='*60}\n")


if __name__ == "__main__":
    run_monitoring()

