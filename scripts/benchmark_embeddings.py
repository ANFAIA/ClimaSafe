#!/usr/bin/env python3
"""
benchmark_embeddings.py — Benchmark de embeddings geoespaciales (ML-003).

Compara el rendimiento del modelo XGBoost CON y SIN embeddings geoespaciales
sobre el mismo split temporal que usa el pipeline de entrenamiento existente.

Métricas:
  - F1_weighted (classification, la métrica principal del proyecto)
  - Brier score (proper scoring rule, análogo a R² para clasificación)
  - Distribución de clases predicha
  - Tamaño medio del prediction set conformal

Uso:
    uv run python scripts/benchmark_embeddings.py
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    f1_score,
    brier_score_loss,
    classification_report,
)
from sklearn.preprocessing import LabelEncoder

# Añadir raíz al path para imports del proyecto
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from climasafeai.features.geospatial_embeddings import (
    CensusDemographicsEmbedding,
    SpatialCoordinateEmbedding,
    merge_embeddings,
)
from climasafeai.features.build_features import (
    preprocess_data,
    COLS_TO_DROP,
    LEAKAGE_COLS_BY_CLASE,
    COLS_TO_DROP_BY_CLASE,
)
from climasafeai.models.conformal import SplitConformalCalibrator

warnings.filterwarnings("ignore", category=FutureWarning)

CLASE = "calor"
TARGET = "clase_riesgo_calor"
DATASET_PATH = ROOT / "data" / "processed" / "dataset_calor_labeled.parquet"


def load_dataset() -> pd.DataFrame:
    """Carga el dataset labeled."""
    if not DATASET_PATH.exists():
        print(f"ERROR: {DATASET_PATH} no existe. Ejecuta make init primero.")
        sys.exit(1)
    df = pd.read_parquet(DATASET_PATH)
    print(f"Dataset cargado: {df.shape[0]} filas, {df.shape[1]} columnas")
    print(f"Provincias: {df['provincia'].nunique()}")
    return df


def add_embeddings_to_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Añade embeddings geoespaciales al dataset."""
    providers = [
        CensusDemographicsEmbedding(),
        SpatialCoordinateEmbedding(),
    ]
    df_emb = merge_embeddings(df, providers)
    n_new = sum(p.n_features for p in providers)
    print(f"Añadidas {n_new} columnas de embedding ({', '.join(p.name for p in providers)})")
    return df_emb


def prepare_features(
    df: pd.DataFrame,
    with_embeddings: bool = True,
) -> tuple[pd.DataFrame, pd.Series]:
    """Prepara features usando preprocess_data (mismo pipeline que el entrenamiento)."""
    df_work = df.copy()

    if not with_embeddings:
        # Eliminar columnas de embedding si existieran
        emb_cols = [c for c in df_work.columns if c.startswith("census_") or c.startswith("spatial_")]
        if emb_cols:
            df_work.drop(columns=emb_cols, inplace=True)

    # Usar preprocess_data del pipeline existente (misma lógica de split, escalado, etc.)
    X_train, X_test, y_train, y_test = preprocess_data(
        df_work,
        target_col=TARGET,
        scaler_type="standard",
        test_size=0.2,
        clase=CLASE,
        split_by_date=True,
    )
    return X_train, X_test, y_train, y_test


def train_and_evaluate(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    feature_names: list[str],
    label: str,
) -> dict:
    """Entrena XGBoost y evalúa. Devuelve métricas."""
    from xgboost import XGBClassifier
    from sklearn.utils.class_weight import compute_sample_weight

    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")

    # Entrenar
    sample_weight = compute_sample_weight("balanced", y_train)
    model = XGBClassifier(
        n_estimators=300, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        reg_alpha=0.1, reg_lambda=1.0,
        eval_metric="logloss", random_state=42, n_jobs=-1,
    )
    model.fit(X_train, y_train, sample_weight=sample_weight)

    # Predicciones
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)

    # Métricas
    f1 = f1_score(y_test, y_pred, average="weighted")

    # Brier score (promedio sobre las 3 clases, una-vs-rest)
    n_classes = y_proba.shape[1]
    brier_scores = []
    for c in range(n_classes):
        y_binary = (y_test == c).astype(int)
        brier_scores.append(brier_score_loss(y_binary, y_proba[:, c]))
    brier = np.mean(brier_scores)

    # Conformal prediction
    conformal = SplitConformalCalibrator(alpha=0.1)
    # Usar la mitad del test como calibration set
    n_cal = len(y_test) // 2
    conformal.fit(y_proba[:n_cal], y_test.values[:n_cal] if hasattr(y_test, 'values') else y_test[:n_cal])
    _, set_sizes = conformal.confidence(y_proba[n_cal:])
    mean_set_size = float(np.mean(set_sizes))

    # Distribución de clases predicha
    pred_dist = pd.Series(y_pred).value_counts(normalize=True).to_dict()

    # Feature importance (top 5)
    importances = model.feature_importances_
    top_idx = np.argsort(importances)[::-1][:5]
    top_features = [(feature_names[i], float(importances[i])) for i in top_idx]

    results = {
        "f1_weighted": f1,
        "brier_score": brier,
        "conformal_mean_set_size": mean_set_size,
        "pred_distribution": pred_dist,
        "top_features": top_features,
    }

    print(f"  F1-weighted:        {f1:.4f}")
    print(f"  Brier score:        {brier:.4f} (menor es mejor)")
    print(f"  Conformal set size: {mean_set_size:.3f} (1=alta, 2=media, 3=baja confianza)")
    print(f"  Distribución pred:  {pred_dist}")
    print(f"  Top 5 features:")
    for name, imp in top_features:
        print(f"    {name:40s} {imp:.4f}")

    return results


def print_comparison(baseline: dict, with_embeddings: dict) -> None:
    """Imprime tabla comparativa."""
    print(f"\n{'='*70}")
    print("  COMPARACIÓN: BASELINE vs EMBEDDINGS GEOESPACIALES")
    print(f"{'='*70}")
    print(f"  {'Métrica':<30s} {'Baseline':>12s} {'Embeddings':>12s} {'Δ':>10s}")
    print(f"  {'-'*64}")

    for metric in ["f1_weighted", "brier_score", "conformal_mean_set_size"]:
        v_base = baseline[metric]
        v_emb = with_embeddings[metric]
        delta = v_emb - v_base
        # Para brier score, menor es mejor → invertir dirección del Δ
        if metric == "brier_score":
            direction = "mejor" if delta < 0 else "peor"
        elif metric == "conformal_mean_set_size":
            direction = "mejor" if delta < 0 else "peor"
        else:
            direction = "mejor" if delta > 0 else "peor"

        if metric == "conformal_mean_set_size":
            print(f"  {metric:<30s} {v_base:>12.4f} {v_emb:>12.4f} {delta:>+9.4f} ({direction})")
        else:
            print(f"  {metric:<30s} {v_base:>12.4f} {v_emb:>12.4f} {delta:>+9.4f} ({direction})")

    # Decisión
    f1_delta = with_embeddings["f1_weighted"] - baseline["f1_weighted"]
    brier_delta = with_embeddings["brier_score"] - baseline["brier_score"]
    cs_delta = with_embeddings["conformal_mean_set_size"] - baseline["conformal_mean_set_size"]

    # Criterio: embeddings mejoran si F1 sube O Brier baja significativamente
    mejora_f1 = f1_delta > 0.001  # mejora > 0.1%
    mejora_brier = brier_delta < -0.001
    mejora_conformal = cs_delta < -0.01

    n_mejoras = sum([mejora_f1, mejora_brier, mejora_conformal])

    print(f"\n  {'='*64}")
    print(f"  DECISIÓN POR LOS NÚMEROS:")
    if n_mejoras >= 2:
        print(f"  → Los embeddings geoespaciales MEJORAN el modelo ({n_mejoras}/3 métricas mejoran)")
        print(f"  → RECOMENDACIÓN: incluir embeddings en el ensemble")
    elif n_mejoras == 1:
        print(f"  → Resultado MIXTO ({n_mejoras}/3 métricas mejoran)")
        print(f"  → RECOMENDACIÓN: explorar más fuentes de embeddings o ajustar features")
    else:
        print(f"  → Los embeddings geoespaciales NO mejoran el modelo (0/3 métricas mejoran)")
        print(f"  → RECOMENDACIÓN: NO incluir embeddings en el ensemble en su forma actual")

    print(f"\n  Detalle:")
    print(f"    F1-weighted:     {'+' if f1_delta >= 0 else ''}{f1_delta:.4f} ({'MEJORA' if mejora_f1 else 'sin cambio significativo'})")
    print(f"    Brier score:     {'+' if brier_delta >= 0 else ''}{brier_delta:.4f} ({'MEJORA' if mejora_brier else 'sin cambio significativo'})")
    print(f"    Conformal set:   {'+' if cs_delta >= 0 else ''}{cs_delta:.4f} ({'MEJORA' if mejora_conformal else 'sin cambio significativo'})")

    # Top features con embeddings
    print(f"\n  Features más importantes CON embeddings:")
    for name, imp in with_embeddings["top_features"]:
        is_embedding = name.startswith("census_") or name.startswith("spatial_")
        marker = " ← EMBEDDING" if is_embedding else ""
        print(f"    {name:40s} {imp:.4f}{marker}")


def main():
    print("ML-003: Benchmark de embeddings geoespaciales")
    print("=" * 60)

    # 1. Cargar datos
    df = load_dataset()

    # 2. Añadir embeddings
    df_emb = add_embeddings_to_dataset(df)

    # 3. Entrenar SIN embeddings (baseline)
    print("\n--- Preparando features BASELINE (sin embeddings) ---")
    X_train_b, X_test_b, y_train_b, y_test_b = prepare_features(df, with_embeddings=False)

    # Nombres de features del baseline
    feature_names_b = [c for c in X_train_b.columns] if hasattr(X_train_b, 'columns') else [f"f{i}" for i in range(X_train_b.shape[1])]
    if isinstance(X_train_b, pd.DataFrame):
        feature_names_b = list(X_train_b.columns)
    else:
        feature_names_b = [f"f{i}" for i in range(X_train_b.shape[1])]

    baseline = train_and_evaluate(
        X_train_b.values if hasattr(X_train_b, 'values') else X_train_b,
        X_test_b.values if hasattr(X_test_b, 'values') else X_test_b,
        y_train_b.values if hasattr(y_train_b, 'values') else y_train_b,
        y_test_b.values if hasattr(y_test_b, 'values') else y_test_b,
        feature_names_b,
        label="BASELINE — sin embeddings geoespaciales",
    )

    # 4. Entrenar CON embeddings
    print("\n--- Preparando features CON embeddings ---")
    X_train_e, X_test_e, y_train_e, y_test_e = prepare_features(df_emb, with_embeddings=True)

    feature_names_e = list(X_train_e.columns) if hasattr(X_train_e, 'columns') else [f"f{i}" for i in range(X_train_e.shape[1])]

    with_emb = train_and_evaluate(
        X_train_e.values if hasattr(X_train_e, 'values') else X_train_e,
        X_test_e.values if hasattr(X_test_e, 'values') else X_test_e,
        y_train_e.values if hasattr(y_train_e, 'values') else y_train_e,
        y_test_e.values if hasattr(y_test_e, 'values') else y_test_e,
        feature_names_e,
        label="CON EMBEDDINGS GEOESPACIALES (census + spatial)",
    )

    # 5. Comparar
    print_comparison(baseline, with_emb)

    # 6. Documentación de fuentes externas
    print(f"\n{'='*70}")
    print("  NOTA SOBRE FUENTES DE EMBEDDINGS")
    print(f"{'='*70}")
    print("  Fuentes evaluadas (OPEN SOURCE):")
    print("    1. CensusDemographicsEmbedding — datos INE (Padrón Continuo 2023)")
    print("       Análogo a census/ACS embeddings de data.census.gov")
    print("       8 features: envejecimiento, estructura poblacional, urbanización")
    print("    2. SpatialCoordinateEmbedding — coordenadas de capitales")
    print("       Análogo a embeddings espaciales de PDFM/AlphaEarth")
    print("       5 features: lat/lon normalizado, distancia a centro, costa")
    print()
    print("  Fuentes NO evaluadas (PROPIETARIAS, requieren API Google):")
    print("    - Population Dynamics Foundation Models (PDFM): require Google Earth Engine")
    print("    - AlphaEarth: require Google Cloud API access")
    print("    - Estas fuentes codifican información similar a la evaluada aquí")
    print("      (demografía + espacio) pero con resolución mayor y modelos preentrenados")
    print("    → Los open alternatives replican la información esencial")


if __name__ == "__main__":
    main()
