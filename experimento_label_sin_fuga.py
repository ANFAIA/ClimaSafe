"""
Experimento: fuga temporal train<->test en el label de riesgo.

Los cortes p75/p95 de mortalidad atribuida se calculaban en
climasafeai/features/labels.py sobre TODO el histórico por provincia, así que
el label del periodo de train incorporaba información de la distribución del
periodo de test (fuga temporal). Este script mide la magnitud de esa fuga y
su efecto en las métricas honestas:

  1. Recalcula el label de dos maneras:
     - CON fuga  : percentiles sobre todo el histórico (comportamiento clásico).
     - SIN fuga  : percentiles solo con fechas < inicio del test
                   (fecha_corte_percentiles), aplicados a todo el dataset.
     El corte replica la regla de split del proyecto (preprocess_data,
     split_by_date=True): último 20% de fechas distintas -> test.
  2. Cuenta cuántas filas cambian de clase (train y test por separado).
  3. Reentrena XGBoost (calor) y RandomForest (frío) con los MISMOS
     hiperparámetros de los modelos guardados (sklearn.base.clone) sobre cada
     variante del label y evalúa en test (Rec_riesgo = recall macro de las
     clases 1 y 2, F1_macro -- mismas definiciones que predict_model.py).

Resultados documentados en documentacion/label_sin_fuga.md.

Uso (desde la raíz del repo, con el paquete apuntando a este checkout):
    python experimento_label_sin_fuga.py
Requiere en models/ los joblib de referencia (XGBoost_calor.joblib,
RandomForest_frio.joblib) solo para clonar hiperparámetros, no para predecir.
"""
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import f1_score, recall_score
from sklearn.utils.class_weight import compute_sample_weight

from climasafeai.features.build_features import preprocess_data
from climasafeai.features.labels import (
    asignar_clase_riesgo_calor,
    asignar_clase_riesgo_frio,
)
from climasafeai.utils.paths import MODELS_DIR, PROCESSED_DATA_DIR

TEST_SIZE = 0.2  # mismo valor por defecto que preprocess_data

CONFIG = {
    "calor": {
        "parquet": PROCESSED_DATA_DIR / "dataset_calor_labeled.parquet",
        "target_col": "clase_riesgo_calor",
        "label_fn": asignar_clase_riesgo_calor,
        "modelo_joblib": MODELS_DIR / "XGBoost_calor.joblib",
        "modelo_nombre": "XGBoost",
        "usa_sample_weight": True,  # XGBoost no tiene class_weight
    },
    "frio": {
        "parquet": PROCESSED_DATA_DIR / "dataset_frio_labeled.parquet",
        "target_col": "clase_riesgo_frio",
        "label_fn": asignar_clase_riesgo_frio,
        "modelo_joblib": MODELS_DIR / "RandomForest_frio.joblib",
        "modelo_nombre": "RandomForest",
        "usa_sample_weight": False,  # ya lleva class_weight='balanced'
    },
}


def fecha_inicio_test(df: pd.DataFrame, test_size: float = TEST_SIZE) -> pd.Timestamp:
    """Replica la regla de split de preprocess_data(split_by_date=True):
    el test son las últimas max(1, round(n_fechas*test_size)) fechas
    distintas; devuelve la primera fecha del test."""
    fechas_unicas = np.sort(pd.to_datetime(df["fecha"]).unique())
    n_test = max(1, round(len(fechas_unicas) * test_size))
    return pd.Timestamp(fechas_unicas[-n_test])


def entrena_y_evalua(df, clase, cfg):
    """preprocess_data + clon del modelo de referencia + métricas de test."""
    X_train, X_test, y_train, y_test = preprocess_data(
        df, target_col=cfg["target_col"], clase=clase, test_size=TEST_SIZE,
    )
    modelo = clone(joblib.load(cfg["modelo_joblib"]))
    modelo.set_params(n_jobs=2)
    if cfg["usa_sample_weight"]:
        sw = compute_sample_weight("balanced", y_train)
        modelo.fit(X_train, y_train, sample_weight=sw)
    else:
        modelo.fit(X_train, y_train)

    y_pred = modelo.predict(X_test)
    risk_labels = [c for c in np.unique(y_test) if c != 0]
    rec_por_clase = recall_score(
        y_test, y_pred, labels=sorted(np.unique(y_test)), average=None, zero_division=0
    )
    return {
        "Rec_riesgo": round(float(recall_score(
            y_test, y_pred, labels=risk_labels, average="macro", zero_division=0
        )), 4),
        "F1_macro": round(float(f1_score(
            y_test, y_pred, average="macro", zero_division=0
        )), 4),
        "recall_por_clase": {
            int(c): round(float(r), 4)
            for c, r in zip(sorted(np.unique(y_test)), rec_por_clase)
        },
        "dist_y_test": {int(k): int(v) for k, v in pd.Series(y_test).value_counts().items()},
    }


def diff_labels(a: pd.Series, b: pd.Series, mask_train: pd.Series) -> dict:
    """Cuántas filas cambian de clase entre dos versiones del label."""
    cambia = a != b
    return {
        "total": int(cambia.sum()),
        "pct_total": round(100 * cambia.mean(), 2),
        "train": int(cambia[mask_train].sum()),
        "pct_train": round(100 * cambia[mask_train].mean(), 2),
        "test": int(cambia[~mask_train].sum()),
        "pct_test": round(100 * cambia[~mask_train].mean(), 2),
    }


def main():
    resultados = {}
    for clase, cfg in CONFIG.items():
        print(f"\n{'='*70}\nCLASE: {clase} ({cfg['modelo_nombre']})\n{'='*70}")
        df = pd.read_parquet(cfg["parquet"])
        corte = fecha_inicio_test(df)
        mask_train = pd.to_datetime(df["fecha"]) < corte
        print(f"Corte (inicio test): {corte.date()} | train={mask_train.sum()} filas, "
              f"test={(~mask_train).sum()} filas")

        target = cfg["target_col"]
        label_parquet = df[target].copy()

        # CON fuga: percentiles full-history (comportamiento clásico commiteado)
        df_confuga = cfg["label_fn"](df)
        # SIN fuga: percentiles train-only aplicados a todo el dataset
        df_sinfuga = cfg["label_fn"](df, fecha_corte_percentiles=corte)

        res = {
            "corte": str(corte.date()),
            "n_train": int(mask_train.sum()),
            "n_test": int((~mask_train).sum()),
            # ¿El label guardado en el parquet coincide con el recomputado
            # con el código commiteado? (si no, el parquet lleva algún
            # filtro/cambio local sin commitear)
            "parquet_vs_recomputado_confuga": diff_labels(
                label_parquet, df_confuga[target], mask_train
            ),
            "cambios_label_confuga_vs_sinfuga": diff_labels(
                df_confuga[target], df_sinfuga[target], mask_train
            ),
            "dist_test_confuga": {
                int(k): int(v)
                for k, v in df_confuga.loc[~mask_train.values, target].value_counts().items()
            },
            "dist_test_sinfuga": {
                int(k): int(v)
                for k, v in df_sinfuga.loc[~mask_train.values, target].value_counts().items()
            },
        }

        print("\n--- Entrenamiento CON fuga (label clásico full-history) ---")
        res["metricas_confuga"] = entrena_y_evalua(df_confuga, clase, cfg)
        print(json.dumps(res["metricas_confuga"], indent=2))

        print("\n--- Entrenamiento SIN fuga (percentiles train-only) ---")
        res["metricas_sinfuga"] = entrena_y_evalua(df_sinfuga, clase, cfg)
        print(json.dumps(res["metricas_sinfuga"], indent=2))

        resultados[clase] = res

    print(f"\n{'='*70}\nRESUMEN\n{'='*70}")
    print(json.dumps(resultados, indent=2, ensure_ascii=False))
    return resultados


if __name__ == "__main__":
    main()
