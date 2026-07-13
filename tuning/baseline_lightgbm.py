"""
tuning/baseline_lightgbm.py — Baseline LightGBM como candidato a sustituir a KNN.

Contexto: KNN no aporta al sistema de aviso (Rec_riesgo 0.02-0.26, "no avisa").
Este script entrena un LGBMClassifier razonable (class_weight='balanced', mismos
hiperparámetros base que el XGBoost del proyecto para comparar en igualdad) sobre
los datos procesados de calor y frío, y reporta Rec_riesgo / F1_macro frente a
los modelos desplegados actuales.

Ejecutar con:
    uv run python -m tuning.baseline_lightgbm

Solo LEE de data/processed/ — no escribe modelos ni toca artifacts.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import classification_report, f1_score, recall_score

from climasafeai.utils.paths import PROCESSED_DATA_DIR

warnings.filterwarnings("ignore")

# Referencias actuales (modelos desplegados, 27 features): Rec_riesgo / F1_macro
REFERENCIAS = {
    "calor": ("XGBoost (desplegado)", 0.6331, 0.5629),
    "frio":  ("RandomForest (desplegado)", 0.5256, 0.5117),
}

# Hiperparámetros alineados con el XGBoost del proyecto (train_model.py) para
# una comparación en igualdad de condiciones; el desbalance se trata con
# class_weight='balanced' (equivalente al sample_weight del XGBoost).
LGBM_PARAMS = dict(
    n_estimators=300,
    max_depth=6,
    num_leaves=63,
    learning_rate=0.05,
    subsample=0.8,
    subsample_freq=1,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=1.0,
    class_weight="balanced",
    n_jobs=2,
    random_state=42,
    verbosity=-1,
)


def evaluar_clase(clase: str) -> dict:
    X_train = pd.read_csv(PROCESSED_DATA_DIR / f"X_train_{clase}.csv")
    X_test  = pd.read_csv(PROCESSED_DATA_DIR / f"X_test_{clase}.csv")
    y_train = pd.read_csv(PROCESSED_DATA_DIR / f"y_train_{clase}.csv").squeeze()
    y_test  = pd.read_csv(PROCESSED_DATA_DIR / f"y_test_{clase}.csv").squeeze()

    print(f"\n{'=' * 60}\n  LightGBM — clase '{clase}'  "
          f"(train {X_train.shape}, test {X_test.shape})\n{'=' * 60}")

    model = LGBMClassifier(**LGBM_PARAMS)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Mismas métricas que evaluate_models (predict_model.py):
    #   Rec_riesgo -> recall medio de las clases de riesgo (todas menos la 0)
    #   F1_macro   -> media por clase (penaliza ignorar minoritarias)
    f1_macro = f1_score(y_test, y_pred, average="macro", zero_division=0)
    risk_labels = [c for c in np.unique(y_test) if c != 0]
    rec_riesgo = recall_score(
        y_test, y_pred, labels=risk_labels, average="macro", zero_division=0
    )

    print(classification_report(y_test, y_pred, zero_division=0))
    ref_nombre, ref_rec, ref_f1 = REFERENCIAS[clase]
    print(f"  LightGBM      → Rec_riesgo: {rec_riesgo:.4f} | F1_macro: {f1_macro:.4f}")
    print(f"  {ref_nombre:<13} → Rec_riesgo: {ref_rec:.4f} | F1_macro: {ref_f1:.4f}")

    return {
        "clase": clase,
        "Rec_riesgo_LGBM": round(rec_riesgo, 4),
        "F1_macro_LGBM": round(f1_macro, 4),
        "referencia": ref_nombre,
        "Rec_riesgo_ref": ref_rec,
        "F1_macro_ref": ref_f1,
    }


def main() -> None:
    resumen = pd.DataFrame([evaluar_clase(c) for c in ("calor", "frio")])
    print(f"\n{'=' * 60}\n  Resumen LightGBM vs desplegados\n{'=' * 60}")
    print(resumen.to_string(index=False))


if __name__ == "__main__":
    main()
