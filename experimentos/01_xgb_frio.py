"""
Experimento 1: XGBoost para frío (reemplazar RF).
Grid search ligero de hyperparams + calibración de umbrales.
"""
import sys; sys.path.insert(0, ".")
import itertools
import joblib
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.metrics import recall_score, f1_score, accuracy_score
from sklearn.utils.class_weight import compute_sample_weight

from climasafeai.features.build_features import preprocess_data
from climasafeai.utils.paths import MODELS_DIR, PROCESSED_DATA_DIR

# --- Cargar datos frío ---
df = pd.read_parquet(PROCESSED_DATA_DIR / "dataset_frio_labeled.parquet")
X_tr, X_te, y_tr, y_te = preprocess_data(df, target_col="clase_riesgo_frio", clase="frio")
risk_labels = [c for c in np.unique(y_te) if c != 0]
print(f"Train: {X_tr.shape}, Test: {X_te.shape}")
print(f"Clases test: {np.bincount(y_te)}")

# --- Grid de hyperparams ---
best = {"rec": -1, "f1m": 0, "params": None, "umbrales": None}
results = []
grid = [
    {"n_estimators": 200, "max_depth": 6, "learning_rate": 0.1, "subsample": 0.8, "colsample_bytree": 0.8},
    {"n_estimators": 300, "max_depth": 8, "learning_rate": 0.08, "subsample": 0.8, "colsample_bytree": 0.7},
    {"n_estimators": 200, "max_depth": 10, "learning_rate": 0.05, "subsample": 0.7, "colsample_bytree": 0.8},
    {"n_estimators": 150, "max_depth": 6, "learning_rate": 0.15, "subsample": 0.9, "colsample_bytree": 1.0},
]

for params in grid:
    print(f"\n--- XGBoost frío con {params} ---")
    sample_weight = compute_sample_weight("balanced", y_tr)
    model = XGBClassifier(
        **params, random_state=42, n_jobs=-1,
        objective="multi:softprob", num_class=3,
        eval_metric="mlogloss",
    )
    model.fit(X_tr, y_tr, sample_weight=sample_weight)
    proba = model.predict_proba(X_te)

    # Grid de umbrales
    for t1 in np.arange(0.30, 0.90, 0.05):
        for t2 in np.arange(0.20, min(t1, 0.80), 0.05):
            pred = np.zeros(len(proba), dtype=int)
            pred[proba[:, 2] >= t2] = 2
            pred[(pred == 0) & (proba[:, 1] + proba[:, 2] >= t1)] = 1
            rec = recall_score(y_te, pred, labels=risk_labels, average="macro", zero_division=0)
            f1m = f1_score(y_te, pred, average="macro", zero_division=0)
            acc = accuracy_score(y_te, pred)
            results.append({"t1": round(t1,2), "t2": round(t2,2),
                            "md": f"d{params['max_depth']}_lr{params['learning_rate']}",
                            "Rec_riesgo": round(rec,4), "F1_macro": round(f1m,4), "Acc": round(acc,4)})
            if rec > best["rec"]:
                best = {"rec": rec, "f1m": f1m, "acc": acc,
                        "t1": t1, "t2": t2, "params": params, "model": model}

# --- Mostrar top 10 ---
df_r = pd.DataFrame(results).sort_values("Rec_riesgo", ascending=False)
print("\n" + "="*70)
print("TOP 10 GLOBAL (XGBoost frío + umbrales):")
print(df_r.head(10).to_string(index=False))

print(f"\nMEJOR: Rec_riesgo={best['rec']:.4f} F1_macro={best['f1m']:.4f} Acc={best['acc']:.4f}")
print(f"  Umbraales: t1={best['t1']:.2f}, t2={best['t2']:.2f}")
print(f"  Params: {best['params']}")
