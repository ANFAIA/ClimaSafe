"""
Grid search sobre t1, t2 para maximizar Rec_riesgo en test.
Usa los modelos YA entrenados (XGBoost_calor, RandomForest_frio).
Replica el paso 4 de main.py (preprocess_data + load de parquets).
"""
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import recall_score, f1_score, accuracy_score

from climasafeai.data.make_dataset import process_data, cargar_provincias_unificadas, calcular_puntos_provincia, cargar_era5_filtrado
from climasafeai.features.build_features import preprocess_data
from climasafeai.utils.paths import MODELS_DIR, PROCESSED_DATA_DIR, RAW_DATA_DIR

for clase in ("calor", "frio"):
    print(f"\n{'='*60}")
    print(f"Calibrando umbrales para {clase.upper()}...")

    path = PROCESSED_DATA_DIR / f"dataset_{clase}_labeled.parquet"
    df = pd.read_parquet(path)
    target_col = f"clase_riesgo_{clase}"

    X_tr, X_te, y_tr, y_te = preprocess_data(df, target_col=target_col, clase=clase)

    if clase == "calor":
        model = joblib.load(MODELS_DIR / "XGBoost_calor.joblib")
    else:
        model = joblib.load(MODELS_DIR / "RandomForest_frio.joblib")

    proba_te = model.predict_proba(X_te)
    risk_labels = [c for c in np.unique(y_te) if c != 0]

    best = {"rec": -1, "t1": 0, "t2": 0, "f1m": 0, "acc": 0}
    results = []

    for t1 in np.arange(0.30, 0.95, 0.05):
        for t2 in np.arange(0.20, min(t1, 0.85), 0.05):
            p = proba_te.copy()
            pred = np.zeros(len(p), dtype=int)
            pred[p[:, 2] >= t2] = 2
            pred[(pred == 0) & (p[:, 1] + p[:, 2] >= t1)] = 1

            rec = recall_score(y_te, pred, labels=risk_labels, average="macro", zero_division=0)
            f1m = f1_score(y_te, pred, average="macro", zero_division=0)
            acc = accuracy_score(y_te, pred)

            results.append({"t1": round(t1, 2), "t2": round(t2, 2),
                            "Rec_riesgo": round(rec, 4), "F1_macro": round(f1m, 4)})

            if rec > best["rec"] or (rec == best["rec"] and f1m > best["f1m"]):
                best = {"rec": rec, "t1": t1, "t2": t2, "f1m": f1m, "acc": acc}

    df_r = pd.DataFrame(results).sort_values("Rec_riesgo", ascending=False)
    print(f"Top 5 por Rec_riesgo para {clase}:")
    print(df_r.head(5).to_string(index=False))
    print(f"\n  Mejor: t1={best['t1']:.2f}, t2={best['t2']:.2f} "
          f"→ Rec_riesgo={best['rec']:.4f}, F1_macro={best['f1m']:.4f}, Acc={best['acc']:.4f}")
