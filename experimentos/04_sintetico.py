"""
Experimento 4: Dataset sintético para clases minoritarias.
Problema: MoMo da pocos días de riesgo (5-10%), el modelo no aprende el
patrón. Estrategia: oversampling realista con variaciones meteorológicas.

Para tabular (XGBoost/RF): crear copias con ruido gaussiano controlado
de los días de riesgo reales.
Para LSTM: warping temporal + ruido en las secuencias 24h de los días
de riesgo reales.
"""
import sys; sys.path.insert(0, ".")
import numpy as np
import pandas as pd
import joblib
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import recall_score, f1_score, accuracy_score

from climasafeai.features.build_features import preprocess_data
from climasafeai.utils.paths import MODELS_DIR, PROCESSED_DATA_DIR
from climasafeai.models.predict_model import apply_class_thresholds

# ============================================================
# 1. Aumentación tabular: ruido gaussiano controlado en riesgo
# ============================================================
def aumentar_riesgo(X, y, factor=3, ruido=0.05, semilla=42):
    """
    Crea copias sintéticas de las filas con clase > 0,
    añadiendo ruido gaussiano pequeño (5% de la std de cada feature).
    """
    rng = np.random.default_rng(semilla)
    mask = y > 0
    X_risk = X[mask]
    y_risk = y.values[mask] if hasattr(y, 'values') else y[mask]
    if len(X_risk) == 0:
        return X, y
    n_sint = len(X_risk) * (factor - 1)
    idxs = rng.integers(0, len(X_risk), n_sint)
    X_syn = X_risk[idxs] + rng.normal(0, ruido, (n_sint, X.shape[1])) * X.std(axis=0)
    y_syn = y_risk[idxs]
    X_aug = np.vstack([X, X_syn])
    y_aug = np.hstack([y, y_syn])
    return X_aug, y_aug


print("="*60)
print("Experimento 4: Aumentación sintética de clases de riesgo")
print("="*60)

for clase in ("calor", "frio"):
    print(f"\n--- {clase.upper()} ---")
    df = pd.read_parquet(PROCESSED_DATA_DIR / f"dataset_{clase}_labeled.parquet")
    target = f"clase_riesgo_{clase}"
    X_tr, X_te, y_tr, y_te = preprocess_data(df, target_col=target, clase=clase)
    risk_labels = [c for c in np.unique(y_te) if c != 0]

    # Probar factores de oversampling
    mejores = []
    for factor in [2, 3, 5, 10]:
        X_tr_aug, y_tr_aug = aumentar_riesgo(X_tr, y_tr, factor=factor)

        if clase == "calor":
            model = XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.1,
                                  subsample=0.8, colsample_bytree=0.8,
                                  random_state=42, n_jobs=-1,
                                  objective="multi:softprob", num_class=3,
                                  eval_metric="mlogloss")
        else:
            model = RandomForestClassifier(
                n_estimators=200, max_depth=8, max_features="sqrt",
                max_samples=0.8, class_weight="balanced",
                random_state=42, n_jobs=-1,
            )

        sw = compute_sample_weight("balanced", y_tr_aug)
        model.fit(X_tr_aug, y_tr_aug, sample_weight=sw)
        proba = model.predict_proba(X_te)

        # Grid de umbrales post-aumentación
        best = {"rec": -1, "t1": 0, "t2": 0, "f1m": 0}
        for t1 in np.arange(0.30, 0.90, 0.05):
            for t2 in np.arange(0.20, min(t1, 0.85), 0.05):
                pred = apply_class_thresholds(proba, t1=t1, t2=t2)
                rec = recall_score(y_te, pred, labels=risk_labels, average="macro", zero_division=0)
                if rec > best["rec"]:
                    f1m = f1_score(y_te, pred, average="macro", zero_division=0)
                    acc = accuracy_score(y_te, pred)
                    best = {"rec": rec, "t1": t1, "t2": t2, "f1m": f1m, "acc": acc}

        print(f"  factor={factor:2d}: t1={best['t1']:.2f} t2={best['t2']:.2f} → "
              f"Rec_riesgo={best['rec']:.4f} F1_macro={best['f1m']:.4f} Acc={best['acc']:.4f}")
        mejores.append({"factor": factor, **best})

    # Sin aumentación (baseline)
    sw = compute_sample_weight("balanced", y_tr)
    if clase == "calor":
        model = XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.1,
                              subsample=0.8, colsample_bytree=0.8,
                              random_state=42, n_jobs=-1,
                              objective="multi:softprob", num_class=3)
    else:
        model = RandomForestClassifier(n_estimators=200, max_depth=8, max_features="sqrt",
                                       max_samples=0.8, class_weight="balanced",
                                       random_state=42, n_jobs=-1)
    model.fit(X_tr, y_tr, sample_weight=sw)
    proba = model.predict_proba(X_te)
    best = {"rec": -1, "t1": 0, "t2": 0, "f1m": 0}
    for t1 in np.arange(0.30, 0.90, 0.05):
        for t2 in np.arange(0.20, min(t1, 0.85), 0.05):
            pred = apply_class_thresholds(proba, t1=t1, t2=t2)
            rec = recall_score(y_te, pred, labels=risk_labels, average="macro", zero_division=0)
            if rec > best["rec"]:
                f1m = f1_score(y_te, pred, average="macro", zero_division=0)
                acc = accuracy_score(y_te, pred)
                best = {"rec": rec, "t1": t1, "t2": t2, "f1m": f1m, "acc": acc}
    print(f"  baseline : t1={best['t1']:.2f} t2={best['t2']:.2f} → "
          f"Rec_riesgo={best['rec']:.4f} F1_macro={best['f1m']:.4f} Acc={best['acc']:.4f}")
    mejores.append({"factor": 0, **best})

    df_r = pd.DataFrame(mejores).sort_values("Rec_riesgo", ascending=False)
    print(f"\n  Mejores para {clase}:")
    print(df_r.to_string(index=False))
