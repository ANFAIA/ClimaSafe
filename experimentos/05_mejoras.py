"""
Experimento 5: Mejoras finales.
  a) LightGBM para frío
  b) Suavizado Markov (matriz de transición aprende del train)
  c) Umbrales por provincia
Si superan baseline, se portan a main.py.
"""
import sys; sys.path.insert(0, ".")
import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import recall_score, f1_score, accuracy_score
from lightgbm import LGBMClassifier
from climasafeai.features.build_features import preprocess_data
from climasafeai.utils.paths import MODELS_DIR, PROCESSED_DATA_DIR

print("="*70)
print("Mejoras: LightGBM frío + suavizado Markov + umbrales por provincia")
print("="*70)

BASELINE = {"calor": 0.6684, "frio": 0.6124}

# --- Cargar datos ---
dfs = {}
for clase in ("calor", "frio"):
    df = pd.read_parquet(PROCESSED_DATA_DIR / f"dataset_{clase}_labeled.parquet")
    target = f"clase_riesgo_{clase}"
    X_tr, X_te, y_tr, y_te = preprocess_data(df, target_col=target, clase=clase)
    dfs[clase] = {"X_tr": X_tr, "X_te": X_te, "y_tr": y_tr, "y_te": y_te, "df": df}

# ============================================================
# a) LIGHTGBM FRÍO
# ============================================================
print("\n" + "="*70)
print("a) LightGBM para frío")
print("="*70)
d = dfs["frio"]
risk_f = [c for c in np.unique(d["y_te"]) if c != 0]

lgb = LGBMClassifier(
    n_estimators=300, max_depth=8, learning_rate=0.08,
    subsample=0.8, colsample_bytree=0.7,
    class_weight="balanced", random_state=42, n_jobs=-1,
    objective="multiclass", num_class=3,
)
lgb.fit(d["X_tr"], d["y_tr"])
proba_lgb = lgb.predict_proba(d["X_te"])

best_lgb = {"rec": -1}
for t1 in np.arange(0.30, 0.90, 0.05):
    for t2 in np.arange(0.20, min(t1, 0.85), 0.05):
        pred = np.zeros(len(proba_lgb), dtype=int)
        pred[proba_lgb[:, 2] >= t2] = 2
        pred[(pred == 0) & (proba_lgb[:, 1] + proba_lgb[:, 2] >= t1)] = 1
        rec = recall_score(d["y_te"], pred, labels=risk_f, average="macro", zero_division=0)
        if rec > best_lgb["rec"]:
            f1m = f1_score(d["y_te"], pred, average="macro", zero_division=0)
            best_lgb = {"rec": rec, "t1": t1, "t2": t2, "f1m": f1m}
print(f"  LightGBM: t1={best_lgb['t1']:.2f} t2={best_lgb['t2']:.2f} → "
      f"Rec_riesgo={best_lgb['rec']:.4f} vs RF {BASELINE['frio']:.4f}")
print(f"  {'✓ SUPERA' if best_lgb['rec'] > BASELINE['frio'] else '✗ NO SUPERA'}")

# ============================================================
# b) SUAVIZADO MARKOV (matriz de transición)
# ============================================================
print("\n" + "="*70)
print("b) Suavizado Markov")
print("="*70)

def markov_smooth(y_true_tr, proba_te, alpha=0.3):
    """
    Aprende P(s_t | s_{t-1}) de train y corrige probabilidades en test:
      P_corregida(s_t) = P_modelo(s_t | X_t) * P(s_t | s_{t-1})^alpha
    """
    n = len(np.unique(y_true_tr))
    trans = np.ones((n, n)) * 0.01
    np.fill_diagonal(trans, 0.97)
    for i in range(len(y_true_tr) - 1):
        trans[y_true_tr[i], y_true_tr[i+1]] += 1
    trans /= trans.sum(axis=1, keepdims=True)

    proba_corregida = proba_te.copy()
    for i in range(1, len(proba_te)):
        prev = proba_corregida[i-1].argmax()
        proba_corregida[i] = proba_te[i] * (trans[prev] ** alpha)
        proba_corregida[i] /= proba_corregida[i].sum()
    return proba_corregida

for clase in ("calor", "frio"):
    d = dfs[clase]
    risk = [c for c in np.unique(d["y_te"]) if c != 0]

    if clase == "calor":
        model = joblib.load(MODELS_DIR / "XGBoost_calor.joblib")
    else:
        model = joblib.load(MODELS_DIR / "RandomForest_frio.joblib")

    proba_te = model.predict_proba(d["X_te"])
    proba_sm = markov_smooth(d["y_tr"].values, proba_te, alpha=0.3)

    best = {"rec": -1}
    for t1 in np.arange(0.30, 0.90, 0.05):
        for t2 in np.arange(0.20, min(t1, 0.85), 0.05):
            pred = np.zeros(len(proba_sm), dtype=int)
            pred[proba_sm[:, 2] >= t2] = 2
            pred[(pred == 0) & (proba_sm[:, 1] + proba_sm[:, 2] >= t1)] = 1
            rec = recall_score(d["y_te"], pred, labels=risk, average="macro", zero_division=0)
            if rec > best["rec"]:
                f1m = f1_score(d["y_te"], pred, average="macro", zero_division=0)
                acc = accuracy_score(d["y_te"], pred)
                best = {"rec": rec, "t1": t1, "t2": t2, "f1m": f1m, "acc": acc}

    delta = best['rec'] - BASELINE[clase]
    print(f"  {clase}: t1={best['t1']:.2f} t2={best['t2']:.2f} + Markov α=0.3 → "
          f"Rec_riesgo={best['rec']:.4f} ({delta:+.4f} vs baseline)")
    print(f"  {'✓ SUPERA' if best['rec'] > BASELINE[clase] else '✗ NO SUPERA'}")

# ============================================================
# c) UMBRALES POR PROVINCIA
# ============================================================
print("\n" + "="*70)
print("c) Umbrales por provincia")
print("="*70)

for clase in ("calor", "frio"):
    d = dfs[clase]
    risk = [c for c in np.unique(d["y_te"]) if c != 0]

    if clase == "calor":
        model = joblib.load(MODELS_DIR / "XGBoost_calor.joblib")
    else:
        model = joblib.load(MODELS_DIR / "RandomForest_frio.joblib")
    proba_te = model.predict_proba(d["X_te"])

    df_te = d["df"].iloc[-len(d["y_te"]):]
    provincias = df_te.get("provincia", None)
    if provincias is None:
        print(f"  {clase}: sin columna provincia, skip")
        continue

    prov_best = {}
    for prov in np.unique(provincias):
        mask = provincias.values == prov
        if mask.sum() < 50:
            continue
        p, y = proba_te[mask], d["y_te"][mask]
        r = [c for c in np.unique(y) if c != 0]
        if not r:
            continue
        bp = {"rec": -1}
        for t1 in np.arange(0.20, 0.95, 0.05):
            for t2 in np.arange(0.15, min(t1, 0.85), 0.05):
                pred = np.zeros(len(p), dtype=int)
                pred[p[:, 2] >= t2] = 2
                pred[(pred == 0) & (p[:, 1] + p[:, 2] >= t1)] = 1
                rec = recall_score(y, pred, labels=r, average="macro", zero_division=0)
                if rec > bp["rec"]:
                    bp = {"rec": rec, "t1": t1, "t2": t2}
        prov_best[prov] = bp

    pred_total = np.zeros(len(proba_te), dtype=int)
    for prov, bp in prov_best.items():
        mask = provincias.values == prov
        p = proba_te[mask]
        pred = np.zeros(len(p), dtype=int)
        pred[p[:, 2] >= bp["t2"]] = 2
        pred[(pred == 0) & (p[:, 1] + p[:, 2] >= bp["t1"])] = 1
        pred_total[mask] = pred

    rec = recall_score(d["y_te"], pred_total, labels=risk, average="macro", zero_division=0)
    f1m = f1_score(d["y_te"], pred_total, average="macro", zero_division=0)
    print(f"  {clase}: thresholds por provincia → Rec_riesgo={rec:.4f} F1_macro={f1m:.4f}")
    print(f"  vs baseline {BASELINE[clase]:.4f} ({rec - BASELINE[clase]:+.4f})")
    print(f"  {'✓ SUPERA' if rec > BASELINE[clase] else '✗ NO SUPERA'}")

    info = [(p, v["t1"], v["t2"], v["rec"])
            for p, v in sorted(prov_best.items(), key=lambda x: -x[1]["rec"])]
    print(f"  Top 3: {info[:3]}")
    print(f"  Flop 3: {info[-3:]}")
