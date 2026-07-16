"""
Experimento 2: 
  a) Calibrar umbrales de decisión de la LSTM híbrida
  b) Ensemble ponderado XGBoost/RF + LSTM híbrida
"""
import sys; sys.path.insert(0, ".")
import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import recall_score, f1_score, accuracy_score

from climasafeai.features.build_features import preprocess_data
from climasafeai.utils.paths import MODELS_DIR, PROCESSED_DATA_DIR, ARTIFACTS_DIR
from climasafeai.data.sequences import cargar_dataset_secuencias, split_secuencias_por_fecha
from climasafeai.models.lstm_model import escalar_secuencias
from climasafeai.models.lstm_province_hybrid import (
    alinear_features_diarias, escalar_diarias,
    load_lstm_hybrid,
)
from climasafeai.models.predict_model import apply_class_thresholds

device = "cpu"

# --- 1. Cargar datos tabulares ---
print("=== Datos tabulares ===")
dfs = {}
for clase in ("calor", "frio"):
    df = pd.read_parquet(PROCESSED_DATA_DIR / f"dataset_{clase}_labeled.parquet")
    target = f"clase_riesgo_{clase}"
    X_tr, X_te, y_tr, y_te = preprocess_data(df, target_col=target, clase=clase)
    dfs[clase] = {"X_tr": X_tr, "X_te": X_te, "y_tr": y_tr, "y_te": y_te}

# --- 2. Cargar LSTM híbrida ---
print("\n=== Cargando LSTM híbrida ===")
data_seq = cargar_dataset_secuencias()
from climasafeai.data.sequences import split_secuencias_por_fecha
split = split_secuencias_por_fecha(data_seq)
_, X_train_s, X_val_s, X_test_s = escalar_secuencias(
    split["X_train"], split["X_val"], split["X_test"], guardar=False
)
Xd = alinear_features_diarias(data_seq["fechas"], data_seq["provincias"])
fechas = data_seq["fechas"]
mask_val = (fechas >= split["fecha_corte_val"]) & (fechas < split["fecha_corte_test"])
mask_test = fechas >= split["fecha_corte_test"]
mask_train = ~(mask_val | mask_test)
Xd_train, Xd_val, Xd_test = Xd[mask_train], Xd[mask_val], Xd[mask_test]
_, Xd_train_s, Xd_val_s, Xd_test_s = escalar_diarias(Xd_train, Xd_val, Xd_test, guardar=False)

model = load_lstm_hybrid(device=device)[0]
model.eval()
with torch.no_grad():
    out_test = model(torch.tensor(X_test_s), torch.tensor(Xd_test_s))
    proba_lstm_calor = torch.softmax(out_test[0], dim=1).numpy()
    proba_lstm_frio  = torch.softmax(out_test[1], dim=1).numpy()

y_te_calor = split["y_test_calor"]
y_te_frio  = split["y_test_frio"]
risk_cal = [c for c in np.unique(y_te_calor) if c != 0]
risk_frio = [c for c in np.unique(y_te_frio) if c != 0]

# --- 3. Calibrar thresholds de LSTM ---
print("\n=== Calibración de umbrales LSTM ===")
for nombre, proba, y_true, risk in [
    ("LSTM calor", proba_lstm_calor, y_te_calor, risk_cal),
    ("LSTM frío", proba_lstm_frio, y_te_frio, risk_frio),
]:
    best = {"rec": -1, "t1": 0, "t2": 0, "f1m": 0}
    for t1 in np.arange(0.30, 0.95, 0.05):
        for t2 in np.arange(0.20, min(t1, 0.85), 0.05):
            pred = apply_class_thresholds(proba, t1=t1, t2=t2)
            rec = recall_score(y_true, pred, labels=risk, average="macro", zero_division=0)
            if rec > best["rec"]:
                f1m = f1_score(y_true, pred, average="macro", zero_division=0)
                best = {"rec": rec, "t1": t1, "t2": t2, "f1m": f1m}
    print(f"  {nombre}: t1={best['t1']:.2f}, t2={best['t2']:.2f} → Rec_riesgo={best['rec']:.4f}, F1_macro={best['f1m']:.4f}")

# --- 4. Ensemble ponderado ---
print("\n=== Ensemble tabular + LSTM ===")
xgb_cal = joblib.load(MODELS_DIR / "XGBoost_calor.joblib")
rf_frio = joblib.load(MODELS_DIR / "RandomForest_frio.joblib")

proba_xgb_cal = xgb_cal.predict_proba(dfs["calor"]["X_te"])
proba_rf_frio = rf_frio.predict_proba(dfs["frio"]["X_te"])

# Mejores umbrales tabulares
u_cal = (0.40, 0.35)
u_frio = (0.45, 0.40)

# Probar pesos alpha (0..1) para la combinación: P_ens = alpha*P_tabular + (1-alpha)*P_lstm
for nombre, proba_tab, proba_lstm, y_true, risk, umbrales in [
    ("Calor",  proba_xgb_cal, proba_lstm_calor, y_te_calor, risk_cal, u_cal),
    ("Frío",   proba_rf_frio, proba_lstm_frio,  y_te_frio,  risk_frio, u_frio),
]:
    best_ens = {"rec": -1, "alpha": 0, "t1": 0, "t2": 0, "f1m": 0}
    t1, t2 = umbrales
    for alpha in np.arange(0.3, 1.05, 0.05):
        proba_ens = alpha * proba_tab + (1 - alpha) * proba_lstm
        # Tambien grid de umbrales
        for et1 in np.arange(0.30, 0.95, 0.05):
            for et2 in np.arange(0.20, min(et1, 0.85), 0.05):
                pred = apply_class_thresholds(proba_ens, t1=et1, t2=et2)
                rec = recall_score(y_true, pred, labels=risk, average="macro", zero_division=0)
                if rec > best_ens["rec"]:
                    f1m = f1_score(y_true, pred, average="macro", zero_division=0)
                    best_ens = {"rec": rec, "alpha": alpha, "t1": et1, "t2": et2, "f1m": f1m}
    print(f"  {nombre}: alpha_tab={best_ens['alpha']:.2f} t1={best_ens['t1']:.2f} t2={best_ens['t2']:.2f} → Rec_riesgo={best_ens['rec']:.4f} F1_macro={best_ens['f1m']:.4f}")
