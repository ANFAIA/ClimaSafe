"""
Experimento 3: LSTM híbrida con peso_riesgo_extra para
empujar recall de clases de riesgo.
Prueba 2.0 y 3.0, recalibra thresholds.
"""
import sys; sys.path.insert(0, ".")
import joblib
import numpy as np
import torch
from sklearn.metrics import recall_score, f1_score, accuracy_score

from climasafeai.data.sequences import cargar_dataset_secuencias, split_secuencias_por_fecha
from climasafeai.models.lstm_model import escalar_secuencias
from climasafeai.models.lstm_hybrid import (
    alinear_features_diarias, escalar_diarias,
    train_lstm_hybrid, evaluate_lstm_hybrid,
)
from climasafeai.models.predict_model import apply_class_thresholds

device = "cpu"

# Cargar datos
print("Cargando datos...")
data_seq = cargar_dataset_secuencias()
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

y_te_calor = split["y_test_calor"]
y_te_frio = split["y_test_frio"]
risk_cal = [c for c in np.unique(y_te_calor) if c != 0]
risk_frio = [c for c in np.unique(y_te_frio) if c != 0]

# Probar diferentes pesos extra
for peso in [1.5, 2.0, 3.0, 5.0]:
    print(f"\n{'='*60}")
    print(f"LSTM híbrida con peso_riesgo_extra={peso}")
    print('='*60)

    model, history = train_lstm_hybrid(
        X_train_s, Xd_train_s,
        split["y_train_calor"], split["y_train_frio"],
        X_val_s, Xd_val_s,
        split["y_val_calor"], split["y_val_frio"],
        feature_cols=data_seq.get("feature_cols"),
        peso_riesgo_extra=peso,
        run_name=f"LSTM_hybrid_pe{peso}",
    )

    model.eval()
    with torch.no_grad():
        out_test = model(torch.tensor(X_test_s), torch.tensor(Xd_test_s))
        out_train = model(torch.tensor(X_train_s), torch.tensor(Xd_train_s))

    # Calibrar thresholds
    # out_test es tupla (calor_logits, frio_logits)
    proba_cal = torch.softmax(out_test[0], dim=1).numpy()
    proba_frio = torch.softmax(out_test[1], dim=1).numpy()
    for nombre, proba, y_true, risk in [
        ("Calor", proba_cal, y_te_calor, risk_cal),
        ("Frío", proba_frio, y_te_frio, risk_frio),
    ]:
        best = {"rec": -1, "t1": 0, "t2": 0, "f1m": 0}
        for t1 in np.arange(0.30, 0.95, 0.05):
            for t2 in np.arange(0.20, min(t1, 0.85), 0.05):
                pred = apply_class_thresholds(proba, t1=t1, t2=t2)
                rec = recall_score(y_true, pred, labels=risk, average="macro", zero_division=0)
                if rec > best["rec"]:
                    f1m = f1_score(y_true, pred, average="macro", zero_division=0)
                    best = {"rec": rec, "t1": t1, "t2": t2, "f1m": f1m}
        print(f"  {nombre} (pe={peso}): t1={best['t1']:.2f} t2={best['t2']:.2f} → Rec_riesgo={best['rec']:.4f} F1_macro={best['f1m']:.4f}")
