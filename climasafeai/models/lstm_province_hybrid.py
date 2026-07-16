"""
climasafeai.models.lstm_province_hybrid — 3 familias LSTM multi-tarea:

  1. LSTMProvince       — tronco LSTM + embedding provincia + features INE
  2. LSTMHybrid         — tronco LSTM + features diarias de contexto de ola
  3. LSTMProvinceHybrid — tronco LSTM + embedding provincia + INE + daily

Cada familia incluye su propio modelo, entrenamiento, evaluación y carga de
checkpoint. Convenciones compartidas con lstm_model.py: split temporal,
pesos de clase 'balanced', early stopping por val loss, métricas F1_macro /
Rec_riesgo, MLflow en 'climasafeai'.
"""

from __future__ import annotations

import copy
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

import mlflow
import mlflow.pytorch

from climasafeai.data.sequences import (
    cargar_dataset_secuencias,
    split_secuencias_por_fecha,
)
from climasafeai.features.external_features import (
    N_FEATURES_PROVINCIA,
    crear_mapping_provincias,
    fetch_ine_features,
    alinear_features_provincia,
    escalar_features_provincia,
)
from climasafeai.models.lstm_model import (
    SEED,
    _pesos_de_clase,
    escalar_secuencias,
)
from climasafeai.models.predict_model import _plot_confusion_matrix
from climasafeai.models.train_model import configurar_mlflow
from climasafeai.utils.paths import (
    MODELS_DIR,
    REPORTS_DIR,
    ARTIFACTS_DIR,
    PROCESSED_DATA_DIR,
)

# =====================================================================
# CONSTANTES
# =====================================================================

# — LSTMProvince
LSTM_PROVINCE_MODEL_PATH = MODELS_DIR / "LSTM_province.pt"
LSTM_PROVINCE_SCALER_PROV_PATH = ARTIFACTS_DIR / "scaler_provincia_features.joblib"
EMBEDDING_DIM_DEFECTO = 16

# — LSTMHybrid
LSTM_HYBRID_MODEL_PATH = MODELS_DIR / "LSTM_hybrid.pt"
LSTM_HYBRID_SCALER_DIARIAS_PATH = ARTIFACTS_DIR / "scaler_diarias_lstm_hybrid.joblib"

DAILY_FEATURE_COLS: list = [
    "t2m_c", "rh", "wind_speed_kmh", "sp",
    "heat_index_c", "wbgt_c", "wind_chill_c",
    "heat_index_mean", "heat_index_std", "heat_index_min", "horas_sobre_umbral",
    "wind_chill_mean", "wind_chill_std", "wind_chill_max", "horas_bajo_umbral",
    "heat_index_c_lag1", "heat_index_c_roll3", "heat_index_c_roll7",
    "dias_consec_sobre_umbral", "grados_dia_calor_roll7", "grados_dia_calor_roll14",
    "wind_chill_mean_roll3", "wind_chill_mean_roll7", "wind_chill_mean_roll14",
    "grados_dia_frio_roll7", "grados_dia_frio_roll14", "dias_consec_bajo_umbral",
    "t2m_min_noche_lag1", "t2m_min_noche_roll7",
    "dias_consec_wc_severo", "horas_wc_severo_sum14",
]

DAILY_PARQUET_PATH = PROCESSED_DATA_DIR / "dataset_calor_labeled.parquet"

# — LSTMProvinceHybrid
LSTM_PROVINCE_HYBRID_MODEL_PATH = MODELS_DIR / "LSTM_province_hybrid.pt"

# =====================================================================
# FAMILIA 1: LSTMProvince (embedding provincia + INE)
# =====================================================================


class LSTMProvinceMultiTask(nn.Module):
    def __init__(
        self,
        n_features: int = 5,
        n_provincias: int = 45,
        emb_dim_provincia: int = EMBEDDING_DIM_DEFECTO,
        n_features_provincia: int = N_FEATURES_PROVINCIA,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.3,
        fusion_size: int = 64,
        n_clases: int = 3,
    ):
        super().__init__()
        self.hparams = {
            "n_features": n_features,
            "n_provincias": n_provincias,
            "emb_dim_provincia": emb_dim_provincia,
            "n_features_provincia": n_features_provincia,
            "hidden_size": hidden_size,
            "num_layers": num_layers,
            "dropout": dropout,
            "fusion_size": fusion_size,
            "n_clases": n_clases,
            "model_type": "concat",
        }
        self.embedding = nn.Embedding(n_provincias, emb_dim_provincia)
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.dropout = nn.Dropout(dropout)
        fusion_in = hidden_size + emb_dim_provincia + n_features_provincia
        self.fusion = nn.Sequential(
            nn.Linear(fusion_in, fusion_size),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.head_calor = nn.Linear(fusion_size, n_clases)
        self.head_frio = nn.Linear(fusion_size, n_clases)

    def forward(self, x_seq, provincia_idx, x_ine):
        _, (h_n, _) = self.lstm(x_seq)
        h_t = self.dropout(h_n[-1])
        emb = self.embedding(provincia_idx)
        z = self.fusion(torch.cat([h_t, emb, x_ine], dim=1))
        return self.head_calor(z), self.head_frio(z)

    @property
    def n_params(self):
        return sum(p.numel() for p in self.parameters())


class LSTMProvinceAttention(nn.Module):
    def __init__(
        self,
        n_features: int = 5,
        n_provincias: int = 45,
        emb_dim_provincia: int = EMBEDDING_DIM_DEFECTO,
        n_features_provincia: int = N_FEATURES_PROVINCIA,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.3,
        fusion_size: int = 64,
        n_clases: int = 3,
    ):
        super().__init__()
        self.hparams = {
            "n_features": n_features,
            "n_provincias": n_provincias,
            "emb_dim_provincia": emb_dim_provincia,
            "n_features_provincia": n_features_provincia,
            "hidden_size": hidden_size,
            "num_layers": num_layers,
            "dropout": dropout,
            "fusion_size": fusion_size,
            "n_clases": n_clases,
            "model_type": "attention",
        }
        self.embedding = nn.Embedding(n_provincias, emb_dim_provincia)
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.attention = nn.Linear(hidden_size, 1)
        fusion_in = hidden_size + emb_dim_provincia + n_features_provincia
        self.fusion = nn.Sequential(
            nn.Linear(fusion_in, fusion_size),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.head_calor = nn.Linear(fusion_size, n_clases)
        self.head_frio = nn.Linear(fusion_size, n_clases)

    def forward(self, x_seq, provincia_idx, x_ine):
        out, (h_n, _) = self.lstm(x_seq)
        attn_scores = self.attention(out).squeeze(-1)
        attn_weights = F.softmax(attn_scores, dim=1).unsqueeze(-1)
        context = (out * attn_weights).sum(dim=1)
        context = self.dropout(context)
        emb = self.embedding(provincia_idx)
        z = self.fusion(torch.cat([context, emb, x_ine], dim=1))
        return self.head_calor(z), self.head_frio(z)

    @property
    def n_params(self):
        return sum(p.numel() for p in self.parameters())


class LSTMProvinceGated(nn.Module):
    def __init__(
        self,
        n_features: int = 5,
        n_provincias: int = 45,
        emb_dim_provincia: int = EMBEDDING_DIM_DEFECTO,
        n_features_provincia: int = N_FEATURES_PROVINCIA,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.3,
        fusion_size: int = 64,
        n_clases: int = 3,
    ):
        super().__init__()
        self.hparams = {
            "n_features": n_features,
            "n_provincias": n_provincias,
            "emb_dim_provincia": emb_dim_provincia,
            "n_features_provincia": n_features_provincia,
            "hidden_size": hidden_size,
            "num_layers": num_layers,
            "dropout": dropout,
            "fusion_size": fusion_size,
            "n_clases": n_clases,
            "model_type": "gated",
        }
        self.embedding = nn.Embedding(n_provincias, emb_dim_provincia)
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.dropout = nn.Dropout(dropout)
        gate_in = hidden_size + emb_dim_provincia + n_features_provincia
        self.gate = nn.Linear(gate_in, 1)
        self.prov_mlp = nn.Sequential(
            nn.Linear(emb_dim_provincia + n_features_provincia, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        fusion_in = hidden_size + hidden_size
        self.fusion = nn.Sequential(
            nn.Linear(fusion_in, fusion_size),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.head_calor = nn.Linear(fusion_size, n_clases)
        self.head_frio = nn.Linear(fusion_size, n_clases)

    def forward(self, x_seq, provincia_idx, x_ine):
        _, (h_n, _) = self.lstm(x_seq)
        h_t = self.dropout(h_n[-1])
        emb = self.embedding(provincia_idx)
        prov_info = self.prov_mlp(torch.cat([emb, x_ine], dim=1))
        g = torch.sigmoid(self.gate(torch.cat([h_t, emb, x_ine], dim=1)))
        z = self.fusion(torch.cat([g * prov_info, (1 - g) * h_t], dim=1))
        return self.head_calor(z), self.head_frio(z)

    @property
    def n_params(self):
        return sum(p.numel() for p in self.parameters())


def preparar_datos():
    data = cargar_dataset_secuencias()
    name_to_id, id_to_name, provincia_idx = crear_mapping_provincias(data["provincias"])
    df_ine = fetch_ine_features(id_to_name)
    Xp = alinear_features_provincia(data["provincias"], df_ine, name_to_id)
    splits = split_secuencias_por_fecha(data)
    fechas = data["fechas"]
    mask_val = (fechas >= splits["fecha_corte_val"]) & (fechas < splits["fecha_corte_test"])
    mask_test = fechas >= splits["fecha_corte_test"]
    mask_train = ~(mask_val | mask_test)
    Xp_train, Xp_val, Xp_test = Xp[mask_train], Xp[mask_val], Xp[mask_test]
    pidx_train, pidx_val, pidx_test = (
        provincia_idx[mask_train],
        provincia_idx[mask_val],
        provincia_idx[mask_test],
    )
    _, X_train_s, X_val_s, X_test_s = escalar_secuencias(
        splits["X_train"], splits["X_val"], splits["X_test"]
    )
    _, Xp_train_s, Xp_val_s, Xp_test_s = escalar_features_provincia(
        Xp_train, Xp_val, Xp_test
    )
    for k in list(splits.keys()):
        if k.startswith("fecha_corte"):
            continue
        splits[k] = splits[k].copy()
    splits["X_train_s"] = X_train_s
    splits["X_val_s"] = X_val_s
    splits["X_test_s"] = X_test_s
    splits["Xp_train_s"] = Xp_train_s
    splits["Xp_val_s"] = Xp_val_s
    splits["Xp_test_s"] = Xp_test_s
    splits["pidx_train"] = pidx_train
    splits["pidx_val"] = pidx_val
    splits["pidx_test"] = pidx_test
    splits["n_provincias"] = len(id_to_name)
    splits["name_to_id"] = name_to_id
    splits["id_to_name"] = id_to_name
    return splits


def train_lstm_province(
    X_train, Xp_train, pidx_train, y_train_calor, y_train_frio,
    X_val, Xp_val, pidx_val, y_val_calor, y_val_frio,
    n_provincias=45, emb_dim_provincia=EMBEDDING_DIM_DEFECTO,
    fusion_size=64, hidden_size=64, num_layers=2, dropout=0.3,
    lr=1e-3, batch_size=256, max_epochs=50, patience=5,
    device=None, run_name="LSTM_province", modelo_cls=LSTMProvinceMultiTask,
    early_stop_metric="val_loss", feature_cols=None,
):
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

    model = modelo_cls(
        n_features=X_train.shape[2], n_provincias=n_provincias,
        emb_dim_provincia=emb_dim_provincia, hidden_size=hidden_size,
        num_layers=num_layers, dropout=dropout, fusion_size=fusion_size,
    ).to(device)

    loss_calor = nn.CrossEntropyLoss(weight=_pesos_de_clase(y_train_calor, device))
    loss_frio = nn.CrossEntropyLoss(weight=_pesos_de_clase(y_train_frio, device))
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    train_dl = DataLoader(
        TensorDataset(
            torch.tensor(X_train), torch.tensor(Xp_train),
            torch.tensor(pidx_train), torch.tensor(y_train_calor),
            torch.tensor(y_train_frio),
        ),
        batch_size=batch_size, shuffle=True,
        generator=torch.Generator().manual_seed(SEED),
    )
    X_val_t = torch.tensor(X_val, device=device)
    Xp_val_t = torch.tensor(Xp_val, device=device)
    pidx_val_t = torch.tensor(pidx_val, device=device)
    y_val_calor_t = torch.tensor(y_val_calor, device=device)
    y_val_frio_t = torch.tensor(y_val_frio, device=device)

    def _rec_riesgo(y_true, y_pred):
        risk = [c for c in np.unique(y_true) if c != 0]
        if not risk:
            return float("nan")
        return recall_score(y_true, y_pred, labels=risk, average="macro", zero_division=0)

    configurar_mlflow()
    mlflow.end_run()

    history = []
    best_val_loss = float("inf")
    best_val_rr = -float("inf")
    best_state = None
    sin_mejora = 0

    print(f"--> {run_name} en {device} "
          f"({sum(p.numel() for p in model.parameters())} parametros, "
          f"early_stop={early_stop_metric})...")

    with mlflow.start_run(run_name=run_name):
        mlflow.log_params({
            **model.hparams,
            "lr": lr, "batch_size": batch_size, "max_epochs": max_epochs,
            "patience": patience, "seed": SEED, "early_stop_metric": early_stop_metric,
            "model_name": run_name, "clase": "multitask",
        })

        for epoch in range(1, max_epochs + 1):
            model.train()
            train_loss = 0.0
            for xb, xpb, pidxb, yb_c, yb_f in train_dl:
                xb, xpb, pidxb = xb.to(device), xpb.to(device), pidxb.to(device)
                optimizer.zero_grad()
                logits_c, logits_f = model(xb, pidxb, xpb)
                loss = loss_calor(logits_c, yb_c.to(device)) + loss_frio(logits_f, yb_f.to(device))
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * len(xb)
            train_loss /= len(train_dl.dataset)

            model.eval()
            with torch.no_grad():
                out_c, out_f = model(X_val_t, pidx_val_t, Xp_val_t)
                val_loss = (
                    loss_calor(out_c, y_val_calor_t) + loss_frio(out_f, y_val_frio_t)
                ).item()
                pred_c = out_c.argmax(1).cpu().numpy()
                pred_f = out_f.argmax(1).cpu().numpy()
                rr_c = _rec_riesgo(y_val_calor, pred_c)
                rr_f = _rec_riesgo(y_val_frio, pred_f)

            history.append({
                "epoch": epoch, "train_loss": train_loss, "val_loss": val_loss,
                "val_rec_riesgo_calor": rr_c, "val_rec_riesgo_frio": rr_f,
            })
            mlflow.log_metrics({
                "train_loss": train_loss, "val_loss": val_loss,
                "val_rec_riesgo_calor": rr_c, "val_rec_riesgo_frio": rr_f,
            }, step=epoch)
            print(f"    epoca {epoch:3d} | train_loss {train_loss:.4f} | "
                  f"val_loss {val_loss:.4f} | val_Rec_riesgo calor {rr_c:.3f} frio {rr_f:.3f}")

            if early_stop_metric == "rec_riesgo":
                val_rr = rr_c + rr_f
                if val_rr > best_val_rr:
                    best_val_rr = val_rr
                    best_state = copy.deepcopy(model.state_dict())
                    sin_mejora = 0
                else:
                    sin_mejora += 1
            else:
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_state = copy.deepcopy(model.state_dict())
                    sin_mejora = 0
                else:
                    sin_mejora += 1

            if sin_mejora >= patience:
                print(f"    Early stopping en epoca {epoch} "
                      f"(mejor val_loss={best_val_loss:.4f})")
                break

        model.load_state_dict(best_state)
        mlflow.log_metric("best_val_loss", best_val_loss)

        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        torch.save(
            {"state_dict": model.state_dict(), "hparams": model.hparams,
             "feature_cols": feature_cols},
            LSTM_PROVINCE_MODEL_PATH,
        )
        print(f"    Modelo guardado -> {LSTM_PROVINCE_MODEL_PATH.name}")

        mlflow.pytorch.log_model(
            model.cpu(), name=run_name,
            registered_model_name=f"climasafeai_LSTM_province",
            serialization_format="pickle",
        )

    return model.cpu().eval(), pd.DataFrame(history)


def evaluate_lstm_province(
    model, X_train_s, Xp_train_s, pidx_train,
    y_train_calor, y_train_frio, X_test_s, Xp_test_s,
    pidx_test, y_test_calor, y_test_frio,
    run_name="LSTM_province_eval",
):
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    model = model.eval()
    with torch.no_grad():
        pred_train = [
            t.argmax(1).numpy()
            for t in model(torch.tensor(X_train_s), torch.tensor(pidx_train), torch.tensor(Xp_train_s))
        ]
        pred_test = [
            t.argmax(1).numpy()
            for t in model(torch.tensor(X_test_s), torch.tensor(pidx_test), torch.tensor(Xp_test_s))
        ]

    cabezas = [
        (f"{run_name}_calor", y_train_calor, y_test_calor, pred_train[0], pred_test[0]),
        (f"{run_name}_frio", y_train_frio, y_test_frio, pred_train[1], pred_test[1]),
    ]

    configurar_mlflow()
    mlflow.end_run()

    results = []
    with mlflow.start_run(run_name=run_name):
        for name, y_tr, y_te, p_tr, p_te in cabezas:
            print(f"\n--- {name} ---")
            acc_train = accuracy_score(y_tr, p_tr)
            acc_test = accuracy_score(y_te, p_te)
            f1_train = f1_score(y_tr, p_tr, average="weighted", zero_division=0)
            f1_test = f1_score(y_te, p_te, average="weighted", zero_division=0)
            prec_test = precision_score(y_te, p_te, average="weighted", zero_division=0)
            rec_test = recall_score(y_te, p_te, average="weighted", zero_division=0)
            f1_macro = f1_score(y_te, p_te, average="macro", zero_division=0)
            risk_labels = [c for c in np.unique(y_te) if c != 0]
            rec_riesgo = (
                recall_score(y_te, p_te, labels=risk_labels, average="macro", zero_division=0)
                if risk_labels else float("nan")
            )
            print(f"  Acc_test: {acc_test:.4f} | F1_macro: {f1_macro:.4f} | Rec_riesgo: {rec_riesgo:.4f}")
            print()
            print(classification_report(y_te, p_te, zero_division=0))
            _plot_confusion_matrix(y_te, p_te, name)

            sufijo = name.split("_")[-1]
            mlflow.log_metrics({
                f"acc_test_{sufijo}": acc_test, f"f1_test_{sufijo}": f1_test,
                f"f1_macro_{sufijo}": f1_macro, f"rec_riesgo_{sufijo}": rec_riesgo,
            })

            results.append({
                "Modelo": name,
                "Acc_train": round(acc_train, 4), "Acc_test": round(acc_test, 4),
                "F1_train": round(f1_train, 4), "F1_test": round(f1_test, 4),
                "Prec_test": round(prec_test, 4), "Rec_test": round(rec_test, 4),
                "F1_macro": round(f1_macro, 4), "Rec_riesgo": round(rec_riesgo, 4),
            })

    df_results = pd.DataFrame(results)
    out_csv = REPORTS_DIR / "resultados_lstm_province.csv"
    df_results.to_csv(out_csv, index=False)
    print(f"\n  Resumen:\n{df_results.to_string(index=False)}")
    print(f"\n  Guardado -> {out_csv}")
    return df_results


def load_lstm_province(
    path: str | Path = LSTM_PROVINCE_MODEL_PATH, device: str = "cpu",
):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model_type = ckpt["hparams"].get("model_type", "concat")
    cls = {"attention": LSTMProvinceAttention, "gated": LSTMProvinceGated}.get(
        model_type, LSTMProvinceMultiTask
    )
    hparams = {k: v for k, v in ckpt["hparams"].items() if k not in ("model_type",)}
    model = cls(**hparams)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    scaler_prov = joblib.load(LSTM_PROVINCE_SCALER_PROV_PATH)
    return model, scaler_prov


def main_lstm_province(
    emb_dim_provincia=EMBEDDING_DIM_DEFECTO, fusion_size=64, dropout=0.3,
    lr=1e-3, early_stop_metric="val_loss", modelo_cls=LSTMProvinceMultiTask,
    tag="LSTM_province",
):
    torch.set_num_threads(2)
    splits = preparar_datos()
    model, history = train_lstm_province(
        splits["X_train_s"], splits["Xp_train_s"], splits["pidx_train"],
        splits["y_train_calor"], splits["y_train_frio"],
        splits["X_val_s"], splits["Xp_val_s"], splits["pidx_val"],
        splits["y_val_calor"], splits["y_val_frio"],
        n_provincias=splits["n_provincias"],
        emb_dim_provincia=emb_dim_provincia, fusion_size=fusion_size,
        dropout=dropout, lr=lr, early_stop_metric=early_stop_metric,
        modelo_cls=modelo_cls, run_name=tag,
    )
    return evaluate_lstm_province(
        model, splits["X_train_s"], splits["Xp_train_s"], splits["pidx_train"],
        splits["y_train_calor"], splits["y_train_frio"],
        splits["X_test_s"], splits["Xp_test_s"], splits["pidx_test"],
        splits["y_test_calor"], splits["y_test_frio"],
        run_name=f"{tag}_eval",
    )


# =====================================================================
# FAMILIA 2: LSTMHybrid (features diarias de contexto de ola)
# =====================================================================


class LSTMHybridMultiTask(nn.Module):
    def __init__(
        self,
        n_features: int = 5,
        n_features_diarias: int = 27,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.3,
        fusion_size: int = 64,
        n_clases: int = 3,
    ):
        super().__init__()
        self.hparams = {
            "n_features": n_features,
            "n_features_diarias": n_features_diarias,
            "hidden_size": hidden_size,
            "num_layers": num_layers,
            "dropout": dropout,
            "fusion_size": fusion_size,
            "n_clases": n_clases,
        }
        self.lstm = nn.LSTM(
            input_size=n_features, hidden_size=hidden_size,
            num_layers=num_layers, dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.fusion = nn.Sequential(
            nn.Linear(hidden_size + n_features_diarias, fusion_size),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.head_calor = nn.Linear(fusion_size, n_clases)
        self.head_frio = nn.Linear(fusion_size, n_clases)

    def forward(self, x_seq, x_dia):
        _, (h_n, _) = self.lstm(x_seq)
        h_t = self.dropout(h_n[-1])
        z = self.fusion(torch.cat([h_t, x_dia], dim=1))
        return self.head_calor(z), self.head_frio(z)


def alinear_features_diarias(
    fechas, provincias, cols=DAILY_FEATURE_COLS, parquet_path=DAILY_PARQUET_PATH,
):
    df = pd.read_parquet(parquet_path, columns=["fecha", "provincia", *cols])
    df["fecha"] = pd.to_datetime(df["fecha"])
    idx = pd.DataFrame({
        "fecha": pd.to_datetime(fechas.astype("datetime64[D]")),
        "provincia": provincias,
    })
    m = idx.merge(df, on=["fecha", "provincia"], how="left", indicator=True)
    if len(m) != len(idx):
        raise ValueError(
            f"(fecha, provincia) duplicados en {parquet_path.name}: "
            f"{len(m)} filas tras merge para {len(idx)} secuencias"
        )
    sin_match = int((m["_merge"] != "both").sum())
    if sin_match:
        raise ValueError(
            f"{sin_match} secuencias sin fila diaria en {parquet_path.name}"
        )
    n_nan = int(m[cols].isna().any(axis=1).sum())
    if n_nan:
        print(f"    Features diarias: {n_nan} filas con NaN (rollings del "
              f"arranque del historico) — se imputaran a la media de train")
    return m[cols].to_numpy(dtype=np.float32)


def escalar_diarias(Xd_train, *Xd_otros, guardar=True):
    scaler = StandardScaler()
    scaler.fit(Xd_train[~np.isnan(Xd_train).any(axis=1)])

    def _transform(Xd):
        Xd_s = scaler.transform(Xd).astype(np.float32)
        return np.nan_to_num(Xd_s, nan=0.0)

    if guardar:
        ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
        joblib.dump(scaler, LSTM_HYBRID_SCALER_DIARIAS_PATH)
        print(f"    Scaler diarias guardado -> {LSTM_HYBRID_SCALER_DIARIAS_PATH.name}")

    return (scaler, _transform(Xd_train), *(_transform(Xd) for Xd in Xd_otros))


def train_lstm_hybrid(
    X_train, Xd_train, y_train_calor, y_train_frio,
    X_val, Xd_val, y_val_calor, y_val_frio,
    hidden_size=64, num_layers=2, dropout=0.3, fusion_size=64,
    lr=1e-3, batch_size=256, max_epochs=50, patience=5,
    device=None, run_name="LSTM_hybrid", feature_cols=None,
    daily_feature_cols=None, peso_riesgo_extra=1.0,
):
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

    model = LSTMHybridMultiTask(
        n_features=X_train.shape[2], n_features_diarias=Xd_train.shape[1],
        hidden_size=hidden_size, num_layers=num_layers,
        dropout=dropout, fusion_size=fusion_size,
    ).to(device)

    def _pesos(y):
        w = _pesos_de_clase(y, device)
        if peso_riesgo_extra != 1.0:
            w = w.clone()
            w[1:] *= peso_riesgo_extra
        return w

    loss_calor = nn.CrossEntropyLoss(weight=_pesos(y_train_calor))
    loss_frio = nn.CrossEntropyLoss(weight=_pesos(y_train_frio))
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    train_dl = DataLoader(
        TensorDataset(
            torch.tensor(X_train), torch.tensor(Xd_train),
            torch.tensor(y_train_calor), torch.tensor(y_train_frio),
        ),
        batch_size=batch_size, shuffle=True,
        generator=torch.Generator().manual_seed(SEED),
    )
    X_val_t = torch.tensor(X_val, device=device)
    Xd_val_t = torch.tensor(Xd_val, device=device)
    y_val_calor_t = torch.tensor(y_val_calor, device=device)
    y_val_frio_t = torch.tensor(y_val_frio, device=device)

    def _rec_riesgo(y_true, y_pred):
        risk = [c for c in np.unique(y_true) if c != 0]
        if not risk:
            return float("nan")
        return recall_score(y_true, y_pred, labels=risk, average="macro", zero_division=0)

    configurar_mlflow()
    mlflow.end_run()

    history = []
    best_val_loss = float("inf")
    best_state = None
    sin_mejora = 0

    print(f"--> Entrenando LSTM hibrida en {device} "
          f"({sum(p.numel() for p in model.parameters())} parametros)...")

    with mlflow.start_run(run_name=run_name):
        mlflow.log_params({
            **model.hparams,
            "lr": lr, "batch_size": batch_size, "max_epochs": max_epochs,
            "patience": patience, "seed": SEED, "peso_riesgo_extra": peso_riesgo_extra,
            "task_type": "clasificacion", "model_name": "LSTM_hybrid", "clase": "multitask",
        })

        for epoch in range(1, max_epochs + 1):
            model.train()
            train_loss = 0.0
            for xb, xdb, yb_c, yb_f in train_dl:
                xb, xdb = xb.to(device), xdb.to(device)
                optimizer.zero_grad()
                logits_c, logits_f = model(xb, xdb)
                loss = loss_calor(logits_c, yb_c.to(device)) + loss_frio(logits_f, yb_f.to(device))
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * len(xb)
            train_loss /= len(train_dl.dataset)

            model.eval()
            with torch.no_grad():
                out_c, out_f = model(X_val_t, Xd_val_t)
                val_loss = (
                    loss_calor(out_c, y_val_calor_t) + loss_frio(out_f, y_val_frio_t)
                ).item()
                pred_c = out_c.argmax(1).cpu().numpy()
                pred_f = out_f.argmax(1).cpu().numpy()

            metricas_epoca = {
                "val_rec_riesgo_calor": _rec_riesgo(y_val_calor, pred_c),
                "val_rec_riesgo_frio": _rec_riesgo(y_val_frio, pred_f),
            }
            history.append({
                "epoch": epoch, "train_loss": train_loss, "val_loss": val_loss,
                **metricas_epoca,
            })
            mlflow.log_metrics({
                "train_loss": train_loss, "val_loss": val_loss, **metricas_epoca,
            }, step=epoch)
            print(f"    epoca {epoch:3d} | train_loss {train_loss:.4f} | "
                  f"val_loss {val_loss:.4f} | Rec_riesgo val calor "
                  f"{metricas_epoca['val_rec_riesgo_calor']:.3f} "
                  f"frio {metricas_epoca['val_rec_riesgo_frio']:.3f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = copy.deepcopy(model.state_dict())
                sin_mejora = 0
            else:
                sin_mejora += 1
                if sin_mejora >= patience:
                    print(f"    Early stopping en epoca {epoch} "
                          f"(mejor val_loss {best_val_loss:.4f})")
                    break

        model.load_state_dict(best_state)
        mlflow.log_metric("best_val_loss", best_val_loss)

        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        torch.save(
            {"state_dict": model.state_dict(), "hparams": model.hparams,
             "feature_cols": feature_cols,
             "daily_feature_cols": daily_feature_cols or DAILY_FEATURE_COLS},
            LSTM_HYBRID_MODEL_PATH,
        )
        print(f"    Modelo guardado -> {LSTM_HYBRID_MODEL_PATH.name}")

        mlflow.pytorch.log_model(
            model.cpu(), name="LSTM_hybrid",
            registered_model_name="climasafeai_LSTM_hybrid",
            serialization_format="pickle",
        )

    return model.cpu().eval(), pd.DataFrame(history)


def evaluate_lstm_hybrid(
    model, X_train_s, Xd_train_s, y_train_calor, y_train_frio,
    X_test_s, Xd_test_s, y_test_calor, y_test_frio,
    run_name="LSTM_hybrid_eval",
):
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    model = model.eval()
    with torch.no_grad():
        pred_train = [
            t.argmax(1).numpy()
            for t in model(torch.tensor(X_train_s), torch.tensor(Xd_train_s))
        ]
        pred_test = [
            t.argmax(1).numpy()
            for t in model(torch.tensor(X_test_s), torch.tensor(Xd_test_s))
        ]

    cabezas = [
        ("LSTM_hybrid_calor", y_train_calor, y_test_calor, pred_train[0], pred_test[0]),
        ("LSTM_hybrid_frio", y_train_frio, y_test_frio, pred_train[1], pred_test[1]),
    ]

    configurar_mlflow()
    mlflow.end_run()

    results = []
    with mlflow.start_run(run_name=run_name):
        for name, y_tr, y_te, p_tr, p_te in cabezas:
            print(f"\n--- {name} ---")
            acc_train = accuracy_score(y_tr, p_tr)
            acc_test = accuracy_score(y_te, p_te)
            f1_train = f1_score(y_tr, p_tr, average="weighted", zero_division=0)
            f1_test = f1_score(y_te, p_te, average="weighted", zero_division=0)
            prec_test = precision_score(y_te, p_te, average="weighted", zero_division=0)
            rec_test = recall_score(y_te, p_te, average="weighted", zero_division=0)
            f1_macro = f1_score(y_te, p_te, average="macro", zero_division=0)
            risk_labels = [c for c in np.unique(y_te) if c != 0]
            rec_riesgo = (
                recall_score(y_te, p_te, labels=risk_labels, average="macro", zero_division=0)
                if risk_labels else float("nan")
            )
            print(f"  Accuracy  -> train: {acc_train:.3f} | test: {acc_test:.3f}")
            print(f"  F1 (w)    -> train: {f1_train:.3f}  | test: {f1_test:.3f}")
            print(f"  F1_macro  -> {f1_macro:.3f}  | Rec_riesgo (clases 1..n) -> {rec_riesgo:.3f}")
            print()
            print(classification_report(y_te, p_te, zero_division=0))
            _plot_confusion_matrix(y_te, p_te, name)

            sufijo = name.split("_")[-1]
            mlflow.log_metrics({
                f"acc_test_{sufijo}": acc_test, f"f1_test_{sufijo}": f1_test,
                f"f1_macro_{sufijo}": f1_macro, f"rec_riesgo_{sufijo}": rec_riesgo,
            })

            results.append({
                "Modelo": name,
                "Acc_train": round(acc_train, 4), "Acc_test": round(acc_test, 4),
                "F1_train": round(f1_train, 4), "F1_test": round(f1_test, 4),
                "Prec_test": round(prec_test, 4), "Rec_test": round(rec_test, 4),
                "F1_macro": round(f1_macro, 4), "Rec_riesgo": round(rec_riesgo, 4),
            })

    df_results = pd.DataFrame(results)
    out_csv = REPORTS_DIR / "resultados_lstm_hybrid.csv"
    df_results.to_csv(out_csv, index=False)
    print(f"\n  Resumen:\n{df_results.to_string(index=False)}")
    print(f"\n  Guardado -> {out_csv}")
    return df_results


def load_lstm_hybrid(path=LSTM_HYBRID_MODEL_PATH, device="cpu"):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = LSTMHybridMultiTask(**ckpt["hparams"])
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    scaler_diarias = joblib.load(LSTM_HYBRID_SCALER_DIARIAS_PATH)
    return model, scaler_diarias


def main_lstm_hybrid(peso_riesgo_extra=1.0, run_name="LSTM_hybrid"):
    torch.set_num_threads(2)
    data = cargar_dataset_secuencias()
    Xd = alinear_features_diarias(data["fechas"], data["provincias"])
    splits = split_secuencias_por_fecha(data)
    fechas = data["fechas"]
    mask_val = (fechas >= splits["fecha_corte_val"]) & (fechas < splits["fecha_corte_test"])
    mask_test = fechas >= splits["fecha_corte_test"]
    mask_train = ~(mask_val | mask_test)
    Xd_train, Xd_val, Xd_test = Xd[mask_train], Xd[mask_val], Xd[mask_test]

    _, X_train_s, X_val_s, X_test_s = escalar_secuencias(
        splits["X_train"], splits["X_val"], splits["X_test"]
    )
    _, Xd_train_s, Xd_val_s, Xd_test_s = escalar_diarias(Xd_train, Xd_val, Xd_test)

    model, history = train_lstm_hybrid(
        X_train_s, Xd_train_s, splits["y_train_calor"], splits["y_train_frio"],
        X_val_s, Xd_val_s, splits["y_val_calor"], splits["y_val_frio"],
        feature_cols=data["feature_cols"], peso_riesgo_extra=peso_riesgo_extra,
        run_name=run_name,
    )

    return evaluate_lstm_hybrid(
        model, X_train_s, Xd_train_s, splits["y_train_calor"], splits["y_train_frio"],
        X_test_s, Xd_test_s, splits["y_test_calor"], splits["y_test_frio"],
        run_name=f"{run_name}_eval",
    )


# =====================================================================
# FAMILIA 3: LSTMProvinceHybrid (provincia + INE + daily features)
# =====================================================================


class LSTMProvinceHybridMultiTask(nn.Module):
    def __init__(
        self,
        n_features: int = 5,
        n_provincias: int = 45,
        emb_dim_provincia: int = 16,
        n_features_provincia: int = N_FEATURES_PROVINCIA,
        n_features_diarias: int = 27,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.3,
        fusion_size: int = 128,
        n_clases: int = 3,
    ):
        super().__init__()
        self.hparams = {
            "n_features": n_features,
            "n_provincias": n_provincias,
            "emb_dim_provincia": emb_dim_provincia,
            "n_features_provincia": n_features_provincia,
            "n_features_diarias": n_features_diarias,
            "hidden_size": hidden_size,
            "num_layers": num_layers,
            "dropout": dropout,
            "fusion_size": fusion_size,
            "n_clases": n_clases,
        }
        self.embedding = nn.Embedding(n_provincias, emb_dim_provincia)
        self.lstm = nn.LSTM(
            input_size=n_features, hidden_size=hidden_size,
            num_layers=num_layers, dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.dropout = nn.Dropout(dropout)
        fusion_in = hidden_size + emb_dim_provincia + n_features_provincia + n_features_diarias
        self.fusion = nn.Sequential(
            nn.Linear(fusion_in, fusion_size),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.head_calor = nn.Linear(fusion_size, n_clases)
        self.head_frio = nn.Linear(fusion_size, n_clases)

    def forward(self, x_seq, provincia_idx, x_ine, x_diarias):
        _, (h_n, _) = self.lstm(x_seq)
        h_t = self.dropout(h_n[-1])
        emb = self.embedding(provincia_idx)
        z = self.fusion(torch.cat([h_t, emb, x_ine, x_diarias], dim=1))
        return self.head_calor(z), self.head_frio(z)

    @property
    def n_params(self):
        return sum(p.numel() for p in self.parameters())


def preparar_datos_hibridos():
    splits = preparar_datos()
    data = cargar_dataset_secuencias()
    Xd = alinear_features_diarias(data["fechas"], data["provincias"])
    fechas = data["fechas"]
    mask_val = (fechas >= splits["fecha_corte_val"]) & (fechas < splits["fecha_corte_test"])
    mask_test = fechas >= splits["fecha_corte_test"]
    mask_train = ~(mask_val | mask_test)
    _, Xd_train_s, Xd_val_s, Xd_test_s = escalar_diarias(
        Xd[mask_train], Xd[mask_val], Xd[mask_test]
    )
    splits["Xd_train_s"] = Xd_train_s
    splits["Xd_val_s"] = Xd_val_s
    splits["Xd_test_s"] = Xd_test_s
    splits["daily_feature_cols"] = DAILY_FEATURE_COLS
    return splits


def train_lstm_province_hybrid(
    X_train, Xp_train, pidx_train, Xd_train,
    y_train_calor, y_train_frio,
    X_val, Xp_val, pidx_val, Xd_val,
    y_val_calor, y_val_frio,
    n_provincias=45, emb_dim_provincia=16,
    fusion_size=128, hidden_size=64, num_layers=2, dropout=0.3,
    lr=1e-3, batch_size=256, max_epochs=50, patience=5,
    device=None, run_name="LSTM_province_hybrid", feature_cols=None,
    peso_riesgo_extra=1.0, seed=None,
):
    _seed = seed if seed is not None else SEED
    torch.manual_seed(_seed)
    np.random.seed(_seed)
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

    model = LSTMProvinceHybridMultiTask(
        n_features=X_train.shape[2], n_provincias=n_provincias,
        emb_dim_provincia=emb_dim_provincia,
        n_features_provincia=Xp_train.shape[1],
        n_features_diarias=Xd_train.shape[1],
        hidden_size=hidden_size, num_layers=num_layers, dropout=dropout,
        fusion_size=fusion_size,
    ).to(device)

    def _pesos(y):
        w = _pesos_de_clase(y, device)
        if peso_riesgo_extra != 1.0:
            w = w.clone()
            w[1:] *= peso_riesgo_extra
        return w
    loss_calor = nn.CrossEntropyLoss(weight=_pesos(y_train_calor))
    loss_frio = nn.CrossEntropyLoss(weight=_pesos(y_train_frio))
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    train_dl = DataLoader(
        TensorDataset(
            torch.tensor(X_train), torch.tensor(Xp_train),
            torch.tensor(pidx_train), torch.tensor(Xd_train),
            torch.tensor(y_train_calor), torch.tensor(y_train_frio),
        ),
        batch_size=batch_size, shuffle=True,
        generator=torch.Generator().manual_seed(_seed),
    )
    X_val_t = [torch.tensor(t, device=device) for t in [X_val, Xp_val, Xd_val]]
    pidx_val_t = torch.tensor(pidx_val, device=device)
    y_val_calor_t = torch.tensor(y_val_calor, device=device)
    y_val_frio_t = torch.tensor(y_val_frio, device=device)

    def _rec_riesgo(y_true, y_pred):
        risk = [c for c in np.unique(y_true) if c != 0]
        return recall_score(y_true, y_pred, labels=risk, average="macro", zero_division=0) if risk else float("nan")

    configurar_mlflow()
    mlflow.end_run()

    history = []
    best_val_loss = float("inf")
    best_state = None
    sin_mejora = 0

    print(f"--> {run_name} en {device} ({model.n_params} parametros)...")

    with mlflow.start_run(run_name=run_name):
        mlflow.log_params({**model.hparams, "lr": lr, "batch_size": batch_size,
                           "max_epochs": max_epochs, "patience": patience, "seed": _seed,
                           "peso_riesgo_extra": peso_riesgo_extra})

        for epoch in range(1, max_epochs + 1):
            model.train()
            train_loss = 0.0
            for xb, xpb, pidxb, xdb, yb_c, yb_f in train_dl:
                xb, xpb, pidxb, xdb = [t.to(device) for t in [xb, xpb, pidxb, xdb]]
                optimizer.zero_grad()
                logits_c, logits_f = model(xb, pidxb, xpb, xdb)
                loss = loss_calor(logits_c, yb_c.to(device)) + loss_frio(logits_f, yb_f.to(device))
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * len(xb)
            train_loss /= len(train_dl.dataset)

            model.eval()
            with torch.no_grad():
                Xv, Xpv, Xdv = [t.to(device) for t in X_val_t]
                out_c, out_f = model(Xv, pidx_val_t, Xpv, Xdv)
                val_loss = (loss_calor(out_c, y_val_calor_t) + loss_frio(out_f, y_val_frio_t)).item()
                pred_c = out_c.argmax(1).cpu().numpy()
                pred_f = out_f.argmax(1).cpu().numpy()
                rr_c = _rec_riesgo(y_val_calor, pred_c)
                rr_f = _rec_riesgo(y_val_frio, pred_f)

            history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss,
                            "val_rec_riesgo_calor": rr_c, "val_rec_riesgo_frio": rr_f})
            mlflow.log_metrics({"train_loss": train_loss, "val_loss": val_loss,
                                "val_rec_riesgo_calor": rr_c, "val_rec_riesgo_frio": rr_f}, step=epoch)
            print(f"    epoca {epoch:3d} | train_loss {train_loss:.4f} | val_loss {val_loss:.4f} | Rec_riesgo calor {rr_c:.3f} frio {rr_f:.3f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = copy.deepcopy(model.state_dict())
                sin_mejora = 0
            else:
                sin_mejora += 1
                if sin_mejora >= patience:
                    print(f"    Early stopping en epoca {epoch} (mejor val_loss={best_val_loss:.4f})")
                    break

        model.load_state_dict(best_state)
        mlflow.log_metric("best_val_loss", best_val_loss)
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        ckpt_path = MODELS_DIR / f"LSTM_province_hybrid_seed{_seed}.pt"
        torch.save({"state_dict": model.state_dict(), "hparams": model.hparams, "feature_cols": feature_cols},
                   ckpt_path)
        print(f"    Modelo guardado -> {ckpt_path.name}")

        mlflow.pytorch.log_model(model.cpu(), name=run_name, serialization_format="pickle")

    return model.cpu().eval(), pd.DataFrame(history)


def evaluate_lstm_province_hybrid(
    model, X_train_s, Xp_train_s, pidx_train, Xd_train_s,
    y_train_calor, y_train_frio, X_test_s, Xp_test_s,
    pidx_test, Xd_test_s, y_test_calor, y_test_frio,
    run_name="LSTM_province_hybrid_eval",
):
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    model = model.eval()
    with torch.no_grad():
        pred_train = [t.argmax(1).numpy() for t in model(
            torch.tensor(X_train_s), torch.tensor(pidx_train),
            torch.tensor(Xp_train_s), torch.tensor(Xd_train_s))]
        pred_test = [t.argmax(1).numpy() for t in model(
            torch.tensor(X_test_s), torch.tensor(pidx_test),
            torch.tensor(Xp_test_s), torch.tensor(Xd_test_s))]

    cabezas = [(f"{run_name}_calor", y_train_calor, y_test_calor, pred_train[0], pred_test[0]),
               (f"{run_name}_frio", y_train_frio, y_test_frio, pred_train[1], pred_test[1])]

    configurar_mlflow()
    mlflow.end_run()
    results = []
    with mlflow.start_run(run_name=run_name):
        for name, y_tr, y_te, p_tr, p_te in cabezas:
            print(f"\n--- {name} ---")
            acc_test = accuracy_score(y_te, p_te)
            f1_macro = f1_score(y_te, p_te, average="macro", zero_division=0)
            risk_labels = [c for c in np.unique(y_te) if c != 0]
            rec_riesgo = recall_score(y_te, p_te, labels=risk_labels, average="macro", zero_division=0) if risk_labels else float("nan")
            print(f"  Acc_test: {acc_test:.4f} | F1_macro: {f1_macro:.4f} | Rec_riesgo: {rec_riesgo:.4f}")
            print(classification_report(y_te, p_te, zero_division=0))
            _plot_confusion_matrix(y_te, p_te, name)
            sufijo = name.split("_")[-1]
            mlflow.log_metrics({f"acc_test_{sufijo}": acc_test, f"f1_macro_{sufijo}": f1_macro, f"rec_riesgo_{sufijo}": rec_riesgo})
            results.append({"Modelo": name, "Acc_test": round(acc_test, 4), "F1_macro": round(f1_macro, 4), "Rec_riesgo": round(rec_riesgo, 4)})

    df_results = pd.DataFrame(results)
    out_csv = REPORTS_DIR / "resultados_lstm_province_hybrid.csv"
    df_results.to_csv(out_csv, index=False)
    print(f"\n{df_results.to_string(index=False)}\nGuardado -> {out_csv}")
    return df_results


def load_lstm_province_hybrid(path: Path = LSTM_PROVINCE_HYBRID_MODEL_PATH, device: str = "cpu"):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = LSTMProvinceHybridMultiTask(**ckpt["hparams"])
    model.load_state_dict(ckpt["state_dict"])
    return model.eval()


def load_by_seed(seed: int, device: str = "cpu"):
    path = MODELS_DIR / f"LSTM_province_hybrid_seed{seed}.pt"
    return load_lstm_province_hybrid(path, device)


class EnsembleLSTM:
    def __init__(self, models: list[LSTMProvinceHybridMultiTask]):
        self.models = models

    def eval(self):
        for m in self.models:
            m.eval()
        return self

    def __call__(self, x_seq, pidx, xp, xd):
        logits_c, logits_f = None, None
        for m in self.models:
            c, f = m(x_seq, pidx, xp, xd)
            if logits_c is None:
                logits_c, logits_f = c, f
            else:
                logits_c += c
                logits_f += f
        return logits_c / len(self.models), logits_f / len(self.models)


def main_lstm_province_hybrid(tag="LSTM_province_hybrid"):
    torch.set_num_threads(2)
    splits = preparar_datos_hibridos()
    model, history = train_lstm_province_hybrid(
        splits["X_train_s"], splits["Xp_train_s"], splits["pidx_train"], splits["Xd_train_s"],
        splits["y_train_calor"], splits["y_train_frio"],
        splits["X_val_s"], splits["Xp_val_s"], splits["pidx_val"], splits["Xd_val_s"],
        splits["y_val_calor"], splits["y_val_frio"],
        n_provincias=splits["n_provincias"], run_name=tag)
    return evaluate_lstm_province_hybrid(
        model, splits["X_train_s"], splits["Xp_train_s"], splits["pidx_train"], splits["Xd_train_s"],
        splits["y_train_calor"], splits["y_train_frio"],
        splits["X_test_s"], splits["Xp_test_s"], splits["pidx_test"], splits["Xd_test_s"],
        splits["y_test_calor"], splits["y_test_frio"], run_name=f"{tag}_eval")


if __name__ == "__main__":
    main_lstm_province_hybrid()
