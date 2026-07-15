"""
climasafeai.models.lstm_province — LSTM multi-tarea con embedding de provincia
+ features demográficas INE.

Variante de la LSTM multi-tarea (lstm_model.py) que añadiendo contexto
geográfico-demográfico: cada secuencia se ve enriquecida con un embedding
aprendido del código de provincia y 4 features demográficas estáticas
(%65+, %80+, %mujeres, log(población)). El tronco LSTM sigue viendo la
secuencia horaria cruda; su último estado oculto se concatena con el embedding
y las features INE antes de una capa de fusión y las dos cabezas.

Dos variantes de arquitectura:
  - LSTMProvinceMultiTask: concatena h_T + embedding + INE -> fusión -> cabezas.
  - LSTMProvinceAttention: atención sobre los 24 pasos temporales (context =
    sum(alpha_i * h_i)) en vez de solo h_T, + embedding + INE -> fusión.

Convenciones idénticas a lstm_model.py: split temporal, pesos de clase
'balanced', early stopping por val loss, métricas F1_macro / Rec_riesgo,
MLflow en el experimento 'climasafeai'.
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
    DEMOGRAPHIC_FEATURES,
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
from climasafeai.utils.paths import MODELS_DIR, REPORTS_DIR, ARTIFACTS_DIR

LSTM_PROVINCE_MODEL_PATH = MODELS_DIR / "LSTM_province.pt"
LSTM_PROVINCE_SCALER_PROV_PATH = ARTIFACTS_DIR / "scaler_provincia_features.joblib"

EMBEDDING_DIM_DEFECTO = 16


class LSTMProvinceMultiTask(nn.Module):
    """
    Tronco LSTM + embedding provincia + features INE -> fusion -> 2 cabezas.
    """

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
    """
    LSTM + atencion sobre las 24h + embedding provincia + INE -> fusion -> 2 cabezas.
    context = sum(alpha_i * h_i) con alpha = softmax(Linear(h_i)).
    """

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
        # x_seq: (batch, 24, n_features)
        out, (h_n, _) = self.lstm(x_seq)
        # out: (batch, 24, hidden)
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
    """
    LSTM + gating: puerta aprendida que controla cuánto peso tiene la info
    provincial vs la secuencia horaria.

    gate = sigmoid(Linear(h_T ⊕ emb ⊕ ine))
    prov_info = MLP_prov(emb ⊕ ine)
    z = gate · prov_info + (1 - gate) · h_T
    z -> fusion -> cabezas
    """

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
    """
    Pipeline completo de datos: npz -> split -> escalado -> features provincia.

    Returns dict con claves 'X_{train,val,test}_s', 'y_{train,val,test}_calor/frio',
    'Xd_{train,val,test}_s', 'pidx_{train,val,test}', y los arrays originales
    para re-uso.
    """
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
    X_train: np.ndarray,
    Xp_train: np.ndarray,
    pidx_train: np.ndarray,
    y_train_calor: np.ndarray,
    y_train_frio: np.ndarray,
    X_val: np.ndarray,
    Xp_val: np.ndarray,
    pidx_val: np.ndarray,
    y_val_calor: np.ndarray,
    y_val_frio: np.ndarray,
    n_provincias: int = 45,
    emb_dim_provincia: int = EMBEDDING_DIM_DEFECTO,
    fusion_size: int = 64,
    hidden_size: int = 64,
    num_layers: int = 2,
    dropout: float = 0.3,
    lr: float = 1e-3,
    batch_size: int = 256,
    max_epochs: int = 50,
    patience: int = 5,
    device: str | None = None,
    run_name: str = "LSTM_province",
    modelo_cls=LSTMProvinceMultiTask,
    early_stop_metric: str = "val_loss",
    feature_cols: list | None = None,
) -> tuple[LSTMProvinceMultiTask | LSTMProvinceAttention, pd.DataFrame]:
    """
    Entrena la LSTM con embedding de provincia y devuelve (mejor_modelo, history).

    `early_stop_metric` ∈ {"val_loss", "rec_riesgo"}. "rec_riesgo" usa la
    suma de Rec_riesgo de calor+frio como criterio de parada.
    `modelo_cls`: LSTMProvinceMultiTask o LSTMProvinceAttention.
    Checkpoint: LSTM_PROVINCE_MODEL_PATH con state_dict + hparams.
    MLflow: run `run_name` en 'climasafeai'.
    """
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

    model = modelo_cls(
        n_features=X_train.shape[2],
        n_provincias=n_provincias,
        emb_dim_provincia=emb_dim_provincia,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout,
        fusion_size=fusion_size,
    ).to(device)

    loss_calor = nn.CrossEntropyLoss(weight=_pesos_de_clase(y_train_calor, device))
    loss_frio = nn.CrossEntropyLoss(weight=_pesos_de_clase(y_train_frio, device))
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    train_dl = DataLoader(
        TensorDataset(
            torch.tensor(X_train),
            torch.tensor(Xp_train),
            torch.tensor(pidx_train),
            torch.tensor(y_train_calor),
            torch.tensor(y_train_frio),
        ),
        batch_size=batch_size,
        shuffle=True,
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
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_rec_riesgo_calor": rr_c,
                "val_rec_riesgo_frio": rr_f,
            })
            mlflow.log_metrics(
                {"train_loss": train_loss, "val_loss": val_loss,
                 "val_rec_riesgo_calor": rr_c, "val_rec_riesgo_frio": rr_f},
                step=epoch,
            )
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
            {
                "state_dict": model.state_dict(),
                "hparams": model.hparams,
                "feature_cols": feature_cols,
            },
            LSTM_PROVINCE_MODEL_PATH,
        )
        print(f"    Modelo guardado -> {LSTM_PROVINCE_MODEL_PATH.name}")

        mlflow.pytorch.log_model(
            model.cpu(),
            name=run_name,
            registered_model_name=f"climasafeai_LSTM_province",
            serialization_format="pickle",
        )

    return model.cpu().eval(), pd.DataFrame(history)


def evaluate_lstm_province(
    model: LSTMProvinceMultiTask | LSTMProvinceAttention,
    X_train_s: np.ndarray,
    Xp_train_s: np.ndarray,
    pidx_train: np.ndarray,
    y_train_calor: np.ndarray,
    y_train_frio: np.ndarray,
    X_test_s: np.ndarray,
    Xp_test_s: np.ndarray,
    pidx_test: np.ndarray,
    y_test_calor: np.ndarray,
    y_test_frio: np.ndarray,
    run_name: str = "LSTM_province_eval",
) -> pd.DataFrame:
    """
    Evalua el modelo con la misma tabla de metricas (Acc/F1/Prec/Rec +
    F1_macro/Rec_riesgo) que evaluate_lstm, con filas
    '{run_name}_calor' / '{run_name}_frio'.
    """
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    model = model.eval()
    with torch.no_grad():
        pred_train = [
            t.argmax(1).numpy()
            for t in model(
                torch.tensor(X_train_s),
                torch.tensor(pidx_train),
                torch.tensor(Xp_train_s),
            )
        ]
        pred_test = [
            t.argmax(1).numpy()
            for t in model(
                torch.tensor(X_test_s),
                torch.tensor(pidx_test),
                torch.tensor(Xp_test_s),
            )
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
    path: str | Path = LSTM_PROVINCE_MODEL_PATH,
    device: str = "cpu",
) -> tuple[LSTMProvinceMultiTask | LSTMProvinceAttention | LSTMProvinceGated, StandardScaler]:
    """Reconstruye el modelo desde el checkpoint."""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model_type = ckpt["hparams"].get("model_type", "concat")
    cls = {"attention": LSTMProvinceAttention, "gated": LSTMProvinceGated}.get(model_type, LSTMProvinceMultiTask)
    hparams = {k: v for k, v in ckpt["hparams"].items() if k not in ("model_type",)}
    model = cls(**hparams)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    scaler_prov = joblib.load(LSTM_PROVINCE_SCALER_PROV_PATH)
    return model, scaler_prov


def main(
    emb_dim_provincia: int = EMBEDDING_DIM_DEFECTO,
    fusion_size: int = 64,
    dropout: float = 0.3,
    lr: float = 1e-3,
    early_stop_metric: str = "val_loss",
    modelo_cls=LSTMProvinceMultiTask,
    tag: str = "LSTM_province",
) -> pd.DataFrame:
    """Pipeline completo: datos -> entrenamiento -> evaluacion."""
    torch.set_num_threads(2)

    splits = preparar_datos()
    run_name = tag

    model, history = train_lstm_province(
        splits["X_train_s"], splits["Xp_train_s"], splits["pidx_train"],
        splits["y_train_calor"], splits["y_train_frio"],
        splits["X_val_s"], splits["Xp_val_s"], splits["pidx_val"],
        splits["y_val_calor"], splits["y_val_frio"],
        n_provincias=splits["n_provincias"],
        emb_dim_provincia=emb_dim_provincia,
        fusion_size=fusion_size,
        dropout=dropout,
        lr=lr,
        early_stop_metric=early_stop_metric,
        modelo_cls=modelo_cls,
        run_name=run_name,
    )

    return evaluate_lstm_province(
        model,
        splits["X_train_s"], splits["Xp_train_s"], splits["pidx_train"],
        splits["y_train_calor"], splits["y_train_frio"],
        splits["X_test_s"], splits["Xp_test_s"], splits["pidx_test"],
        splits["y_test_calor"], splits["y_test_frio"],
        run_name=f"{run_name}_eval",
    )


if __name__ == "__main__":
    main()
