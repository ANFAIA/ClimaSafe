"""
climasafeai.models.lstm_hybrid — LSTM híbrida: secuencia 24h + features diarias.

Variante de la LSTM multi-tarea (lstm_model.py) que añade CONTEXTO DE OLA:
la LSTM pura solo ve las 24 horas del día y no puede saber que es el 5º día
consecutivo de una ola de calor/frío — exactamente lo que capturan las
features diarias de persistencia del pipeline tabular (dias_consec_*,
grados_dia_*_roll7/14, *_roll3/7/14...). Aquí el tronco LSTM resume la
secuencia horaria y su último estado oculto se CONCATENA con el vector de
features diarias (escalado con un StandardScaler ajustado solo con train)
antes de una capa de fusión y las dos cabezas de clasificación.

Convenciones idénticas a lstm_model.py: split temporal, pesos de clase
'balanced', early stopping por val loss, métricas F1_macro / Rec_riesgo,
MLflow en el experimento 'climasafeai'. Checkpoint propio: LSTM_hybrid.pt.
"""

from __future__ import annotations

import copy

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
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

from climasafeai.models.lstm_model import (
    SEED,
    _pesos_de_clase,
    escalar_secuencias,
)
from climasafeai.models.predict_model import _plot_confusion_matrix
from climasafeai.models.train_model import configurar_mlflow
from climasafeai.utils.paths import (
    ARTIFACTS_DIR,
    MODELS_DIR,
    PROCESSED_DATA_DIR,
    REPORTS_DIR,
)

LSTM_HYBRID_MODEL_PATH = MODELS_DIR / "LSTM_hybrid.pt"
LSTM_HYBRID_SCALER_DIARIAS_PATH = ARTIFACTS_DIR / "scaler_diarias_lstm_hybrid.joblib"

# Las features tabulares diarias de dataset_calor_labeled.parquet (mismo
# conjunto que ven XGBoost/RF, sin fecha/provincia/target/label). Incluyen
# las de persistencia que motivan la variante híbrida: dias_consec_*,
# grados_dia_*_roll7/14, heat_index_c_roll3/7, wind_chill_mean_roll3/7/14,
# más nocturnas y rachas severas (2026-07-14).
DAILY_FEATURE_COLS: list = [
    "t2m_c", "rh", "wind_speed_kmh", "sp",
    "heat_index_c", "wbgt_c", "wind_chill_c",
    "heat_index_mean", "heat_index_std", "heat_index_min", "horas_sobre_umbral",
    "wind_chill_mean", "wind_chill_std", "wind_chill_max", "horas_bajo_umbral",
    "heat_index_c_lag1", "heat_index_c_roll3", "heat_index_c_roll7",
    "dias_consec_sobre_umbral", "grados_dia_calor_roll7", "grados_dia_calor_roll14",
    "wind_chill_mean_roll3", "wind_chill_mean_roll7", "wind_chill_mean_roll14",
    "grados_dia_frio_roll7", "grados_dia_frio_roll14", "dias_consec_bajo_umbral",
    # Nocturnas y rachas severas (2026-07-14)
    "t2m_min_noche_lag1", "t2m_min_noche_roll7",
    "dias_consec_wc_severo", "horas_wc_severo_sum14",
]

DAILY_PARQUET_PATH = PROCESSED_DATA_DIR / "dataset_calor_labeled.parquet"


class LSTMHybridMultiTask(nn.Module):
    """
    Tronco LSTM (secuencia 24h) + vector de features diarias -> fusión -> 2 cabezas.

    El último estado oculto h_T de la capa superior (resumen intradía) se
    concatena con las features diarias ya escaladas; una capa de fusión
    Linear+ReLU+Dropout mezcla ambos mundos antes de las cabezas de 3 clases
    (calor, frío). Sin la fusión, cada cabeza solo podría combinar los dos
    bloques linealmente.
    """

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
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
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

    def forward(
        self, x_seq: torch.Tensor, x_dia: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # x_seq: (batch, 24, n_features) | x_dia: (batch, n_features_diarias)
        _, (h_n, _) = self.lstm(x_seq)
        h_t = self.dropout(h_n[-1])  # último estado oculto de la capa superior
        z = self.fusion(torch.cat([h_t, x_dia], dim=1))
        return self.head_calor(z), self.head_frio(z)


def alinear_features_diarias(
    fechas: np.ndarray,
    provincias: np.ndarray,
    cols: list = DAILY_FEATURE_COLS,
    parquet_path=DAILY_PARQUET_PATH,
) -> np.ndarray:
    """
    Alinea el vector diario con cada secuencia por (fecha, provincia).

    `fechas`/`provincias` son los arrays del npz de secuencias (mismo orden
    que X). Merge left VALIDADO: si alguna secuencia no encuentra su fila
    diaria se lanza error (no se imputa en silencio) — con los datasets
    actuales el match es 172350/172350.

    Los NaN residuales de las features rolling (solo el primer día del
    histórico, 2016-01-01) se dejan pasar: `escalar_diarias` los convierte
    en la media de train (0 tras estandarizar), sin fuga train->test.

    Returns: np.ndarray float32 (n_secuencias, len(cols)).
    """
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
            f"{sin_match} secuencias sin fila diaria en {parquet_path.name} "
            f"— revisa el alineamiento fecha/provincia"
        )
    n_nan = int(m[cols].isna().any(axis=1).sum())
    if n_nan:
        print(f"    Features diarias: {n_nan} filas con NaN (rollings del "
              f"arranque del histórico) — se imputarán a la media de train")
    return m[cols].to_numpy(dtype=np.float32)


def escalar_diarias(
    Xd_train: np.ndarray,
    *Xd_otros: np.ndarray,
    guardar: bool = True,
) -> tuple:
    """
    StandardScaler sobre las features diarias, ajustado SOLO con train.

    Scaler PROPIO de la híbrida (no el scaler_{clase}.joblib del pipeline
    tabular: aquel se ajusta tras el preprocess de build_features, con otro
    orden de columnas y su propio split). Los NaN (rollings del primer día
    del histórico) se imputan a 0 TRAS estandarizar = media de train.

    Returns (scaler, Xd_train_escalado, *Xd_otros_escalados).
    """
    scaler = StandardScaler()
    scaler.fit(Xd_train[~np.isnan(Xd_train).any(axis=1)])

    def _transform(Xd: np.ndarray) -> np.ndarray:
        Xd_s = scaler.transform(Xd).astype(np.float32)
        return np.nan_to_num(Xd_s, nan=0.0)

    if guardar:
        ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
        joblib.dump(scaler, LSTM_HYBRID_SCALER_DIARIAS_PATH)
        print(f"    Scaler diarias guardado → {LSTM_HYBRID_SCALER_DIARIAS_PATH.name}")

    return (scaler, _transform(Xd_train), *(_transform(Xd) for Xd in Xd_otros))


def train_lstm_hybrid(
    X_train: np.ndarray,
    Xd_train: np.ndarray,
    y_train_calor: np.ndarray,
    y_train_frio: np.ndarray,
    X_val: np.ndarray,
    Xd_val: np.ndarray,
    y_val_calor: np.ndarray,
    y_val_frio: np.ndarray,
    hidden_size: int = 64,
    num_layers: int = 2,
    dropout: float = 0.3,
    fusion_size: int = 64,
    lr: float = 1e-3,
    batch_size: int = 256,
    max_epochs: int = 50,
    patience: int = 5,
    device: str | None = None,
    run_name: str = "LSTM_hybrid",
    feature_cols: list | None = None,
    daily_feature_cols: list | None = None,
    peso_riesgo_extra: float = 1.0,
) -> tuple[LSTMHybridMultiTask, pd.DataFrame]:
    """
    Entrena la LSTM híbrida (clasificación multi-tarea) y devuelve
    (mejor_modelo, history por época). Mismo protocolo que train_lstm:

    - Pérdida conjunta CE(calor, pesos balanced) + CE(frío, pesos balanced).
    - `peso_riesgo_extra`: multiplicador adicional sobre los pesos de las
      clases de riesgo (1 y 2) — 1.0 reproduce el 'balanced' exacto de la
      LSTM base; >1 empuja el recall de riesgo a costa de la clase 0.
    - Early stopping por val loss conjunta (patience épocas); se restaura
      el mejor checkpoint. Rec_riesgo por época se loguea como referencia.
    - Checkpoint LSTM_HYBRID_MODEL_PATH con state_dict + hparams
      (+ feature_cols de secuencia y diarias).
    - MLflow: run `run_name` en 'climasafeai', registro
      'climasafeai_LSTM_hybrid'.

    Las entradas deben venir YA escaladas (escalar_secuencias para X,
    escalar_diarias para Xd).
    """
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

    model = LSTMHybridMultiTask(
        n_features=X_train.shape[2],
        n_features_diarias=Xd_train.shape[1],
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout,
        fusion_size=fusion_size,
    ).to(device)

    def _pesos(y: np.ndarray) -> torch.Tensor:
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
            torch.tensor(X_train),
            torch.tensor(Xd_train),
            torch.tensor(y_train_calor),
            torch.tensor(y_train_frio),
        ),
        batch_size=batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(SEED),
    )
    X_val_t = torch.tensor(X_val, device=device)
    Xd_val_t = torch.tensor(Xd_val, device=device)
    y_val_calor_t = torch.tensor(y_val_calor, device=device)
    y_val_frio_t = torch.tensor(y_val_frio, device=device)

    def _rec_riesgo(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        risk = [c for c in np.unique(y_true) if c != 0]
        if not risk:
            return float("nan")
        return recall_score(y_true, y_pred, labels=risk, average="macro", zero_division=0)

    configurar_mlflow()
    mlflow.end_run()  # cerrar cualquier run activo antes de abrir uno nuevo

    history = []
    best_val_loss = float("inf")
    best_state = None
    sin_mejora = 0

    print(f"--> Entrenando LSTM híbrida en {device} "
          f"({sum(p.numel() for p in model.parameters())} parámetros)...")

    with mlflow.start_run(run_name=run_name):
        mlflow.log_params({
            **model.hparams,
            "lr": lr, "batch_size": batch_size, "max_epochs": max_epochs,
            "patience": patience, "seed": SEED,
            "peso_riesgo_extra": peso_riesgo_extra,
            "task_type": "clasificacion", "model_name": "LSTM_hybrid",
            "clase": "multitask",
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
            mlflow.log_metrics(
                {"train_loss": train_loss, "val_loss": val_loss, **metricas_epoca},
                step=epoch,
            )
            print(f"    época {epoch:3d} | train_loss {train_loss:.4f} | "
                  f"val_loss {val_loss:.4f} | Rec_riesgo val calor "
                  f"{metricas_epoca['val_rec_riesgo_calor']:.3f} "
                  f"frío {metricas_epoca['val_rec_riesgo_frio']:.3f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = copy.deepcopy(model.state_dict())
                sin_mejora = 0
            else:
                sin_mejora += 1
                if sin_mejora >= patience:
                    print(f"    Early stopping en época {epoch} "
                          f"(mejor val_loss {best_val_loss:.4f})")
                    break

        model.load_state_dict(best_state)
        mlflow.log_metric("best_val_loss", best_val_loss)

        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "state_dict": model.state_dict(),
                "hparams": model.hparams,
                "feature_cols": feature_cols,
                "daily_feature_cols": daily_feature_cols or DAILY_FEATURE_COLS,
            },
            LSTM_HYBRID_MODEL_PATH,
        )
        print(f"    Modelo guardado → {LSTM_HYBRID_MODEL_PATH.name}")

        mlflow.pytorch.log_model(
            model.cpu(),
            name="LSTM_hybrid",
            registered_model_name="climasafeai_LSTM_hybrid",
            # pickle: el formato 'pt2' por defecto de MLflow 3.x no soporta
            # salida tupla (dos cabezas) — mismo criterio que lstm_model.
            serialization_format="pickle",
        )

    return model.cpu().eval(), pd.DataFrame(history)


def evaluate_lstm_hybrid(
    model: LSTMHybridMultiTask,
    X_train_s: np.ndarray,
    Xd_train_s: np.ndarray,
    y_train_calor: np.ndarray,
    y_train_frio: np.ndarray,
    X_test_s: np.ndarray,
    Xd_test_s: np.ndarray,
    y_test_calor: np.ndarray,
    y_test_frio: np.ndarray,
    run_name: str = "LSTM_hybrid_eval",
) -> pd.DataFrame:
    """
    Misma tabla de métricas que evaluate_lstm (Acc/F1/Prec/Rec + F1_macro y
    Rec_riesgo) con filas 'LSTM_hybrid_calor' / 'LSTM_hybrid_frio', comparable
    1:1 con reports/resultados_lstm.csv. Guarda matrices de confusión en
    FIGURES_DIR y el resumen en reports/resultados_lstm_hybrid.csv. Loguea a
    MLflow en un run `run_name`.
    """
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

            print(f"  Accuracy  → train: {acc_train:.3f} | test: {acc_test:.3f}")
            print(f"  F1 (w)    → train: {f1_train:.3f}  | test: {f1_test:.3f}")
            print(f"  F1_macro  → {f1_macro:.3f}  | Rec_riesgo (clases 1..n) → {rec_riesgo:.3f}")
            print()
            print(classification_report(y_te, p_te, zero_division=0))
            _plot_confusion_matrix(y_te, p_te, name)

            sufijo = name.split("_")[-1]  # calor | frio
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
    print(f"\n  Guardado → {out_csv}")
    return df_results


def load_lstm_hybrid(
    path=LSTM_HYBRID_MODEL_PATH,
    device: str = "cpu",
) -> tuple[LSTMHybridMultiTask, StandardScaler]:
    """Reconstruye la híbrida desde el checkpoint (hparams incluidos) y carga
    su scaler de features diarias. El scaler de secuencias es el compartido
    de la LSTM (scaler_secuencias_lstm.joblib, ver lstm_model.load_lstm).
    -> (model en eval(), scaler_diarias)."""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = LSTMHybridMultiTask(**ckpt["hparams"])
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    scaler_diarias = joblib.load(LSTM_HYBRID_SCALER_DIARIAS_PATH)
    return model, scaler_diarias


def main(peso_riesgo_extra: float = 1.0, run_name: str = "LSTM_hybrid") -> pd.DataFrame:
    """Pipeline completo reproducible: npz de secuencias + parquet diario ->
    split temporal -> escalado (solo train) -> entrenamiento -> evaluación."""
    from climasafeai.data.sequences import (
        cargar_dataset_secuencias,
        split_secuencias_por_fecha,
    )

    torch.set_num_threads(2)  # CPU compartida con otros procesos

    data = cargar_dataset_secuencias()
    Xd = alinear_features_diarias(data["fechas"], data["provincias"])

    splits = split_secuencias_por_fecha(data)
    # Reproducir las mismas máscaras del split para las features diarias
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
        X_train_s, Xd_train_s,
        splits["y_train_calor"], splits["y_train_frio"],
        X_val_s, Xd_val_s,
        splits["y_val_calor"], splits["y_val_frio"],
        feature_cols=data["feature_cols"],
        peso_riesgo_extra=peso_riesgo_extra,
        run_name=run_name,
    )

    return evaluate_lstm_hybrid(
        model,
        X_train_s, Xd_train_s,
        splits["y_train_calor"], splits["y_train_frio"],
        X_test_s, Xd_test_s,
        splits["y_test_calor"], splits["y_test_frio"],
        run_name=f"{run_name}_eval",
    )


if __name__ == "__main__":
    main()
