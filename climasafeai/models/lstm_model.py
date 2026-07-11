"""
climasafeai.models.lstm_model — LSTM multi-tarea (calor + frío).

Cuarta estimación del sistema (ver documentacion/diseño_modelo.md §6): una
única red con tronco LSTM compartido y dos cabezas de 3 clases (calor y
frío) entrenada sobre secuencias de 24 horas crudas (ver
climasafeai/data/sequences.py) con los labels de MoMo — aprende la
correlación empírica española sin heredar la calibración americana de las
fórmulas Heat Index / Wind Chill.

Convenciones compartidas con train_model/predict_model:
  - split temporal (nunca aleatorio), pesos de clase por desbalance,
  - métricas F1_macro / Rec_riesgo (recall medio de las clases != 0),
  - MLflow en el experimento 'climasafeai' (configurar_mlflow).
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
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader, TensorDataset

import mlflow
import mlflow.pytorch

from climasafeai.models.predict_model import _plot_confusion_matrix
from climasafeai.models.train_model import configurar_mlflow
from climasafeai.utils.paths import ARTIFACTS_DIR, MODELS_DIR, REPORTS_DIR

LSTM_MODEL_PATH = MODELS_DIR / "LSTM_multitask.pt"
LSTM_REG_MODEL_PATH = MODELS_DIR / "LSTM_multitask_reg.pt"
LSTM_SCALER_PATH = ARTIFACTS_DIR / "scaler_secuencias_lstm.joblib"

SEED = 42  # random_state=42 es la convención del repo


class LSTMMultiTask(nn.Module):
    """
    Tronco LSTM compartido + 2 cabezas lineales de 3 clases (calor, frío).

    El tronco es común porque ambas tareas leen la misma física horaria;
    la multi-tarea actúa además de regularizador. Se clasifica con el último
    estado oculto h_T de la capa superior (resumen de la secuencia del día).
    """

    def __init__(
        self,
        n_features: int = 5,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.3,
        n_clases: int = 3,
        n_salidas_por_cabeza: int | None = None,
    ):
        """`n_salidas_por_cabeza`: 3 (default, clasificación — logits por
        clase) o 1 (regresión — sigmoid, índice continuo 0-1 = percentil de
        mortalidad). Si es None se usa `n_clases` (compatibilidad con
        checkpoints anteriores, que no guardaban esta clave)."""
        super().__init__()
        if n_salidas_por_cabeza is None:
            n_salidas_por_cabeza = n_clases
        self.hparams = {
            "n_features": n_features,
            "hidden_size": hidden_size,
            "num_layers": num_layers,
            "dropout": dropout,
            "n_clases": n_clases,
            "n_salidas_por_cabeza": n_salidas_por_cabeza,
        }
        self.regresion = n_salidas_por_cabeza == 1
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.head_calor = nn.Linear(hidden_size, n_salidas_por_cabeza)
        self.head_frio = nn.Linear(hidden_size, n_salidas_por_cabeza)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # x: (batch, 24, n_features) -> h_n: (num_layers, batch, hidden)
        _, (h_n, _) = self.lstm(x)
        h_t = self.dropout(h_n[-1])  # último estado oculto de la capa superior
        out_calor = self.head_calor(h_t)
        out_frio = self.head_frio(h_t)
        if self.regresion:
            # Índice continuo 0-1 (el target es un percentil)
            return torch.sigmoid(out_calor).squeeze(-1), torch.sigmoid(out_frio).squeeze(-1)
        return out_calor, out_frio


def escalar_secuencias(
    X_train: np.ndarray,
    *X_otros: np.ndarray,
    guardar: bool = True,
) -> tuple:
    """
    StandardScaler por feature sobre las secuencias, ajustado SOLO con train.

    Las secuencias (n, 24, f) se aplanan a (n*24, f) para ajustar/transformar
    — la media/desviación es por variable, común a las 24 horas. Scaler
    PROPIO de la LSTM: el scaler diario existente (scaler_{clase}.joblib) se
    ajustó sobre valores de la hora pico, otra distribución.

    Returns (scaler, X_train_escalado, *X_otros_escalados).
    """
    n_features = X_train.shape[2]
    scaler = StandardScaler()
    scaler.fit(X_train.reshape(-1, n_features))

    def _transform(X: np.ndarray) -> np.ndarray:
        return scaler.transform(X.reshape(-1, n_features)).reshape(X.shape).astype(np.float32)

    if guardar:
        ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
        joblib.dump(scaler, LSTM_SCALER_PATH)
        print(f"    Scaler guardado → {LSTM_SCALER_PATH.name}")

    return (scaler, _transform(X_train), *(_transform(X) for X in X_otros))


def _pesos_de_clase(y: np.ndarray, device: torch.device, n_clases: int = 3) -> torch.Tensor:
    """Pesos 'balanced' por clase — mismo criterio que el sample_weight del
    XGBoost: sin ponderar, la red colapsa a la clase 0 (~90%) e ignora los
    días de riesgo, inservible para un sistema de aviso.

    Devuelve SIEMPRE un vector de `n_clases` (CrossEntropyLoss lo exige
    completo): las clases ausentes en `y` (posible en subsets pequeños,
    p.ej. sin días de peligro por calor fuera del verano) reciben peso 1.0
    — irrelevante, sin muestras no generan gradiente."""
    presentes = np.unique(y)
    w_presentes = compute_class_weight("balanced", classes=presentes, y=y)
    w = np.ones(n_clases, dtype=np.float32)
    w[presentes] = w_presentes
    return torch.tensor(w, dtype=torch.float32, device=device)


def train_lstm(
    X_train: np.ndarray,
    y_train_calor: np.ndarray,
    y_train_frio: np.ndarray,
    X_val: np.ndarray,
    y_val_calor: np.ndarray,
    y_val_frio: np.ndarray,
    hidden_size: int = 64,
    num_layers: int = 2,
    dropout: float = 0.3,
    lr: float = 1e-3,
    batch_size: int = 256,
    max_epochs: int = 50,
    patience: int = 5,
    device: str | None = None,
    run_name: str | None = None,
    feature_cols: list | None = None,
    modo: str = "clasificacion",
) -> tuple[LSTMMultiTask, pd.DataFrame]:
    """
    Entrena la LSTM multi-tarea y devuelve (mejor_modelo, history por época).

    - `modo="clasificacion"` (default): cabezas de 3 clases, pérdida conjunta
      CE(calor, pesos balanced) + CE(frío, pesos balanced) — suma sin
      ponderar, ambas tareas importan igual y sus escalas son comparables.
      Los `y_*` son clases 0/1/2 (int).
    - `modo="regresion"`: cabezas de 1 salida con sigmoid, pérdida conjunta
      MSE(calor) + MSE(frío). Los `y_*` son el percentil continuo de
      mortalidad 0-1 (`y_*_pct` de sequences.py). Sin pesos de clase (el
      percentil ya es ~uniforme por construcción del rank).
    - Early stopping por val loss conjunta (patience épocas sin mejorar);
      se restaura el mejor checkpoint. Las métricas de referencia por época
      (Rec_riesgo en clasificación, MAE en regresión) se loguean pero no
      gobiernan la parada; la comparación FINAL entre modelos sí se hace
      por Rec_riesgo.
    - MLflow: run `run_name` (default 'LSTM' / 'LSTM_reg' según modo) en el
      experimento 'climasafeai'; registro 'climasafeai_LSTM_multitask'
      (+ sufijo '_reg' en regresión).
    - Checkpoint en disco: LSTM_MODEL_PATH / LSTM_REG_MODEL_PATH con
      state_dict + hparams (+ feature_cols) para reconstruir sin hardcodear
      la arquitectura.
    """
    if modo not in ("clasificacion", "regresion"):
        raise ValueError(f"modo debe ser 'clasificacion' o 'regresion', no {modo!r}")
    regresion = modo == "regresion"
    if run_name is None:
        run_name = "LSTM_reg" if regresion else "LSTM"
    checkpoint_path = LSTM_REG_MODEL_PATH if regresion else LSTM_MODEL_PATH

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

    model = LSTMMultiTask(
        n_features=X_train.shape[2],
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout,
        n_salidas_por_cabeza=1 if regresion else 3,
    ).to(device)

    if regresion:
        y_train_calor = np.asarray(y_train_calor, dtype=np.float32)
        y_train_frio = np.asarray(y_train_frio, dtype=np.float32)
        y_val_calor = np.asarray(y_val_calor, dtype=np.float32)
        y_val_frio = np.asarray(y_val_frio, dtype=np.float32)
        loss_calor = nn.MSELoss()
        loss_frio = nn.MSELoss()
    else:
        loss_calor = nn.CrossEntropyLoss(weight=_pesos_de_clase(y_train_calor, device))
        loss_frio = nn.CrossEntropyLoss(weight=_pesos_de_clase(y_train_frio, device))
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    train_dl = DataLoader(
        TensorDataset(
            torch.tensor(X_train),
            torch.tensor(y_train_calor),
            torch.tensor(y_train_frio),
        ),
        batch_size=batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(SEED),
    )
    X_val_t = torch.tensor(X_val, device=device)
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

    print(f"--> Entrenando LSTM multi-tarea en {device} "
          f"({sum(p.numel() for p in model.parameters())} parámetros)...")

    with mlflow.start_run(run_name=run_name):
        mlflow.log_params({
            **model.hparams,
            "lr": lr, "batch_size": batch_size, "max_epochs": max_epochs,
            "patience": patience, "seed": SEED, "modo": modo,
            "task_type": "clasificacion", "model_name": "LSTM", "clase": "multitask",
        })

        for epoch in range(1, max_epochs + 1):
            model.train()
            train_loss = 0.0
            for xb, yb_c, yb_f in train_dl:
                xb = xb.to(device)
                optimizer.zero_grad()
                logits_c, logits_f = model(xb)
                loss = loss_calor(logits_c, yb_c.to(device)) + loss_frio(logits_f, yb_f.to(device))
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * len(xb)
            train_loss /= len(train_dl.dataset)

            model.eval()
            with torch.no_grad():
                out_c, out_f = model(X_val_t)
                val_loss = (
                    loss_calor(out_c, y_val_calor_t) + loss_frio(out_f, y_val_frio_t)
                ).item()
                if regresion:
                    mae_c = (out_c - y_val_calor_t).abs().mean().item()
                    mae_f = (out_f - y_val_frio_t).abs().mean().item()
                else:
                    pred_c = out_c.argmax(1).cpu().numpy()
                    pred_f = out_f.argmax(1).cpu().numpy()

            if regresion:
                metricas_epoca = {"val_mae_calor": mae_c, "val_mae_frio": mae_f}
                detalle = f"MAE val calor {mae_c:.4f} frío {mae_f:.4f}"
            else:
                metricas_epoca = {
                    "val_rec_riesgo_calor": _rec_riesgo(y_val_calor, pred_c),
                    "val_rec_riesgo_frio": _rec_riesgo(y_val_frio, pred_f),
                }
                detalle = (
                    f"Rec_riesgo val calor {metricas_epoca['val_rec_riesgo_calor']:.3f} "
                    f"frío {metricas_epoca['val_rec_riesgo_frio']:.3f}"
                )

            history.append({
                "epoch": epoch, "train_loss": train_loss, "val_loss": val_loss,
                **metricas_epoca,
            })
            mlflow.log_metrics(
                {"train_loss": train_loss, "val_loss": val_loss, **metricas_epoca},
                step=epoch,
            )
            print(f"    época {epoch:3d} | train_loss {train_loss:.4f} | "
                  f"val_loss {val_loss:.4f} | {detalle}")

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
            },
            checkpoint_path,
        )
        print(f"    Modelo guardado → {checkpoint_path.name}")

        mlflow.pytorch.log_model(
            model.cpu(),
            name="LSTM",
            registered_model_name="climasafeai_LSTM_multitask" + ("_reg" if regresion else ""),
            # El formato por defecto de MLflow 3.x ('pt2') traza el grafo y
            # no soporta modelos con salida tupla (dos cabezas) — pickle sí,
            # y es coherente con el CLOUDPICKLE usado para los sklearn.
            serialization_format="pickle",
        )

    return model.cpu().eval(), pd.DataFrame(history)


def evaluate_lstm(
    model: LSTMMultiTask,
    X_train_s: np.ndarray,
    y_train_calor: np.ndarray,
    y_train_frio: np.ndarray,
    X_test_s: np.ndarray,
    y_test_calor: np.ndarray,
    y_test_frio: np.ndarray,
) -> pd.DataFrame:
    """
    Evalúa cada cabeza por separado con el MISMO esquema que evaluate_models
    (filas 'LSTM_calor' / 'LSTM_frio'; columnas Acc/F1/Prec/Rec + F1_macro y
    Rec_riesgo, la métrica de selección del proyecto). Guarda las matrices de
    confusión en FIGURES_DIR y el resumen en reports/resultados_lstm.csv
    (fichero propio — evaluate_models sobrescribe resultados_modelos.csv y no
    hay que pisarlo). Loguea a MLflow en un run 'LSTM_eval'.
    """
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    model = model.eval()
    with torch.no_grad():
        pred_train = [t.argmax(1).numpy() for t in model(torch.tensor(X_train_s))]
        pred_test = [t.argmax(1).numpy() for t in model(torch.tensor(X_test_s))]

    cabezas = [
        ("LSTM_calor", y_train_calor, y_test_calor, pred_train[0], pred_test[0]),
        ("LSTM_frio", y_train_frio, y_test_frio, pred_train[1], pred_test[1]),
    ]

    configurar_mlflow()
    mlflow.end_run()

    results = []
    with mlflow.start_run(run_name="LSTM_eval"):
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

            sufijo = name.split("_")[1]  # calor | frio
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
    out_csv = REPORTS_DIR / "resultados_lstm.csv"
    df_results.to_csv(out_csv, index=False)
    print(f"\n  Resumen:\n{df_results.to_string(index=False)}")
    print(f"\n  Guardado → {out_csv}")
    return df_results


def indice_riesgo_softmax(
    model: LSTMMultiTask,
    X_s: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Índice continuo de peligrosidad 0-1 del CLASIFICADOR ya entrenado:
    softmax de cada cabeza -> P(riesgo) = 1 - P(clase 0 'seguro').

    La clase que devuelve predict es solo el argmax de estas probabilidades;
    esto expone la información graduada que la red ya calcula, sin
    reentrenar nada. X_s debe venir YA escalado (mismo scaler del modelo).
    """
    if model.regresion:
        raise ValueError("indice_riesgo_softmax es para el modelo de clasificación")
    model = model.eval()
    with torch.no_grad():
        logits_c, logits_f = model(torch.tensor(X_s))
        idx_c = 1.0 - torch.softmax(logits_c, dim=1)[:, 0]
        idx_f = 1.0 - torch.softmax(logits_f, dim=1)[:, 0]
    return idx_c.numpy(), idx_f.numpy()


def evaluate_lstm_regresion(
    model: LSTMMultiTask,
    X_test_s: np.ndarray,
    y_test_calor_pct: np.ndarray,
    y_test_frio_pct: np.ndarray,
    y_test_calor: np.ndarray,
    y_test_frio: np.ndarray,
    q_precaucion: float = 0.75,
    q_peligro: float = 0.95,
) -> pd.DataFrame:
    """
    Evalúa la LSTM de regresión en dos planos:

    1. Como REGRESIÓN: MAE y Spearman entre el percentil predicho y el real
       (Spearman porque lo que importa del índice es que ORDENE bien los
       días por peligrosidad, no su valor absoluto).
    2. Como CLASIFICADOR recuperado: percentil predicho -> clases con los
       mismos cortes p75/p95 de labels.py -> misma tabla de métricas que
       evaluate_lstm (filas 'LSTM_reg_calor'/'LSTM_reg_frio'), comparable
       1:1 con la clasificación y los modelos desplegados.

    Guarda matrices de confusión cm_LSTM_reg_{calor,frio}.png y el resumen
    en reports/resultados_lstm_reg.csv. Loguea al run MLflow 'LSTM_reg_eval'.
    """
    from scipy.stats import spearmanr

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    model = model.eval()
    with torch.no_grad():
        pred_c, pred_f = [t.numpy() for t in model(torch.tensor(X_test_s))]

    def _a_clase(pct: np.ndarray) -> np.ndarray:
        return np.where(pct <= q_precaucion, 0, np.where(pct <= q_peligro, 1, 2))

    cabezas = [
        ("LSTM_reg_calor", y_test_calor_pct, y_test_calor, pred_c),
        ("LSTM_reg_frio", y_test_frio_pct, y_test_frio, pred_f),
    ]

    configurar_mlflow()
    mlflow.end_run()

    results = []
    with mlflow.start_run(run_name="LSTM_reg_eval"):
        for name, y_pct, y_clase, pred_pct in cabezas:
            print(f"\n--- {name} ---")
            mae = float(np.abs(pred_pct - y_pct).mean())
            rho = float(spearmanr(pred_pct, y_pct).statistic)
            print(f"  Regresión → MAE: {mae:.4f} | Spearman: {rho:.4f}")

            p_clase = _a_clase(pred_pct)
            acc_test = accuracy_score(y_clase, p_clase)
            f1_test = f1_score(y_clase, p_clase, average="weighted", zero_division=0)
            prec_test = precision_score(y_clase, p_clase, average="weighted", zero_division=0)
            rec_test = recall_score(y_clase, p_clase, average="weighted", zero_division=0)
            f1_macro = f1_score(y_clase, p_clase, average="macro", zero_division=0)
            risk_labels = [c for c in np.unique(y_clase) if c != 0]
            rec_riesgo = (
                recall_score(y_clase, p_clase, labels=risk_labels, average="macro", zero_division=0)
                if risk_labels else float("nan")
            )
            print(f"  Clases (p{q_precaucion:.0%}/p{q_peligro:.0%}) → "
                  f"Acc: {acc_test:.3f} | F1_macro: {f1_macro:.3f} | Rec_riesgo: {rec_riesgo:.3f}")
            print()
            print(classification_report(y_clase, p_clase, zero_division=0))
            _plot_confusion_matrix(y_clase, p_clase, name)

            sufijo = name.split("_")[-1]  # calor | frio
            mlflow.log_metrics({
                f"mae_{sufijo}": mae, f"spearman_{sufijo}": rho,
                f"acc_test_{sufijo}": acc_test, f"f1_macro_{sufijo}": f1_macro,
                f"rec_riesgo_{sufijo}": rec_riesgo,
            })

            results.append({
                "Modelo": name,
                "MAE": round(mae, 4), "Spearman": round(rho, 4),
                "Acc_test": round(acc_test, 4), "F1_test": round(f1_test, 4),
                "Prec_test": round(prec_test, 4), "Rec_test": round(rec_test, 4),
                "F1_macro": round(f1_macro, 4), "Rec_riesgo": round(rec_riesgo, 4),
            })

    df_results = pd.DataFrame(results)
    out_csv = REPORTS_DIR / "resultados_lstm_reg.csv"
    df_results.to_csv(out_csv, index=False)
    print(f"\n  Resumen:\n{df_results.to_string(index=False)}")
    print(f"\n  Guardado → {out_csv}")
    return df_results


def load_lstm(
    path=LSTM_MODEL_PATH,
    device: str = "cpu",
) -> tuple[LSTMMultiTask, StandardScaler]:
    """Reconstruye el modelo desde el checkpoint (hparams incluidos) y carga
    su scaler. -> (model en eval(), scaler)."""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = LSTMMultiTask(**ckpt["hparams"])
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    scaler = joblib.load(LSTM_SCALER_PATH)
    return model, scaler


def predict_lstm(
    model: LSTMMultiTask,
    scaler: StandardScaler,
    X: np.ndarray,
    q_precaucion: float = 0.75,
    q_peligro: float = 0.95,
) -> tuple[np.ndarray, np.ndarray]:
    """X SIN escalar (n, 24, f) -> (pred_calor, pred_frio), clases 0/1/2.

    Con un modelo de regresión, el percentil predicho se convierte a clase
    con los mismos cortes p75/p95 de labels.py."""
    n_features = X.shape[2]
    X_s = scaler.transform(X.reshape(-1, n_features)).reshape(X.shape).astype(np.float32)
    model = model.eval()
    with torch.no_grad():
        out_c, out_f = model(torch.tensor(X_s))
    if model.regresion:
        def _a_clase(pct: torch.Tensor) -> np.ndarray:
            pct = pct.numpy()
            return np.where(pct <= q_precaucion, 0, np.where(pct <= q_peligro, 1, 2))
        return _a_clase(out_c), _a_clase(out_f)
    return out_c.argmax(1).numpy(), out_f.argmax(1).numpy()
