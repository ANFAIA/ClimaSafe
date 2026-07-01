
"""
predict_model.py — Evaluación de modelos supervisado.
Tarea: clasificacion
"""
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    roc_auc_score, confusion_matrix, classification_report,
    ConfusionMatrixDisplay,
)

import mlflow
import mlflow.sklearn

import shap
from climasafeai.utils.paths import FIGURES_DIR, MODELS_DIR, REPORTS_DIR

# Umbral de decisión. Bajar (e.g. 0.3) aumenta recall de clase minoritaria.
DECISION_THRESHOLD: float = 0.5

def evaluate_models(
    models: dict,
    X_train,
    y_train,
    X_test,
    y_test,
    threshold: float = DECISION_THRESHOLD,
) -> pd.DataFrame:
    """
    Evalúa todos los modelos sobre train y test.
    Métricas: Accuracy, F1 weighted, Precision, Recall, ROC-AUC (binario).
    Genera matrices de confusión en figures/.
    Returns
    -------
    pd.DataFrame ordenado por métrica principal.
    """
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}\n  Evaluación — clasificacion (umbral={threshold})\n{'='*60}")

    mlflow.set_experiment("climasafeai")

    results = []
    for name, model in models.items():
        print(f"\n--- {name} ---")

        mlflow.end_run()  # cerrar run activo antes de abrir uno nuevo

        if threshold != 0.5 and hasattr(model, "predict_proba"):
            proba_test   = model.predict_proba(X_test)[:, 1]
            y_pred_test  = (proba_test >= threshold).astype(int)
            proba_train  = model.predict_proba(X_train)[:, 1]
            y_pred_train = (proba_train >= threshold).astype(int)
        else:
            y_pred_test  = model.predict(X_test)
            y_pred_train = model.predict(X_train)

        acc_train = accuracy_score(y_train, y_pred_train)
        acc_test  = accuracy_score(y_test,  y_pred_test)
        f1_train  = f1_score(y_train, y_pred_train, average="weighted", zero_division=0)
        f1_test   = f1_score(y_test,  y_pred_test,  average="weighted", zero_division=0)
        prec_test = precision_score(y_test, y_pred_test, average="weighted", zero_division=0)
        rec_test  = recall_score(y_test,  y_pred_test,  average="weighted", zero_division=0)
        roc_auc   = None
        if hasattr(model, "predict_proba") and len(np.unique(y_test)) == 2:
            roc_auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])

        print(f"  Accuracy  → train: {acc_train:.3f} | test: {acc_test:.3f}")
        print(f"  F1 (w)    → train: {f1_train:.3f}  | test: {f1_test:.3f}")
        print(f"  Precision → {prec_test:.3f}  | Recall → {rec_test:.3f}")
        if roc_auc is not None:
            print(f"  ROC-AUC   → {roc_auc:.3f}")
        print()
        print(classification_report(y_test, y_pred_test, zero_division=0))
        _plot_confusion_matrix(y_test, y_pred_test, name)

        row = {
            "Modelo":    name,
            "Acc_train": round(acc_train, 4), "Acc_test":  round(acc_test,  4),
            "F1_train":  round(f1_train,  4), "F1_test":   round(f1_test,   4),
            "Prec_test": round(prec_test, 4), "Rec_test":  round(rec_test,  4),
        }
        if roc_auc is not None:
            row["ROC_AUC"] = round(roc_auc, 4)

        with mlflow.start_run(run_name=f"{name}_eval"):
            mlflow.log_metrics({
                "acc_train": acc_train, "acc_test": acc_test,
                "f1_train":  f1_train,  "f1_test":  f1_test,
                "prec_test": prec_test, "rec_test":  rec_test,
            })
            if roc_auc is not None:
                mlflow.log_metric("roc_auc", roc_auc)
            mlflow.log_artifact(str(FIGURES_DIR / f"cm_{name}.png"))
        results.append(row)

    df_results = pd.DataFrame(results).sort_values("Acc_test", ascending=False)

    out_csv = REPORTS_DIR / "resultados_modelos.csv"
    df_results.to_csv(out_csv, index=False)
    print(f"\n{'='*60}\n  Resumen:\n{'='*60}")
    print(df_results.to_string(index=False))
    print(f"\n  Guardado → {out_csv}")
    return df_results

def _plot_confusion_matrix(y_true, y_pred, model_name: str) -> None:
    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
    cm   = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    fig, ax = plt.subplots(figsize=(7, 6))
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title(f"Matriz de confusion — {model_name}", fontsize=13)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / f"cm_{model_name}.png", dpi=150)
    plt.close(fig)
    print(f"    cm_{model_name}.png guardado")

def explain_models(
    models: dict,
    X_train,
    feature_names: list = None,
    max_display: int = 15,
    kernel_background: int = 50,
    kernel_max_samples: int = 100,
) -> None:
    """
    Genera explicaciones SHAP para cada modelo entrenado.

    Por cada modelo produce dos gráficas en reports/figures/:
      - shap_bar_{nombre}.png   → importancia global media (resumen ejecutivo)
      - shap_beeswarm_{nombre}.png → distribución + dirección del impacto

    Selección de explainer por tipo de modelo:
      TreeExplainer   → RandomForest, DecisionTree, XGBoost, LightGBM  (exacto, rápido)
      LinearExplainer → LogisticRegression, Ridge, Lasso                (exacto, rápido)
      KernelExplainer → KNN y otros sin soporte nativo                  (aprox., lento)

    Parameters
    ----------
    models            : dict nombre→modelo (salida de train_models)
    X_train           : datos de entrenamiento ya preprocesados
    feature_names     : lista de nombres de features (opcional)
    max_display       : nº máximo de features a mostrar en las gráficas
    kernel_background : nº de muestras de fondo para KernelExplainer
    kernel_max_samples: nº máximo de filas a explicar con KernelExplainer
    """
    import warnings
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    if hasattr(X_train, "values"):
        X_arr = X_train.values
        feat_names = feature_names or list(X_train.columns)
    else:
        X_arr = X_train
        feat_names = feature_names or [f"feature_{i}" for i in range(X_arr.shape[1])]

    print(f"\n{'='*60}\n  SHAP — Explicabilidad de modelos\n{'='*60}")

    for name, model in models.items():
        print(f"\n--- {name} ---")
        try:
            shap_values, X_explain = _compute_shap(
                model, X_arr, feat_names,
                kernel_background=kernel_background,
                kernel_max_samples=kernel_max_samples,
            )
        except Exception as exc:
            print(f"    ⚠ SHAP no disponible para {name}: {exc}")
            continue

        _shap_bar(shap_values, X_explain, feat_names, name, max_display)
        _shap_beeswarm(shap_values, X_explain, feat_names, name, max_display)

    print(f"\n  Gráficas SHAP guardadas en {FIGURES_DIR}")


def _compute_shap(model, X_arr, feat_names, kernel_background, kernel_max_samples):
    """Selecciona el explainer adecuado y devuelve (shap_values, X_explain)."""
    module = type(model).__module__

    is_tree = (
        hasattr(model, "estimators_")       # RandomForest
        or hasattr(model, "tree_")          # DecisionTree
        or "xgboost" in module
        or "lightgbm" in module
    )
    is_linear = hasattr(model, "coef_")     # LogisticRegression, Ridge, Lasso

    if is_tree:
        explainer  = shap.TreeExplainer(model)
        shap_vals  = explainer.shap_values(X_arr)
        X_explain  = X_arr

    elif is_linear:
        explainer  = shap.LinearExplainer(model, X_arr)
        shap_vals  = explainer.shap_values(X_arr)
        X_explain  = X_arr

    else:
        # KNN y otros → KernelExplainer (lento)
        n_bg = min(kernel_background, len(X_arr))
        bg   = shap.sample(X_arr, n_bg)
        fn   = model.predict_proba if hasattr(model, "predict_proba") else model.predict
        explainer  = shap.KernelExplainer(fn, bg)
        n_exp      = min(kernel_max_samples, len(X_arr))
        X_explain  = X_arr[:n_exp]
        print(f"    KernelExplainer: {n_bg} muestras fondo, "
              f"{n_exp} muestras a explicar (puede tardar...)")
        shap_vals = explainer.shap_values(X_explain)

    # RandomForest y multiclase devuelven lista — tomamos clase positiva (binario)
    # o la media absoluta entre clases (multiclase)
    if isinstance(shap_vals, list):
        if len(shap_vals) == 2:
            shap_vals = shap_vals[1]            # clase positiva binaria
        else:
            import numpy as _np
            shap_vals = _np.abs(_np.stack(shap_vals)).mean(axis=0)  # media multiclase

    return shap_vals, X_explain


def _shap_bar(shap_values, X_explain, feat_names, model_name, max_display):
    """Barra de importancia global media (|SHAP|)."""
    fig, ax = plt.subplots(figsize=(9, max(4, min(max_display, len(feat_names)) * 0.4 + 1)))
    shap.summary_plot(
        shap_values, X_explain,
        feature_names=feat_names,
        plot_type="bar",
        max_display=max_display,
        show=False,
        plot_size=None,
    )
    plt.title(f"SHAP — Importancia global ({model_name})", fontsize=12, pad=10)
    plt.tight_layout()
    path = FIGURES_DIR / f"shap_bar_{model_name}.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"    shap_bar_{model_name}.png guardado")


def _shap_beeswarm(shap_values, X_explain, feat_names, model_name, max_display):
    """Beeswarm: distribución de valores SHAP por feature (dirección + magnitud)."""
    fig, ax = plt.subplots(figsize=(10, max(4, min(max_display, len(feat_names)) * 0.4 + 1)))
    shap.summary_plot(
        shap_values, X_explain,
        feature_names=feat_names,
        plot_type="dot",        # beeswarm
        max_display=max_display,
        show=False,
        plot_size=None,
    )
    plt.title(f"SHAP — Beeswarm ({model_name})", fontsize=12, pad=10)
    plt.tight_layout()
    path = FIGURES_DIR / f"shap_beeswarm_{model_name}.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"    shap_beeswarm_{model_name}.png guardado")



def predict_new(model_name: str, X_new) -> np.ndarray:
    """Carga un modelo y predice sobre nuevas muestras (ya preprocesadas)."""
    path = MODELS_DIR / f"{model_name}.joblib"
    if not path.exists():
        raise FileNotFoundError(f"Modelo no encontrado: {path}")
    return joblib.load(path).predict(X_new)



def predict_proba_new(model_name: str, X_new) -> np.ndarray:
    """Carga un modelo y devuelve probabilidades de clase."""
    path = MODELS_DIR / f"{model_name}.joblib"
    if not path.exists():
        raise FileNotFoundError(f"Modelo no encontrado: {path}")
    model = joblib.load(path)
    if not hasattr(model, "predict_proba"):
        raise ValueError(f"{model_name} no soporta predict_proba")
    return model.predict_proba(X_new)


def try_model() -> None:
    """
    Modo interactivo: introduce tus datos y el modelo predice el resultado.

    Flujo:
      1. Lista los modelos disponibles en models/ y pide elegir uno.
      2. Carga el modelo y los artefactos de preprocesado (scaler, PCA,
         encoders, threshold) guardados durante el entrenamiento.
      3. Pide al usuario los valores de cada feature por consola.
      4. Preprocesa la entrada con process_input() de build_features.
      5. Imprime la predicción (y probabilidad si está disponible).

    Requisitos previos:
      - Haber ejecutado run_full_pipeline() al menos una vez para que
        existan los joblibs en artifacts/ y los modelos en models/.
    """
    from climasafeai.features.build_features import process_input
    from climasafeai.utils.paths import ARTIFACTS_DIR, PROCESSED_DATA_DIR
    import pandas as pd

    # ── 1. Elegir modelo ────────────────────────────────────────────────
    available = sorted(MODELS_DIR.glob("*.joblib"))
    if not available:
        print("No hay modelos entrenados en models/. Ejecuta primero la opción 0.")
        return

    print("\nModelos disponibles:")
    for i, p in enumerate(available):
        print(f"  [{i}] {p.stem}")
    try:
        idx = int(input("Elige modelo (número): "))
        model = joblib.load(available[idx])
        model_name = available[idx].stem
    except (ValueError, IndexError):
        print("Selección inválida.")
        return

    # ── 2. Cargar nombres de features ──────────────────────────────────
    feat_path = ARTIFACTS_DIR / "feature_names.joblib"
    if feat_path.exists():
        feature_names = joblib.load(feat_path)
    else:
        x_train_path = PROCESSED_DATA_DIR / "X_train.csv"
        if x_train_path.exists():
            feature_names = pd.read_csv(x_train_path).columns.tolist()
        else:
            print("No se encontró feature_names.joblib ni X_train.csv. Ejecuta primero run_full_pipeline().")
            return

    # ── 3. Pedir valores al usuario ────────────────────────────────────
    print(f"\nIntroduce los valores para el modelo '{model_name}':")
    print("  (deja en blanco para usar 0 como valor por defecto)\n")
    row = {}
    for feat in feature_names:
        raw = input(f"  {feat}: ").strip()
        try:
            row[feat] = float(raw) if raw else 0.0
        except ValueError:
            row[feat] = raw if raw else 0.0

    df_input = pd.DataFrame([row])

    # ── 4. Preprocesar ─────────────────────────────────────────────────
    try:
        X_new = process_input(df_input)
    except Exception as e:
        print(f"\nError en preprocesado: {e}")
        return

    # ── 6. Predecir ────────────────────────────────────────────────────
    print(f"\n{'='*50}")

    # ── 5. Cargar umbral (si existe) ───────────────────────────────────
    threshold_path = ARTIFACTS_DIR / "threshold.joblib"
    threshold = joblib.load(threshold_path) if threshold_path.exists() else 0.5

    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X_new)[0]
        pred  = int(proba[1] >= threshold)
        print(f"  Modelo     : {model_name}")
        print(f"  Umbral     : {threshold:.4f}")
        print(f"  Predicción : {pred}")
        print(f"  Probabilidades: {dict(enumerate(proba.round(4).tolist()))}")
        # Decodificar etiqueta original si existe target_encoder
        te_path = ARTIFACTS_DIR / "target_encoder.joblib"
        if te_path.exists():
            te = joblib.load(te_path)
            print(f"  Clase      : {te.inverse_transform([pred])[0]}")
    else:
        pred = model.predict(X_new)[0]
        print(f"  Modelo     : {model_name}")
        print(f"  Predicción : {int(pred)}")

    print(f"{'='*50}\n")


