
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

# ---------------------------------------------------------------------------
# Umbrales de decisión POR CLASE (modelos multiclase de riesgo:
# 0=seguro, 1=precaución, 2=peligro)
# ---------------------------------------------------------------------------
# Calibrados con tuning/calibracion_umbrales.py sobre una validación temporal
# interna de train (último 15% de fechas de train; test NUNCA se usa para
# elegirlos) -- ver documentacion/calibracion_umbrales.md. Regla en cascada
# por severidad (apply_class_thresholds):
#     P(clase 2) >= t2          -> 2 (peligro)
#     P(1) + P(2) >= t1         -> 1 (precaución)
#     en otro caso              -> 0 (seguro)
# Con class_thresholds=None todo sigue funcionando como siempre (argmax).
CLASS_THRESHOLDS_RECOMENDADOS: dict = {
    "calor": {"t1": 0.50, "t2": 0.45},
    "frio":  {"t1": 0.45, "t2": 0.40},
}

# Thresholds óptimos para LSTM province_hybrid (con peso_riesgo_extra=8.0).
# Calibrados independientemente de los de los modelos tabulares.
CLASS_THRESHOLDS_LSTM: dict = {
    "calor": {"t1": 0.60, "t2": 0.55},
    "frio":  {"t1": 0.40, "t2": 0.35},
}


def apply_class_thresholds(proba: np.ndarray, t1: float, t2: float) -> np.ndarray:
    """
    Decisión en cascada por severidad sobre probabilidades multiclase
    (columnas = clases 0, 1, 2):

        P(clase 2) >= t2    -> 2 (peligro)
        P(1) + P(2) >= t1   -> 1 (precaución)
        en otro caso        -> 0 (seguro)

    "Peligro" exige evidencia directa de la clase 2; "precaución" solo
    exige suficiente masa de probabilidad de riesgo total -- coherente con
    la política del sistema (mejor sobre-avisar que no avisar). Es la
    extensión a 3 clases ordinales del DECISION_THRESHOLD binario.

    Parameters
    ----------
    proba : array (n_samples, 3) de predict_proba.
    t1    : umbral sobre P(1)+P(2) para avisar al menos "precaución".
    t2    : umbral sobre P(2) para escalar a "peligro".
    """
    proba = np.asarray(proba)
    if proba.ndim != 2 or proba.shape[1] != 3:
        raise ValueError(
            f"apply_class_thresholds espera probabilidades (n, 3) de un "
            f"modelo de 3 clases (0/1/2); recibido shape={proba.shape}."
        )
    p2 = proba[:, 2]
    p_riesgo = proba[:, 1] + proba[:, 2]
    return np.where(p2 >= t2, 2, np.where(p_riesgo >= t1, 1, 0))

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
        # Métricas NO ponderadas -- las ponderadas (F1_test/Rec_test) están
        # dominadas por la clase 0 (seguro ~90-94%) y hacen "ganar" a modelos
        # que casi nunca avisan. Para un sistema de aviso importan estas:
        #   F1_macro   -> media por clase (penaliza ignorar minoritarias)
        #   Rec_riesgo -> recall medio de las clases de riesgo (todas menos la 0)
        f1_macro  = f1_score(y_test, y_pred_test, average="macro", zero_division=0)
        risk_labels = [c for c in np.unique(y_test) if c != 0]
        rec_riesgo = (recall_score(y_test, y_pred_test, labels=risk_labels,
                                   average="macro", zero_division=0)
                      if risk_labels else float("nan"))
        roc_auc   = None
        if hasattr(model, "predict_proba") and len(np.unique(y_test)) == 2:
            roc_auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])

        print(f"  Accuracy  → train: {acc_train:.3f} | test: {acc_test:.3f}")
        print(f"  F1 (w)    → train: {f1_train:.3f}  | test: {f1_test:.3f}")
        print(f"  Precision → {prec_test:.3f}  | Recall → {rec_test:.3f}")
        print(f"  F1_macro  → {f1_macro:.3f}  | Rec_riesgo (clases 1..n) → {rec_riesgo:.3f}")
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
            "F1_macro":  round(f1_macro,  4), "Rec_riesgo": round(rec_riesgo, 4),
        }
        if roc_auc is not None:
            row["ROC_AUC"] = round(roc_auc, 4)

        with mlflow.start_run(run_name=f"{name}_eval"):
            mlflow.log_metrics({
                "acc_train": acc_train, "acc_test": acc_test,
                "f1_train":  f1_train,  "f1_test":  f1_test,
                "prec_test": prec_test, "rec_test":  rec_test,
                "f1_macro":  f1_macro,  "rec_riesgo": rec_riesgo,
            })
            if roc_auc is not None:
                mlflow.log_metric("roc_auc", roc_auc)
            mlflow.log_artifact(str(FIGURES_DIR / f"cm_{name}.png"))
        results.append(row)

    df_results = pd.DataFrame(results).sort_values("F1_macro", ascending=False)

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



def predict_new(model_name: str, X_new, class_thresholds=None) -> np.ndarray:
    """Carga un modelo y predice sobre nuevas muestras (ya preprocesadas).

    Parameters
    ----------
    class_thresholds : None | dict | str
        - None (por defecto): comportamiento de siempre -- model.predict()
          (argmax de las probabilidades).
        - dict {"t1": float, "t2": float}: decisión en cascada por clase
          sobre predict_proba -- ver apply_class_thresholds().
        - "calor" | "frio": usa los umbrales calibrados de
          CLASS_THRESHOLDS_RECOMENDADOS para ese modelo de riesgo.
    """
    path = MODELS_DIR / f"{model_name}.joblib"
    if not path.exists():
        raise FileNotFoundError(f"Modelo no encontrado: {path}")
    model = joblib.load(path)
    if class_thresholds is None:
        return model.predict(X_new)

    if isinstance(class_thresholds, str):
        if class_thresholds not in CLASS_THRESHOLDS_RECOMENDADOS:
            raise ValueError(
                f"class_thresholds='{class_thresholds}' no reconocido -- usa "
                f"una de {list(CLASS_THRESHOLDS_RECOMENDADOS)}, un dict "
                "{'t1': ..., 't2': ...} o None."
            )
        class_thresholds = CLASS_THRESHOLDS_RECOMENDADOS[class_thresholds]
    if not hasattr(model, "predict_proba"):
        raise ValueError(
            f"{model_name} no soporta predict_proba -- no se pueden aplicar "
            "umbrales por clase; llama a predict_new sin class_thresholds."
        )
    return apply_class_thresholds(
        model.predict_proba(X_new), class_thresholds["t1"], class_thresholds["t2"]
    )



def predict_proba_new(model_name: str, X_new) -> np.ndarray:
    """Carga un modelo y devuelve probabilidades de clase."""
    path = MODELS_DIR / f"{model_name}.joblib"
    if not path.exists():
        raise FileNotFoundError(f"Modelo no encontrado: {path}")
    model = joblib.load(path)
    if not hasattr(model, "predict_proba"):
        raise ValueError(f"{model_name} no soporta predict_proba")
    return model.predict_proba(X_new)


CLASES = ["SEGURO", "PRECAUCION", "PELIGRO"]


def _preguntar_perfil() -> dict:
    print("\n--- Datos personales (opcional, pulsa Enter para saltar) ---")
    perfil = {}
    raw = input("  Edad: ").strip()
    if raw:
        perfil["edad"] = int(raw)
    raw = input("  Sexo (hombre/mujer): ").strip().lower()
    if raw in ("hombre", "mujer"):
        perfil["sexo"] = raw
    raw = input("  Porcentaje graso: ").strip()
    if raw:
        perfil["porcentaje_grasa"] = float(raw)
    raw = input("  Nivel de actividad (reposo/ligera/moderada/intensa/muy_intensa): ").strip().lower().replace(" ", "_")
    if raw in ("reposo", "ligera", "moderada", "intensa", "muy_intensa"):
        perfil["nivel_actividad"] = raw
    raw = input("  Hora inicio actividad (0-23, ej: 17): ").strip()
    if raw:
        perfil["hora_inicio"] = float(raw)
    raw = input("  Duración actividad (horas): ").strip()
    if raw:
        perfil["duracion_actividad_h"] = float(raw)
    raw = input("  ¿Aclimatado al clima local? (s/n): ").strip().lower()
    if raw in ("s", "si", "sí"):
        perfil["aclimatado"] = True
    elif raw in ("n", "no"):
        perfil["aclimatado"] = False
    raw = input("  Fototipo Fitzpatrick (1-6): ").strip()
    if raw in ("1", "2", "3", "4", "5", "6"):
        perfil["fototipo"] = raw
    print("  Comorbilidades (separadas por coma, ej: cardiovascular,diabetes):")
    raw = input("    ").strip().lower()
    if raw:
        perfil["comorbilidades"] = set(c.strip().replace(" ", "_") for c in raw.split(",") if c.strip())
    print("  Situación social (separada por coma, ej: vive_solo,encamado):")
    raw = input("    ").strip().lower()
    if raw:
        perfil["situacion_social"] = set(c.strip().replace(" ", "_") for c in raw.split(",") if c.strip())
    return perfil


def try_model() -> None:
    from climasafeai.models.ensemble import predict_ensemble

    print("\n=== ClimaSafeAI - Prediccion ensemble (4 modelos) ===\n")

    raw = input("Provincia o ubicacion (ej: Madrid, 40.4168,-3.7038): ").strip()
    lat = lon = None
    provincia = "Madrid"
    if "," in raw and raw.count(",") == 1:
        try:
            lat, lon = [float(x.strip()) for x in raw.split(",")]
            provincia = f"{lat},{lon}"
        except ValueError:
            lat, lon = None, None
    else:
        provincia = raw

    perfil = _preguntar_perfil()

    print("\nObteniendo datos meteorologicos...")
    try:
        resultado = predict_ensemble(lat=lat, lon=lon, provincia=provincia, perfil=perfil)
    except Exception as e:
        print(f"\n  Error en la prediccion: {e}")
        return

    w = resultado["weather"]
    print(f"\n{'='*60}")
    print(f"  {provincia} ({w['lat']:.2f}, {w['lon']:.2f})")
    if w.get("current"):
        c = w["current"]
        print(f"  {c.get('t2m_c', '?')}C  {c.get('rh', '?')}%  {c.get('wind_speed_kmh', '?')} km/h")
    if w.get("uv_index") is not None:
        print(f"  UV: {w['uv_index']}")
    print(f"{'='*60}")

    print(f"\n{'-'*60}")
    print("  Predicciones individuales:")
    print(f"{'-'*60}")
    for modelo, res in resultado["modelos"].items():
        if modelo == "LSTM":
            if "error" in res:
                print(f"  LSTM: {res['error']}")
                continue
            c_cal = res["calor"]["clase_threshold"]
            c_frio = res["frio"]["clase_threshold"]
            print(f"  LSTM (calor):   {CLASES[c_cal]} ({c_cal})  riesgo={res['calor']['prob_riesgo']:.3f}")
            print(f"  LSTM (frio):    {CLASES[c_frio]} ({c_frio})  riesgo={res['frio']['prob_riesgo']:.3f}")
        elif modelo == "Formula":
            c_cal = res["calor"]["clase"]
            c_frio = res["frio"]["clase"]
            print(f"  Formula (calor): {CLASES[c_cal]} ({c_cal})  HI={res['calor']['heat_index_c']}C")
            print(f"  Formula (frio):  {CLASES[c_frio]} ({c_frio})  WC={res['frio']['wind_chill_c']}C")
        else:
            c = res["clase_threshold"]
            print(f"  {modelo}: {CLASES[c]} ({c})  riesgo={res['prob_riesgo']:.3f}")

    print(f"\n{'-'*60}")
    print("  Umbrales aplicados (P(riesgo) >= t1 -> PRECAUCION, P(peligro) >= t2 -> PELIGRO):")
    for modelo, res in resultado["modelos"].items():
        if modelo == "LSTM" and isinstance(res, dict) and "error" not in res:
            u_c = res.get("calor", {}).get("thresholds_usados", {})
            u_f = res.get("frio", {}).get("thresholds_usados", {})
            if u_c:
                print(f"  LSTM calor:  t1={u_c['t1']:.2f}  t2={u_c['t2']:.2f}")
            if u_f:
                print(f"  LSTM frio:   t1={u_f['t1']:.2f}  t2={u_f['t2']:.2f}")
        elif isinstance(res, dict) and "thresholds_usados" in res:
            u = res["thresholds_usados"]
            print(f"  {modelo}: t1={u['t1']:.2f}  t2={u['t2']:.2f}")

    print(f"\n{'='*60}")
    print(f"  ESCENARIO MAS RESTRICTIVO: {resultado['clase_final_label']} ({resultado['clase_final']})")
    print(f"{'='*60}")

    perfil_horario = w.get("perfil_horario")
    h_ini = perfil.get("hora_inicio")
    dur = perfil.get("duracion_actividad_h")
    if perfil_horario:
        print(f"\n{'-'*60}")
        print("  Perfil de riesgo horario (HI por hora):")
        print(f"{'-'*60}")
        horas_lista = [e["hora"] for e in perfil_horario]
        max_hi = max(h["HI"] for h in perfil_horario) if perfil_horario else 40
        for entry in perfil_horario:
            hora = entry["hora"]
            hi = entry["HI"]
            bars = min(8, max(0, int(hi / max(1, max_hi) * 8)))
            barra = "▓" * bars + "░" * (8 - bars)
            if hi < 27:
                clase_h = "SEGURO"
            elif hi < 39:
                clase_h = "PRECAUCION"
            else:
                clase_h = "PELIGRO"
            marca = ""
            if h_ini is not None and dur is not None and h_ini <= hora < h_ini + dur:
                marca = " ← actividad"
            print(f"  {hora:02d}:00  HI={hi:.1f}C  {barra}  {clase_h}{marca}")
        print()
        if h_ini is not None and dur is not None:
            h_fin = h_ini + dur
            act_hi = [e["HI"] for e in perfil_horario if h_ini <= e["hora"] < h_fin]
            act_clase = "SEGURO"
            if act_hi:
                max_act_hi = max(act_hi)
                if max_act_hi >= 39:
                    act_clase = "PELIGRO"
                elif max_act_hi >= 27:
                    act_clase = "PRECAUCION"
            act_bar = "█" * int(dur * 4)
            print(f"  Actividad:  {h_ini:.0f}:00 {act_bar} {h_fin:.0f}:00  {act_clase}")

    override = resultado.get("override_fisico")
    if override:
        hi = resultado["modelos"]["Formula"]["calor"]["heat_index_c"]
        wc = resultado["modelos"]["Formula"]["frio"]["wind_chill_c"]
        if perfil_horario and h_ini is not None and dur is not None:
            window_hi = [e["HI"] for e in perfil_horario if h_ini <= e["hora"] < h_ini + dur]
            if window_hi:
                peak_hi = max(window_hi)
                h_fin = h_ini + dur
                print(f"\n  NOTA: Condiciones actuales seguras (HI={hi}C), pero durante")
                print(f"  la actividad prevista ({h_ini:.0f}:00-{h_fin:.0f}:00)")
                print(f"  se espera HI de hasta {peak_hi}C, lo que justifica el nivel")
                print(f"  de riesgo. El modelo ML lo confirma por tendencia.")
            else:
                print(f"\n  NOTA: Condiciones actuales seguras (HI={hi}C, WC={wc}C).")
                print(f"  El modelo ML indicaba riesgo por tendencia meteorologica,")
                print(f"  pero los indicadores fisicos actuales no lo confirman.")
        else:
            c = resultado.get("weather", {}).get("current", {})
            t = c.get("t2m_c", "?")
            uv = resultado.get("weather", {}).get("uv_index", "?")
            print(f"\n  NOTA: Condiciones actuales ({t}C, HI={hi}C, WC={wc}C, UV={uv})")
            print(f"  estan en rango seguro. El modelo ML indicaba riesgo por")
            print(f"  tendencia de dias anteriores, pero los indicadores fisicos")
            print(f"  no muestran riesgo objetivo.")

    # -- Explicacion --
    explicacion = resultado.get("explicacion", {})
    if explicacion:
        print(f"\n{'-'*60}")
        print("  Explicacion:")
        print(f"{'-'*60}")
        modelo_det = explicacion.get("modelo_determinante", "")
        if modelo_det:
            print(f"  Modelo determinante: {modelo_det}")
        detalles = explicacion.get("detalles", {})
        for mod, det in detalles.items():
            if isinstance(det, dict) and "error" in det:
                continue
            prob_mod = resultado.get("modelos", {}).get(mod, {}).get("prob_riesgo", 0)
            if prob_mod < 0.35:
                continue
            if mod == "XGBoost_calor":
                print(f"\n  Riesgo por tendencia meteorologica (XGBoost calor):")
                print(f"  El modelo detecta patron de calor acumulado de dias previos")
                print(f"  (NO es el HI actual, sino medias moviles y persistencia):")
                tops = det.get("top_features", [])
                if tops:
                    todas_cero = all(t.get("importancia", 0) == 0 for t in tops)
                    if todas_cero:
                        print("    El modelo no detecta factores de riesgo significativos para este dia.")
                    else:
                        for ft in tops:
                            print(f"    {ft['feature']}  ({ft['importancia']})")
            elif mod == "RandomForest_frio":
                print(f"\n  Riesgo por tendencia meteorologica (RandomForest frio):")
                print(f"  El modelo detecta patron de frio acumulado de dias previos:")
                tops = det.get("top_features", [])
                if tops:
                    todas_cero = all(t.get("importancia", 0) == 0 for t in tops)
                    if todas_cero:
                        print("    El modelo no detecta factores de riesgo significativos para este dia.")
                    else:
                        for ft in tops:
                            print(f"    {ft['feature']}  ({ft['importancia']})")
            elif mod == "LSTM":
                metodo = det.get("metodo", "")
                hora = det.get("hora_mas_influyente")
                if hora is not None:
                    print(f"\n  LSTM: hora mas influyente = {hora}:00")
                vars_top = det.get("variables_top", [])
                if vars_top:
                    print(f"  Variables mas influyentes en la secuencia:")
                    for vt in vars_top:
                        print(f"    {vt['feature']}  ({vt['importancia']})")
            elif mod == "Formula":
                expls = det.get("explicaciones", [])
                for exp_t in expls:
                    print(f"\n  {exp_t}")

    # -- Factores personales --
    if resultado["perfil"]["calor"]["factores"] or resultado["perfil"]["frio"]["factores"]:
        print(f"\n{'-'*60}")
        print("  Factores de riesgo aplicados:")
        print(f"{'-'*60}")
        for tipo in ("calor", "frio"):
            sec = resultado["perfil"][tipo]
            if sec["factores"]:
                print(f"  {tipo.capitalize()}:")
                for f in sec["factores"]:
                    print(f"    {f['nombre']} -> x{f['factor']}")
                ft = sec["factor_total"]
                pb = sec.get("producto_bruto", ft)
                if sec.get("capado") and pb > ft:
                    print(f"    Factor total calculado: x{pb}")
                    print(f"    Aplicado (cap x{sec.get('cap_factores', 3.0)}): x{ft}")
                else:
                    print(f"    Factor total: x{ft}")
                print(f"    Prob. poblacional: {sec['prob_poblacional']:.3f} -> personalizada: {sec['prob_personalizada']:.3f}")
    else:
        print("\n  Sin datos personales - factores neutros (x1.0)")

    # -- Recomendaciones --
    recomendaciones = resultado.get("recomendaciones", [])
    if recomendaciones:
        print(f"\n{'-'*60}")
        print("  Recomendaciones:")
        print(f"{'-'*60}")
        for i, rec in enumerate(recomendaciones, 1):
            print(f"  {i}. {rec}")

    print(f"\n{'='*60}\n")


