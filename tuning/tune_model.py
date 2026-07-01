
"""
tuning/tune_model.py — Optimización de hiperparámetros con Optuna.

Ejecutar con:
    make tune

O directamente:
    uv run python -m tuning.tune_model

Qué hace:
    1. Ejecuta un estudio Optuna por cada modelo activo.
    2. Guarda los mejores params en artifacts/best_params_<modelo>.joblib.
    3. train_models() los carga automáticamente en el siguiente make train.
    4. Guarda un resumen en reports/tuning_results.csv.

Configurar el número de trials en main.py:
    OPTUNA_TRIALS = 30
"""
from __future__ import annotations
import warnings
import joblib
import numpy as np
import pandas as pd
import optuna
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from climasafeai.utils.paths import ARTIFACTS_DIR, REPORTS_DIR
from sklearn.metrics import f1_score
warnings.filterwarnings("ignore")

optuna.logging.set_verbosity(optuna.logging.WARNING)
_SCORING  = "f1_weighted"
_MINIMIZE = False   # maximize F1

# ---------------------------------------------------------------------------
# Objetivos por modelo
# ---------------------------------------------------------------------------

def _objective_rf(trial, X_train, y_train):

    from sklearn.ensemble import RandomForestClassifier
    model = RandomForestClassifier(
        n_estimators     = trial.suggest_int("n_estimators",   50,  400, step=50),
        max_depth        = trial.suggest_int("max_depth",       3,   20),
        min_samples_leaf = trial.suggest_int("min_samples_leaf",1,   20),
        max_features     = trial.suggest_categorical("max_features", ["sqrt", "log2", None]),
        class_weight     = "balanced",
        random_state     = 42, n_jobs=-1,
    )

    scores = cross_val_score(model, X_train, y_train, cv=5, scoring=_SCORING)
    return -scores.mean() if _MINIMIZE else scores.mean()













def _objective_xgb(trial, X_train, y_train):

    from xgboost import XGBClassifier
    model = XGBClassifier(
        n_estimators      = trial.suggest_int("n_estimators",    50,  500, step=50),
        max_depth         = trial.suggest_int("max_depth",        3,   10),
        learning_rate     = trial.suggest_float("learning_rate",  1e-3, 0.3, log=True),
        subsample         = trial.suggest_float("subsample",      0.5,  1.0),
        colsample_bytree  = trial.suggest_float("colsample_bytree",0.5, 1.0),
        reg_alpha         = trial.suggest_float("reg_alpha",      1e-4, 10.0, log=True),
        reg_lambda        = trial.suggest_float("reg_lambda",     1e-4, 10.0, log=True),
        eval_metric       = "logloss",
        random_state      = 42, n_jobs=-1,
    )

    scores = cross_val_score(model, X_train, y_train, cv=5, scoring=_SCORING)
    return -scores.mean() if _MINIMIZE else scores.mean()









# ---------------------------------------------------------------------------
# Mapa modelo → función objetivo
# ---------------------------------------------------------------------------
_OBJECTIVES: dict = {}

_OBJECTIVES["RandomForest"] = _objective_rf





_OBJECTIVES["XGBoost"] = _objective_xgb








# ---------------------------------------------------------------------------
# Motor principal de tuning
# ---------------------------------------------------------------------------
def tune_models(
    X_train,
    y_train=None,
    n_trials: int = 30,
    timeout: int = None,

    artifacts_dir=None,
    reports_dir=None,
) -> dict[str, dict]:
    """
    Optimiza hiperparámetros de todos los modelos activos con Optuna.

    Guarda los mejores params en artifacts/best_params_<modelo>.joblib.
    train_models() los carga automáticamente si existen.

    Parameters
    ----------
    X_train       : features de entrenamiento
    y_train       : target (None para no_supervisado)
    n_trials      : número de trials por modelo (default: 30)
    timeout       : segundos máximos por estudio (None = sin límite)
    artifacts_dir : directorio donde guardar los .joblib (default: ARTIFACTS_DIR)
    reports_dir   : directorio donde guardar tuning_results.csv (default: REPORTS_DIR)

    Returns
    -------
    dict[str, dict] : {nombre_modelo: mejores_params}
    """
    _artifacts = artifacts_dir or ARTIFACTS_DIR
    _reports   = reports_dir   or REPORTS_DIR
    _artifacts.mkdir(parents=True, exist_ok=True)
    _reports.mkdir(parents=True, exist_ok=True)

    X_arr = X_train.values if hasattr(X_train, "values") else X_train
    if y_train is not None:
        y_arr = y_train.values if hasattr(y_train, "values") else y_train
    else:
        y_arr = None

    results = []
    best_params_all: dict[str, dict] = {}

    print(f"\n{'='*60}")
    print(f"  Optuna — optimizando {len(_OBJECTIVES)} modelo(s), {n_trials} trials c/u")
    print(f"{'='*60}")


    for model_name, objective_fn in _OBJECTIVES.items():
        print(f"\n  Optimizando {model_name}...")
        sampler = optuna.samplers.TPESampler(seed=42)

        direction = "maximize" if not _MINIMIZE else "minimize"
        def _obj(trial): return objective_fn(trial, X_arr, y_arr)

        study = optuna.create_study(direction=direction, sampler=sampler)
        study.optimize(_obj, n_trials=n_trials, timeout=timeout, show_progress_bar=False)

        best      = study.best_params
        best_val  = study.best_value
        print(f"    Mejor valor: {best_val:.4f}")
        print(f"    Params: {best}")

        path = _artifacts / f"best_params_{model_name}.joblib"
        joblib.dump(best, path)
        print(f"    Guardado → {path.name}")
        best_params_all[model_name] = best
        results.append({"modelo": model_name, "best_value": round(best_val, 4), **best})


    df = pd.DataFrame(results)
    out_csv = _reports / "tuning_results.csv"
    df.to_csv(out_csv, index=False)
    print(f"\n  Resumen guardado → {out_csv.name}")
    print(df.to_string(index=False))
    print(f"\n{'='*60}")
    print("  Ejecuta 'make train' para entrenar con los mejores params.")
    print(f"{'='*60}\n")

    return best_params_all


if __name__ == "__main__":
    print("Ejecuta 'make tune' para lanzar la optimización con tus datos procesados.")
    print("O importa tune_models() y pásale X_train e y_train directamente.")
