import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from xgboost import XGBClassifier
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient


from climasafeai.utils.paths import MODELS_DIR, ARTIFACTS_DIR


# ---------------------------------------------------------------------------
# Configuración de modelos
# ---------------------------------------------------------------------------

def _load_best_params(model_name: str) -> dict:
    """Carga los mejores hiperparámetros de Optuna si existen."""
    path = ARTIFACTS_DIR / f"best_params_{model_name}.joblib"
    if path.exists():
        params = joblib.load(path)
        print(f"    [{model_name}] best_params cargados desde Optuna: {params}")
        return params
    return {}


def _build_models() -> dict:
    """
    Define los modelos a entrenar.
    Tarea: clasificacion
    RandomForest       → ensemble robusto con feature importances.
    XGBoost            → gradient boosting optimizado. Referencia en Kaggle.

    """
    models = {}

    _best = {name: _load_best_params(name) for name in [
        "RandomForest",
        "XGBoost",

    ]}

    models["RandomForest"] = RandomForestClassifier(**{
        "n_estimators": 200, "max_depth": 10, "max_features": "sqrt",
        "max_samples": 0.8, "class_weight": "balanced", "random_state": 42, "n_jobs": -1,
        **_best.get("RandomForest", {}),
    })

    models["XGBoost"] = XGBClassifier(**{
        "n_estimators": 300, "max_depth": 6, "learning_rate": 0.05,
        "subsample": 0.8, "colsample_bytree": 0.8,
        "reg_alpha": 0.1, "reg_lambda": 1.0,
        "eval_metric": "logloss", "random_state": 42, "n_jobs": -1,
        **_best.get("XGBoost", {}),
    })

    return models

def train_models(
    X_train,
    y_train,
    tune_knn: bool = True,
    cv_evaluate: bool = True,
) -> dict:
    """
    Entrena modelos de clasificacion y los guarda en models/.

    Métrica CV: F1_weighted (5-fold).

    MLflow: cada modelo se loguea como un run independiente dentro del
    experimento 'climasafeai'. Los artifacts (.joblib) se registran
    en el Model Registry bajo el nombre del modelo.

    Returns
    -------
    dict : {nombre_modelo: modelo_entrenado}
    """
    print("--> Entrenando modelos de clasificacion...")
    models = _build_models()

    mlflow.set_experiment("climasafeai")


    trained = {}
    for name, model in models.items():

        mlflow.end_run()  # cerrar cualquier run activo antes de abrir uno nuevo

        print(f"    [{name}] entrenando...")


        with mlflow.start_run(run_name=name):
            # ── Parámetros ────────────────────────────────────────────────
            params = {}
            if hasattr(model, "get_params"):
                params = {k: v for k, v in model.get_params().items()
                          if v is not None and not callable(v)}
            mlflow.log_params(params)
            mlflow.log_param("task_type", "clasificacion")
            mlflow.log_param("model_name", name)

            model.fit(X_train, y_train)

            if cv_evaluate:
                cv_score = cross_val_score(
                    model, X_train, y_train, cv=5, scoring="f1_weighted"
                ).mean()
                print(f"      F1_weighted 5-fold CV: {cv_score:.3f}")

                mlflow.log_metric("cv_score", cv_score)

            joblib.dump(model, MODELS_DIR / f"{name}.joblib")
            print(f"      Guardado → {name}.joblib")

            mlflow.sklearn.log_model(
                model, artifact_path=name,
                registered_model_name=f"climasafeai_{name}",
            )
            mlflow.log_artifact(str(MODELS_DIR / f"{name}.joblib"))

        trained[name] = model

    print(f"--> {len(trained)} modelos guardados en {MODELS_DIR}")
    return trained


def load_models(model_names: list = None) -> dict:
    """Carga modelos desde disco."""
    if model_names is None:
        model_names = [p.stem for p in MODELS_DIR.glob("*.joblib")]
    models = {}
    for name in model_names:
        path = MODELS_DIR / f"{name}.joblib"
        if path.exists():
            models[name] = joblib.load(path)
            print(f"    Cargado: {name}")
        else:
            print(f"    No encontrado: {path}")
    return models
