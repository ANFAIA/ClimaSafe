import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient


from climasafeai.utils.paths import MODELS_DIR, ARTIFACTS_DIR, PROJECT_DIR


# ---------------------------------------------------------------------------
# Configuración de modelos
# ---------------------------------------------------------------------------

# RandomForest: hiperparámetros POR CLASE (calor/frío) -- se decidieron por
# separado con validación cruzada temporal por años, priorizando el recall de
# las clases de riesgo (ver reports/rf_tuning_*.csv). Solo cambia max_depth:
#   - calor -> 12: mismo recall que a profundidad 10 pero algo mejor accuracy/F1w.
#   - frío  -> 8 : el frío (olas más prolongadas, menos picos aislados) generaliza
#                  mejor con árboles más superficiales -- d8 mejora el recall de
#                  ambas clases de riesgo frente al 12 heredado de calor
#                  (precaución 0.32->0.38, peligro 0.58->0.60). Ver rf_tuning_frio.csv.
# El resto de la config (class_weight="balanced", etc.) es común -- pesos custom
# más agresivos NO suben el recall total, solo lo trasladan entre clases.
_RF_BASE_PARAMS = {
    "n_estimators": 200, "max_features": "sqrt", "max_samples": 0.8,
    "class_weight": "balanced", "random_state": 42, "n_jobs": -1,
}
_RF_MAX_DEPTH_BY_CLASE = {"calor": 12, "frio": 8}


def _load_best_params(model_name: str) -> dict:
    """Carga los mejores hiperparámetros cacheados si existen (de Optuna
    para RandomForest/XGBoost, o de GridSearchCV para KNN -- ver _tune_knn_k)."""
    path = ARTIFACTS_DIR / f"best_params_{model_name}.joblib"
    if path.exists():
        params = joblib.load(path)
        print(f"    [{model_name}] best_params cargados desde caché: {params}")
        return params
    return {}


def _tune_knn_k(X_train, y_train, k_range=range(1, 31, 2)) -> int:
    """
    Busca el mejor valor de k para KNN vía GridSearchCV (5-fold,
    F1_weighted -- misma métrica que se usa para evaluar el resto de
    modelos en train_models(), para que las puntuaciones sean comparables).

    k_range por defecto: impares de 1 a 29 (evita empates de votación en
    problemas binarios; aquí es multiclase pero mantiene la convención).

    El resultado se cachea en best_params_KNN.joblib (mismo patrón que
    _load_best_params usa para Optuna) para no repetir la búsqueda -- que
    el propio docstring de la config del notebook ya advierte que es
    lenta en datasets grandes -- en cada llamada a train_models().
    """
    print(f"    [KNN] buscando mejor k en {list(k_range)}...")
    grid = GridSearchCV(
        KNeighborsClassifier(),
        param_grid={"n_neighbors": list(k_range)},
        cv=5,
        scoring="f1_weighted",
        n_jobs=-1,
    )
    grid.fit(X_train, y_train)
    best_k = grid.best_params_["n_neighbors"]
    print(f"    [KNN] mejor k = {best_k} (F1_weighted CV = {grid.best_score_:.3f})")

    joblib.dump({"n_neighbors": best_k}, ARTIFACTS_DIR / "best_params_KNN.joblib")
    return best_k


def _build_models(X_train=None, y_train=None, tune_knn: bool = True,
                  clase: str = "calor") -> dict:
    """
    Define los modelos a entrenar.
    Tarea: clasificacion
    RandomForest       → ensemble robusto con feature importances.
    XGBoost            → gradient boosting optimizado. Referencia en Kaggle.
    KNN                → sensible a la escala (ya viene escalado por
                          preprocess_data) y al valor de k -- ver tune_knn.

    Parameters
    ----------
    X_train, y_train : necesarios solo si tune_knn=True y no hay
        best_params_KNN.joblib cacheado todavía (hace falta ajustar k con
        los datos reales). Si tune_knn=False, no se usan.
    tune_knn : bool
        Si True, busca el mejor k con GridSearchCV (lento en datasets
        grandes) la primera vez, y cachea el resultado. Llamadas
        posteriores reutilizan el k cacheado en vez de re-buscar.
    clase : "calor" | "frio"
        Selecciona los hiperparámetros de RandomForest específicos de la
        clase (ver _RF_MAX_DEPTH_BY_CLASE) -- calor y frío se afinaron por
        separado. Solo afecta al RandomForest; XGBoost/KNN son comunes.
    """
    models = {}

    _best = {name: _load_best_params(name) for name in [
        "RandomForest",
        "XGBoost",
        "KNN",
    ]}

    # RandomForest afinado para RECALL de las clases de riesgo, con max_depth
    # POR CLASE (calor=12 / frío=8, ver _RF_MAX_DEPTH_BY_CLASE arriba). El
    # best_params cacheado, si existe, tiene prioridad sobre estos defaults.
    if clase not in _RF_MAX_DEPTH_BY_CLASE:
        raise ValueError(
            f"_build_models: clase='{clase}' no reconocida -- debe ser una de "
            f"{list(_RF_MAX_DEPTH_BY_CLASE)}."
        )
    models["RandomForest"] = RandomForestClassifier(**{
        **_RF_BASE_PARAMS,
        "max_depth": _RF_MAX_DEPTH_BY_CLASE[clase],
        **_best.get("RandomForest", {}),
    })

    models["XGBoost"] = XGBClassifier(**{
        "n_estimators": 300, "max_depth": 6, "learning_rate": 0.05,
        "subsample": 0.8, "colsample_bytree": 0.8,
        "reg_alpha": 0.1, "reg_lambda": 1.0,
        "eval_metric": "logloss", "random_state": 42, "n_jobs": -1,
        **_best.get("XGBoost", {}),
    })

    knn_params = dict(_best.get("KNN", {}))
    if "n_neighbors" not in knn_params:
        if tune_knn:
            if X_train is None or y_train is None:
                raise ValueError(
                    "_build_models: tune_knn=True pero no hay best_params_KNN.joblib "
                    "cacheado todavía -- hacen falta X_train/y_train para poder buscar k."
                )
            knn_params["n_neighbors"] = _tune_knn_k(X_train, y_train)
        else:
            knn_params["n_neighbors"] = 5  # valor por defecto de sklearn, sin buscar

    models["KNN"] = KNeighborsClassifier(**{
        "weights": "distance",  # vecinos más cercanos pesan más -- razonable con clases desequilibradas
        "n_jobs": -1,
        **knn_params,
    })

    return models


def configurar_mlflow() -> None:
    """
    Apunta MLflow al experimento 'climasafeai' con fallback a tracking local.

    Reutilizable desde cualquier módulo que loguee a MLflow (train_models,
    lstm_model, ...) para que todos compartan el mismo experimento y el
    mismo fallback.
    """
    try:
        mlflow.set_experiment("climasafeai")
    except Exception as e:
        # Típicamente ConnectionRefusedError/MlflowException si
        # MLFLOW_TRACKING_URI apunta a un servidor (p.ej. localhost:5000)
        # que no está levantado -- en vez de reventar todo el entrenamiento,
        # se cae a tracking local (sqlite, sin necesitar ningún proceso
        # corriendo -- MLflow 3.x ya no permite el backend de fichero
        # plano './mlruns' sin más). Arranca el servidor con
        # `make mlflow-ui` si prefieres la UI en vez de tracking local.
        #
        # Ruta ABSOLUTA basada en PROJECT_DIR (no relativa a './') para
        # que el fichero .db quede siempre en la raíz del proyecto, sin
        # importar desde qué carpeta (notebooks/, raíz, etc.) se ejecute
        # -- una ruta relativa aquí crea un mlflow.db distinto por cada
        # cwd desde la que se lance el notebook.
        db_path = PROJECT_DIR / "mlflow.db"
        print(f"    AVISO: no se pudo conectar al tracking server de MLflow ({e}).")
        print(f"    Usando tracking local (sqlite:///{db_path}) en su lugar.")
        try:
            mlflow.set_tracking_uri(f"sqlite:///{db_path}")
            mlflow.set_experiment("climasafeai")
        except Exception as e2:
            # Si esto también falla, casi seguro es un problema de
            # permisos del fichero/carpeta (p.ej. mlflow.db creado
            # previamente sin permiso de escritura para el usuario actual,
            # o la carpeta del proyecto es de solo lectura) -- no es algo
            # que el código pueda arreglar solo, hace falta revisarlo a mano.
            raise RuntimeError(
                f"No se pudo usar ni el tracking server ni el fallback local "
                f"({db_path}). Revisa permisos del fichero/carpeta: "
                f"`ls -la {db_path.parent}` y `chmod u+w {db_path}` si ya existe, "
                f"o bórralo y deja que MLflow lo recree."
            ) from e2


def train_models(
    X_train,
    y_train,
    tune_knn: bool = True,
    cv_evaluate: bool = True,
    clase: str = "calor",
) -> dict:
    """
    Entrena modelos de clasificacion y los guarda en models/.

    Métrica CV: F1_weighted (5-fold).

    MLflow: cada modelo se loguea como un run independiente dentro del
    experimento 'climasafeai'. Los artifacts (.joblib) se registran
    en el Model Registry bajo el nombre del modelo.

    Parameters
    ----------
    tune_knn : bool
        Si True, busca el mejor k para KNN vía GridSearchCV (ver
        _tune_knn_k) antes de entrenar -- la primera vez es lento en
        datasets grandes; llamadas siguientes reutilizan el k cacheado
        en best_params_KNN.joblib. Si False, usa k=5 (o el valor
        cacheado, si ya existe) sin buscar.
    clase : "calor" | "frio"
        Qué modelo se entrena. Selecciona los hiperparámetros de
        RandomForest específicos de la clase (ver _RF_MAX_DEPTH_BY_CLASE) --
        calor y frío se afinaron por separado. Debe coincidir con el `clase`
        usado en preprocess_data() al generar los datos.

    Returns
    -------
    dict : {nombre_modelo: modelo_entrenado}
    """
    print(f"--> Entrenando modelos de clasificacion (clase='{clase}')...")
    models = _build_models(X_train=X_train, y_train=y_train, tune_knn=tune_knn, clase=clase)

    configurar_mlflow()

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
            mlflow.log_param("clase", clase)

            # XGBoost no acepta class_weight="balanced" como RandomForest -> sin
            # ponderar colapsa a la clase mayoritaria (seguro ~90-94%) e IGNORA
            # los días de riesgo (recall clases 1/2 ~0.00-0.02, inservible para
            # un aviso). Se le pasan pesos POR MUESTRA balanceados en el fit, que
            # es el equivalente a class_weight="balanced": sube el recall de
            # riesgo a costa de algo de accuracy global (ver reports/rf_tuning_*
            # y la comparación RF vs XGBoost-con-pesos).
            if name == "XGBoost":
                sample_weight = compute_sample_weight("balanced", y_train)
                model.fit(X_train, y_train, sample_weight=sample_weight)
            else:
                model.fit(X_train, y_train)

            if cv_evaluate:
                cv_score = cross_val_score(
                    model, X_train, y_train, cv=5, scoring="f1_weighted"
                ).mean()
                print(f"      F1_weighted 5-fold CV: {cv_score:.3f}")

                mlflow.log_metric("cv_score", cv_score)

            # Fichero namespaceado por clase (XGBoost_calor.joblib /
            # XGBoost_frio.joblib) -- así entrenar una clase NO pisa el modelo
            # de la otra (mismo criterio que scaler_{clase}/encoders_{clase}).
            model_filename = f"{name}_{clase}.joblib"
            joblib.dump(model, MODELS_DIR / model_filename)
            print(f"      Guardado → {model_filename}")

            mlflow.sklearn.log_model(
                model, artifact_path=name,
                registered_model_name=f"climasafeai_{name}_{clase}",
                # serialization_format="cloudpickle": el formato por defecto
                # de MLflow 3.x (skops) rechaza objetos internos de KNN
                # (KDTree/EuclideanDistance64) y de XGBoost como "tipos no
                # confiables" (UntrustedTypesFoundException). cloudpickle
                # evita ese chequeo -- es el mismo compromiso de seguridad
                # que ya asumes al usar joblib.dump() unas líneas arriba.
                serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_CLOUDPICKLE,
            )
            mlflow.log_artifact(str(MODELS_DIR / model_filename))

        trained[name] = model

    print(f"--> {len(trained)} modelos guardados en {MODELS_DIR}")
    return trained


def load_models(model_names: list = None, clase: str = "calor") -> dict:
    """
    Carga modelos entrenados desde disco, namespaceados por clase
    (XGBoost_calor.joblib, etc.). Devuelve {nombre_modelo: modelo}, con el
    nombre SIN el sufijo de clase.

    Si model_names es None, carga todos los `*_{clase}.joblib` de MODELS_DIR.
    """
    sufijo = f"_{clase}.joblib"
    if model_names is None:
        models = {}
        for p in sorted(MODELS_DIR.glob(f"*{sufijo}")):
            name = p.name[: -len(sufijo)]
            models[name] = joblib.load(p)
            print(f"    Cargado: {name} ({clase})")
        return models
    models = {}
    for name in model_names:
        path = MODELS_DIR / f"{name}{sufijo}"
        if path.exists():
            models[name] = joblib.load(path)
            print(f"    Cargado: {name} ({clase})")
        else:
            print(f"    No encontrado: {path}")
    return models