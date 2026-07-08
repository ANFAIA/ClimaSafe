"""
temporal_cv.py — Validación cruzada temporal por años (ventana expansiva).

En vez de un único split train/test por fecha, evalúa varios folds
consecutivos (entrena con todos los años hasta N-1, evalúa en el año N,
desliza N) para no depender de si un único tramo de fechas resultó ser
fácil o difícil por casualidad.

Uso típico:
    df = dataset_calor(df_momo, df_eras)
    df = asignar_clase_riesgo_calor(df)
    resultados = evaluate_models_temporal_cv(df, target_col="clase_riesgo_calor", clase="calor")
    resumen = resumen_temporal_cv(resultados)
    print(resumen)
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier

from climasafeai.features.build_features import (
    _feature_engineering,
    _apply_logcols,
    LOGCOLS,
    ORDINAL_MAPPINGS,
    COLS_TO_DROP,
    LEAKAGE_COLS_BY_CLASE,
)


def _build_models_cv(knn_k: int = 5, clase: str = "calor") -> dict:
    """
    Modelos con hiperparámetros FIJOS (sin tuning por fold).

    A propósito no se re-tunean hiperparámetros en cada fold: si lo
    hiciéramos, la variación de resultados entre folds mezclaría dos
    cosas distintas (cómo de bien generaliza el modelo en el tiempo, y
    cuánto varía el hiperparámetro óptimo de un fold a otro), y dejaría
    de ser una comparación limpia. Usa el mismo `knn_k` que ya
    encontraste con `tune_knn` en train_model.py si quieres consistencia
    con el modelo que planeas usar en producción.

    `clase` selecciona el max_depth del RandomForest específico de la
    clase (calor=12 / frío=8), igual que train_model._build_models, para
    que la validación temporal use el MISMO modelo que producción.
    """
    rf_max_depth = {"calor": 12, "frio": 8}
    if clase not in rf_max_depth:
        raise ValueError(f"_build_models_cv: clase='{clase}' no reconocida.")
    return {
        # max_depth POR CLASE (afinado para recall de las clases de riesgo con
        # validación temporal -- ver train_model._RF_MAX_DEPTH_BY_CLASE y
        # reports/rf_tuning_*.csv).
        "RandomForest": RandomForestClassifier(
            n_estimators=200, max_depth=rf_max_depth[clase], max_features="sqrt",
            max_samples=0.8, class_weight="balanced", random_state=42, n_jobs=-1,
        ),
        "XGBoost": XGBClassifier(
            n_estimators=300, max_depth=6, learning_rate=0.05, subsample=0.8,
            colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1.0,
            eval_metric="logloss", random_state=42, n_jobs=-1,
        ),
        "KNN": KNeighborsClassifier(n_neighbors=knn_k, weights="distance", n_jobs=-1),
    }


def _prepare_fold(df_train_raw: pd.DataFrame, df_test_raw: pd.DataFrame, target_col: str, clase: str):
    """
    Aplica el mismo pipeline de preprocess_data() a un fold train/test,
    pero ajustando (fit) imputación/encoders/escalado SOLO con el fold de
    train y aplicándolo (transform) al fold de test -- para que no haya
    fuga de información de test hacia ningún paso de ajuste (ni siquiera
    la media usada para rellenar nulos).
    """
    df_train = _feature_engineering(df_train_raw.copy())
    df_test = _feature_engineering(df_test_raw.copy())

    df_train = _apply_logcols(df_train, LOGCOLS)
    df_test = _apply_logcols(df_test, LOGCOLS)

    for col, mapping in ORDINAL_MAPPINGS.items():
        if col in df_train.columns:
            df_train[col] = df_train[col].map(mapping)
            df_test[col] = df_test[col].map(mapping)

    cols_a_eliminar = list(COLS_TO_DROP) + LEAKAGE_COLS_BY_CLASE[clase]
    cols_presentes = [c for c in cols_a_eliminar if c in df_train.columns and c != target_col]
    df_train = df_train.drop(columns=cols_presentes, errors="ignore")
    df_test = df_test.drop(columns=[c for c in cols_presentes if c in df_test.columns], errors="ignore")

    X_train = df_train.drop(columns=[target_col])
    y_train = df_train[target_col]
    X_test = df_test.drop(columns=[target_col])
    y_test = df_test[target_col]

    num_cols = X_train.select_dtypes(include=[np.number]).columns
    cat_cols = X_train.select_dtypes(exclude=[np.number]).columns

    # Nulos: media/moda calculadas SOLO en train, aplicadas también a test
    medias = X_train[num_cols].mean()
    X_train[num_cols] = X_train[num_cols].fillna(medias)
    X_test[num_cols] = X_test[num_cols].fillna(medias)

    for col in cat_cols:
        moda = X_train[col].mode()[0]
        X_train[col] = X_train[col].fillna(moda)
        encoder = LabelEncoder()
        X_train[col] = encoder.fit_transform(X_train[col].astype(str))

        # Categorías del test que el encoder de train nunca vio -> moda de train
        # (en vez de que .transform() reviente con un ValueError de categoría desconocida)
        test_vals = X_test[col].fillna(moda).astype(str)
        conocidas = set(encoder.classes_)
        test_vals = test_vals.where(test_vals.isin(conocidas), str(moda))
        X_test[col] = encoder.transform(test_vals)

    if y_train.dtype == object or str(y_train.dtype) == "category":
        le_target = LabelEncoder()
        y_train = pd.Series(le_target.fit_transform(y_train.astype(str)), index=y_train.index)
        y_test = pd.Series(le_target.transform(y_test.astype(str)), index=y_test.index)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test


def evaluate_models_temporal_cv(
    df: pd.DataFrame,
    target_col: str,
    clase: str = "calor",
    min_train_years: int = 3,
    knn_k: int = 5,
) -> pd.DataFrame:
    """
    Validación cruzada temporal por años (ventana expansiva): entrena con
    todos los años hasta N-1 y evalúa en el año N, deslizando N desde
    `min_train_years` hasta el último año disponible.

    A diferencia de un único split train/test, esto da media y desviación
    por modelo a lo largo de varios años distintos, en vez de depender de
    si el último tramo de fechas resultó ser fácil o difícil por
    casualidad.

    Parameters
    ----------
    df : pd.DataFrame
        Salida de dataset_calor()/dataset_frio() + asignar_clase_riesgo_*()
        -- debe tener 'fecha' y `target_col`.
    min_train_years : int
        Años mínimos de histórico antes del primer fold de test (por
        defecto 3) -- evita folds iniciales con muy poco train.
    knn_k : int
        k fijo para KNN en todos los folds -- ver nota en _build_models_cv.

    Returns
    -------
    pd.DataFrame
        Una fila por (modelo, fold), con el año de test y las métricas.
        Pásalo a resumen_temporal_cv() para la media/desviación agregada.
    """
    df = df.copy()
    df["_año"] = pd.to_datetime(df["fecha"]).dt.year
    años = sorted(df["_año"].unique())

    if len(años) <= min_train_years:
        raise ValueError(
            f"Solo hay {len(años)} años distintos en 'fecha', pero min_train_years="
            f"{min_train_years} -- no queda ningún año para test. Reduce min_train_years."
        )

    resultados = []

    for i in range(min_train_years, len(años)):
        año_test = años[i]
        años_train = años[:i]

        df_train_raw = df[df["_año"].isin(años_train)].drop(columns=["_año"])
        df_test_raw = df[df["_año"] == año_test].drop(columns=["_año"])

        print(
            f"--> Fold: train {años_train[0]}-{años_train[-1]} ({len(df_train_raw)} filas) "
            f"| test {año_test} ({len(df_test_raw)} filas)"
        )

        X_train, X_test, y_train, y_test = _prepare_fold(df_train_raw, df_test_raw, target_col, clase)

        # Modelos nuevos en cada fold (no reutilizar instancias ya entrenadas
        # de un fold anterior -- aunque .fit() resetea el estado interno de
        # estos estimadores, instanciar de cero por fold evita depender de
        # ese detalle de implementación de sklearn/xgboost).
        modelos = _build_models_cv(knn_k=knn_k, clase=clase)

        for nombre, modelo in modelos.items():
            modelo.fit(X_train, y_train)
            y_pred_train = modelo.predict(X_train)
            y_pred_test = modelo.predict(X_test)

            resultados.append({
                "Modelo": nombre,
                "año_test": año_test,
                "n_train": len(y_train),
                "n_test": len(y_test),
                "Acc_train": accuracy_score(y_train, y_pred_train),
                "Acc_test": accuracy_score(y_test, y_pred_test),
                "F1_train": f1_score(y_train, y_pred_train, average="weighted", zero_division=0),
                "F1_test": f1_score(y_test, y_pred_test, average="weighted", zero_division=0),
                "Prec_test": precision_score(y_test, y_pred_test, average="weighted", zero_division=0),
                "Rec_test": recall_score(y_test, y_pred_test, average="weighted", zero_division=0),
            })

    return pd.DataFrame(resultados)


def resumen_temporal_cv(df_resultados: pd.DataFrame) -> pd.DataFrame:
    """
    Media ± desviación estándar de cada métrica, agregada por modelo a lo
    largo de todos los folds -- la tabla que de verdad importa para
    comparar modelos, no los folds sueltos (que son ruidosos por
    separado).
    """
    metricas = ["Acc_train", "Acc_test", "F1_train", "F1_test", "Prec_test", "Rec_test"]
    resumen = df_resultados.groupby("Modelo")[metricas].agg(["mean", "std"])
    resumen.columns = [f"{m}_{stat}" for m, stat in resumen.columns]
    return resumen.reset_index()


def tune_randomforest_temporal_cv(
    df: pd.DataFrame,
    target_col: str,
    clase: str = "calor",
    param_grid: list[dict] | None = None,
    min_train_years: int = 3,
    scoring: str = "F1_test",
) -> pd.DataFrame:
    """
    Busca hiperparámetros de RandomForest evaluando cada combinación con
    la MISMA validación cruzada temporal por años que evaluate_models_temporal_cv()
    -- no un GridSearchCV con split aleatorio, que sería inconsistente con
    la conclusión de que RandomForest necesita evaluarse por años.

    Motivación (ver discusión de resultados): RandomForest salió con
    Acc_train bajo (~0.75, no sobreajuste, simplemente no está aprendiendo
    bien) y Prec_test alta pero Rec_test baja -- síntomas compatibles con
    max_depth demasiado superficial y/o class_weight="balanced" penalizando
    de más. Este grid explora ambas hipótesis.

    Parameters
    ----------
    param_grid : list[dict], opcional
        Lista de diccionarios de hiperparámetros a probar (se pasan
        directamente a RandomForestClassifier). Si no se pasa, usa un
        grid por defecto que cruza max_depth x class_weight (ver código).
    min_train_years : int
        Igual que en evaluate_models_temporal_cv().
    scoring : str
        Columna de _prepare_fold/evaluate_models_temporal_cv por la que
        ordenar los resultados (por defecto F1_test, razonable con
        clases desequilibradas -- cambia a 'Rec_test' si te preocupa más
        no perderte casos de "peligro" que los falsos positivos).

    Returns
    -------
    pd.DataFrame
        Una fila por combinación de hiperparámetros, con la media y
        desviación de cada métrica a lo largo de los folds, ordenada de
        mejor a peor según `scoring`. La primera fila es la recomendada.
    """
    if param_grid is None:
        param_grid = [
            {"max_depth": max_depth, "class_weight": class_weight,
             "n_estimators": 200, "max_features": "sqrt", "max_samples": 0.8}
            for max_depth in [10, 15, 20, None]
            for class_weight in [None, "balanced"]
        ]

    df = df.copy()
    df["_año"] = pd.to_datetime(df["fecha"]).dt.year
    años = sorted(df["_año"].unique())

    if len(años) <= min_train_years:
        raise ValueError(
            f"Solo hay {len(años)} años distintos en 'fecha', pero min_train_years="
            f"{min_train_years} -- no queda ningún año para test. Reduce min_train_years."
        )

    resultados = []

    for params in param_grid:
        print(f"--> Probando RandomForest con {params}...")
        fold_metrics = []

        for i in range(min_train_years, len(años)):
            año_test = años[i]
            años_train = años[:i]

            df_train_raw = df[df["_año"].isin(años_train)].drop(columns=["_año"])
            df_test_raw = df[df["_año"] == año_test].drop(columns=["_año"])

            X_train, X_test, y_train, y_test = _prepare_fold(df_train_raw, df_test_raw, target_col, clase)

            modelo = RandomForestClassifier(random_state=42, n_jobs=-1, **params)
            modelo.fit(X_train, y_train)
            y_pred_train = modelo.predict(X_train)
            y_pred_test = modelo.predict(X_test)

            fold_metrics.append({
                "Acc_train": accuracy_score(y_train, y_pred_train),
                "Acc_test": accuracy_score(y_test, y_pred_test),
                "F1_train": f1_score(y_train, y_pred_train, average="weighted", zero_division=0),
                "F1_test": f1_score(y_test, y_pred_test, average="weighted", zero_division=0),
                "Prec_test": precision_score(y_test, y_pred_test, average="weighted", zero_division=0),
                "Rec_test": recall_score(y_test, y_pred_test, average="weighted", zero_division=0),
            })

        fold_df = pd.DataFrame(fold_metrics)
        fila = {f"{col}_mean": fold_df[col].mean() for col in fold_df.columns}
        fila.update({f"{col}_std": fold_df[col].std() for col in fold_df.columns})
        fila["params"] = params
        resultados.append(fila)

    resultado_df = pd.DataFrame(resultados).sort_values(f"{scoring}_mean", ascending=False).reset_index(drop=True)
    print(f"\n--> Mejor combinación ({scoring}_mean={resultado_df.iloc[0][f'{scoring}_mean']:.3f}): "
          f"{resultado_df.iloc[0]['params']}")
    return resultado_df