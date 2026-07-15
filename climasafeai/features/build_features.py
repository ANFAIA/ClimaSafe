import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
import joblib
from loguru import logger
from climasafeai.utils.paths import PROCESSED_DATA_DIR, ARTIFACTS_DIR
from climasafeai.features.weather_indices import add_weather_index_columns


# ---------------------------------------------------------------------------
# Features adicionales a excluir por clase (resultados de ablación 27v19)
# ---------------------------------------------------------------------------
# La ablación demostró que las 8 features de persistencia avanzada (Grupo D,
# features 20-27) benefician a calor (+0.007 Rec_riesgo) pero dañan a frío
# (-0.020 Rec_riesgo). Para frío se eliminan, quedando 19 features base.
# Ver documentacion/ablacion_features_27v19.md y tuning/ablacion_27v19.py.
FRIO_EXTRA_COLS: list = [
    "grados_dia_calor_roll7", "grados_dia_calor_roll14",
    "wind_chill_mean_roll3", "wind_chill_mean_roll7", "wind_chill_mean_roll14",
    "grados_dia_frio_roll7", "grados_dia_frio_roll14",
    "dias_consec_bajo_umbral",
]
COLS_TO_DROP_BY_CLASE: dict = {
    "calor": [
        # Nocturnas y rachas severas (solo relevantes para frío)
        "t2m_min_noche_lag1", "t2m_min_noche_roll7",
        "dias_consec_wc_severo", "horas_wc_severo_sum14",
    ],
    "frio": FRIO_EXTRA_COLS,
}

# ---------------------------------------------------------------------------
# Configuración de codificación ordinal
# ---------------------------------------------------------------------------
ORDINAL_MAPPINGS: dict = {
    # Ejemplo:
    # "education": {
    #     "illiterate": 1, "basic.4y": 2, "basic.6y": 3, "basic.9y": 4,
    #     "high.school": 5, "professional.course": 6, "university.degree": 7,
    #     "unknown": 8,
    # },
}

COLS_TO_DROP: list = [
    # --- Sesgo geográfico (ver spatial_integration.py / diseño_modelo.md) ---
    # 'provincia', 'lat', 'lon' se mantienen en el dataset combinado como
    # identificadores (unir ERA5+MoMo, depurar, visualizar), pero NO deben
    # entrar como features de entrenamiento: el modelo debe aprender de
    # condiciones meteorológicas, no de "dónde" (evita que aprenda
    # "Madrid → riesgo alto" en vez de "38°C + poca humedad → riesgo alto").
    # --- Sesgo temporal ---
    # 'fecha' también se mantiene como identificador (split train/test por
    # fechas, trazabilidad, orden temporal) pero se excluye como feature
    # por el mismo motivo que la provincia: si el modelo ve la fecha
    # exacta, puede memorizar "esta fecha tuvo mucha mortalidad" (una ola
    # de calor puntual, un efecto COVID en el histórico de MoMo, etc.) en
    # vez de aprender la relación física temperatura/humedad -> riesgo,
    # y eso no generaliza a fechas futuras.
    # 'datetime' (la hora exacta seleccionada como "hora de mayor riesgo"
    # por select_risk_hour_row) es, en la práctica, casi un identificador
    # único por fila -- LabelEncoder la codificaría como si fuera una
    # categoría sin ningún significado numérico real, y al ser casi única
    # por fila puede actuar como clave de memorización en vez de feature
    # meteorológica.
    "provincia", "fecha", "datetime",
    # Columnas brutas de nocturnas/frío severo (solo se usan sus versiones
    # con lag. La columna del mismo día miraría horas 0-8 del día actual,
    # que no debe ser feature — solo el rezago del día anterior importa).
    "t2m_min_noche", "horas_wc_severo",
]

# Columnas a las que aplicar transformación logarítmica (np.log1p).
# Útil para features con distribución muy sesgada (skewness > 1).
# Ejemplo: ["amount", "salary", "tenure_days"]
LOGCOLS: list = []

# ---------------------------------------------------------------------------
# Índices meteorológicos derivados (Heat Index / WBGT / Wind Chill)
# ---------------------------------------------------------------------------
# Nombres de columna esperados tras la agregación ERA5 + GeoPandas (ver
# climasafeai/data/make_dataset.py y documentacion/diseño_modelo.md sección 3).
# Ajusta estos tres nombres cuando se cierre el esquema final del dataset
# agregado por provincia/día — add_weather_index_columns() avisa (no falla)
# si alguna columna no existe todavía.
WEATHER_TEMP_COL = "t2m_c"
WEATHER_RH_COL = "rh"
WEATHER_WIND_COL = "wind_speed_kmh"


# Columnas que serían fuga de datos si entran a X, según qué modelo
# (calor/frío) se esté entrenando. Se excluyen ADEMÁS de COLS_TO_DROP y
# del propio target_col -- ver preprocess_data(clase=...).
LEAKAGE_COLS_BY_CLASE: dict = {
    "calor": [
        "clase_riesgo_frio",          # la otra clase, no debe verla este modelo
        "clase_riesgo_calor_label",   # versión en texto del propio target
        "clase_riesgo_frio_label",
        "defunciones_atrib_exc_temp", # de aquí sale clase_riesgo_calor -- fuga directa
        "defunciones_atrib_def_temp",
    ],
    "frio": [
        "clase_riesgo_calor",
        "clase_riesgo_calor_label",
        "clase_riesgo_frio_label",
        "defunciones_atrib_exc_temp",
        "defunciones_atrib_def_temp",  # de aquí sale clase_riesgo_frio -- fuga directa
    ],
}


def preprocess_data(
    df: pd.DataFrame,
    target_col: str,
    scaler_type: str = "standard",
    test_size: float = 0.2,
    random_state: int = 42,
    clase: str = "calor",
    split_by_date: bool = True,
):
    """
    Pipeline completo de preprocesado para aprendizaje supervisado.

    Pasos:
      1. Elimina duplicados
      2. Feature engineering personalizable (_feature_engineering)
      3. Codificación ordinal (ORDINAL_MAPPINGS)
      4. Elimina columnas no deseadas (COLS_TO_DROP + LEAKAGE_COLS_BY_CLASE[clase])
      5. Rellena nulos (media/moda)
      6. LabelEncoder para categóricas
      7. Train/test split -- por FECHA (recomendado) o aleatorio estratificado
      8. Escalado (StandardScaler o MinMaxScaler)
      9. Guarda artefactos en artifacts/, namespaceados por `clase`

    Parameters
    ----------
    scaler_type : "standard" | "minmax"
    clase : "calor" | "frio"
        Qué modelo se está entrenando. Controla dos cosas:
        - Qué columnas se excluyen de X como fuga de datos (ver
          LEAKAGE_COLS_BY_CLASE) -- la etiqueta y la mortalidad bruta de
          la que sale NUNCA deben entrar como features.
        - El sufijo de los artefactos guardados (scaler_calor.joblib vs
          scaler_frio.joblib, etc.) para que entrenar un modelo no
          sobrescriba los artefactos del otro.
    split_by_date : bool
        Si True (recomendado, por defecto): el test es el `test_size`
        final de FECHAS distintas (p.ej. los últimos ~20% de días del
        histórico), no una muestra aleatoria de filas. Evita que días de
        la misma ola de calor (correlacionados entre sí) queden
        repartidos entre train y test -- un split aleatorio infla el
        rendimiento aparente de modelos basados en similitud/vecinos
        (KNN), porque el "vecino más parecido" de un día de test puede
        ser literalmente otro día de la misma ola que sí está en train.
        Si False, usa el split aleatorio estratificado de siempre
        (`train_test_split(..., stratify=y)`) -- requiere que `df`
        contenga la columna 'fecha'.

    Returns
    -------
    X_train, X_test, y_train, y_test  (arrays numpy)
    """
    if clase not in LEAKAGE_COLS_BY_CLASE:
        raise ValueError(
            f"clase='{clase}' no reconocida -- debe ser una de {list(LEAKAGE_COLS_BY_CLASE)}"
        )

    print(f"--> Preprocesando datos (clase='{clase}', target='{target_col}', scaler='{scaler_type}')...")

    df = df.copy()

    # 1. Duplicados
    n_before = len(df)
    df.drop_duplicates(inplace=True)
    if n_before - len(df):
        print(f"    Duplicados eliminados: {n_before - len(df)}")

    # 2. Feature engineering
    df = _feature_engineering(df)

    # 2.5 Transformación logarítmica
    df = _apply_logcols(df, LOGCOLS)

    # 3. Codificación ordinal
    for col, mapping in ORDINAL_MAPPINGS.items():
        if col in df.columns:
            df[col] = df[col].map(mapping)
            print(f"    Codificación ordinal: {col}")

    # Capturar 'fecha' ANTES de que COLS_TO_DROP la elimine (paso 4) --
    # split_by_date la necesita para decidir qué filas van a test.
    if split_by_date:
        if "fecha" not in df.columns:
            raise ValueError(
                "preprocess_data: split_by_date=True requiere que 'fecha' "
                "esté en el DataFrame de entrada. Pasa split_by_date=False "
                "si no la tienes, o añádela antes de llamar a esta función."
            )
        fechas_para_split = pd.to_datetime(df["fecha"]).copy()

    # 4. Eliminar columnas (generales + fuga de datos + extra por clase)
    cols_a_eliminar = (
        list(COLS_TO_DROP)
        + LEAKAGE_COLS_BY_CLASE[clase]
        + COLS_TO_DROP_BY_CLASE.get(clase, [])
    )
    cols_presentes = [c for c in cols_a_eliminar if c in df.columns and c != target_col]
    if cols_presentes:
        df.drop(columns=cols_presentes, inplace=True)
        print(f"    Columnas eliminadas ({clase}): {cols_presentes}")

    # 5. X / y
    X = df.drop(columns=[target_col])
    y = df[target_col]

    # 6. Nulos
    num_cols = X.select_dtypes(include=[np.number]).columns
    cat_cols = X.select_dtypes(exclude=[np.number]).columns
    X[num_cols] = X[num_cols].fillna(X[num_cols].mean())
    for col in cat_cols:
        X[col] = X[col].fillna(X[col].mode()[0])

    # 7. LabelEncoder
    encoders = {}  # guardamos un encoder por columna categórica para reproducibilidad
    for col in cat_cols:
        le_col = LabelEncoder()
        X[col] = le_col.fit_transform(X[col].astype(str))
        encoders[col] = le_col

    if y.dtype == object or str(y.dtype) == "category":
        le_target = LabelEncoder()
        y = pd.Series(
            le_target.fit_transform(y.astype(str)),
            index=y.index,
            name=target_col,
        )
        encoders["__target__"] = le_target
        joblib.dump(le_target, ARTIFACTS_DIR / f"target_encoder_{clase}.joblib")
        print(f"    Target codificado → target_encoder_{clase}.joblib")

    # Guardar todos los encoders en un único joblib (reproducibilidad inferencia)
    joblib.dump(encoders, ARTIFACTS_DIR / f"encoders_{clase}.joblib")
    if encoders:
        cols_encoded = [c for c in encoders if c != "__target__"]
        print(f"    Encoders guardados → encoders_{clase}.joblib  ({cols_encoded})")

    if split_by_date:
        fechas_para_split = fechas_para_split.loc[X.index]
        fechas_unicas = np.sort(fechas_para_split.unique())
        n_test_dates = max(1, round(len(fechas_unicas) * test_size))
        fechas_test = set(fechas_unicas[-n_test_dates:])
        mask_test = fechas_para_split.isin(fechas_test)

        X_train, X_test = X[~mask_test], X[mask_test]
        y_train, y_test = y[~mask_test], y[mask_test]

        print(
            f"    Split por fecha: train hasta {fechas_unicas[-n_test_dates-1] if n_test_dates < len(fechas_unicas) else '(sin margen)'} "
            f"| test desde {fechas_unicas[-n_test_dates]} ({len(fechas_test)} días distintos)"
        )
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y,
        )

    # Guardar nombres de features originales para test_model()
    joblib.dump(list(X.columns), ARTIFACTS_DIR / f"feature_names_{clase}.joblib")
    print(f"    feature_names_{clase}.joblib guardado ({len(X.columns)} features)")

    # 9. Escalado
    scaler = MinMaxScaler() if scaler_type == "minmax" else StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)
    joblib.dump(scaler, ARTIFACTS_DIR / f"scaler_{clase}.joblib")
    print(f"    Scaler guardado → scaler_{clase}.joblib")

    # threshold.joblib se genera DESPUÉS del entrenamiento, no aquí.
    # Descomenta find_best_threshold en predict_model.py (solo binaria) y
    # guárdalo al final de train_model.py:
    #   joblib.dump(best_threshold, ARTIFACTS_DIR / f"threshold_{clase}.joblib")

    print(f"    Train: {X_train.shape} | Test: {X_test.shape}")
    print(f"    Proporción clases (train): {y_train.value_counts(normalize=True).to_dict()}")

    # Guardar conjuntos procesados (namespaceados por clase). Se conservan los
    # NOMBRES de columna (X ya trae los nombres reales de las features) para que
    # al releer el CSV, X_train.columns dé los nombres y no 0,1,2,... -- así la
    # importancia de variables sale con nombres. Mismo orden que feature_names_{clase}.joblib.
    pd.DataFrame(X_train, columns=list(X.columns)).to_csv(PROCESSED_DATA_DIR / f"X_train_{clase}.csv", index=False)
    pd.DataFrame(X_test,  columns=list(X.columns)).to_csv(PROCESSED_DATA_DIR / f"X_test_{clase}.csv",  index=False)
    pd.Series(y_train).to_csv(PROCESSED_DATA_DIR / f"y_train_{clase}.csv", index=False)
    pd.Series(y_test).to_csv(PROCESSED_DATA_DIR  / f"y_test_{clase}.csv",  index=False)

    return X_train, X_test, y_train, y_test


def _feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transformaciones y nuevas variables antes del modelado.
    Edita esta función según las necesidades del problema.

    Ejemplos comunes:
      df['was_contacted'] = df['pdays'].apply(lambda x: 0 if x == 999 else 1)
      df['total_loans']   = df['housing'] + df['loan']
    """
    # Heat Index, WBGT y Wind Chill (ver climasafeai/features/weather_indices.py
    # y documentacion/diseño_modelo.md sección 1 — "features principales").
    # Se calculan sobre la fila ya seleccionada como hora de mayor riesgo del
    # día (ver diseño_modelo.md sección 2); aquí solo se derivan como features
    # del modelo, no se recalcula ninguna selección de hora.
    df = add_weather_index_columns(
        df,
        temp_col=WEATHER_TEMP_COL,
        rh_col=WEATHER_RH_COL,
        wind_col=WEATHER_WIND_COL,
    )
    # --- Añade aquí más transformaciones si hacen falta ---
    return df


def _apply_logcols(df: pd.DataFrame, cols: list) -> pd.DataFrame:
    """
    Aplica transformación logarítmica np.log1p() a las columnas indicadas.

    Úsala con features numéricas de distribución muy sesgada (skewness > 1)
    para acercarlas a una distribución normal antes del escalado.

    Parameters
    ----------
    df   : DataFrame con las columnas a transformar.
    cols : Lista de nombres de columna. Las columnas que no existan en df
           se ignoran con un aviso. Ejemplo: LOGCOLS = ["amount", "tenure_days"]

    Notes
    -----
    - np.log1p(x) = log(1 + x) → evita log(0) cuando hay ceros.
    - Para valores negativos, aplica primero un offset: x - x.min() + 1.
    - Configura LOGCOLS en la sección de constantes de este fichero.
    """
    if not cols:
        return df

    df = df.copy()
    applied, skipped = [], []

    for col in cols:
        if col not in df.columns:
            skipped.append(col)
            continue
        if df[col].min() < 0:
            offset = -df[col].min() + 1
            df[col] = np.log1p(df[col] + offset)
            logger.warning(f"logcols | '{col}' tiene valores negativos → offset {offset:.4f} aplicado antes de log1p")
        else:
            df[col] = np.log1p(df[col])
        applied.append(col)

    if applied:
        logger.info(f"logcols | log1p aplicado → {applied}")
    if skipped:
        logger.warning(f"logcols | columnas no encontradas (ignoradas) → {skipped}")

    return df


def process_input(df_new: pd.DataFrame, clase: str = "calor") -> np.ndarray:
    """
    Preprocesa nuevos datos para inferencia usando los artefactos guardados.
    Aplica: feature_engineering → ordinal → drop → encode (encoders_{clase}.joblib)
            → scaler_{clase}.

    Los encoders_{clase}.joblib garantizan que el mapping de categorías sea
    idéntico al del entrenamiento, evitando silenciosos errores de codificación.

    Parameters
    ----------
    clase : "calor" | "frio"
        Qué modelo se usa para inferencia -- debe coincidir con el `clase`
        usado en preprocess_data() al entrenar, para cargar los artefactos
        correctos (scaler_{clase}.joblib, encoders_{clase}.joblib).
    """
    scaler_path = ARTIFACTS_DIR / f"scaler_{clase}.joblib"
    encoders_path = ARTIFACTS_DIR / f"encoders_{clase}.joblib"

    scaler   = joblib.load(scaler_path)
    encoders = joblib.load(encoders_path) if encoders_path.exists() else {}

    df_new = df_new.copy()
    df_new = _feature_engineering(df_new)
    df_new = _apply_logcols(df_new, LOGCOLS)

    for col, mapping in ORDINAL_MAPPINGS.items():
        if col in df_new.columns:
            df_new[col] = df_new[col].map(mapping)

    cols_a_eliminar = list(COLS_TO_DROP) + LEAKAGE_COLS_BY_CLASE.get(clase, [])
    cols_present = [c for c in cols_a_eliminar if c in df_new.columns]
    if cols_present:
        df_new.drop(columns=cols_present, inplace=True)

    cat_cols = df_new.select_dtypes(exclude=[np.number]).columns
    for col in cat_cols:
        if col in encoders:
            # Usar el mismo encoder del entrenamiento → mismo mapping de clases
            le = encoders[col]
            df_new[col] = le.transform(df_new[col].astype(str))
        else:
            # Fallback: re-fit (puede diferir del entrenamiento si hay categorías nuevas)
            le = LabelEncoder()
            df_new[col] = le.fit_transform(df_new[col].astype(str))

    num_cols = df_new.select_dtypes(include=[np.number]).columns
    df_new[num_cols] = df_new[num_cols].fillna(df_new[num_cols].mean())

    X = scaler.transform(df_new)

    return X