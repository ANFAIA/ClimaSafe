
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.decomposition import PCA
import joblib
from loguru import logger
from climasafeai.utils.paths import PROCESSED_DATA_DIR, ARTIFACTS_DIR


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
    # "duration",     # fuga de datos
    # "nr_employed",  # alta correlación con euribor3m
]

# Columnas a las que aplicar transformación logarítmica (np.log1p).
# Útil para features con distribución muy sesgada (skewness > 1).
# Ejemplo: ["amount", "salary", "tenure_days"]
LOGCOLS: list = []


def preprocess_data(
    df: pd.DataFrame,
    target_col: str,
    scaler_type: str = "standard",
    test_size: float = 0.2,
    random_state: int = 42,
    use_pca=None,
    tipo="calor"
):
    """
    Pipeline completo de preprocesado para aprendizaje supervisado.

    Pasos:
      1. Elimina duplicados
      2. Feature engineering personalizable (_feature_engineering)
      3. Codificación ordinal (ORDINAL_MAPPINGS)
      4. Elimina columnas no deseadas (COLS_TO_DROP)
      5. Rellena nulos (media/moda)
      6. LabelEncoder para categóricas
      7. Train/test split estratificado
      8. Escalado (StandardScaler o MinMaxScaler)
      9. PCA opcional (use_pca)
      10. Guarda artefactos en artifacts/

    Parameters
    ----------
    scaler_type : "standard" | "minmax"
    use_pca     : None → sin PCA
                  float (0 < n < 1) → nº componentes por varianza explicada, e.g. 0.95
                  int  → nº fijo de componentes, e.g. 10

    Returns
    -------
    X_train, X_test, y_train, y_test  (arrays numpy)
    """
    print(f"--> Preprocesando datos (target='{target_col}', scaler='{scaler_type}', PCA={use_pca})...")

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

    # 4. Eliminar columnas
    cols_present = [c for c in COLS_TO_DROP if c in df.columns]
    if cols_present:
        df.drop(columns=cols_present, inplace=True)
        print(f"    Columnas eliminadas: {cols_present}")

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
        joblib.dump(le_target, ARTIFACTS_DIR / "target_encoder.joblib")
        print("    Target codificado → target_encoder.joblib")

    # Guardar todos los encoders en un único joblib (reproducibilidad inferencia)
    joblib.dump(encoders, ARTIFACTS_DIR / "encoders.joblib")
    if encoders:
        cols_encoded = [c for c in encoders if c != "__target__"]
        print(f"    Encoders guardados → encoders.joblib  ({cols_encoded})")


    # 8. Train/test split (sin stratify, es regresión)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state,
    )

    # Submuestreo de la mitad de los ceros SOLO en entrenamiento
    rng = np.random.default_rng(random_state)
    mask_ceros = (y_train == 0)
    idx_ceros = y_train[mask_ceros].index        # pd.Index
    n_conservar = len(idx_ceros) // 2
    # Elegir la mitad de los índices y convertir a pd.Index para poder usar .union
    idx_ceros_sub = pd.Index(rng.choice(idx_ceros, size=n_conservar, replace=False))
    idx_pos = y_train[~mask_ceros].index         # pd.Index
    idx_keep = idx_ceros_sub.union(idx_pos)      # pd.Index combinado

    X_train = X_train.loc[idx_keep]
    y_train = y_train.loc[idx_keep]
    print(f"    Muestras de entrenamiento tras submuestreo: {len(X_train)}")
    print(f"    Muestras de entrenamiento tras submuestreo: {len(X_train)}")

    # Guardar nombres de features originales (antes de PCA) para test_model()
    joblib.dump(list(X.columns), ARTIFACTS_DIR / "feature_names.joblib")
    print(f"    feature_names.joblib guardado ({len(X.columns)} features)")

    # 9. Escalado
    scaler = MinMaxScaler() if scaler_type == "minmax" else StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)
    joblib.dump(scaler, ARTIFACTS_DIR / "scaler.joblib")
    print(f"    Scaler guardado → scaler.joblib")

    # 10. PCA opcional
    if use_pca is not None:
        X_train, X_test = _apply_pca(X_train, X_test, use_pca)

    print(f"    Train: {X_train.shape} | Test: {X_test.shape}")

    print(f"    Proporción clases (train): {y_train.value_counts(normalize=True).to_dict()}")


    # Guardar conjuntos procesados
    pd.DataFrame(X_train).to_csv(PROCESSED_DATA_DIR / f"X_train_{tipo}.csv", index=False)
    pd.DataFrame(X_test).to_csv(PROCESSED_DATA_DIR  / f"X_test_{tipo}.csv",  index=False)
    pd.Series(y_train).to_csv(PROCESSED_DATA_DIR / f"y_train_{tipo}.csv", index=False)
    pd.Series(y_test).to_csv(PROCESSED_DATA_DIR  / f"y_test_{tipo}.csv",  index=False)

    return X_train, X_test, y_train, y_test


def _apply_pca(X_train, X_test, n_components):
    """
    Aplica PCA a train/test y guarda el objeto PCA en artifacts/.

    Parameters
    ----------
    n_components : float (varianza) | int (componentes fijos)
                   Ejemplos: 0.95 → 95% varianza | 10 → 10 componentes

    ¿Cuándo usar PCA antes del clasificador?
      - Muchas features correladas (|r| > 0.8 en varios pares)
      - Alta dimensionalidad (>50 features) → riesgo de maldición dimensional
      - Modelos lentos en alta dimensión (SVM, KNN)
      - Datos con ruido: PCA elimina las componentes de menor varianza

    ¿Cuándo NO usar PCA?
      - Cuando la interpretabilidad de features es crítica
      - Árboles y ensembles (RandomForest, XGBoost): ya gestionan la
        dimensionalidad internamente; PCA no suele mejorar resultados
    """
    pca = PCA(n_components=n_components, random_state=42)
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca  = pca.transform(X_test)
    joblib.dump(pca, ARTIFACTS_DIR / "pca.joblib")

    n_comp = pca.n_components_
    var_exp = pca.explained_variance_ratio_.sum()
    print(f"    PCA: {X_train.shape[1]} → {n_comp} componentes "
          f"({var_exp:.1%} varianza explicada) → pca.joblib")
    return X_train_pca, X_test_pca


def _feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transformaciones y nuevas variables antes del modelado.
    Edita esta función según las necesidades del problema.

    Ejemplos comunes:
      df['was_contacted'] = df['pdays'].apply(lambda x: 0 if x == 999 else 1)
      df['total_loans']   = df['housing'] + df['loan']
    """
    # --- Añade tus transformaciones aquí ---
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


def process_input(df_new: pd.DataFrame) -> np.ndarray:
    """
    Preprocesa nuevos datos para inferencia usando los artefactos guardados.
    Aplica: feature_engineering → ordinal → drop → encode (encoders.joblib)
            → scaler → PCA (si existe).

    Los encoders.joblib garantizan que el mapping de categorías sea idéntico
    al del entrenamiento, evitando silenciosos errores de codificación.
    """
    import os
    scaler   = joblib.load(ARTIFACTS_DIR / "scaler.joblib")
    encoders = joblib.load(ARTIFACTS_DIR / "encoders.joblib") if (ARTIFACTS_DIR / "encoders.joblib").exists() else {}

    df_new = df_new.copy()
    df_new = _feature_engineering(df_new)
    df_new = _apply_logcols(df_new, LOGCOLS)

    for col, mapping in ORDINAL_MAPPINGS.items():
        if col in df_new.columns:
            df_new[col] = df_new[col].map(mapping)

    cols_present = [c for c in COLS_TO_DROP if c in df_new.columns]
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

    pca_path = ARTIFACTS_DIR / "pca.joblib"
    if pca_path.exists():
        pca = joblib.load(pca_path)
        X = pca.transform(X)

    return X


