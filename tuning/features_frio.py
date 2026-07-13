"""
Experimento: features de FRÍO con más memoria temporal (retardo epidemiológico).

Motivación
----------
Al pasar de 19 a 27 features, el calor mejoró (Rec_riesgo 0.614 -> 0.633) pero
el frío se quedó igual (0.527 -> 0.526). Hipótesis: la mortalidad por frío
tiene un retardo largo (días-semanas tras la exposición, a diferencia del
calor, que es de lag corto) y las ventanas actuales (roll3/7/14) se quedan
cortas. Este script construye candidatas con más memoria (21-28 días, ventanas
desplazadas t-7..t-21, mínimas nocturnas) y reentrena los modelos de frío con
los MISMOS hiperparámetros de producción para medir si aportan.

Reglas
------
- Solo PASADO: todas las candidatas usan shift(1) por provincia (el día actual
  nunca entra en su propia ventana), igual que _agregar_rezagos_temporales de
  climasafeai/data/make_dataset.py. Sin fuga de datos.
- Split temporal: último 20% de fechas distintas a test, replicando
  preprocess_data() de climasafeai/features/build_features.py (incluido su
  fillna con la media y su StandardScaler).
- Modelos: hiperparámetros clonados de models/RandomForest_frio.joblib y
  models/XGBoost_frio.joblib (n_jobs limitado a 2). XGBoost con
  sample_weight=compute_sample_weight("balanced"), como en train_model.py.

Uso
---
    PYTHONPATH=. $MAIN/.venv/bin/python tuning/features_frio.py

Necesita en data/processed/ (copias locales o symlinks del checkout principal):
    dataset_frio_labeled.parquet   dataset diario por provincia (27 features)
    secuencias_24h.npz             series horarias 24h [t2m_c, rh,
                                   wind_speed_kmh, heat_index_c, wind_chill_c]

Salida: tabla de métricas por variante (stdout) y CSVs en reports/ con
métricas e importancias (reports/features_frio_retardo_*.csv).
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, recall_score
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

# ---------------------------------------------------------------------------
# Configuración
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "processed"
REPORTS_DIR = PROJECT_ROOT / "reports"

PARQUET = DATA_DIR / "dataset_frio_labeled.parquet"
NPZ = DATA_DIR / "secuencias_24h.npz"

TARGET = "clase_riesgo_frio"
N_JOBS = 2
TEST_SIZE = 0.2
RANDOM_STATE = 42

# Mismo umbral que make_dataset.WIND_CHILL_UMBRAL_C (grados-día de frío =
# suma de max(umbral - wind_chill_mean, 0) sobre los N días previos).
WIND_CHILL_UMBRAL_C = 0.0
# Umbral "frío severo" para la racha alternativa (wind chill horario < -5 °C,
# más exigente que el 0 °C de horas_bajo_umbral).
UMBRAL_SEVERO_C = -5.0

# Columnas que NO son features (identificadores + fuga de datos), replicando
# COLS_TO_DROP + LEAKAGE_COLS_BY_CLASE["frio"] de build_features.py.
NO_FEATURES = [
    "provincia", "fecha", "datetime",
    "clase_riesgo_calor", "clase_riesgo_calor_label", "clase_riesgo_frio_label",
    "defunciones_atrib_exc_temp", "defunciones_atrib_def_temp",
]

# Hiperparámetros clonados de models/*_frio.joblib (solo se fija n_jobs).
RF_PARAMS = dict(
    n_estimators=200, max_depth=8, max_features="sqrt", max_samples=0.8,
    class_weight="balanced", random_state=RANDOM_STATE, n_jobs=N_JOBS,
)
XGB_PARAMS = dict(
    n_estimators=300, max_depth=6, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1.0,
    eval_metric="logloss", random_state=RANDOM_STATE, n_jobs=N_JOBS,
)


# ---------------------------------------------------------------------------
# Features candidatas (solo pasado: shift(1) por provincia)
# ---------------------------------------------------------------------------
def _features_nocturnas(npz_path: Path) -> pd.DataFrame:
    """
    Mínima nocturna diaria por (provincia, fecha) desde secuencias_24h.npz.

    t2m_min_noche = mínimo de t2m_c en las horas 0-8 del día (madrugada).
    La mortalidad por frío se asocia a las mínimas nocturnas sostenidas más
    que al wind chill del pico de riesgo diurno que ya recogen las features
    actuales. También se extrae el nº de horas con wind_chill_c < -5 °C
    (frío severo) para construir rachas con umbral más exigente.
    """
    d = np.load(npz_path, allow_pickle=True)
    cols = list(d["feature_cols"])
    i_t2m = cols.index("t2m_c")
    i_wc = cols.index("wind_chill_c")
    X = d["X"]  # (n, 24, n_vars)
    return pd.DataFrame({
        "provincia": d["provincias"],
        "fecha": pd.to_datetime(pd.Series(d["fechas"])),
        "t2m_min_noche": X[:, 0:9, i_t2m].min(axis=1),
        "horas_wc_severo": (X[:, :, i_wc] < UMBRAL_SEVERO_C).sum(axis=1),
    })


def _racha_previa(col: pd.Series) -> pd.Series:
    """Días consecutivos ANTERIORES con col > 0 (copia de make_dataset)."""
    activo = (col > 0).astype(int)
    bloques = (activo != activo.shift()).cumsum()
    incl = activo.groupby(bloques).cumcount().add(1).where(activo == 1, 0)
    return incl.shift(1).fillna(0)


def build_candidatas(df: pd.DataFrame, npz_path: Path) -> tuple[pd.DataFrame, list]:
    """
    Añade las features candidatas de "memoria larga" al dataset diario.
    Devuelve (df, lista de nombres de las candidatas).

    Grupos (todos con shift(1) por provincia -> el día t solo ve t-1 y antes):
      memoria_larga : mismas familias que roll7/14 pero a 21 y 28 días
          wind_chill_mean_roll21/28, grados_dia_frio_roll21/28,
          horas_bajo_umbral_sum14/28
      desplazadas   : ventanas que EXCLUYEN la última semana, para aislar la
          exposición retardada de la reciente (que ya cubren roll3/7)
          wind_chill_mean_t7_21  = media de wind_chill_mean en [t-21, t-7]
          wind_chill_mean_t14_28 = media en [t-28, t-14]
      nocturnas     : mínimas de madrugada desde el npz horario
          t2m_min_noche_lag1, t2m_min_noche_roll7
      rachas        : persistencia con umbral más frío
          dias_consec_wc_severo (racha de días con horas de wind chill < -5)
          horas_wc_severo_sum14 (horas de frío severo acumuladas en 14 días)
    """
    df = df.sort_values(["provincia", "fecha"]).copy()
    df["fecha"] = pd.to_datetime(df["fecha"])

    # --- nocturnas (npz horario) ---
    noct = _features_nocturnas(npz_path)
    df = df.merge(noct, on=["provincia", "fecha"], how="left")
    df = df.sort_values(["provincia", "fecha"])  # el merge no garantiza orden

    g = df.groupby("provincia", sort=False)
    candidatas: list = []

    def add(name: str, serie: pd.Series):
        df[name] = serie
        candidatas.append(name)

    # --- memoria_larga: rolls de 21/28 días (solo pasado) ---
    wc = g["wind_chill_mean"]
    add("wind_chill_mean_roll21", wc.transform(lambda s: s.shift(1).rolling(21, min_periods=1).mean()))
    add("wind_chill_mean_roll28", wc.transform(lambda s: s.shift(1).rolling(28, min_periods=1).mean()))

    df["_deficit_frio"] = (WIND_CHILL_UMBRAL_C - df["wind_chill_mean"]).clip(lower=0)
    gd = df.groupby("provincia", sort=False)["_deficit_frio"]
    add("grados_dia_frio_roll21", gd.transform(lambda s: s.shift(1).rolling(21, min_periods=1).sum()))
    add("grados_dia_frio_roll28", gd.transform(lambda s: s.shift(1).rolling(28, min_periods=1).sum()))
    df.drop(columns="_deficit_frio", inplace=True)

    hb = df.groupby("provincia", sort=False)["horas_bajo_umbral"]
    add("horas_bajo_umbral_sum14", hb.transform(lambda s: s.shift(1).rolling(14, min_periods=1).sum()))
    add("horas_bajo_umbral_sum28", hb.transform(lambda s: s.shift(1).rolling(28, min_periods=1).sum()))

    # --- desplazadas: [t-21, t-7] y [t-28, t-14] ---
    # shift(7) + rolling(15) = media de los días t-7..t-21 (15 días).
    add("wind_chill_mean_t7_21", wc.transform(lambda s: s.shift(7).rolling(15, min_periods=1).mean()))
    add("wind_chill_mean_t14_28", wc.transform(lambda s: s.shift(14).rolling(15, min_periods=1).mean()))

    # --- nocturnas: solo pasado (lag1 / roll7 sobre días previos) ---
    tn = df.groupby("provincia", sort=False)["t2m_min_noche"]
    add("t2m_min_noche_lag1", tn.transform(lambda s: s.shift(1)))
    add("t2m_min_noche_roll7", tn.transform(lambda s: s.shift(1).rolling(7, min_periods=1).mean()))

    # --- rachas con umbral severo ---
    hs = df.groupby("provincia", sort=False)["horas_wc_severo"]
    add("dias_consec_wc_severo", hs.transform(_racha_previa))
    add("horas_wc_severo_sum14", hs.transform(lambda s: s.shift(1).rolling(14, min_periods=1).sum()))

    # t2m_min_noche y horas_wc_severo del propio día NO se usan como features
    # (regla del experimento: candidatas solo con pasado).
    df.drop(columns=["t2m_min_noche", "horas_wc_severo"], inplace=True)

    return df, candidatas


# ---------------------------------------------------------------------------
# Split temporal + escalado (réplica de build_features.preprocess_data)
# ---------------------------------------------------------------------------
def split_temporal(df: pd.DataFrame, feature_cols: list):
    """
    Réplica del preprocesado de preprocess_data() (sin guardar artefactos):
    duplicados -> fillna con la media -> último 20% de fechas a test ->
    StandardScaler ajustado en train.
    """
    df = df.drop_duplicates().copy()
    fechas = pd.to_datetime(df["fecha"])

    X = df[feature_cols].copy()
    y = df[TARGET].copy()
    X = X.fillna(X.mean(numeric_only=True))

    fechas_unicas = np.sort(fechas.unique())
    n_test = max(1, round(len(fechas_unicas) * TEST_SIZE))
    fechas_test = set(fechas_unicas[-n_test:])
    mask_test = fechas.isin(fechas_test)

    X_train, X_test = X[~mask_test], X[mask_test]
    y_train, y_test = y[~mask_test].to_numpy(), y[mask_test].to_numpy()

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    return X_train, X_test, y_train, y_test


# ---------------------------------------------------------------------------
# Entrenamiento y métricas
# ---------------------------------------------------------------------------
def evaluar(y_test, y_pred) -> dict:
    risk = [c for c in np.unique(y_test) if c != 0]
    rec = recall_score(y_test, y_pred, average=None, labels=[0, 1, 2], zero_division=0)
    return {
        "Rec_riesgo": recall_score(y_test, y_pred, labels=risk, average="macro", zero_division=0),
        "F1_macro": f1_score(y_test, y_pred, average="macro", zero_division=0),
        "rec_c0": rec[0], "rec_c1": rec[1], "rec_c2": rec[2],
    }


def run_variante(nombre, df, feature_cols):
    """Entrena RF + XGB con `feature_cols` y devuelve métricas e importancias."""
    X_train, X_test, y_train, y_test = split_temporal(df, feature_cols)
    filas, importancias = [], {}

    rf = RandomForestClassifier(**RF_PARAMS)
    rf.fit(X_train, y_train)
    filas.append({"variante": nombre, "modelo": "RandomForest",
                  "n_features": len(feature_cols), **evaluar(y_test, rf.predict(X_test))})
    importancias["RandomForest"] = pd.Series(rf.feature_importances_, index=feature_cols)

    xgb = XGBClassifier(**XGB_PARAMS)
    sw = compute_sample_weight("balanced", y_train)
    xgb.fit(X_train, y_train, sample_weight=sw)
    filas.append({"variante": nombre, "modelo": "XGBoost",
                  "n_features": len(feature_cols), **evaluar(y_test, xgb.predict(X_test))})
    importancias["XGBoost"] = pd.Series(xgb.feature_importances_, index=feature_cols)

    return filas, importancias


def main():
    df = pd.read_parquet(PARQUET)
    df, candidatas = build_candidatas(df, NPZ)

    base27 = [c for c in df.columns if c not in NO_FEATURES + [TARGET] + candidatas]
    print(f"Features base ({len(base27)}): {base27}")
    print(f"Candidatas ({len(candidatas)}): {candidatas}\n")

    grupos = {
        "memoria_larga": ["wind_chill_mean_roll21", "wind_chill_mean_roll28",
                          "grados_dia_frio_roll21", "grados_dia_frio_roll28",
                          "horas_bajo_umbral_sum14", "horas_bajo_umbral_sum28"],
        "desplazadas": ["wind_chill_mean_t7_21", "wind_chill_mean_t14_28"],
        "nocturnas": ["t2m_min_noche_lag1", "t2m_min_noche_roll7"],
        "rachas_severas": ["dias_consec_wc_severo", "horas_wc_severo_sum14"],
    }

    variantes = {"baseline_27": base27,
                 "todas_39": base27 + candidatas}
    for g, cols in grupos.items():
        variantes[f"base+{g}"] = base27 + cols
    # Combinación de los dos grupos que destacaron por separado (ver
    # documentacion/features_frio_retardo.md): nocturnas + rachas severas.
    variantes["base+noct+rachas"] = base27 + grupos["nocturnas"] + grupos["rachas_severas"]

    resultados, imp_todas = [], None
    for nombre, cols in variantes.items():
        print(f"--> {nombre} ({len(cols)} features)")
        filas, imps = run_variante(nombre, df, cols)
        resultados.extend(filas)
        for f in filas:
            print(f"    {f['modelo']:<13} Rec_riesgo={f['Rec_riesgo']:.4f}  "
                  f"F1_macro={f['F1_macro']:.4f}  "
                  f"(rec c0/c1/c2 = {f['rec_c0']:.3f}/{f['rec_c1']:.3f}/{f['rec_c2']:.3f})")
        if nombre == "todas_39":
            imp_todas = imps

    # Variante extra: base27 + top-4 candidatas por importancia RF (con todas)
    imp_rf_cand = imp_todas["RandomForest"][candidatas].sort_values(ascending=False)
    top4 = list(imp_rf_cand.index[:4])
    print(f"\n--> base+top4_importancia (top-4 candidatas RF: {top4})")
    filas, _ = run_variante("base+top4_importancia", df, base27 + top4)
    resultados.extend(filas)
    for f in filas:
        print(f"    {f['modelo']:<13} Rec_riesgo={f['Rec_riesgo']:.4f}  "
              f"F1_macro={f['F1_macro']:.4f}  "
              f"(rec c0/c1/c2 = {f['rec_c0']:.3f}/{f['rec_c1']:.3f}/{f['rec_c2']:.3f})")

    # --- salida ---
    res = pd.DataFrame(resultados).round(4)
    REPORTS_DIR.mkdir(exist_ok=True)
    res.to_csv(REPORTS_DIR / "features_frio_retardo_metricas.csv", index=False)

    imp = pd.DataFrame({m: s for m, s in imp_todas.items()}).round(4)
    imp.sort_values("RandomForest", ascending=False).to_csv(
        REPORTS_DIR / "features_frio_retardo_importancias.csv")

    print("\n===== RESUMEN =====")
    print(res.to_string(index=False))
    print("\nImportancia de las candidatas (variante todas_39):")
    print(imp.loc[candidatas].sort_values("RandomForest", ascending=False).to_string())
    print(f"\nCSVs guardados en {REPORTS_DIR}/features_frio_retardo_*.csv")


if __name__ == "__main__":
    main()
