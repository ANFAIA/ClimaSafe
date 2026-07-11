"""
climasafeai.data.sequences — dataset de secuencias 24h para la LSTM.

Construye, a partir de los .nc horarios de ERA5 (data/raw/era5/), un tensor
(n_días, 24, n_features) con las variables horarias crudas por provincia/día,
alineado con los labels de los parquets etiquetados (clase_riesgo_calor y
clase_riesgo_frio). Ver documentacion/diseño_modelo.md §6 y
documentacion/formulas_ml_resumen.md: la LSTM ve la secuencia completa del
día (incluido el alivio nocturno), no la hora pico ni índices agregados.

El resultado se CACHEA en data/processed/secuencias_24h.npz: los ~5.5 GB de
.nc se procesan una sola vez; re-entrenamientos posteriores cargan el npz.
Los valores se guardan en unidades físicas SIN escalar — el escalado es una
decisión de entrenamiento (se ajusta solo con train, ver lstm_model).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from climasafeai.data.make_dataset import (
    RAW_DATA_DIR,
    calcular_puntos_provincia,
    cargar_provincias_unificadas,
    filtrar_era5_por_puntos,
    procesar_era5_a_horario,
)
from climasafeai.utils.paths import PROCESSED_DATA_DIR

# Variables horarias que ve la LSTM (recomendación de formulas_ml_resumen.md).
# 'sp' se excluye por defecto: casi constante dentro del día, aporta poco a
# una secuencia intradía (queda parametrizable por si se quiere probar).
FEATURE_COLS_SEQ: list = ["t2m_c", "rh", "wind_speed_kmh", "heat_index_c", "wind_chill_c"]

# Misma columna de nombre de provincia que usan los notebooks 0-1.
COL_NOMBRE_PROVINCIA = "NAMEUNIT"

SECUENCIAS_PATH = PROCESSED_DATA_DIR / "secuencias_24h.npz"

HORAS_POR_DIA = 24


def construir_secuencias_24h(
    df_hora: pd.DataFrame,
    feature_cols: list = FEATURE_COLS_SEQ,
) -> tuple[np.ndarray, pd.DataFrame]:
    """
    DataFrame horario (provincia, datetime, features) -> tensor de secuencias.

    Exige días COMPLETOS: exactamente 24 horas y sin ningún NaN en
    `feature_cols`. El resto se descarta y se reporta (esperable en el
    primer/último día del rango descargado y en huecos de ERA5T).

    Returns
    -------
    X : np.ndarray float32 (n_días, 24, len(feature_cols))
        Horas ordenadas 00..23 dentro de cada día.
    df_index : pd.DataFrame
        Una fila por secuencia, columnas ['provincia', 'fecha'] en el mismo
        orden que X (para alinear labels).
    """
    df = df_hora.copy()
    df["fecha"] = pd.to_datetime(df["datetime"]).dt.normalize()
    df["_valida"] = df[feature_cols].notna().all(axis=1)

    resumen = df.groupby(["provincia", "fecha"], sort=False).agg(
        n_horas=("datetime", "size"),
        sin_nan=("_valida", "all"),
    )
    dias_ok = resumen[(resumen["n_horas"] == HORAS_POR_DIA) & resumen["sin_nan"]].index
    n_descartados = len(resumen) - len(dias_ok)
    if n_descartados:
        print(
            f"    Secuencias: {n_descartados} días descartados de {len(resumen)} "
            f"(incompletos o con NaN)"
        )

    df = df.set_index(["provincia", "fecha"]).loc[dias_ok].reset_index()
    df = df.sort_values(["provincia", "fecha", "datetime"], kind="stable")

    X = df[feature_cols].to_numpy(dtype=np.float32)
    X = X.reshape(-1, HORAS_POR_DIA, len(feature_cols))
    df_index = df.iloc[::HORAS_POR_DIA][["provincia", "fecha"]].reset_index(drop=True)
    return X, df_index


def _cargar_labels() -> pd.DataFrame:
    """Une los dos parquets etiquetados -> (provincia, fecha, y_calor, y_frio,
    y_calor_pct, y_frio_pct).

    Inner-join estricto: solo días presentes en AMBOS parquets, para que cada
    secuencia tenga siempre los dos labels (multi-tarea sin máscaras).

    Además de las clases 0/1/2, calcula el target CONTINUO de regresión:
    el percentil de mortalidad atribuida por provincia
    (`rank(pct=True, method="average")`) — exactamente el `pct` interno de
    `labels.py:_clasificar`, el paso previo al corte p75/p95 que produce
    las clases. Caveat heredado de la definición del label: el percentil se
    calcula sobre TODO el histórico por provincia (2016-2026), así que hay
    una fuga leve train↔test — la MISMA que ya tienen las clases de todo el
    proyecto, no una nueva.
    """
    calor = pd.read_parquet(
        PROCESSED_DATA_DIR / "dataset_calor_labeled.parquet",
        columns=["provincia", "fecha", "clase_riesgo_calor", "defunciones_atrib_exc_temp"],
    )
    frio = pd.read_parquet(
        PROCESSED_DATA_DIR / "dataset_frio_labeled.parquet",
        columns=["provincia", "fecha", "clase_riesgo_frio", "defunciones_atrib_def_temp"],
    )
    labels = calor.merge(frio, on=["provincia", "fecha"], how="inner")
    labels["fecha"] = pd.to_datetime(labels["fecha"])

    labels["y_calor_pct"] = labels.groupby("provincia")[
        "defunciones_atrib_exc_temp"
    ].rank(pct=True, method="average")
    labels["y_frio_pct"] = labels.groupby("provincia")[
        "defunciones_atrib_def_temp"
    ].rank(pct=True, method="average")

    return labels.rename(
        columns={"clase_riesgo_calor": "y_calor", "clase_riesgo_frio": "y_frio"}
    )


def generar_dataset_secuencias(
    force: bool = False,
    feature_cols: list = FEATURE_COLS_SEQ,
    limite_ficheros: int | None = None,
) -> Path:
    """
    Pipeline completo con caché: .nc de ERA5 -> secuencias 24h + labels -> npz.

    Si SECUENCIAS_PATH ya existe y `force=False`, no recalcula nada.

    Los .nc se procesan UNO A UNO (cada mes filtrado a los 5 puntos/provincia
    son ~200k filas) en vez de open_mfdataset sobre los ~120 ficheros — mismo
    resultado con una fracción de la memoria. Los días son completos dentro de
    cada fichero mensual, así que el troceo no rompe ninguna secuencia.

    Parameters
    ----------
    limite_ficheros : int | None
        Para smoke-tests: procesa solo los N primeros meses.
    """
    if SECUENCIAS_PATH.exists() and not force:
        print(f"    Caché existente: {SECUENCIAS_PATH} (usa force=True para regenerar)")
        return SECUENCIAS_PATH

    nc_files = sorted((RAW_DATA_DIR / "era5").glob("era5_*.nc"))
    if not nc_files:
        raise FileNotFoundError(f"No se encontraron archivos NetCDF en {RAW_DATA_DIR / 'era5'}")
    if limite_ficheros is not None:
        nc_files = nc_files[:limite_ficheros]

    provincias = cargar_provincias_unificadas()
    puntos = calcular_puntos_provincia(provincias, col_nombre=COL_NOMBRE_PROVINCIA)

    print(f"--> Procesando {len(nc_files)} ficheros ERA5 a horario...")
    dfs = []
    for i, nc in enumerate(nc_files, 1):
        with xr.open_dataset(nc) as ds:
            ds_filtrado = filtrar_era5_por_puntos(ds, puntos).load()
        df_mes = procesar_era5_a_horario(ds_filtrado)
        dfs.append(df_mes[["provincia", "datetime", *feature_cols]])
        if i % 12 == 0 or i == len(nc_files):
            print(f"    ... {i}/{len(nc_files)} ficheros")
    df_hora = pd.concat(dfs, ignore_index=True)
    del dfs

    X, df_index = construir_secuencias_24h(df_hora, feature_cols=feature_cols)

    # --- Alinear labels (inner-join estricto con ambos parquets) ---
    labels = _cargar_labels()
    df_index["_orden"] = np.arange(len(df_index))
    df_merge = df_index.merge(labels, on=["provincia", "fecha"], how="inner")
    df_merge = df_merge.sort_values("_orden")
    X = X[df_merge["_orden"].to_numpy()]
    print(
        f"    Alineado con labels: {len(df_merge)} secuencias "
        f"({len(df_index) - len(df_merge)} sin label descartadas)"
    )

    np.savez_compressed(
        SECUENCIAS_PATH,
        X=X,
        y_calor=df_merge["y_calor"].to_numpy(dtype=np.int64),
        y_frio=df_merge["y_frio"].to_numpy(dtype=np.int64),
        y_calor_pct=df_merge["y_calor_pct"].to_numpy(dtype=np.float32),
        y_frio_pct=df_merge["y_frio_pct"].to_numpy(dtype=np.float32),
        provincias=df_merge["provincia"].to_numpy(dtype=str),
        fechas=df_merge["fecha"].dt.strftime("%Y-%m-%d").to_numpy(dtype=str),
        feature_cols=np.array(feature_cols, dtype=str),
    )
    print(f"    Guardado: {SECUENCIAS_PATH} — X{X.shape}")
    return SECUENCIAS_PATH


def cargar_dataset_secuencias(path: Path = SECUENCIAS_PATH) -> dict:
    """Carga el npz cacheado -> dict con X, y_calor, y_frio, provincias, fechas."""
    if not Path(path).exists():
        raise FileNotFoundError(
            f"No existe {path}. Genera el dataset primero con "
            f"generar_dataset_secuencias() (tarda la primera vez: procesa los .nc de ERA5)."
        )
    with np.load(path, allow_pickle=False) as npz:
        data = {
            "X": npz["X"],
            "y_calor": npz["y_calor"],
            "y_frio": npz["y_frio"],
            "provincias": npz["provincias"],
            "fechas": npz["fechas"].astype("datetime64[D]"),
            "feature_cols": list(npz["feature_cols"]),
        }
        # Targets continuos de regresión (percentil de mortalidad) — pueden
        # faltar en un npz generado con una versión anterior del módulo.
        for clave in ("y_calor_pct", "y_frio_pct"):
            if clave in npz.files:
                data[clave] = npz[clave]
    return data


def split_secuencias_por_fecha(
    data: dict,
    test_size: float = 0.2,
    val_size: float = 0.1,
) -> dict:
    """
    Split TEMPORAL train/val/test con la misma regla que preprocess_data
    (build_features.py): fechas únicas ordenadas, las últimas round(n*test_size)
    van a test. De las fechas restantes, las últimas round(m*val_size) van a
    validación (early stopping de la LSTM). Nunca aleatorio: un split aleatorio
    mezclaría días de la misma ola de calor entre train y test.
    """
    fechas = data["fechas"]
    fechas_unicas = np.sort(np.unique(fechas))

    n_test = max(1, round(len(fechas_unicas) * test_size))
    fechas_test = fechas_unicas[-n_test:]
    restantes = fechas_unicas[:-n_test]

    n_val = max(1, round(len(restantes) * val_size))
    fechas_val = restantes[-n_val:]
    fechas_train = restantes[:-n_val]

    mask_test = np.isin(fechas, fechas_test)
    mask_val = np.isin(fechas, fechas_val)
    mask_train = ~(mask_test | mask_val)

    print(
        f"    Split por fecha: train hasta {fechas_train[-1]} | "
        f"val desde {fechas_val[0]} | test desde {fechas_test[0]} "
        f"({n_test} días distintos de test)"
    )

    out = {"fecha_corte_val": fechas_val[0], "fecha_corte_test": fechas_test[0]}
    for nombre, mask in [("train", mask_train), ("val", mask_val), ("test", mask_test)]:
        out[f"X_{nombre}"] = data["X"][mask]
        out[f"y_{nombre}_calor"] = data["y_calor"][mask]
        out[f"y_{nombre}_frio"] = data["y_frio"][mask]
        for clave in ("y_calor_pct", "y_frio_pct"):
            if clave in data:
                sufijo = clave.removeprefix("y_")  # calor_pct | frio_pct
                out[f"y_{nombre}_{sufijo}"] = data[clave][mask]
        out[f"fechas_{nombre}"] = fechas[mask]
    return out
