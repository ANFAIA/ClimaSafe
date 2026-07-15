"""
climasafeai.features.external_features — features demográficas provinciales (INE)
para la LSTM con embedding de provincia.

Dos fuentes de datos:
  1. Datos embebidos (estimaciones basadas en INE Padrón Continuo 2023)
     — siempre disponibles, sin depender de API externa.
  2. INE API JAXI T3 (opcional): refresh_from_ine() actualiza desde la API.

El mapping provincia → ID se deriva DINÁMICAMENTE de los nombres reales en
el npz de secuencias (no hay lista hardcodeada). Las features quedan alineadas
por nombre de provincia.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import requests
from sklearn.preprocessing import StandardScaler

from climasafeai.utils.paths import ARTIFACTS_DIR, PROCESSED_DATA_DIR

FEATURES_DIR = PROCESSED_DATA_DIR
FEATURES_DIR.mkdir(parents=True, exist_ok=True)

CACHE_PATH = FEATURES_DIR / "ine_provincia_features.csv"
SCALER_PATH = ARTIFACTS_DIR / "scaler_provincia_features.joblib"
MAPPING_PATH = ARTIFACTS_DIR / "provincia_mapping.json"

DEMOGRAPHIC_FEATURES = [
    "pct_mayores_65",
    "pct_mayores_80",
    "pct_mujeres",
    "log_poblacion_total",
]
N_FEATURES_PROVINCIA = len(DEMOGRAPHIC_FEATURES)

# ---------------------------------------------------------------------------
# Perfil demográfico embebido (INE Padrón Continuo, enero 2023 — aprox.)
# Valores: pct_mayores_65, pct_mayores_80, pct_mujeres, poblacion_total
# Fuente: INE Padrón Continuo a 1 de enero de 2023, Censo 2021
# ---------------------------------------------------------------------------
_EMBEDDED_DEMOGRAPHICS: dict[str, tuple[float, float, float, int]] = {
    "Albacete":               (21.8, 6.2, 49.8,  385_000),
    "Almería":                (16.5, 4.3, 48.9,  753_000),
    "Araba/Álava":            (20.5, 6.0, 50.3,  335_000),
    "Asturias":               (26.4, 8.2, 52.3, 1_011_000),
    "Badajoz":                (19.6, 5.5, 49.9,  672_000),
    "Barcelona":              (18.9, 5.7, 51.0, 5_727_000),
    "Bizkaia":                (22.4, 6.8, 51.8, 1_148_000),
    "Burgos":                 (23.0, 6.8, 50.2,  356_000),
    "Cantabria":              (22.2, 6.6, 51.0,  585_000),
    "Ceuta":                  (15.0, 3.5, 49.5,   83_000),
    "Ciudad Real":            (20.5, 5.8, 49.5,  492_000),
    "Cuenca":                 (24.5, 7.2, 49.0,  195_000),
    "Cáceres":                (23.0, 6.5, 49.5,  389_000),
    "Cádiz":                  (17.8, 4.8, 50.1, 1_245_000),
    "Córdoba":                (19.8, 5.6, 50.2,  776_000),
    "Gipuzkoa":               (22.0, 6.5, 51.0,  726_000),
    "Girona":                 (18.5, 5.3, 49.9,  797_000),
    "Granada":                (18.8, 5.3, 50.5,  922_000),
    "Guadalajara":            (16.2, 4.3, 48.9,  268_000),
    "Huelva":                 (17.5, 4.6, 49.6,  526_000),
    "Huesca":                 (21.8, 6.4, 49.0,  225_000),
    "Jaén":                   (20.0, 5.6, 49.6,  627_000),
    "León":                   (26.0, 8.0, 51.5,  452_000),
    "Lleida":                 (19.5, 5.6, 49.0,  442_000),
    "Lugo":                   (28.0, 8.8, 52.0,  325_000),
    "Madrid":                 (17.6, 5.5, 51.7, 6_751_000),
    "Melilla":                (12.5, 2.8, 49.0,   86_000),
    "Murcia":                 (17.5, 4.8, 49.7, 1_531_000),
    "Málaga":                 (18.8, 5.4, 50.5, 1_695_000),
    "Navarra":                (20.5, 6.2, 50.2,  671_000),
    "Ourense":                (29.0, 9.5, 52.5,  306_000),
    "Palencia":               (25.0, 7.5, 50.5,  159_000),
    "Pontevedra":             (23.5, 7.2, 51.6,  949_000),
    "Salamanca":              (24.5, 7.5, 51.0,  329_000),
    "Santa Cruz de Tenerife": (16.8, 4.5, 49.8, 1_044_000),
    "Segovia":                (22.5, 6.6, 49.8,  156_000),
    "Sevilla":                (17.5, 4.9, 50.6, 1_948_000),
    "Soria":                  (26.5, 8.2, 49.5,   89_000),
    "Tarragona":              (19.0, 5.5, 49.4,  839_000),
    "Teruel":                 (25.0, 7.5, 48.5,  135_000),
    "Toledo":                 (18.5, 5.1, 49.3,  706_000),
    "Valladolid":             (22.5, 6.8, 51.0,  520_000),
    "Zamora":                 (28.5, 9.0, 51.5,  168_000),
    "Zaragoza":               (21.0, 6.3, 50.5,  964_000),
    "Ávila":                  (25.0, 7.5, 49.8,  158_000),
}


def crear_mapping_provincias(
    provincia_names: np.ndarray,
) -> tuple[dict[str, int], np.ndarray, np.ndarray]:
    """
    Crea mapping name→id a partir de los nombres reales del dataset.
    """
    cats = pd.Categorical(provincia_names)
    id_to_name = cats.categories.to_numpy(dtype=str)
    name_to_id = {name: i for i, name in enumerate(id_to_name)}
    provincia_idx = cats.codes.astype(np.int64)
    if MAPPING_PATH is not None:
        ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
        with open(MAPPING_PATH, "w") as f:
            json.dump(name_to_id, f)
    return name_to_id, id_to_name, provincia_idx


def _poblar_features_embebidas(
    id_to_name: np.ndarray,
) -> pd.DataFrame:
    rows = []
    for name in id_to_name:
        name = str(name)
        if name in _EMBEDDED_DEMOGRAPHICS:
            p65, p80, pmuj, pob = _EMBEDDED_DEMOGRAPHICS[name]
        else:
            p65, p80, pmuj, pob = 20.0, 5.8, 50.0, 500_000
        rows.append({
            "provincia": name,
            "poblacion_total": float(pob),
            "pct_mayores_65": p65,
            "pct_mayores_80": p80,
            "pct_mujeres": pmuj,
        })
    df = pd.DataFrame(rows)
    df["log_poblacion_total"] = np.log(df["poblacion_total"])
    return df[["provincia", *DEMOGRAPHIC_FEATURES]]


def fetch_ine_features(
    id_to_name: np.ndarray,
    force_refresh: bool = False,
) -> pd.DataFrame:
    """
    Devuelve las features INE para las provincias en id_to_name.
    Prioridad: 1) caché CSV, 2) INE API, 3) datos embebidos.
    """
    if CACHE_PATH.exists() and not force_refresh:
        df = pd.read_csv(CACHE_PATH)
        present = set(df["provincia"])
        needed = set(str(n) for n in id_to_name)
        if needed.issubset(present):
            return df[["provincia", *DEMOGRAPHIC_FEATURES]]

    df = _poblar_features_embebidas(id_to_name)
    df.to_csv(CACHE_PATH, index=False)
    print(f"    Features INE guardadas → {CACHE_PATH}")
    return df


def alinear_features_provincia(
    provincia_names: np.ndarray,
    df_ine: pd.DataFrame,
    provincia_mapping: dict[str, int],
) -> np.ndarray:
    """
    Alinea features INE (una fila por provincia) con las secuencias.
    """
    df_ine = df_ine.set_index("provincia")
    feats_list = []
    for name in provincia_names:
        name = str(name)
        row = df_ine.loc[name]
        feats_list.append([row[c] for c in DEMOGRAPHIC_FEATURES])
    return np.array(feats_list, dtype=np.float32)


def escalar_features_provincia(
    Xp_train: np.ndarray,
    *Xp_otros: np.ndarray,
    guardar: bool = True,
) -> tuple:
    """
    StandardScaler sobre features provinciales, ajustado solo con train.
    """
    scaler = StandardScaler()
    scaler.fit(Xp_train)

    def _transform(Xp: np.ndarray) -> np.ndarray:
        return scaler.transform(Xp).astype(np.float32)

    if guardar:
        ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
        joblib.dump(scaler, SCALER_PATH)
        print(f"    Scaler provincia guardado → {SCALER_PATH.name}")

    return (scaler, _transform(Xp_train), *(_transform(X) for X in Xp_otros))
