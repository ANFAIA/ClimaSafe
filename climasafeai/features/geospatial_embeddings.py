"""
climasafeai.features.geospatial_embeddings — embeddings geoespaciales como features.

Evalúa si embeddings más ricos que las coordenadas crudas (lat/lon) mejoran
la predicción por provincia. Compara dos fuentes open source:

  1. **CensusDemographicsEmbedding** — perfil demográfico denso derivado de
     datos censales (INE Padrón Continuo): envejecimiento, dependencia,
     urbanización, estructura poblacional. Análogo a los census/ACS
     embeddings de data.census.gov.

  2. **SpatialCoordinateEmbedding** — representación espacial densa a partir
     de coordenadas: distancia al centro geográfico, banda latitudinal,
     proximidad a costa, clusters espaciales. Análogo a lo que harían los
     embeddings de Google PDFM o AlphaEarth pero con fuentes abiertas.

PDFM (Population Dynamics Foundation Models) y AlphaEarth son productos
proprietarios de Google que requieren API access. No se usan aquí porque
no son open source. Los providers implementados replican la información
que estos modelos codifican (demografía + espacial) con datos públicos.

Uso::

    from climasafeai.features.geospatial_embeddings import (
        CensusDemographicsEmbedding,
        SpatialCoordinateEmbedding,
        merge_embeddings,
    )

    census = CensusDemographicsEmbedding()
    spatial = SpatialCoordinateEmbedding()

    df = merge_embeddings(df, [census, spatial])
    # df ahora tiene columnas como 'census_pct_envejecimiento', 'spatial_dist_madrid', etc.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import pandas as pd

from climasafeai.features.external_features import _EMBEDDED_DEMOGRAPHICS


# ---------------------------------------------------------------------------
# Coordenadas de las 52 capitales de provincia (las 45 del dataset + extras)
# Fuente: geolocalización estándar de las capitales
# ---------------------------------------------------------------------------
_PROVINCE_COORDS: dict[str, tuple[float, float]] = {
    "Albacete": (38.99, -1.86),
    "Almería": (36.83, -2.46),
    "Araba/Álava": (42.85, -2.67),
    "Asturias": (43.36, -5.85),
    "Badajoz": (38.88, -6.97),
    "Barcelona": (41.39, 2.17),
    "Bizkaia": (43.26, -2.93),
    "Burgos": (42.34, -3.70),
    "Cantabria": (43.46, -3.81),
    "Ceuta": (35.89, -5.32),
    "Ciudad Real": (38.99, -3.93),
    "Cuenca": (40.07, -2.13),
    "Cáceres": (39.48, -6.37),
    "Cádiz": (36.53, -6.29),
    "Córdoba": (37.88, -4.77),
    "Gipuzkoa": (43.32, -1.98),
    "Girona": (41.98, 2.82),
    "Granada": (37.18, -3.60),
    "Guadalajara": (40.63, -3.17),
    "Huelva": (37.26, -6.95),
    "Huesca": (42.14, -0.41),
    "Jaén": (37.77, -3.79),
    "León": (42.60, -5.57),
    "Lleida": (41.61, 0.63),
    "Lugo": (43.01, -7.56),
    "Madrid": (40.42, -3.70),
    "Melilla": (35.29, -2.94),
    "Murcia": (37.98, -1.13),
    "Málaga": (36.72, -4.42),
    "Navarra": (42.82, -1.65),
    "Ourense": (42.34, -7.86),
    "Palencia": (42.01, -4.53),
    "Pontevedra": (42.43, -8.64),
    "Salamanca": (40.97, -5.66),
    "Santa Cruz de Tenerife": (28.46, -16.25),
    "Segovia": (40.95, -4.12),
    "Sevilla": (37.39, -5.98),
    "Soria": (41.76, -2.47),
    "Tarragona": (41.12, 1.24),
    "Teruel": (40.34, -1.11),
    "Toledo": (39.86, -4.03),
    "Valladolid": (41.65, -4.72),
    "Zamora": (41.50, -5.74),
    "Zaragoza": (41.65, -0.88),
    "Ávila": (40.66, -4.70),
}

# Punto medio geográfico de España peninsular (referencia)
_SPAIN_CENTROID = (40.0, -3.7)

# Capitales costales (aproximación: < 15 km del mar)
_COASTAL_PROVINCES = {
    "Almería", "Barcelona", "Cádiz", "Córdoba",  # Córdoba no es costera, corrigo abajo
    "Gipuzkoa", "Girona", "Huelva", "Málaga",
    "Murcia", "Pontevedra", "Santa Cruz de Tenerife",
    "Tarragona", "Vizcaya/Bizkaia", "Alicante", "Castellón",
    "Valencia", "Islas Baleares", "Las Palmas",
}
# Corrección: solo las que realmente son costeras del dataset
_COASTAL_PROVINCES = {
    "Almería", "Barcelona", "Cádiz", "Gipuzkoa", "Girona",
    "Huelva", "Málaga", "Murcia", "Pontevedra",
    "Santa Cruz de Tenerife", "Tarragona", "Ceuta", "Melilla",
}


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Distancia haversine en km entre dos puntos (grados)."""
    R = 6371.0
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    a = (np.sin(dlat / 2) ** 2
         + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2))
         * np.sin(dlon / 2) ** 2)
    return R * 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))


# ---------------------------------------------------------------------------
# Interfaz base
# ---------------------------------------------------------------------------
class GeospatialEmbeddingProvider(ABC):
    """Interfaz para providers de embeddings geoespaciales.

    Cada provider genera un vector denso por provincia que se añade como
    feature(s) al DataFrame de entrenamiento.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Prefijo de las columnas que genera (e.g. 'census', 'spatial')."""

    @property
    @abstractmethod
    def feature_names(self) -> list[str]:
        """Nombres de las columnas que genera."""

    @abstractmethod
    def get_embeddings(self) -> pd.DataFrame:
        """Devuelve DataFrame con columna 'provincia' + columnas de embedding.

        Un vector por provincia. El merge con los datos de entrenamiento se
        hace por nombre de provincia.
        """

    @property
    def n_features(self) -> int:
        return len(self.feature_names)


# ---------------------------------------------------------------------------
# Provider 1: Census Demographics Embedding
# ---------------------------------------------------------------------------
class CensusDemographicsEmbedding(GeospatialEmbeddingProvider):
    """Embedding denso demográfico a partir de datos censales (INE).

    Toma los datos del Padrón Continuo ya disponibles en
    `_EMBEDDED_DEMOGRAPHICS` y genera un vector de 8 features que capturan
    la estructura poblacional de cada provincia:

    - pct_envejecimiento: % mayores 65 / total (% sobre 65)
    - pct_ancianos: % mayores 80 (fracción extrema del envejecimiento)
    - pct_mujeres: proporción de mujeres
    - log_poblacion: logaritmo de la población total
    - indice_envejecimiento: ratio mayores 65 / mayores 80 (estructura)
    - ratio_dependencia: estimación de ratio de dependencia
    - score_urbanizacion: proxy de urbanización (log(población) normalizado)
    - pct_poblacion_joven: 100 - pct_envejecimiento - 20 (proxy jóvenes)

    Análogo a los census/ACS embeddings de data.census.gov.
    """

    @property
    def name(self) -> str:
        return "census"

    @property
    def feature_names(self) -> list[str]:
        return [
            "pct_envejecimiento",
            "pct_ancianos",
            "pct_mujeres",
            "log_poblacion",
            "indice_envejecimiento",
            "ratio_dependencia",
            "score_urbanizacion",
            "pct_poblacion_joven",
        ]

    def get_embeddings(self) -> pd.DataFrame:
        rows = []
        for prov, (p65, p80, pmuj, pob) in _EMBEDDED_DEMOGRAPHICS.items():
            p65_f, p80_f = float(p65), float(p80)
            log_pop = float(np.log(pob))
            # Índice de envejecimiento: estructura de la población mayor
            idx_env = p65_f / max(p80_f, 0.1)
            # Ratio de dependencia estimada: mayores 65 / (resto estimado)
            ratio_dep = p65_f / max(100.0 - p65_f, 1.0)
            # Urbanización: log(población) normalizado [0, 1]
            log_pops = [float(np.log(vals[3])) for vals in _EMBEDDED_DEMOGRAPHICS.values()]
            log_pop_min, log_pop_max = min(log_pops), max(log_pops)
            urban = (log_pop - log_pop_min) / max(log_pop_max - log_pop_min, 0.01)
            # Población joven estimada
            pct_joven = max(100.0 - p65_f - 20.0, 5.0)  # suelo para evitar negativos

            rows.append({
                "provincia": prov,
                "pct_envejecimiento": p65_f,
                "pct_ancianos": p80_f,
                "pct_mujeres": pmuj,
                "log_poblacion": log_pop,
                "indice_envejecimiento": idx_env,
                "ratio_dependencia": ratio_dep,
                "score_urbanizacion": urban,
                "pct_poblacion_joven": pct_joven,
            })
        return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Provider 2: Spatial Coordinate Embedding
# ---------------------------------------------------------------------------
class SpatialCoordinateEmbedding(GeospatialEmbeddingProvider):
    """Embedding denso espacial a partir de coordenadas geográficas.

    Va más allá de lat/lon crudos y codifica:
    - lat, lon: coordenadas normalizadas
    - dist_madrid_km: distancia a Madrid (centro político-administrativo)
    - latitud_normalizada: banda latitudinal (sin( lat ))
    - es_costera: proximidad al litoral (0/1)

    Análogo a lo que codifican los embeddings espaciales de PDFM/AlphaEarth
    pero usando solo coordenadas de las capitales (fuente abierta).
    """

    @property
    def name(self) -> str:
        return "spatial"

    @property
    def feature_names(self) -> list[str]:
        return [
            "lat_norm",
            "lon_norm",
            "dist_madrid_km",
            "latitud_normalizada",
            "es_costera",
        ]

    def get_embeddings(self) -> pd.DataFrame:
        rows = []
        lats = [c[0] for c in _PROVINCE_COORDS.values()]
        lons = [c[1] for c in _PROVINCE_COORDS.values()]
        lat_min, lat_max = min(lats), max(lats)
        lon_min, lon_max = min(lons), max(lons)

        for prov, (lat, lon) in _PROVINCE_COORDS.items():
            lat_norm = (lat - lat_min) / max(lat_max - lat_min, 0.01)
            lon_norm = (lon - lon_min) / max(lon_max - lon_min, 0.01)
            dist_madrid = _haversine_km(lat, lon, *_SPAIN_CENTROID)
            lat_rad = np.sin(np.radians(lat))
            es_costera = 1.0 if prov in _COASTAL_PROVINCES else 0.0

            rows.append({
                "provincia": prov,
                "lat_norm": lat_norm,
                "lon_norm": lon_norm,
                "dist_madrid_km": dist_madrid,
                "latitud_normalizada": float(lat_rad),
                "es_costera": es_costera,
            })
        return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Función de utilidad: merge embeddings en el dataset
# ---------------------------------------------------------------------------
def merge_embeddings(
    df: pd.DataFrame,
    providers: list[GeospatialEmbeddingProvider],
) -> pd.DataFrame:
    """Añade embeddings geoespaciales al DataFrame por nombre de provincia.

    Parameters
    ----------
    df : DataFrame con columna 'provincia'.
    providers : lista de providers cuyos embeddings se van a añadir.

    Returns
    -------
    Copia de `df` con las columnas de embedding añadidas.
    """
    if "provincia" not in df.columns:
        raise ValueError(
            "merge_embeddings: el DataFrame debe tener columna 'provincia'"
        )

    df = df.copy()
    for provider in providers:
        emb = provider.get_embeddings()
        prefix = f"{provider.name}_"
        # Renombrar columnas de embedding con prefijo (excepto provincia)
        emb_cols = [c for c in emb.columns if c != "provincia"]
        emb = emb.rename(columns={c: f"{prefix}{c}" for c in emb_cols})
        df = df.merge(emb, on="provincia", how="left")
        # Rellenar nulos para provincias que no estén en el provider
        for col in emb.columns:
            if col != "provincia" and col in df.columns:
                df[col] = df[col].fillna(0.0)

    return df
