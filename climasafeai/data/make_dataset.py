import pandas as pd
from climasafeai.utils.paths import RAW_DATA_DIR
import os
import requests
import cdsapi
from datetime import date
import xarray as xr
from pathlib import Path
import geopandas as gpd
from shapely.geometry import Polygon, MultiPolygon

from climasafeai.features.weather_indices import (
    kelvin_to_celsius,
    relative_humidity_from_dewpoint,
    wind_speed_kmh_from_components,
    select_risk_hour_row,
    add_weather_index_columns,
)
from climasafeai.features.labels import (
    asignar_clase_riesgo_calor,
    asignar_clase_riesgo_frio,
)

def download_aemet(
    municipio: str,
    filename: str
) -> pd.DataFrame:
    """
    Descarga la predicción meteorológica de un municipio desde la API de AEMET
    y la almacena en ``data/raw/``.

    Se usara para produccion

    Parameters
    ----------
    municipio : str, optional
        Código INE del municipio. Por defecto "28079" (Madrid).

    filename : str, optional
        Nombre del fichero CSV de salida.

    Returns
    -------
    pd.DataFrame
        Datos descargados desde AEMET.
    """

    file_path = RAW_DATA_DIR / filename

    if file_path.exists():
        print(f"    {filename} ya existe en {RAW_DATA_DIR}")
        return pd.read_csv(file_path)

    api_key = os.environ.get("AEMET_API_KEY")
    if api_key is None:
        raise EnvironmentError("Falta la variable de entorno 'AEMET_API_KEY'.")

    url = (
        "https://opendata.aemet.es/opendata/api/"
        f"prediccion/especifica/municipio/diaria/{municipio}"
    )

    headers = {
        "api_key": api_key
    }

    print("--> Descargando datos de AEMET...")

    response = requests.get(url, headers=headers, timeout=30)
    response.raise_for_status()

    download_url = response.json()["datos"]

    response = requests.get(download_url, timeout=30)
    response.raise_for_status()

    df = pd.DataFrame(response.json())

    df.to_csv(file_path, index=False)

    print(f"    Guardado en {file_path}")

    return df

def download_era5_data() -> None:
    """
    Descarga datos horarios de ERA5 para España desde hace 10 años.

    Se genera un fichero NetCDF por mes en ``data/raw/era5/``.
    Si un fichero ya existe, no se vuelve a descargar.

    Solo se descargan meses completos para evitar errores de
    disponibilidad de datos en el mes actual.
    """

    output_dir = RAW_DATA_DIR / "era5"
    output_dir.mkdir(parents=True, exist_ok=True)

    client = cdsapi.Client(
        retry_max=3,
        sleep_max=60,
    )

    today = date.today()

    # Último mes COMPLETO disponible
    current_year = today.year
    current_month = today.month - 1

    # Si estamos en enero, el último mes completo es diciembre del año anterior
    if current_month == 0:
        current_month = 12
        current_year -= 1

    start_year = current_year - 10

    variables = [
        "2m_temperature",
        "2m_dewpoint_temperature",
        "surface_pressure",
        "10m_u_component_of_wind",
        "10m_v_component_of_wind",
    ]

    hours = [f"{h:02d}:00" for h in range(24)]
    days = [f"{d:02d}" for d in range(1, 32)]

    for year in range(start_year, current_year + 1):

        last_month = current_month if year == current_year else 12

        for month in range(1, last_month + 1):

            output_file = output_dir / f"era5_{year}_{month:02d}.nc"

            if output_file.exists():
                print(f"    {output_file.name} ya existe.")
                continue

            print(f"--> Descargando ERA5 {year}-{month:02d}...")

            try:
                client.retrieve(
                    "reanalysis-era5-single-levels",
                    {
                        "product_type": "reanalysis",
                        "variable": variables,
                        "year": str(year),
                        "month": f"{month:02d}",
                        "day": days,
                        "time": hours,
                        "area": [45, -19, 27, 5],  # España (incluye Canarias)
                        "data_format": "netcdf",
                        "download_format": "unarchived",
                    },
                    str(output_file),
                )

            except Exception as e:
                print("\n" + "=" * 70)
                print(" Error descargando ERA5.")
                print(e)
                print("=" * 70)
                raise

            print(f"    Guardado en {output_file}")

    print("\n Descarga de ERA5 completada.")

def download_momo_data() -> None:
    """Descarga el dataset de Momo desde la URL oficial y lo guarda en data/raw/momo_data.csv"""
    if "momo_data.csv" in os.listdir(RAW_DATA_DIR):
        print(f"    momo_data.csv ya existe en {RAW_DATA_DIR}")
        return
    else:
        response = requests.get("https://momo.isciii.es/public/momo/data")
        response.raise_for_status()
        with open(RAW_DATA_DIR / "momo_data.csv", "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"    Guardado en {RAW_DATA_DIR / 'momo_data.csv'}")

def download_openuv(
    latitude: float,
    longitude: float,
    filename: str = "openuv_data.csv"
) -> pd.DataFrame:
    """
    Descarga el índice UV actual desde la API de OpenUV y lo almacena
    en ``data/raw/``.

    Notes
    -----
    Esta función **no** se utiliza para generar el dataset de entrenamiento
    de ClimaSafe, ya que OpenUV no ofrece un histórico adecuado.

    Su finalidad es obtener el índice UV en tiempo real durante la fase
    de inferencia del modelo.

    Parameters
    ----------
    latitude : float
        Latitud del punto de consulta.
    longitude : float
        Longitud del punto de consulta.
    filename : str, optional
        Nombre del fichero CSV donde se almacenarán los datos.

    Returns
    -------
    pd.DataFrame
        Datos devueltos por la API de OpenUV.
    """
    file_path = RAW_DATA_DIR / filename

    if file_path.exists():
        print(f"    {filename} ya existe en {RAW_DATA_DIR}")
        return pd.read_csv(file_path)

    api_key = os.environ.get("OpenUV_API_KEY")
    if api_key is None:
        raise EnvironmentError("Falta la variable de entorno 'OpenUV_API_KEY'.")

    headers = {
        "x-access-token": api_key
    }

    params = {
        "lat": latitude,
        "lng": longitude
    }

    print("--> Descargando datos de OpenUV...")

    response = requests.get(
        "https://api.openuv.io/api/v1/uv",
        headers=headers,
        params=params,
        timeout=30,
    )
    response.raise_for_status()

    data = response.json()
    df = pd.json_normalize(data["result"])

    df.to_csv(file_path, index=False)

    print(f"    Guardado en {file_path}")

    return df

def _extremos_polygon(geom) -> dict:
    """Extremos N/S/E/O reales sobre el borde de un Polygon (no bounding box)."""
    coords = list(geom.exterior.coords)
    return {
        "norte": max(coords, key=lambda c: c[1]),   # (lon, lat) con lat máxima
        "sur":   min(coords, key=lambda c: c[1]),
        "este":  max(coords, key=lambda c: c[0]),   # lon máxima
        "oeste": min(coords, key=lambda c: c[0]),
    }

def calcular_puntos_provincia(provincias: gpd.GeoDataFrame, col_nombre: str) -> dict:
    """
    Para cada provincia, calcula 5 puntos representativos: centro + N/S/E/O
    reales sobre el polígono (maneja MultiPolygon quedándose con el
    subpolígono de mayor área, para evitar islotes sueltos).

    Returns
    -------
    dict: {nombre_provincia: {"centro": (lon, lat), "norte": (lon, lat), ...}}
    """
    puntos_por_provincia = {}

    for _, row in provincias.iterrows():
        geom = row.geometry
        nombre = row[col_nombre]

        if isinstance(geom, MultiPolygon):
            geom = max(geom.geoms, key=lambda g: g.area)

        centro = geom.representative_point()
        extremos = _extremos_polygon(geom)

        puntos_por_provincia[nombre] = {
            "centro": (centro.x, centro.y),
            **extremos,
        }

    return puntos_por_provincia

def cargar_provincias_unificadas() -> gpd.GeoDataFrame:
    lineas_limite_dir = RAW_DATA_DIR / "lineas_limite"

    penin = gpd.read_file(
        lineas_limite_dir / "SHP_ETRS89" / "recintos_provinciales_inspire_peninbal_etrs89"
        / "recintos_provinciales_inspire_peninbal_etrs89.shp"
    ).to_crs("EPSG:4326")

    canarias = gpd.read_file(
        lineas_limite_dir / "SHP_REGCAN95" / "recintos_provinciales_inspire_canarias_regcan95"
        / "recintos_provinciales_inspire_canarias_regcan95.shp"
    ).to_crs("EPSG:4326")

    provincias = gpd.GeoDataFrame(
        pd.concat([penin, canarias], ignore_index=True),
        crs="EPSG:4326",
    )
    return provincias

def filtrar_era5_por_puntos(ds: xr.Dataset, puntos_por_provincia: dict) -> xr.Dataset:
    """
    Reduce el Dataset de ERA5 a los 5 puntos (centro+N/S/E/O) de cada
    provincia, usando selección vectorizada por el punto más cercano.
    """
    provincias, tipos, lats, lons = [], [], [], []
    for nombre, puntos in puntos_por_provincia.items():
        for tipo, (lon, lat) in puntos.items():
            provincias.append(nombre)
            tipos.append(tipo)
            lats.append(lat)
            lons.append(lon)

    idx = xr.DataArray(range(len(provincias)), dims="punto")
    ds_filtrado = ds.sel(
        latitude=xr.DataArray(lats, dims="punto"),
        longitude=xr.DataArray(lons, dims="punto"),
        method="nearest",
    )
    ds_filtrado = ds_filtrado.assign_coords(
        provincia=("punto", provincias),
        tipo_punto=("punto", tipos),
    )
    return ds_filtrado

def cargar_era5_filtrado(puntos_por_provincia: dict) -> xr.Dataset:
    """
    Carga todos los ficheros NetCDF de ERA5 y filtra a los 5 puntos
    (centro+N/S/E/O) de cada provincia.
    """
    era5_dir = RAW_DATA_DIR / "era5"
    nc_files = sorted(era5_dir.glob("era5_*.nc"))

    if not nc_files:
        raise FileNotFoundError(f"No se encontraron archivos NetCDF en {era5_dir}")

    print(f"--> Cargando y filtrando {len(nc_files)} archivos ERA5...")

    ds = xr.open_mfdataset(
        [str(f) for f in nc_files],
        combine="by_coords",
    )

    ds_filtrado = filtrar_era5_por_puntos(ds, puntos_por_provincia).load()
    ds.close()

    return ds_filtrado


def _resolver_expver(era5: xr.Dataset) -> xr.Dataset:
    """
    Colapsa la dimensión 'expver' si existe.

    ERA5 mezcla, en las descargas de los meses más recientes, el dato
    final (ERA5) con el provisional (ERA5T) bajo una dimensión extra
    'expver' con 2 valores -- cada timestamp trae NaN en uno de los dos.
    No se asume si el valor de esa dimensión es el string "0001"/"0005" o
    el entero 1/5 (varía según versión de xarray/cdsapi/formato) -- se
    indexa POR POSICIÓN (isel) en vez de por valor (sel), y se rellenan
    los NaN del primero (dato final) con el segundo (provisional) vía
    combine_first, que es el patrón recomendado por ECMWF para este caso.
    """
    if "expver" in era5.dims:
        if era5.sizes["expver"] > 1:
            era5 = era5.isel(expver=0).combine_first(era5.isel(expver=-1))
        else:
            era5 = era5.isel(expver=0)
    return era5


# Umbrales para las estadísticas diarias de "horas de riesgo":
#   - Calor: 32°C de Heat Index es el corte "precaución extrema" de la tabla
#     clínica (HEAT_INDEX_CATEGORIES en weather_indices.py). horas_sobre_umbral
#     cuenta cuántas horas del día se pasan de ahí.
#   - Frío: 0°C de Wind Chill (sensación térmica bajo cero) como proxy simple de
#     riesgo por frío. horas_bajo_umbral cuenta las horas por debajo.
HEAT_INDEX_UMBRAL_C = 32.0
WIND_CHILL_UMBRAL_C = 0.0


def _agregar_estadisticas_diarias(
    df_hora: pd.DataFrame,
    group_cols: list,
    heat_col: str = "heat_index_c",
    cold_col: str = "wind_chill_c",
    umbral_calor: float = HEAT_INDEX_UMBRAL_C,
    umbral_frio: float = WIND_CHILL_UMBRAL_C,
) -> pd.DataFrame:
    """
    Estadísticas de la DISTRIBUCIÓN diaria (24h) del calor/frío, no solo el
    pico -- para que el modelo distinga un día de calor sostenido de otro con
    el mismo máximo pero puntual.

    Devuelve una fila por `group_cols` (típicamente ['provincia', 'fecha']) con,
    sobre las 24 horas de `heat_col` (Heat Index):
        heat_index_mean    media diaria (acumulación de calor)
        heat_index_std     desviación estándar (sostenido vs. puntual)
        heat_index_min     mínimo diario (proxy de alivio nocturno)
        horas_sobre_umbral nº de horas con heat_col > `umbral_calor`
    y, de forma análoga para el frío sobre `cold_col` (Wind Chill):
        wind_chill_mean / wind_chill_std / wind_chill_max
        horas_bajo_umbral  nº de horas con cold_col < `umbral_frio`

    Estas columnas son features meteorológicas legítimas (no hay fuga de datos):
    se calculan solo a partir de variables de ERA5.
    """
    df = df_hora.copy()
    df["_over"] = df[heat_col] > umbral_calor
    df["_under"] = df[cold_col] < umbral_frio
    stats = (
        df.groupby(group_cols)
        .agg(
            heat_index_mean=(heat_col, "mean"),
            heat_index_std=(heat_col, "std"),
            heat_index_min=(heat_col, "min"),
            horas_sobre_umbral=("_over", "sum"),
            wind_chill_mean=(cold_col, "mean"),
            wind_chill_std=(cold_col, "std"),
            wind_chill_max=(cold_col, "max"),
            horas_bajo_umbral=("_under", "sum"),
        )
        .reset_index()
    )
    return stats


def _agregar_rezagos_temporales(
    df_dia: pd.DataFrame,
    group_col: str = "provincia",
    date_col: str = "fecha",
) -> pd.DataFrame:
    """
    Features de PERSISTENCIA entre días (no dentro del día): capturan que el
    riesgo depende de la racha, no solo del día actual.

    Motivación: el efecto del frío sobre la mortalidad es RETARDADO y
    ACUMULATIVO (una racha de días fríos mata más que un día suelto), mientras
    que el del calor es de lag corto (mismo día / día siguiente). Por eso el
    frío lleva medias móviles de varios días y una racha, y el calor solo el
    día anterior.

    Estrictamente PASADO (shift positivo) -> no hay fuga de datos aunque el
    split train/test sea temporal: cada día solo mira sus propios días
    ANTERIORES. La serie diaria es continua por provincia, así que shift = día
    de calendario.

    Columnas nuevas (una fila por (provincia, fecha)):
      Calor (lag corto):
        heat_index_c_lag1        Heat Index (pico) del día anterior
      Frío (acumulativo):
        wind_chill_mean_roll3    media de wind_chill_mean de los 3 días previos
        wind_chill_mean_roll7    ... de los 7 días previos
        dias_consec_bajo_umbral  nº de días consecutivos ANTERIORES con
                                 horas_bajo_umbral > 0 (racha de frío sin tregua)

    NaN en el primer día de cada provincia (sin histórico) -> preprocess_data
    los rellena con la media, igual que el resto de features.
    """
    df = df_dia.sort_values([group_col, date_col]).copy()
    g = df.groupby(group_col, sort=False)

    if "heat_index_c" in df.columns:
        df["heat_index_c_lag1"] = g["heat_index_c"].shift(1)

    if "wind_chill_mean" in df.columns:
        # shift(1) primero -> el día actual NO entra en su propia media (solo pasado)
        df["wind_chill_mean_roll3"] = g["wind_chill_mean"].transform(
            lambda s: s.shift(1).rolling(3, min_periods=1).mean()
        )
        df["wind_chill_mean_roll7"] = g["wind_chill_mean"].transform(
            lambda s: s.shift(1).rolling(7, min_periods=1).mean()
        )

    if "horas_bajo_umbral" in df.columns:
        def _racha_frio_previa(col: pd.Series) -> pd.Series:
            frio = (col > 0).astype(int)
            bloques = (frio != frio.shift()).cumsum()            # id de racha
            incl = frio.groupby(bloques).cumcount().add(1).where(frio == 1, 0)
            return incl.shift(1).fillna(0)                        # racha hasta AYER
        df["dias_consec_bajo_umbral"] = g["horas_bajo_umbral"].transform(_racha_frio_previa)

    return df


def _procesar_era5_a_diario(era5: xr.Dataset) -> pd.DataFrame:
    """
    Convierte el xr.Dataset de ERA5 (ya filtrado a 5 puntos/provincia) en
    un DataFrame con una fila por (provincia, fecha) -- la hora de mayor
    riesgo del día, con las unidades ya convertidas.

    Pasos:
      1. Resuelve 'expver' si aparece (ver _resolver_expver).
      2. xr.Dataset -> DataFrame largo (una fila por punto/hora).
      3. Conversión de unidades ERA5 -> unidades del proyecto:
         Kelvin -> °C (t2m, d2m), punto de rocío -> RH, componentes
         u10/v10 -> velocidad de viento en km/h (u10/v10 son las
         componentes del vector, NO la velocidad -- hace falta el módulo).
      4. Media espacial: colapsa los 5 puntos -> una fila por provincia/hora.
      5. Índices meteorológicos horarios (heat_index_c/wbgt_c/wind_chill_c).
      6. Estadísticas de la distribución diaria (24h) del calor/frío -- ver
         _agregar_estadisticas_diarias -- calculadas ANTES de colapsar el día.
      7. Selección de la hora de mayor riesgo del día (select_risk_hour_row,
         que mantiene heat_index_c/wbgt_c/wind_chill_c del pico) -> una fila
         por provincia/día, a la que se le mergean las estadísticas del paso 6.
    """
    era5 = _resolver_expver(era5)

    df = era5.to_dataframe().reset_index()

    # La coordenada temporal real de este dataset es 'valid_time' (no 'time').
    df = df.rename(columns={"valid_time": "datetime"})

    # --- Conversión de unidades ERA5 -> unidades del proyecto ---
    df["t2m_c"] = kelvin_to_celsius(df["t2m"])
    df["d2m_c"] = kelvin_to_celsius(df["d2m"])
    df["rh"] = relative_humidity_from_dewpoint(df["t2m_c"], df["d2m_c"])
    # u10/v10 son las componentes del vector viento, no la velocidad --
    # hay que calcular el módulo antes de poder usarla en Wind Chill.
    df["wind_speed_kmh"] = wind_speed_kmh_from_components(df["u10"], df["v10"])

    # --- Media espacial: colapsa los 5 puntos (centro/N/S/E/O) ---
    value_cols = ["t2m_c", "rh", "wind_speed_kmh", "sp"]
    df_hora = (
        df.groupby(["provincia", "datetime"])[value_cols]
        .mean()
        .reset_index()
    )
    df_hora["fecha"] = df_hora["datetime"].dt.date

    # --- Índices meteorológicos HORARIOS (necesarios para agregar la
    #     distribución diaria del calor/frío antes de colapsar el día) ---
    df_hora = add_weather_index_columns(
        df_hora, temp_col="t2m_c", rh_col="rh", wind_col="wind_speed_kmh"
    )

    # --- Estadísticas de la distribución diaria (24h), NO solo el pico ---
    df_stats = _agregar_estadisticas_diarias(df_hora, group_cols=["provincia", "fecha"])

    # --- Selección de la hora de mayor riesgo del día (mantiene el
    #     heat_index_c/wind_chill_c del pico) ---
    df_dia = select_risk_hour_row(
        df_hora,
        group_cols=["provincia", "fecha"],
        temp_col="t2m_c",
        rh_col="rh",
        wind_col="wind_speed_kmh",
    )

    # --- Añade (no sustituye) las estadísticas diarias a la fila hora-pico ---
    df_dia = df_dia.merge(df_stats, on=["provincia", "fecha"], how="left")

    # --- Features de persistencia entre días (lags/medias móviles del pasado) ---
    df_dia = _agregar_rezagos_temporales(df_dia, group_col="provincia", date_col="fecha")

    print(f"    ERA5: {len(df)} filas (punto/hora) -> {len(df_hora)} (provincia/hora) -> {len(df_dia)} (provincia/día)")
    return df_dia


def _procesar_momo_provincial(momo: pd.DataFrame, col_mortalidad: str) -> pd.DataFrame:
    """
    Filtra MoMo a ámbito provincial y agrega sexo/edad sumando
    `col_mortalidad` -- común a dataset_calor()/dataset_frio(), cambia
    solo la columna de mortalidad de origen.
    """
    momo = momo[momo["ambito"] == "provincia"].copy()
    momo = momo.rename(columns={
        "nombre_ambito": "provincia",
        "fecha_defuncion": "fecha",
    })
    momo["fecha"] = pd.to_datetime(momo["fecha"]).dt.date

    momo_agg = (
        momo[["fecha", "provincia", col_mortalidad]]
        .groupby(["fecha", "provincia"], as_index=False)
        .sum()
    )
    return momo_agg


def dataset_calor(momo: pd.DataFrame, era5: xr.Dataset) -> pd.DataFrame:
    """
    Fusiona los datos de mortalidad de MoMo (diarios por provincia, calor)
    con variables climáticas de ERA5 -- una fila por (provincia, fecha),
    quedándose con la hora de mayor riesgo del día (no la media de las 24h).

    Parameters
    ----------
    momo : pd.DataFrame
        Dataset de MoMo descargado (crudo, sin filtrar).
    era5 : xr.Dataset
        Dataset ERA5 filtrado (salida de ``cargar_era5_filtrado``).

    Returns
    -------
    pd.DataFrame
        DataFrame combinado listo para asignar_clase_riesgo_calor() y,
        después, build_features.py.
    """
    momo_agg = _procesar_momo_provincial(momo, col_mortalidad="defunciones_atrib_exc_temp")
    era5_dia = _procesar_era5_a_diario(era5)

    df = pd.merge(momo_agg, era5_dia, on=["fecha", "provincia"], how="inner")
    print(f"    dataset_calor: {len(df)} filas (provincia/día)")
    return df


def dataset_frio(momo: pd.DataFrame, era5: xr.Dataset) -> pd.DataFrame:
    """
    Fusiona los datos de mortalidad de MoMo (diarios por provincia, frío)
    con variables climáticas de ERA5 -- una fila por (provincia, fecha),
    quedándose con la hora de mayor riesgo del día (no la media de las 24h).

    Parameters
    ----------
    momo : pd.DataFrame
        Dataset de MoMo descargado (crudo, sin filtrar).
    era5 : xr.Dataset
        Dataset ERA5 filtrado (salida de ``cargar_era5_filtrado``).

    Returns
    -------
    pd.DataFrame
        DataFrame combinado listo para asignar_clase_riesgo_frio() y,
        después, build_features.py.
    """
    momo_agg = _procesar_momo_provincial(momo, col_mortalidad="defunciones_atrib_def_temp")
    era5_dia = _procesar_era5_a_diario(era5)

    df = pd.merge(momo_agg, era5_dia, on=["fecha", "provincia"], how="inner")
    print(f"    dataset_frio: {len(df)} filas (provincia/día)")
    return df


def load_data(filename="credit-train.csv"):
    """
    Carga el dataset desde la carpeta data/raw.
    """
    file_path = RAW_DATA_DIR / filename
    print(f"--> Cargando datos desde {file_path}...")

    try:
        df = pd.read_csv(file_path)
        print(f"    Datos cargados. Dimensiones: {df.shape}")
        return df
    except FileNotFoundError:
        print(f"ERROR: No se encontró el archivo {filename} en {RAW_DATA_DIR}")
        raise


def process_data(momo: pd.DataFrame, era5: xr.Dataset, clase: str = "calor") -> pd.DataFrame:
    """
    Orquesta el pipeline completo para una `clase` (calor|frio): construye
    el dataset combinado ERA5+MoMo y le asigna la etiqueta de riesgo
    correspondiente -- es la función que normalmente llamarías desde el
    notebook en vez de encadenar dataset_calor()/dataset_frio() y
    asignar_clase_riesgo_*() a mano.

    Parameters
    ----------
    momo : pd.DataFrame
        MoMo crudo, sin filtrar (dataset_calor/dataset_frio ya filtran
        internamente por ambito=="provincia").
    era5 : xr.Dataset
        ERA5 ya filtrado a 5 puntos/provincia (salida de cargar_era5_filtrado).
    clase : "calor" | "frio"
        Qué pipeline ejecutar. Determina tanto la columna de mortalidad
        usada (defunciones_atrib_exc_temp vs defunciones_atrib_def_temp)
        como la función de etiquetado (asignar_clase_riesgo_calor vs
        asignar_clase_riesgo_frio).

    Returns
    -------
    pd.DataFrame
        Una fila por (provincia, fecha), con las variables meteorológicas
        de la hora de mayor riesgo + la columna 'clase_riesgo_{clase}'
        (0=SEGURO/1=PRECAUCION/2=PELIGRO) lista para
        build_features.preprocess_data(..., clase=clase).
    """
    if clase == "calor":
        df = dataset_calor(momo, era5)
        df = asignar_clase_riesgo_calor(df)
    elif clase == "frio":
        df = dataset_frio(momo, era5)
        df = asignar_clase_riesgo_frio(df)
    else:
        raise ValueError(f"clase='{clase}' no reconocida -- debe ser 'calor' o 'frio'")

    return df