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

def dataset_calor(momo: pd.DataFrame, era5: xr.Dataset) -> pd.DataFrame:
    """
    Fusiona los datos de mortalidad de MoMo (diarios por provincia)
    con variables climáticas de ERA5 (horarias, 5 puntos por provincia).

    Parameters
    ----------
    momo : pd.DataFrame
        Dataset de MoMo descargado.
    era5 : xr.Dataset
        Dataset ERA5 filtrado (salida de ``cargar_era5_filtrado``).

    Returns
    -------
    pd.DataFrame
        DataFrame combinado listo para modelado.
    """
    # ── 1. Preparar MoMo ──────────────────────────────────────────
    momo = momo[momo["ambito"] == "provincia"].copy()
    momo = momo.rename(columns={
        "nombre_ambito": "provincia",
        "fecha_defuncion": "fecha",
    })
    momo["fecha"] = pd.to_datetime(momo["fecha"]).dt.date

    momo_agg = (
        momo[["fecha", "provincia", "defunciones_atrib_exc_temp"]]
        .groupby(["fecha", "provincia"], as_index=False)
        .sum()
    )

    # ── 2. Preparar ERA5 ─────────────────────────────────────────
    # Quedarse con el experimento final (si existe la coordenada expver)
    if "expver" in era5.coords:
        era5 = era5.sel(expver="0001")   # o 1, según el tipo de dato

    # Convertir a DataFrame y aplanar índices
    era5_df = era5.to_dataframe().reset_index()

    # Renombrar la coordenada temporal real (¡cuidado! en tu caso es 'valid_time')
    era5_df = era5_df.rename(columns={"valid_time": "fecha"})

    # Dejar solo la fecha (sin hora) para alinear con MoMo
    era5_df["fecha"] = pd.to_datetime(era5_df["fecha"]).dt.date

    # Agrupar por fecha y provincia → un solo valor diario representativo
    # (promedio de todas las horas y de los 5 puntos interiores)
    era5_agg = (
        era5_df.groupby(["fecha", "provincia"], as_index=False)
        .mean(numeric_only=True)
    )

    # ── 3. Merge final ───────────────────────────────────────────
    df = pd.merge(momo_agg, era5_agg, on=["fecha", "provincia"], how="inner")

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
