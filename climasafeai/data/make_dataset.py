import pandas as pd
from climasafeai.utils.paths import RAW_DATA_DIR
import os
import requests
import cdsapi
from datetime import date

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
    """

    output_dir = RAW_DATA_DIR / "era5"
    output_dir.mkdir(parents=True, exist_ok=True)

    client = cdsapi.Client()

    current_year = date.today().year
    current_month = date.today().month
    start_year = current_year - 10

    variables = [
        "2m_temperature",
        "total_precipitation",
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

            client.retrieve(
                "reanalysis-era5-single-levels",
                {
                    "product_type": "reanalysis",
                    "variable": variables,
                    "year": str(year),
                    "month": f"{month:02d}",
                    "day": days,
                    "time": hours,

                    # España (incluye Canarias)
                    "area": [45, -19, 27, 5],

                    "data_format": "netcdf",
                    "download_format": "unarchived",
                },
                str(output_file),
            )

            print(f"    Guardado en {output_file}")
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

def load_data(filename: str = "<nombre>.csv") -> pd.DataFrame:
    """
    Carga el dataset desde data/raw/ con pandas.

    Parameters
    ----------
    filename : nombre del CSV en data/raw/  (solo el nombre, sin ruta)

    Returns
    -------
    pd.DataFrame
    """
    file_path = RAW_DATA_DIR / filename
    print(f"--> Cargando datos desde {file_path}...")
    if not file_path.exists():
        raise FileNotFoundError(
            f"\n  Archivo no encontrado: {file_path}\n"
            f"  Coloca tu dataset en: data/raw/{filename}\n"
            f"  O cambia DATA_FILE en main.py con el nombre correcto."
        )
    df = pd.read_csv(file_path)
    print(f"    Shape: {df.shape}")
    print(f"    Tipos:\n{df.dtypes.to_string()}")
    return df
