import pandas as pd
from climasafeai.utils.paths import RAW_DATA_DIR
import os
import requests

def download_data(url: str, filename: str, params: dict | None = None, api_key_env: str | None = None) -> pd.DataFrame:
    """
    Descarga datos desde una API externa y los guarda en data/raw/.

    Parameters
    ----------
    url : endpoint de la API
    filename : nombre con el que se guardará el CSV en data/raw/
    params : query params para la petición (opcional)
    api_key_env : nombre de la variable de entorno con la API key (opcional)
    """
    headers = {}
    if api_key_env:
        key = os.environ.get(api_key_env)
        if not key:
            raise EnvironmentError(f"Falta la variable de entorno {api_key_env}")
        headers["Authorization"] = f"Bearer {key}"

    print(f"--> Descargando datos desde {url}...")
    resp = requests.get(url, params=params, headers=headers, timeout=30)
    resp.raise_for_status()

    file_path = RAW_DATA_DIR / filename
    df = pd.DataFrame(resp.json())  # o pd.read_csv si la API devuelve CSV
    df.to_csv(file_path, index=False)
    print(f"    Guardado en {file_path} — Shape: {df.shape}")
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
