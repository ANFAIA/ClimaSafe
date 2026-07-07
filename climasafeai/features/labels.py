"""
labels.py -- Etiquetas de riesgo (SEGURO/PRECAUCION/PELIGRO) a partir de
percentiles de mortalidad atribuida de MoMo (NO del Heat Index -- ver
docstrings de cada función para la distinción).
"""
import numpy as np
import pandas as pd

def asignar_clase_riesgo_calor(
    df: pd.DataFrame,
    col_mortalidad: str = "defunciones_atrib_exc_temp",
    q_precaucion: float = 0.75,
    q_peligro: float = 0.95,
    por_provincia: bool = True,
) -> pd.DataFrame:
    """
    Asigna la clase de riesgo (0=SEGURO, 1=PRECAUCION, 2=PELIGRO) a partir
    de los percentiles de mortalidad atribuida a exceso de temperatura de
    MoMo -- NO a partir del Heat Index. Ver la distinción en el docstring
    de categorize_heat_index() (esa es la clasificación meteorológica de
    respaldo determinista, no la etiqueta de entrenamiento).

    Parameters
    ----------
    df : pd.DataFrame
        Salida de dataset_calor() -- una fila por (provincia, fecha).
    col_mortalidad : str
        Columna de mortalidad atribuida a calor sobre la que calcular los
        percentiles. Por defecto 'defunciones_atrib_exc_temp'.
    q_precaucion, q_peligro : float
        Percentiles de corte (0-1). Por defecto p75 y p95: el 75% de los
        días con menor mortalidad atribuida son SEGURO, el siguiente 20%
        PRECAUCION, y el 5% más alto PELIGRO. Son valores de partida --
        ajústalos según la distribución real de tu columna (haz un
        df[col_mortalidad].describe() / histograma antes de fijarlos).
    por_provincia : bool
        Si True (recomendado), los percentiles se calculan POR PROVINCIA
        por separado -- una provincia pequeña con pocos habitantes tiene
        números absolutos de mortalidad mucho más bajos que Madrid o
        Barcelona, así que un único percentil global clasificaría casi
        todo lo de las provincias grandes como "peligro" y casi nada de
        las pequeñas, sin que refleje riesgo relativo real. Si False, se
        usa un único percentil global sobre todo el dataset.

    Returns
    -------
    pd.DataFrame
        Copia de `df` con dos columnas nuevas:
        - 'clase_riesgo_calor': 0/1/2 (SEGURO/PRECAUCION/PELIGRO)
        - 'clase_riesgo_calor_label': la versión en texto, para depurar
    """
    df = df.copy()

    def _clasificar(serie: pd.Series) -> pd.Series:
        # Percentil por RANGO en vez de pd.cut sobre bordes de quantile:
        # con pocos días distintos de mortalidad (provincias pequeñas,
        # pocos años de histórico) los percentiles 75/95 pueden coincidir
        # en el mismo valor -- pd.cut revienta ahí (bordes de bin
        # duplicados) incluso con duplicates="drop", porque las 3
        # etiquetas fijas dejan de encajar. rank(pct=True) siempre
        # produce un percentil válido, incluso con empates o series
        # casi constantes (p.ej. muchos días seguidos en 0 mortalidad).
        pct = serie.rank(pct=True, method="average")
        clase = np.where(
            pct <= q_precaucion, 0,
            np.where(pct <= q_peligro, 1, 2),
        )
        return pd.Series(clase, index=serie.index).astype(int)

    if por_provincia:
        df["clase_riesgo_calor"] = (
            df.groupby("provincia")[col_mortalidad]
            .transform(_clasificar)
        )
    else:
        df["clase_riesgo_calor"] = _clasificar(df[col_mortalidad])

    etiquetas = {0: "seguro", 1: "precaucion", 2: "peligro"}
    df["clase_riesgo_calor_label"] = df["clase_riesgo_calor"].map(etiquetas)

    print("    Distribución de clases:")
    print(df["clase_riesgo_calor_label"].value_counts().to_string())

    return df
def asignar_clase_riesgo_frio(
    df: pd.DataFrame,
    col_mortalidad: str = "defunciones_atrib_def_temp",
    q_precaucion: float = 0.75,
    q_peligro: float = 0.95,
    por_provincia: bool = True,
) -> pd.DataFrame:
    """
    Asigna la clase de riesgo (0=SEGURO, 1=PRECAUCION, 2=PELIGRO) a partir
    de los percentiles de mortalidad atribuida a déficit de temperatura
    (frío) de MoMo -- mismo criterio que asignar_clase_riesgo_calor(),
    pero sobre 'defunciones_atrib_def_temp' en vez de
    'defunciones_atrib_exc_temp'.

    Es el modelo de FRÍO del diseño de dos modelos (calor/frío) --
    misma lógica, misma justificación de por_provincia (evita que
    provincias grandes concentren todo el "peligro" por diferencias de
    escala absoluta de población), distinta columna de mortalidad de
    origen.

    Parameters
    ----------
    df : pd.DataFrame
        Una fila por (provincia, fecha) -- análogo a la entrada de
        asignar_clase_riesgo_calor(), pero normalmente vendrá de un
        dataset_frio() (equivalente a dataset_calor() para la parte de
        frío, todavía por construir) en vez de dataset_calor().
    col_mortalidad : str
        Columna de mortalidad atribuida a frío. Por defecto
        'defunciones_atrib_def_temp'.
    q_precaucion, q_peligro : float
        Percentiles de corte (0-1). Mismo punto de partida que en calor
        (p75/p95) -- pero revisa la distribución real de
        'defunciones_atrib_def_temp' antes de darlos por buenos: el
        patrón estacional del frío es distinto al del calor (más
        prolongado, menos días de pico extremo aislado), así que no
        asumas que los mismos cortes son igual de adecuados sin mirarlo.
    por_provincia : bool
        Igual que en asignar_clase_riesgo_calor(): True (recomendado)
        calcula los percentiles dentro de cada provincia por separado.

    Returns
    -------
    pd.DataFrame
        Copia de `df` con dos columnas nuevas:
        - 'clase_riesgo_frio': 0/1/2 (SEGURO/PRECAUCION/PELIGRO)
        - 'clase_riesgo_frio_label': la versión en texto, para depurar
    """
    df = df.copy()

    def _clasificar(serie: pd.Series) -> pd.Series:
        # Percentil por RANGO en vez de pd.cut sobre bordes de quantile:
        # con pocos días distintos de mortalidad (provincias pequeñas,
        # pocos años de histórico) los percentiles 75/95 pueden coincidir
        # en el mismo valor -- pd.cut revienta ahí (bordes de bin
        # duplicados) incluso con duplicates="drop", porque las 3
        # etiquetas fijas dejan de encajar. rank(pct=True) siempre
        # produce un percentil válido, incluso con empates o series
        # casi constantes (p.ej. muchos días seguidos en 0 mortalidad).
        pct = serie.rank(pct=True, method="average")
        clase = np.where(
            pct <= q_precaucion, 0,
            np.where(pct <= q_peligro, 1, 2),
        )
        return pd.Series(clase, index=serie.index).astype(int)

    if por_provincia:
        df["clase_riesgo_frio"] = (
            df.groupby("provincia")[col_mortalidad]
            .transform(_clasificar)
        )
    else:
        df["clase_riesgo_frio"] = _clasificar(df[col_mortalidad])

    etiquetas = {0: "seguro", 1: "precaucion", 2: "peligro"}
    df["clase_riesgo_frio_label"] = df["clase_riesgo_frio"].map(etiquetas)

    print("    Distribución de clases (frío):")
    print(df["clase_riesgo_frio_label"].value_counts().to_string())

    return df
