"""
labels.py -- Etiquetas de riesgo (SEGURO/PRECAUCION/PELIGRO) a partir de
percentiles de mortalidad atribuida de MoMo (NO del Heat Index -- ver
docstrings de cada función para la distinción).
"""
import numpy as np
import pandas as pd


def _clasificar_percentil(
    serie: pd.Series,
    q_precaucion: float,
    q_peligro: float,
    mask_referencia: pd.Series | None = None,
) -> pd.Series:
    """
    Clasifica una serie de mortalidad en 0/1/2 por percentil de RANGO.

    Percentil por RANGO en vez de pd.cut sobre bordes de quantile:
    con pocos días distintos de mortalidad (provincias pequeñas,
    pocos años de histórico) los percentiles 75/95 pueden coincidir
    en el mismo valor -- pd.cut revienta ahí (bordes de bin
    duplicados) incluso con duplicates="drop", porque las 3
    etiquetas fijas dejan de encajar. rank(pct=True) siempre
    produce un percentil válido, incluso con empates o series
    casi constantes (p.ej. muchos días seguidos en 0 mortalidad).

    Parameters
    ----------
    mask_referencia : pd.Series[bool] | None
        Si es None (comportamiento clásico), el percentil de cada fila se
        calcula respecto a TODA la serie -- rank(pct=True) directo.
        Si se pasa una máscara booleana (alineada con `serie`), la
        distribución de referencia son SOLO las filas marcadas True
        (típicamente el periodo de train): el percentil de cada valor
        (de train O de test) se evalúa contra esa distribución de
        referencia. Para las filas de referencia el resultado es
        idéntico a rank(pct=True, method="average") sobre ese
        subconjunto, así que el label de train no cambia respecto a
        etiquetar train por separado -- y el label de test deja de
        depender de datos de test (sin fuga temporal).
    """
    if mask_referencia is None:
        pct = serie.rank(pct=True, method="average")
    else:
        ref = np.sort(serie[mask_referencia].dropna().to_numpy())
        n = len(ref)
        if n == 0:
            raise ValueError(
                "_clasificar_percentil: la máscara de referencia no deja "
                "ninguna fila (¿fecha de corte anterior a todo el histórico "
                "de alguna provincia?). No se pueden calcular percentiles."
            )
        vals = serie.to_numpy()
        # Rango medio del valor dentro de la distribución de referencia:
        # equivale a rank(pct=True, method="average") para valores que
        # están en la referencia, y al punto medio del hueco (ECDF) para
        # valores nuevos que no aparecen en ella.
        menores = np.searchsorted(ref, vals, side="left")
        menores_o_iguales = np.searchsorted(ref, vals, side="right")
        pct = pd.Series(
            (menores + menores_o_iguales + 1) / (2 * n),
            index=serie.index,
        )
    clase = np.where(
        pct <= q_precaucion, 0,
        np.where(pct <= q_peligro, 1, 2),
    )
    return pd.Series(clase, index=serie.index).astype(int)


def _mask_train_desde_corte(df: pd.DataFrame, fecha_corte, col_fecha: str) -> pd.Series:
    """Máscara booleana de filas de train: fecha < fecha_corte."""
    if col_fecha not in df.columns:
        raise ValueError(
            f"fecha_corte_percentiles requiere la columna '{col_fecha}' en el "
            "DataFrame para separar el periodo de referencia (train)."
        )
    corte = pd.Timestamp(fecha_corte)
    mask = pd.to_datetime(df[col_fecha]) < corte
    if not mask.any():
        raise ValueError(
            f"fecha_corte_percentiles={corte.date()} no deja ninguna fila de "
            "train (todas las fechas son >= corte)."
        )
    return mask


def asignar_clase_riesgo_calor(
    df: pd.DataFrame,
    col_mortalidad: str = "defunciones_atrib_exc_temp",
    q_precaucion: float = 0.75,
    q_peligro: float = 0.95,
    por_provincia: bool = True,
    min_mortalidad_peligro: float | None = 2.0,
    fecha_corte_percentiles=None,
    col_fecha: str = "fecha",
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
    min_mortalidad_peligro : float | None
        Suelo ABSOLUTO de mortalidad para poder ser PELIGRO (clase 2). Si se
        da (por defecto 2.0), un día que caería en PELIGRO solo por ser el
        top percentil de su provincia PERO con mortalidad atribuida por
        debajo de este suelo se DEGRADA a PRECAUCION (clase 1) -- no a
        SEGURO: sigue siendo un aviso. Corrige el efecto colateral de
        por_provincia=True: en provincias diminutas (Ceuta, Melilla, Teruel)
        el "top 5%" puede ser un día de 0-1 muertes, y marcarlo como MÁXIMA
        alerta es una falsa alarma que además ensucia la señal que aprende
        el modelo. Solo toca el tramo PELIGRO; PRECAUCION se deja intacto
        (cualquier mortalidad > 0 sigue avisando), así que no introduce
        falsos negativos: los días degradados pasan a un aviso menor, no a
        "sin riesgo". None desactiva el suelo (comportamiento anterior:
        PELIGRO puramente por percentil).
    fecha_corte_percentiles : str | datetime-like | None
        Si es None (por defecto), comportamiento clásico: los percentiles
        se calculan sobre TODO el histórico. OJO: eso introduce fuga
        temporal si luego se hace un split train/test por fecha, porque
        el label de train incorpora la distribución del periodo de test.
        Si se pasa una fecha (la fecha de INICIO del test), los
        percentiles de referencia se calculan SOLO con filas de
        fecha < corte (train) y se aplican a todo el dataset -- el label
        de test se evalúa contra la distribución de train, sin fuga.
    col_fecha : str
        Columna de fecha usada para aplicar `fecha_corte_percentiles`.

    Returns
    -------
    pd.DataFrame
        Copia de `df` con dos columnas nuevas:
        - 'clase_riesgo_calor': 0/1/2 (SEGURO/PRECAUCION/PELIGRO)
        - 'clase_riesgo_calor_label': la versión en texto, para depurar
    """
    df = df.copy()

    mask_train = (
        _mask_train_desde_corte(df, fecha_corte_percentiles, col_fecha)
        if fecha_corte_percentiles is not None
        else None
    )

    def _clasificar(serie: pd.Series) -> pd.Series:
        mask = mask_train.loc[serie.index] if mask_train is not None else None
        return _clasificar_percentil(serie, q_precaucion, q_peligro, mask)

    if por_provincia:
        df["clase_riesgo_calor"] = (
            df.groupby("provincia")[col_mortalidad]
            .transform(_clasificar)
        )
    else:
        df["clase_riesgo_calor"] = _clasificar(df[col_mortalidad])

    if min_mortalidad_peligro is not None:
        # Suelo absoluto: PELIGRO por percentil pero con mortalidad por debajo
        # del suelo -> se degrada a PRECAUCION (sigue siendo aviso, no SEGURO).
        degradar = (
            (df["clase_riesgo_calor"] == 2)
            & (df[col_mortalidad] < min_mortalidad_peligro)
        )
        n_degradados = int(degradar.sum())
        df.loc[degradar, "clase_riesgo_calor"] = 1
        if n_degradados:
            print(
                f"    Suelo PELIGRO (>= {min_mortalidad_peligro} muertes): "
                f"{n_degradados} días degradados a PRECAUCION"
            )

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
    min_mortalidad_peligro: float | None = 2.0,
    fecha_corte_percentiles=None,
    col_fecha: str = "fecha",
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
    min_mortalidad_peligro : float | None
        Suelo absoluto de mortalidad para PELIGRO -- ver la explicación
        detallada en asignar_clase_riesgo_calor(). Por defecto 2.0: un día
        que caería en PELIGRO solo por percentil pero con menos muertes
        atribuidas que el suelo se degrada a PRECAUCION (sigue avisando).
        None desactiva el suelo.
    fecha_corte_percentiles : str | datetime-like | None
        Igual que en asignar_clase_riesgo_calor(): si es None (por
        defecto), percentiles sobre todo el histórico (comportamiento
        clásico, con fuga temporal si hay split por fecha); si se pasa
        la fecha de inicio del test, los percentiles se calculan SOLO
        con filas de fecha < corte y se aplican a todo el dataset.
    col_fecha : str
        Columna de fecha usada para aplicar `fecha_corte_percentiles`.

    Returns
    -------
    pd.DataFrame
        Copia de `df` con dos columnas nuevas:
        - 'clase_riesgo_frio': 0/1/2 (SEGURO/PRECAUCION/PELIGRO)
        - 'clase_riesgo_frio_label': la versión en texto, para depurar
    """
    df = df.copy()

    mask_train = (
        _mask_train_desde_corte(df, fecha_corte_percentiles, col_fecha)
        if fecha_corte_percentiles is not None
        else None
    )

    def _clasificar(serie: pd.Series) -> pd.Series:
        mask = mask_train.loc[serie.index] if mask_train is not None else None
        return _clasificar_percentil(serie, q_precaucion, q_peligro, mask)

    if por_provincia:
        df["clase_riesgo_frio"] = (
            df.groupby("provincia")[col_mortalidad]
            .transform(_clasificar)
        )
    else:
        df["clase_riesgo_frio"] = _clasificar(df[col_mortalidad])

    if min_mortalidad_peligro is not None:
        # Suelo absoluto: PELIGRO por percentil pero con mortalidad por debajo
        # del suelo -> se degrada a PRECAUCION (sigue siendo aviso, no SEGURO).
        degradar = (
            (df["clase_riesgo_frio"] == 2)
            & (df[col_mortalidad] < min_mortalidad_peligro)
        )
        n_degradados = int(degradar.sum())
        df.loc[degradar, "clase_riesgo_frio"] = 1
        if n_degradados:
            print(
                f"    Suelo PELIGRO (>= {min_mortalidad_peligro} muertes): "
                f"{n_degradados} días degradados a PRECAUCION"
            )

    etiquetas = {0: "seguro", 1: "precaucion", 2: "peligro"}
    df["clase_riesgo_frio_label"] = df["clase_riesgo_frio"].map(etiquetas)

    print("    Distribución de clases (frío):")
    print(df["clase_riesgo_frio_label"].value_counts().to_string())

    return df
