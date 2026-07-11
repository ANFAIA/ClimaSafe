"""
weather_indices.py — Índices meteorológicos derivados (feature engineering).

Implementa las fórmulas documentadas en
`documentacion/formulas_riesgo_deterministico.md`:

  - Heat Index (Rothfusz, 1990)      → sección 1.1
  - WBGT aproximado desde Heat Index → sección 1.4 (Bernard & Iheanacho, 2015)
  - Wind Chill (NWS, 2001)           → sección 2.1

Estas mismas fórmulas se usan en dos sitios distintos del sistema:

  1. Como **fallback determinista** (respuesta directa, sin ML) — no es
     responsabilidad de este módulo, ver `formulas_riesgo_deterministico.md`.
  2. Como **features de entrada al modelo ML** (RandomForest/red neuronal),
     que es lo que resuelve este módulo — ver `diseño_modelo.md`, sección 1
     ("Features principales: Heat Index... / Wind Chill...").

Todas las funciones son vectorizadas (aceptan escalar, np.ndarray o
pd.Series) y trabajan en las unidades del proyecto (°C, km/h, % HR).
"""

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Conversión de unidades
# ---------------------------------------------------------------------------
def celsius_to_fahrenheit(t_c):
    """°C → °F"""
    return t_c * 9 / 5 + 32


def fahrenheit_to_celsius(t_f):
    """°F → °C"""
    return (t_f - 32) * 5 / 9


# ---------------------------------------------------------------------------
# 1.1 — Heat Index (Rothfusz, 1990)
# ---------------------------------------------------------------------------
def heat_index(t_c, rh):
    """
    Heat Index (sensación térmica por calor) — ecuación de Rothfusz (NWS,
    1990), regresión sobre las tablas de Steadman.

    Parameters
    ----------
    t_c : float | array-like
        Temperatura del aire en °C.
    rh : float | array-like
        Humedad relativa en % (0-100).

    Returns
    -------
    float | np.ndarray
        Heat Index en °C.

    Notes
    -----
    La ecuación completa de Rothfusz solo es fiable para
    T ≥ 80°F (~26.7°C) y RH ≥ 40%. Por debajo de ese rango se usa una
    aproximación simplificada (media entre T y un pequeño efecto de la
    humedad), tal como indica `formulas_riesgo_deterministico.md` sección
    1.1, para evitar valores erráticos fuera del dominio de validez de la
    regresión.
    """
    t_c = np.asarray(t_c, dtype=float)
    rh = np.asarray(rh, dtype=float)

    t_f = celsius_to_fahrenheit(t_c)

    hi_full = (
        -42.379
        + 2.04901523 * t_f
        + 10.14333127 * rh
        - 0.22475541 * t_f * rh
        - 0.00683783 * t_f**2
        - 0.05481717 * rh**2
        + 0.00122874 * t_f**2 * rh
        + 0.00085282 * t_f * rh**2
        - 0.00000199 * t_f**2 * rh**2
    )

    # Aproximación simplificada fuera del rango de validez (T < 80°F o RH < 40%)
    hi_simplified = 0.5 * (t_f + 61.0 + (t_f - 68.0) * 1.2 + rh * 0.094)

    valid_range = (t_f >= 80.0) & (rh >= 40.0)
    hi_f = np.where(valid_range, hi_full, hi_simplified)

    result = fahrenheit_to_celsius(hi_f)
    return float(result) if result.ndim == 0 else result


# ---------------------------------------------------------------------------
# 1.4 — WBGT aproximado desde Heat Index (Bernard & Iheanacho, 2015)
# ---------------------------------------------------------------------------
def wbgt_from_heat_index(hi_c):
    """
    Aproximación de WBGT (Wet Bulb Globe Temperature) a partir del Heat
    Index, según Bernard & Iheanacho (2015, J. Occup. Environ. Hyg.).

    Es un cribado (screening), no sustituye una medición real de WBGT
    (que requeriría termómetro de globo y bulbo húmedo, no disponible en
    este proyecto). Precisión: ±2°C para HI bajos, ±0.5°C para HI > 37.8°C.

    Parameters
    ----------
    hi_c : float | array-like
        Heat Index en °C (salida de `heat_index()`).

    Returns
    -------
    float | np.ndarray
        WBGT aproximado en °C.
    """
    hi_c = np.asarray(hi_c, dtype=float)
    hi_f = celsius_to_fahrenheit(hi_c)
    wbgt_c = -0.0034 * hi_f**2 + 0.96 * hi_f - 34
    return float(wbgt_c) if wbgt_c.ndim == 0 else wbgt_c


# ---------------------------------------------------------------------------
# 2.1 — Wind Chill (NWS, 2001)
# ---------------------------------------------------------------------------
def wind_chill(t_c, v_kmh):
    """
    Wind Chill (sensación térmica por frío/viento) — fórmula oficial NWS
    vigente desde 2001.

    Parameters
    ----------
    t_c : float | array-like
        Temperatura del aire en °C.
    v_kmh : float | array-like
        Velocidad del viento en km/h (a 10 m de altura, como ERA5).

    Returns
    -------
    float | np.ndarray
        Wind Chill en °C. Fuera del rango de validez (T > 10°C o
        V ≤ 4.8 km/h) el viento no tiene efecto significativo sobre la
        sensación térmica, así que se devuelve la temperatura real sin
        modificar.
    """
    t_c = np.asarray(t_c, dtype=float)
    v_kmh = np.asarray(v_kmh, dtype=float)

    wc_full = (
        13.12
        + 0.6215 * t_c
        - 11.37 * np.power(np.maximum(v_kmh, 0), 0.16)
        + 0.3965 * t_c * np.power(np.maximum(v_kmh, 0), 0.16)
    )

    valid_range = (t_c <= 10.0) & (v_kmh > 4.8)
    result = np.where(valid_range, wc_full, t_c)
    return float(result) if result.ndim == 0 else result


# ---------------------------------------------------------------------------
# Categorías de riesgo (tabla 1.2 de formulas_riesgo_deterministico.md)
# ---------------------------------------------------------------------------
HEAT_INDEX_CATEGORIES = [
    (27.0, 32.0, "precaucion"),
    (32.0, 39.0, "precaucion_extrema"),
    (39.0, 51.0, "peligro"),
    (51.0, np.inf, "peligro_extremo"),
]


def categorize_heat_index(hi_c) -> pd.Series | str:
    """
    Clasifica el Heat Index según la tabla clínica de
    `formulas_riesgo_deterministico.md` sección 1.2.

    Valores por debajo de 27°C se etiquetan como "seguro".
    """
    def _label(v):
        if pd.isna(v) or v < 27.0:
            return "seguro"
        for lo, hi, label in HEAT_INDEX_CATEGORIES:
            if lo <= v < hi:
                return label
        return "peligro_extremo"

    if np.isscalar(hi_c):
        return _label(hi_c)
    return pd.Series(hi_c).apply(_label)


# ---------------------------------------------------------------------------
# Helper de alto nivel: añade todas las columnas derivadas a un DataFrame
# ---------------------------------------------------------------------------
def add_weather_index_columns(
    df: pd.DataFrame,
    temp_col: str = "t2m_c",
    rh_col: str = "rh",
    wind_col: str = "wind_speed_kmh",
) -> pd.DataFrame:
    """
    Añade Heat Index, WBGT y Wind Chill como columnas nuevas a `df`.

    Pensado para usarse dentro de `_feature_engineering()` en
    `build_features.py`. No modifica `df` in-place; devuelve una copia.

    Parameters
    ----------
    temp_col, rh_col, wind_col : str
        Nombres de las columnas de entrada en `df`. Ajusta estos nombres
        a los que finalmente use el pipeline de agregación ERA5
        (ver sección 6 de la conversación de diseño / `make_dataset.py`).
        Si alguna columna no existe, esa variable derivada se omite con
        un aviso en vez de lanzar una excepción — así el resto del
        pipeline no se rompe mientras se termina de definir el esquema
        final de columnas tras la integración ERA5 + MoMo + GeoPandas.

    Returns
    -------
    pd.DataFrame
        Copia de `df` con las columnas nuevas añadidas (las que se hayan
        podido calcular): 'heat_index_c', 'wbgt_c', 'wind_chill_c'.
    """
    df = df.copy()
    missing = []

    if temp_col in df.columns and rh_col in df.columns:
        df["heat_index_c"] = heat_index(df[temp_col], df[rh_col])
        df["wbgt_c"] = wbgt_from_heat_index(df["heat_index_c"])
    else:
        missing += [c for c in (temp_col, rh_col) if c not in df.columns]

    if temp_col in df.columns and wind_col in df.columns:
        df["wind_chill_c"] = wind_chill(df[temp_col], df[wind_col])
    else:
        missing += [c for c in (wind_col,) if c not in df.columns and wind_col not in missing]

    if missing:
        import warnings
        warnings.warn(
            f"add_weather_index_columns: columnas no encontradas {sorted(set(missing))} "
            "— revisa temp_col/rh_col/wind_col según el esquema final de datos.",
            stacklevel=2,
        )

    return df


# ---------------------------------------------------------------------------
# Selección de la hora de mayor riesgo del día (diseño_modelo.md, sección 2)
# ---------------------------------------------------------------------------
def select_risk_hour_row(
    df_hourly: pd.DataFrame,
    group_cols: list,
    temp_col: str = "t2m_c",
    rh_col: str = "rh",
    wind_col: str = "wind_speed_kmh",
    uv_col: str | None = None,
) -> pd.DataFrame:
    """
    Reduce un DataFrame horario (ERA5, ya agregado por provincia) a una
    única fila por `group_cols` (típicamente ['provincia', 'fecha']),
    quedándose con la hora de mayor riesgo meteorológico del día.

    Implementa el algoritmo de `diseño_modelo.md` sección 2:
      1. Para cada hora, RiskScore = max(HeatIndex, WindChill, UV).
      2. hora_peligro = argmax del RiskScore dentro de cada grupo.
      3. Se extraen las features meteorológicas de ERA5 de esa hora.
      4. Se descarta el RiskScore — nunca entra como feature del modelo,
         solo sirvió para elegir la hora.

    Esto evita mezclar condiciones de distintas horas del mismo día en un
    mismo registro (data leakage temporal), y evita colar el RiskScore
    calculado como si fuera un dato bruto de entrada.

    Parameters
    ----------
    df_hourly : pd.DataFrame
        Datos horarios de ERA5 ya agregados por provincia (una fila por
        provincia/fecha/hora). Debe incluir `group_cols` + las columnas
        de temp_col/rh_col/wind_col (y uv_col si se pasa).
    group_cols : list[str]
        Columnas que identifican un día (p.ej. ['provincia', 'fecha']).
    uv_col : str, optional
        Columna de índice UV, si está disponible. Si no se pasa, el
        RiskScore usa solo max(HeatIndex, -WindChill) — Wind Chill entra
        con signo invertido (más frío = más riesgo) para que "máximo"
        siga significando "peor caso" en ambos sentidos.

    Returns
    -------
    pd.DataFrame
        Una fila por grupo, con las columnas originales de esa hora más
        'heat_index_c', 'wbgt_c', 'wind_chill_c' — SIN la columna auxiliar
        de riesgo (se elimina antes de devolver: nunca debe usarse como
        feature de entrada, tal como indica el diseño).
    """
    df = add_weather_index_columns(
        df_hourly, temp_col=temp_col, rh_col=rh_col, wind_col=wind_col
    )

    n = len(df)
    heat_component = (
        df["heat_index_c"].to_numpy(dtype=float) if "heat_index_c" in df else np.full(n, -np.inf)
    )
    # Wind Chill: cuanto más bajo (más frío), mayor riesgo. Se invierte el
    # signo para que "más riesgo" siga siendo "valor más alto" y así sea
    # comparable con Heat Index/UV dentro del mismo max().
    cold_component = (
        -df["wind_chill_c"].to_numpy(dtype=float) if "wind_chill_c" in df else np.full(n, -np.inf)
    )
    components = [heat_component, cold_component]

    if uv_col is not None and uv_col in df.columns:
        components.append(df[uv_col].to_numpy(dtype=float))

    df["_risk_score"] = np.maximum.reduce(components)

    idx = df.groupby(group_cols)["_risk_score"].idxmax()
    result = df.loc[idx].drop(columns=["_risk_score"]).reset_index(drop=True)
    return result


# ---------------------------------------------------------------------------
# Conversiones directas de variables ERA5 (Kelvin, componentes de viento,
# punto de rocío) a las unidades que esperan las fórmulas de este módulo.
# ERA5 entrega t2m/d2m en KELVIN y el viento como componentes u10/v10
# (m/s), no como velocidad ni en km/h — conviene pasar por aquí ANTES de
# llamar a heat_index()/wind_chill()/add_weather_index_columns().
# ---------------------------------------------------------------------------
def kelvin_to_celsius(t_k):
    """K → °C. ERA5 (t2m, d2m) viene en Kelvin, no en °C."""
    return np.asarray(t_k, dtype=float) - 273.15


def relative_humidity_from_dewpoint(t_c, td_c):
    """
    Humedad relativa (%) a partir de temperatura y punto de rocío, ambos
    en °C — aproximación de Magnus-Tetens (August-Roche-Magnus).

    ERA5 no da RH directamente: da temperatura del aire (t2m) y
    temperatura de rocío (d2m). Esta es la conversión estándar
    (constantes de Alduchov & Eskridge, 1996, las mismas que usa Copernicus
    en su documentación de ERA5).

    Parameters
    ----------
    t_c : float | array-like
        Temperatura del aire en °C (t2m ya convertido con kelvin_to_celsius).
    td_c : float | array-like
        Temperatura de punto de rocío en °C (d2m ya convertido).

    Returns
    -------
    float | np.ndarray
        Humedad relativa en % (0-100).
    """
    t_c = np.asarray(t_c, dtype=float)
    td_c = np.asarray(td_c, dtype=float)

    a, b = 17.625, 243.04  # constantes de Alduchov & Eskridge (1996)
    numerator = np.exp((a * td_c) / (b + td_c))
    denominator = np.exp((a * t_c) / (b + t_c))
    rh = 100.0 * (numerator / denominator)
    # Recorte defensivo: redondeos pueden dar 100.4% o -0.2% en casos límite
    rh = np.clip(rh, 0.0, 100.0)
    return float(rh) if rh.ndim == 0 else rh


def wind_speed_kmh_from_components(u_ms, v_ms):
    """
    Módulo de la velocidad del viento en km/h a partir de las componentes
    u10/v10 de ERA5 (m/s, a 10m de altura).

    Parameters
    ----------
    u_ms, v_ms : float | array-like
        Componentes zonal (u) y meridional (v) del viento en m/s.

    Returns
    -------
    float | np.ndarray
        Velocidad del viento en km/h.
    """
    u_ms = np.asarray(u_ms, dtype=float)
    v_ms = np.asarray(v_ms, dtype=float)
    speed_ms = np.sqrt(u_ms**2 + v_ms**2)
    speed_kmh = speed_ms * 3.6
    return float(speed_kmh) if speed_kmh.ndim == 0 else speed_kmh
