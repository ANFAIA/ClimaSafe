"""
test_make_dataset.py — Tests para climasafeai/data/make_dataset.py
"""
import pandas as pd
import numpy as np
import pytest


def test_load_data_reads_csv(patch_paths):
    """load_data debe leer un CSV válido y devolver un DataFrame."""
    from climasafeai.data.make_dataset import load_data

    # Crear CSV temporal en RAW_DATA_DIR (ya parcheado)
    sample = pd.DataFrame(
        np.random.randn(50, 3),
        columns=["a", "b", "c"],
    )
    csv_path = patch_paths["RAW_DATA_DIR"] / "test.csv"
    sample.to_csv(csv_path, index=False)

    df = load_data("test.csv")
    assert isinstance(df, pd.DataFrame)
    assert df.shape == (50, 3)
    assert list(df.columns) == ["a", "b", "c"]


def test_load_data_raises_on_missing_file(patch_paths):
    """load_data debe lanzar una excepción si el archivo no existe."""
    from climasafeai.data.make_dataset import load_data
    with pytest.raises(Exception):
        load_data("no_existe.csv")


# ---------------------------------------------------------------------------
# Estadísticas de la distribución diaria (24h) -- _agregar_estadisticas_diarias
# ---------------------------------------------------------------------------
def _df_horario(provincia, heat_values, cold_values):
    """Helper: DataFrame horario de un único (provincia, fecha)."""
    n = len(heat_values)
    return pd.DataFrame(
        {
            "provincia": [provincia] * n,
            "fecha": [pd.Timestamp("2020-07-01").date()] * n,
            "heat_index_c": heat_values,
            "wind_chill_c": cold_values,
        }
    )


def test_calor_sostenido_vs_puntual_difieren_std_y_horas():
    """
    Requisito clave: un día de calor SOSTENIDO y otro con un pico PUNTUAL del
    MISMO máximo deben producir heat_index_std y horas_sobre_umbral distintos.
    """
    from climasafeai.data.make_dataset import _agregar_estadisticas_diarias

    # Ambos días llegan al mismo máximo (35°C), pero:
    #   - sostenido: 24 h por encima del umbral (32°C), poca dispersión
    #   - puntual  : una sola hora por encima, mucha dispersión
    sostenido = _df_horario("sostenido", [33.0] * 23 + [35.0], [10.0] * 24)
    puntual = _df_horario("puntual", [25.0] * 23 + [35.0], [10.0] * 24)

    assert sostenido["heat_index_c"].max() == puntual["heat_index_c"].max()  # mismo máximo

    stats = _agregar_estadisticas_diarias(
        pd.concat([sostenido, puntual], ignore_index=True),
        group_cols=["provincia", "fecha"],
    ).set_index("provincia")

    # horas_sobre_umbral: 24 (sostenido) vs 1 (puntual) -> distintos
    assert stats.loc["sostenido", "horas_sobre_umbral"] == 24
    assert stats.loc["puntual", "horas_sobre_umbral"] == 1
    assert (
        stats.loc["sostenido", "horas_sobre_umbral"]
        != stats.loc["puntual", "horas_sobre_umbral"]
    )

    # heat_index_std: el día puntual dispersa mucho más que el sostenido
    assert stats.loc["sostenido", "heat_index_std"] < stats.loc["puntual", "heat_index_std"]

    # El máximo compartido NO basta para distinguirlos: la media sí cambia
    assert stats.loc["sostenido", "heat_index_mean"] > stats.loc["puntual", "heat_index_mean"]


def test_frio_sostenido_vs_puntual_difieren_std_y_horas():
    """Simétrico para frío: wind_chill_std y horas_bajo_umbral deben diferir."""
    from climasafeai.data.make_dataset import _agregar_estadisticas_diarias

    # Mismo mínimo (-8°C) pero frío sostenido vs. bajón puntual.
    sostenido = _df_horario("frio_sostenido", [20.0] * 24, [-3.0] * 23 + [-8.0])
    puntual = _df_horario("frio_puntual", [20.0] * 24, [5.0] * 23 + [-8.0])

    assert sostenido["wind_chill_c"].min() == puntual["wind_chill_c"].min()  # mismo mínimo

    stats = _agregar_estadisticas_diarias(
        pd.concat([sostenido, puntual], ignore_index=True),
        group_cols=["provincia", "fecha"],
    ).set_index("provincia")

    assert stats.loc["frio_sostenido", "horas_bajo_umbral"] == 24
    assert stats.loc["frio_puntual", "horas_bajo_umbral"] == 1
    assert (
        stats.loc["frio_sostenido", "wind_chill_std"]
        < stats.loc["frio_puntual", "wind_chill_std"]
    )


def test_procesar_era5_a_diario_anade_estadisticas_diarias():
    """
    Integración: _procesar_era5_a_diario debe MANTENER heat_index_c (pico) y
    AÑADIR las columnas de distribución diaria, una fila por (provincia, fecha).
    """
    import numpy as np
    import xarray as xr
    from climasafeai.data.make_dataset import _procesar_era5_a_diario

    # 2 días × 24 h, 3 puntos de una provincia; ciclo diurno en la temperatura.
    times = pd.date_range("2020-07-01", periods=48, freq="h")
    n_p = 3
    diurno = 300.0 + 6.0 * np.sin(np.linspace(0, 4 * np.pi, 48))  # Kelvin, ~21-33°C
    t2m = np.tile(diurno[:, None], (1, n_p))
    d2m = t2m - 5.0
    sp = np.full((48, n_p), 101300.0)
    u10 = np.full((48, n_p), 2.0)
    v10 = np.full((48, n_p), 1.0)

    ds = xr.Dataset(
        {
            "t2m": (("valid_time", "punto"), t2m),
            "d2m": (("valid_time", "punto"), d2m),
            "sp": (("valid_time", "punto"), sp),
            "u10": (("valid_time", "punto"), u10),
            "v10": (("valid_time", "punto"), v10),
        },
        coords={
            "valid_time": times,
            "provincia": ("punto", ["TestProv"] * n_p),
        },
    )

    df_dia = _procesar_era5_a_diario(ds)

    # Una fila por (provincia, fecha) -> 2 días
    assert len(df_dia) == 2
    nuevas = {
        "heat_index_mean", "heat_index_std", "heat_index_min", "horas_sobre_umbral",
        "wind_chill_mean", "wind_chill_std", "wind_chill_max", "horas_bajo_umbral",
    }
    assert nuevas.issubset(df_dia.columns)
    assert "heat_index_c" in df_dia.columns  # el pico se mantiene

    fila = df_dia.iloc[0]
    # El pico (hora de mayor riesgo) no puede ser menor que el mínimo diario
    assert fila["heat_index_c"] >= fila["heat_index_min"]
    # horas_sobre_umbral es un conteo válido dentro de las 24 h
    assert 0 <= fila["horas_sobre_umbral"] <= 24


# ---------------------------------------------------------------------------
# Features de persistencia entre días -- _agregar_rezagos_temporales
# ---------------------------------------------------------------------------
def _df_diario(provincia, heat=None, wc_mean=None, horas_bajo=None):
    """Helper: DataFrame diario (una fila por fecha) de una provincia."""
    n = len(heat or wc_mean or horas_bajo)
    fechas = pd.date_range("2020-01-01", periods=n, freq="D").date
    data = {"provincia": [provincia] * n, "fecha": list(fechas)}
    if heat is not None:
        data["heat_index_c"] = heat
    if wc_mean is not None:
        data["wind_chill_mean"] = wc_mean
    if horas_bajo is not None:
        data["horas_bajo_umbral"] = horas_bajo
    return pd.DataFrame(data)


def test_lag1_es_valor_del_dia_anterior():
    """heat_index_c_lag1 debe ser el heat_index_c del día anterior (NaN el 1º)."""
    from climasafeai.data.make_dataset import _agregar_rezagos_temporales
    df = _df_diario("P", heat=[10.0, 20.0, 30.0, 40.0],
                    wc_mean=[1, 2, 3, 4], horas_bajo=[0, 0, 0, 0])
    out = _agregar_rezagos_temporales(df).sort_values("fecha").reset_index(drop=True)
    assert pd.isna(out.loc[0, "heat_index_c_lag1"])
    assert list(out["heat_index_c_lag1"].iloc[1:]) == [10.0, 20.0, 30.0]


def test_rolling_solo_usa_dias_previos():
    """wind_chill_mean_roll3 = media de los 3 días PREVIOS (hoy no cuenta)."""
    from climasafeai.data.make_dataset import _agregar_rezagos_temporales
    df = _df_diario("P", heat=[0]*5, wc_mean=[1.0, 2.0, 3.0, 4.0, 5.0], horas_bajo=[0]*5)
    out = _agregar_rezagos_temporales(df).sort_values("fecha").reset_index(drop=True)
    r3 = out["wind_chill_mean_roll3"]
    assert pd.isna(r3.iloc[0])          # sin histórico
    assert r3.iloc[1] == 1.0            # media de [1]
    assert r3.iloc[2] == 1.5            # media de [1,2]
    assert r3.iloc[3] == 2.0            # media de [1,2,3]
    assert r3.iloc[4] == 3.0            # media de [2,3,4] (hoy=5 NO entra)


def test_racha_frio_consecutiva_previa():
    """dias_consec_bajo_umbral = nº de días fríos consecutivos ANTERIORES."""
    from climasafeai.data.make_dataset import _agregar_rezagos_temporales
    # frío (horas_bajo>0) en: sí,sí,no,sí,sí
    df = _df_diario("P", heat=[0]*5, wc_mean=[0]*5, horas_bajo=[5, 3, 0, 2, 4])
    out = _agregar_rezagos_temporales(df).sort_values("fecha").reset_index(drop=True)
    assert list(out["dias_consec_bajo_umbral"]) == [0, 1, 2, 0, 1]


def test_rezagos_no_cruzan_provincias():
    """El lag de una provincia no debe usar filas de otra."""
    from climasafeai.data.make_dataset import _agregar_rezagos_temporales
    a = _df_diario("A", heat=[10.0, 11.0], wc_mean=[1, 2], horas_bajo=[0, 0])
    b = _df_diario("B", heat=[99.0, 98.0], wc_mean=[9, 8], horas_bajo=[0, 0])
    out = _agregar_rezagos_temporales(pd.concat([a, b], ignore_index=True))
    out = out.sort_values(["provincia", "fecha"]).reset_index(drop=True)
    b_rows = out[out["provincia"] == "B"].reset_index(drop=True)
    # El primer día de B no tiene día anterior -> NaN (no usa la última fila de A)
    assert pd.isna(b_rows.loc[0, "heat_index_c_lag1"])
    # El segundo día de B usa el primero de B (99), NO el de A
    assert b_rows.loc[1, "heat_index_c_lag1"] == 99.0


def test_rezagos_no_filtran_futuro():
    """Los rezagos de una fila no cambian si se borran las filas posteriores."""
    from climasafeai.data.make_dataset import _agregar_rezagos_temporales
    df = _df_diario("P", heat=[10.0, 20.0, 30.0, 40.0, 50.0],
                    wc_mean=[1.0, 2.0, 3.0, 4.0, 5.0], horas_bajo=[5, 0, 3, 3, 3])
    full = _agregar_rezagos_temporales(df).sort_values("fecha").reset_index(drop=True)
    truncado = _agregar_rezagos_temporales(df.iloc[:3]).sort_values("fecha").reset_index(drop=True)
    cols = ["heat_index_c_lag1", "wind_chill_mean_roll3", "dias_consec_bajo_umbral"]
    pd.testing.assert_frame_equal(full.loc[:2, cols], truncado.loc[:2, cols])




