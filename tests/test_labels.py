"""
Tests de climasafeai/features/labels.py -- en particular del parámetro
fecha_corte_percentiles (percentiles train-only para evitar la fuga
temporal train↔test del label).
"""
import numpy as np
import pandas as pd
import pytest

from climasafeai.features.labels import (
    asignar_clase_riesgo_calor,
    asignar_clase_riesgo_frio,
)


def _df_sintetico(col_mortalidad: str, n_dias: int = 200, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    fechas = pd.date_range("2020-01-01", periods=n_dias, freq="D")
    filas = []
    for prov, escala in [("madrid", 20.0), ("soria", 0.5)]:
        filas.append(pd.DataFrame({
            "fecha": fechas,
            "provincia": prov,
            col_mortalidad: rng.gamma(shape=1.2, scale=escala, size=n_dias).round(1),
        }))
    return pd.concat(filas, ignore_index=True)


def test_comportamiento_por_defecto_sin_corte():
    """Sin fecha_corte_percentiles el resultado es el clásico (rank pct global)."""
    df = _df_sintetico("defunciones_atrib_exc_temp")
    out = asignar_clase_riesgo_calor(df)

    assert set(out["clase_riesgo_calor"].unique()) <= {0, 1, 2}
    # p75/p95 -> ~75% seguro, ~20% precaución, ~5% peligro por provincia
    prop = out.groupby("provincia")["clase_riesgo_calor"].apply(
        lambda s: (s == 0).mean()
    )
    assert ((prop > 0.70) & (prop < 0.80)).all()


def test_corte_no_cambia_labels_de_train():
    """
    Con fecha_corte_percentiles, las filas de train (fecha < corte) deben
    quedar EXACTAMENTE igual que si se etiquetara el subconjunto de train
    por separado con el método clásico (rank pct).
    """
    df = _df_sintetico("defunciones_atrib_exc_temp")
    corte = "2020-06-01"

    out = asignar_clase_riesgo_calor(df, fecha_corte_percentiles=corte)
    mask_train = pd.to_datetime(df["fecha"]) < pd.Timestamp(corte)

    solo_train = asignar_clase_riesgo_calor(df[mask_train].copy())
    pd.testing.assert_series_equal(
        out.loc[mask_train.values, "clase_riesgo_calor"],
        solo_train["clase_riesgo_calor"],
        check_names=False,
    )


def test_labels_de_test_usan_distribucion_de_train():
    """
    El label del periodo de test no debe depender de los valores de test:
    cambiar la mortalidad de OTRAS filas de test no altera el label de una
    fila de test dada (con percentiles full-history sí lo haría).
    """
    df = _df_sintetico("defunciones_atrib_def_temp")
    corte = "2020-06-01"
    mask_test = pd.to_datetime(df["fecha"]) >= pd.Timestamp(corte)

    out1 = asignar_clase_riesgo_frio(df, fecha_corte_percentiles=corte)

    df2 = df.copy()
    # Inflar la mortalidad de la mitad de las filas de test
    idx_alterar = df2.index[mask_test][::2]
    df2.loc[idx_alterar, "defunciones_atrib_def_temp"] *= 100
    out2 = asignar_clase_riesgo_frio(df2, fecha_corte_percentiles=corte)

    idx_intactas = df2.index[mask_test].difference(idx_alterar)
    pd.testing.assert_series_equal(
        out1.loc[idx_intactas, "clase_riesgo_frio"],
        out2.loc[idx_intactas, "clase_riesgo_frio"],
    )


def test_corte_invalido_lanza_error():
    df = _df_sintetico("defunciones_atrib_exc_temp")
    # Corte anterior a todo el histórico -> ninguna fila de train
    with pytest.raises(ValueError):
        asignar_clase_riesgo_calor(df, fecha_corte_percentiles="2010-01-01")
    # Sin columna de fecha
    with pytest.raises(ValueError):
        asignar_clase_riesgo_calor(
            df.drop(columns=["fecha"]), fecha_corte_percentiles="2020-06-01"
        )
