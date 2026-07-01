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




