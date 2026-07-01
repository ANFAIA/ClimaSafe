"""
test_build_features.py — Tests para climasafeai/features/build_features.py
"""
import numpy as np
import pandas as pd
import pytest



from climasafeai.features.build_features import (
    preprocess_data,
    _feature_engineering,
    process_input,
)


@pytest.mark.smoke
def test_preprocess_data_returns_four_splits(df_with_target, patch_paths):
    """preprocess_data debe devolver (X_train, X_test, y_train, y_test)."""
    result = preprocess_data(df_with_target, target_col="target")
    assert len(result) == 4
    X_train, X_test, y_train, y_test = result
    assert X_train.shape[0] > X_test.shape[0]
    assert len(y_train) == X_train.shape[0]
    assert len(y_test)  == X_test.shape[0]


def test_preprocess_data_creates_scaler_artifact(df_with_target, patch_paths):
    """Debe guardar scaler.joblib en ARTIFACTS_DIR."""
    preprocess_data(df_with_target, target_col="target")
    assert (patch_paths["ARTIFACTS_DIR"] / "scaler.joblib").exists()


def test_preprocess_data_with_pca(df_with_target, patch_paths):
    """Con use_pca=0.95 debe guardar pca.joblib."""
    X_train, X_test, _, _ = preprocess_data(
        df_with_target, target_col="target", use_pca=0.95
    )
    assert (patch_paths["ARTIFACTS_DIR"] / "pca.joblib").exists()
    n_orig = df_with_target.shape[1] - 1  # sin columna target
    assert X_train.shape[1] <= n_orig


def test_preprocess_data_saves_processed_csvs(df_with_target, patch_paths):
    """Debe guardar X_train.csv, X_test.csv, y_train.csv, y_test.csv."""
    preprocess_data(df_with_target, target_col="target")
    proc = patch_paths["PROCESSED_DATA_DIR"]
    for fname in ["X_train.csv", "X_test.csv", "y_train.csv", "y_test.csv"]:
        assert (proc / fname).exists(), f"Falta {fname}"


def test_preprocess_data_test_size(df_with_target, patch_paths):
    """test_size=0.3 debe producir ~30% en test."""
    X_train, X_test, _, _ = preprocess_data(
        df_with_target, target_col="target", test_size=0.3
    )
    total = X_train.shape[0] + X_test.shape[0]
    ratio = X_test.shape[0] / total
    assert 0.25 <= ratio <= 0.35


def test_preprocess_data_removes_duplicates(patch_paths):
    """Debe eliminar filas duplicadas antes de procesar."""
    np.random.seed(0)
    df = pd.DataFrame(np.random.randn(50, 3), columns=["a", "b", "c"])
    df["target"] = (df["a"] > 0).astype(int)
    df = pd.concat([df, df.iloc[:10]])  # 10 duplicados

    X_train, X_test, y_train, y_test = preprocess_data(df, target_col="target")
    assert X_train.shape[0] + X_test.shape[0] <= 50


def test_feature_engineering_returns_dataframe(sample_df):
    """_feature_engineering debe devolver un DataFrame."""
    result = _feature_engineering(sample_df)
    assert isinstance(result, pd.DataFrame)
    assert result.shape == sample_df.shape


def test_process_input_requires_scaler(df_with_target, patch_paths):
    """process_input debe fallar si no existe el scaler (no entrenado aún)."""
    with pytest.raises(Exception):
        process_input(df_with_target.drop(columns=["target"]))


def test_process_input_after_preprocess(df_with_target, patch_paths):
    """Después de preprocess_data, process_input debe transformar datos nuevos."""
    preprocess_data(df_with_target, target_col="target")
    X_new = df_with_target.drop(columns=["target"]).head(5)
    result = process_input(X_new)
    assert result.shape[0] == 5









