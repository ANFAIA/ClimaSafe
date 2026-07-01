"""
test_proba.py — Smoke tests: verifican que todos los módulos son importables
y que las funciones principales existen con las firmas esperadas.
"""
import pytest
import inspect

pytestmark = pytest.mark.smoke


def test_import_data_module():
    from climasafeai.data import make_dataset
    assert hasattr(make_dataset, "load_data")


def test_import_features_module():
    from climasafeai.features import build_features
    assert hasattr(build_features, "preprocess_data")


def test_import_models_train():
    from climasafeai.models import train_model
    assert hasattr(train_model, "train_models")


def test_import_models_predict():
    from climasafeai.models import predict_model
    assert hasattr(predict_model, "evaluate_models")


def test_import_visualization():
    from climasafeai.visualization import visualize
    assert hasattr(visualize, "plot_distributions")
    assert hasattr(visualize, "plot_correlation_matrix")


def test_import_utils_paths():
    from climasafeai.utils import paths
    assert hasattr(paths, "MODELS_DIR")
    assert hasattr(paths, "RAW_DATA_DIR")
    assert hasattr(paths, "FIGURES_DIR")


def test_load_data_signature():
    """load_data debe aceptar un argumento 'filename'."""
    from climasafeai.data.make_dataset import load_data
    sig = inspect.signature(load_data)
    assert "filename" in sig.parameters


def test_preprocess_data_signature():
    """preprocess_data debe tener los parámetros mínimos esperados."""
    from climasafeai.features.build_features import preprocess_data
    sig = inspect.signature(preprocess_data)
    assert "df" in sig.parameters



def test_train_models_signature():
    from climasafeai.models.train_model import train_models
    sig = inspect.signature(train_models)
    assert "X_train" in sig.parameters
    assert "y_train" in sig.parameters


def test_evaluate_models_signature():
    from climasafeai.models.predict_model import evaluate_models
    sig = inspect.signature(evaluate_models)
    assert "models" in sig.parameters









