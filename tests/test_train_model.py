"""
test_train_model.py — Tests para climasafeai/models/train_model.py
"""
import numpy as np
import pandas as pd
import pytest



from climasafeai.models.train_model import (
    _build_models,

    train_models,
    load_models,
)


def _make_Xy():
    """Datos sintéticos pequeños para entrenamiento rápido en tests."""

    from sklearn.datasets import make_classification
    X, y = make_classification(
        n_samples=120, n_features=4, n_classes=2, random_state=42
    )

    return X, y


@pytest.mark.smoke
def test_build_models_returns_dict():
    models = _build_models()
    assert isinstance(models, dict)
    assert len(models) > 0


def test_build_models_expected_keys():
    models = _build_models()


    assert "RandomForest" in models





def test_train_models_returns_trained_dict(patch_paths):
    X, y = _make_Xy()
    trained = train_models(X, y, tune_knn=False, cv_evaluate=False)
    assert isinstance(trained, dict)
    assert len(trained) > 0
    for model in trained.values():
        assert hasattr(model, "predict")


def test_train_models_saves_joblib_files(patch_paths):
    X, y = _make_Xy()
    trained = train_models(X, y, tune_knn=False, cv_evaluate=False)
    saved = list(patch_paths["MODELS_DIR"].glob("*.joblib"))
    assert len(saved) == len(trained)
    for name in trained:
        assert (patch_paths["MODELS_DIR"] / f"{name}.joblib").exists()


def test_load_models_loads_saved(patch_paths):
    X, y = _make_Xy()
    trained = train_models(X, y, tune_knn=False, cv_evaluate=False)
    loaded  = load_models()
    assert set(loaded.keys()) == set(trained.keys())
    for model in loaded.values():
        assert hasattr(model, "predict")



def test_load_models_specific_names(patch_paths):
    X, y = _make_Xy()
    train_models(X, y, tune_knn=False, cv_evaluate=False)
    loaded = load_models(["RandomForest"])
    assert "RandomForest" in loaded



def test_load_models_missing_returns_empty(patch_paths):
    """Si no hay modelos guardados, load_models() debe devolver dict vacío."""
    loaded = load_models(["ModeloInexistente"])
    assert loaded == {}


def test_models_can_predict(patch_paths):
    X, y = _make_Xy()
    trained = train_models(X, y, tune_knn=False, cv_evaluate=False)
    for name, model in trained.items():
        preds = model.predict(X)
        assert len(preds) == len(y), f"{name}: longitud incorrecta"

        assert set(preds).issubset({0, 1}), f"{name}: predicciones fuera de clases"










