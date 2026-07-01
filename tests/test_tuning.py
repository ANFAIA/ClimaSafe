
"""
test_tuning.py — Tests de la optimizacion de hiperparametros con Optuna.

Usa n_trials=2 para que los tests sean rapidos.
"""
import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import train_test_split


def _make_Xy(n=120, n_feat=4):
    np.random.seed(42)
    X = pd.DataFrame(
        np.random.randn(n, n_feat),
        columns=[f"feat_{i}" for i in range(n_feat)],
    )

    y = pd.Series((X["feat_0"] + X["feat_1"] > 0).astype(int), name="target")

    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, y_train



def test_tune_models_devuelve_dict(patch_paths):
    """tune_models debe devolver un dict con params por modelo."""
    from tuning.tune_model import tune_models
    X_train, y_train = _make_Xy()
    result = tune_models(X_train, y_train, n_trials=2,
                         artifacts_dir=patch_paths["ARTIFACTS_DIR"],
                         reports_dir=patch_paths["REPORTS_DIR"])
    assert isinstance(result, dict)
    assert len(result) > 0


def test_tune_models_guarda_joblib(patch_paths):
    """tune_models debe guardar best_params_<modelo>.joblib en artifacts/."""
    import joblib
    from tuning.tune_model import tune_models, _OBJECTIVES
    X_train, y_train = _make_Xy()
    tune_models(X_train, y_train, n_trials=2,
                artifacts_dir=patch_paths["ARTIFACTS_DIR"],
                reports_dir=patch_paths["REPORTS_DIR"])
    for model_name in _OBJECTIVES:
        path = patch_paths["ARTIFACTS_DIR"] / f"best_params_{model_name}.joblib"
        assert path.exists(), f"No encontrado: best_params_{model_name}.joblib"
        params = joblib.load(path)
        assert isinstance(params, dict)


def test_tune_models_guarda_csv(patch_paths):
    """tune_models debe guardar reports/tuning_results.csv."""
    from tuning.tune_model import tune_models
    X_train, y_train = _make_Xy()
    tune_models(X_train, y_train, n_trials=2,
                artifacts_dir=patch_paths["ARTIFACTS_DIR"],
                reports_dir=patch_paths["REPORTS_DIR"])
    csv_path = patch_paths["REPORTS_DIR"] / "tuning_results.csv"
    assert csv_path.exists()
    df = pd.read_csv(csv_path)
    assert "modelo" in df.columns
    assert "best_value" in df.columns
    assert len(df) > 0


def test_train_models_usa_best_params(patch_paths):
    """Tras tune_models, train_models debe cargar best_params sin error."""
    import joblib
    from tuning.tune_model import tune_models
    from climasafeai.models.train_model import train_models

    X_train, y_train = _make_Xy()
    tune_models(X_train, y_train, n_trials=2,
                artifacts_dir=patch_paths["ARTIFACTS_DIR"],
                reports_dir=patch_paths["REPORTS_DIR"])
    models = train_models(X_train, y_train, tune_knn=False, cv_evaluate=False)
    assert isinstance(models, dict)
    assert len(models) > 0



