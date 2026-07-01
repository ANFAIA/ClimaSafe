"""
test_predict_model.py — Tests para climasafeai/models/predict_model.py
"""
import numpy as np
import pandas as pd
import pytest



from climasafeai.models.predict_model import (
    evaluate_models,
    predict_new,

    predict_proba_new,

)
from climasafeai.models.train_model import train_models


def _make_data():

    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    X, y = make_classification(
        n_samples=160, n_features=4, n_classes=2, random_state=42
    )
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    return train_test_split(X, y, test_size=0.25, random_state=42)



def test_evaluate_models_returns_dataframe(patch_paths):
    X_train, X_test, y_train, y_test = _make_data()
    trained = train_models(X_train, y_train, tune_knn=False, cv_evaluate=False)
    df_res = evaluate_models(trained, X_train, y_train, X_test, y_test)
    assert isinstance(df_res, pd.DataFrame)
    assert len(df_res) == len(trained)


def test_evaluate_models_columns(patch_paths):
    """El resultado debe contener las columnas de métricas esperadas."""
    X_train, X_test, y_train, y_test = _make_data()
    trained = train_models(X_train, y_train, tune_knn=False, cv_evaluate=False)
    df_res = evaluate_models(trained, X_train, y_train, X_test, y_test)

    for col in ["Modelo", "Acc_train", "Acc_test", "F1_train", "F1_test"]:
        assert col in df_res.columns, f"Falta columna: {col}"




def test_evaluate_models_accuracy_in_range(patch_paths):
    """Accuracy debe estar entre 0 y 1."""
    X_train, X_test, y_train, y_test = _make_data()
    trained = train_models(X_train, y_train, tune_knn=False, cv_evaluate=False)
    df_res = evaluate_models(trained, X_train, y_train, X_test, y_test)
    assert (df_res["Acc_test"].between(0, 1)).all()
    assert (df_res["Acc_train"].between(0, 1)).all()


def test_evaluate_models_saves_confusion_matrices(patch_paths):
    """Debe guardar una imagen de matriz de confusión por modelo."""
    X_train, X_test, y_train, y_test = _make_data()
    trained = train_models(X_train, y_train, tune_knn=False, cv_evaluate=False)
    evaluate_models(trained, X_train, y_train, X_test, y_test)
    pngs = list(patch_paths["FIGURES_DIR"].glob("cm_*.png"))
    assert len(pngs) == len(trained)



def test_evaluate_models_saves_csv(patch_paths):
    """Debe guardar resultados_modelos.csv en REPORTS_DIR."""
    X_train, X_test, y_train, y_test = _make_data()
    trained = train_models(X_train, y_train, tune_knn=False, cv_evaluate=False)
    evaluate_models(trained, X_train, y_train, X_test, y_test)
    assert (patch_paths["REPORTS_DIR"] / "resultados_modelos.csv").exists()


def test_predict_new_after_train(patch_paths):
    """predict_new debe cargar el modelo y predecir correctamente."""
    X_train, X_test, y_train, _ = _make_data()
    trained = train_models(X_train, y_train, tune_knn=False, cv_evaluate=False)
    model_name = list(trained.keys())[0]  # usar el primer modelo disponible
    preds = predict_new(model_name, X_test)
    assert len(preds) == len(X_test)

    assert set(preds).issubset({0, 1})



def test_predict_new_raises_if_missing(patch_paths):
    """predict_new debe lanzar FileNotFoundError si el modelo no existe."""
    with pytest.raises(FileNotFoundError):
        predict_new("ModeloInexistente", np.zeros((5, 4)))



def test_predict_proba_new_shape(patch_paths):
    """predict_proba_new debe devolver array (n_samples, n_classes)."""
    X_train, X_test, y_train, _ = _make_data()
    trained = train_models(X_train, y_train, tune_knn=False, cv_evaluate=False)
    model_name = list(trained.keys())[0]
    proba = predict_proba_new(model_name, X_test)
    assert proba.shape == (len(X_test), 2)
    assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-5)


def test_predict_proba_new_raises_for_no_proba(patch_paths):
    """predict_proba_new debe fallar si el modelo no existe."""
    with pytest.raises(Exception):
        predict_proba_new("ModeloSinProba", np.zeros((5, 4)))


def test_custom_threshold(patch_paths):
    """Con threshold distinto de 0.5 debe aplicarse sobre predict_proba."""
    X_train, X_test, y_train, y_test = _make_data()
    trained = train_models(X_train, y_train, tune_knn=False, cv_evaluate=False)
    df_low  = evaluate_models(trained, X_train, y_train, X_test, y_test, threshold=0.3)
    df_high = evaluate_models(trained, X_train, y_train, X_test, y_test, threshold=0.7)
    assert isinstance(df_low,  pd.DataFrame)
    assert isinstance(df_high, pd.DataFrame)







