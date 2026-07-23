"""
calibrate.py — Calibración post-hoc de probabilidades con Isotonic Regression.

Se usa para RandomForest_frio y XGBoost_calor (calibra prob de PELIGRO).
"""
import joblib
import numpy as np
from sklearn.isotonic import IsotonicRegression
from climasafeai.utils.paths import ARTIFACTS_DIR


def _calibration_artifact_path(clase: str = "frio") -> str:
    return str(ARTIFACTS_DIR / f"iso_calib_{clase}.joblib")


def fit_isotonic(model, X_val, y_val, clase: str = "frio") -> IsotonicRegression:
    """Entrena un IsotonicRegression sobre la probabilidad de PELIGRO (clase 2).

    Parameters
    ----------
    model  : clasificador con .predict_proba()
    X_val  : features de validación (ya escaladas)
    y_val  : labels de validación
    clase  : "frio" por defecto (solo calibramos frío)

    Returns
    -------
    IsotonicRegression entrenado (también guardado en artifacts/iso_calib_{clase}.joblib)
    """
    proba = model.predict_proba(X_val)
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(proba[:, 2], (y_val == 2).astype(int))

    path = _calibration_artifact_path(clase)
    joblib.dump(iso, path)
    return iso


def load_isotonic(clase: str = "frio") -> IsotonicRegression | None:
    """Carga el calibrador isotónico si existe."""
    path = _calibration_artifact_path(clase)
    try:
        return joblib.load(path)
    except (FileNotFoundError, ValueError):
        return None


def calibrate_proba(proba: np.ndarray, iso: IsotonicRegression) -> np.ndarray:
    """Aplica calibración isotónica a la columna de PELIGRO (clase 2).

    1. Transforma proba[:, 2] con isotonic
    2. Re-escala para que sumen 1
    """
    proba = proba.copy()
    proba[:, 2] = iso.transform(proba[:, 2])
    proba = proba / proba.sum(axis=1, keepdims=True)
    return proba
