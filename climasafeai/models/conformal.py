"""
Split Conformal Prediction — incertidumbre por prediction sets.

Añade a cualquier clasificador probabilístico multiclase una estimación
de confianza por muestra:

  - "alta"   → set size = 1 (el modelo lo tiene claro)
  - "media"  → set size = 2 (duda entre dos clases)
  - "baja"   → set size = 3 (muy incierto, solo pasa si P se reparte)

No modifica la predicción, solo la etiqueta con el nivel de confianza.
"""
from __future__ import annotations

import numpy as np



class SplitConformalCalibrator:
    def __init__(self, alpha: float = 0.1):
        if not 0 < alpha < 1:
            raise ValueError(f"alpha debe estar entre 0 y 1, recibido {alpha}")
        self.alpha = alpha
        self.qhat: float | None = None
        self.n_classes: int | None = None

    def fit(self, proba_cal: np.ndarray, y_cal: np.ndarray):
        """Aprende el umbral qhat del calibration set.

        Parameters
        ----------
        proba_cal : (n_cal, n_classes) — predict_proba del modelo
        y_cal     : (n_cal,) — etiquetas verdaderas
        """
        n_cal = len(y_cal)
        scores = np.array([1.0 - proba_cal[i, y_cal[i]] for i in range(n_cal)])
        q = np.ceil((n_cal + 1) * (1 - self.alpha)) / n_cal
        self.qhat = np.quantile(scores, q, method="higher")
        self.n_classes = proba_cal.shape[1]
        return self

    def predict_set(self, proba: np.ndarray) -> list[set[int]]:
        """Devuelve prediction sets (conjuntos de clases posibles)."""
        if self.qhat is None:
            raise RuntimeError("Llama a fit() antes de predict_set().")
        sets = []
        for i in range(len(proba)):
            s = {c for c in range(proba.shape[1]) if 1.0 - proba[i, c] <= self.qhat}
            sets.append(s)
        return sets

    def confidence(self, proba: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Confianza por muestra: 'alta', 'media' o 'baja'.

        Returns
        -------
        (confianzas, set_sizes)
        confianzas  : array[str] de longitud n
        set_sizes   : array[int] de longitud n
        """
        sets = self.predict_set(proba)
        sizes = np.array([len(s) for s in sets])
        conf = np.where(sizes == 1, "alta", np.where(sizes == 2, "media", "baja"))
        return conf, sizes

    def save(self, path: str):
        import joblib
        joblib.dump({"alpha": self.alpha, "qhat": self.qhat, "n_classes": self.n_classes}, path)

    def load(self, path: str):
        import joblib
        data = joblib.load(path)
        self.alpha = data["alpha"]
        self.qhat = data["qhat"]
        self.n_classes = data["n_classes"]


def confidence_label(size: int) -> str:
    if size == 0:
        return "baja"
    if size == 1:
        return "alta"
    return "media"
