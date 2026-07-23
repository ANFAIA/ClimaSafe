"""
BayesianRiskDiagnosis — Red bayesiana ligera para diagnóstico inverso.

Propósito: cuando el modelo predice PELIGRO o PRECAUCIÓN, la red responde
"¿qué factor es más probable que lo esté causando?" y "¿cómo cambiaría si
modificáramos un factor?".

Entrenada con distribuciones realistas de edad y grasa corporal (población
española). Modelos independientes para calor y frío.

DAG:
  T (temperatura)   ──┐
  V (vulnerabilidad) ←├── G (grasa)
                      ├── E (edad)
  T ──→ R (riesgo) ←──┘
  R ──→ A (alerta)
  T ──→ A (efecto directo de temperatura extrema)

Dependencias: pgmpy
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.inference import VariableElimination


# ---------------------------------------------------------------------------
# Discretización
# ---------------------------------------------------------------------------

def discretizar_temp_calor(t: float) -> int:
    if t < 20: return 0
    if t < 30: return 1
    return 2


def discretizar_temp_frio(t: float) -> int:
    if t > 10: return 0
    if t > 0: return 1
    return 2


def discretizar_grasa(g: float) -> int:
    return 1 if g > 25 else 0


def discretizar_edad(e: float) -> int:
    if e < 35: return 0
    if e < 60: return 1
    return 2


def discretizar_vulnerabilidad(e_disc: int, g_disc: int) -> int:
    return 1 if e_disc >= 2 or g_disc >= 1 else 0


def discretizar_riesgo(r: float) -> int:
    if r < 0.3: return 0
    if r < 0.6: return 1
    return 2


def discretizar_alerta(r_disc: int, t_disc: int) -> int:
    return max(r_disc, t_disc)


ETIQUETAS_T = {0: "normal", 1: "moderado", 2: "extremo"}
ETIQUETAS_G = {0: "normal", 1: "alta"}
ETIQUETAS_E = {0: "joven", 1: "adulto", 2: "mayor"}
ETIQUETAS_V = {0: "baja", 1: "alta"}
ETIQUETAS_R = {0: "bajo", 1: "medio", 2: "alto"}
ETIQUETAS_A = {0: "seguro", 1: "precaucion", 2: "peligro"}


# ---------------------------------------------------------------------------
# Modelo
# ---------------------------------------------------------------------------

ESTRUCTURA = [("T", "R"), ("V", "R"), ("E", "V"), ("G", "V"), ("R", "A"), ("T", "A")]


class BayesianRiskDiagnosis:
    def __init__(self, clase: str = "calor"):
        if clase not in ("calor", "frio"):
            raise ValueError(f"clase debe ser 'calor' o 'frio', no {clase!r}")
        self.clase = clase
        self._discretizar_temp = (
            discretizar_temp_calor if clase == "calor" else discretizar_temp_frio
        )
        self._model: DiscreteBayesianNetwork | None = None
        self._inference: VariableElimination | None = None
        self._fitted = False

    def fit(self, df: pd.DataFrame):
        self._model = DiscreteBayesianNetwork(ebunch=ESTRUCTURA)
        self._model.fit(df)
        self._inference = VariableElimination(self._model)
        self._fitted = True

    def fit_from_continuous(
        self,
        temperaturas: np.ndarray,
        grasas: np.ndarray,
        edades: np.ndarray,
        riesgos: np.ndarray,
    ):
        alertas = np.array([
            discretizar_alerta(
                discretizar_riesgo(r),
                self._discretizar_temp(t),
            )
            for r, t in zip(riesgos, temperaturas)
        ])
        t_disc = np.array([self._discretizar_temp(t) for t in temperaturas])
        g_disc = np.array([discretizar_grasa(g) for g in grasas])
        e_disc = np.array([discretizar_edad(e) for e in edades])
        v_disc = np.array([
            discretizar_vulnerabilidad(ed, gr)
            for ed, gr in zip(e_disc, g_disc)
        ])
        df = pd.DataFrame({
            "T": t_disc,
            "G": g_disc,
            "E": e_disc,
            "V": v_disc,
            "R": [discretizar_riesgo(r) for r in riesgos],
            "A": alertas,
        })
        self.fit(df)

    def _require_fitted(self):
        if not self._fitted:
            raise RuntimeError("Modelo no entrenado. Llama a fit() primero.")

    # -----------------------------------------------------------------------
    # Diagnóstico inverso
    # -----------------------------------------------------------------------

    def diagnosis_inverso(self, alerta: int) -> dict:
        """Dada una alerta, devuelve la probabilidad de cada factor."""
        self._require_fitted()
        p_t = self._inference.query(variables=["T"], evidence={"A": alerta})
        p_g = self._inference.query(variables=["G"], evidence={"A": alerta})
        p_e = self._inference.query(variables=["E"], evidence={"A": alerta})
        p_v = self._inference.query(variables=["V"], evidence={"A": alerta})
        p_r = self._inference.query(variables=["R"], evidence={"A": alerta})

        def _argmax_label(p, labels):
            idx = int(np.argmax(p.values))
            return labels[idx], {labels[i]: round(float(p.values[i]), 3) for i in range(len(labels))}

        t_val, t_dist = _argmax_label(p_t, ETIQUETAS_T)
        g_val, g_dist = _argmax_label(p_g, ETIQUETAS_G)
        e_val, e_dist = _argmax_label(p_e, ETIQUETAS_E)
        v_val, v_dist = _argmax_label(p_v, ETIQUETAS_V)
        r_val, r_dist = _argmax_label(p_r, ETIQUETAS_R)

        return {
            "alerta": ETIQUETAS_A[alerta],
            "factor_mas_probable": {
                "temperatura": t_val,
                "grasa": g_val,
                "edad": e_val,
                "vulnerabilidad": v_val,
                "riesgo_modelo": r_val,
            },
            "distribuciones": {
                "temperatura": t_dist,
                "grasa": g_dist,
                "edad": e_dist,
                "vulnerabilidad": v_dist,
                "riesgo_modelo": r_dist,
            },
        }

    # -----------------------------------------------------------------------
    # Contrafactual
    # -----------------------------------------------------------------------

    def contrafactual(self, temp: float, grasa: float, edad: float) -> dict:
        """Dado un perfil, evalúa cómo cambia la alerta al modificar cada factor."""
        self._require_fitted()
        t = self._discretizar_temp(temp)
        g = discretizar_grasa(grasa)
        e = discretizar_edad(edad)
        v = discretizar_vulnerabilidad(e, g)

        p_base = self._inference.query(variables=["A"], evidence={"T": t, "G": g, "E": e, "V": v})
        clase_base = int(np.argmax(p_base.values))

        escenarios = []
        for factor, valor_alt, desc, ev_extra in [
            ("T", 0, "temperatura normal", {}),
            ("T", 1, "temperatura moderada", {}),
            ("G", 0, "grasa corporal normal (≤25%)", {}),
            ("E", 0, "edad joven (<35 años)", {}),
            ("E", 1, "edad adulta (35-60 años)", {}),
        ]:
            ev = {"T": t, "G": g, "E": e, "V": v}
            ev[factor] = valor_alt

            if factor == "E":
                ev["V"] = discretizar_vulnerabilidad(valor_alt, g)
            elif factor == "G":
                ev["V"] = discretizar_vulnerabilidad(e, valor_alt)

            if ev == {"T": t, "G": g, "E": e, "V": v}:
                continue

            p_nueva = self._inference.query(variables=["A"], evidence=ev)
            clase_nueva = int(np.argmax(p_nueva.values))
            if clase_nueva < clase_base:
                escenarios.append({
                    "cambio": f"Si {desc}",
                    "alerta_actual": ETIQUETAS_A[clase_base],
                    "alerta_tras_cambio": ETIQUETAS_A[clase_nueva],
                    "probabilidad_actual": round(float(p_base.values[clase_base]), 3),
                    "probabilidad_tras_cambio": round(float(p_nueva.values[clase_nueva]), 3),
                    "mejora": True,
                })

        return {
            "perfil_actual": {
                "temperatura": ETIQUETAS_T[t],
                "grasa": ETIQUETAS_G[g],
                "edad": ETIQUETAS_E[e],
                "vulnerabilidad": ETIQUETAS_V[v],
                "alerta": ETIQUETAS_A[clase_base],
            },
            "escenarios": sorted(escenarios, key=lambda x: x["probabilidad_tras_cambio"], reverse=True),
        }

    # -----------------------------------------------------------------------
    # Persistencia
    # -----------------------------------------------------------------------

    def save(self, path: str):
        import joblib
        joblib.dump({"model": self._model, "fitted": self._fitted, "clase": self.clase}, path)

    def load(self, path: str):
        import joblib
        data = joblib.load(path)
        self.clase = data.get("clase", "calor")
        self._discretizar_temp = (
            discretizar_temp_calor if self.clase == "calor" else discretizar_temp_frio
        )
        self._model = data["model"]
        self._fitted = data["fitted"]
        if self._fitted:
            self._inference = VariableElimination(self._model)

    @property
    def cpd_info(self) -> list[dict]:
        if not self._fitted:
            return []
        return [
            {"variable": c.variable, "parents": c.variables[1:]}
            for c in self._model.get_cpds()
        ]
