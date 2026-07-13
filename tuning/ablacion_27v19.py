"""Ablación limpia de features: 27 vs 19 con el MISMO label.

Motivación: en la iteración de hoy se cambiaron a la vez el conjunto de
features (19 -> 27) y el label (se añadió un suelo de mortalidad), por lo que
la mejora observada en Rec_riesgo no es atribuible a las features nuevas.
Este experimento fija el label (el nuevo, con suelo de mortalidad) y entrena
cada modelo con (a) las 27 features y (b) solo las 19 antiguas, usando los
MISMOS hiperparámetros que los modelos entrenados hoy
(models/XGBoost_{clase}.joblib, models/RandomForest_{clase}.joblib).

Métrica de selección del proyecto: Rec_riesgo = recall macro de las clases
1 (precaución) y 2 (peligro). Política: preferir falsos positivos.

Uso:
    PYTHONPATH=<repo> <venv>/bin/python tuning/ablacion_27v19.py \
        [--data-dir DATA] [--models-dir MODELS] [--out CSV]

Por defecto lee los CSV y modelos del checkout principal (solo lectura) y
escribe el CSV de resultados en reports/ del directorio de trabajo actual.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.utils.class_weight import compute_sample_weight

MAIN = Path("/home/cacelas/Documentos/anfaia/ClimaSafeAI")

# n_jobs limitado: hay otros procesos entrenando en paralelo en esta máquina.
N_JOBS = 2

FEATURES_19 = [
    "t2m_c", "rh", "wind_speed_kmh", "sp", "heat_index_c", "wbgt_c",
    "wind_chill_c", "heat_index_mean", "heat_index_std", "heat_index_min",
    "horas_sobre_umbral", "wind_chill_mean", "wind_chill_std",
    "wind_chill_max", "horas_bajo_umbral", "heat_index_c_lag1",
    "wind_chill_mean_roll3", "wind_chill_mean_roll7",
    "dias_consec_bajo_umbral",
]

FEATURES_8_NUEVAS = [
    "heat_index_c_roll3", "heat_index_c_roll7", "dias_consec_sobre_umbral",
    "grados_dia_calor_roll7", "grados_dia_calor_roll14",
    "wind_chill_mean_roll14", "grados_dia_frio_roll7",
    "grados_dia_frio_roll14",
]


def cargar_datos(data_dir: Path, clase: str):
    X_train = pd.read_csv(data_dir / f"X_train_{clase}.csv")
    X_test = pd.read_csv(data_dir / f"X_test_{clase}.csv")
    y_train = pd.read_csv(data_dir / f"y_train_{clase}.csv").iloc[:, 0]
    y_test = pd.read_csv(data_dir / f"y_test_{clase}.csv").iloc[:, 0]
    return X_train, X_test, y_train, y_test


def clonar_modelo(models_dir: Path, nombre: str, clase: str):
    """Clona el modelo entrenado hoy conservando sus hiperparámetros."""
    base = joblib.load(models_dir / f"{nombre}_{clase}.joblib")
    modelo = clone(base)
    modelo.set_params(n_jobs=N_JOBS)
    return modelo


def evaluar(modelo, X_test, y_test) -> dict:
    y_pred = modelo.predict(X_test)
    rec_por_clase = recall_score(
        y_test, y_pred, labels=[1, 2], average=None, zero_division=0
    )
    prec_por_clase = precision_score(
        y_test, y_pred, labels=[1, 2], average=None, zero_division=0
    )
    return {
        # Métrica de selección: recall macro de las clases de riesgo (1 y 2)
        "Rec_riesgo": rec_por_clase.mean(),
        "F1_macro": f1_score(y_test, y_pred, average="macro", zero_division=0),
        "Acc_test": accuracy_score(y_test, y_pred),
        "Prec_c1": prec_por_clase[0],
        "Rec_c1": rec_por_clase[0],
        "Prec_c2": prec_por_clase[1],
        "Rec_c2": rec_por_clase[1],
    }


def ejecutar_variante(modelo_nombre, clase, features, etiqueta_variante,
                      data_dir, models_dir):
    X_train, X_test, y_train, y_test = cargar_datos(data_dir, clase)
    faltan = set(features) - set(X_train.columns)
    if faltan:
        raise ValueError(f"Faltan columnas en X_train_{clase}: {faltan}")

    modelo = clonar_modelo(models_dir, modelo_nombre, clase)
    fit_kwargs = {}
    if modelo_nombre == "XGBoost":
        # XGBoost se entrena en este proyecto con sample_weight balanceado;
        # RandomForest ya lleva class_weight='balanced' en sus params.
        fit_kwargs["sample_weight"] = compute_sample_weight(
            "balanced", y_train
        )

    modelo.fit(X_train[features], y_train, **fit_kwargs)
    metricas = evaluar(modelo, X_test[features], y_test)
    return {
        "clase": clase,
        "modelo": modelo_nombre,
        "variante": etiqueta_variante,
        "n_features": len(features),
        **metricas,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-dir", type=Path,
                        default=MAIN / "data" / "processed")
    parser.add_argument("--models-dir", type=Path, default=MAIN / "models")
    parser.add_argument("--out", type=Path,
                        default=Path("reports") / "ablacion_27v19.csv")
    args = parser.parse_args()

    features_27 = FEATURES_19 + FEATURES_8_NUEVAS
    resultados = []
    for clase in ("calor", "frio"):
        for modelo_nombre in ("XGBoost", "RandomForest"):
            for etiqueta, feats in (("27f", features_27),
                                    ("19f", FEATURES_19)):
                print(f"Entrenando {modelo_nombre} {clase} {etiqueta}...",
                      flush=True)
                fila = ejecutar_variante(
                    modelo_nombre, clase, feats, etiqueta,
                    args.data_dir, args.models_dir,
                )
                resultados.append(fila)
                print(json.dumps(fila, indent=2, default=float), flush=True)

    df = pd.DataFrame(resultados)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"\nResultados guardados en {args.out}\n")
    with pd.option_context("display.width", 200):
        print(df.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
