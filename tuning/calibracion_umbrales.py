"""
calibracion_umbrales.py — Calibración de umbrales de decisión por clase.

Problema: los modelos desplegados (XGBoost_calor, RandomForest_frio) predicen
con argmax sobre predict_proba. Su cuello de botella documentado es la
PRECISIÓN de las clases de riesgo (~0.16-0.37): muchas falsas alarmas. Este
script explora umbrales de probabilidad por clase sobre un tramo de VALIDACIÓN
temporal (nunca sobre test) y expone el trade-off recall/precisión sin
reentrenar el modelo desplegado.

Regla de decisión en cascada (por severidad):
    p2 = P(clase 2 = peligro),  p_riesgo = P(1) + P(2)
    si   p2       >= t2  ->  2 (peligro)
    sino si p_riesgo >= t1  ->  1 (precaución)
    sino                 ->  0 (seguro)

Justificación: es la extensión natural a 3 clases ordinales del umbral binario
que ya usaba predict_model.DECISION_THRESHOLD. "Peligro" exige evidencia
directa de la clase 2; "precaución" solo exige suficiente masa de probabilidad
de riesgo total, coherente con la política del sistema (mejor sobre-avisar que
no avisar). Con t2=t1=0.5 NO equivale exactamente al argmax: por eso el argmax
se evalúa aparte como línea base.

Metodología (sin mirar test):
  1. Reconstruye las fechas de las filas de X_train_{clase}.csv desde
     dataset_{clase}_labeled.parquet replicando el split por fecha de
     build_features.preprocess_data (último 20% de fechas distintas -> test).
  2. Reserva el último 15% temporal de train como VALIDACIÓN.
  3. Ajusta un CLON del modelo desplegado (mismos hiperparámetros, misma
     receta de pesos que train_model.train_models) solo con el sub-train
     (train sin validación). Motivo: el modelo desplegado se entrenó con TODO
     el train, de modo que sus probabilidades sobre la validación serían
     in-sample y optimistas (medido: rec_riesgo ~0.82 in-sample vs 0.63 real
     para calor) y los umbrales elegidos ahí no transferirían. El clon da
     probabilidades honestas para ELEGIR umbrales; el modelo desplegado, sin
     tocar, es el que luego se evalúa en test con esos umbrales.
  4. Barre (t1, t2) en rejilla, tabula la frontera de Pareto recall/precisión
     de las clases de riesgo y elige puntos de operación:
       (a) max Rec_riesgo con Prec_riesgo >= la del argmax en validación
       (b) max Prec_riesgo con Rec_riesgo >= la del argmax en validación
       (c) max F2_riesgo (F-beta macro de clases 1-2 con beta=2: pesa el
           recall el doble, coherente con "mejor sobre-avisar")
  5. SOLO los puntos elegidos se evalúan una única vez en test, contra el
     argmax del modelo desplegado.

Uso (desde la raíz del repo):
    python tuning/calibracion_umbrales.py \
        --data-dir /ruta/a/data/processed --models-dir /ruta/a/models

Salidas (rutas del repo desde el que se ejecuta):
    reports/calibracion_umbrales_frontera_{clase}.csv
    reports/calibracion_umbrales_puntos.csv
    reports/figures/calibracion_frontera.png
"""
from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.utils.class_weight import compute_sample_weight

from climasafeai.utils.paths import MODELS_DIR, PROCESSED_DATA_DIR, REPORTS_DIR, FIGURES_DIR

# Modelo desplegado por clase (ver conclusiones_modelos.md)
MODELO_DESPLEGADO = {"calor": "XGBoost_calor", "frio": "RandomForest_frio"}

VAL_FRAC = 0.15        # último 15% de fechas de train -> validación
TEST_FRAC = 0.20       # regla de preprocess_data: último 20% de fechas -> test
GRID = np.round(np.arange(0.02, 0.99, 0.02), 2)  # rejilla de umbrales

# Paleta (dataviz skill, modo claro)
C_FRONTERA = "#2a78d6"   # azul   — frontera de Pareto
C_ARGMAX = "#e34948"     # rojo   — línea base argmax
C_PUNTOS = "#008300"     # verde  — puntos de operación elegidos
C_NUBE = "#c3c2b7"       # gris   — resto de combinaciones barridas


# ---------------------------------------------------------------------------
# Datos: reconstrucción del split temporal
# ---------------------------------------------------------------------------

def cargar_datos(clase: str, data_dir):
    """Carga X/y de train y test + fechas de las filas de train.

    Las fechas no están en X_train_{clase}.csv: se reconstruyen desde el
    parquet etiquetado replicando exactamente el split por fecha de
    preprocess_data (drop_duplicates + último TEST_FRAC de fechas únicas a
    test). La reconstrucción se verifica contra y_train_{clase}.csv: si las
    etiquetas no coinciden fila a fila, se aborta.
    """
    X_train = pd.read_csv(data_dir / f"X_train_{clase}.csv")
    X_test = pd.read_csv(data_dir / f"X_test_{clase}.csv")
    y_train = pd.read_csv(data_dir / f"y_train_{clase}.csv").iloc[:, 0].to_numpy()
    y_test = pd.read_csv(data_dir / f"y_test_{clase}.csv").iloc[:, 0].to_numpy()

    d = pd.read_parquet(data_dir / f"dataset_{clase}_labeled.parquet").drop_duplicates()
    fechas = pd.to_datetime(d["fecha"])
    fechas_unicas = np.sort(fechas.unique())
    n_test = max(1, round(len(fechas_unicas) * TEST_FRAC))
    mask_test = fechas.isin(set(fechas_unicas[-n_test:]))

    y_train_rec = d.loc[~mask_test, f"clase_riesgo_{clase}"].to_numpy()
    if len(y_train_rec) != len(y_train) or not (y_train_rec == y_train).all():
        raise RuntimeError(
            f"[{clase}] la reconstrucción del split por fecha no coincide con "
            "y_train_{clase}.csv — revisa que el parquet y los CSV procesados "
            "salgan de la misma ejecución del pipeline."
        )
    fechas_train = fechas[~mask_test].reset_index(drop=True)
    return X_train, y_train, X_test, y_test, fechas_train


def mascara_validacion(fechas_train: pd.Series, val_frac: float = VAL_FRAC) -> np.ndarray:
    """True en las filas del último `val_frac` de fechas distintas de train."""
    fu = np.sort(fechas_train.unique())
    n_val = max(1, round(len(fu) * val_frac))
    return fechas_train.isin(set(fu[-n_val:])).to_numpy()


# ---------------------------------------------------------------------------
# Clon honesto para validación
# ---------------------------------------------------------------------------

def ajustar_clon(modelo_desplegado, X_sub, y_sub):
    """Clona el modelo desplegado (mismos hiperparámetros) y lo ajusta SOLO
    con el sub-train, replicando la receta de train_model.train_models:
    XGBoost no acepta class_weight -> sample_weight balanceado en fit().
    """
    clon = clone(modelo_desplegado)
    if "n_jobs" in clon.get_params():
        clon.set_params(n_jobs=2)
    if "xgboost" in type(clon).__module__:
        clon.fit(X_sub, y_sub, sample_weight=compute_sample_weight("balanced", y_sub))
    else:
        clon.fit(X_sub, y_sub)
    return clon


# ---------------------------------------------------------------------------
# Regla de decisión y métricas
# ---------------------------------------------------------------------------

def predecir_con_umbrales(proba: np.ndarray, t1: float, t2: float) -> np.ndarray:
    """Cascada por severidad: P(2)>=t2 -> 2; P(1)+P(2)>=t1 -> 1; si no, 0."""
    p2 = proba[:, 2]
    p_riesgo = proba[:, 1] + proba[:, 2]
    return np.where(p2 >= t2, 2, np.where(p_riesgo >= t1, 1, 0))


def metricas(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Recall/precisión por clase de riesgo + agregados. Vectorizado a mano
    (llamar a sklearn ~2400 veces en el barrido sería lento)."""
    out = {}
    f1s = []
    for c in (0, 1, 2):
        tp = np.sum((y_pred == c) & (y_true == c))
        rec = tp / max(np.sum(y_true == c), 1)
        prec = tp / max(np.sum(y_pred == c), 1)
        f1s.append(2 * prec * rec / max(prec + rec, 1e-12))
        if c:
            out[f"rec{c}"], out[f"prec{c}"] = rec, prec
    out["rec_riesgo"] = (out["rec1"] + out["rec2"]) / 2
    out["prec_riesgo"] = (out["prec1"] + out["prec2"]) / 2
    out["f1_macro"] = float(np.mean(f1s))
    # F-beta macro de las clases de riesgo con beta=2 (recall pesa doble)
    f2s = []
    for c in (1, 2):
        p, r = out[f"prec{c}"], out[f"rec{c}"]
        f2s.append(5 * p * r / max(4 * p + r, 1e-12))
    out["f2_riesgo"] = float(np.mean(f2s))
    # % de filas con aviso (clase 1 o 2): coste operativo de las alarmas
    out["pct_avisos"] = float(np.mean(y_pred != 0))
    return out


def barrer_umbrales(proba_val: np.ndarray, y_val: np.ndarray) -> pd.DataFrame:
    """Evalúa la cascada para toda la rejilla (t1, t2) sobre validación."""
    filas = []
    for t2 in GRID:
        for t1 in GRID:
            m = metricas(y_val, predecir_con_umbrales(proba_val, t1, t2))
            m["t1"], m["t2"] = t1, t2
            filas.append(m)
    return pd.DataFrame(filas)


def frontera_pareto(df: pd.DataFrame) -> pd.DataFrame:
    """Puntos no dominados en (rec_riesgo, prec_riesgo)."""
    df = df.sort_values(["rec_riesgo", "prec_riesgo"], ascending=[False, False])
    mejores, mejor_prec = [], -1.0
    for _, fila in df.iterrows():
        if fila["prec_riesgo"] > mejor_prec:
            mejores.append(fila)
            mejor_prec = fila["prec_riesgo"]
    return pd.DataFrame(mejores).sort_values("rec_riesgo").reset_index(drop=True)


KEEP_ALIVE = 0.5  # clase 1 (precaución) debe conservar >= 50% de su recall argmax


def elegir_puntos(df_barrido: pd.DataFrame, base_val: dict) -> dict:
    """Puntos de operación (a), (b), (c) — ver docstring del módulo.

    Dos guardas, imprescindibles con estas probabilidades:

    1. PELIGRO INTOCABLE: todo punto debe conservar el recall de la clase 2
       (peligro) al menos en su valor argmax. Es la prioridad del sistema
       (no dejar de avisar una muerte por calor/frío) y además la guarda que
       impide el sobreajuste a validación: sin ella el barrido premia
       esquinas de la rejilla (p.ej. t1≈0.05: casi todo pasa a "precaución")
       que suben el agregado rec_riesgo en validación bajando el recall de
       peligro -- y en test se desploman (f1_macro 0.56->0.49, recall de
       peligro 0.69->0.52 en calor).
    2. PRECAUCIÓN VIVA: la clase 1 no puede caer por debajo del 50% de su
       recall argmax -- evita el colapso simétrico (t1 tan alto que el modelo
       deja de emitir "precaución", rec1≈0, y la precisión agregada se
       dispara por 1-2 muestras que no transfieren a test: prec_riesgo
       0.23->0.08 en frío).

    Con ambas guardas el trade-off que queda es real y estable entre
    validación y test.
    """
    vivos = df_barrido[
        (df_barrido["rec2"] >= base_val["rec2"])
        & (df_barrido["rec1"] >= KEEP_ALIVE * base_val["rec1"])
    ]
    if not len(vivos):                          # salvaguarda: no debería ocurrir
        vivos = df_barrido
    puntos = {}

    # (a) máxima cobertura: max rec_riesgo manteniendo prec_riesgo >= argmax
    cand_a = vivos[vivos["prec_riesgo"] >= base_val["prec_riesgo"]]
    if len(cand_a):
        puntos["a_max_recall"] = cand_a.sort_values(
            ["rec_riesgo", "prec_riesgo"], ascending=False).iloc[0]

    # (b) máxima precisión: max prec_riesgo manteniendo rec_riesgo >= argmax
    cand_b = vivos[vivos["rec_riesgo"] >= base_val["rec_riesgo"]]
    if len(cand_b):
        puntos["b_max_precision"] = cand_b.sort_values(
            ["prec_riesgo", "rec_riesgo"], ascending=False).iloc[0]

    # (c) recomendado: max F2 de riesgo (recall pesa doble, coherente con
    #     "mejor sobre-avisar") sin perder precisión agregada frente al
    #     argmax -> el mejor equilibrio que no empeora ninguna cara del argmax.
    cand_c = vivos[vivos["prec_riesgo"] >= base_val["prec_riesgo"]]
    if not len(cand_c):
        cand_c = vivos
    puntos["c_recomendado"] = cand_c.sort_values(
        ["f2_riesgo", "prec_riesgo"], ascending=False).iloc[0]
    return puntos


# ---------------------------------------------------------------------------
# Gráfica
# ---------------------------------------------------------------------------

def dibujar_frontera(resultados: dict, path_png) -> None:
    """Un panel por clase: nube barrida (gris), frontera de Pareto (azul),
    argmax (rojo) y puntos de operación (verde), en validación."""
    fig, axes = plt.subplots(1, len(resultados), figsize=(6.4 * len(resultados), 5.4))
    axes = np.atleast_1d(axes)
    for ax, (clase, r) in zip(axes, resultados.items()):
        df, front = r["barrido"], r["frontera"]
        ax.scatter(df["rec_riesgo"], df["prec_riesgo"], s=8, color=C_NUBE,
                   alpha=0.35, linewidths=0, label="rejilla (t1, t2)")
        ax.plot(front["rec_riesgo"], front["prec_riesgo"], color=C_FRONTERA,
                lw=2, marker="o", ms=4, label="frontera de Pareto")
        b = r["base_val"]
        ax.scatter([b["rec_riesgo"]], [b["prec_riesgo"]], marker="X", s=140,
                   color=C_ARGMAX, zorder=5, label="argmax (actual)")
        # Etiquetas escalonadas con línea guía: los 3 puntos de operación caen
        # muy juntos (la frontera es plana cerca del argmax), así que se
        # separan las anotaciones para que no se solapen.
        offsets = {"a_max_recall": (26, 34), "b_max_precision": (30, -6),
                   "c_recomendado": (26, -40)}
        primero = True
        lineas_caja = []
        for nombre, p in r["puntos"].items():
            ax.scatter([p["rec_riesgo"]], [p["prec_riesgo"]], marker="D", s=70,
                       color=C_PUNTOS, zorder=6,
                       label="puntos de operación" if primero else None)
            primero = False
            ax.annotate(f"({nombre[0]})", (p["rec_riesgo"], p["prec_riesgo"]),
                        textcoords="offset points",
                        xytext=offsets.get(nombre, (8, 8)),
                        fontsize=9, color="#1a1a19", fontweight="bold",
                        ha="center",
                        arrowprops=dict(arrowstyle="-", color="#8a8a86", lw=0.8))
            lineas_caja.append(
                f"({nombre[0]}) {nombre.split('_', 1)[1]}: "
                f"t1={p['t1']:.2f}  t2={p['t2']:.2f}")
        ax.text(0.02, 0.02, "\n".join(lineas_caja), transform=ax.transAxes,
                fontsize=8, color="#40403e", va="bottom", ha="left",
                bbox=dict(boxstyle="round,pad=0.4", fc="#f6f5f0",
                          ec="#d9d7cf", lw=0.8))
        ax.set_title(f"{clase} — {MODELO_DESPLEGADO[clase]} (validación)",
                     fontsize=11, color="#1a1a19")
        ax.set_xlabel("Rec_riesgo (recall macro clases 1-2)", fontsize=9.5)
        ax.set_ylabel("Prec_riesgo (precisión macro clases 1-2)", fontsize=9.5)
        ax.grid(True, color="#eceae4", lw=0.8)
        for lado in ("top", "right"):
            ax.spines[lado].set_visible(False)
        ax.legend(loc="upper right", fontsize=8.5, frameon=False)
    fig.suptitle("Calibración de umbrales por clase — frontera recall/precisión",
                 fontsize=12.5)
    fig.tight_layout()
    fig.savefig(path_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"    Gráfica guardada -> {path_png}")


# ---------------------------------------------------------------------------
# Pipeline principal
# ---------------------------------------------------------------------------

def calibrar_clase(clase: str, data_dir, models_dir) -> dict:
    print(f"\n{'='*70}\n  {clase.upper()} — {MODELO_DESPLEGADO[clase]}\n{'='*70}")
    X_train, y_train, X_test, y_test, fechas_train = cargar_datos(clase, data_dir)
    mval = mascara_validacion(fechas_train)
    X_sub, y_sub = X_train[~mval], y_train[~mval]
    X_val, y_val = X_train[mval], y_train[mval]
    print(f"  sub-train: {len(y_sub)} filas (hasta {fechas_train[~mval].max().date()}) | "
          f"validación: {len(y_val)} filas (desde {fechas_train[mval].min().date()})")

    desplegado = joblib.load(models_dir / f"{MODELO_DESPLEGADO[clase]}.joblib")
    cache = REPORTS_DIR / f"calibracion_umbrales_probas_{clase}.npz"
    if cache.exists():
        print(f"  Probabilidades de validación cargadas de caché ({cache.name})")
        proba_val = np.load(cache)["proba_val"]
    else:
        print("  Ajustando clon (mismos hiperparámetros) con sub-train...")
        clon = ajustar_clon(desplegado, X_sub, y_sub)
        proba_val = clon.predict_proba(X_val)
        np.savez_compressed(cache, proba_val=proba_val)

    base_val = metricas(y_val, proba_val.argmax(axis=1))
    print(f"  argmax en validación (clon): rec_riesgo={base_val['rec_riesgo']:.4f} "
          f"prec_riesgo={base_val['prec_riesgo']:.4f}")

    print(f"  Barriendo rejilla {len(GRID)}x{len(GRID)} de (t1, t2)...")
    barrido = barrer_umbrales(proba_val, y_val)
    frontera = frontera_pareto(barrido)
    puntos = elegir_puntos(barrido, base_val)

    # ---- evaluación ÚNICA en test: solo argmax actual + puntos elegidos ----
    proba_test = desplegado.predict_proba(X_test)
    base_test = metricas(y_test, proba_test.argmax(axis=1))
    filas = [{"clase": clase, "punto": "argmax (actual)", "t1": np.nan, "t2": np.nan,
              **{f"val_{k}": v for k, v in base_val.items()},
              **{f"test_{k}": v for k, v in base_test.items()}}]
    for nombre, p in puntos.items():
        m_test = metricas(y_test, predecir_con_umbrales(proba_test, p["t1"], p["t2"]))
        filas.append({"clase": clase, "punto": nombre, "t1": p["t1"], "t2": p["t2"],
                      **{f"val_{k}": v for k, v in p.drop(["t1", "t2"]).items()},
                      **{f"test_{k}": v for k, v in m_test.items()}})
    return {"barrido": barrido, "frontera": frontera, "puntos": puntos,
            "base_val": base_val, "comparativa": pd.DataFrame(filas)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--data-dir", default=str(PROCESSED_DATA_DIR),
                        help="Carpeta con X_train_*.csv y dataset_*_labeled.parquet")
    parser.add_argument("--models-dir", default=str(MODELS_DIR),
                        help="Carpeta con los .joblib desplegados")
    args = parser.parse_args()
    data_dir, models_dir = Path(args.data_dir), Path(args.models_dir)

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    resultados, comparativas = {}, []
    for clase in ("calor", "frio"):
        r = calibrar_clase(clase, data_dir, models_dir)
        resultados[clase] = r
        comparativas.append(r["comparativa"])
        out_front = REPORTS_DIR / f"calibracion_umbrales_frontera_{clase}.csv"
        r["frontera"].round(4).to_csv(out_front, index=False)
        print(f"    Frontera -> {out_front}")

    df_comp = pd.concat(comparativas, ignore_index=True).round(4)
    out_csv = REPORTS_DIR / "calibracion_umbrales_puntos.csv"
    df_comp.to_csv(out_csv, index=False)

    cols = ["clase", "punto", "t1", "t2",
            "val_rec_riesgo", "val_prec_riesgo",
            "test_rec_riesgo", "test_prec_riesgo", "test_rec1", "test_prec1",
            "test_rec2", "test_prec2", "test_f1_macro", "test_pct_avisos"]
    print(f"\n{'='*70}\n  Comparativa argmax vs puntos de operación (val elige, test confirma)\n{'='*70}")
    print(df_comp[cols].to_string(index=False))
    print(f"\n  Guardado -> {out_csv}")

    dibujar_frontera(resultados, FIGURES_DIR / "calibracion_frontera.png")


if __name__ == "__main__":
    main()
