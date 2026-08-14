#!/usr/bin/env python
"""
scripts/exportar_onnx.py — WEB-011. Conversión a ONNX de los modelos del ensemble
más exportación de artefactos a JSON para el demo navegador.

Convierte los 3 modelos que usa `predict_ensemble`:

  - models/XGBoost_calor.joblib          → models/onnx/XGBoost_calor.onnx
  - models/RandomForest_frio.joblib      → models/onnx/RandomForest_frio.onnx
  - models/LSTM_province_hybrid.pt       → models/onnx/LSTM_province_hybrid.onnx

y exporta a models/onnx/ los artefactos que el navegador necesita para
reproducir `process_input`, `escalar_para_lstm` y las reglas de decisión
(umbrales, conformal, isotónica frío, class thresholds, INE, factores de
riesgo). Nada se inventa: todo lo exportado sale de los joblib/JSON del repo.

Notas de paridad:

  - XGBoost se convierte con onnxmltools (skl2onnx 1.20 ya no trae el
    conversor de XGBoost); el ONNX devuelve (label, proba) con la MISMA
    probabilidad que `predict_proba` (diff < 1e-6).
  - RandomForest se convierte con skl2onnx con `zipmap=False` para que la
    salida de probabilidades sea un tensor (1, 3) y no un dict, y devuelve la
    proba cruda (sin isotónica): la calibración isotónica de frío NO va dentro
    del ONNX — se exporta aparte en iso_calib_frio.json y la aplica el
    navegador (interpolación lineal con clip, como hace sklearn).
  - LSTM se exporta con torch.onnx.export (opset 17, batch dinámico) y
    devuelve LOGITS (logits_calor, logits_frio): el navegador aplica softmax.
  - El embedding de provincia del LSTM usa provincia_mapping.json (45
    provincias ordenadas alfabéticamente, idéntico a `get_province_idx`).

Uso:
    uv run python scripts/exportar_onnx.py            # exporta (skips si existe) + verifica paridad
    uv run python scripts/exportar_onnx.py --force    # re-exporta todo
    uv run python scripts/exportar_onnx.py --check-only  # solo verifica artefactos existentes
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from climasafeai.utils.paths import MODELS_DIR, ARTIFACTS_DIR

# ─────────────────────────────────────────────────────────────────────────────
# Dependencias opcionales (extra `onnx`): fallo claro, no traceback feo.
# ─────────────────────────────────────────────────────────────────────────────
def _check_deps() -> None:
    faltan = []
    for mod, nombre in [
        ("onnx", "onnx"),
        ("skl2onnx", "skl2onnx"),
        ("onnxmltools", "onnxmltools"),
        ("onnxruntime", "onnxruntime"),
        ("torch", "torch"),
    ]:
        try:
            __import__(mod)
        except ModuleNotFoundError:
            faltan.append(nombre)
    if faltan:
        print(
            "FALTAN DEPENDENCIAS ONNX: " + ", ".join(faltan)
            + "\nInstálalas con:  uv sync --all-extras  (o: uv pip install "
            + " ".join(faltan) + ")"
        )
        sys.exit(1)


OUTPUT_DIR = MODELS_DIR / "onnx"
ONNX_XGB = OUTPUT_DIR / "XGBoost_calor.onnx"
ONNX_RF = OUTPUT_DIR / "RandomForest_frio.onnx"
ONNX_LSTM = OUTPUT_DIR / "LSTM_province_hybrid.onnx"

TOLERANCIA_PARIDAD = 1e-3


# ─────────────────────────────────────────────────────────────────────────────
# Utilidades JSON
# ─────────────────────────────────────────────────────────────────────────────
def _python(o):
    """Convierte recursivamente numpy → tipos nativos para JSON."""
    if isinstance(o, bool):
        return bool(o)
    if isinstance(o, dict):
        return {str(k): _python(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_python(v) for v in o]
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, np.ndarray):
        return [_python(v) for v in o.tolist()]
    if isinstance(o, (np.floating, float)):
        return float(o)
    if isinstance(o, (np.integer, int)):
        return int(o)
    return o


def _write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(_python(data), f, indent=2, ensure_ascii=False, sort_keys=True)
    print(f"    {path.relative_to(MODELS_DIR)}")


# ─────────────────────────────────────────────────────────────────────────────
# Exportación ONNX
# ─────────────────────────────────────────────────────────────────────────────
def exportar_xgboost(force: bool = False) -> Path:
    import onnx
    from onnxmltools import convert_xgboost
    from onnxmltools.convert.common.data_types import FloatTensorType

    if ONNX_XGB.exists() and not force:
        print(f"    XGBoost_calor.onnx ya existe (usa --force para regenerar)")
        return ONNX_XGB

    print("> XGBoost_calor.joblib → ONNX (onnxmltools)...")
    model = joblib.load(MODELS_DIR / "XGBoost_calor.joblib")
    fn = joblib.load(ARTIFACTS_DIR / "feature_names_calor.joblib")
    # onnxmltools 1.16 no soporta opset > 15 para XGBoost; el default basta.
    onx = convert_xgboost(
        model,
        initial_types=[("X", FloatTensorType([None, len(fn)]))],
    )
    onnx.checker.check_model(onx)
    ONNX_XGB.parent.mkdir(parents=True, exist_ok=True)
    with open(ONNX_XGB, "wb") as f:
        f.write(onx.SerializeToString())
    print(f"    {ONNX_XGB.relative_to(MODELS_DIR)}")
    return ONNX_XGB


def exportar_random_forest(force: bool = False) -> Path:
    import onnx
    from skl2onnx import convert_sklearn
    from skl2onnx.common.data_types import FloatTensorType

    if ONNX_RF.exists() and not force:
        print(f"    RandomForest_frio.onnx ya existe (usa --force para regenerar)")
        return ONNX_RF

    print("> RandomForest_frio.joblib → ONNX (skl2onnx, zipmap=False)...")
    model = joblib.load(MODELS_DIR / "RandomForest_frio.joblib")
    fn = joblib.load(ARTIFACTS_DIR / "feature_names_frio.joblib")
    onx = convert_sklearn(
        model,
        initial_types=[("X", FloatTensorType([None, len(fn)]))],
        target_opset=17,
        options={id(model): {"zipmap": False}},
    )
    onnx.checker.check_model(onx)
    ONNX_RF.parent.mkdir(parents=True, exist_ok=True)
    with open(ONNX_RF, "wb") as f:
        f.write(onx.SerializeToString())
    print(f"    {ONNX_RF.relative_to(MODELS_DIR)}")
    return ONNX_RF


def exportar_lstm(force: bool = False) -> Path:
    import torch

    if ONNX_LSTM.exists() and not force:
        print(f"    LSTM_province_hybrid.onnx ya existe (usa --force para regenerar)")
        return ONNX_LSTM

    print("> LSTM_province_hybrid.pt → ONNX (torch.onnx.export, opset 17)...")
    from climasafeai.models.lstm_province_hybrid import (
        load_lstm_province_hybrid,
        DAILY_FEATURE_COLS,
    )
    from climasafeai.data.sequences import FEATURE_COLS_SEQ
    from climasafeai.data.weather_fetcher import get_province_idx, get_ine_features

    model = load_lstm_province_hybrid(device="cpu")
    h = model.hparams
    n_diarias = h["n_features_diarias"]
    assert len(DAILY_FEATURE_COLS) == n_diarias, (
        f"DAILY_FEATURE_COLS ({len(DAILY_FEATURE_COLS)}) != n_features_diarias del "
        f"checkpoint ({n_diarias})"
    )
    assert len(FEATURE_COLS_SEQ) == h["n_features"]

    # Entrada de ejemplo: batch 1, shapes del demo. Seq sintética (0.5) para
    # trazar el grafo; los valores concretos no importan para la exportación.
    x_seq = torch.zeros(1, 24, h["n_features"], dtype=torch.float32)
    pidx = torch.tensor([get_province_idx("Madrid")], dtype=torch.int64)
    x_ine = torch.zeros(1, h["n_features_provincia"], dtype=torch.float32)
    x_diarias = torch.zeros(1, n_diarias, dtype=torch.float32)

    with torch.no_grad():
        torch.onnx.export(
            model,
            (x_seq, pidx, x_ine, x_diarias),
            str(ONNX_LSTM),
            input_names=["x_seq", "provincia_idx", "x_ine", "x_diarias"],
            output_names=["logits_calor", "logits_frio"],
            dynamic_axes={
                "x_seq": {0: "batch"},
                "provincia_idx": {0: "batch"},
                "x_ine": {0: "batch"},
                "x_diarias": {0: "batch"},
                "logits_calor": {0: "batch"},
                "logits_frio": {0: "batch"},
            },
            opset_version=17,
        )
    print(f"    {ONNX_LSTM.relative_to(MODELS_DIR)}")
    return ONNX_LSTM


# ─────────────────────────────────────────────────────────────────────────────
# Exportación de artefactos JSON
# ─────────────────────────────────────────────────────────────────────────────
def _scaler_json(clase: str) -> dict:
    sc = joblib.load(ARTIFACTS_DIR / f"scaler_{clase}.joblib")
    return {"mean": sc.mean_.tolist(), "scale": sc.scale_.tolist()}


def _encoder_json(clase: str) -> dict:
    """Exporta los encoders por columna de forma que JS pueda reproducir
    process_input: LabelEncoder → clases ordenadas; OneHotEncoder → categorías.
    Hoy ambos son {} (no hay categóricas); se exporta la estructura real."""
    encoders = joblib.load(ARTIFACTS_DIR / f"encoders_{clase}.joblib") or {}
    out = {}
    for col, enc in encoders.items():
        if hasattr(enc, "classes_"):
            out[col] = {"type": "LabelEncoder", "classes": list(enc.classes_)}
        elif hasattr(enc, "categories_"):
            out[col] = {"type": "OneHotEncoder", "categories": [list(c) for c in enc.categories_]}
        else:
            out[col] = {"type": type(enc).__name__, "not_serializable": True}
    return out


def exportar_artefactos(force: bool = False) -> list[Path]:
    from climasafeai.features.external_features import (
        _EMBEDDED_DEMOGRAPHICS,
        DEMOGRAPHIC_FEATURES,
        N_FEATURES_PROVINCIA,
    )
    from climasafeai.models.lstm_province_hybrid import DAILY_FEATURE_COLS
    from climasafeai.data.sequences import FEATURE_COLS_SEQ
    from climasafeai.models.predict_model import (
        CLASS_THRESHOLDS_RECOMENDADOS,
        CLASS_THRESHOLDS_LSTM,
    )
    from climasafeai.models.ensemble import PERS_THRESHOLD_PELIGRO

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    escritos: list[Path] = []
    print("> Artefactos JSON → models/onnx/")

    # --- Features y scalers tabulares ---
    for clase in ("calor", "frio"):
        fn = joblib.load(ARTIFACTS_DIR / f"feature_names_{clase}.joblib")
        _write_json(OUTPUT_DIR / f"feature_names_{clase}.json", list(fn))
        _write_json(OUTPUT_DIR / f"scaler_{clase}.json", _scaler_json(clase))
        _write_json(OUTPUT_DIR / f"encoders_{clase}.json", _encoder_json(clase))
        # Umbrales por provincia (t1/t2 y métricas de calibración si existen)
        umb = joblib.load(ARTIFACTS_DIR / f"umbrales_provincia_{clase}.joblib")
        _write_json(OUTPUT_DIR / f"umbrales_provincia_{clase}.json", umb)
        # Split conformal: {alpha, qhat, n_classes} — exactamente el dict del joblib
        conf = joblib.load(ARTIFACTS_DIR / f"conformal_{clase}.joblib")
        _write_json(OUTPUT_DIR / f"conformal_{clase}.json", conf)
        escritos += [
            OUTPUT_DIR / f"feature_names_{clase}.json",
            OUTPUT_DIR / f"scaler_{clase}.json",
            OUTPUT_DIR / f"encoders_{clase}.json",
            OUTPUT_DIR / f"umbrales_provincia_{clase}.json",
            OUTPUT_DIR / f"conformal_{clase}.json",
        ]

    # --- Isotónica frío (post-hoc del RF; NO va dentro del ONNX) ---
    iso = joblib.load(ARTIFACTS_DIR / "iso_calib_frio.joblib")
    x_iso = getattr(iso, "X_thresholds_", None)
    if x_iso is None:
        x_iso = iso.f_.x
    y_iso = getattr(iso, "y_thresholds_", None)
    if y_iso is None:
        y_iso = iso.f_.y
    p_iso = OUTPUT_DIR / "iso_calib_frio.json"
    _write_json(p_iso, {
        "x": np.asarray(x_iso).tolist(),
        "y": np.asarray(y_iso).tolist(),
        "out_of_bounds": getattr(iso, "out_of_bounds", "clip"),
        "transform": "interpolar_lineal_con_clip",  # np.interp(clip(p, x0, xN), x, y)
    })
    escritos.append(p_iso)

    # --- Thresholds de clase ---
    p_ct = OUTPUT_DIR / "class_thresholds.json"
    _write_json(p_ct, {
        "CLASS_THRESHOLDS_RECOMENDADOS": CLASS_THRESHOLDS_RECOMENDADOS,
        "CLASS_THRESHOLDS_LSTM": CLASS_THRESHOLDS_LSTM,
        "PERS_THRESHOLD_PELIGRO": PERS_THRESHOLD_PELIGRO,
    })
    escritos.append(p_ct)

    # --- Provincia mapping (LSTM embedding). Verificación contra get_province_idx ---
    from climasafeai.data.weather_fetcher import get_province_idx
    mapping = json.load(open(ARTIFACTS_DIR / "provincia_mapping.json", encoding="utf-8"))
    orden_dinamico = {p: i for i, p in enumerate(sorted(_EMBEDDED_DEMOGRAPHICS.keys()))}
    if mapping != orden_dinamico:
        print("AVISO: provincia_mapping.json NO coincide con el orden de get_province_idx")
    else:
        print(f"    provincia_mapping.json verificado ({len(mapping)} provincias, orden alfabético OK)")
    p_pm = OUTPUT_DIR / "provincia_mapping.json"
    _write_json(p_pm, mapping)
    escritos.append(p_pm)

    # --- Features INE por provincia (los 4 valores que alimentan x_ine) ---
    # _EMBEDDED_DEMOGRAPHICS guarda la población TOTAL; get_ine_features aplica
    # np.log antes de escalar (DEMOGRAPHIC_FEATURES lo llama log_poblacion_total).
    provincias_ine = {
        nombre: {
            "pct_mayores_65": valores[0],
            "pct_mayores_80": valores[1],
            "pct_mujeres": valores[2],
            "poblacion_total": valores[3],
        }
        for nombre, valores in _EMBEDDED_DEMOGRAPHICS.items()
    }
    p_ine = OUTPUT_DIR / "ine_features.json"
    _write_json(p_ine, {
        "N_FEATURES_PROVINCIA": N_FEATURES_PROVINCIA,
        "demographic_features": DEMOGRAPHIC_FEATURES,
        # get_ine_features construye x_ine = [p65, p80, pmuj, log(poblacion_total)]
        "log_poblacion_aplicado_por_el_cliente": True,
        "provincias": provincias_ine,
    })
    escritos.append(p_ine)

    # --- Columnas LSTM + scalers LSTM ---
    p_dc = OUTPUT_DIR / "daily_feature_cols.json"
    _write_json(p_dc, {
        "daily_feature_cols": DAILY_FEATURE_COLS,
        "feature_cols_seq": FEATURE_COLS_SEQ,
        "seq_len": 24,
    })
    escritos.append(p_dc)
    for src, name in [
        ("scaler_diarias_lstm_hybrid", "scaler_diarias_lstm.json"),
        ("scaler_secuencias_lstm", "scaler_secuencias_lstm.json"),
        ("scaler_provincia_features", "scaler_provincia_features.json"),
    ]:
        sc = joblib.load(ARTIFACTS_DIR / f"{src}.joblib")
        p_sc = OUTPUT_DIR / name
        _write_json(p_sc, {"mean": sc.mean_.tolist(), "scale": sc.scale_.tolist()})
        escritos.append(p_sc)

    # --- Factores de riesgo (copia literal de data/factores_riesgo.json) ---
    from climasafeai.utils.paths import DATA_DIR
    factores = json.load(open(DATA_DIR / "factores_riesgo.json", encoding="utf-8"))
    p_fr = OUTPUT_DIR / "factores_riesgo.json"
    _write_json(p_fr, factores)
    escritos.append(p_fr)

    print(f"  {len(escritos)} artefactos JSON exportados")
    return escritos


# ─────────────────────────────────────────────────────────────────────────────
# Verificación: cada ONNX carga e infiere en onnxruntime CPU
# ─────────────────────────────────────────────────────────────────────────────
def verificar_inferencia() -> None:
    import onnxruntime as ort

    print("> Verificación onnxruntime CPU:")
    # XGB: X de ejemplo (27 features escaladas → ceros sirven para shape)
    sess = ort.InferenceSession(str(ONNX_XGB), providers=["CPUExecutionProvider"])
    out = sess.run(None, {"X": np.zeros((1, 27), dtype=np.float32)})
    print(f"    XGBoost_calor.onnx  → {[np.asarray(o).shape for o in out]}")

    sess = ort.InferenceSession(str(ONNX_RF), providers=["CPUExecutionProvider"])
    out = sess.run(None, {"X": np.zeros((1, 23), dtype=np.float32)})
    print(f"    RandomForest_frio.onnx → {[np.asarray(o).shape for o in out]}")

    sess = ort.InferenceSession(str(ONNX_LSTM), providers=["CPUExecutionProvider"])
    out = sess.run(None, {
        "x_seq": np.zeros((1, 24, 5), dtype=np.float32),
        "provincia_idx": np.array([0], dtype=np.int64),
        "x_ine": np.zeros((1, 4), dtype=np.float32),
        "x_diarias": np.zeros((1, 31), dtype=np.float32),
    })
    print(f"    LSTM_province_hybrid.onnx → {[np.asarray(o).shape for o in out]}")


# ─────────────────────────────────────────────────────────────────────────────
# Paridad: 5 escenarios, diff < 1e-3
# ─────────────────────────────────────────────────────────────────────────────
def _escenarios() -> list[dict]:
    return [
        {"nombre": "dia_templado", "t2m_c": 22.0, "rh": 55.0, "wind_speed_kmh": 10.0, "sp": 1013.0, "provincia": "Madrid"},
        {"nombre": "ola_calor_humeda", "t2m_c": 38.0, "rh": 62.0, "wind_speed_kmh": 8.0, "sp": 1008.0, "provincia": "Sevilla"},
        {"nombre": "ola_calor_seca", "t2m_c": 40.5, "rh": 22.0, "wind_speed_kmh": 16.0, "sp": 1005.0, "provincia": "Córdoba"},
        {"nombre": "dia_frio_humedo", "t2m_c": 1.0, "rh": 85.0, "wind_speed_kmh": 24.0, "sp": 1002.0, "provincia": "Zamora"},
        {"nombre": "helada_seca", "t2m_c": -4.0, "rh": 45.0, "wind_speed_kmh": 28.0, "sp": 1021.0, "provincia": "Ávila"},
    ]


def _fila_tabular(esc: dict, clase: str, rng: np.random.Generator) -> np.ndarray:
    """Fila sintética con TODOS los nombres de feature reales de la clase."""
    fn = joblib.load(ARTIFACTS_DIR / f"feature_names_{clase}.joblib")
    row = {c: 0.0 for c in fn}
    row.update({
        "t2m_c": esc["t2m_c"], "rh": esc["rh"],
        "wind_speed_kmh": esc["wind_speed_kmh"], "sp": esc["sp"],
    })
    # El resto de features (medias, roll, rezagos...) con valores plausibles
    # deterministas alrededor de la temperatura del escenario.
    for c in fn:
        if c in row and row[c] == 0.0:
            row[c] = esc["t2m_c"] + float(rng.uniform(-2.0, 2.0))
    from climasafeai.features.build_features import process_input
    return process_input(pd.DataFrame([row]), clase=clase)


def _entradas_lstm(esc: dict, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Secuencia 24h + diarias + INE sintéticos pero con nombres/shapes reales."""
    from climasafeai.features.weather_indices import heat_index, wbgt_from_heat_index, wind_chill
    from climasafeai.models.lstm_province_hybrid import DAILY_FEATURE_COLS
    from climasafeai.data.weather_fetcher import (
        get_province_idx, get_ine_features, escalar_para_lstm,
    )

    horas = np.arange(24)
    t2m = esc["t2m_c"] + 4.0 * np.sin(2 * np.pi * (horas - 8) / 24)
    rh = esc["rh"] + 8.0 * np.sin(2 * np.pi * (horas - 2) / 24)
    wind = esc["wind_speed_kmh"] + 2.0 * np.cos(2 * np.pi * horas / 24)
    seq = np.stack([
        t2m, rh, wind,
        heat_index(t2m, rh),
        wind_chill(t2m, wind),
    ], axis=1).astype(np.float32)  # (24, 5), orden FEATURE_COLS_SEQ

    base = {
        "t2m_c": esc["t2m_c"], "rh": esc["rh"],
        "wind_speed_kmh": esc["wind_speed_kmh"], "sp": esc["sp"],
        "heat_index_c": float(heat_index(esc["t2m_c"], esc["rh"])),
        "wbgt_c": float(wbgt_from_heat_index(heat_index(esc["t2m_c"], esc["rh"]))),
        "wind_chill_c": float(wind_chill(esc["t2m_c"], esc["wind_speed_kmh"])),
    }
    daily = np.array(
        [base.get(c, esc["t2m_c"] + float(rng.uniform(-1.5, 1.5))) for c in DAILY_FEATURE_COLS],
        dtype=np.float32,
    )

    ine = get_ine_features(esc["provincia"])
    pidx = np.array([get_province_idx(esc["provincia"])], dtype=np.int64)
    seq_s, ine_s, daily_s = escalar_para_lstm(seq, ine, daily)
    return seq_s, pidx, ine_s, daily_s


def _extraer_proba(outs: list) -> np.ndarray:
    """Saca la proba (1, 3) de la salida ONNX, tanto si es tensor como zipmap."""
    for o in outs:
        if isinstance(o, list) and len(o) == 1 and isinstance(o[0], dict):
            d = o[0]
            keys = sorted(int(k) for k in d.keys())
            return np.array([[float(d[k]) for k in keys]], dtype=np.float32)
        a = np.asarray(o)
        if a.ndim == 2 and a.shape[1] == 3:
            return a.astype(np.float32)
    raise RuntimeError(f"No se encontró salida de probabilidades en {outs!r}")


def paridad() -> bool:
    """Compara joblib/torch vs ONNX en 5 escenarios. Devuelve True si todo < 1e-3."""
    import torch
    import onnxruntime as ort
    from climasafeai.models.lstm_province_hybrid import load_lstm_province_hybrid

    xgb = joblib.load(MODELS_DIR / "XGBoost_calor.joblib")
    rf = joblib.load(MODELS_DIR / "RandomForest_frio.joblib")
    lstm = load_lstm_province_hybrid(device="cpu")

    sess_xgb = ort.InferenceSession(str(ONNX_XGB), providers=["CPUExecutionProvider"])
    sess_rf = ort.InferenceSession(str(ONNX_RF), providers=["CPUExecutionProvider"])
    sess_lstm = ort.InferenceSession(str(ONNX_LSTM), providers=["CPUExecutionProvider"])

    print("> Paridad ONNX vs joblib/torch (5 escenarios, diff < 1e-3):")
    print(f"    {'Escenario':<18}{'XGB calor':>12}{'RF frio':>12}{'LSTM calor':>12}{'LSTM frio':>12}")
    ok = True
    for i, esc in enumerate(_escenarios()):
        rng = np.random.default_rng(i)

        X_calor = _fila_tabular(esc, "calor", rng)
        X_frio = _fila_tabular(esc, "frio", rng)

        d_xgb = np.abs(_extraer_proba(sess_xgb.run(None, {"X": X_calor.astype(np.float32)}))
                       - xgb.predict_proba(X_calor)).max()
        d_rf = np.abs(_extraer_proba(sess_rf.run(None, {"X": X_frio.astype(np.float32)}))
                      - rf.predict_proba(X_frio)).max()

        seq_s, pidx, ine_s, daily_s = _entradas_lstm(esc, rng)
        feeds = {
            "x_seq": seq_s[None].astype(np.float32),
            "provincia_idx": pidx,
            "x_ine": ine_s.reshape(1, -1).astype(np.float32),
            "x_diarias": daily_s.reshape(1, -1).astype(np.float32),
        }
        oc_onnx, of_onnx = sess_lstm.run(None, feeds)
        with torch.no_grad():
            oc_t, of_t = lstm(
                torch.tensor(feeds["x_seq"]), torch.tensor(feeds["provincia_idx"]),
                torch.tensor(feeds["x_ine"]), torch.tensor(feeds["x_diarias"]),
            )
        d_lc = np.abs(oc_onnx - oc_t.numpy()).max()
        d_lf = np.abs(of_onnx - of_t.numpy()).max()
        # También la proba post-softmax (lo que verá el navegador)
        proba_onnx = torch.softmax(torch.tensor(oc_onnx), dim=1).numpy()
        proba_t = torch.softmax(oc_t, dim=1).numpy()
        d_lc_proba = np.abs(proba_onnx - proba_t).max()

        if max(d_xgb, d_rf, d_lc, d_lf) >= TOLERANCIA_PARIDAD:
            ok = False
        print(f"    {esc['nombre']:<18}{d_xgb:>12.2e}{d_rf:>12.2e}{d_lc:>12.2e}{d_lf:>12.2e}"
              + (f"   (proba calor {d_lc_proba:.2e})" if d_lc_proba > TOLERANCIA_PARIDAD else ""))

    print("  Paridad: " + ("OK (todo < 1e-3)" if ok else "FALLO (> 1e-3)"))
    return ok


# ─────────────────────────────────────────────────────────────────────────────
# main
# ─────────────────────────────────────────────────────────────────────────────
def main() -> int:
    parser = argparse.ArgumentParser(description="Exporta modelos a ONNX + artefactos JSON (WEB-011)")
    parser.add_argument("--force", action="store_true", help="Re-exportar aunque existan los .onnx")
    parser.add_argument("--check-only", action="store_true", help="No exportar; solo verificar y paridad")
    args = parser.parse_args()

    _check_deps()

    if not args.check_only:
        print(f"Salida: {OUTPUT_DIR.relative_to(MODELS_DIR)}/")
        exportar_xgboost(force=args.force)
        exportar_random_forest(force=args.force)
        exportar_lstm(force=args.force)
        exportar_artefactos(force=args.force)

    verificar_inferencia()
    if not paridad():
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
