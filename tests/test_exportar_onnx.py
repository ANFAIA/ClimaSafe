"""
test_exportar_onnx.py — WEB-011. Paridad ONNX vs joblib/torch.

El script scripts/exportar_onnx.py genera models/onnx/*.onnx + *.json a partir
de los joblib/JSON reales del repo. Estos tests verifican que:

  1. los 3 ONNX y los artefactos JSON existen;
  2. cada ONNX carga e infiere en onnxruntime CPU;
  3. la probabilidad del ONNX coincide con joblib/torch en 5 escenarios
     sintéticos con nombres/shapes reales (diff < 1e-3);
  4. los JSON salen de los joblib reales (nada inventado).

OJO: el conftest.py redirige MODELS_DIR/ARTIFACTS_DIR a tmp_path para aislar
el filesystem; estos tests necesitan los modelos REALES (models/*.joblib,
models/artifacts/*.joblib), así que un fixture autouse restaura las rutas.
"""
from __future__ import annotations

import importlib
import json
import subprocess
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

# Rutas reales capturadas ANTES de que el conftest las parchee.
from climasafeai.utils.paths import (
    MODELS_DIR as _REAL_MODELS_DIR,
    ARTIFACTS_DIR as _REAL_ARTIFACTS_DIR,
)

PROJECT_DIR = Path(__file__).resolve().parents[1]
SCRIPT = Path(PROJECT_DIR) / "scripts" / "exportar_onnx.py"
ONNX_DIR = _REAL_MODELS_DIR / "onnx"

TOL = 1e-3

_ONNX_ESPERADOS = [
    "XGBoost_calor.onnx",
    "RandomForest_frio.onnx",
    "LSTM_province_hybrid.onnx",
]

_JSON_ESPERADOS = [
    "feature_names_calor.json", "feature_names_frio.json",
    "scaler_calor.json", "scaler_frio.json",
    "encoders_calor.json", "encoders_frio.json",
    "umbrales_provincia_calor.json", "umbrales_provincia_frio.json",
    "conformal_calor.json", "conformal_frio.json",
    "iso_calib_frio.json",
    "class_thresholds.json",
    "provincia_mapping.json", "ine_features.json",
    "daily_feature_cols.json",
    "scaler_diarias_lstm.json", "scaler_secuencias_lstm.json",
    "scaler_provincia_features.json",
    "factores_riesgo.json",
]

ESCENARIOS = [
    {"nombre": "dia_templado", "t2m_c": 22.0, "rh": 55.0, "wind_speed_kmh": 10.0, "sp": 1013.0, "provincia": "Madrid"},
    {"nombre": "ola_calor_humeda", "t2m_c": 38.0, "rh": 62.0, "wind_speed_kmh": 8.0, "sp": 1008.0, "provincia": "Sevilla"},
    {"nombre": "ola_calor_seca", "t2m_c": 40.5, "rh": 22.0, "wind_speed_kmh": 16.0, "sp": 1005.0, "provincia": "Córdoba"},
    {"nombre": "dia_frio_humedo", "t2m_c": 1.0, "rh": 85.0, "wind_speed_kmh": 24.0, "sp": 1002.0, "provincia": "Zamora"},
    {"nombre": "helada_seca", "t2m_c": -4.0, "rh": 45.0, "wind_speed_kmh": 28.0, "sp": 1021.0, "provincia": "Ávila"},
]


@pytest.fixture(autouse=True)
def _rutas_reales(monkeypatch):
    """Deshace el parcheo de rutas del conftest: estos tests usan los modelos reales."""
    for mod_name in [
        "climasafeai.utils.paths",
        "climasafeai.models.predict_model",
        "climasafeai.features.build_features",
        "climasafeai.models.ensemble",
        "climasafeai.models.lstm_province_hybrid",
        "climasafeai.data.weather_fetcher",
        "climasafeai.features.external_features",
        "climasafeai.models.calibrate",
        "climasafeai.models.conformal",
    ]:
        try:
            mod = importlib.import_module(mod_name)
        except (ImportError, ModuleNotFoundError):
            continue
        for attr, val in [("MODELS_DIR", _REAL_MODELS_DIR), ("ARTIFACTS_DIR", _REAL_ARTIFACTS_DIR)]:
            if hasattr(mod, attr):
                monkeypatch.setattr(mod, attr, val)


@pytest.fixture(scope="session", autouse=True)
def _exportar_onnx():
    """Ejecuta el script una vez por sesión si falta algún artefacto."""
    faltan = [
        ONNX_DIR / n for n in _ONNX_ESPERADOS + _JSON_ESPERADOS
        if not (ONNX_DIR / n).exists()
    ]
    if faltan:
        res = subprocess.run(
            [sys.executable, str(SCRIPT)],
            capture_output=True, text=True, timeout=1800,
        )
        assert res.returncode == 0, f"exportar_onnx.py falló:\n{res.stdout}\n{res.stderr}"


# ─────────────────────────────────────────────────────────────────────────────
# Helpers de escenarios sintéticos (mismos nombres/shapes que los reales)
# ─────────────────────────────────────────────────────────────────────────────
def _X_tabular(esc: dict, clase: str, rng: np.random.Generator) -> np.ndarray:
    from climasafeai.features.build_features import process_input

    fn = joblib.load(_REAL_ARTIFACTS_DIR / f"feature_names_{clase}.joblib")
    row = {c: 0.0 for c in fn}
    row.update({
        "t2m_c": esc["t2m_c"], "rh": esc["rh"],
        "wind_speed_kmh": esc["wind_speed_kmh"], "sp": esc["sp"],
    })
    for c in fn:
        if row[c] == 0.0:
            row[c] = esc["t2m_c"] + float(rng.uniform(-2.0, 2.0))
    return process_input(pd.DataFrame([row]), clase=clase)


def _entradas_lstm(esc: dict, rng: np.random.Generator):
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
        t2m, rh, wind, heat_index(t2m, rh), wind_chill(t2m, wind),
    ], axis=1).astype(np.float32)

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


def _session_ort(path: str):
    import onnxruntime as ort
    return ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])


def _proba_de(outs) -> np.ndarray:
    """Probas (1, 3) de la salida ONNX, tanto si es tensor como zipmap."""
    for o in outs:
        if isinstance(o, list) and len(o) == 1 and isinstance(o[0], dict):
            d = o[0]
            keys = sorted(int(k) for k in d.keys())
            return np.array([[float(d[k]) for k in keys]], dtype=np.float32)
        a = np.asarray(o)
        if a.ndim == 2 and a.shape[1] == 3:
            return a.astype(np.float32)
    raise AssertionError(f"No se encontró salida de probabilidades: {outs!r}")


# ─────────────────────────────────────────────────────────────────────────────
# 1. Existen los artefactos
# ─────────────────────────────────────────────────────────────────────────────
def test_genera_los_tres_onnx():
    for name in _ONNX_ESPERADOS:
        f = ONNX_DIR / name
        assert f.exists(), f"falta {f}"
        assert f.stat().st_size > 1_000, f"{f} sospechosamente pequeño"


def test_genera_los_artefactos_json():
    for name in _JSON_ESPERADOS:
        f = ONNX_DIR / name
        assert f.exists(), f"falta {f}"
        json.load(open(f, encoding="utf-8"))  # debe ser JSON válido


# ─────────────────────────────────────────────────────────────────────────────
# 2. Paridad 5 escenarios: ONNX vs joblib
# ─────────────────────────────────────────────────────────────────────────────
def test_paridad_xgboost_calor_5_escenarios():
    xgb = joblib.load(_REAL_MODELS_DIR / "XGBoost_calor.joblib")
    sess = _session_ort(ONNX_DIR / "XGBoost_calor.onnx")
    for i, esc in enumerate(ESCENARIOS):
        X = _X_tabular(esc, "calor", np.random.default_rng(i))
        onnx_proba = _proba_de(sess.run(None, {"X": X.astype(np.float32)}))
        ref = xgb.predict_proba(X)
        diff = float(np.abs(onnx_proba - ref).max())
        assert diff < TOL, f"{esc['nombre']}: diff XGB {diff} >= {TOL}"


def test_paridad_random_forest_frio_5_escenarios():
    rf = joblib.load(_REAL_MODELS_DIR / "RandomForest_frio.joblib")
    sess = _session_ort(ONNX_DIR / "RandomForest_frio.onnx")
    for i, esc in enumerate(ESCENARIOS):
        X = _X_tabular(esc, "frio", np.random.default_rng(i))
        # sin zipmap → proba es el tensor (1, 3)
        onnx_proba = _proba_de(sess.run(None, {"X": X.astype(np.float32)}))
        ref = rf.predict_proba(X)  # proba CRUDA, antes de isotónica
        diff = float(np.abs(onnx_proba - ref).max())
        assert diff < TOL, f"{esc['nombre']}: diff RF {diff} >= {TOL}"


def test_paridad_lstm_5_escenarios():
    import torch
    from climasafeai.models.lstm_province_hybrid import load_lstm_province_hybrid

    lstm = load_lstm_province_hybrid(device="cpu")
    sess = _session_ort(ONNX_DIR / "LSTM_province_hybrid.onnx")
    for i, esc in enumerate(ESCENARIOS):
        seq_s, pidx, ine_s, daily_s = _entradas_lstm(esc, np.random.default_rng(i))
        feeds = {
            "x_seq": seq_s[None].astype(np.float32),
            "provincia_idx": pidx,
            "x_ine": ine_s.reshape(1, -1).astype(np.float32),
            "x_diarias": daily_s.reshape(1, -1).astype(np.float32),
        }
        oc_onnx, of_onnx = sess.run(None, feeds)
        with torch.no_grad():
            oc_t, of_t = lstm(
                torch.tensor(feeds["x_seq"]), torch.tensor(feeds["provincia_idx"]),
                torch.tensor(feeds["x_ine"]), torch.tensor(feeds["x_diarias"]),
            )
        assert float(np.abs(oc_onnx - oc_t.numpy()).max()) < TOL, f"{esc['nombre']}: diff logits calor"
        assert float(np.abs(of_onnx - of_t.numpy()).max()) < TOL, f"{esc['nombre']}: diff logits frio"
        # El navegador aplica softmax a los logits → proba debe coincidir también
        proba_onnx = torch.softmax(torch.tensor(oc_onnx), dim=1).numpy()
        proba_t = torch.softmax(oc_t, dim=1).numpy()
        assert float(np.abs(proba_onnx - proba_t).max()) < TOL, f"{esc['nombre']}: diff proba calor"


def test_lstm_onnx_devuelve_logits():
    """El ONNX del LSTM exporta LOGITS; el navegador aplica softmax (documentado)."""
    import onnx

    m = onnx.load(str(ONNX_DIR / "LSTM_province_hybrid.onnx"))
    salidas = {o.name for o in m.graph.output}
    assert salidas == {"logits_calor", "logits_frio"}, salidas


# ─────────────────────────────────────────────────────────────────────────────
# 3. Los JSON salen de los joblib reales (nada inventado)
# ─────────────────────────────────────────────────────────────────────────────
def test_json_coinciden_con_joblib():
    for clase in ("calor", "frio"):
        fn = joblib.load(_REAL_ARTIFACTS_DIR / f"feature_names_{clase}.joblib")
        assert json.load(open(ONNX_DIR / f"feature_names_{clase}.json")) == list(fn)

        sc = joblib.load(_REAL_ARTIFACTS_DIR / f"scaler_{clase}.joblib")
        js = json.load(open(ONNX_DIR / f"scaler_{clase}.json"))
        assert np.allclose(js["mean"], sc.mean_)
        assert np.allclose(js["scale"], sc.scale_)

        umb = joblib.load(_REAL_ARTIFACTS_DIR / f"umbrales_provincia_{clase}.joblib")
        ju = json.load(open(ONNX_DIR / f"umbrales_provincia_{clase}.json"))
        assert set(ju) == set(umb)

        conf = joblib.load(_REAL_ARTIFACTS_DIR / f"conformal_{clase}.joblib")
        jc = json.load(open(ONNX_DIR / f"conformal_{clase}.json"))
        assert jc["alpha"] == conf["alpha"]
        assert abs(jc["qhat"] - float(conf["qhat"])) < 1e-6
        assert jc["n_classes"] == conf["n_classes"]


def test_json_isotonic_frio_es_la_joblib():
    iso = joblib.load(_REAL_ARTIFACTS_DIR / "iso_calib_frio.joblib")
    x = getattr(iso, "X_thresholds_", iso.f_.x)
    y = getattr(iso, "y_thresholds_", iso.f_.y)
    ji = json.load(open(ONNX_DIR / "iso_calib_frio.json"))
    assert np.allclose(ji["x"], np.asarray(x))
    assert np.allclose(ji["y"], np.asarray(y))
    assert ji["out_of_bounds"] == iso.out_of_bounds
    # La transformación que aplicará el navegador reproduce sklearn
    p = np.array([0.05, 0.3, 0.6, 0.9])
    manual = np.interp(np.clip(p, x[0], x[-1]), np.asarray(x, dtype=float), np.asarray(y, dtype=float))
    assert np.allclose(manual, iso.transform(p))


def test_json_class_thresholds_y_provincia_mapping():
    from climasafeai.models.predict_model import (
        CLASS_THRESHOLDS_RECOMENDADOS,
        CLASS_THRESHOLDS_LSTM,
    )
    from climasafeai.models.ensemble import PERS_THRESHOLD_PELIGRO
    from climasafeai.features.external_features import _EMBEDDED_DEMOGRAPHICS
    from climasafeai.data.weather_fetcher import get_province_idx

    jt = json.load(open(ONNX_DIR / "class_thresholds.json"))
    assert jt["CLASS_THRESHOLDS_RECOMENDADOS"] == CLASS_THRESHOLDS_RECOMENDADOS
    assert jt["CLASS_THRESHOLDS_LSTM"] == CLASS_THRESHOLDS_LSTM
    assert jt["PERS_THRESHOLD_PELIGRO"] == PERS_THRESHOLD_PELIGRO

    jp = json.load(open(ONNX_DIR / "provincia_mapping.json"))
    orden = {p: get_province_idx(p) for p in _EMBEDDED_DEMOGRAPHICS}
    assert jp == orden, "provincia_mapping.json no coincide con get_province_idx"

    ji = json.load(open(ONNX_DIR / "ine_features.json"))
    assert ji["N_FEATURES_PROVINCIA"] == len(_EMBEDDED_DEMOGRAPHICS["Madrid"])
    assert set(ji["provincias"]) == set(_EMBEDDED_DEMOGRAPHICS)


def test_json_factores_riesgo_es_copia():
    from climasafeai.utils.paths import DATA_DIR

    original = json.load(open(DATA_DIR / "factores_riesgo.json", encoding="utf-8"))
    copia = json.load(open(ONNX_DIR / "factores_riesgo.json", encoding="utf-8"))
    assert copia == original
