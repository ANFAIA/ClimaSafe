"""Paridad WEB-012: pipeline JS (web/probar-ya, onnxruntime-web en node) vs
predict_ensemble Python (joblib/torch) sobre 5 escenarios precargados.

El test corre SIN red: los datos meteorológicos viven en
web/probar-ya/scenarios.json (mismo fichero que usa la demo como fallback).
Cada lado construye df_hora/df_features con la MISMA fuente y compara:
  - clase_final: idéntica
  - % de riesgo (prob_pers = max(prob_personalizada calor/frío)): ±1 punto

Depende de node >= 18 y del paquete npm local onnxruntime-web
(web/probar-ya/test/package.json); si falta node_modules se instala con npm.
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
from datetime import date
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
WEB = ROOT / "web" / "probar-ya"
ESCENARIOS_PATH = WEB / "scenarios.json"
TEST_DIR = WEB / "test"
NODE = shutil.which("node")

pytestmark = pytest.mark.skipif(
    NODE is None,
    reason="node no está en PATH: el test de paridad JS necesita node >= 18",
)

TOLERANCIA_PUNTOS = 0.01  # ±1 punto


# ─────────────────────────────────────────────────────────────────────────────
# Lado Python
# ─────────────────────────────────────────────────────────────────────────────
def _df_hora_desde_escenario(esc: dict) -> pd.DataFrame:
    filas = []
    for r in esc["horas"]:
        filas.append({
            "datetime": pd.to_datetime(r["datetime"]),
            "t2m_c": r["t2m_c"],
            "rh": r["rh"],
            "wind_speed_kmh": r["wind_speed_kmh"],
            "sp": r["sp"],
        })
    return pd.DataFrame(filas)


def _weather_python(esc: dict) -> dict:
    from climasafeai.data.weather_fetcher import _generar_features_completas

    df_hora = _df_hora_desde_escenario(esc)
    target = esc["target_date"]
    mask = pd.to_datetime(df_hora["datetime"]).dt.date.astype(str) == target
    df_target = df_hora[mask].copy()
    df_hist = df_hora[~mask].copy()
    df_features, df_hora_proc = _generar_features_completas(df_target, df_hist)
    return {
        "lat": esc["lat"],
        "lon": esc["lon"],
        "current": dict(esc["current"]),
        "df_hora": df_hora_proc,
        "df_features": df_features,
        "uv_index": None,
        "target_date": target,
    }


def _perfil_python(perfil: dict) -> dict:
    p = dict(perfil)
    p["comorbilidades"] = set(p.get("comorbilidades") or [])
    p["farmacos"] = set(p.get("farmacos") or [])
    p["situacion_social"] = set(p.get("situacion_social") or [])
    return p


def _salida_python(esc: dict) -> dict:
    from climasafeai.models.ensemble import predict_ensemble

    weather = _weather_python(esc)
    res = predict_ensemble(
        weather=weather,
        provincia=esc["provincia"],
        perfil=_perfil_python(esc["perfil"]),
        target_date=date.fromisoformat(esc["target_date"]),
    )
    prob_pers = max(
        res["perfil"]["calor"]["prob_personalizada"],
        res["perfil"]["frio"]["prob_personalizada"],
    )
    return {
        "nombre": esc["nombre"],
        "clase_final": int(res["clase_final"]),
        "prob_pers": float(prob_pers),
        "perfil": {
            canal: {
                "prob_poblacional": float(res["perfil"][canal]["prob_poblacional"]),
                "prob_personalizada": float(res["perfil"][canal]["prob_personalizada"]),
                "factor_total": float(res["perfil"][canal]["factor_total"]),
            }
            for canal in ("calor", "frio")
        },
        "override_fisico": res.get("override_fisico"),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Lado JS (node)
# ─────────────────────────────────────────────────────────────────────────────
def _asegurar_node_modules() -> None:
    if (TEST_DIR / "node_modules" / "onnxruntime-web").exists():
        return
    subprocess.run(
        ["npm", "install", "--no-audit", "--no-fund"],
        cwd=TEST_DIR,
        check=True,
        capture_output=True,
    )


def _salidas_js(tmp_path: Path, salidas_python: list[dict]) -> list[dict]:
    python_out = tmp_path / "python_out.json"
    python_out.write_text(json.dumps(salidas_python, ensure_ascii=False), encoding="utf-8")
    js_out = tmp_path / "js_out.json"
    cmd = [
        NODE, "paridad.mjs",
        str(ESCENARIOS_PATH), str(python_out), str(js_out),
    ]
    proc = subprocess.run(cmd, cwd=TEST_DIR, capture_output=True, text=True)
    assert proc.returncode == 0, (
        f"node paridad.mjs falló (rc={proc.returncode})\n"
        f"stdout: {proc.stdout}\nstderr: {proc.stderr}"
    )
    return json.loads(js_out.read_text(encoding="utf-8"))


# ─────────────────────────────────────────────────────────────────────────────
# Test
# ─────────────────────────────────────────────────────────────────────────────
def test_paridad_web_012(tmp_path, monkeypatch):
    # El conftest autouse (patch_paths) redirige MODELS_DIR/ARTIFACTS_DIR a
    # tmp_path para aislar el filesystem; este test necesita los modelos y
    # artefactos REALES (predict_ensemble carga XGBoost_calor.joblib, etc.).
    import climasafeai.utils.paths as paths_mod

    raiz = Path(paths_mod.__file__).resolve().parents[2]
    real = {"MODELS_DIR": raiz / "models", "ARTIFACTS_DIR": raiz / "models" / "artifacts"}
    monkeypatch.setattr(paths_mod, "MODELS_DIR", real["MODELS_DIR"])
    monkeypatch.setattr(paths_mod, "ARTIFACTS_DIR", real["ARTIFACTS_DIR"])
    # El conftest también parcheó build_features (importado por el autouse);
    # process_input necesita el scaler/encoders REALES.
    import importlib

    for nombre_mod in ("climasafeai.features.build_features",):
        mod = importlib.import_module(nombre_mod)
        for attr, val in real.items():
            if hasattr(mod, attr):
                monkeypatch.setattr(mod, attr, val)

    _asegurar_node_modules()
    data = json.loads(ESCENARIOS_PATH.read_text(encoding="utf-8"))
    escenarios = data["escenarios"]

    salidas_python = [_salida_python(esc) for esc in escenarios]
    salidas_js = _salidas_js(tmp_path, salidas_python)

    assert len(salidas_js) == len(escenarios), (
        f"Se esperaban {len(escenarios)} salidas JS, llegaron {len(salidas_js)}"
    )

    fallos = []
    for py, js in zip(salidas_python, salidas_js):
        assert py["nombre"] == js["nombre"]
        d_clase = abs(py["clase_final"] - js["clase_final"])
        d_prob = abs(py["prob_pers"] - js["prob_pers"])
        estado = "OK" if d_clase == 0 and d_prob <= TOLERANCIA_PUNTOS else "FALLO"
        print(
            f"  {py['nombre']:<16} clase py={py['clase_final']} js={js['clase_final']} "
            f"| % py={py['prob_pers']:.4f} js={js['prob_pers']:.4f} "
            f"(Δ={d_prob:+.4f}) {estado}"
        )
        if estado == "FALLO":
            fallos.append((py["nombre"], py, js))

    if fallos:
        detalle = "\n".join(
            f"  {nombre}:\n    python={py}\n    js={js}" for nombre, py, js in fallos
        )
        pytest.fail(f"Paridad JS vs Python falla en {len(fallos)} escenarios:\n{detalle}")
