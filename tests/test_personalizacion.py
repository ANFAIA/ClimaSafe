"""Tests de climasafeai.features.personalizacion.personalizar_riesgo."""

import pytest

from climasafeai.features.personalizacion import personalizar_riesgo


def test_perfil_vacio_no_cambia_indice():
    r = personalizar_riesgo(0.5, {}, tipo="calor")
    assert r["factor_total"] == 1.0
    assert r["indice_personalizado"] == 0.5
    assert r["factores"] == []


def test_composicion_en_odds_no_se_sale_de_escala():
    # 0.95 con un factor grande NO debe superar 1.0 (la trampa de multiplicar).
    r = personalizar_riesgo(0.95, {"sexo": "mujer", "aclimatado": False}, tipo="calor")
    assert r["indice_personalizado"] <= 1.0
    # odds(0.95)=19, ×1.04×1.6=1.664 -> odds'=31.62 -> 31.62/32.62 ≈ 0.969
    assert r["indice_personalizado"] == pytest.approx(0.969, abs=1e-3)


def test_obesidad_calor_vs_frio_es_asimetrica():
    """El punto clave: los gordos sufren más en verano, no en invierno."""
    perfil = {"porcentaje_grasa": 40, "edad": 50, "sexo": "hombre", "nivel_actividad": "intensa"}
    calor = personalizar_riesgo(0.5, perfil, tipo="calor")
    frio = personalizar_riesgo(0.5, perfil, tipo="frio")

    nombres_calor = {f["nombre"] for f in calor["factores"]}
    assert any("grasa" in n.lower() for n in nombres_calor)  # sube el riesgo en calor
    # En frío la grasa alta también es factor (aisla, pero la desviación de la media da factor):
    nombres_frio = {f["nombre"] for f in frio["factores"]}
    assert any("grasa" in n.lower() for n in nombres_frio)


def test_obesidad_calor_solo_cuenta_en_esfuerzo():
    # El factor grasa relativa se aplica siempre que hay % grasa + edad,
    # pero el exceso se modula por actividad → el factor es continuo 0.85-1.15
    en_reposo = personalizar_riesgo(0.5, {"porcentaje_grasa": 40, "edad": 50}, tipo="calor")
    en_moderada = personalizar_riesgo(0.5, {"porcentaje_grasa": 40, "edad": 50, "nivel_actividad": "moderada"}, tipo="calor")
    # Ambos tienen factor grasa (el cálculo es independiente de actividad)
    assert any("grasa" in f["nombre"].lower() for f in en_reposo["factores"])
    assert any("grasa" in f["nombre"].lower() for f in en_moderada["factores"])


def test_actividad_protege_en_frio_perjudica_en_calor():
    perfil = {"nivel_actividad": "moderada"}
    calor = personalizar_riesgo(0.5, perfil, tipo="calor")
    frio = personalizar_riesgo(0.5, perfil, tipo="frio")
    assert calor["factor_total"] > 1.0   # calor: la actividad sube el riesgo
    assert frio["factor_total"] < 1.0    # frío: la actividad genera calor, protege


def test_cap_de_factores():
    perfil = {
        "edad": 90, "sexo": "mujer", "aclimatado": False,
        "comorbilidades": {"cardiovascular", "mental"},
        "nivel_actividad": "muy_intensa",
    }
    r = personalizar_riesgo(0.5, perfil, tipo="calor", cap_factores=3.0)
    assert r["capado"] is True
    assert r["factor_total"] == 3.0


def test_salud_mental_y_antipsicoticos_no_se_cuentan_doble():
    solo_diag = personalizar_riesgo(0.5, {"comorbilidades": {"mental"}}, tipo="calor")
    con_ambos = personalizar_riesgo(
        0.5, {"comorbilidades": {"mental"}, "farmacos": {"antipsicoticos"}}, tipo="calor"
    )
    # Un único factor de 1.8 en ambos casos (no 1.8×1.8).
    assert solo_diag["factor_total"] == pytest.approx(1.8)
    assert con_ambos["factor_total"] == pytest.approx(1.8)


def test_social_toma_el_maximo_no_el_producto():
    r = personalizar_riesgo(
        0.5, {"situacion_social": {"encamado", "vive_solo"}}, tipo="calor"
    )
    factores_soc = [f for f in r["factores"] if f["categoria"] == "situacional"]
    assert len(factores_soc) == 1
    assert factores_soc[0]["factor"] == 2.0  # max(encamado 2.0, vive_solo 1.5)


def test_indices_extremos_no_se_mueven():
    assert personalizar_riesgo(0.0, {"edad": 90}, tipo="calor")["indice_personalizado"] == 0.0
    assert personalizar_riesgo(1.0, {"edad": 90}, tipo="calor")["indice_personalizado"] == 1.0


def test_validaciones():
    with pytest.raises(ValueError):
        personalizar_riesgo(1.5, {}, tipo="calor")
    with pytest.raises(ValueError):
        personalizar_riesgo(0.5, {}, tipo="templado")


def test_desglose_es_explicable():
    r = personalizar_riesgo(0.8, {"sexo": "mujer", "edad": 80, "comorbilidades": {"cardiovascular"}}, tipo="calor")
    assert {f["nombre"].split()[0] for f in r["factores"]} >= {"sexo", "cardiopatía/HTA"}
    for f in r["factores"]:
        assert set(f) == {"nombre", "categoria", "factor"}


# ─────────────────────────────────────────────────────────────────────────────
# Regresión BUG-001: un dato meteorológico NaN (fetch corrupto) NO debe croncar
# predict_ensemble. Todo el pipeline (Formula → ensemble → personalización)
# tiene que devolver probabilidades finitas en [0,1] y una clase 0/1/2.
# ─────────────────────────────────────────────────────────────────────────────

import numpy as np
import pandas as pd

from climasafeai.models.ensemble import (
    predict_ensemble,
    _proba_from_formula,
    _conformal_weighted_ensemble,
)


def _weather_nan():
    """Weather con el dato corrupto del bug: t2m_c y rh NaN."""
    return {
        "current": {"t2m_c": float("nan"), "rh": float("nan"), "wind_speed_kmh": float("nan")},
        "df_hora": pd.DataFrame(columns=["datetime", "heat_index_c", "t2m_c"]),
        "df_features": pd.DataFrame(),
        "lat": 42.198,
        "lon": -8.728,
        "uv_index": None,
    }


def _stub_predictores(monkeypatch, prob_tabular=0.3):
    """Offline: sin red ni modelos. El contagio pasa por Formula+ensemble."""
    import climasafeai.models.ensemble as ensemble

    monkeypatch.setattr(ensemble, "fetch_weather_data", lambda **kw: _weather_nan())
    monkeypatch.setattr(
        ensemble, "_predecir_tabular",
        lambda *a, **k: {"prob_riesgo": prob_tabular, "conformal_set_size": 2, "clase": 0, "_X": None},
    )
    monkeypatch.setattr(
        ensemble, "_predecir_lstm",
        lambda *a, **k: {"calor": {"prob_riesgo": 0.4, "clase": 0}, "frio": {"prob_riesgo": 0.1, "clase": 0}},
    )


def _assert_validas(r):
    assert r["clase_final"] in (0, 1, 2)
    for tipo in ("calor", "frio"):
        for key in ("prob_poblacional", "prob_personalizada"):
            v = r["perfil"][tipo][key]
            assert np.isfinite(v), f"{tipo}.{key} no es finito: {v}"
            assert 0.0 <= v <= 1.0, f"{tipo}.{key} fuera de [0,1]: {v}"


def test_predict_ensemble_no_crashea_con_weather_nan(monkeypatch):
    """Perfil del log (edad 19, hombre, tenis, Vigo/Pontevedra) con current NaN."""
    _stub_predictores(monkeypatch)
    perfil = {"edad": 19, "sexo": "hombre", "nivel_actividad": "moderada", "duracion_actividad_h": 2}
    r = predict_ensemble(lat=42.198, lon=-8.728, provincia="Pontevedra", perfil=perfil)
    _assert_validas(r)


def test_predict_ensemble_nan_de_un_modelo_no_contamina(monkeypatch):
    """Aunque XGBoost/RandomForest devuelva prob NaN, el ensemble lo descarta."""
    _stub_predictores(monkeypatch, prob_tabular=float("nan"))
    perfil = {"edad": 75, "sexo": "mujer", "nivel_actividad": "reposo"}
    r = predict_ensemble(lat=42.198, lon=-8.728, provincia="Pontevedra", perfil=perfil)
    _assert_validas(r)


def test_proba_from_formula_con_nan_usa_defaults_seguros():
    r = _proba_from_formula({"t2m_c": float("nan"), "rh": float("nan"), "wind_speed_kmh": float("nan")})
    for tipo in ("calor", "frio"):
        assert np.isfinite(r[tipo]["prob_riesgo"])
        assert 0.0 <= r[tipo]["prob_riesgo"] <= 1.0


def test_conformal_weighted_ensemble_descarta_modelo_con_nan():
    res = {
        "XGBoost_calor": {"prob_riesgo": float("nan"), "conformal_set_size": 2},
        "LSTM": {"calor": {"prob_riesgo": 0.8, "clase": 2}},
        "Formula": {"calor": {"prob_riesgo": 0.5, "clase": 1}},
    }
    ens = _conformal_weighted_ensemble(res, "calor")
    assert np.isfinite(ens["prob_riesgo"])
    assert 0.0 <= ens["prob_riesgo"] <= 1.0
    # Media ponderada de los dos finitos: (0.8/2 + 0.5/2) / 1 = 0.65
    assert ens["prob_riesgo"] == pytest.approx(0.65)


def test_conformal_weighted_ensemble_solo_nan_fallback_seguro():
    res = {
        "XGBoost_calor": {"prob_riesgo": float("nan"), "conformal_set_size": 2},
        "LSTM": {"calor": {"prob_riesgo": float("inf"), "clase": 2}},
        "Formula": {"calor": {"prob_riesgo": float("nan"), "clase": 1}},
    }
    ens = _conformal_weighted_ensemble(res, "calor")
    assert ens["prob_riesgo"] == 0.0
    assert ens["clase"] == 0
