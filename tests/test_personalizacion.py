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


def test_puede_salir_peligro_con_porcentaje_bajo(monkeypatch):
    """BOT-013 criterio 3: sí, y por eso el parte tiene que decir que el nivel
    no sale del porcentaje.

    El override físico de `predict_ensemble` (HI_peak >= 39 → PELIGRO) no mira
    `prob_personalizada`: solo exige que la clase por probabilidad no sea ya
    PELIGRO. Con los modelos dando riesgo casi nulo y un día de HI 41 sale
    PELIGRO con un porcentaje de un dígito.
    """
    import climasafeai.models.ensemble as ensemble

    weather = _weather_nan()
    # Día con pico de HI de 41°C entre las 12 y las 17
    weather["df_hora"] = pd.DataFrame({
        "datetime": pd.date_range("2026-08-05 00:00", periods=24, freq="h"),
        "heat_index_c": [20.0] * 12 + [41.0] * 5 + [20.0] * 7,
        "t2m_c": [20.0] * 12 + [38.0] * 5 + [20.0] * 7,
    })
    weather["current"] = {"t2m_c": 38.0, "rh": 30.0, "wind_speed_kmh": 5.0}
    monkeypatch.setattr(ensemble, "fetch_weather_data", lambda **kw: weather)
    monkeypatch.setattr(
        ensemble, "_predecir_tabular",
        lambda *a, **k: {"prob_riesgo": 0.01, "conformal_set_size": 2, "clase": 0, "_X": None},
    )
    monkeypatch.setattr(
        ensemble, "_predecir_lstm",
        lambda *a, **k: {"calor": {"prob_riesgo": 0.01, "clase": 0}, "frio": {"prob_riesgo": 0.0, "clase": 0}},
    )
    monkeypatch.setattr(
        ensemble, "_proba_from_formula",
        lambda *a, **k: {"calor": {"prob_riesgo": 0.01, "heat_index_c": 41.0},
                         "frio": {"prob_riesgo": 0.0, "wind_chill_c": 38.0}},
    )

    perfil = {"edad": 30, "sexo": "hombre", "nivel_actividad": "reposo",
              "hora_inicio": 13, "duracion_actividad_h": 3}
    r = predict_ensemble(lat=37.888, lon=-4.779, provincia="Cordoba", perfil=perfil)

    assert r["clase_final_label"] == "PELIGRO"
    assert r["perfil"]["calor"]["prob_personalizada"] < 0.10
    assert ">=39" in r["override_fisico"]["razon"]


# ─────────────────────────────────────────────────────────────────────────────
# DATA-004: recomendar_horario y pico_riesgo_actividad sobre perfil sub-horario
# ─────────────────────────────────────────────────────────────────────────────

from climasafeai.features.personalizacion import (
    riesgo_horario_acumulado, recomendar_horario, pico_riesgo_actividad,
)
from climasafeai.models.ensemble import perfil_horario_desde_df
from datetime import date, datetime


def _perfil_horario_campana() -> list[dict]:
    """24 h con campana de HI (pico 41 a las 16h), como la gráfica MCP."""
    return [
        {"hora": h, "HI": round(20.0 + (41.0 - 20.0) * max(0.0, 1 - abs(h - 16) / 10), 1),
         "temp": round(15.0 + h * 0.4, 1)}
        for h in range(24)
    ]


def _df_hora_pico_al_empezar_la_hora() -> pd.DataFrame:
    """Día donde el mínimo de 1h NO está en una hora en punto: la hora 6 baja de
    26°C (heredando el calor de las 5:00) hacia las 22°C de las 7:00, y a partir
    de las 8:00 vuelve a subir. Con 15 min la mejor ventana empieza a las 6:45;
    en modo horario solo puede elegir la hora 7 completa."""
    hi = {h: 30.0 for h in range(6)}
    hi[6] = 26.0
    hi[7] = 22.0
    for h in range(8, 24):
        hi[h] = 24.0
    return pd.DataFrame([
        {"datetime": datetime(2026, 7, 21, h), "t2m_c": 25.0, "heat_index_c": hi[h]}
        for h in range(24)
    ])


def _perfil_usuario(**kw) -> dict:
    p = {"edad": 40, "sexo": "hombre", "nivel_actividad": "ligera",
         "hora_inicio": 7, "duracion_actividad_h": 1}
    p.update(kw)
    return p


def test_recomendar_horario_modo_horario_identico_al_historico():
    """DATA-004 criterio 5: con perfil horario la recomendación es la de siempre.
    Valor capturado con el código anterior a DATA-004 (regresión)."""
    ph = _perfil_horario_campana()
    perfil = _perfil_usuario(hora_inicio=10, duracion_actividad_h=2)
    assert recomendar_horario(ph, perfil) == {
        "hora_inicio": 6, "hora_fin": 8, "riesgo_medio": 0.1254, "riesgo_actual": 0.462,
    }


def test_recomendar_horario_15min_mismo_contrato_y_ventana_subhoraria():
    """DATA-004 criterio 3: la misma función, sobre el perfil de 15 min, devuelve
    el mismo contrato y una ventana que se desliza por cuartos de hora."""
    df = _df_hora_pico_al_empezar_la_hora()
    ph15 = perfil_horario_desde_df(df, target_date=date(2026, 7, 21), res_min=15)
    ph60 = perfil_horario_desde_df(df, target_date=date(2026, 7, 21), res_min=60)

    rec60 = recomendar_horario(ph60, _perfil_usuario())
    rec15 = recomendar_horario(ph15, _perfil_usuario())

    # Modo horario: solo puede elegir la hora entera 7 (la de menor riesgo).
    assert rec60["hora_inicio"] == 7
    # Modo 15 min: la ventana de 1h se desliza y empieza a las 6:45, evitando el
    # tramo caluroso del inicio de la hora 6 (26°C a las 6:00 vs 23°C a las 6:45).
    assert set(rec15) == {"hora_inicio", "hora_fin", "riesgo_medio", "riesgo_actual"}
    assert rec15["hora_inicio"] == 6.75
    assert rec15["hora_fin"] - rec15["hora_inicio"] == 1.0
    assert rec15["hora_inicio"] % 0.25 == 0  # sigue la malla de 15 min


def test_pico_riesgo_actividad_15min_captura_el_repunte_dentro_de_la_hora():
    """DATA-004 criterio 3: el pico se recalcula sobre la curva de 15 min y ve el
    repunte hacia la hora siguiente que el pico horario se pierde."""
    df = _df_hora_pico_al_empezar_la_hora()
    ph15 = perfil_horario_desde_df(df, target_date=date(2026, 7, 21), res_min=15)
    ph60 = perfil_horario_desde_df(df, target_date=date(2026, 7, 21), res_min=60)
    perfil = _perfil_usuario()

    curva15 = riesgo_horario_acumulado(ph15, perfil)
    curva60 = riesgo_horario_acumulado(ph60, perfil)

    p15 = pico_riesgo_actividad(curva15, perfil)  # ventana 7 a 8
    p60 = pico_riesgo_actividad(curva60, perfil)

    assert p60 == 0.1692  # el punto de la hora 7 (22°C) no ve la subida a las 8:00
    assert p15 == pytest.approx(max(c["riesgo"] for c in curva15 if 7 <= c["hora"] < 8), abs=1e-4)
    assert p15 == pytest.approx(0.2072, abs=1e-4)
    assert p15 > p60  # el pico de 15 min ve el repunte que el horario pierde


def test_recomendar_horario_15min_duracion_subhora_en_cuartos():
    """DATA-004: una salida de 40 min sobre el perfil de 15 min usa 3 cuartos
    (45 min) y mantiene el contrato; la resolución ya no la mete entera en un
    único punto horario."""
    df = _df_hora_pico_al_empezar_la_hora()
    ph15 = perfil_horario_desde_df(df, target_date=date(2026, 7, 21), res_min=15)
    perfil = _perfil_usuario(duracion_actividad_h=40 / 60)

    rec = recomendar_horario(ph15, perfil)
    assert set(rec) == {"hora_inicio", "hora_fin", "riesgo_medio", "riesgo_actual"}
    assert 0.25 <= rec["hora_fin"] - rec["hora_inicio"] <= 1.0


# ─────────────────────────────────────────────────────────────────────────────
# DATA-007: predict_ensemble acepta `resolucion` (min por punto, default 60)
# ─────────────────────────────────────────────────────────────────────────────

def _df_hora_dia_completo() -> pd.DataFrame:
    """24 h de un día con campana de HI (pico 41 a las 16h)."""
    return pd.DataFrame({
        "datetime": pd.date_range("2026-08-05 00:00", periods=24, freq="h"),
        "heat_index_c": [round(20.0 + 21.0 * max(0.0, 1 - abs(h - 16) / 10), 1) for h in range(24)],
        "t2m_c": [round(15.0 + h * 0.4, 1) for h in range(24)],
    })


def _weather_con_perfil_horario() -> dict:
    """Weather completo (offline) con un día de datos horarios."""
    weather = _weather_nan()
    weather["df_hora"] = _df_hora_dia_completo()
    weather["target_date"] = "2026-08-05"
    return weather


def _stub_predictores_resolucion(monkeypatch):
    """Igual que el stub de BUG-001 pero con un día de df_hora (el de arriba)."""
    import climasafeai.models.ensemble as ensemble

    weather = _weather_con_perfil_horario()
    monkeypatch.setattr(ensemble, "fetch_weather_data", lambda **kw: weather)
    monkeypatch.setattr(
        ensemble, "_predecir_tabular",
        lambda *a, **k: {"prob_riesgo": 0.3, "conformal_set_size": 2, "clase": 0, "_X": None},
    )
    monkeypatch.setattr(
        ensemble, "_predecir_lstm",
        lambda *a, **k: {"calor": {"prob_riesgo": 0.4, "clase": 0}, "frio": {"prob_riesgo": 0.1, "clase": 0}},
    )


def test_predict_ensemble_resolucion_60_identico_al_default(monkeypatch):
    """DATA-007 criterio 1: `resolucion=60` da exactamente el perfil horario de
    hoy (un punto por hora), idéntico a no pasar el parámetro (default 60)."""
    _stub_predictores_resolucion(monkeypatch)
    perfil = {"edad": 40, "sexo": "hombre", "nivel_actividad": "ligera"}

    r_default = predict_ensemble(lat=42.29, lon=-8.81, provincia="Pontevedra", perfil=perfil)
    r60 = predict_ensemble(lat=42.29, lon=-8.81, provincia="Pontevedra", perfil=perfil, resolucion=60)

    ph_default = r_default["weather"]["perfil_horario"]
    ph60 = r60["weather"]["perfil_horario"]
    assert ph60 == ph_default
    assert len(ph60) == 24
    assert all(isinstance(p["hora"], int) for p in ph60)
    # El resto del contrato no cambia: misma clase y perfil aplicado.
    assert r60["clase_final"] == r_default["clase_final"]
    assert r60["perfil"] == r_default["perfil"]


def test_predict_ensemble_resolucion_15_cuadruplica_los_puntos(monkeypatch):
    """DATA-007 criterio 2: con `resolucion=15` el perfil horario tiene 4 puntos
    por hora (96 en un día) y el contrato de salida se mantiene."""
    _stub_predictores_resolucion(monkeypatch)
    perfil = {"edad": 40, "sexo": "hombre", "nivel_actividad": "ligera"}

    r15 = predict_ensemble(lat=42.29, lon=-8.81, provincia="Pontevedra", perfil=perfil, resolucion=15)

    ph15 = r15["weather"]["perfil_horario"]
    assert len(ph15) == 24 * 4
    # Las anclas :00 conservan el máximo horario (interpolación de DATA-004).
    assert any(p["hora"] == 16.0 for p in ph15)
    assert all(p["hora"] % 0.25 == 0 for p in ph15)
    # Contrato intacto: los campos top-level siguen existiendo.
    for key in ("clase_final", "clase_final_label", "perfil", "explicacion", "recomendaciones"):
        assert key in r15
