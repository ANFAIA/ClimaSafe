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
    r = personalizar_riesgo(0.95, {"edad": 90}, tipo="calor")
    assert r["indice_personalizado"] <= 1.0
    # odds(0.95)=19, ×2.0=38 -> 38/39 ≈ 0.974
    assert r["indice_personalizado"] == pytest.approx(0.9744, abs=1e-3)


def test_obesidad_calor_vs_frio_es_asimetrica():
    """El punto clave: los gordos sufren más en verano, no en invierno."""
    perfil = {"imc": 33, "nivel_actividad": "intensa"}
    calor = personalizar_riesgo(0.5, perfil, tipo="calor")
    frio = personalizar_riesgo(0.5, perfil, tipo="frio")

    nombres_calor = {f["nombre"] for f in calor["factores"]}
    assert any("obesidad" in n for n in nombres_calor)  # sube el riesgo en calor
    # En frío la obesidad NO añade factor de riesgo (neutra/protectora):
    nombres_frio = {f["nombre"] for f in frio["factores"]}
    assert not any("obesidad" in n for n in nombres_frio)


def test_obesidad_calor_solo_cuenta_en_esfuerzo():
    en_reposo = personalizar_riesgo(0.5, {"imc": 33, "nivel_actividad": "reposo"}, tipo="calor")
    assert not any("obesidad" in f["nombre"] for f in en_reposo["factores"])

    en_esfuerzo = personalizar_riesgo(0.5, {"imc": 33, "nivel_actividad": "moderada"}, tipo="calor")
    assert any("obesidad" in f["nombre"] for f in en_esfuerzo["factores"])


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
    r = personalizar_riesgo(0.8, {"edad": 80, "comorbilidades": {"cardiovascular"}}, tipo="calor")
    assert {f["nombre"].split()[0] for f in r["factores"]} >= {"edad", "cardiopatía/HTA"}
    for f in r["factores"]:
        assert set(f) == {"nombre", "categoria", "factor"}
