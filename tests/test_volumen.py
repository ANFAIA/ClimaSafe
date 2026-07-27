"""Tests para climasafeai/models/volumen.py y el endpoint /api/riesgo-volumen."""

import pytest
from climasafeai.models.volumen import (
    estimar_afectados,
    _multiplicador_climatico,
    _tasa_incidencia_directa,
    _categoria_hi,
)


class TestMultiplicadorClimatico:
    def test_hi_normal(self):
        assert _multiplicador_climatico(20) == 1.0
        assert _multiplicador_climatico(26) == 1.0

    def test_hi_precaucion(self):
        m = _multiplicador_climatico(30)
        assert 1.0 <= m <= 1.1

    def test_hi_precaucion_extrema(self):
        m = _multiplicador_climatico(35)
        assert 1.05 <= m <= 1.15

    def test_hi_peligro(self):
        m = _multiplicador_climatico(42)
        assert 1.15 <= m <= 1.30

    def test_hi_peligro_extremo(self):
        m = _multiplicador_climatico(48)
        assert 1.25 <= m <= 1.40

    def test_none_returns_1(self):
        assert _multiplicador_climatico(None) == 1.0


class TestTasaIncidenciaDirecta:
    def test_baja_normal(self):
        assert _tasa_incidencia_directa(20, 1.0) == 0.0003

    def test_alta_extremo(self):
        t = _tasa_incidencia_directa(48, 1.0)
        assert t >= 0.004

    def test_factor_evento_no_afecta(self):
        assert _tasa_incidencia_directa(20, 2.0) == 0.0003


class TestCategoriaHI:
    def test_normal(self):
        assert _categoria_hi(20) == "NORMAL"
        assert _categoria_hi(26) == "NORMAL"

    def test_precaucion(self):
        assert _categoria_hi(27) == "PRECAUCION"
        assert _categoria_hi(31) == "PRECAUCION"

    def test_precaucion_extrema(self):
        assert _categoria_hi(32) == "PRECAUCION_EXTREMA"
        assert _categoria_hi(38) == "PRECAUCION_EXTREMA"

    def test_peligro(self):
        assert _categoria_hi(39) == "PELIGRO"
        assert _categoria_hi(44) == "PELIGRO"

    def test_peligro_extremo(self):
        assert _categoria_hi(45) == "PELIGRO_EXTREMO"
        assert _categoria_hi(50) == "PELIGRO_EXTREMO"

    def test_none(self):
        assert _categoria_hi(None) == "desconocido"


class TestEstimarAfectados:
    def test_sin_calor_no_hay_exceso(self):
        r = estimar_afectados(total_personas=5000, hi_peak=20, pct_mayores_50=30)
        assert r["estimacion_atencion_medica"] <= 5
        assert r["exceso_ecv"] == 0.0

    def test_ejemplo_roadmap(self):
        """El roadmap dice: De 5000 asistentes, ~75 podrian requerir atencion
        medica. Con HI ~38 (precaucion extrema), 30% >50, deporte:
        debe dar un numero cercano a 75."""
        r = estimar_afectados(
            total_personas=5000, hi_peak=38, pct_mayores_50=30, tipo_evento="deporte",
        )
        assert 40 <= r["estimacion_atencion_medica"] <= 150
        assert r["rango_bajo"] <= r["estimacion_atencion_medica"] <= r["rango_alto"]
        assert "5000" in r["mensaje"]

    def test_escalado_lineal(self):
        """Doblar la poblacion debe aproximadamente doblar la estimacion."""
        r1 = estimar_afectados(1000, hi_peak=38, pct_mayores_50=30)
        r2 = estimar_afectados(2000, hi_peak=38, pct_mayores_50=30)
        ratio = r2["estimacion_atencion_medica"] / r1["estimacion_atencion_medica"]
        assert 1.8 <= ratio <= 2.2

    def test_pct_mayores_aumenta_estimacion(self):
        r_joven = estimar_afectados(5000, hi_peak=38, pct_mayores_50=10)
        r_mayor = estimar_afectados(5000, hi_peak=38, pct_mayores_50=80)
        assert r_mayor["estimacion_atencion_medica"] > r_joven["estimacion_atencion_medica"]

    def test_evento_deportivo_mayor_que_general(self):
        r_gen = estimar_afectados(5000, hi_peak=38, tipo_evento="general")
        r_dep = estimar_afectados(5000, hi_peak=38, tipo_evento="deporte")
        assert r_dep["estimacion_atencion_medica"] > r_gen["estimacion_atencion_medica"]

    def test_total_cero(self):
        r = estimar_afectados(0, hi_peak=38)
        assert "error" in r

    def test_mensaje_largo_incluye_parametros(self):
        r = estimar_afectados(5000, hi_peak=38, pct_mayores_50=30, tipo_evento="deporte")
        assert "ECV" in r["mensaje_largo"]
        assert "HI" in r["mensaje_largo"]

    def test_sin_hi_peak_funciona(self):
        r = estimar_afectados(5000, hi_peak=None, pct_mayores_50=30)
        assert r["estimacion_atencion_medica"] >= 0
        assert r["clima"]["categoria"] == "desconocido"
        assert r["multiplicador_climatico"] == 1.0

    def test_extremo_concuerda_con_literatura(self):
        """HI 50, evento deportivo, 50% >50:
        debe dar un % recognoscible (>2% de la poblacion)."""
        r = estimar_afectados(10000, hi_peak=50, pct_mayores_50=50, tipo_evento="deporte")
        pct = r["pct_estimado"]
        assert 1.5 <= pct <= 6.0
        assert r["clima"]["categoria"] == "PELIGRO_EXTREMO"

    def test_rango_simetrico(self):
        r = estimar_afectados(5000, hi_peak=38)
        assert r["rango_bajo"] <= r["estimacion_atencion_medica"] <= r["rango_alto"]
        diff_low = r["estimacion_atencion_medica"] - r["rango_bajo"]
        diff_high = r["rango_alto"] - r["estimacion_atencion_medica"]
        assert abs(diff_low - diff_high) <= 1

    def test_claves_presentes(self):
        r = estimar_afectados(5000, hi_peak=38)
        for key in (
            "total_personas", "estimacion_atencion_medica", "rango_bajo",
            "rango_alto", "pct_estimado", "clima", "prevalencia_ecv_usada",
            "multiplicador_climatico", "factor_evento", "tasa_incidencia_directa",
            "exceso_ecv", "exceso_directo", "mensaje", "mensaje_largo",
        ):
            assert key in r, f"Falta clave: {key}"
