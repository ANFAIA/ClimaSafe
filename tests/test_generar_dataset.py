"""Tests de climasafeai.llm.generar_dataset — el input lleva el parte completo.

Regresión de LLM-004: cada ejemplo del dataset de fine-tuning tiene que incluir
en el input el parte de la ventana de actividad (temperatura media y máxima,
humedad, viento y UV) con el que se calculó la respuesta. Si se revierte el
cambio (input sin el parte completo, o sin máx/UV), estos tests fallan.

Cubre:
  - `_clima_de_la_ventana`: devuelve SIEMPRE los cinco campos de la ventana
  - `_uv_de_la_ventana`: máximo de UV de la ventana (Open-Meteo horario), con
    fallback al `uv_index` del día (OpenUV)
  - `formatear_input`: emite la máxima aunque coincida con la media, y el UV
    redondeado a un decimal
  - `generar_dataset`: descarta perfiles cuyo parte no esté completo, y todos
    los inputs de los ejemplos que entran llevan los cinco campos
  - `predict_ensemble` acepta el weather precargado (no re-descarga), que es lo
    que permite que el UV del input coincida con el que usa el pipeline
"""

from __future__ import annotations

import pandas as pd
import pytest

import climasafeai.llm.generar_dataset as gd


# ── Helpers ────────────────────────────────────────────────────────────────


def _perfil(**kw):
    p = {
        "edad": 45,
        "sexo": "hombre",
        "aclimatado": False,
        "hora_inicio": 10,
        "duracion_h": 2,
        "provincia": "Sevilla",
    }
    p.update(kw)
    return p


def _weather(**kw):
    w = {
        "lat": 37.38,
        "lon": -5.99,
        "current": {"t2m_c": 30.0, "rh": 55, "wind_speed_kmh": 12.0},
        "perfil_horario": [
            {"hora": 10, "HI": 33.0, "temp": 29.0},
            {"hora": 11, "HI": 34.0, "temp": 31.0},
            {"hora": 12, "HI": 35.0, "temp": 33.0},
        ],
        "df_hora": _df_hora(),
        "df_features": pd.DataFrame(),
        "uv_horario": _df_uv_horario(),
        "uv_index": 8.5,
    }
    w.update(kw)
    return w


def _df_hora():
    return pd.DataFrame({
        "datetime": pd.to_datetime([
            "2026-08-03 10:00", "2026-08-03 11:00", "2026-08-03 12:00",
        ]),
        "t2m_c": [29.0, 31.0, 33.0],
        "rh": [50.0, 52.0, 54.0],
        "wind_speed_kmh": [10.0, 11.0, 12.0],
        "heat_index_c": [33.0, 34.0, 35.0],
    })


def _df_uv_horario():
    return pd.DataFrame({
        "datetime": pd.to_datetime([
            "2026-08-03 10:00", "2026-08-03 11:00", "2026-08-03 12:00",
        ]),
        "uv_index": [6.5, 8.2, 9.0],
    })


def _riesgo_fake(clima, clase="SEGURO"):
    return {
        "clase": clase,
        "indice_personalizado": 0.10,
        "indice_base": 0.10,
        "factor_total": 1.0,
        "producto_bruto": 1.0,
        "capado": False,
        "factores": [],
        "recomendaciones": [],
        "clima": clima,
        "perfil": {},
    }


# ── El parte de la ventana ──────────────────────────────────────────────────


class TestClimaDeLaVentana:

    def test_devuelve_los_cinco_campos_de_la_ventana(self):
        resultado = {"weather": _weather()}
        clima = gd._clima_de_la_ventana(resultado, _perfil())
        # Ventana [10, 12): horas 10 y 11 → medias de 29 y 31
        assert clima["t_media"] == 30.0
        assert clima["t_max"] == 31.0
        assert clima["rh"] == 55
        assert clima["viento_kmh"] == 12.0
        assert clima["uv"] == 8.2  # max de UV en las horas 10-11

    def test_humedad_y_viento_caen_a_la_ventana_si_current_no_los_trae(self):
        weather = _weather()
        weather["current"] = {"t2m_c": 30.0}  # sin rh ni viento
        resultado = {"weather": weather}
        clima = gd._clima_de_la_ventana(resultado, _perfil())
        assert clima["rh"] == 51.0  # media de rh en horas 10-11
        assert clima["viento_kmh"] == 10.5  # media de viento en horas 10-11


class TestUvDeLaVentana:

    def test_usa_el_maximo_de_uv_de_las_horas_de_la_ventana(self):
        weather = _weather(uv_horario=_df_uv_horario())
        assert gd._uv_de_la_ventana(weather, 10, 12) == 8.2

    def test_sin_uv_horario_cae_al_uv_index_del_dia(self):
        weather = _weather(uv_horario=None)  # sin Open-Meteo UV
        assert gd._uv_de_la_ventana(weather, 10, 12) == 8.5


# ── El input formateado ─────────────────────────────────────────────────────


class TestFormatearInput:

    def test_con_clima_incluye_el_parte_completo(self):
        clima = {"t_media": 30.0, "t_max": 34.0, "rh": 55, "viento_kmh": 12.0, "uv": 8.5}
        texto = gd.formatear_input(_perfil(), clima)
        assert "Tiempo en esa franja" in texto
        assert "30.0 °C de media" in texto
        assert "máx 34.0 °C" in texto
        assert "humedad 55 %" in texto
        assert "viento 12.0 km/h" in texto
        assert "UV 8.5" in texto

    def test_emite_la_maxima_aunque_coincida_con_la_media(self):
        # Regresión: antes se omitía la máxima cuando t_max == t_media, y 193 de
        # 400 inputs se quedaban sin ella.
        clima = {"t_media": 30.0, "t_max": 30.0, "rh": 55, "viento_kmh": 12.0, "uv": 8.5}
        texto = gd.formatear_input(_perfil(), clima)
        assert "máx 30.0 °C" in texto

    def test_redondea_uv_a_un_decimal(self):
        clima = {"t_media": 30.0, "t_max": 34.0, "rh": 55, "viento_kmh": 12.0, "uv": 7.7645}
        texto = gd.formatear_input(_perfil(), clima)
        assert "UV 7.8" in texto
        assert "UV 7.7645" not in texto

    def test_sin_clima_no_anade_la_linea_del_parte(self):
        texto = gd.formatear_input(_perfil())
        assert "Tiempo en esa franja" not in texto


# ── La generación del dataset ───────────────────────────────────────────────


class TestGenerarDataset:

    def test_todos_los_inputs_llevan_el_parte_completo(self, monkeypatch):
        clima = {"t_media": 20.0, "t_max": 20.0, "rh": 60, "viento_kmh": 5.0, "uv": 4.0}
        monkeypatch.setattr(gd, "predecir", lambda perfil: _riesgo_fake(clima))
        dataset = gd.generar_dataset(num_ejemplos=10, equilibrar=False)
        assert len(dataset) == 10
        for ex in dataset:
            assert "°C de media" in ex["input"]
            assert "máx" in ex["input"]
            assert "humedad" in ex["input"]
            assert "viento" in ex["input"]
            assert "UV 4.0" in ex["input"]

    def test_descarta_perfiles_con_parte_incompleto(self, monkeypatch):
        # Un parte sin UV no puede entrar: el input no llevaría el determinante
        # de la respuesta, que es justo lo que LLM-004 corrige.
        clima = {"t_media": 20.0, "t_max": 21.0, "rh": 60, "viento_kmh": 5.0, "uv": None}
        monkeypatch.setattr(gd, "predecir", lambda perfil: _riesgo_fake(clima))
        dataset = gd.generar_dataset(num_ejemplos=5, equilibrar=False)
        assert dataset == []

    # ── Correcciones QC LLM-015 ──────────────────────────────────────────────

    def test_descarta_input_cuyo_texto_no_lleva_max_o_uv(self, monkeypatch):
        # Invariant sobre el TEXTO, no solo sobre el dict: si formatear_input no
        # emite la máxima o el UV (hallazgo real: 97/100 pares del val de agosto),
        # el par no entra aunque el dict diga que está completo.
        clima = {"t_media": 20.0, "t_max": 21.0, "rh": 60, "viento_kmh": 5.0, "uv": 4.0}
        monkeypatch.setattr(gd, "predecir", lambda perfil: _riesgo_fake(clima))
        monkeypatch.setattr(gd, "formatear_input",
                            lambda perfil, clima: "Edad: 45. Sin parte completo.")
        dataset = gd.generar_dataset(num_ejemplos=5, equilibrar=False)
        assert dataset == []

    def test_dedupe_descarta_inputs_ya_emitidos(self, monkeypatch, capsys):
        # Hallazgo real del QC: 184 pares casi idénticos en train.jsonl. Si el
        # mismo perfil con el mismo parte se repite, solo entra el primero.
        clima = {"t_media": 20.0, "t_max": 21.0, "rh": 60, "viento_kmh": 5.0, "uv": 4.0}
        monkeypatch.setattr(gd, "predecir", lambda perfil: _riesgo_fake(clima))
        # 10 perfiles idénticos + el mismo clima → 9 inputs duplicados.
        perfiles = [_perfil() for _ in range(10)]
        monkeypatch.setattr(gd, "generar_perfiles", lambda n: perfiles)
        dataset = gd.generar_dataset(num_ejemplos=10, equilibrar=False)
        assert len(dataset) == 1
        assert "9 inputs duplicados" in capsys.readouterr().out


# ── predict_ensemble con weather precargado ─────────────────────────────────


class TestPredictEnsembleWeatherPrecargado:

    def test_acepta_weather_precargado_sin_redescargar(self, monkeypatch):
        import climasafeai.models.ensemble as ensemble

        fetch_llamadas = []
        monkeypatch.setattr(ensemble, "fetch_weather_data",
                            lambda **kw: (fetch_llamadas.append(kw) or _weather()))
        monkeypatch.setattr(
            ensemble, "_predecir_tabular",
            lambda *a, **k: {"prob_riesgo": 0.1, "conformal_set_size": 2, "clase": 0, "_X": None},
        )
        monkeypatch.setattr(
            ensemble, "_predecir_lstm",
            lambda *a, **k: {"calor": {"prob_riesgo": 0.1, "clase": 0}, "frio": {"prob_riesgo": 0.05, "clase": 0}},
        )

        weather = _weather(df_hora=_df_hora(), uv_index=None)
        r = ensemble.predict_ensemble(lat=42.29, lon=-8.81, provincia="Pontevedra",
                                      perfil={"edad": 40, "sexo": "mujer", "fototipo": "II"},
                                      weather=weather)
        assert fetch_llamadas == []  # usó el weather pasado, no re-descargó
        assert r["weather"]["uv_index"] is None
