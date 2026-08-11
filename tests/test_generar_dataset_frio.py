"""Tests de climasafeai.llm.generar_dataset_frio (DATA-008).

El generador de frío lee dataset_frio_labeled.parquet (2016-2026) y produce
ejemplos que:
  - salen del pipeline real: `predecir` recibe un `weather` construido desde
    el parquet, no de fetch_weather_data
  - llevan en el input los cinco campos del parte (media, máx, humedad,
    viento y UV)
  - solo entran si la media de la ventana es < 10 °C (canal frío)
  - se mezclan con los ~300 de calor actuales y se parten en train/val mixtos

Sin red ni GPU: se mockean `predict_ensemble`, `fetch_weather_data`, la lectura
del parquet y el UV histórico.
"""

from __future__ import annotations

import json
import re
from datetime import date

import pandas as pd
import pytest

import climasafeai.llm.generar_dataset as gd
import climasafeai.llm.generar_dataset_frio as gf


# ── Helpers ────────────────────────────────────────────────────────────────


def _perfil(**kw):
    p = {
        "edad": 45,
        "sexo": "hombre",
        "aclimatado": False,
        "nivel_actividad": "ligera",
        "hora_inicio": 10,
        "duracion_h": 2,
        "provincia": "León",
    }
    p.update(kw)
    return p


def _weather(**kw):
    w = {
        "lat": 42.5987,
        "lon": -5.5665,
        "current": {"t2m_c": 4.0, "rh": 70, "wind_speed_kmh": 10.0, "sp": 91000.0},
        "perfil_horario": [
            {"hora": 10, "HI": 4.0, "temp": 3.0},
            {"hora": 11, "HI": 4.5, "temp": 5.0},
        ],
        "df_hora": pd.DataFrame({
            "datetime": pd.to_datetime(["2021-01-15 10:00", "2021-01-15 11:00"]),
            "t2m_c": [3.0, 5.0],
            "rh": [72.0, 68.0],
            "wind_speed_kmh": [12.0, 9.0],
            "heat_index_c": [4.0, 4.5],
        }),
        "df_features": pd.DataFrame(),
        "uv_horario": _df_uv_horario(),
        "uv_index": 3.2,
        "target_date": "2021-01-15",
    }
    w.update(kw)
    return w


def _df_uv_horario():
    return pd.DataFrame({
        "datetime": pd.to_datetime(["2021-01-15 10:00", "2021-01-15 11:00"]),
        "uv_index": [1.8, 3.2],
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


def _clima_frio(t_media=4.0):
    return {"t_media": t_media, "t_max": t_media + 1.0, "rh": 70.0,
            "viento_kmh": 10.0, "uv": 3.2}


def _fila_parquet(provincia="León", fecha="2021-01-15", t2m=4.0):
    """Una fila del parquet: diaria, con las features agregadas precomputadas."""
    f = pd.DataFrame([{
        "fecha": fecha,
        "provincia": provincia,
        "defunciones_atrib_def_temp": 0.0,
        "datetime": pd.Timestamp(f"{fecha} 14:00"),
        "t2m_c": t2m,
        "rh": 70.0,
        "wind_speed_kmh": 10.0,
        "sp": 91000.0,
        "heat_index_c": t2m,
        "wbgt_c": t2m - 1.0,
        "wind_chill_c": t2m - 2.0,
        "heat_index_mean": t2m,
        "heat_index_std": 1.0,
        "heat_index_min": t2m - 1.0,
        "horas_sobre_umbral": 0,
        "wind_chill_mean": t2m - 2.0,
        "wind_chill_std": 1.0,
        "wind_chill_max": t2m,
        "horas_bajo_umbral": 3,
        "t2m_min_noche": t2m - 3.0,
        "horas_wc_severo": 0,
        "heat_index_c_lag1": t2m + 1.0,
        "heat_index_c_roll3": t2m,
        "heat_index_c_roll7": t2m,
        "dias_consec_sobre_umbral": 0.0,
        "grados_dia_calor_roll7": 0.0,
        "grados_dia_calor_roll14": 0.0,
        "wind_chill_mean_roll3": t2m - 2.0,
        "wind_chill_mean_roll7": t2m - 2.0,
        "wind_chill_mean_roll14": t2m - 2.0,
        "grados_dia_frio_roll7": 10.0,
        "grados_dia_frio_roll14": 20.0,
        "dias_consec_bajo_umbral": 2.0,
        "t2m_min_noche_lag1": t2m - 2.0,
        "t2m_min_noche_roll7": t2m - 3.0,
        "dias_consec_wc_severo": 0.0,
        "horas_wc_severo_sum14": 0.0,
        "clase_riesgo_frio": 0,
        "clase_riesgo_frio_label": "seguro",
    }])
    f["_dt"] = pd.to_datetime(f["datetime"])
    f["_mes"] = f["_dt"].dt.month
    f["_fecha_date"] = f["_dt"].dt.date
    return f


# ── Weather desde el parquet ───────────────────────────────────────────────


class TestConstruirWeather:

    def test_monta_las_mismas_claves_que_fetch_weather_data(self, monkeypatch):
        parquet = _fila_parquet()
        monkeypatch.setattr(gf, "_uv_estimado_historico", lambda *a: _df_uv_horario())
        weather = gf._construir_weather(parquet, "León", date(2021, 1, 15))
        assert weather is not None
        # Las claves que consume predict_ensemble
        assert set(weather) >= {"lat", "lon", "current", "df_hora", "df_features",
                                "uv_index", "uv_horario", "target_date"}
        assert weather["current"]["t2m_c"] == 4.0
        assert weather["uv_index"] == 3.2  # max del uv_horario
        assert weather["target_date"] == "2021-01-15"
        # df_hora lleva las columnas brutas + índices que pide perfil_horario
        assert "heat_index_c" in weather["df_hora"].columns
        # Las etiquetas/fugas del parquet NO entran como feature
        assert "clase_riesgo_frio" not in weather["df_features"].columns
        assert "defunciones_atrib_def_temp" not in weather["df_features"].columns
        # Las columnas auxiliares de filtrado tampoco
        assert "_dt" not in weather["df_features"].columns
        assert "_fecha_date" not in weather["df_features"].columns

    def test_sin_fila_o_sin_uv_devuelve_none(self, monkeypatch):
        parquet = _fila_parquet()
        # Sin fila para esa provincia/fecha
        assert gf._construir_weather(parquet, "León", date(2020, 1, 1)) is None
        # Sin UV → no hay parte completo → no se puede predecir
        monkeypatch.setattr(gf, "_uv_estimado_historico", lambda *a: None)
        assert gf._construir_weather(parquet, "León", date(2021, 1, 15)) is None


class TestSeleccionarDiasFrios:

    def test_reparte_entre_varias_provincias_y_requiere_t2m_menor_5(self):
        import random
        parquet = pd.concat([
            _fila_parquet(p, f"2021-{mes}-15", t2m=2.0)
            for p in ["León", "Soria", "Lleida", "Girona"] for mes in ("01", "02")
        ] + [_fila_parquet("León", "2021-08-15", t2m=30.0)])
        dias = gf._seleccionar_dias_frios(parquet, 8, random.Random(0))
        provincias = {p for p, _ in dias}
        assert {"León", "Soria", "Lleida", "Girona"} <= provincias
        # El día de agosto (calor) nunca entra
        assert all(d != date(2021, 8, 15) for _, d in dias)


# ── Generación de ejemplos de frío ─────────────────────────────────────────


class TestGenerarFrio:

    def _parchear(self, monkeypatch, riesgo):
        """Parchea parquet, días, weather y predecir para no tocar red ni modelos."""
        parquet = _fila_parquet()
        monkeypatch.setattr(gf.pd, "read_parquet", lambda *a, **k: parquet)
        monkeypatch.setattr(gf, "_seleccionar_dias_frios",
                            lambda *a, **k: [("León", date(2021, 1, 15))])
        monkeypatch.setattr(gf, "_construir_weather",
                            lambda *a, **k: _weather())
        monkeypatch.setattr(gf.gd, "predecir",
                            lambda perfil, weather=None: riesgo(perfil))

    def test_entra_un_dia_frio_con_los_cinco_campos_y_media_menor_10(self, monkeypatch):
        self._parchear(monkeypatch, lambda perfil: _riesgo_fake(_clima_frio(t_media=4.0)))
        dataset = gf.generar_frio(num_frio=3, equilibrar=False)
        assert len(dataset) == 3
        for ex in dataset:
            assert "°C de media" in ex["input"]
            assert "máx" in ex["input"]
            assert "humedad" in ex["input"]
            assert "viento" in ex["input"]
            assert "UV" in ex["input"]
            # Criterio 4: la media de la ventana es real y < 10 °C
            assert "4.0 °C de media" in ex["input"]
            assert "Ubicación: León" in ex["input"]
            # Procedencia del día real del parquet, verificable en el JSONL
            assert ex["fecha"] == "2021-01-15"
            assert ex["provincia"] == "León"

    def test_descarta_ventana_con_media_igual_o_mayor_10(self, monkeypatch):
        # Un día frío con una ventana templada (t_media 12) NO es un ejemplo de
        # frío: el canal se decide por la media real de la ventana, no por la
        # provincia ni por la fecha.
        self._parchear(monkeypatch, lambda perfil: _riesgo_fake(_clima_frio(t_media=12.0)))
        dataset = gf.generar_frio(num_frio=3, equilibrar=False)
        assert dataset == []

    def test_descarta_parte_incompleto_sin_uv(self, monkeypatch):
        clima = {"t_media": 4.0, "t_max": 5.0, "rh": 70.0, "viento_kmh": 10.0, "uv": None}
        self._parchear(monkeypatch, lambda perfil: _riesgo_fake(clima))
        assert gf.generar_frio(num_frio=3, equilibrar=False) == []

    def test_los_dias_reparten_varias_provincias_exigidas(self, monkeypatch):
        parquet = pd.concat([
            _fila_parquet(p, "2021-01-15", t2m=2.0)
            for p in ["León", "Soria", "Lleida", "Girona"]
        ])
        parquet["_dt"] = pd.to_datetime(parquet["datetime"])
        parquet["_mes"] = parquet["_dt"].dt.month
        parquet["_fecha_date"] = parquet["_dt"].dt.date
        monkeypatch.setattr(gf.pd, "read_parquet", lambda *a, **k: parquet)
        monkeypatch.setattr(gf, "_construir_weather",
                            lambda parquet, provincia, fecha: _weather())
        monkeypatch.setattr(gf.gd, "predecir",
                            lambda perfil, weather=None: _riesgo_fake(_clima_frio(t_media=3.0)))
        # 8 días → 2 por provincia en round-robin → 4 provincias cubiertas
        dataset = gf.generar_frio(num_frio=16, equilibrar=False)
        provincias = set()
        for ex in dataset:
            m = re.search(r"Ubicación: ([\wáéíóúÁÉÍÓÚ/]+)", ex["input"])
            if m:
                provincias.add(m.group(1))
        assert {"León", "Soria", "Lleida", "Girona"} <= provincias


# ── Mezcla calor + frío ────────────────────────────────────────────────────


class TestGenerarDatasetMixto:

    def test_train_y_val_quedan_mixtos_y_con_reparto_por_canal(self, monkeypatch, tmp_path):
        calor_train = tmp_path / "calor_train.jsonl"
        calor_val = tmp_path / "calor_val.jsonl"
        with open(calor_train, "w") as f:
            for i in range(5):
                f.write(json.dumps({
                    "instruction": "i",
                    "input": f"Edad: 40. Tiempo en esa franja: {30.0 + i} °C de media, "
                             f"máx {32.0 + i} °C, humedad 50 %, viento 10.0 km/h, UV 8.0.",
                    "output": "RIESGO: SEGURO",
                }) + "\n")
        with open(calor_val, "w") as f:
            f.write(json.dumps({
                "instruction": "i",
                "input": "Edad: 40. Tiempo en esa franja: 31.0 °C de media, "
                         "máx 33.0 °C, humedad 50 %, viento 10.0 km/h, UV 8.0.",
                "output": "RIESGO: SEGURO",
            }) + "\n")

        # 6 de frío (contrapesando los 6 de calor) para que el split de val no
        # se quede con un solo canal por azar.
        frio_fake = [
            {"instruction": "i",
             "input": f"Edad: {70 - i}. Tiempo en esa franja: {2.0 + i} °C de media, "
                      f"máx {3.0 + i} °C, humedad 70 %, viento 12.0 km/h, UV 2.0.",
             "output": "RIESGO: PRECAUCION"}
            for i in range(6)
        ]
        monkeypatch.setattr(gf, "generar_frio", lambda *a, **k: frio_fake)

        train, val, conteo = gf.generar_dataset_mixto(
            num_frio=6, calor_paths=[str(calor_train), str(calor_val)],
            val_split=0.5, seed=42,
        )
        assert conteo == {"calor": 6, "frio": 6}
        assert len(train) == 6 and len(val) == 6
        # Ambos splits llevan calor Y frío (mezcla de verdad, no uno por lado)
        for split in (train, val):
            canales, _ = gf._reparto_dataset(split)
            assert "calor" in canales and "frio" in canales

    def test_reparto_dataset_lee_canal_y_clase_del_jsonl(self):
        dataset = [
            {"input": "... 3.0 °C de media, máx 4.0 °C ...", "output": "RIESGO: PRECAUCION"},
            {"input": "... 31.0 °C de media, máx 33.0 °C ...", "output": "RIESGO: SEGURO"},
        ]
        canales, clases = gf._reparto_dataset(dataset)
        assert canales == {"frio": 1, "calor": 1}
        assert clases == {"PRECAUCION": 1, "SEGURO": 1}


# ── predecir con weather precargado ────────────────────────────────────────


class TestPredecirWeatherPrecargado:

    def test_usa_el_weather_del_parquet_y_no_redescarga(self, monkeypatch):
        import climasafeai.data.weather_fetcher as wf
        import climasafeai.models.ensemble as ensemble

        recibido = {}
        fetch_llamadas = []
        monkeypatch.setattr(wf, "fetch_weather_data",
                            lambda **kw: (fetch_llamadas.append(kw) or {}))
        monkeypatch.setattr(gd, "_uv_horario", lambda *a: None)

        def _predict(**kw):
            recibido["weather"] = kw.get("weather")
            return {
                "clase_final_label": "SEGURO",
                "perfil": {"calor": {
                    "prob_personalizada": 0.1, "prob_poblacional": 0.1,
                    "factor_total": 1.0, "producto_bruto": 1.0, "capado": False,
                    "factores": [],
                }},
                "recomendaciones": [],
                "weather": {"current": {"rh": 70.0, "wind_speed_kmh": 10.0},
                            "perfil_horario": [{"hora": 10, "temp": 3.0}],
                            "df_hora": _weather()["df_hora"],
                            "uv_index": 3.2},
            }
        monkeypatch.setattr(ensemble, "predict_ensemble", _predict)

        weather = _weather()
        riesgo = gd.predecir(_perfil(lat=42.5987, lon=-5.5665, provincia="León"),
                             weather=weather)
        assert fetch_llamadas == []  # no tocó fetch_weather_data
        assert recibido["weather"] is weather  # el pipeline usó EL weather del parquet
        assert riesgo["clase"] == "SEGURO"
