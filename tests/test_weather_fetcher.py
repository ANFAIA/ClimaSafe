"""Tests de FORECAST-001 (seguridad de datos): el fallback nunca inventa weather.

Cubre:
  - fetch_hourly_forecast pide 7 días calendario completos (forecast_days)
  - una fecha fuera del horizonte del forecast lanza ForecastHorizonError con
    mensaje claro, en vez de devolver datos fabricados (20 °C / 50 % RH / 1013 hPa)
    o el último día observado como si fuera el objetivo
  - una fecha pasada lanza ForecastHorizonError
  - una fecha dentro del horizonte usa datos reales del forecast
  - hoy sin forecast usa la observación actual real, nunca constantes inventadas

Todo el tráfico a Open-Meteo está mockeado en `_openmeteo_request`.
"""
from datetime import date, timedelta

import pandas as pd
import pytest

import climasafeai.data.weather_fetcher as wf

HOY = date(2026, 8, 10)


def _horas(fecha_inicio: date, n_dias: int, base_temp: float) -> dict:
    """168 horas (n_dias*24) empezando a las 00:00 de fecha_inicio."""
    n = n_dias * 24
    tiempos = []
    temps = []
    for h in range(n):
        d = fecha_inicio + timedelta(hours=h)
        tiempos.append(d.isoformat())
        temps.append(base_temp + (h // 24))
    return {
        "time": tiempos,
        "temperature_2m": temps,
        "relative_humidity_2m": [40.0] * n,
        "wind_speed_10m": [10.0] * n,
        "surface_pressure": [1013.0] * n,
    }


def _fake_openmeteo(forecast_dias: int = 7, con_forecast: bool = True):
    """Fábrica de `_openmeteo_request`: archive 14 días, current real, forecast 7 días."""
    def _fake(url: str, params: dict, timeout: int = 30):
        if "archive" in url:
            # fetch_historical_hourly(lat, lon, days=14): start=hoy-14, end=hoy
            # → 15 días calendario (360 h). El mock los genera completos.
            start = HOY - timedelta(days=14)
            return {"hourly": _horas(start, 15, base_temp=10.0)}
        if "current" in params:
            return {"current": {
                "temperature_2m": 35.3,
                "relative_humidity_2m": 15,
                "wind_speed_10m": 6.1,
                "surface_pressure": 940.7,
            }}
        # forecast
        if not con_forecast:
            return {"hourly": {}}
        return {"hourly": _horas(HOY, forecast_dias, base_temp=20.0)}
    return _fake


@pytest.fixture(autouse=True)
def _mocker(monkeypatch):
    # download_openuv solo se toca para target hoy; fuera de la red falla y se
    # deja en None. Se mockea para que el test de hoy no dependa de red/fichero.
    monkeypatch.setattr(wf, "download_openuv", lambda *a, **k: pd.DataFrame())
    return monkeypatch


class TestFetchHorizonteSieteDias:
    def test_fetch_hourly_forecast_pide_7_dias_calendario(self, monkeypatch):
        """forecast_days=7 → 7 días completos de 24 h (no horas desde ahora)."""
        monkeypatch.setattr(wf, "_openmeteo_request", _fake_openmeteo())
        df = wf.fetch_hourly_forecast(40.4, -3.7)
        assert len(df) == 7 * 24
        fechas = pd.to_datetime(df["datetime"]).dt.date
        assert min(fechas) == HOY
        assert max(fechas) == HOY + timedelta(days=6)
        horas_por_dia = fechas.value_counts().to_dict()
        assert all(n == 24 for n in horas_por_dia.values())


class TestFallbackSinDatosInventados:
    def test_fecha_fuera_de_horizonte_lanza_error_claro(self, monkeypatch):
        monkeypatch.setattr(wf, "_openmeteo_request", _fake_openmeteo())
        objetivo = HOY + timedelta(days=7)  # el fetch cubre hasta +6
        with pytest.raises(wf.ForecastHorizonError) as exc:
            wf.fetch_weather_data(lat=40.4, lon=-3.7, target_date=objetivo)
        msg = str(exc.value)
        assert "forecast" in msg.lower()
        assert objetivo.isoformat() in msg
        assert (HOY + timedelta(days=6)).isoformat() in msg

    def test_fecha_muy_lejana_tambien_lanza(self, monkeypatch):
        monkeypatch.setattr(wf, "_openmeteo_request", _fake_openmeteo())
        with pytest.raises(wf.ForecastHorizonError):
            wf.fetch_weather_data(lat=40.4, lon=-3.7, target_date=HOY + timedelta(days=30))

    def test_fecha_pasada_lanza_error_claro(self, monkeypatch):
        monkeypatch.setattr(wf, "_openmeteo_request", _fake_openmeteo())
        with pytest.raises(wf.ForecastHorizonError) as exc:
            wf.fetch_weather_data(lat=40.4, lon=-3.7, target_date=HOY - timedelta(days=1))
        assert "ya pasó" in str(exc.value)

    def test_sin_forecast_descargado_lanza_error(self, monkeypatch):
        monkeypatch.setattr(wf, "_openmeteo_request", _fake_openmeteo(con_forecast=False))
        with pytest.raises(wf.ForecastHorizonError) as exc:
            wf.fetch_weather_data(lat=40.4, lon=-3.7, target_date=HOY + timedelta(days=1))
        assert "No hay forecast" in str(exc.value)

    def test_fecha_dentro_de_horizonte_usa_datos_reales_del_forecast(self, monkeypatch):
        """+3 días: el current sale del mediodía del día objetivo, no de hoy."""
        monkeypatch.setattr(wf, "_openmeteo_request", _fake_openmeteo())
        objetivo = HOY + timedelta(days=3)
        w = wf.fetch_weather_data(lat=40.4, lon=-3.7, target_date=objetivo)
        # temp del día objetivo = 20 + 3 = 23.0 (mock), no 20.0 inventado
        assert w["current"]["t2m_c"] == 23.0
        fechas_hora = pd.to_datetime(w["df_hora"]["datetime"]).dt.date
        assert max(fechas_hora) == objetivo

    def test_hoy_sin_forecast_usa_la_observacion_real_no_constantes(self, monkeypatch):
        """Forecast vacío + observación actual real → esa observación, no 20/50/1013.

        El histórico se deja hasta ayer a propósito: si cubriera hoy, el merge de
        la línea 262 rellenaría df_hora_target y el fallback ni se ejecutaría.
        """
        def _fake(url: str, params: dict, timeout: int = 30):
            if "archive" in url:
                start = HOY - timedelta(days=14)
                # 14 días hasta AYER (no incluye hoy)
                return {"hourly": _horas(start, 14, base_temp=10.0)}
            if "current" in params:
                return {"current": {
                    "temperature_2m": 35.3, "relative_humidity_2m": 15,
                    "wind_speed_10m": 6.1, "surface_pressure": 940.7,
                }}
            return {"hourly": {}}

        monkeypatch.setattr(wf, "_openmeteo_request", _fake)
        w = wf.fetch_weather_data(lat=40.4, lon=-3.7, target_date=HOY)
        ultima = w["df_hora"].iloc[-1]
        assert ultima["t2m_c"] == 35.3   # real, no 20.0
        assert ultima["rh"] == 15.0      # real, no 50.0
        assert ultima["sp"] == 940.7     # real, no 1013.0


class TestForecastHorizonError:
    def test_es_exception_y_mensaje_legible(self):
        err = wf.ForecastHorizonError("El forecast meteorológico llega hasta X")
        assert isinstance(err, Exception)
        assert "forecast" in str(err).lower()
