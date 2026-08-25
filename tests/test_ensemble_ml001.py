"""ML-001 regresión: sin ningún modelo ML utilizable, predict_ensemble debe
lanzar (contrato original) en vez de degradar silenciosamente a solo-Fórmula.

El bucle de descubrimiento tolera modelos individuales que fallan, pero si
TODOS los tabular/lstm fallan, el ensemble ya no es un ensemble: la API debe
devolver {"error": ...} como antes de los manifiestos (test_api_predict_
sin_modelo_real_devuelve_error).
"""
import pandas as pd
import pytest

from climasafeai.models import ensemble


@pytest.fixture
def df_features_minimo() -> pd.DataFrame:
    """df_features con las columnas mínimas para llegar al bucle de descubrimiento."""
    idx = pd.date_range("2026-08-25", periods=24, freq="h")
    return pd.DataFrame(0.0, index=idx, columns=["t2m_c"])


def test_sin_ningun_ml_utilizable_lanza(monkeypatch, tmp_path):
    """Manifests con un tabular que falla + formula: debe lanzar, no degradar."""
    # Descubrimiento controlado: un tabular roto y una formula
    descubiertos = [
        {"name": "XGBoost_calor", "type": "tabular", "class": "calor",
         "file": "no_existe.joblib"},
        {"name": "Formula", "type": "formula", "class": "both"},
    ]
    monkeypatch.setattr(ensemble, "discover_models", lambda: descubiertos)
    monkeypatch.setattr(
        ensemble, "_predecir_tabular",
        lambda *a, **k: (_ for _ in ()).throw(FileNotFoundError("no_existe.joblib")),
    )
    weather = {
        "current": {"t2m_c": 30.0, "rh": 40, "wind_speed_kmh": 5},
        "df_hora": None,
        "df_features": None,
        "lat": 0.0,
        "lon": 0.0,
        "target_date": None,
    }
    with pytest.raises(RuntimeError, match="(?i)ningún modelo ml"):
        ensemble.predict_ensemble(weather=weather)


def test_lstm_solo_error_lanza(monkeypatch):
    """Caso exacto de la regresión ML-001: los tabular fallan y el LSTM
    devuelve {"error": ...} (no lanza) → debe lanzar igualmente."""
    descubiertos = [
        {"name": "XGBoost_calor", "type": "tabular", "class": "calor",
         "file": "roto.joblib"},
        {"name": "LSTM", "type": "lstm", "class": "both"},
        {"name": "Formula", "type": "formula", "class": "both"},
    ]
    monkeypatch.setattr(ensemble, "discover_models", lambda: descubiertos)
    monkeypatch.setattr(
        ensemble, "_predecir_tabular",
        lambda *a, **k: (_ for _ in ()).throw(FileNotFoundError("roto.joblib")),
    )
    # _predecir_lstm NO lanza: devuelve stub de error (comportamiento real)
    monkeypatch.setattr(
        ensemble, "_predecir_lstm",
        lambda *a, **k: {"error": "No se pudo cargar LSTM"},
    )
    weather = {
        "current": {"t2m_c": 30.0, "rh": 40, "wind_speed_kmh": 5},
        "df_hora": None,
        "df_features": None,
        "lat": 0.0,
        "lon": 0.0,
        "target_date": None,
    }
    with pytest.raises(RuntimeError, match="(?i)ningún modelo ml"):
        ensemble.predict_ensemble(weather=weather)


def test_con_ml_utilizable_no_lanza(monkeypatch):
    """Si al menos un ML ejecuta, el ensemble sigue aunque otro falle."""
    descubiertos = [
        {"name": "XGBoost_calor", "type": "tabular", "class": "calor",
         "file": "ok.joblib"},
        {"name": "RandomForest_frio", "type": "tabular", "class": "frio",
         "file": "roto.joblib"},
        {"name": "Formula", "type": "formula", "class": "both"},
    ]
    monkeypatch.setattr(ensemble, "discover_models", lambda: descubiertos)

    def _tabular_ok(model_file, clase, df_features, provincia, grupo_edad="todos"):
        assert clase in ("calor", "frio")
        if clase == "frio":
            raise FileNotFoundError("roto.joblib")
        return {
            "clase_argmax": 0, "clase_threshold": 0,
            "probabilidades": [0.9, 0.08, 0.02], "prob_riesgo": 0.1,
            "thresholds_usados": {}, "conformal_confianza": None,
            "conformal_set_size": 2,
        }

    monkeypatch.setattr(ensemble, "_predecir_tabular", _tabular_ok)
    # LSTM falla también: con un solo ML vivo basta
    monkeypatch.setattr(
        ensemble, "_predecir_lstm",
        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("lstm caido")),
    )
    monkeypatch.setattr(ensemble, "perfil_horario_desde_df", lambda *a, **k: None)
    monkeypatch.setattr(ensemble, "explicar_ensemble", lambda *a, **k: {})
    monkeypatch.setattr(ensemble, "generar_recomendaciones", lambda *a, **k: [])

    weather = {
        "current": {"t2m_c": 20.0, "rh": 50, "wind_speed_kmh": 10},
        "df_hora": None,
        "df_features": None,
        "lat": 0.0,
        "lon": 0.0,
        "target_date": None,
    }
    res = ensemble.predict_ensemble(weather=weather)
    assert res["clase_final"] == 0
