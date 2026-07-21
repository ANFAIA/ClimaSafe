import joblib
import numpy as np
import pandas as pd
import torch

from climasafeai.features.build_features import process_input
from climasafeai.features.personalizacion import personalizar_riesgo
from climasafeai.features.weather_indices import (
    heat_index, wind_chill, wbgt_from_heat_index,
)
from climasafeai.models.predict_model import (
    apply_class_thresholds,
    CLASS_THRESHOLDS_RECOMENDADOS,
    CLASS_THRESHOLDS_LSTM,
)
from climasafeai.models.lstm_province_hybrid import (
    load_lstm_province_hybrid,
    LSTM_PROVINCE_HYBRID_MODEL_PATH,
)
from climasafeai.data.weather_fetcher import (
    fetch_weather_data,
    build_sequence_24h,
    build_daily_feature_vector,
    get_province_idx,
    get_ine_features,
    escalar_para_lstm,
)
from climasafeai.utils.paths import MODELS_DIR, ARTIFACTS_DIR
from climasafeai.models.explicabilidad import explicar_ensemble
from climasafeai.models.recomendaciones import generar_recomendaciones

CLASES = ["SEGURO", "PRECAUCION", "PELIGRO"]

_FACTORES_RIESGO_EDAD = {
    "calor": {"joven": 0.6, "adulto": 0.6, "mayor": 0.75, "anciano": 0.875, "viejano": 1.0, "todos": 1.0},
    "frio":  {"joven": 0.75, "adulto": 0.75, "mayor": 0.875, "anciano": 0.95, "viejano": 1.0, "todos": 1.0},
}


def _edad_a_estrato(edad: float | None) -> str:
    if edad is None:
        return "todos"
    if edad < 45:
        return "joven"
    if edad < 60:
        return "adulto"
    if edad < 70:
        return "mayor"
    if edad < 80:
        return "anciano"
    return "viejano"


def _cargar_province_mapping() -> dict[str, int]:
    from climasafeai.features.external_features import _EMBEDDED_DEMOGRAPHICS
    return {p: i for i, p in enumerate(sorted(_EMBEDDED_DEMOGRAPHICS.keys()))}


def _aplicar_factor_edad(proba: np.ndarray, clase: str, grupo_edad: str) -> np.ndarray:
    """Ajusta probabilidad (3,) por factor edad. Devuelve copia."""
    factor = _FACTORES_RIESGO_EDAD.get(clase, {}).get(grupo_edad, 1.0)
    if factor == 1.0:
        return proba
    p = proba.copy()
    prob_riesgo = 1.0 - p[0]
    prob_riesgo_adj = prob_riesgo * factor
    p_sum = p[1] + p[2]
    if p_sum > 0:
        p1_frac = p[1] / p_sum
        p2_frac = p[2] / p_sum
        p[0] = 1.0 - prob_riesgo_adj
        p[1] = prob_riesgo_adj * p1_frac
        p[2] = prob_riesgo_adj * p2_frac
    return p


def _predecir_tabular(
    model_path: str,
    clase: str,
    df_features: pd.DataFrame,
    provincia: str | None = None,
    grupo_edad: str = "todos",
) -> dict:
    model = joblib.load(MODELS_DIR / model_path)
    df_input = df_features.copy()
    if provincia:
        df_input["provincia"] = provincia
    X = process_input(df_input, clase=clase)

    proba = model.predict_proba(X)

    # Calibración isotónica post-hoc para frío
    _calibrado = False
    if clase == "frio":
        try:
            from climasafeai.models.calibrate import load_isotonic, calibrate_proba
            iso = load_isotonic("frio")
            if iso is not None:
                proba = calibrate_proba(proba, iso)
                _calibrado = True
        except Exception:
            pass

    pred_argmax = int(proba[0].argmax())

    proba[0] = _aplicar_factor_edad(proba[0], clase, grupo_edad)

    u_global = CLASS_THRESHOLDS_RECOMENDADOS.get(clase, {"t1": 0.5, "t2": 0.4})
    u = dict(u_global)
    if provincia and not _calibrado:
        try:
            umb_path = ARTIFACTS_DIR / f"umbrales_provincia_{clase}.joblib"
            if umb_path.exists():
                umb_prov = joblib.load(umb_path)
                prov_u = umb_prov.get(provincia, u_global)
                u = {"t1": prov_u["t1"], "t2": prov_u["t2"]}
        except Exception:
            pass

    pred_th = int(apply_class_thresholds(proba, **u)[0])
    prob_riesgo = float(1.0 - proba[0, 0])

    return {
        "clase_argmax": pred_argmax,
        "clase_threshold": pred_th,
        "probabilidades": proba[0].round(4).tolist(),
        "prob_riesgo": round(prob_riesgo, 4),
        "thresholds_usados": u,
        "_X": X,
    }


def _predecir_lstm(
    df_hora: pd.DataFrame,
    df_features: pd.DataFrame,
    provincia: str | None = None,
    grupo_edad: str = "todos",
) -> dict:
    try:
        model = load_lstm_province_hybrid(LSTM_PROVINCE_HYBRID_MODEL_PATH, device="cpu")
    except Exception as e:
        return {"error": f"No se pudo cargar LSTM: {e}"}

    seq = build_sequence_24h(df_hora)
    if seq is None:
        return {"error": "No hay datos horarios para LSTM"}

    daily_vec = build_daily_feature_vector(df_features)
    if daily_vec is None:
        return {"error": "No se pudieron generar features diarias para LSTM"}

    prov_name = provincia or "Madrid"
    ine_vec = get_ine_features(prov_name)
    pidx = np.array([get_province_idx(prov_name)], dtype=np.int64)

    seq_s, ine_s, daily_s = escalar_para_lstm(seq, ine_vec, daily_vec)

    with torch.no_grad():
        logits_c, logits_f = model(
            torch.tensor(seq_s),
            torch.tensor(pidx),
            torch.tensor(ine_s.reshape(1, -1)),
            torch.tensor(daily_s.reshape(1, -1)),
        )

    proba_c = torch.softmax(logits_c, dim=1).numpy()[0]
    proba_f = torch.softmax(logits_f, dim=1).numpy()[0]

    proba_c = _aplicar_factor_edad(proba_c, "calor", grupo_edad)
    proba_f = _aplicar_factor_edad(proba_f, "frio", grupo_edad)

    u_c = CLASS_THRESHOLDS_LSTM.get("calor", {"t1": 0.6, "t2": 0.55})
    u_f = CLASS_THRESHOLDS_LSTM.get("frio", {"t1": 0.4, "t2": 0.35})

    pred_c_th = int(apply_class_thresholds(proba_c[np.newaxis, :], **u_c)[0])
    pred_f_th = int(apply_class_thresholds(proba_f[np.newaxis, :], **u_f)[0])

    return {
        "calor": {
            "clase_argmax": int(proba_c.argmax()),
            "clase_threshold": pred_c_th,
            "probabilidades": proba_c.round(4).tolist(),
            "prob_riesgo": round(float(1.0 - proba_c[0]), 4),
        },
        "frio": {
            "clase_argmax": int(proba_f.argmax()),
            "clase_threshold": pred_f_th,
            "probabilidades": proba_f.round(4).tolist(),
            "prob_riesgo": round(float(1.0 - proba_f[0]), 4),
        },
    }


def _predecir_formulas(current: dict) -> dict:
    t = current.get("t2m_c", 20.0)
    rh = current.get("rh", 50.0)
    ws = current.get("wind_speed_kmh", 10.0)

    hi = heat_index(t, rh)
    wc = wind_chill(t, ws)

    hi_clase = 0
    if hi >= 39:
        hi_clase = 2
    elif hi >= 27:
        hi_clase = 1

    wc_clase = 0
    if wc <= -25:
        wc_clase = 2
    elif wc <= 0:
        wc_clase = 1

    return {
        "calor": {
            "clase": hi_clase,
            "heat_index_c": round(float(hi), 2),
            "categoria": ["seguro", "precaucion", "peligro", "peligro_extremo"][
                min(hi_clase, 3)
            ],
        },
        "frio": {
            "clase": wc_clase,
            "wind_chill_c": round(float(wc), 2),
        },
    }


def predict_ensemble(
    lat: float | None = None,
    lon: float | None = None,
    provincia: str = "Madrid",
    perfil: dict | None = None,
) -> dict:
    weather = fetch_weather_data(lat=lat, lon=lon, provincia=provincia)

    if perfil is None:
        perfil = {}

    df_features = weather["df_features"]
    df_hora = weather["df_hora"]

    # Determinar estrato por edad del usuario
    estrato = _edad_a_estrato(perfil.get("edad") if perfil else None)

    resultados = {}

    xgb_result = _predecir_tabular("XGBoost_calor.joblib", "calor", df_features, provincia, grupo_edad=estrato)
    resultados["XGBoost_calor"] = xgb_result

    rf_result = _predecir_tabular("RandomForest_frio.joblib", "frio", df_features, provincia, grupo_edad=estrato)
    resultados["RandomForest_frio"] = rf_result

    lstm_result = _predecir_lstm(df_hora, df_features, provincia, grupo_edad=estrato)
    resultados["LSTM"] = lstm_result

    formula_result = _predecir_formulas(weather["current"])
    resultados["Formula"] = formula_result

    todas_clases = []
    for key, res in resultados.items():
        if isinstance(res, dict) and "error" in res:
            continue
        if "calor" in res and isinstance(res["calor"], dict):
            c = res["calor"]
            todas_clases.append(c.get("clase_threshold") or c.get("clase", 0))
        if "frio" in res and isinstance(res["frio"], dict):
            c = res["frio"]
            todas_clases.append(c.get("clase_threshold") or c.get("clase", 0))
        if "clase_threshold" in res:
            todas_clases.append(res["clase_threshold"])

    clase_final = max(todas_clases) if todas_clases else 0

    perfil_aplicado = {}

    perfil_horario = None
    if df_hora is not None and "datetime" in df_hora.columns and "heat_index_c" in df_hora.columns:
        horas_agrupadas = {}
        for _, row in df_hora.iterrows():
            dt = pd.to_datetime(row["datetime"])
            hi = row.get("heat_index_c")
            if hi is not None and not (isinstance(hi, float) and np.isnan(hi)):
                hora = dt.hour
                if hora not in horas_agrupadas or float(hi) > horas_agrupadas[hora]:
                    horas_agrupadas[hora] = float(hi)
        if horas_agrupadas:
            perfil_horario = [{"hora": h, "HI": hi} for h, hi in sorted(horas_agrupadas.items())]
            if perfil:
                perfil["_perfil_horario"] = perfil_horario

    override_fisico = None
    formula_result = resultados.get("Formula", {})
    HI_current = formula_result.get("calor", {}).get("heat_index_c")
    WC = formula_result.get("frio", {}).get("wind_chill_c")
    UV = weather.get("uv_index")
    HI = HI_current
    if perfil_horario:
        HI = max(h["HI"] for h in perfil_horario)
    if (
        HI is not None and HI < 27
        and WC is not None and WC > 0
        and (UV is None or UV < 6)
        and clase_final > 0
    ):
        nueva_clase = clase_final
        razon_extra = ""
        if clase_final == 2:
            nueva_clase = 1
            razon_extra = " (PELIGRO→PRECAUCION: condiciones fisicas actuales seguras)"
        elif clase_final == 1:
            razon_extra = " (se mantiene PRECAUCION: modelos ML detectan tendencia de riesgo)"
        override_fisico = {
            "clase_ml": int(clase_final),
            "clase_final": int(nueva_clase),
            "razon": f"HI_peak={HI:.1f}C<27, WC={WC:.1f}C>0, UV<6{razon_extra}",
        }
        clase_final = nueva_clase

    def _personalizar_si_hay(prob_poblacional, tipo):
        if perfil and any(v is not None for v in perfil.values()):
            res_pers = personalizar_riesgo(prob_poblacional, perfil, tipo=tipo)
            return res_pers
        return {
            "indice_personalizado": prob_poblacional,
            "factor_total": 1.0,
            "producto_bruto": 1.0,
            "capado": False,
            "factores": [],
        }

    prob_calor = xgb_result.get("prob_riesgo", 0.5)
    prob_frio = rf_result.get("prob_riesgo", 0.5)

    res_calor = _personalizar_si_hay(prob_calor, "calor")
    res_frio = _personalizar_si_hay(prob_frio, "frio")

    perfil_aplicado["calor"] = {
        "prob_poblacional": prob_calor,
        "factor_total": res_calor["factor_total"],
        "producto_bruto": res_calor["producto_bruto"],
        "capado": res_calor["capado"],
        "prob_personalizada": res_calor["indice_personalizado"],
        "factores": res_calor["factores"],
    }
    perfil_aplicado["frio"] = {
        "prob_poblacional": prob_frio,
        "factor_total": res_frio["factor_total"],
        "producto_bruto": res_frio["producto_bruto"],
        "capado": res_frio["capado"],
        "prob_personalizada": res_frio["indice_personalizado"],
        "factores": res_frio["factores"],
    }

    weather_result = {
        "lat": weather["lat"],
        "lon": weather["lon"],
        "current": weather["current"],
        "uv_index": weather.get("uv_index"),
        "provincia": provincia,
        "df_hora": df_hora,
        "df_features": df_features,
    }

    if perfil and "_perfil_horario" in perfil:
        weather_result["perfil_horario"] = perfil["_perfil_horario"]

    X_calor = xgb_result.get("_X")
    X_frio = rf_result.get("_X")

    explicacion = explicar_ensemble(
        {
            "modelos": resultados,
            "clase_final": clase_final,
            "weather": weather_result,
        },
        X_calor=X_calor,
        X_frio=X_frio,
        perfil_usuario=perfil,
    )

    if override_fisico:
        risk_source = "tendencia meteorológica" if not perfil_horario else "actividad prevista"
        explicacion["modelo_determinante"] = f"Override — condiciones actuales seguras, riesgo por {risk_source}"
        explicacion["override"] = override_fisico

    recomendaciones = generar_recomendaciones(perfil, {
        "modelos": resultados,
        "clase_final": clase_final,
        "weather": weather_result,
    })

    for r in resultados.values():
        if isinstance(r, dict):
            r.pop("_X", None)

    return {
        "weather": weather_result,
        "modelos": resultados,
        "perfil": perfil_aplicado,
        "perfil_usuario": perfil,
        "clase_final": clase_final,
        "clase_final_label": CLASES[clase_final] if clase_final < len(CLASES) else "DESCONOCIDO",
        "explicacion": explicacion,
        "recomendaciones": recomendaciones,
        "override_fisico": override_fisico,
    }
