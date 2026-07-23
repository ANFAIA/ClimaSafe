import numpy as np
import pandas as pd
import joblib
import torch

from climasafeai.utils.paths import ARTIFACTS_DIR
from climasafeai.models.predict_model import (
    CLASS_THRESHOLDS_RECOMENDADOS,
    CLASS_THRESHOLDS_LSTM,
)
from climasafeai.models.lstm_province_hybrid import (
    LSTMProvinceHybridMultiTask,
)
from climasafeai.data.weather_fetcher import (
    build_sequence_24h,
    build_daily_feature_vector,
    get_province_idx,
    get_ine_features,
    escalar_para_lstm,
)

try:
    import shap
except ImportError:
    shap = None

try:
    from captum.attr import IntegratedGradients as CaptumIG
except ImportError:
    CaptumIG = None


_TOP_N = 5

_FEATURE_NAME_MAP = {
    "t2m_c": "Temperatura",
    "rh": "Humedad relativa",
    "wind_speed_kmh": "Velocidad viento",
    "sp": "Presión atmosférica",
    "heat_index_c": "HI actual (índice calor)",
    "wbgt_c": "WBGT actual",
    "wind_chill_c": "WC actual (sensación térmica)",
    "heat_index_mean": "HI medio diario",
    "heat_index_std": "Desviación HI diario",
    "heat_index_min": "HI mínimo diario",
    "horas_sobre_umbral": "Horas sobre umbral calor (hoy)",
    "wind_chill_mean": "WC medio diario",
    "wind_chill_std": "Desviación WC diario",
    "wind_chill_max": "WC máximo diario",
    "horas_bajo_umbral": "Horas bajo umbral frío (hoy)",
    "heat_index_c_lag1": "HI día anterior",
    "heat_index_c_roll3": "Media HI últimos 3 días",
    "heat_index_c_roll7": "Media HI últimos 7 días (tendencia)",
    "dias_consec_sobre_umbral": "Días consecutivos sobre umbral calor",
    "grados_dia_calor_roll7": "Grados-día calor acumulados (7d)",
    "grados_dia_calor_roll14": "Grados-día calor acumulados (14d)",
    "wind_chill_mean_roll3": "Media WC últimos 3 días",
    "wind_chill_mean_roll7": "Media WC últimos 7 días (tendencia)",
    "wind_chill_mean_roll14": "Media WC últimos 14 días",
    "grados_dia_frio_roll7": "Grados-día frío acumulados (7d)",
    "grados_dia_frio_roll14": "Grados-día frío acumulados (14d)",
    "dias_consec_bajo_umbral": "Días consecutivos bajo umbral frío",
    "t2m_min_noche_lag1": "Temp. mínima nocturna (anoche)",
    "t2m_min_noche_roll7": "Media temp. mínima nocturna (7d)",
    "dias_consec_wc_severo": "Días consecutivos WC severo",
    "horas_wc_severo_sum14": "Horas WC severo (últimos 14d)",
}


def _safe_shap_explainer(model, X: np.ndarray) -> np.ndarray | None:
    if shap is None:
        return None
    try:
        if hasattr(model, "estimators_"):
            explainer = shap.TreeExplainer(model)
        elif "xgboost" in type(model).__module__:
            explainer = shap.TreeExplainer(model)
        else:
            return None
        shap_values = explainer.shap_values(X)
        if isinstance(shap_values, list):
            if len(shap_values) == 3:
                shap_values = np.stack(shap_values).mean(axis=0)
            elif len(shap_values) == 2:
                shap_values = shap_values[1]
        if isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
            shap_values = shap_values.mean(axis=2)
        return shap_values
    except Exception:
        return None


def _top_features(shap_values: np.ndarray, feature_names: list, top_n: int = _TOP_N) -> list[dict]:
    shap_values = np.asarray(shap_values)
    if shap_values.ndim == 1:
        contrib = np.abs(shap_values)
        signo = shap_values
    else:
        contrib = np.abs(shap_values).mean(axis=0)
        signo = shap_values.mean(axis=0)
    indices = np.argsort(contrib)[::-1][:top_n]
    result = []
    for idx in indices:
        i = int(idx)
        name = feature_names[i]
        label = _FEATURE_NAME_MAP.get(name, name)
        result.append({
            "feature": label,
            "importancia": round(float(contrib[i]), 4),
            "direccion": "positiva" if float(signo[i]) > 0 else "negativa",
        })
    return result


def explicar_xgboost(
    X: np.ndarray,
    feature_names: list[str],
    model_path: str = "XGBoost_calor.joblib",
) -> dict:
    from climasafeai.utils.paths import MODELS_DIR
    path = MODELS_DIR / model_path
    if not path.exists():
        return {"error": f"Modelo no encontrado: {path}"}
    model = joblib.load(path)
    shap_vals = _safe_shap_explainer(model, X)
    if shap_vals is None:
        return {"error": "SHAP no disponible para XGBoost"}
    return {
        "modelo": model_path,
        "top_features": _top_features(shap_vals, feature_names),
    }


def explicar_randomforest(
    X: np.ndarray,
    feature_names: list[str],
    model_path: str = "RandomForest_frio.joblib",
) -> dict:
    from climasafeai.utils.paths import MODELS_DIR
    path = MODELS_DIR / model_path
    if not path.exists():
        return {"error": f"Modelo no encontrado: {path}"}
    model = joblib.load(path)
    shap_vals = _safe_shap_explainer(model, X)
    if shap_vals is None:
        return {"error": "SHAP no disponible para RandomForest"}
    return {
        "modelo": model_path,
        "top_features": _top_features(shap_vals, feature_names),
    }


def explicar_lstm(
    df_hora,
    df_features,
    provincia: str = "Madrid",
) -> dict:
    from climasafeai.models.lstm_province_hybrid import (
        load_lstm_province_hybrid,
        LSTM_PROVINCE_HYBRID_MODEL_PATH,
    )

    try:
        model = load_lstm_province_hybrid(LSTM_PROVINCE_HYBRID_MODEL_PATH, device="cpu")
    except Exception as e:
        return {"error": f"No se pudo cargar LSTM: {e}"}

    seq = build_sequence_24h(df_hora)
    if seq is None:
        return {"error": "No hay datos horarios para LSTM"}
    daily_vec = build_daily_feature_vector(df_features)
    if daily_vec is None:
        return {"error": "No se pudieron generar features diarias"}
    ine_vec = get_ine_features(provincia)
    pidx = np.array([get_province_idx(provincia)], dtype=np.int64)
    seq_s, ine_s, daily_s = escalar_para_lstm(seq, ine_vec, daily_vec)

    seq_t = torch.tensor(seq_s, requires_grad=True)
    pidx_t = torch.tensor(pidx)
    ine_t = torch.tensor(ine_s.reshape(1, -1))
    daily_t = torch.tensor(daily_s.reshape(1, -1))

    if CaptumIG is not None:
        try:
            ig = CaptumIG(lambda x: model(x, pidx_t, ine_t, daily_t))
            baseline = torch.zeros_like(seq_t)
            attr = ig.attribute(seq_t, baselines=baseline, target=0)
            attr_map = attr.squeeze(0).detach().numpy()
            abs_attr = np.abs(attr_map).mean(axis=1)
            hora_pico = int(np.argmax(abs_attr))
            from climasafeai.data.sequences import FEATURE_COLS_SEQ
            contribuciones = []
            for h in np.argsort(abs_attr)[::-1][:_TOP_N]:
                contribuciones.append({
                    "hora": int(h),
                    "importancia": float(round(abs_attr[h], 4)),
                })
            feature_contrib = []
            for f in range(attr_map.shape[1]):
                val = float(np.abs(attr_map[:, f]).mean())
                name = FEATURE_COLS_SEQ[f] if f < len(FEATURE_COLS_SEQ) else f"var_{f}"
                feature_contrib.append({"feature": name, "importancia": round(val, 4)})
            feature_contrib.sort(key=lambda x: x["importancia"], reverse=True)
            return {
                "metodo": "Integrated Gradients (captum)",
                "hora_mas_influyente": int(hora_pico),
                "horas_top": contribuciones[:_TOP_N],
                "variables_top": feature_contrib[:_TOP_N],
            }
        except Exception as e:
            pass

    return {"metodo": "proxy no disponible", "error": "No fue posible explicar LSTM directamente"}


def _hi_en_ventana_actividad(perfil_horario: list[dict] | None, perfil: dict | None) -> dict | None:
    if not perfil_horario or not perfil:
        return None
    h_ini = perfil.get("hora_inicio")
    dur = perfil.get("duracion_actividad_h")
    if h_ini is None or dur is None:
        return None
    fin = h_ini + dur
    window = [e for e in perfil_horario if h_ini <= e["hora"] < fin]
    if not window:
        return None
    his = [e["HI"] for e in window]
    return {
        "hora_inicio": int(h_ini),
        "hora_fin": int(fin),
        "hi_min": round(min(his), 1),
        "hi_max": round(max(his), 1),
        "hi_peak": round(max(his), 1),
    }


def explicar_formula(
    current: dict,
    formula_result: dict,
    perfil_horario: list[dict] | None = None,
    perfil: dict | None = None,
    clase_final: int | None = None,
) -> dict:
    explicaciones = []
    calor = formula_result.get("calor", {})
    frio = formula_result.get("frio", {})
    hi = calor.get("heat_index_c")
    wc = frio.get("wind_chill_c")

    ventana = _hi_en_ventana_actividad(perfil_horario, perfil)
    if ventana and ventana["hi_peak"] >= 27:
        if clase_final and clase_final > 0:
            explicaciones.append(
                f"Condiciones actuales seguras (HI={hi}C), pero durante la actividad "
                f"prevista ({ventana['hora_inicio']}:00-{ventana['hora_fin']}:00) se esperan "
                f"HI entre {ventana['hi_min']} y {ventana['hi_max']}C, "
                f"lo que justifica el nivel de riesgo."
            )
        else:
            explicaciones.append(
                f"HI alcanzara {ventana['hi_peak']}C durante la actividad "
                f"({ventana['hora_inicio']}:00-{ventana['hora_fin']}:00), "
                f"pero tu perfil personalizado indica riesgo bajo."
            )
    else:
        if hi is not None:
            if hi >= 39:
                explicaciones.append(f"Heat Index de {hi}C supera el umbral de peligro (39C). Riesgo clinico de golpe de calor con exposicion prolongada.")
            elif hi >= 32:
                explicaciones.append(f"Heat Index de {hi}C esta en rango de precaucion extrema (32-39C). Posibles calambres y agotamiento por calor.")
            elif hi >= 27:
                explicaciones.append(f"Heat Index de {hi}C esta en rango de precaucion (27-32C). Fatiga posible con exposicion prolongada.")
            else:
                explicaciones.append(f"Heat Index de {hi}C esta por debajo del umbral de riesgo (27C).")

    if wc is not None:
        if wc <= -25:
            explicaciones.append(f"Wind Chill de {wc}C esta por debajo del umbral de peligro (-25C). Riesgo de congelacion en menos de 30 minutos.")
        elif wc <= 0:
            explicaciones.append(f"Wind Chill de {wc}C esta en rango de riesgo. Sensacion termica bajo cero.")

    return {
        "explicaciones": explicaciones,
        "detalle": {
            "heat_index_c": hi,
            "wind_chill_c": wc,
        },
    }


def _obtener_feature_names(clase: str) -> list[str]:
    path = ARTIFACTS_DIR / f"feature_names_{clase}.joblib"
    if path.exists():
        return joblib.load(path)
    return []


def explicar_ensemble(
    resultado: dict,
    X_calor: np.ndarray | None = None,
    X_frio: np.ndarray | None = None,
    perfil_usuario: dict | None = None,
) -> dict:
    modelos = resultado.get("modelos", {})
    weather = resultado.get("weather", {})
    current = weather.get("current", {})
    df_hora = weather.get("df_hora")
    df_features = weather.get("df_features")

    explicaciones = {}

    if X_calor is not None:
        feat_calor = _obtener_feature_names("calor")
        if feat_calor:
            explicaciones["XGBoost_calor"] = explicar_xgboost(X_calor, feat_calor)

    if X_frio is not None:
        feat_frio = _obtener_feature_names("frio")
        if feat_frio:
            explicaciones["RandomForest_frio"] = explicar_randomforest(X_frio, feat_frio)

    if "LSTM" in modelos and "error" not in modelos.get("LSTM", {}):
        lstm_ex = explicar_lstm(df_hora, df_features, weather.get("provincia", "Madrid"))
        explicaciones["LSTM"] = lstm_ex

    if "Formula" in modelos:
        explicaciones["Formula"] = explicar_formula(
            current, modelos.get("Formula", {}),
            perfil_horario=weather.get("perfil_horario"),
            perfil=perfil_usuario,
            clase_final=resultado.get("clase_final", 0),
        )

    clase_final = resultado.get("clase_final", 0)
    if clase_final > 0:
        try:
            modelos = resultado.get("modelos", {})
            calor_clase = max(
                modelos.get("XGBoost_calor", {}).get("clase_threshold", 0),
                modelos.get("LSTM", {}).get("calor", {}).get("clase_threshold", 0),
                modelos.get("Formula", {}).get("calor", {}).get("clase", 0),
            )
            frio_clase = max(
                modelos.get("RandomForest_frio", {}).get("clase_threshold", 0),
                modelos.get("LSTM", {}).get("frio", {}).get("clase_threshold", 0),
                modelos.get("Formula", {}).get("frio", {}).get("clase", 0),
            )
            clase_bayes = "calor" if calor_clase >= frio_clase else "frio"

            from climasafeai.models.bayes import BayesianRiskDiagnosis
            _BAYES_PATH = str(ARTIFACTS_DIR / f"bayes_risk_diagnosis_{clase_bayes}.joblib")
            bd = BayesianRiskDiagnosis(clase=clase_bayes)
            bd.load(_BAYES_PATH)
            bayes_diag = bd.diagnosis_inverso(int(clase_final))
            explicaciones["Bayes"] = bayes_diag
            if perfil_usuario:
                temp = current.get("temp") or current.get("t2m_c")
                grasa = perfil_usuario.get("grasa_corporal") or perfil_usuario.get("grasa")
                edad = perfil_usuario.get("edad")
                if temp is not None:
                    cf = bd.contrafactual(temp, grasa or 20.0, edad or 40)
                    explicaciones["Bayes"]["contrafactuales"] = cf["escenarios"]
        except Exception:
            pass

    modelo_determinante = _modelo_mas_restrictivo(resultado)
    return {
        "modelo_determinante": modelo_determinante,
        "detalles": explicaciones,
    }


def _modelo_mas_restrictivo(resultado: dict) -> str:
    modelos = resultado.get("modelos", {})
    clase_final = resultado.get("clase_final", 0)

    ml_match: str | None = None
    lstm_match = False

    for nombre, res in modelos.items():
        if isinstance(res, dict) and "error" in res:
            continue
        if nombre == "LSTM":
            c = res.get("calor", {}).get("clase_threshold") or res.get("frio", {}).get("clase_threshold", 0)
            if c == clase_final:
                lstm_match = True
        elif nombre == "Formula":
            continue
        else:
            c = res.get("clase_threshold", 0)
            if c == clase_final:
                ml_match = nombre

    _ETIQUETAS = {
        "XGBoost_calor": "XGBoost",
        "RandomForest_frio": "RandomForest",
        "LSTM": "LSTM híbrida",
    }

    if clase_final == 0 and ml_match:
        return _ETIQUETAS.get(ml_match, ml_match)
    if clase_final == 0 and lstm_match:
        return _ETIQUETAS.get("LSTM")
    if ml_match:
        return _ETIQUETAS.get(ml_match, ml_match)
    if lstm_match:
        return _ETIQUETAS.get("LSTM")
    return "múltiples"


# ──────────────────────────────────────────────
# Contrafactuales
# ──────────────────────────────────────────────

PERS_THRESHOLD_PELIGRO = 0.55

CAMBIOS_ACCIONABLES = [
    {
        "id": "nivel_actividad",
        "descripcion": "Reducir la intensidad del esfuerzo",
        "condicion": lambda p: p.get("nivel_actividad") in ("intensa", "muy_intensa", "moderada", "ligera"),
        "aplicar": lambda p: _cambiar(p, "nivel_actividad",
            {"muy_intensa": "intensa", "intensa": "moderada", "moderada": "ligera", "ligera": "reposo"}
            .get(p.get("nivel_actividad", ""), p["nivel_actividad"])),
    },
    {
        "id": "aclimatado",
        "descripcion": "Aclimatarse al calor (14 días de exposición gradual)",
        "condicion": lambda p: p.get("aclimatado") is False,
        "aplicar": lambda p: _cambiar(p, "aclimatado", True),
    },
    {
        "id": "falta_sueno",
        "descripcion": "Descansar adecuadamente la noche anterior",
        "condicion": lambda p: p.get("falta_sueno") is True,
        "aplicar": lambda p: _cambiar(p, "falta_sueno", False),
    },
    {
        "id": "enfermedad_reciente",
        "descripcion": "Esperar a recuperarse completamente de la enfermedad",
        "condicion": lambda p: p.get("enfermedad_reciente") is True,
        "aplicar": lambda p: _cambiar(p, "enfermedad_reciente", False),
    },
    {
        "id": "hora_inicio",
        "descripcion": "Realizar la actividad en horas más frescas",
        "condicion": lambda p: _hora_en_peak(p.get("hora_inicio")),
        "aplicar": lambda p: _cambiar(p, "hora_inicio",
            8 if p.get("hora_inicio", 12) >= 12 else 19),
    },
    {
        "id": "duracion_actividad",
        "descripcion": "Acortar la duración de la actividad",
        "condicion": lambda p: (p.get("duracion_actividad_h") or 0) > 1,
        "aplicar": lambda p: _cambiar(p, "duracion_actividad_h", 1.0),
    },
]


def _cambiar(perfil: dict, clave: str, valor) -> dict:
    copia = dict(perfil)
    copia[clave] = valor
    return copia


def _hora_en_peak(hora: float | None) -> bool:
    if hora is None:
        return False
    return 12 <= hora <= 17


def _clase_desde_prob(prob: float) -> int:
    umbral = CLASS_THRESHOLDS_RECOMENDADOS.get("calor", {"t1": 0.25})
    if prob >= PERS_THRESHOLD_PELIGRO:
        return 2
    if prob >= umbral["t1"]:
        return 1
    return 0


def _factor_total_desde_original(perfil_aplicado: dict, tipo: str = "calor") -> float:
    """Reconstruye el factor_total que se aplicó originalmente."""
    p = (perfil_aplicado.get(tipo) or {})
    return p.get("factor_total", p.get("producto_bruto", 1.0))


def _pers_pair(
    prob_calor: float, prob_frio: float, perfil: dict, cap: float
) -> tuple[float, float]:
    from climasafeai.features.personalizacion import personalizar_riesgo
    rc = personalizar_riesgo(prob_calor, perfil, tipo="calor", cap_factores=cap)
    rf = personalizar_riesgo(prob_frio, perfil, tipo="frio", cap_factores=cap)
    prob = max(rc["indice_personalizado"], rf["indice_personalizado"])
    ft = rc["factor_total"]
    return prob, ft


def _pers_pair_nocap(prob_calor, prob_frio, perfil):
    return _pers_pair(prob_calor, prob_frio, perfil, 10.0)


def generar_contrafactuales(result: dict) -> list[dict]:
    perfil: dict = result.get("perfil_usuario") or {}
    perfil_aplicado: dict = result.get("perfil", {})
    prob_calor = (perfil_aplicado.get("calor") or {}).get("prob_poblacional", 0.5)
    prob_frio = (perfil_aplicado.get("frio") or {}).get("prob_poblacional", 0.5)

    prob_pers_orig_cap, ft_orig_cap = _pers_pair(prob_calor, prob_frio, perfil, 3.0)
    prob_pers_orig_nocap, ft_orig_nocap = _pers_pair_nocap(prob_calor, prob_frio, perfil)
    clase_orig = _clase_desde_prob(prob_pers_orig_nocap)

    contrafactuales = []

    for cambio in CAMBIOS_ACCIONABLES:
        if not cambio["condicion"](perfil):
            continue

        perfil_mod = cambio["aplicar"](perfil)

        prob_nueva_cap, ft_nueva_cap = _pers_pair(prob_calor, prob_frio, perfil_mod, 3.0)
        prob_nueva_nocap, _ = _pers_pair_nocap(prob_calor, prob_frio, perfil_mod)

        diff = prob_pers_orig_nocap - prob_nueva_nocap
        if diff <= 0.001:
            continue

        contrafactuales.append({
            "id": cambio["id"],
            "descripcion": cambio["descripcion"],
            "probabilidad_sin_cap": {
                "actual": round(prob_pers_orig_nocap, 4),
                "tras_cambio": round(prob_nueva_nocap, 4),
            },
            "probabilidad_con_cap": {
                "actual": round(prob_pers_orig_cap, 4),
                "tras_cambio": round(prob_nueva_cap, 4),
            },
            "clase_actual": _CLASE_LABEL[clase_orig],
            "clase_tras_cambio": _CLASE_LABEL[_clase_desde_prob(prob_nueva_nocap)],
            "mejora_porcentual_puntos": round(diff * 100, 1),
            "enmascarado_por_cap": ft_orig_cap >= 2.99 and ft_nueva_cap >= 2.99,
            "factor_total_actual": round(ft_orig_nocap, 3),
            "factor_total_tras_cambio": round(ft_nueva_cap, 3),
            "producto_bruto_tras_cambio": None,  # filled below
        })

        # Get producto_bruto for informational purposes
        from climasafeai.features.personalizacion import personalizar_riesgo
        r_full = personalizar_riesgo(prob_calor, perfil_mod, tipo="calor", cap_factores=10.0)
        contrafactuales[-1]["producto_bruto_tras_cambio"] = round(r_full["producto_bruto"], 3)

    contrafactuales.sort(key=lambda x: x["mejora_porcentual_puntos"], reverse=True)
    return contrafactuales


_CLASE_LABEL = ["SEGURO", "PRECAUCION", "PELIGRO"]
