import joblib
import numpy as np
import pandas as pd

try:
    import torch
    _TORCH_AVAILABLE = True
except ModuleNotFoundError:
    torch = None
    _TORCH_AVAILABLE = False

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
    FORECAST_HORIZON_DAYS,
    ForecastHorizonError,
)
from climasafeai.utils.paths import MODELS_DIR, ARTIFACTS_DIR
from climasafeai.models.explicabilidad import explicar_ensemble
from climasafeai.models.recomendaciones import generar_recomendaciones
from climasafeai.models.registry import discover_models

# Cache de modelos para no recargar en cada predicción
_MODEL_CACHE: dict[str, object] = {}


def _cargar_modelo(path: str) -> object:
    if path not in _MODEL_CACHE:
        _MODEL_CACHE[path] = joblib.load(MODELS_DIR / path)
    return _MODEL_CACHE[path]

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
    model = _cargar_modelo(model_path)
    df_input = df_features.copy()
    if provincia:
        df_input["provincia"] = provincia
    X = process_input(df_input, clase=clase)

    proba = model.predict_proba(X)

    # Calibración isotónica post-hoc (solo frío, en calor reduce sensibilidad en provincias frías)
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
    try:
        estrato_path = ARTIFACTS_DIR / "params_estrato.joblib"
        if estrato_path.exists():
            params_estrato = joblib.load(estrato_path)
            estrato_u = params_estrato.get(clase, {}).get(grupo_edad, u_global)
            u = {"t1": estrato_u["t1"], "t2": estrato_u["t2"]}
    except Exception:
        pass
    if provincia:
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

    conformal_conf = None
    conformal_set_size = 2
    try:
        from climasafeai.models.conformal import SplitConformalCalibrator, confidence_label
        _conformal_path = ARTIFACTS_DIR / f"conformal_{clase}.joblib"
        if _conformal_path.exists():
            cal = SplitConformalCalibrator()
            cal.load(str(_conformal_path))
            conf, sizes = cal.confidence(proba)
            conformal_conf = confidence_label(int(sizes[0]))
            conformal_set_size = int(sizes[0])
    except Exception:
        pass

    return {
        "clase_argmax": pred_argmax,
        "clase_threshold": pred_th,
        "probabilidades": proba[0].round(4).tolist(),
        "prob_riesgo": round(prob_riesgo, 4),
        "thresholds_usados": u,
        "conformal_confianza": conformal_conf,
        "conformal_set_size": conformal_set_size,
        "_X": X,
    }


_LSTM_CACHE: object | None = None


def _cargar_lstm():
    global _LSTM_CACHE
    if _LSTM_CACHE is None:
        _LSTM_CACHE = load_lstm_province_hybrid(LSTM_PROVINCE_HYBRID_MODEL_PATH, device="cpu")
    return _LSTM_CACHE


def _predecir_lstm(
    df_hora: pd.DataFrame,
    df_features: pd.DataFrame,
    provincia: str | None = None,
    grupo_edad: str = "todos",
) -> dict:
    if not _TORCH_AVAILABLE:
        return {"error": "torch no está instalado (pip install '.[redes_neuronales]' o 'uv sync --group redes_neuronales')"}

    try:
        model = _cargar_lstm()
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


from datetime import date as date_type  # noqa: E402


# Umbral de PELIGRO sobre la probabilidad personalizada. Más exigente que el t2
# del ML porque prob_pers es P(riesgo)=P(1)+P(2), no P(peligro)=P(2).
PERS_THRESHOLD_PELIGRO = 0.55


def perfil_horario_desde_df(df_hora, target_date=None, res_min: int = 60) -> list[dict] | None:
    """Perfil horario ``[{"hora", "HI", "temp"}, ...]`` del día objetivo.

    Se queda con el HI máximo de cada hora del DÍA OBJETIVO (hoy o
    ``target_date``), no del máximo por hora de todos los días que traiga el
    df. El df_hora concatenado (14 días de histórico + el día objetivo) solo
    debe usarse entero para el LSTM y las features; el perfil del usuario se
    filtra por fecha ANTES de agrupar (bug DATA-003: sin filtrar mezclaba días
    y proyectaba picos de días pasados). Si ``target_date`` no se pasa, se usa
    la última fecha presente en el df. Extraído de ``predict_ensemble`` para
    que otros endpoints puedan construir el mismo perfil sin ejecutar el
    ensemble entero.

    ``res_min`` — resolución del perfil en minutos (5, 15, 30 o 60; por defecto
    60). Con 60 devuelve exactamente el perfil horario histórico: un punto por
    hora con el HI máximo de esa hora. Con menos, añade los puntos intermedios
    por interpolación LINEAL entre los máximos horarios consecutivos; la fuente
    (Open-Meteo forecast/archive) solo publica datos horarios, no hay fuente
    sub-horaria que consultar (DATA-004). La marca :00 de cada hora conserva el
    máximo horario; los puntos :15/:30/:45 salen de la interpolación y la
    última hora del día se mantiene plana (no hay hora siguiente a la que
    interpolar). El campo ``hora`` no se redondea a propósito (la resolución de
    5 min no es decimal exacta y la ventana deslizante necesita el paso
    exacto); ``HI`` y ``temp`` sí se redondean como el resto del perfil.
    """
    if df_hora is None or "datetime" not in df_hora.columns or "heat_index_c" not in df_hora.columns:
        return None
    if res_min not in (5, 15, 30, 60):
        raise ValueError(f"res_min debe ser 5, 15, 30 o 60, no {res_min!r}")

    if target_date is not None:
        dia_objetivo = pd.to_datetime(target_date).date()
    else:
        dia_objetivo = pd.to_datetime(df_hora["datetime"]).dt.date.max()
    df_hora = df_hora[pd.to_datetime(df_hora["datetime"]).dt.date == dia_objetivo]

    horas_agrupadas = {}
    temp_por_hora = {}
    for _, row in df_hora.iterrows():
        dt = pd.to_datetime(row["datetime"])
        hi = row.get("heat_index_c")
        if hi is not None and not (isinstance(hi, float) and np.isnan(hi)):
            hora = dt.hour
            if hora not in horas_agrupadas or float(hi) > horas_agrupadas[hora]:
                horas_agrupadas[hora] = float(hi)
                t = row.get("t2m_c")
                if t is not None and not (isinstance(t, float) and np.isnan(t)):
                    temp_por_hora[hora] = round(float(t), 1)

    if not horas_agrupadas:
        return None
    horas = sorted(horas_agrupadas)
    if res_min == 60:
        return [
            {"hora": h, "HI": horas_agrupadas[h], "temp": temp_por_hora.get(h)}
            for h in horas
        ]

    paso = res_min / 60.0
    n_intermedios = int(60 / res_min)
    perfil = []
    for i, h in enumerate(horas):
        hi = horas_agrupadas[h]
        t = temp_por_hora.get(h)
        perfil.append({"hora": float(h), "HI": hi, "temp": t})
        if i + 1 < len(horas):
            h_next = horas[i + 1]
            hi_next = horas_agrupadas[h_next]
            t_next = temp_por_hora.get(h_next)
        else:
            h_next = h  # última hora del día: se mantiene plana
            hi_next = hi
            t_next = t
        span = h_next - h
        for q in range(1, n_intermedios):
            if span > 0:
                frac = q * paso / span
                hi_q = hi + frac * (hi_next - hi)
                t_q = t + frac * (t_next - t) if t is not None and t_next is not None else None
            else:
                # Última hora del día: sin hora siguiente, se mantiene plana.
                hi_q = hi
                t_q = t
            perfil.append({
                "hora": h + q * paso,
                "HI": round(hi_q, 4),
                "temp": round(t_q, 4) if t_q is not None else None,
            })
    return perfil


def _proba_from_formula(current: dict) -> dict:
    """Convierte salida de la Fórmula a probabilidad de riesgo.

    Añade prob_riesgo a los campos que ya devuelve _predecir_formulas,
    manteniendo compatibilidad hacia atrás.

    Si un dato meteorológico llega NaN (fetch corrupto), se sustituye por su
    default seguro para que heat_index/wind_chill no propaguen NaN.
    """
    def _finito(v, default):
        try:
            f = float(v)
        except (TypeError, ValueError):
            return default
        return f if np.isfinite(f) else default

    t = _finito(current.get("t2m_c"), 20.0)
    rh = _finito(current.get("rh"), 50.0)
    ws = _finito(current.get("wind_speed_kmh"), 10.0)
    hi = heat_index(t, rh)
    wc = wind_chill(t, ws)

    # Calor: HI -> prob_riesgo
    if hi >= 39:
        prob_calor = 0.95
    elif hi >= 32:
        prob_calor = 0.60
    elif hi >= 27:
        prob_calor = 0.35
    else:
        prob_calor = 0.05 + (hi / 27.0) * 0.20

    # Frío: WC -> prob_riesgo
    if wc <= -25:
        prob_frio = 0.95
    elif wc <= -10:
        prob_frio = 0.55
    elif wc <= 0:
        prob_frio = 0.30
    else:
        prob_frio = 0.05

    # Clase según thresholds existentes
    hi_clase = 2 if hi >= 39 else (1 if hi >= 27 else 0)
    wc_clase = 2 if wc <= -25 else (1 if wc <= 0 else 0)

    return {
        "calor": {
            "prob_riesgo": round(min(prob_calor, 1.0), 4),
            "clase": hi_clase,
            "heat_index_c": round(float(hi), 2),
            "categoria": ["seguro", "precaucion", "peligro", "peligro_extremo"][
                min(hi_clase, 3)
            ],
        },
        "frio": {
            "prob_riesgo": round(min(prob_frio, 1.0), 4),
            "clase": wc_clase,
            "wind_chill_c": round(float(wc), 2),
        },
    }


def _conformal_weighted_ensemble(model_results: dict, tipo: str) -> dict:
    """Media ponderada por set_size conformal de los modelos del ensemble.

    Contribuyen los modelos que produjeron resultado en ESTA llamada
    (claves de ``model_results``); el registry aporta solo metadatos (tipo y
    clase) si el modelo tiene manifiesto. Así funciona igual con manifiestos,
    sin ellos o con diccionarios sintéticos (tests unitarios).

    Parameters
    ----------
    model_results : dict
        Resultados de todos los modelos descubiertos por manifiesto.
    tipo : str
        "calor" o "frio".

    Returns
    -------
    dict con "prob_riesgo" (float) y "clase" (int).
    """
    specs = {m["name"]: m for m in discover_models()}

    prob_sum = 0.0
    weight_sum = 0.0

    for key, res in model_results.items():
        if not isinstance(res, dict) or "error" in res:
            continue

        spec = specs.get(key)
        if spec and spec["class"] not in (tipo, "both"):
            continue

        # Tipo desde el manifiesto; sin él se infiere por forma: el tabular
        # trae prob_riesgo arriba, lstm/formula traen subdict por tipo (y
        # ambas pesan 1/2, así que basta distinguirlas del tabular).
        mtype = spec["type"] if spec else (
            "tabular" if "prob_riesgo" in res else "lstm"
        )

        if mtype == "lstm":
            sub = res.get(tipo, {})
            if not sub or not isinstance(sub, dict) or "error" in sub:
                continue
            prob = sub.get("prob_riesgo")
            if prob is None:
                continue
            # LSTM no tiene conformal → set_size por defecto = 2
            weight = 1.0 / 2.0
        elif mtype == "formula":
            sub = res.get(tipo, {})
            if not sub or not isinstance(sub, dict):
                continue
            prob = sub.get("prob_riesgo")
            if prob is None:
                continue
            # Fórmula no tiene conformal → set_size por defecto = 2
            weight = 1.0 / 2.0
        else:
            # Tabular models — tienen conformal_set_size
            prob = res.get("prob_riesgo")
            if prob is None:
                continue
            set_size = res.get("conformal_set_size", 2)
            if set_size is None or set_size <= 0:
                set_size = 2
            weight = 1.0 / set_size

        # Garantía del invariante: solo se acumula un prob_riesgo finito en
        # [0, 1]. Un NaN/inf (dato corrupto) descarta el modelo igual que un
        # "error", para que el clico no propague NaN al personalizar.
        if not isinstance(prob, (int, float)) or not np.isfinite(float(prob)):
            continue
        prob = float(prob)
        prob_sum += prob * weight
        weight_sum += weight

    if weight_sum <= 0:
        return {"prob_riesgo": 0.0, "clase": 0}

    prob_ens = prob_sum / weight_sum
    prob_ens = min(max(prob_ens, 0.0), 1.0)

    # Clase del ensemble sobre prob RAW (no personalizada).
    # Usamos el mismo umbral que personalización PERS_THRESHOLD_PELIGRO para
    # que el ensemble refleje el riesgo poblacional con el mismo criterio.
    thresholds = CLASS_THRESHOLDS_RECOMENDADOS.get(tipo, {"t1": 0.25})
    if prob_ens >= PERS_THRESHOLD_PELIGRO:
        clase = 2
    elif prob_ens >= thresholds["t1"]:
        clase = 1
    else:
        clase = 0

    return {"prob_riesgo": round(prob_ens, 4), "clase": clase}


def predict_ensemble(
    lat: float | None = None,
    lon: float | None = None,
    provincia: str = "Madrid",
    perfil: dict | None = None,
    target_date: date_type | None = None,
    weather: dict | None = None,
    resolucion: int = 60,
) -> dict:
    """Predice el riesgo cardiovascular de una persona (ensemble conformal).

    ``resolucion`` — resolución del perfil horario en minutos por punto (5, 15,
    30 o 60; por defecto 60). Con 60 la salida es exactamente la de siempre:
    un punto por hora. Valores menores interpolan el HI entre horas (ver
    ``perfil_horario_desde_df``); el resto del contrato de salida no cambia.
    """
    if weather is None:
        weather = fetch_weather_data(lat=lat, lon=lon, provincia=provincia, target_date=target_date)

    if perfil is None:
        perfil = {}

    df_features = weather["df_features"]
    df_hora = weather["df_hora"]

    # Determinar estrato por edad del usuario
    estrato = _edad_a_estrato(perfil.get("edad") if perfil else None)

    resultados = {}

    # ── Model discovery via manifests ──────────────────────────────────────
    # Instead of hardcoding model paths, scan for *.manifest.json files.
    # Each manifest declares name, type, class and file.  The prediction
    # backend is chosen by type (tabular → _predecir_tabular, lstm →
    # _predecir_lstm, formula → _proba_from_formula).
    discovered = discover_models()
    for model_spec in discovered:
        name = model_spec["name"]
        mtype = model_spec["type"]

        try:
            if mtype == "tabular":
                model_class = model_spec["class"]
                model_file = model_spec["file"]
                resultados[name] = _predecir_tabular(
                    model_file, model_class, df_features, provincia,
                    grupo_edad=estrato,
                )
            elif mtype == "lstm":
                resultados[name] = _predecir_lstm(
                    df_hora, df_features, provincia, grupo_edad=estrato,
                )
            elif mtype == "formula":
                resultados[name] = _proba_from_formula(weather["current"])
        except Exception:
            # If a model fails, skip it — the ensemble tolerates missing models.
            pass

    # Backward compat: without any usable ML model (tabular/lstm) this is not
    # an ensemble — raise like the pre-manifest code did when the first
    # _predecir_tabular failed. A lone Formula (or an {"error": ...} stub,
    # same convention as _conformal_weighted_ensemble) must not replace ML.
    ml_names = {m["name"] for m in discovered if m["type"] in ("tabular", "lstm")}
    hay_ml = any(
        isinstance(resultados.get(n), dict)
        and resultados[n]
        and "error" not in resultados[n]
        for n in ml_names
    )
    if ml_names and not hay_ml:
        raise RuntimeError(
            f"Ningún modelo ML del ensemble pudo ejecutarse (descubiertos: {sorted(ml_names)})"
        )

    # Fallback: if no manifests found, use the hardcoded defaults so the
    # system works even without any manifest files (backward compat).
    if not resultados:
        xgb_result = _predecir_tabular(
            "XGBoost_calor.joblib", "calor", df_features, provincia,
            grupo_edad=estrato,
        )
        resultados["XGBoost_calor"] = xgb_result

        rf_result = _predecir_tabular(
            "RandomForest_frio.joblib", "frio", df_features, provincia,
            grupo_edad=estrato,
        )
        resultados["RandomForest_frio"] = rf_result

        lstm_result = _predecir_lstm(df_hora, df_features, provincia, grupo_edad=estrato)
        resultados["LSTM"] = lstm_result

        formula_result = _proba_from_formula(weather["current"])
        resultados["Formula"] = formula_result

    # Ensemble conformal-weighted: media ponderada por set_size
    ens_calor = _conformal_weighted_ensemble(resultados, "calor")
    ens_frio = _conformal_weighted_ensemble(resultados, "frio")

    # Clase del ensemble (para clase_ml_original / explicación)
    clase_ml_original = max(ens_calor["clase"], ens_frio["clase"])

    perfil_aplicado = {}

    perfil_horario = perfil_horario_desde_df(
        df_hora, target_date=target_date or weather.get("target_date"),
        res_min=resolucion,
    )
    if perfil_horario and perfil:
        perfil["_perfil_horario"] = perfil_horario

    override_fisico = None
    formula_result = resultados.get("Formula", {})
    HI_current = formula_result.get("calor", {}).get("heat_index_c")
    WC = formula_result.get("frio", {}).get("wind_chill_c")
    UV = weather.get("uv_index")
    HI = HI_current
    if perfil_horario:
        # HI_peak: si el usuario especificó hora_inicio+duración, solo considerar esa ventana
        inicio = perfil.get("hora_inicio")
        duracion = perfil.get("duracion_actividad_h")
        if inicio is not None and duracion is not None:
            fin = inicio + duracion
            ventana = [h for h in perfil_horario if inicio <= h["hora"] < fin]
            if ventana:
                HI = max(h["HI"] for h in ventana)
            else:
                HI = max(h["HI"] for h in perfil_horario)
        else:
            HI = max(h["HI"] for h in perfil_horario)

    def _personalizar_si_hay(prob_poblacional, tipo):
        # Defensa final: si la prob poblacional no es finita (dato corrupto que
        # hubiera sobrevivido al ensemble), se trata como dato ausente → fallback
        # neutro (SEGURO) en vez de croncar personalizar_riesgo con NaN.
        if not isinstance(prob_poblacional, (int, float)) or not np.isfinite(float(prob_poblacional)):
            prob_poblacional = 0.0
        perfil_uv = dict(perfil) if perfil else {}
        uv = weather.get("uv_index")
        if uv is not None:
            perfil_uv["_uv_index"] = uv
        _current = weather.get("current", {})
        ws = _current.get("wind_speed_kmh")
        if ws is not None:
            perfil_uv["_wind_speed_kmh"] = ws
        rh = _current.get("rh")
        if rh is not None:
            perfil_uv["_rh"] = rh
        if perfil_uv and any(v is not None for v in perfil_uv.values()):
            res_pers = personalizar_riesgo(prob_poblacional, perfil_uv, tipo=tipo)
            return res_pers
        return {
            "indice_personalizado": prob_poblacional,
            "factor_total": 1.0,
            "producto_bruto": 1.0,
            "capado": False,
            "factores": [],
        }

    prob_calor = ens_calor["prob_riesgo"]
    prob_frio = ens_frio["prob_riesgo"]

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

    # Clase desde probabilidad personalizada
    # t1 (PRECAUCION) alineado con el ML (CLASS_THRESHOLDS_RECOMENDADOS.calor.t1).
    # t2 (PELIGRO) es más exigente porque prob_pers es P(riesgo) = P(1)+P(2),
    # no P(peligro)=P(2) del ML. Usamos un umbral propio y fijo.
    prob_pers = max(
        res_calor["indice_personalizado"],
        res_frio["indice_personalizado"],
    )
    umbral_pers = CLASS_THRESHOLDS_RECOMENDADOS.get("calor", {"t1": 0.25})
    clase_pers = 0
    if prob_pers >= PERS_THRESHOLD_PELIGRO:
        clase_pers = 2
    elif prob_pers >= umbral_pers["t1"]:
        clase_pers = 1

    # Seguridad física por HI: se aplica después de personalización.
    # Solo sobre-escribe clase_pers si el perfil tiene factores de riesgo
    # significativos (edad avanzada, no aclimatado, comorbilidades, factor alto,
    # o si chocó el cap de factores).
    if HI is not None and override_fisico is None:
        if HI >= 39 and clase_pers < 2:
            override_fisico = {
                "clase_ml": clase_ml_original,
                "clase_final": 2,
                "razon": f"ML={CLASES[clase_ml_original]}, HI_peak={HI:.1f}C>=39 → PELIGRO",
            }
        elif HI >= 27 and clase_pers < 1:
            _edad_vulnerable = (
                (perfil or {}).get("edad", 0) >= 60
                and not (
                    (perfil or {}).get("entrenado")
                    and (perfil or {}).get("aclimatado")
                )
            )
            _vulnerable_calor = (
                (perfil or {}).get("comorbilidades")
                or (perfil or {}).get("farmacos")
                or _edad_vulnerable
                or (perfil or {}).get("aclimatado") is False
                or res_calor["factor_total"] > 1.8
                or res_calor.get("capado")
            )
            if _vulnerable_calor and (HI >= 32 or (UV is not None and UV > 3)):
                override_fisico = {
                    "clase_ml": clase_ml_original,
                    "clase_final": 1,
                    "razon": f"ML={CLASES[clase_ml_original]}, HI_peak={HI:.1f}C>={'32' if HI>=32 else '27'}+UV>3 → PRECAUCION",
                }

    # Seguridad física por frío (wind chill): análogo al override por HI.
    if WC is not None and override_fisico is None and clase_pers < 2:
        if WC <= -25:
            override_fisico = {
                "clase_ml": clase_ml_original,
                "clase_final": 2,
                "razon": f"ML={CLASES[clase_ml_original]}, WC={WC:.1f}C<=-25 → PELIGRO (riesgo de congelación)",
            }
        elif WC <= -10 and clase_pers < 1:
            _edad_vulnerable_frio = (
                (perfil or {}).get("edad", 0) >= 60
                and not (perfil or {}).get("entrenado")
            )
            _vulnerable_frio = (
                (perfil or {}).get("comorbilidades")
                or _edad_vulnerable_frio
                or (perfil or {}).get("situacion_social", set()) & {"vive_solo", "no_sale", "vivienda_fria"}
                or res_frio["factor_total"] > 1.8
                or res_frio.get("capado")
            )
            if _vulnerable_frio:
                override_fisico = {
                    "clase_ml": clase_ml_original,
                    "clase_final": 1,
                    "razon": f"ML={CLASES[clase_ml_original]}, WC={WC:.1f}C<=-10 → PRECAUCION",
                }

    # Downgrade por ausencia de calor real
    if HI is not None and HI < 27 and WC is not None and WC > 0 and (UV is None or UV < 6) and clase_pers > 0:
        if override_fisico is None:
            if clase_pers == 2:
                override_fisico = {
                    "clase_ml": clase_ml_original,
                    "clase_final": 1,
                    "razon": f"HI_peak={HI:.1f}C<27, WC={WC:.1f}C>0, UV<6 → PRECAUCION (ML={CLASES[clase_ml_original]}, pero sin calor actual)",
                }
            elif clase_pers == 1:
                override_fisico = {
                    "clase_ml": clase_ml_original,
                    "clase_final": 1,
                    "razon": f"HI_peak={HI:.1f}C<27, WC={WC:.1f}C>0, UV<6 (ML detecta tendencia de riesgo aunque sin calor ahora)",
                }

    if override_fisico:
        clase_final = override_fisico["clase_final"]
    else:
        clase_final = clase_pers

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

    X_calor = resultados.get("XGBoost_calor", {}).get("_X")
    X_frio = resultados.get("RandomForest_frio", {}).get("_X")

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
        razon = override_fisico["razon"]
        explicacion["modelo_determinante"] = f"Override — {razon}"
        explicacion["override"] = override_fisico
    elif override_fisico is None:
        clase_modelos = clase_ml_original
        if clase_pers != clase_modelos:
            direccion = "subió" if clase_pers > clase_modelos else "bajó"
            explicacion["modelo_determinante"] = f"Personalización ({direccion} de {CLASES[clase_modelos]} a {CLASES[clase_pers]})"

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


# ── Tendencia semanal con banda de confianza (FORECAST-001) ─────────────────
#
# La banda NO es un intervalo inventado: sale del prediction set conformal que
# ya calcula cada modelo ML en cada predicción (`SplitConformalCalibrator`,
# alpha=0.1 — documentacion/conformal_prediction.md). Ese set tiene tamaño 1
# (confianza alta), 2 (media) o 3 (baja); la semianchura de la banda es una
# función monótona documentada de ese tamaño. El punto de extensión para una
# banda calibrada numéricamente (en vez de la heurística sobre el set size)
# está documentado en documentacion/conformal_prediction.md.

BANDA_CONFIANZA_SEMIANCHURA = {1: 0.05, 2: 0.15, 3: 0.25}
_CONFIANZA_ETIQUETA = {1: "alta", 2: "media", 3: "baja"}


def _set_size_conformal_del_dia(resultado: dict) -> int:
    """Tamaño medio del prediction set conformal de los modelos ML del día.

    Lee los `conformal_set_size` (1/2/3) que traigan los resultados de esa
    llamada: solo los tabulares emiten ese campo (lstm/fórmula no tienen
    conformal), así que no hace falta descubrir nada por disco. Sin señal
    conformal, fallback a 2 (media).
    """
    sizes = []
    for m in (resultado.get("modelos") or {}).values():
        if not isinstance(m, dict):
            continue
        ss = m.get("conformal_set_size")
        if isinstance(ss, (int, float)) and 1 <= ss <= 3:
            sizes.append(int(ss))
    if not sizes:
        return 2
    return int(round(sum(sizes) / len(sizes)))


def prediccion_semanal(
    lat: float | None = None,
    lon: float | None = None,
    provincia: str = "Madrid",
    perfil: dict | None = None,
    resolucion: int = 60,
    dias: int | None = None,
) -> dict:
    """Serie diaria de riesgo a `dias` vista con su banda de confianza.

    Para cada día llama a ``predict_ensemble``; la banda sale del prediction
    set conformal que ya computan los modelos ML ese mismo día, no de un
    intervalo arbitrario. Si el forecast meteorológico no cubre un día, la
    serie se corta ahí y ``completo=False`` avisa explícitamente hasta dónde
    llega (FORECAST-001 criterio 3): nunca extrapola con datos inventados.
    """
    if dias is None:
        dias = FORECAST_HORIZON_DAYS

    from datetime import date as _date, timedelta

    hoy = _date.today()
    serie: list[dict] = []
    for i in range(dias):
        dia = hoy + timedelta(days=i)
        try:
            r = predict_ensemble(
                lat=lat, lon=lon, provincia=provincia, perfil=perfil,
                target_date=dia, resolucion=resolucion,
            )
        except ForecastHorizonError:
            # El forecast no cubre este día: cortar y avisar, no extrapolar.
            break

        prob = None
        for canal in ("calor", "frio"):
            prob = (r.get("perfil") or {}).get(canal, {}).get("prob_personalizada")
            if prob is not None:
                break

        set_size = _set_size_conformal_del_dia(r)
        semianchura = BANDA_CONFIANZA_SEMIANCHURA[set_size]
        serie.append({
            "fecha": dia.isoformat(),
            "prob": round(float(prob), 4) if prob is not None else None,
            "clase": r.get("clase_final_label"),
            "confianza_conformal": _CONFIANZA_ETIQUETA[set_size],
            "set_size_conformal": set_size,
            "banda": (
                [round(max(0.0, float(prob) - semianchura), 4),
                 round(min(1.0, float(prob) + semianchura), 4)]
                if prob is not None else None
            ),
        })

    return {
        "horizonte_dias": dias,
        "completo": len(serie) == dias,
        "forecast_hasta": serie[-1]["fecha"] if serie else None,
        "dias": serie,
        "banda_origen": (
            "prediction set conformal (SplitConformalCalibrator alpha=0.1) de "
            "los modelos ML de cada día: set_size 1/2/3 → semianchura ±5/15/25 "
            "puntos. documentacion/conformal_prediction.md"
        ),
    }
