"""
chat/app.py — Interfaz web de chat para ClimaSafeAI.

Servidor FastAPI + WebSocket que expone un chat interactivo
para interactuar con los modelos entrenados del proyecto.

Se inicia automaticamente via:
    docker compose up -d
o directamente:
    python -m uvicorn chat.app:app --host 0.0.0.0 --port 8080
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Query
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from climasafeai.db.manager import DBManager



# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_DIR   = Path(__file__).resolve().parents[1]
MODELS_DIR    = PROJECT_DIR / "models"
ARTIFACTS_DIR = MODELS_DIR / "artifacts"
CHAT_DIR    = PROJECT_DIR / "chat"
STATIC_DIR    = CHAT_DIR / "static"

# ---------------------------------------------------------------------------
# Estado global del servicio
# ---------------------------------------------------------------------------
_state: dict[str, Any] = {
    "models":         {},
    "scaler":         None,
    "encoders":       {},
    "feature_names":  [],
    "target_encoder": None,
    "model_loaded":   False,
}

# Constantes del proyecto (fijadas en la generacion del template)
_PROJECT     = "ClimaSafeAI"
_ML_TYPE     = "supervisado"
_TASK_TYPE   = "clasificacion"
_VERSION     = "0.0.1"


# ---------------------------------------------------------------------------
# Carga de modelos
# ---------------------------------------------------------------------------
def load_models() -> None:
    """Carga modelos y artefactos de preprocesado desde models/."""

    # Feature names
    fn_path = ARTIFACTS_DIR / "feature_names.joblib"
    if fn_path.exists():
        _state["feature_names"] = joblib.load(fn_path)

    # Scaler
    sc_path = ARTIFACTS_DIR / "scaler.joblib"
    if sc_path.exists():
        _state["scaler"] = joblib.load(sc_path)

    # Encoders de features categoricas
    enc_path = ARTIFACTS_DIR / "encoders.joblib"
    if enc_path.exists():
        _state["encoders"] = joblib.load(enc_path)

    # Target encoder
    te_path = ARTIFACTS_DIR / "target_encoder.joblib"
    if te_path.exists():
        _state["target_encoder"] = joblib.load(te_path)


    _skip = {"scaler", "encoders", "pca", "threshold", "feature_names",
             "target_encoder", "output_dim"}
    for path in sorted(MODELS_DIR.glob("*.joblib")):
        if path.stem in _skip or path.stem.startswith("best_params_"):
            continue
        try:
            _state["models"][path.stem] = joblib.load(path)
        except Exception as exc:
            print(f"[chat/app] No se pudo cargar {path.name}: {exc}", file=sys.stderr)



    _state["model_loaded"] = bool(_state["models"])
    print(f"[chat/app] Modelos cargados: {list(_state['models'].keys())}", file=sys.stderr)


# ---------------------------------------------------------------------------
# Prediccion generica
# ---------------------------------------------------------------------------
def _preprocess(features: dict[str, Any]) -> np.ndarray:
    """Convierte un dict de features a ndarray listo para predecir."""
    if _state["feature_names"]:
        missing = [f for f in _state["feature_names"] if f not in features]
        if missing:
            raise ValueError(f"Faltan features: {missing}")
        df = pd.DataFrame([{f: features[f] for f in _state["feature_names"]}])
    else:
        df = pd.DataFrame([features])

    for col, enc in _state["encoders"].items():
        if col == "__target__" or col not in df.columns:
            continue
        try:
            df[col] = enc.transform(df[col].astype(str))
        except ValueError as exc:
            raise ValueError(f"Valor desconocido en '{col}': {exc}") from exc

    if _state["scaler"] is not None:
        X = _state["scaler"].transform(df)
    else:
        try:
            X = df.values.astype(np.float64)  # solo funciona si todas las cols son numericas
        except (ValueError, TypeError) as exc:
            raise ValueError(
                "No hay scaler cargado y el dataframe contiene columnas no numericas. "
                "Vuelve a entrenar el modelo con `train`."
            ) from exc
    return X.astype(np.float32)


def predict_one(features: dict[str, Any]) -> dict[str, Any]:
    """Genera una prediccion a partir de un dict de features."""
    if not _state["model_loaded"]:
        return {"error": "No hay modelos cargados. Entrena primero con `train`."}

    try:
        X = _preprocess(features)
    except ValueError as exc:
        return {"error": str(exc)}


    model_name = list(_state["models"].keys())[0]
    model      = _state["models"][model_name]
    pred       = model.predict(X)[0]


    prob: float | None = None
    if hasattr(model, "predict_proba"):
        prob = float(model.predict_proba(X)[0].max())
    label: str | None = None
    if _state["target_encoder"] is not None:
        try:
            label = str(_state["target_encoder"].inverse_transform([int(pred)])[0])
        except Exception:
            label = str(pred)
    return {"prediction": int(pred), "probability": prob, "label": label, "model": model_name}





# ---------------------------------------------------------------------------
# Mensajes del bot
# ---------------------------------------------------------------------------
def _welcome_message() -> str:
    status_line = (
        f" ✔ **{len(_state['models'])} modelo(s) listo(s):** "
        f"`{'`, `'.join(_state['models'].keys())}`"
        if _state["model_loaded"]
        else "   Sin modelos entrenados — escribe `train` para entrenar."
    )
    return (
        f"#    Bienvenido a **ClimaSafeAI**\n\n"
        f"| Campo | Valor |\n"
        f"|---|---|\n"
        f"| **Tipo ML** | `supervisado` |\n"
        f"| **Tarea** | `{_TASK_TYPE}` |\n"
        f"| **Versión** | `0.0.1` |\n\n"
        f"{status_line}\n\n"
        f"---\n\n"
        f"**Comandos disponibles:**\n"
        f"- `status` — estado del sistema\n"
        f"- `info` — detalles del modelo y features\n"
        f"- `predict` — hacer una prediccion paso a paso\n"
        f"- `train` — lanzar el entrenamiento\n"
        f"- `reload` — recargar modelos del disco\n"
        f"- `help` — mostrar este mensaje"
    )


def _status_message() -> str:
    feat_count = len(_state["feature_names"])
    if _state["model_loaded"]:
        models_list = "\n".join(f"  - `{m}`" for m in _state["models"].keys())
        return (
            f"**Estado del sistema**  ✔\n\n"
            f"**Proyecto:** ClimaSafeAI\n"
            f"**ML Type:** `supervisado`\n"
            f"**Tarea:** `{_TASK_TYPE}`\n"
            f"**Features detectadas:** {feat_count}\n\n"
            f"**Modelos disponibles:**\n{models_list}"
        )
    return (
        f"**Estado del sistema**   \n\n"
        f"No hay modelos entrenados todavia.\n"
        f"Escribe `train` para lanzar el entrenamiento."
    )


def _info_message() -> str:
    if not _state["model_loaded"]:
        return "   No hay modelos cargados. Entrena primero con `train`."
    features = _state["feature_names"]
    feat_str = ", ".join(f"`{f}`" for f in features[:8])
    if len(features) > 8:
        feat_str += f" ... *y {len(features) - 8} mas*"
    classes_info = ""
    if _state["target_encoder"] is not None:
        classes = list(_state["target_encoder"].classes_)
        classes_info = f"\n**Clases:** {', '.join(str(c) for c in classes)}"
    return (
        f"**Informacion del modelo**\n\n"
        f"**Modelos:** {', '.join(_state['models'].keys())}\n"
        f"**Features ({len(features)}):** {feat_str}{classes_info}"
    )


def _start_prediction(session: dict) -> str:
    if not _state["model_loaded"]:
        return "   No hay modelos cargados. Entrena primero con `train`."
    features = _state["feature_names"]
    if not features:
        return "   No se detectaron nombres de features. Vuelve a entrenar el modelo."
    # Snapshot de features en la sesion — inmune a reload() durante la prediccion
    session.update({"state": "collecting", "features": {}, "idx": 0,
                    "snapshot_features": list(features)})
    return (
        f"   **Modo prediccion** — introduce el valor de cada feature.\n\n"
        f"**{len(features)} features** en total. Escribe `cancelar` para salir.\n\n"
        f"---\n\n"
        f"**[1/{len(features)}]** `{features[0]}`"
    )


def _handle_feature(msg: str, session: dict) -> str:
    if msg.lower() in ("cancel", "cancelar", "salir", "exit"):
        session["state"] = "idle"
        return " ✕ Prediccion cancelada."

    # Usar snapshot de la sesion (inmune a reload durante la prediccion)
    features = session.get("snapshot_features") or _state["feature_names"]
    idx      = session["idx"]
    if idx >= len(features):          # proteccion ante desfase por reload
        session["state"] = "idle"
        return "   Sesion de prediccion desfasada. Escribe `predict` para empezar de nuevo."
    name     = features[idx]

    try:
        value = float(msg.replace(",", "."))
    except ValueError:
        value = msg  # Categorica — mantener como string

    session["features"][name] = value
    session["idx"] += 1

    if session["idx"] >= len(features):
        session["state"] = "idle"
        result = predict_one(session["features"])

        if "error" in result:
            return f" ✕ Error: {result['error']}"


        pred       = result.get("prediction", "?")
        prob       = result.get("probability")
        label      = result.get("label")
        model_name = result.get("model", "?")
        prob_str   = f"\n**Confianza:** `{prob:.1%}`" if prob is not None else ""
        label_str  = f"\n**Clase:** `{label}`" if label else ""
        return (
            f" ✔ **Resultado de la prediccion**\n\n"
            f"**Prediccion:** `{pred}`{label_str}{prob_str}\n"
            f"**Modelo:** `{model_name}`\n\n"
            f"¿Otra prediccion? Escribe `predict`."
        )


    next_name = features[session["idx"]]  # features ya es el snapshot
    return (
        f"✓ `{name}` = `{value}`\n\n"
        f"**[{session['idx'] + 1}/{len(features)}]** `{next_name}`"
    )


# ---------------------------------------------------------------------------
# Maquina de estados del chat
# ---------------------------------------------------------------------------
async def process_message(msg: str, session: dict) -> str:
    """Enruta cada mensaje al handler correcto segun el estado de la sesion."""
    low = msg.lower().strip()

    if session["state"] == "collecting":
        return _handle_feature(msg, session)

    if low in ("help", "ayuda", "?", ""):
        return _welcome_message()
    if low in ("status", "estado"):
        return _status_message()
    if low in ("info"):
        return _info_message()
    if low in ("reload", "recargar"):
        _state.update({
            "models": {}, "model_loaded": False,
            "scaler": None, "encoders": {},
            "feature_names": [], "target_encoder": None,
        })
        load_models()
        if _state["model_loaded"]:
            return f" ✔ Modelos recargados: **{', '.join(_state['models'].keys())}**"
        return "   No se encontraron modelos en `models/`."
    if low in ("train", "entrenar", "training"):
        # Buscar dataset en ubicaciones posibles
        candidates = [
            PROJECT_DIR / "dataset.csv",
            PROJECT_DIR / "data" / "raw",
        ]
        dataset_found = candidates[0].exists()
        if not dataset_found:
            raw_dir = candidates[1]
            if raw_dir.is_dir():
                dataset_found = any(raw_dir.glob("*.csv"))
        if not dataset_found:
            return (
                "   No se encontro `dataset.csv`.\n\n"
                "Coloca el dataset en la raiz del proyecto o en `data/raw/` y escribe `train` de nuevo."
            )
        try:
            proc = subprocess.Popen(
                [sys.executable, "main.py"],
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,   # evita deadlock por pipe lleno
                stderr=subprocess.DEVNULL,
                cwd=str(PROJECT_DIR),
                text=True,
            )
            # Enviar "0" al prompt interactivo y desacoplar (background)
            proc.stdin.write("0\n")
            proc.stdin.flush()
            proc.stdin.close()
            # No esperamos (proc.wait()) — corre en background
            return (
                "  **Entrenamiento iniciado** en segundo plano.\n\n"
                "El proceso puede tardar varios minutos dependiendo del dataset.\n"
                "Cuando termine, escribe `reload` para cargar los modelos."
            )
        except Exception as exc:
            return f" ✕ Error al iniciar entrenamiento: {exc}"
    if low in ("predict", "predecir", "prediccion"):
        return _start_prediction(session)
    if low in ("cancel", "cancelar"):
        return "ℹ No hay ninguna operacion activa."

    return (
        f"❓ No reconozco `{msg}`.\n\n"
        f"Escribe `help` para ver los comandos disponibles."
    )


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
app = FastAPI(
    title="ClimaSafeAI — Chat",
    description="Interfaz web de chat para ClimaSafeAI. Generado por dskit.",
    version="0.0.1",
)

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# Modelos se cargan bajo demanda (al conectar WebSocket), no al arrancar


@app.get("/", response_class=HTMLResponse)
async def root():
    html_file = STATIC_DIR / "index.html"
    return HTMLResponse(
        content=html_file.read_text(encoding="utf-8"),
        headers={"Cache-Control": "no-store, max-age=0"},
    )


@app.get("/api/status")
async def api_status():
    return {
        "project":      _PROJECT,
        "ml_type":      _ML_TYPE,
        "task_type":    _TASK_TYPE,
        "version":      _VERSION,
        "model_loaded": _state["model_loaded"],
        "models":       list(_state["models"].keys()),
        "feature_count": len(_state["feature_names"]),
        "features":     _state["feature_names"],
        "has_pending_factors": len(_get_pending_factors() or []) > 0,
    }


@app.get("/api/reload")
async def api_reload():
    _state.update({
        "models": {}, "model_loaded": False,
        "scaler": None, "encoders": {},
        "feature_names": [], "target_encoder": None,
    })
    load_models()
    return {
        "model_loaded": _state["model_loaded"],
        "models": list(_state["models"].keys()),
    }


_db = DBManager()
_db.initialize()


def _normalize_perfil(perfil: dict) -> dict:
    """Convierte listas del frontend a sets."""
    p = dict(perfil)

    comorb = p.get("comorbilidades")
    if isinstance(comorb, list):
        p["comorbilidades"] = {c for c in comorb if c}

    social = p.get("situacion_social")
    if isinstance(social, list):
        p["situacion_social"] = {s for s in social if s}

    return p


def _get_weather_summary(result: dict) -> dict:
    """Extrae un dict serializable del weather (sin df_hora/df_features)."""
    w = result.get("weather", {})
    return {
        "lat": w.get("lat"),
        "lon": w.get("lon"),
        "uv_index": w.get("uv_index"),
        "current": w.get("current"),
        "perfil_horario": w.get("perfil_horario"),
        "provincia": w.get("provincia"),
        "target_date": w.get("target_date"),
    }


def _get_implemented_factors() -> dict:
    """Devuelve solo factores con implementado=true, agrupados por tipo y categoria."""
    return _db.obtener_factores(solo_implementados=True)


def _get_pending_factors() -> list[dict]:
    """Lee factores con implementado=false de SQLite."""
    return _db.factores_pendientes()


@app.get("/api/pending-factors")
async def api_pending_factors():
    return {
        "count": len(f := _get_pending_factors()),
        "factors": f,
    }


@app.get("/api/factores")
async def api_factores():
    return _get_implemented_factors()


@app.post("/api/approve-factor")
async def api_approve_factor(body: dict):
    tipo = body.get("tipo")
    categoria = body.get("categoria")
    clave = body.get("clave")
    errors = []
    if not tipo:
        errors.append("tipo")
    if not categoria:
        errors.append("categoria")
    if not clave:
        errors.append("clave")
    if errors:
        return {"success": False, "error": f"Faltan campos: {', '.join(errors)}"}

    result = _db.aprobar_factor(tipo, categoria, clave)
    return result


@app.post("/api/rag-search")
async def api_rag_search(body: dict):
    query = body.get("query", "")
    k = body.get("k", 5)
    if not query.strip():
        return {"success": False, "error": "query vacía"}
    try:
        results = _db.search_factores(query, k=k)
        return {"success": True, "results": results, "total": len(results)}
    except Exception as e:
        return {"success": False, "error": str(e)}


@app.post("/api/predict")
async def api_predict(body: dict, date: str | None = None):
    provincia = body.get("provincia", "Madrid")
    lat = body.get("lat")
    lon = body.get("lon")
    raw_perfil = body.get("perfil") or {}
    perfil_id = raw_perfil.get("perfil_id")
    perfil = _normalize_perfil(raw_perfil)

    target_date = None
    if date:
        try:
            from datetime import date as date_type, timedelta
            target_date = date_type.fromisoformat(date)
            today = date_type.today()
            if target_date < today:
                return {"error": f"La fecha {date} ya pasó. Solo se aceptan hoy o el futuro."}
            if (target_date - today).days > 2:
                return {"error": f"Fecha {date} está a más de 2 días vista. Horizonte máximo: 2 días."}
        except ValueError:
            return {"error": f"Fecha inválida: '{date}'. Usa formato ISO: YYYY-MM-DD"}

    try:
        from climasafeai.models.ensemble import predict_ensemble
        result = predict_ensemble(lat=lat, lon=lon, provincia=provincia, perfil=perfil, target_date=target_date)
    except Exception as exc:
        return {"error": str(exc)}

    # Guardar perfil en SQLite (sin perfil_id ni alias en datos)
    alias = raw_perfil.get("alias")
    datos_perfil = {k: v for k, v in raw_perfil.items() if k not in ("perfil_id", "alias")}
    # Quitar campos internos que no deben persistir en SQLite
    for _k in ("_perfil_horario", "perfil_id", "alias"):
        datos_perfil.pop(_k, None)
    datos_perfil["lat"] = lat
    datos_perfil["lon"] = lon
    datos_perfil["provincia"] = provincia
    if alias:
        existente = _db.buscar_por_alias(alias)
        if existente:
            perfil_id = existente["id"]
            datos_perfil["alias"] = alias
            _db.actualizar_perfil(perfil_id, datos_perfil)
        else:
            datos_perfil["alias"] = alias
            perfil_id = _db.crear_perfil(datos_perfil)
    elif perfil_id:
        _db.actualizar_perfil(perfil_id, datos_perfil)
    else:
        perfil_id = _db.crear_perfil(datos_perfil)
    result["perfil_id"] = perfil_id

    # Guardar consulta
    clase = result.get("clase_final_label", result.get("clase_final"))
    tipo = result.get("tipo", "calor")
    indice_orig = result.get("explicacion", {}).get("indice_original")
    indice_pers = result.get("explicacion", {}).get("indice_personalizado")
    _db.guardar_consulta(
        perfil_id=perfil_id, provincia=provincia, lat=lat, lon=lon,
        tipo_riesgo=tipo, indice_original=indice_orig,
        indice_personalizado=indice_pers, clase_final=clase,
    )

    result["perfil_usuario"] = perfil
    result["weather"] = _get_weather_summary(result)
    if target_date:
        result["target_date"] = target_date.isoformat()

    for mod_name, mod_res in result.get("modelos", {}).items():
        if isinstance(mod_res, dict):
            mod_res.pop("_X", None)
    if "error" in result.get("modelos", {}).get("LSTM", {}):
        del result["modelos"]["LSTM"]["error"]

    result["weather"].pop("df_hora", None)
    result["weather"].pop("df_features", None)

    # Curva de riesgo por hora (instantáneo personalizado + carga térmica
    # acumulada) y recomendación de horario. Evita la "línea recta" en la
    # gráfica y da el mejor tramo horario para la actividad.
    try:
        from climasafeai.features.personalizacion import (
            riesgo_horario_acumulado, recomendar_horario, pico_riesgo_actividad,
        )
        _ph = result["weather"].get("perfil_horario") or []
        _curva = riesgo_horario_acumulado(_ph, perfil)
        result["riesgo_horario"] = _curva
        result["riesgo_pico"] = pico_riesgo_actividad(_curva, perfil)
        result["recomendacion_horario"] = recomendar_horario(_ph, perfil)
    except Exception:
        pass

    return result


@app.post("/api/riesgo-colectivo")
async def api_riesgo_colectivo(body: dict):
    """Calcula riesgo para un grupo."""
    provincia = body.get("provincia", "Madrid")
    lat = body.get("lat")
    lon = body.get("lon")
    target_date = body.get("fecha")
    date_obj = None
    if target_date:
        try:
            from datetime import date as date_type
            date_obj = date_type.fromisoformat(target_date)
        except ValueError:
            pass

    tipo = body.get("tipo", "numero")
    from climasafeai.models.ensemble import predict_ensemble
    from climasafeai.features.personalizacion import (
        riesgo_horario_acumulado, recomendar_horario, pico_riesgo_actividad,
    )

    if tipo == "numero":
        cantidad = int(body.get("cantidad", 100))
        edad_min = int(body.get("edad_min", 18))
        edad_max = int(body.get("edad_max", 80))
        pct_hombres = int(body.get("pct_hombres", 50))
        actividad = body.get("actividad", "ligera")
        hora_inicio = float(body.get("hora_inicio", 10))
        duracion = float(body.get("duracion", 2))
        aclimatado = body.get("aclimatado")

        # Factores de riesgo porcentuales del grupo
        pcts = {
            "grasa_alta": float(body.get("pct_grasa_alta", 0)),
            "cardiovascular": float(body.get("pct_cardiovascular", 0)),
            "diabetes": float(body.get("pct_diabetes", 0)),
            "respiratoria": float(body.get("pct_respiratoria", 0)),
            "mental": float(body.get("pct_mental", 0)),
            "no_aclimatados": float(body.get("pct_no_aclimatados", 0)),
        }
        COEF_PCT = {
            "grasa_alta": 1.08,
            "cardiovascular": 1.4,
            "diabetes": 1.2,
            "respiratoria": 1.3,
            "mental": 1.8,
            "no_aclimatados": 1.6,
        }

        def _factor_grupo(pct: float, coef: float) -> float:
            if pct <= 0:
                return 1.0
            return 1.0 + (pct / 100.0) * (coef - 1.0)

        factor_extra = 1.0
        for k, coef in COEF_PCT.items():
            factor_extra *= _factor_grupo(pcts[k], coef)
        factor_extra = min(factor_extra, 2.5)

        rangos_edad = [
            (18, 30), (30, 45), (45, 60), (60, 75), (75, 90)
        ]
        rangos_edad = [(a, b) for a, b in rangos_edad if a < edad_max and b > edad_min]
        if not rangos_edad:
            rangos_edad = [(edad_min, edad_max)]

        total_rango_pct = sum(min(b, edad_max) - max(a, edad_min) for a, b in rangos_edad)
        resultados_rangos = []
        total_seguros = 0
        total_precaucion = 0
        total_peligro = 0
        primer_pred_num = None

        for a, b in rangos_edad:
            solapamiento = max(0, min(b, edad_max) - max(a, edad_min))
            if solapamiento <= 0:
                continue
            pct_rango = solapamiento / total_rango_pct
            n_personas = max(1, int(round(cantidad * pct_rango)))
            edad_med = (max(a, edad_min) + min(b, edad_max)) // 2

            for sexo in ("hombre", "mujer"):
                pct_sexo = pct_hombres / 100 if sexo == "hombre" else (100 - pct_hombres) / 100
                n_sexo = max(1, int(round(n_personas * pct_sexo)))
                if n_sexo == 0:
                    continue

                perfil = {
                    "edad": edad_med,
                    "sexo": sexo,
                    "nivel_actividad": actividad,
                    "hora_inicio": hora_inicio,
                    "duracion_actividad_h": duracion,
                }
                if body.get("ocupacion"):
                    perfil["ocupacion"] = body["ocupacion"]
                if body.get("deporte"):
                    perfil["deporte"] = body["deporte"]
                if aclimatado:
                    perfil["aclimatado"] = aclimatado == "si"

                try:
                    pred = predict_ensemble(lat=lat, lon=lon, provincia=provincia, perfil=perfil, target_date=date_obj)
                    if primer_pred_num is None:
                        primer_pred_num = pred
                    clase = pred.get("clase_final", 0)
                    prob_base = pred.get("perfil", {}).get("calor", {}).get("prob_personalizada", 0)
                    # Aplicar factor extra del grupo en espacio de odds
                    prob = prob_base
                    if factor_extra != 1.0 and 0 < prob_base < 1:
                        odds = prob_base / (1.0 - prob_base)
                        prob = odds * factor_extra / (1.0 + odds * factor_extra)
                except Exception:
                    clase = 0
                    prob = 0

                if clase == 2:
                    total_peligro += n_sexo
                elif clase == 1:
                    total_precaucion += n_sexo
                else:
                    total_seguros += n_sexo

                resultados_rangos.append({
                    "rango": f"{edad_med}a {sexo[0]}",
                    "seguros": n_sexo if clase == 0 else 0,
                    "precaucion": n_sexo if clase == 1 else 0,
                    "peligro": n_sexo if clase == 2 else 0,
                    "prob": round(prob, 4),
                })

        total = total_seguros + total_precaucion + total_peligro
        pct_peligro = round(total_peligro / total * 100, 1) if total else 0
        # Añadir detalle de factores aplicados al mensaje
        factores_activos = [f"{k}={pcts[k]:.0f}%" for k in sorted(pcts) if pcts[k] > 0]
        sufijo_extra = f" · Factor extra grupo: x{factor_extra:.2f}" if factor_extra > 1.01 else ""

        # Perfil para el mapa de zona: peor caso del rango (edad máxima) más las
        # comorbilidades/condiciones que afectan a una fracción relevante del grupo.
        _comorb_map = {
            "cardiovascular": "cardiovascular",
            "diabetes": "diabetes",
            "respiratoria": "respiratoria",
            "mental": "mental",
        }
        comorb_mapa = {_comorb_map[k] for k, col in _comorb_map.items() if pcts.get(k, 0) >= 50}
        perfil_mapa = {
            "edad": edad_max,
            "sexo": "hombre",
            "nivel_actividad": actividad,
            "hora_inicio": hora_inicio,
            "duracion_actividad_h": duracion,
        }
        if comorb_mapa:
            perfil_mapa["comorbilidades"] = comorb_mapa
        if pcts.get("no_aclimatados", 0) >= 50 or aclimatado == "no":
            perfil_mapa["aclimatado"] = False
        elif aclimatado == "si":
            perfil_mapa["aclimatado"] = True
        if body.get("ocupacion"):
            perfil_mapa["ocupacion"] = body["ocupacion"]
        if body.get("deporte"):
            perfil_mapa["deporte"] = body["deporte"]

        _hourly_num = primer_pred_num.get("weather", {}).get("perfil_horario", []) if primer_pred_num else []
        grp_curva = riesgo_horario_acumulado(_hourly_num, perfil_mapa)
        grp_reco = recomendar_horario(_hourly_num, perfil_mapa)

        return {
            "total_personas": total,
            "seguros": total_seguros,
            "en_precaucion": total_precaucion,
            "en_peligro": total_peligro,
            "pct_peligro": pct_peligro,
            "clase": "PELIGRO" if pct_peligro > 20 else ("PRECAUCION" if pct_peligro > 5 else "SEGURO"),
            "factor_extra": round(factor_extra, 3),
            "factores_grupo": factores_activos,
            "mensaje": f"De {total} personas, ~{total_peligro} en peligro, ~{total_precaucion} en precaución" + sufijo_extra,
            "rangos": resultados_rangos,
            "perfil_mapa": perfil_mapa,   # peor caso del rango: el mapa de zona lo usa
            "riesgo_horario": grp_curva,
            "recomendacion_horario": grp_reco,
            "weather": _get_weather_summary(primer_pred_num) if primer_pred_num else None,
        }

    elif tipo == "etiqueta":
        tag = body.get("tag", "").strip()
        if not tag:
            return {"error": "tag requerido"}
                # Parámetros del grupo que sobreescriben a los saved del perfil
        hora_inicio = body.get("hora_inicio")
        duracion = body.get("duracion")
        nivel_actividad = body.get("actividad")
        tipo_actividad = body.get("tipo_actividad")
        aclimatado_grupo = body.get("aclimatado")
        ocupacion_grupo = body.get("ocupacion")
        deporte_grupo = body.get("deporte")
        perfiles = _db.buscar_por_tag(tag)
        resultados = []
        primer_pred = None
        perfil_mapa = None          # perfil más restrictivo (para el mapa de zona)
        peor_prob_mapa = -1.0
        for p in perfiles:
            try:
                perfil = {k: v for k, v in p.items() if k not in ("id", "alias", "tags", "created_at", "updated_at")}
                if hora_inicio is not None:
                    perfil["hora_inicio"] = float(hora_inicio)
                if duracion is not None:
                    perfil["duracion_actividad_h"] = float(duracion)
                if nivel_actividad:
                    perfil["nivel_actividad"] = nivel_actividad
                if aclimatado_grupo is not None:
                    perfil["aclimatado"] = aclimatado_grupo == "si"
                if ocupacion_grupo:
                    perfil["ocupacion"] = ocupacion_grupo
                if deporte_grupo:
                    perfil["deporte"] = deporte_grupo
                if tipo_actividad:
                    if hora_inicio is None and duracion is None and not nivel_actividad:
                        if tipo_actividad == "trabajo":
                            perfil.setdefault("nivel_actividad", "moderada")
                            perfil.setdefault("hora_inicio", 8)
                            perfil.setdefault("duracion_actividad_h", 8)
                        elif tipo_actividad == "competicion":
                            perfil.setdefault("nivel_actividad", "muy_intensa")
                        elif tipo_actividad == "deporte":
                            perfil.setdefault("nivel_actividad", "moderada")
                pred = predict_ensemble(lat=lat, lon=lon, provincia=provincia, perfil=perfil, target_date=date_obj)
                hourly = pred.get("weather", {}).get("perfil_horario", [])
                if primer_pred is None:
                    primer_pred = pred
                prob_r = pred.get("perfil", {}).get("calor", {}).get("prob_personalizada", 0) or 0
                if prob_r > peor_prob_mapa:
                    peor_prob_mapa = prob_r
                    perfil_mapa = dict(perfil)
                    perfil_mapa["_alias"] = p.get("alias", "?")
                _curva = riesgo_horario_acumulado(hourly, perfil)
                resultados.append({
                    "alias": p.get("alias", "?"),
                    "edad": p.get("edad"),
                    "sexo": p.get("sexo"),
                    "clase": pred.get("clase_final_label", "SEGURO"),
                    "prob_riesgo": pred.get("perfil", {}).get("calor", {}).get("prob_personalizada", 0),
                    "riesgo_pico": pico_riesgo_actividad(_curva, perfil),
                    "perfil_horario": hourly,
                    "riesgo_horario": _curva,
                    "recomendacion_horario": recomendar_horario(hourly, perfil),
                    "explicacion": pred.get("explicacion"),
                    "recomendaciones": pred.get("recomendaciones", []),
                    "factores": pred.get("perfil", {}).get("calor", {}).get("factores", []),
                })
            except Exception as e:
                resultados.append({"alias": p.get("alias", "?"), "error": str(e)})

        seguros = sum(1 for r in resultados if r.get("clase") == "SEGURO")
        precaucion = sum(1 for r in resultados if r.get("clase") == "PRECAUCION")
        peligro = sum(1 for r in resultados if r.get("clase") == "PELIGRO")

        # Curva y recomendación de horario del GRUPO, según el perfil más
        # restrictivo (protege al más vulnerable de la cuadrilla).
        _hourly_grp = primer_pred.get("weather", {}).get("perfil_horario", []) if primer_pred else []
        grp_curva = riesgo_horario_acumulado(_hourly_grp, perfil_mapa) if perfil_mapa else []
        grp_reco = recomendar_horario(_hourly_grp, perfil_mapa) if perfil_mapa else None

        return {
            "total_personas": len(resultados),
            "seguros": seguros,
            "en_precaucion": precaucion,
            "en_peligro": peligro,
            "pct_peligro": round(peligro / len(resultados) * 100, 1) if resultados else 0,
            "detalle": resultados,
            "perfil_mapa": perfil_mapa,   # el más restrictivo: el mapa de zona lo usa
            "riesgo_horario": grp_curva,        # curva del peor caso del grupo
            "recomendacion_horario": grp_reco,  # horario óptimo del grupo
            "weather": _get_weather_summary(primer_pred) if primer_pred else None,
        }

    return {"error": "tipo no válido"}


@app.get("/api/perfiles")
async def api_list_perfiles():
    """Lista todos los perfiles (cabecera con alias, coordenadas)."""
    return _db.listar_perfiles()


@app.get("/api/perfil/{perfil_id}")
async def api_get_perfil(perfil_id: int):
    """Devuelve un perfil guardado (escalares + arrays)."""
    p = _db.obtener_perfil(perfil_id)
    if p is None:
        return {"error": "Perfil no encontrado"}
    # Quitar campos internos
    for k in ("id", "created_at", "updated_at", "aclimatado_actualizado_en"):
        p.pop(k, None)
    # Devolver arrays como listas para el frontend
    return p


@app.post("/api/perfil")
async def api_save_perfil(body: dict):
    """Guarda un perfil (sin predecir). Si incluye alias, busca o crea."""
    alias = body.get("alias")
    perfil_id = body.get("perfil_id")
    datos = {k: v for k, v in body.items() if k not in ("perfil_id", "alias")}

    if alias:
        existente = _db.buscar_por_alias(alias)
        if existente:
            perfil_id = existente["id"]
            datos["alias"] = alias
            _db.actualizar_perfil(perfil_id, datos)
        else:
            datos["alias"] = alias
            perfil_id = _db.crear_perfil(datos)
    elif perfil_id:
        _db.actualizar_perfil(perfil_id, datos)
    else:
        perfil_id = _db.crear_perfil(datos)

    return {"perfil_id": perfil_id}


@app.post("/api/perfil/{perfil_id}/tags")
async def api_update_tags(perfil_id: int, body: dict):
    """Actualiza las etiquetas de un perfil."""
    tags = body.get("tags", "")
    _db.actualizar_perfil(perfil_id, {"tags": tags})
    return {"ok": True}


@app.delete("/api/perfil/{perfil_id}")
async def api_delete_perfil(perfil_id: int):
    """Elimina un perfil."""
    _db.eliminar_perfil(perfil_id)
    return {"ok": True}


# ── Tags disponibles ────────────────────────────────────────────────

@app.get("/api/tags-disponibles")
async def api_list_tags_disponibles():
    return _db.listar_tags_disponibles()


@app.post("/api/tags-disponibles")
async def api_create_tag_disponible(body: dict):
    nombre = body.get("nombre", "").strip()
    if not nombre:
        return {"error": "Nombre requerido"}
    tag_id = _db.crear_tag_disponible(nombre)
    return {"id": tag_id, "nombre": nombre}


@app.delete("/api/tags-disponibles/{tag_id}")
async def api_delete_tag_disponible(tag_id: int):
    _db.eliminar_tag_disponible(tag_id)
    return {"ok": True}


@app.post("/api/contrafactuales")
async def api_contrafactuales(body: dict):
    from climasafeai.models.explicabilidad import generar_contrafactuales

    provincia = body.get("provincia", "Madrid")
    lat = body.get("lat")
    lon = body.get("lon")
    raw_perfil = body.get("perfil") or {}
    perfil = _normalize_perfil(raw_perfil)

    from climasafeai.models.ensemble import predict_ensemble
    result = predict_ensemble(lat=lat, lon=lon, provincia=provincia, perfil=perfil)

    cfs = generar_contrafactuales(result)
    perfil_aplicado = result.get("perfil", {})
    prob_pers = max(
        (perfil_aplicado.get("calor") or {}).get("prob_personalizada", 0),
        (perfil_aplicado.get("frio") or {}).get("prob_personalizada", 0),
    )
    PERS_THRESHOLD_PELIGRO = 0.55
    umbral_t1 = 0.25
    if prob_pers >= PERS_THRESHOLD_PELIGRO:
        clase_pers_idx = 2
    elif prob_pers >= umbral_t1:
        clase_pers_idx = 1
    else:
        clase_pers_idx = 0
    clase_pers_label = ["SEGURO", "PRECAUCION", "PELIGRO"][clase_pers_idx]
    return {
        "clase_final_sistema": result.get("clase_final_label"),
        "clase_personalizada": clase_pers_label,
        "probabilidad_personalizada": round(prob_pers, 4),
        "contrafactuales": cfs,
        "total": len(cfs),
        "nota": "La 'clase_final_sistema' puede incluir un override por HI/UV que prevalece sobre la probabilidad personalizada." if result.get("override_fisico") else None,
    }


@app.get("/api/riesgo-zona")
async def api_riesgo_zona(
    lat: float = Query(...),
    lon: float = Query(...),
    radio_km: float = Query(5, ge=0.5, le=50),
    perfil: str = Query("vulnerable"),
    fecha: str | None = Query(None),
):
    from climasafeai.data.grid_risk import riesgo_zona_grid, PERFILES_DISPONIBLES

    if perfil not in PERFILES_DISPONIBLES:
        return {"error": f"Perfil no válido. Opciones: {', '.join(PERFILES_DISPONIBLES.keys())}"}

    date_obj = None
    if fecha:
        try:
            from datetime import date as date_type
            date_obj = date_type.fromisoformat(fecha)
        except ValueError:
            return {"error": f"Fecha inválida: '{fecha}'. Usa ISO: YYYY-MM-DD"}

    try:
        result = riesgo_zona_grid(lat=lat, lon=lon, radio_km=radio_km, perfil_id=perfil, target_date=date_obj)
        if "error" in result:
            return result
        return result
    except Exception as exc:
        return {"error": str(exc)}


@app.post("/api/riesgo-zona")
async def api_riesgo_zona_post(body: dict):
    from climasafeai.data.grid_risk import riesgo_zona_grid

    lat = body.get("lat")
    lon = body.get("lon")
    if lat is None or lon is None:
        return {"error": "lat y lon requeridos"}
    radio_km = float(body.get("radio_km", 5))
    perfil_id = body.get("perfil_id", "adulto")
    perfil = body.get("perfil")
    fecha = body.get("fecha")

    date_obj = None
    if fecha:
        try:
            from datetime import date as date_type
            date_obj = date_type.fromisoformat(fecha)
        except ValueError:
            return {"error": f"Fecha inválida: '{fecha}'. Usa ISO: YYYY-MM-DD"}

    try:
        result = riesgo_zona_grid(
            lat=lat, lon=lon, radio_km=radio_km,
            perfil_id=perfil_id, target_date=date_obj,
            perfil=perfil,
        )
        if "error" in result:
            return result
        return result
    except Exception as e:
        return {"error": str(e)}


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    if not _state["model_loaded"]:
        load_models()
    await ws.send_json({"type": "bot", "text": _welcome_message()})

    session: dict[str, Any] = {"state": "idle", "features": {}, "idx": 0}

    try:
        while True:
            try:
                data = await ws.receive_json()
            except WebSocketDisconnect:
                raise   # propagar para que el outer except lo capture
            except Exception:
                # Mensaje malformado (no JSON) — ignorar y seguir
                try:
                    await ws.send_json({"type": "bot", "text": " ✕ Mensaje no válido. Usa texto plano."})
                except Exception:
                    pass
                continue
            msg   = data.get("text", "").strip()
            reply = await process_message(msg, session)
            await ws.send_json({"type": "bot", "text": reply})
    except WebSocketDisconnect:
        pass