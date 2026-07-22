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
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
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
    return HTMLResponse(content=html_file.read_text(encoding="utf-8"))


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

    # Guardar perfil en SQLite (sin perfil_id en los datos)
    datos_perfil = {k: v for k, v in raw_perfil.items() if k != "perfil_id"}
    if perfil_id:
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

    return result


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
    """Guarda un perfil (sin predecir). Si incluye perfil_id, actualiza."""
    perfil_id = body.get("perfil_id")
    datos = {k: v for k, v in body.items() if k != "perfil_id"}
    if perfil_id:
        _db.actualizar_perfil(perfil_id, datos)
    else:
        perfil_id = _db.crear_perfil(datos)
    return {"perfil_id": perfil_id}


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