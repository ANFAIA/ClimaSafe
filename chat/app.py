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

# Cargar modelos al arrancar (errores no fatales: app arranca igual)
try:
    load_models()
except Exception as _startup_exc:
    print(f"[chat/app] Aviso al cargar modelos al inicio: {_startup_exc}", file=sys.stderr)


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


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
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