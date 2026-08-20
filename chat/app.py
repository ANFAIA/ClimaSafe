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

import logging
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, Query
from fastapi.responses import HTMLResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from starlette.exceptions import HTTPException as StarletteHTTPException

from climasafeai.db.manager import CampoDesconocidoError, DBManager
from climasafeai.features.personalizacion import nivel_actividad_de_deporte

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_DIR = Path(__file__).resolve().parents[1]
MODELS_DIR = PROJECT_DIR / "models"
ARTIFACTS_DIR = MODELS_DIR / "artifacts"
CHAT_DIR = PROJECT_DIR / "chat"
STATIC_DIR = CHAT_DIR / "static"

# ---------------------------------------------------------------------------
# Estado global del servicio
# ---------------------------------------------------------------------------
_state: dict[str, Any] = {
    "models": {},
    "scaler": None,
    "encoders": {},
    "feature_names": [],
    "target_encoder": None,
    "model_loaded": False,
}

# Constantes del proyecto (fijadas en la generacion del template)
_PROJECT = "ClimaSafeAI"
_ML_TYPE = "supervisado"
_TASK_TYPE = "clasificacion"


def _resolve_version() -> str:
    """Versión única (PACK-001): pyproject.toml es la fuente de verdad.

    La sube `harness finish` en cada cierre. Se prefiere la versión instalada
    del paquete; si no está instalado se lee pyproject.toml; si nada funciona,
    "0.0.0". Nada hardcodeado.
    """
    try:
        from importlib.metadata import version as _pkg_version

        return _pkg_version("climasafeai")
    except Exception:
        pass
    try:
        import tomllib

        with open(PROJECT_DIR / "pyproject.toml", "rb") as fh:
            return tomllib.load(fh)["project"]["version"]
    except Exception:
        return "0.0.0"


_VERSION = _resolve_version()


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

    _skip = {
        "scaler",
        "encoders",
        "pca",
        "threshold",
        "feature_names",
        "target_encoder",
        "output_dim",
    }
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
    model = _state["models"][model_name]
    pred = model.predict(X)[0]

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
        f"| **Versión** | `{_VERSION}` |\n\n"
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
    session.update(
        {"state": "collecting", "features": {}, "idx": 0, "snapshot_features": list(features)}
    )
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
    idx = session["idx"]
    if idx >= len(features):  # proteccion ante desfase por reload
        session["state"] = "idle"
        return "   Sesion de prediccion desfasada. Escribe `predict` para empezar de nuevo."
    name = features[idx]

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

        pred = result.get("prediction", "?")
        prob = result.get("probability")
        label = result.get("label")
        model_name = result.get("model", "?")
        prob_str = f"\n**Confianza:** `{prob:.1%}`" if prob is not None else ""
        label_str = f"\n**Clase:** `{label}`" if label else ""
        return (
            f" ✔ **Resultado de la prediccion**\n\n"
            f"**Prediccion:** `{pred}`{label_str}{prob_str}\n"
            f"**Modelo:** `{model_name}`\n\n"
            f"¿Otra prediccion? Escribe `predict`."
        )

    next_name = features[session["idx"]]  # features ya es el snapshot
    return f"✓ `{name}` = `{value}`\n\n**[{session['idx'] + 1}/{len(features)}]** `{next_name}`"


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
        _state.update(
            {
                "models": {},
                "model_loaded": False,
                "scaler": None,
                "encoders": {},
                "feature_names": [],
                "target_encoder": None,
            }
        )
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
                stdout=subprocess.DEVNULL,  # evita deadlock por pipe lleno
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

    return f"❓ No reconozco `{msg}`.\n\nEscribe `help` para ver los comandos disponibles."


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
app = FastAPI(
    title="ClimaSafeAI — Chat",
    description="Interfaz web de chat para ClimaSafeAI. Generado por dskit.",
    version=_VERSION,
)

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


# WEB-005: los fallos salían con HTTP 200 y un {"error": ...} en el cuerpo, así
# que cualquier cliente que mirase el código creía que había ido bien. Ahora se
# lanza HTTPException con el código correcto, pero el cuerpo sigue siendo
# {"error": ...} y no el {"detail": ...} de FastAPI: el frontend hace
# `if (d.error)` en una veintena de sitios y no hay motivo para romperlo.
@app.exception_handler(StarletteHTTPException)
async def _error_como_error(request, exc: StarletteHTTPException):
    return JSONResponse({"error": exc.detail}, status_code=exc.status_code)


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
        "project": _PROJECT,
        "ml_type": _ML_TYPE,
        "task_type": _TASK_TYPE,
        "version": _VERSION,
        "model_loaded": _state["model_loaded"],
        "models": list(_state["models"].keys()),
        "feature_count": len(_state["feature_names"]),
        "features": _state["feature_names"],
        "has_pending_factors": len(_get_pending_factors() or []) > 0,
    }


@app.get("/api/reload")
async def api_reload():
    _state.update(
        {
            "models": {},
            "model_loaded": False,
            "scaler": None,
            "encoders": {},
            "feature_names": [],
            "target_encoder": None,
        }
    )
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

    farmacos = p.get("farmacos")
    if isinstance(farmacos, list):
        p["farmacos"] = {f for f in farmacos if f}

    social = p.get("situacion_social")
    if isinstance(social, list):
        p["situacion_social"] = {s for s in social if s}

    return p


def _aplicar_deporte_a_nivel(perfil: dict) -> None:
    """Si el perfil lleva deporte, el MET del Compendium fija la intensidad.

    Igual que el bot (BOT-007): el MET del deporte manda sobre el
    nivel_actividad que traiga el perfil por defecto. Si el deporte no está en
    la tabla de MET, se respeta el nivel_actividad que traiga.
    """
    dep = perfil.get("deporte")
    if not dep:
        return
    nivel = nivel_actividad_de_deporte(dep)
    if nivel:
        perfil["nivel_actividad"] = nivel


def _chat_id_de_perfil(perfil_id: int) -> str | None:
    """chat_id con el que el perfil guarda rutinas y avisos.

    Si el perfil está vinculado a un chat de Telegram comparte sus rutinas con
    el bot; si no (perfil web puro) se usa un espacio sintético por perfil.
    Devuelve None si el perfil no existe.
    """
    p = _db.obtener_perfil(perfil_id)
    if p is None:
        return None
    return p.get("telegram_chat_id") or f"web_{perfil_id}"


def _perfil_prediccion_desde_rutina(perfil: dict, rutina: dict) -> dict:
    """Perfil para predict_ensemble con la ventana de la rutina.

    Igual que el aviso diario del bot: la ventana a evaluar la define la rutina
    (hora_inicio + duración); el resto de factores vienen del perfil guardado.
    """
    p = {
        "sexo": perfil.get("sexo", "hombre"),
        "edad": perfil.get("edad"),
        "aclimatado": perfil.get("aclimatado", False),
        "nivel_actividad": "ligera",
        "duracion_actividad_h": rutina["hora_fin"] - rutina["hora_inicio"],
        "hora_inicio": rutina["hora_inicio"],
        "comorbilidades": set(perfil.get("comorbilidades") or []),
        "farmacos": set(perfil.get("farmacos") or []),
    }
    if perfil.get("porcentaje_grasa") is not None:
        p["porcentaje_grasa"] = perfil["porcentaje_grasa"]
    if perfil.get("fototipo") is not None:
        p["fototipo"] = perfil["fototipo"]
    if perfil.get("situacion_social"):
        p["situacion_social"] = set(perfil["situacion_social"])
    if rutina.get("ocupacion"):
        p["ocupacion"] = rutina["ocupacion"]
    if rutina.get("deporte"):
        p["deporte"] = rutina["deporte"]
        _aplicar_deporte_a_nivel(p)
    return p


def _temps_en_ventana(perfil_horario: list[dict], perfil_usuario: dict) -> list[float]:
    """Temperaturas previstas en las horas de la ventana de la rutina."""
    inicio = perfil_usuario.get("hora_inicio")
    duracion = perfil_usuario.get("duracion_actividad_h")
    if inicio is not None and duracion is not None:
        en_ventana = [
            h["temp"]
            for h in perfil_horario
            if inicio <= h["hora"] < inicio + duracion and h.get("temp") is not None
        ]
        if en_ventana:
            return en_ventana
    return [h["temp"] for h in perfil_horario if h.get("temp") is not None]


def _validar_hora_aviso(texto: str) -> str | None:
    """Valida 'HH:MM' y devuelve la hora normalizada 'HH:MM'; None si no vale."""
    t = texto.strip()
    if ":" not in t:
        return None
    hh, mm = t.split(":", 1)
    try:
        h, m = int(hh), int(mm)
    except ValueError:
        return None
    if not (0 <= h <= 23 and 0 <= m <= 59):
        return None
    return f"{h:02d}:{m:02d}"


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


@app.post("/api/rag/ask")
async def api_rag_ask(body: dict):
    query = body.get("query", "")
    k = body.get("k", 5)
    if not query.strip():
        return {"success": False, "error": "query vacía"}
    try:
        result = _db.ask_rag(query, k=k)
        return {"success": True, **result}
    except Exception as e:
        return {"success": False, "error": str(e)}


@app.post("/api/predict")
async def api_predict(body: dict, date: str | None = None):
    provincia = body.get("provincia", "Madrid")
    lat = body.get("lat")
    lon = body.get("lon")
    # Resolución del perfil horario en minutos por punto (5/15/30/60; default 60
    # = comportamiento histórico de un punto por hora). DATA-007.
    resolucion = body.get("resolucion", 60)
    raw_perfil = body.get("perfil") or {}
    perfil_id = raw_perfil.get("perfil_id")
    perfil = _normalize_perfil(raw_perfil)
    # El MET del deporte fija la intensidad antes de predecir (igual que el bot)
    _aplicar_deporte_a_nivel(perfil)

    target_date = None
    if date:
        try:
            from datetime import date as date_type, timedelta

            # Mismo horizonte que el forecast meteorológico (FORECAST-001): 7 días.
            from climasafeai.data.weather_fetcher import FORECAST_HORIZON_DAYS

            target_date = date_type.fromisoformat(date)
            today = date_type.today()
            if target_date < today:
                return {"error": f"La fecha {date} ya pasó. Solo se aceptan hoy o el futuro."}
            if (target_date - today).days > FORECAST_HORIZON_DAYS:
                return {
                    "error": f"Fecha {date} está a más de {FORECAST_HORIZON_DAYS} días vista. El forecast meteorológico cubre hasta {FORECAST_HORIZON_DAYS} días."
                }
        except ValueError:
            return {"error": f"Fecha inválida: '{date}'. Usa formato ISO: YYYY-MM-DD"}

    try:
        from climasafeai.models.ensemble import predict_ensemble

        result = predict_ensemble(
            lat=lat,
            lon=lon,
            provincia=provincia,
            perfil=perfil,
            target_date=target_date,
            resolucion=resolucion,
        )
    except Exception as exc:
        return {"error": str(exc)}

    # Predicciones auxiliares (comparativa de edades, simulaciones) mandan
    # `persistir: false`: son perfiles inventados a partir del real, así que
    # no deben crear filas en SQLite ni contar como consulta del usuario.
    if body.get("persistir", True):
        # Guardar perfil en SQLite (sin perfil_id ni alias en datos)
        #
        # Esto va DESPUÉS de la predicción a propósito: si el guardado falla, el
        # usuario ya tiene su riesgo calculado. Antes un campo de más aquí —`peso`,
        # que no es columna de `perfiles`— tumbaba la peticion entera con un 500
        # mudo, aunque la predicción hubiera salido bien.
        alias = raw_perfil.get("alias")
        try:
            datos_perfil = _sin_strings_vacios(
                {k: v for k, v in raw_perfil.items() if k not in ("perfil_id", "alias")}
            )
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
        except CampoDesconocidoError as exc:
            # El error se devuelve, no se esconde: un campo mal escrito en el
            # frontend tiene que salir a la luz, no tirarse en silencio.
            result["error_perfil"] = str(exc)
            logger.warning("Perfil no guardado: %s", exc)

        # Guardar consulta
        clase = result.get("clase_final_label", result.get("clase_final"))
        tipo = result.get("tipo", "calor")
        indice_orig = result.get("explicacion", {}).get("indice_original")
        indice_pers = result.get("explicacion", {}).get("indice_personalizado")
        _db.guardar_consulta(
            perfil_id=perfil_id,
            provincia=provincia,
            lat=lat,
            lon=lon,
            tipo_riesgo=tipo,
            indice_original=indice_orig,
            indice_personalizado=indice_pers,
            clase_final=clase,
        )

    result["perfil_id"] = perfil_id

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
            riesgo_horario_acumulado,
            recomendar_horario,
            pico_riesgo_actividad,
        )

        _ph = result["weather"].get("perfil_horario") or []
        _curva = riesgo_horario_acumulado(_ph, perfil)
        result["riesgo_horario"] = _curva
        result["riesgo_pico"] = pico_riesgo_actividad(_curva, perfil)
        result["recomendacion_horario"] = recomendar_horario(_ph, perfil)
    except Exception:
        pass

    return result


# ---------------------------------------------------------------------------
# UX-001 — Agente conversacional web (estilo SymptomAI)
#
# El agente pregunta de una en una y va construyendo el perfil de forma
# progresiva, como el cuestionario del bot de Telegram, pero para la GUI web.
# El estado vive en el cliente (paso + perfil + ubicacion), así que el
# endpoint es stateless: cada llamada recibe la respuesta y el estado
# acumulado, y devuelve la siguiente pregunta o, al terminar, la predicción
# por el MISMO camino que /api/predict (predict_ensemble + _normalize_perfil
# + _aplicar_deporte_a_nivel). No se duplica la lógica de riesgo.
# ---------------------------------------------------------------------------

_CHAT_CAMPOS_LABEL = {
    "ubicacion": "Ubicación",
    "edad": "Edad",
    "sexo": "Sexo",
    "aclimatado": "Aclimatado al calor",
    "nivel_actividad": "Intensidad de la actividad",
    "duracion_actividad_h": "Duración de la actividad",
    "hora_inicio": "Hora de inicio",
    "comorbilidades": "Comorbilidades",
    "farmacos": "Medicación",
    "situacion_social": "Situación social",
    "porcentaje_grasa": "% grasa corporal",
    "fototipo": "Fototipo",
}

_CHAT_ACTIVIDADES = ["reposo", "ligera", "moderada", "intensa", "muy_intensa"]

# Las opciones de sexo y actividad están fijas; las de comorbilidades,
# medicación y situación social salen del catálogo de /api/factores (la misma
# fuente que el formulario), para no duplicar listas a mano.
_CHAT_PASOS: list[dict] = [
    {
        "campo": "ubicacion",
        "pregunta": "¿Dónde vas a estar? Escribe el nombre del sitio (ej: Madrid) o las coordenadas 'lat,lon'.",
        "parse": "ubicacion",
    },
    {
        "campo": "edad",
        "pregunta": "¿Cuántos años tienes?",
        "parse": "edad",
    },
    {
        "campo": "sexo",
        "pregunta": "¿Cuál es tu sexo?",
        "opciones": [
            {"clave": "hombre", "nombre": "Hombre"},
            {"clave": "mujer", "nombre": "Mujer"},
        ],
        "parse": "opcion",
    },
    {
        "campo": "aclimatado",
        "pregunta": "¿Estás aclimatado al calor? (vives en clima cálido o llevas semanas de calor — hacer deporte no cuenta)",
        "opciones": [{"clave": "si", "nombre": "Sí"}, {"clave": "no", "nombre": "No"}],
        "parse": "siono",
    },
    {
        "campo": "nivel_actividad",
        "pregunta": "¿Qué intensidad tendrá la actividad?",
        "opciones": [
            {"clave": a, "nombre": a.replace("_", " ").title()} for a in _CHAT_ACTIVIDADES
        ],
        "parse": "opcion",
    },
    {
        "campo": "duracion_actividad_h",
        "pregunta": "¿Cuántas horas durará? (ej: 2, 3.5)",
        "parse": "numero_rango",
        "rango": (0, 24),
    },
    {
        "campo": "hora_inicio",
        "pregunta": "¿A qué hora empiezas? (de 0 a 24, ej: 8 o 14.5)",
        "parse": "numero_rango",
        "rango": (0, 24),
    },
    {
        "campo": "comorbilidades",
        "pregunta": "¿Tienes alguna de estas condiciones? Escribe los números separados por comas, o 'ninguna'.",
        "fuente": "comorbilidades",
        "multiselect": True,
        "opcional": True,
        "parse": "multiselect",
    },
    {
        "campo": "farmacos",
        "pregunta": "¿Tomas alguno de estos medicamentos? Escribe los números separados por comas, o 'ninguna'.",
        "fuente": "farmacos",
        "multiselect": True,
        "opcional": True,
        "parse": "multiselect",
    },
    {
        "campo": "situacion_social",
        "pregunta": "¿Cómo es tu situación habitual? Escribe los números separados por comas, o 'ninguna'.",
        "fuente": "situacional",
        "multiselect": True,
        "opcional": True,
        "parse": "multiselect",
    },
    {
        "campo": "porcentaje_grasa",
        "pregunta": "¿Sabes tu % de grasa corporal? Escríbelo, o 'saltar' si no lo sabes.",
        "opcional": True,
        "parse": "grasa",
    },
    {
        "campo": "fototipo",
        "pregunta": "¿Cuál es tu fototipo de piel? Elige el número (1-6) o 'saltar'.",
        "opciones": [{"clave": str(i), "nombre": f"Tipo {i}"} for i in range(1, 7)],
        "opcional": True,
        "parse": "opcion",
    },
]

_CHAT_NINGUNA = ("ninguna", "ninguno", "no", "nada", "terminar", "termine", "ok")


def _chat_opciones_paso(paso: dict) -> list[dict]:
    """Opciones de un paso: estáticas o del catálogo de factores implementados.

    Misma fuente que el formulario web (/api/factores); el 'alcohol' de
    situación social no se ofrece (el formulario lo oculta igualmente).
    """
    if paso.get("multiselect"):
        factores = _get_implemented_factors()
        items = (factores.get("calor") or {}).get(paso["fuente"]) or []
        if paso["fuente"] == "situacional":
            items = [f for f in items if f.get("clave") != "alcohol"]
        return [{"clave": f["clave"], "nombre": f["nombre"]} for f in items]
    return paso.get("opciones") or []


def _chat_parse_ubicacion(texto: str) -> tuple[Any, str | None]:
    """'lat,lon' → dict con coordenadas; cualquier otro texto → provincia."""
    m = re.match(r"^(-?\d{1,3}(?:\.\d+)?)\s*[,;]\s*(-?\d{1,3}(?:\.\d+)?)$", texto)
    if m:
        lat, lon = float(m.group(1)), float(m.group(2))
        if not (-90 <= lat <= 90 and -180 <= lon <= 180):
            return None, "Coordenadas fuera de rango: lat entre -90 y 90, lon entre -180 y 180."
        return {"lat": lat, "lon": lon, "provincia": f"{lat},{lon}"}, None
    if len(texto) < 2:
        return (
            None,
            "Escribe el nombre del sitio (ej: Madrid) o las coordenadas 'lat,lon' (ej: 40.4168,-3.7038).",
        )
    return {"provincia": texto, "lat": None, "lon": None}, None


def _chat_parse_edad(texto: str) -> tuple[Any, str | None]:
    try:
        edad = float(texto.replace(",", "."))
    except ValueError:
        return None, "La edad debe ser un número (ej: 45)."
    if not 0 <= edad <= 120:
        return None, "La edad debe estar entre 0 y 120 años."
    return int(edad), None


def _chat_parse_numero(texto: str, minimo: float, maximo: float) -> tuple[Any, str | None]:
    try:
        n = float(texto.replace(",", "."))
    except ValueError:
        return None, f"Debe ser un número entre {minimo:g} y {maximo:g}."
    if not minimo <= n <= maximo:
        return None, f"Debe estar entre {minimo:g} y {maximo:g}."
    return n, None


def _chat_parse_siono(texto: str) -> tuple[Any, str | None]:
    t = texto.strip().lower()
    if t in ("si", "sí", "s", "true", "1"):
        return True, None
    if t in ("no", "n", "false", "0"):
        return False, None
    return None, "Responde 'si' o 'no'."


def _chat_parse_opcion(texto: str, opciones: list[dict]) -> tuple[Any, str | None]:
    t = texto.strip().lower()
    if not opciones:
        return None, "No hay opciones disponibles para esta pregunta."
    for o in opciones:
        if t == o["clave"].lower() or t == o["nombre"].lower():
            return o["clave"], None
    nombres = ", ".join(o["clave"] for o in opciones)
    return None, f"No reconozco '{texto}'. Opciones: {nombres}."


def _chat_parse_multiselect(texto: str, opciones: list[dict]) -> tuple[Any, str | None]:
    t = texto.strip().lower()
    if t in _CHAT_NINGUNA:
        return set(), None
    if not opciones:
        return None, "No hay opciones disponibles; escribe 'ninguna' para continuar."
    claves: list[str] = []
    for parte in re.split(r"[,;\s]+", t):
        if not parte:
            continue
        if parte.isdigit():
            idx = int(parte)
            if not (1 <= idx <= len(opciones)):
                return None, f"El número {parte} no es una opción válida (1-{len(opciones)})."
            claves.append(opciones[idx - 1]["clave"])
        else:
            match = next(
                (
                    o
                    for o in opciones
                    if o["clave"].lower() == parte or o["nombre"].lower() == parte
                ),
                None,
            )
            if match is None:
                return (
                    None,
                    f"No reconozco '{parte}'. Usa los números (1-{len(opciones)}) o los nombres de la lista.",
                )
            claves.append(match["clave"])
    return set(claves), None


def _chat_parse(paso: dict, mensaje: str) -> tuple[Any, str | None]:
    """Devuelve (valor, error). En pasos opcionales, 'saltar' → (None, None)."""
    texto = mensaje.strip()
    if paso.get("opcional") and texto.lower() in ("saltar", "skip"):
        return None, None
    metodo = paso["parse"]
    if metodo == "ubicacion":
        return _chat_parse_ubicacion(texto)
    if metodo == "edad":
        return _chat_parse_edad(texto)
    if metodo == "numero_rango":
        return _chat_parse_numero(texto, *paso["rango"])
    if metodo == "siono":
        return _chat_parse_siono(texto)
    if metodo == "opcion":
        return _chat_parse_opcion(texto, _chat_opciones_paso(paso))
    if metodo == "multiselect":
        return _chat_parse_multiselect(texto, _chat_opciones_paso(paso))
    if metodo == "grasa":
        return _chat_parse_numero(texto, 0, 70)
    return None, "Pregunta no implementada."


def _chat_formato_valor(campo: str, valor, opciones: list[dict]) -> str:
    """Texto legible del valor confirmado (para el eco de la respuesta)."""
    if campo == "aclimatado":
        return "Sí" if valor else "No"
    if campo == "sexo":
        return valor.title()
    if campo == "duracion_actividad_h":
        return f"{valor:g} h"
    if campo == "hora_inicio":
        return f"{valor:g} h"
    if campo == "porcentaje_grasa":
        return f"{valor:g}%"
    if campo == "fototipo":
        return f"Tipo {valor}"
    if isinstance(valor, (set, list)):
        if not valor:
            return "Ninguna"
        nombres = {o["clave"]: o["nombre"] for o in opciones}
        return ", ".join(nombres.get(v, v) for v in sorted(valor))
    return str(valor)


def _chat_respuesta_pregunta(paso: int, perfil: dict, ubicacion: dict | None) -> dict:
    """Respuesta con la pregunta del paso y el estado acumulado."""
    p = _CHAT_PASOS[paso]
    return {
        "paso": paso,
        "total": len(_CHAT_PASOS),
        "campo": p["campo"],
        "pregunta": p["pregunta"],
        "opciones": _chat_opciones_paso(p),
        "multiselect": bool(p.get("multiselect")),
        "opcional": bool(p.get("opcional")),
        "perfil": perfil,
        "ubicacion": ubicacion,
    }


def _chat_predecir(perfil: dict, ubicacion: dict | None) -> dict:
    """Mismo camino de predicción que /api/predict: normaliza y llama al ensemble."""
    from climasafeai.models.ensemble import predict_ensemble

    perfil = _normalize_perfil(perfil)
    _aplicar_deporte_a_nivel(perfil)

    lat = (ubicacion or {}).get("lat")
    lon = (ubicacion or {}).get("lon")
    provincia = (ubicacion or {}).get("provincia") or "Madrid"

    result = predict_ensemble(lat=lat, lon=lon, provincia=provincia, perfil=perfil)

    result["perfil_usuario"] = perfil
    result["weather"] = _get_weather_summary(result)

    for mod_name, mod_res in result.get("modelos", {}).items():
        if isinstance(mod_res, dict):
            mod_res.pop("_X", None)
    if "error" in result.get("modelos", {}).get("LSTM", {}):
        del result["modelos"]["LSTM"]["error"]

    # Curva de riesgo por hora y recomendación de horario, como /api/predict.
    try:
        from climasafeai.features.personalizacion import (
            pico_riesgo_actividad,
            recomendar_horario,
            riesgo_horario_acumulado,
        )

        _ph = result["weather"].get("perfil_horario") or []
        _curva = riesgo_horario_acumulado(_ph, perfil)
        result["riesgo_horario"] = _curva
        result["riesgo_pico"] = pico_riesgo_actividad(_curva, perfil)
        result["recomendacion_horario"] = recomendar_horario(_ph, perfil)
    except Exception:
        pass

    return result


def _chat_mensaje_final(resultado: dict, ubicacion: dict | None) -> str:
    """Resumen del agente: frases del parte + recomendación contextual.

    Las frases del parte son las mismas que redacta el bot (lineas_parte) y la
    recomendación es la de una línea adaptada al canal dominante. Si el
    formateo falla (resultado mínimo), se degrada a la clase y las
    recomendaciones crudas — el resultado de la predicción no se pierde.
    """
    try:
        from climasafeai.llm.rag_qwen import lineas_parte
        from climasafeai.models.recomendaciones import recomendacion_resumen

        lugar = (ubicacion or {}).get("provincia")
        lineas = list(lineas_parte(resultado, lugar))
        lineas.append(recomendacion_resumen(resultado))
        return "\n".join(lineas)
    except Exception:
        recs = resultado.get("recomendaciones") or []
        nivel = resultado.get("clase_final_label") or "?"
        texto = f"Nivel de riesgo: {nivel}."
        if recs:
            texto += "\n" + "\n".join(f"- {r}" for r in recs)
        return texto


@app.post("/api/chat")
async def api_chat(body: dict):
    """Turno de la conversación del agente web (UX-001).

    Body: {"mensaje": str, "estado": {"paso", "perfil", "ubicacion"}}.
    Sin mensaje devuelve la pregunta actual (la primera si no hay estado).
    Al contestar la última pregunta devuelve "fin": true con el resultado de
    predict_ensemble — el mismo camino que usa POST /api/predict.
    """
    mensaje = (body.get("mensaje") or "").strip()
    estado = body.get("estado") or {}
    try:
        paso = int(estado.get("paso") or 0)
    except (TypeError, ValueError):
        paso = 0
    perfil_raw = estado.get("perfil")
    perfil = dict(perfil_raw) if isinstance(perfil_raw, dict) else {}
    ubicacion = estado.get("ubicacion")

    if not mensaje:
        paso_actual = paso if 0 <= paso < len(_CHAT_PASOS) else 0
        return _chat_respuesta_pregunta(paso_actual, perfil, ubicacion)

    if mensaje.lower() in ("cancelar", "cancel", "salir", "exit"):
        return {
            "cancelado": True,
            "respuesta": "✕ Conversación cancelada. Pulsa 'Empezar de nuevo' para reiniciar.",
            "paso": 0,
            "total": len(_CHAT_PASOS),
            "perfil": {},
            "ubicacion": None,
        }

    if not (0 <= paso < len(_CHAT_PASOS)):
        return {"error": "La conversación ya terminó. Empieza una nueva."}

    p = _CHAT_PASOS[paso]
    valor, error = _chat_parse(p, mensaje)
    if error:
        return {**_chat_respuesta_pregunta(paso, perfil, ubicacion), "error": error}

    campo = p["campo"]
    if campo == "ubicacion":
        ubicacion = valor
        lugar = valor["provincia"] if valor["lat"] is None else f"{valor['lat']}, {valor['lon']}"
        respuesta = f"✓ {_CHAT_CAMPOS_LABEL[campo]}: {lugar}"
    elif valor is None:  # 'saltar' en un paso opcional
        respuesta = f"✓ {_CHAT_CAMPOS_LABEL[campo]}: saltado"
    else:
        perfil[campo] = valor
        respuesta = f"✓ {_CHAT_CAMPOS_LABEL[campo]}: {_chat_formato_valor(campo, valor, _chat_opciones_paso(p))}"

    paso += 1

    if paso >= len(_CHAT_PASOS):
        try:
            resultado = _chat_predecir(perfil, ubicacion)
        except Exception as exc:
            return {
                "fin": True,
                "respuesta": respuesta,
                "error": f"No se pudo calcular el riesgo: {exc}",
                "perfil": perfil,
                "ubicacion": ubicacion,
            }
        return {
            "fin": True,
            "respuesta": respuesta,
            "perfil": perfil,
            "ubicacion": ubicacion,
            "resultado": resultado,
            "mensaje_final": _chat_mensaje_final(resultado, ubicacion),
        }

    return {**_chat_respuesta_pregunta(paso, perfil, ubicacion), "respuesta": respuesta}


@app.post("/api/predict/semanal")
async def api_predict_semanal(body: dict):
    """Tendencia semanal de riesgo con banda de confianza (FORECAST-001).

    Devuelve la serie día a día (hoy + 6) con la banda procedente del prediction
    set conformal de cada día, y `completo=False` + `forecast_hasta` cuando el
    forecast meteorológico no cubre los 7 días. No persiste nada: es una vista.
    """
    from climasafeai.models.ensemble import prediccion_semanal

    provincia = body.get("provincia", "Madrid")
    lat = body.get("lat")
    lon = body.get("lon")
    resolucion = body.get("resolucion", 60)
    raw_perfil = body.get("perfil") or {}
    perfil = _normalize_perfil(raw_perfil)
    _aplicar_deporte_a_nivel(perfil)

    try:
        return prediccion_semanal(
            lat=lat,
            lon=lon,
            provincia=provincia,
            perfil=perfil,
            resolucion=resolucion,
        )
    except Exception as exc:
        return {"error": str(exc)}


EDADES_COMPARATIVA = (25, 55, 65, 75, 85)


@app.post("/api/curvas-edad")
async def api_curvas_edad(body: dict):
    """Curva de riesgo horario del mismo perfil a varias edades.

    La curva solo depende del perfil horario (HI de cada hora) y de los factores
    de personalización, así que las N curvas salen de UNA sola descarga de meteo
    y sin ejecutar los modelos. Si el cliente ya tiene el `perfil_horario` de la
    predicción principal, lo manda y no hace falta ni descargar.

    No persiste nada: son perfiles derivados del real para comparar, no
    consultas del usuario.
    """
    from climasafeai.features.personalizacion import riesgo_horario_acumulado
    from climasafeai.models.ensemble import (
        PERS_THRESHOLD_PELIGRO,
        perfil_horario_desde_df,
    )
    from climasafeai.models.predict_model import CLASS_THRESHOLDS_RECOMENDADOS

    perfil_base = _normalize_perfil(body.get("perfil") or {})
    for _k in ("_perfil_horario", "perfil_id", "alias"):
        perfil_base.pop(_k, None)

    edades = body.get("edades") or list(EDADES_COMPARATIVA)
    try:
        edades = sorted({int(e) for e in edades if 0 < int(e) <= 120})[:8]
    except (TypeError, ValueError):
        return {"error": "El campo 'edades' debe ser una lista de números."}
    if not edades:
        return {"error": "No hay ninguna edad válida que comparar (1-120)."}

    perfil_horario = body.get("perfil_horario")
    if not perfil_horario:
        target_date = None
        if body.get("fecha"):
            try:
                from datetime import date as date_type

                target_date = date_type.fromisoformat(body["fecha"])
            except ValueError:
                return {"error": f"Fecha inválida: '{body['fecha']}'. Usa formato ISO: YYYY-MM-DD"}
        try:
            from climasafeai.data.weather_fetcher import fetch_weather_data

            weather = fetch_weather_data(
                lat=body.get("lat"),
                lon=body.get("lon"),
                provincia=body.get("provincia", "Madrid"),
                target_date=target_date,
            )
            perfil_horario = perfil_horario_desde_df(
                weather.get("df_hora"), target_date=weather.get("target_date")
            )
        except Exception as exc:
            return {"error": str(exc)}
    if not perfil_horario:
        return {"error": "No hay perfil horario disponible para esta ubicación."}

    curvas = []
    for edad in edades:
        curva = riesgo_horario_acumulado(perfil_horario, {**perfil_base, "edad": edad})
        if not curva:
            continue
        pico = max(curva, key=lambda e: e["riesgo"])
        curvas.append(
            {
                "edad": edad,
                "curva": curva,
                "pico": round(pico["riesgo"], 4),
                "hora_pico": pico["hora"],
            }
        )

    return {
        "curvas": curvas,
        "horas": [e["hora"] for e in perfil_horario],
        # Mismos cortes que usa el ensemble sobre la probabilidad personalizada,
        # para que las líneas de la gráfica coincidan con la clase que se muestra.
        "umbrales": {
            "precaucion": CLASS_THRESHOLDS_RECOMENDADOS.get("calor", {}).get("t1", 0.25),
            "peligro": PERS_THRESHOLD_PELIGRO,
        },
    }


FACTORES_COEF = {
    "grasa_alta": {"label": "Obesidad/grasa alta", "coef": 1.08},
    "cardiovascular": {"label": "Cardiovascular", "coef": 1.4},
    "diabetes": {"label": "Diabetes", "coef": 1.2},
    "respiratoria": {"label": "Respiratoria", "coef": 1.3},
    "mental": {"label": "Salud mental", "coef": 1.8},
    "no_aclimatados": {"label": "No aclimatados", "coef": 1.6},
}

FACTORES_ETIQUETAS = {k: v["label"] for k, v in FACTORES_COEF.items()}

# CSV-001 — orgullo colectivo. Factor multiplicativo sobre las ODDS del riesgo
# individual que se aplica solo cuando el tipo de actividad del grupo es
# competición o deporte: en ese contexto la gente se exige más de lo que haría
# sola y genera más calor metabólico. Valor ×1.2: un salto de carga comparable
# al de un escalón de nivel_actividad (ligera ×1.1 → moderada ×1.3) sin duplicar
# el factor de actividad que el deporte ya aplica. No es un RR publicado: es un
# modificador de exposición situacional, igual que los factores de NIOSH que ya
# usa el proyecto. Documentado en
# documentacion/riesgo/personalizacion_individual.md (sección "Orgullo colectivo").
ORGULLO_COLECTIVO = 1.2
# Valores de `tipo_actividad` que disparan el orgullo colectivo. Mismos
# valores que la rama de etiqueta de /api/riesgo-colectivo ("competicion",
# "deporte") para que el frontend no tenga que distinguir.
TIPOS_ACTIVIDAD_COMPETICION = ("competicion", "deporte")


def _aplicar_orgullo_colectivo(prob: float, tipo_actividad: str) -> tuple[float, float]:
    """Aplica el orgullo colectivo a la probabilidad, en odds (CSV-001).

    Solo cuando ``tipo_actividad`` es competición/deporte; cualquier otro tipo
    (o "sin tipo") devuelve la probabilidad intacta con factor 1.0.
    """
    if tipo_actividad not in TIPOS_ACTIVIDAD_COMPETICION or not 0 < prob < 1:
        return prob, 1.0
    odds = prob / (1.0 - prob)
    odds_ajustadas = odds * ORGULLO_COLECTIVO
    return odds_ajustadas / (1.0 + odds_ajustadas), ORGULLO_COLECTIVO


def _calcular_riesgo_colectivo(body: dict) -> dict:
    """Núcleo del cálculo de riesgo colectivo (modo 'numero').
    Devuelve el resultado completo + datos intermedios para contrafactuales."""
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

    cantidad = int(body.get("cantidad", 100))
    edad_min = int(body.get("edad_min", 18))
    edad_max = int(body.get("edad_max", 80))
    pct_hombres = int(body.get("pct_hombres", 50))
    actividad = body.get("actividad", "ligera")
    hora_inicio = float(body.get("hora_inicio", 10))
    duracion = float(body.get("duracion", 2))
    aclimatado = body.get("aclimatado")

    def _prevalencia(edad: float) -> dict:
        e = max(18, min(90, edad))
        return {
            "grasa_alta": min(45, 20 + (e - 20) * 0.35),
            "cardiovascular": min(30, 2 + (e - 20) * 0.35),
            "diabetes": min(25, 1 + (e - 20) * 0.30),
            "respiratoria": min(12, 3 + (e - 20) * 0.12),
            "mental": min(15, 8 - abs(e - 45) * 0.15),
            "no_aclimatados": 40.0,
        }

    def _factor_grupo(pct: float, coef: float) -> float:
        if pct <= 0:
            return 1.0
        return 1.0 + (pct / 100.0) * (coef - 1.0)

    rangos_edad = [(18, 30), (30, 45), (45, 60), (60, 75), (75, 90)]
    rangos_edad = [(a, b) for a, b in rangos_edad if a < edad_max and b > edad_min]
    if not rangos_edad:
        rangos_edad = [(edad_min, edad_max)]

    from climasafeai.models.ensemble import predict_ensemble

    total_rango_pct = sum(min(b, edad_max) - max(a, edad_min) for a, b in rangos_edad)
    pcts_ponderados = {k: 0.0 for k in FACTORES_COEF}
    for a, b in rangos_edad:
        solapamiento = max(0, min(b, edad_max) - max(a, edad_min))
        if solapamiento <= 0:
            continue
        peso = solapamiento / total_rango_pct if total_rango_pct else 0
        edad_med_rango = (max(a, edad_min) + min(b, edad_max)) / 2
        prev = _prevalencia(edad_med_rango)
        for k in pcts_ponderados:
            pcts_ponderados[k] += prev[k] * peso

    # Permitir override explícito desde el body (útil para contrafactuales)
    for k in pcts_ponderados:
        bk = f"pct_{k}"
        if bk in body:
            try:
                pcts_ponderados[k] = float(body[bk])
            except (ValueError, TypeError):
                pass

    factor_extra = 1.0
    factores_detalle = []
    for k, cfg in FACTORES_COEF.items():
        pct = pcts_ponderados[k]
        mult = _factor_grupo(pct, cfg["coef"])
        factor_extra *= mult
        if mult > 1.001:
            factores_detalle.append(
                {
                    "clave": k,
                    "nombre": cfg["label"],
                    "pct": round(pct, 1),
                    "coef": cfg["coef"],
                    "multiplicador": round(mult, 3),
                }
            )
    factor_extra = min(factor_extra, 2.5)

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
            # El MET del deporte fija la intensidad antes de predecir (igual que el bot)
            _aplicar_deporte_a_nivel(perfil)

            try:
                pred = predict_ensemble(
                    lat=lat, lon=lon, provincia=provincia, perfil=perfil, target_date=date_obj
                )
                if primer_pred_num is None:
                    primer_pred_num = pred
                clase = pred.get("clase_final", 0)
                prob_base = pred.get("perfil", {}).get("calor", {}).get("prob_personalizada", 0)
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

            resultados_rangos.append(
                {
                    "rango": f"{edad_med}a {sexo[0]}",
                    "edad": edad_med,
                    "sexo": sexo,
                    "seguros": n_sexo if clase == 0 else 0,
                    "precaucion": n_sexo if clase == 1 else 0,
                    "peligro": n_sexo if clase == 2 else 0,
                    "prob": round(prob, 4),
                    "n_personas": n_sexo,
                }
            )

    total = total_seguros + total_precaucion + total_peligro
    pct_peligro = round(total_peligro / total * 100, 1) if total else 0

    return {
        "total_personas": total,
        "seguros": total_seguros,
        "en_precaucion": total_precaucion,
        "en_peligro": total_peligro,
        "pct_peligro": pct_peligro,
        "factor_extra": round(factor_extra, 3),
        "factores_detalle": factores_detalle,
        "rangos": resultados_rangos,
        "primer_pred": primer_pred_num,
        "cantidad": cantidad,
        "edad_min": edad_min,
        "edad_max": edad_max,
        "pct_hombres": pct_hombres,
        "actividad": actividad,
        "hora_inicio": hora_inicio,
        "duracion": duracion,
        "aclimatado": aclimatado,
        "pcts": pcts_ponderados,
    }


@app.post("/api/riesgo-colectivo")
async def api_riesgo_colectivo(body: dict):
    """Calcula riesgo para un grupo."""
    tipo = body.get("tipo", "numero")
    from climasafeai.features.personalizacion import (
        riesgo_horario_acumulado,
        recomendar_horario,
        pico_riesgo_actividad,
    )

    if tipo == "numero":
        c = _calcular_riesgo_colectivo(body)
        total = c["total_personas"]
        total_peligro = c["en_peligro"]
        total_precaucion = c["en_precaucion"]
        total_seguros = c["seguros"]
        pct_peligro = c["pct_peligro"]
        factor_extra = c["factor_extra"]
        primer_pred_num = c["primer_pred"]
        resultados_rangos = c["rangos"]
        pcts = c["pcts"]

        factores_activos = [f"{k}={pcts[k]:.0f}%" for k in sorted(pcts) if pcts[k] > 0]
        sufijo_extra = f" · Factor extra grupo: x{factor_extra:.2f}" if factor_extra > 1.01 else ""

        _comorb_map = {
            "cardiovascular": "cardiovascular",
            "diabetes": "diabetes",
            "respiratoria": "respiratoria",
            "mental": "mental",
        }

        edad_max = c["edad_max"]
        actividad = c["actividad"]
        aclimatado_val = c["aclimatado"]
        comorb_mapa = {_comorb_map[k] for k in _comorb_map if pcts.get(k, 0) >= 50}
        perfil_mapa = {
            "edad": edad_max,
            "sexo": "hombre",
            "nivel_actividad": actividad,
            "hora_inicio": c["hora_inicio"],
            "duracion_actividad_h": c["duracion"],
        }
        if comorb_mapa:
            perfil_mapa["comorbilidades"] = comorb_mapa
        if pcts.get("no_aclimatados", 0) >= 50 or aclimatado_val == "no":
            perfil_mapa["aclimatado"] = False
        elif aclimatado_val == "si":
            perfil_mapa["aclimatado"] = True
        if body.get("ocupacion"):
            perfil_mapa["ocupacion"] = body["ocupacion"]
        if body.get("deporte"):
            perfil_mapa["deporte"] = body["deporte"]
        _aplicar_deporte_a_nivel(perfil_mapa)

        _hourly_num = (
            primer_pred_num.get("weather", {}).get("perfil_horario", []) if primer_pred_num else []
        )
        grp_curva = riesgo_horario_acumulado(_hourly_num, perfil_mapa)
        grp_reco = recomendar_horario(_hourly_num, perfil_mapa)

        demografico = _calc_demografico(resultados_rangos, total)

        return {
            "total_personas": total,
            "seguros": total_seguros,
            "en_precaucion": total_precaucion,
            "en_peligro": total_peligro,
            "pct_peligro": pct_peligro,
            "clase": "PELIGRO"
            if pct_peligro > 20
            else ("PRECAUCION" if pct_peligro > 5 else "SEGURO"),
            "factor_extra": round(factor_extra, 3),
            "factores_grupo": factores_activos,
            "factores_detalle": c["factores_detalle"],
            "mensaje": f"De {total} personas, ~{total_peligro} en peligro, ~{total_precaucion} en precaución"
            + sufijo_extra,
            "rangos": resultados_rangos,
            "demografico": demografico,
            "resumen": _generar_resumen(
                pct_peligro,
                total_peligro,
                total_precaucion,
                total_seguros,
                factor_extra,
                c["factores_detalle"],
                c["actividad"],
            ),
            "perfil_mapa": perfil_mapa,
            "riesgo_horario": grp_curva,
            "recomendacion_horario": grp_reco,
            "weather": _get_weather_summary(primer_pred_num) if primer_pred_num else None,
        }

    elif tipo == "etiqueta":
        tag = body.get("tag", "").strip()
        if not tag:
            return {"error": "tag requerido"}

        # La ubicación y la fecha salen del body igual que en el modo número:
        # aquí no se pasa por _calcular_riesgo_colectivo, así que hay que
        # resolverlas en esta rama (antes no existían y cada persona fallaba).
        lat = body.get("lat")
        lon = body.get("lon")
        provincia = body.get("provincia", "Madrid")
        date_obj = None
        if body.get("fecha"):
            try:
                from datetime import date as date_type

                date_obj = date_type.fromisoformat(body["fecha"])
            except ValueError:
                pass
        from climasafeai.models.ensemble import predict_ensemble

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
        perfil_mapa = None  # perfil más restrictivo (para el mapa de zona)
        peor_prob_mapa = -1.0
        for p in perfiles:
            try:
                perfil = {
                    k: v
                    for k, v in p.items()
                    if k not in ("id", "alias", "tags", "created_at", "updated_at")
                }
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
                # El MET del deporte fija la intensidad antes de predecir (igual que el bot)
                _aplicar_deporte_a_nivel(perfil)
                pred = predict_ensemble(
                    lat=lat, lon=lon, provincia=provincia, perfil=perfil, target_date=date_obj
                )
                hourly = pred.get("weather", {}).get("perfil_horario", [])
                if primer_pred is None:
                    primer_pred = pred
                prob_r = pred.get("perfil", {}).get("calor", {}).get("prob_personalizada", 0) or 0
                if prob_r > peor_prob_mapa:
                    peor_prob_mapa = prob_r
                    perfil_mapa = dict(perfil)
                    perfil_mapa["_alias"] = p.get("alias", "?")
                _curva = riesgo_horario_acumulado(hourly, perfil)
                resultados.append(
                    {
                        "alias": p.get("alias", "?"),
                        "edad": p.get("edad"),
                        "sexo": p.get("sexo"),
                        "clase": pred.get("clase_final_label", "SEGURO"),
                        "prob_riesgo": pred.get("perfil", {})
                        .get("calor", {})
                        .get("prob_personalizada", 0),
                        "riesgo_pico": pico_riesgo_actividad(_curva, perfil),
                        "perfil_horario": hourly,
                        "riesgo_horario": _curva,
                        "recomendacion_horario": recomendar_horario(hourly, perfil),
                        "explicacion": pred.get("explicacion"),
                        "recomendaciones": pred.get("recomendaciones", []),
                        "factores": pred.get("perfil", {}).get("calor", {}).get("factores", []),
                    }
                )
            except Exception as e:
                resultados.append({"alias": p.get("alias", "?"), "error": str(e)})

        seguros = sum(1 for r in resultados if r.get("clase") == "SEGURO")
        precaucion = sum(1 for r in resultados if r.get("clase") == "PRECAUCION")
        peligro = sum(1 for r in resultados if r.get("clase") == "PELIGRO")

        # Curva y recomendación de horario del GRUPO, según el perfil más
        # restrictivo (protege al más vulnerable de la cuadrilla).
        _hourly_grp = (
            primer_pred.get("weather", {}).get("perfil_horario", []) if primer_pred else []
        )
        grp_curva = riesgo_horario_acumulado(_hourly_grp, perfil_mapa) if perfil_mapa else []
        grp_reco = recomendar_horario(_hourly_grp, perfil_mapa) if perfil_mapa else None

        return {
            "total_personas": len(resultados),
            "seguros": seguros,
            "en_precaucion": precaucion,
            "en_peligro": peligro,
            "pct_peligro": round(peligro / len(resultados) * 100, 1) if resultados else 0,
            "detalle": resultados,
            "perfil_mapa": perfil_mapa,  # el más restrictivo: el mapa de zona lo usa
            "riesgo_horario": grp_curva,  # curva del peor caso del grupo
            "recomendacion_horario": grp_reco,  # horario óptimo del grupo
            "weather": _get_weather_summary(primer_pred) if primer_pred else None,
        }

    return {"error": "tipo no válido"}


# ── CSV-001: riesgo colectivo desde un CSV de personas ──────────────────────
# Columnas obligatorias en el CSV; el resto son opcionales y se mapean al
# perfil que entiende predict_ensemble (grasa→porcentaje_grasa,
# duracion/duracion_h→duracion_actividad_h, actividad→nivel_actividad).

CSV_COLUMNAS_REQUERIDAS = ("nombre", "edad", "sexo")
_VALORES_ACLMATADO = {
    "si": True,
    "sí": True,
    "true": True,
    "1": True,
    "no": False,
    "false": False,
    "0": False,
}


def _leer_numero(valor, campo, errores, minimo=None, maximo=None):
    """Columna numérica → float (None si vacía). Apunta error legible."""
    texto = (valor or "").strip()
    if not texto:
        return None
    try:
        n = float(texto.replace(",", "."))
    except ValueError:
        errores.append(f"'{campo}' debe ser un número")
        return None
    if minimo is not None and n < minimo:
        errores.append(f"'{campo}' no puede ser menor que {minimo}")
    if maximo is not None and n > maximo:
        errores.append(f"'{campo}' no puede ser mayor que {maximo}")
    return n


def _parsear_csv_personas(texto: str) -> list[dict]:
    """Convierte el CSV de personas en perfiles validados para predict_ensemble.

    Devuelve una lista de {"nombre": str, "perfil": dict}. Cualquier problema
    (CSV ilegible, columnas faltantes, valor inválido) lanza ValueError con un
    mensaje en español que dice QUÉ falla y EN QUÉ fila — el endpoint lo
    convierte en 400 explicativo, nunca en un 500 mudo.
    """
    import csv as csv_mod
    from io import StringIO

    try:
        filas = list(csv_mod.DictReader(StringIO(texto)))
    except csv_mod.Error as exc:
        raise ValueError(f"El CSV no se puede leer: {exc}") from exc

    if not filas:
        raise ValueError("El CSV está vacío: no hay ninguna persona.")

    cabecera = [c for c in filas[0].keys() if c]
    faltantes = [c for c in CSV_COLUMNAS_REQUERIDAS if c not in cabecera]
    if faltantes:
        raise ValueError(
            "Faltan columnas en el CSV: "
            + ", ".join(faltantes)
            + f". Columnas mínimas: {', '.join(CSV_COLUMNAS_REQUERIDAS)}."
        )

    personas = []
    for i, fila in enumerate(filas, start=2):
        nombre = (fila.get("nombre") or "").strip() or f"persona_{i}"
        errores: list[str] = []

        edad = _leer_numero(fila.get("edad"), "edad", errores, minimo=0, maximo=120)
        if edad is None:
            errores.append("'edad' no puede estar vacía")
        sexo = (fila.get("sexo") or "").strip().lower()
        if sexo not in ("hombre", "mujer"):
            errores.append("'sexo' debe ser 'hombre' o 'mujer'")
        grasa = _leer_numero(
            fila.get("grasa") or fila.get("porcentaje_grasa"),
            "grasa",
            errores,
            minimo=0,
            maximo=100,
        )
        hora_inicio = _leer_numero(
            fila.get("hora_inicio"), "hora_inicio", errores, minimo=0, maximo=24
        )
        duracion = _leer_numero(
            fila.get("duracion") or fila.get("duracion_h") or fila.get("duracion_actividad_h"),
            "duracion",
            errores,
            minimo=0,
            maximo=24,
        )

        aclimatado = None
        texto_acl = (fila.get("aclimatado") or "").strip().lower()
        if texto_acl:
            if texto_acl not in _VALORES_ACLMATADO:
                errores.append("'aclimatado' debe ser si/no")
            else:
                aclimatado = _VALORES_ACLMATADO[texto_acl]

        comorbilidades = {
            c.strip()
            for c in (fila.get("comorbilidades") or "").replace(";", "|").split("|")
            if c.strip()
        }

        if errores:
            raise ValueError(f"Fila {i} ({nombre}): " + "; ".join(errores) + ".")

        perfil = {"edad": int(edad), "sexo": sexo}
        if grasa is not None:
            perfil["porcentaje_grasa"] = grasa
        if (fila.get("nivel_actividad") or "").strip():
            perfil["nivel_actividad"] = fila["nivel_actividad"].strip()
        if hora_inicio is not None:
            perfil["hora_inicio"] = hora_inicio
        if duracion is not None:
            perfil["duracion_actividad_h"] = duracion
        if (fila.get("deporte") or "").strip():
            perfil["deporte"] = fila["deporte"].strip()
        if aclimatado is not None:
            perfil["aclimatado"] = aclimatado
        if (fila.get("fototipo") or "").strip():
            perfil["fototipo"] = fila["fototipo"].strip()
        if comorbilidades:
            perfil["comorbilidades"] = comorbilidades
        # El MET del deporte fija la intensidad antes de predecir (igual que el bot)
        _aplicar_deporte_a_nivel(perfil)

        personas.append({"nombre": nombre, "perfil": perfil})

    return personas


@app.post("/api/riesgo-colectivo/csv")
async def api_riesgo_colectivo_csv(body: dict):
    """Riesgo individual + estadísticas de grupo desde un CSV de personas (CSV-001).

    Body: {"csv": "...", "lat", "lon", "provincia"?, "fecha"?, "tipo_actividad"?}.

    Columnas del CSV: nombre, edad, sexo (obligatorias); opcionales:
    grasa/porcentaje_grasa, nivel_actividad/actividad, hora_inicio,
    duracion/duracion_h/duracion_actividad_h, deporte, aclimatado (si/no),
    fototipo, comorbilidades (separadas por | o ;).

    Devuelve una fila de riesgo por persona (``detalle``) y el bloque de
    estadísticas del grupo. Cuando ``tipo_actividad`` es competicion/deporte se
    aplica el factor orgullo colectivo (ver ORGULLO_COLECTIVO) a la
    probabilidad de cada persona.
    """
    texto = body.get("csv")
    if not isinstance(texto, str) or not texto.strip():
        raise HTTPException(400, "Falta el campo 'csv' con el contenido del fichero.")

    lat = body.get("lat")
    lon = body.get("lon")
    if lat is None or lon is None:
        raise HTTPException(
            400, "Faltan lat y lon: hacen falta la ubicación y la fecha para predecir."
        )
    provincia = body.get("provincia", "Madrid")

    date_obj = None
    if body.get("fecha"):
        try:
            from datetime import date as date_type

            date_obj = date_type.fromisoformat(body["fecha"])
        except ValueError:
            pass

    try:
        personas = _parsear_csv_personas(texto)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc

    from climasafeai.models.ensemble import predict_ensemble

    tipo_actividad = body.get("tipo_actividad") or ""
    resultados = []
    primer_pred = None
    for p in personas:
        try:
            pred = predict_ensemble(
                lat=lat,
                lon=lon,
                provincia=provincia,
                perfil=p["perfil"],
                target_date=date_obj,
            )
        except Exception as exc:
            raise HTTPException(
                400, f"No se pudo predecir el riesgo de {p['nombre']}: {exc}"
            ) from exc
        if primer_pred is None:
            primer_pred = pred
        prob_base = pred.get("perfil", {}).get("calor", {}).get("prob_personalizada", 0) or 0
        prob, factor = _aplicar_orgullo_colectivo(prob_base, tipo_actividad)
        resultados.append(
            {
                "nombre": p["nombre"],
                "edad": p["perfil"].get("edad"),
                "sexo": p["perfil"].get("sexo"),
                "clase": pred.get("clase_final_label", "SEGURO"),
                "prob_riesgo": round(prob, 4),
                "factor_orgullo": factor,
            }
        )

    seguros = sum(1 for r in resultados if r["clase"] == "SEGURO")
    precaucion = sum(1 for r in resultados if r["clase"] == "PRECAUCION")
    peligro = sum(1 for r in resultados if r["clase"] == "PELIGRO")

    return {
        "total_personas": len(resultados),
        "seguros": seguros,
        "en_precaucion": precaucion,
        "en_peligro": peligro,
        "pct_peligro": round(peligro / len(resultados) * 100, 1) if resultados else 0,
        "clase": "PELIGRO"
        if peligro and peligro / len(resultados) > 0.2
        else ("PRECAUCION" if precaucion else "SEGURO"),
        "tipo_actividad": tipo_actividad or None,
        "orgullo_colectivo": {
            "aplicado": tipo_actividad in TIPOS_ACTIVIDAD_COMPETICION,
            "factor": ORGULLO_COLECTIVO if tipo_actividad in TIPOS_ACTIVIDAD_COMPETICION else 1.0,
        },
        "detalle": resultados,
    }


@app.get("/api/perfiles")
async def api_list_perfiles():
    """Lista todos los perfiles (cabecera con alias, coordenadas)."""
    return _db.listar_perfiles()


@app.get("/api/perfil/{perfil_id}")
async def api_get_perfil(perfil_id: int):
    """Devuelve un perfil guardado (escalares + arrays)."""
    p = _db.obtener_perfil(perfil_id)
    if p is None:
        raise HTTPException(404, "Perfil no encontrado")
    # Quitar campos internos
    for k in ("id", "created_at", "updated_at", "aclimatado_actualizado_en"):
        p.pop(k, None)
    # Devolver arrays como listas para el frontend
    return p


def _sin_strings_vacios(datos: dict) -> dict:
    """Los selects del frontend mandan ``''`` para «sin elegir» (ocupacion,
    deporte, nivel_actividad...). Un string vacío viola los CHECK de las
    columnas (``ocupacion IN (...)``, ``fototipo BETWEEN 1 AND 6``, ``sexo``...)
    y tumbaba el guardado con un 500 mudo: el perfil no se guardaba y la página
    individual no podía precargarlo (WEB-009). ``''`` → ``None`` (los CHECK
    admiten NULL).
    """
    return {k: (None if v == "" else v) for k, v in datos.items()}


@app.post("/api/perfil")
async def api_save_perfil(body: dict):
    """Guarda un perfil (sin predecir). Si incluye alias, busca o crea."""
    alias = body.get("alias")
    perfil_id = body.get("perfil_id")
    datos = _sin_strings_vacios(
        {k: v for k, v in body.items() if k not in ("perfil_id", "alias")}
    )

    # Mismo fallo que /api/predict: un campo que no sea columna de `perfiles`
    # llegaba al INSERT y salia como 500 mudo. Aqui SI es un error de verdad —
    # guardar el perfil es lo unico que hace este endpoint— asi que se devuelve
    # el error en vez de un perfil_id que no existe.
    try:
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
    except CampoDesconocidoError as exc:
        logger.warning("Perfil no guardado: %s", exc)
        raise HTTPException(400, str(exc)) from exc

    return {"perfil_id": perfil_id}


@app.post("/api/perfil/{perfil_id}/tags")
async def api_update_tags(perfil_id: int, body: dict):
    """Actualiza las etiquetas de un perfil."""
    tags = body.get("tags", "")
    _db.actualizar_perfil(perfil_id, {"tags": tags})
    return {"ok": True}


@app.delete("/api/perfil/{perfil_id}")
async def api_delete_perfil(perfil_id: int):
    """Elimina un perfil y lo que dependía de él.

    WEB-006: `rutinas` y `avisos_config` no tienen FK a `perfiles` (van por
    chat_id), así que borrar el perfil las dejaba vivas. El aviso diario del
    bot seguía leyéndolas y disparando cada día para un perfil que ya no está.

    Qué se borra depende de a quién pertenece el chat_id:

    - Perfil web puro (`web_<id>`): ese espacio muere con el perfil, nadie
      puede volver a alcanzarlo. Se borran sus rutinas y su aviso.
    - Perfil vinculado a Telegram: el chat sigue existiendo y sus rutinas
      siguen siendo del usuario del bot — borrarlas desde la web sería
      destruirle datos. Se quita solo el aviso, que es lo que se personaliza
      con el perfil que estamos borrando.
    """
    perfil = _db.obtener_perfil(perfil_id)
    if perfil is None:
        raise HTTPException(404, "Perfil no encontrado")

    chat_id = perfil.get("telegram_chat_id") or f"web_{perfil_id}"
    es_web = not perfil.get("telegram_chat_id")

    rutinas_borradas = 0
    if es_web:
        for r in _db.listar_rutinas(chat_id):
            _db.eliminar_rutina(r["id"])
            rutinas_borradas += 1
    _db.guardar_hora_aviso(chat_id, None)

    _db.eliminar_perfil(perfil_id)
    return {"ok": True, "rutinas_borradas": rutinas_borradas, "aviso_borrado": True}


# ── BOT-008: rutinas semanales y avisos por perfil (web) ────────────────


@app.get("/api/perfil/{perfil_id}/rutinas")
async def api_list_rutinas(perfil_id: int):
    """Rutinas semanales de un perfil (dias 1-7, 1=lunes)."""
    chat_id = _chat_id_de_perfil(perfil_id)
    if chat_id is None:
        raise HTTPException(404, "Perfil no encontrado")
    return {"rutinas": _db.listar_rutinas(chat_id)}


@app.post("/api/perfil/{perfil_id}/rutinas")
async def api_crear_rutina(perfil_id: int, body: dict):
    """Crea una rutina semanal para el perfil.

    Body: {nombre, dias: "1,2,3,4,5", hora_inicio, hora_fin,
           ocupacion?, deporte?}. dias 1-7 (1=lunes, 7=domingo).
    """
    chat_id = _chat_id_de_perfil(perfil_id)
    if chat_id is None:
        raise HTTPException(404, "Perfil no encontrado")

    nombre = (body.get("nombre") or "").strip()
    dias = (body.get("dias") or "").strip()
    if not nombre or not dias:
        raise HTTPException(400, "Faltan campos: nombre y dias son obligatorios")
    try:
        hora_inicio = float(body.get("hora_inicio"))
        hora_fin = float(body.get("hora_fin"))
    except (TypeError, ValueError):
        raise HTTPException(400, "hora_inicio y hora_fin deben ser números")
    try:
        nums = sorted({int(d) for d in dias.split(",") if d.strip()})
    except ValueError:
        raise HTTPException(
            400, "dias deben ser números separados por comas (1=lunes ... 7=domingo)"
        )
    if not nums or not all(1 <= d <= 7 for d in nums):
        raise HTTPException(400, "dias deben estar entre 1 y 7 (1=lunes, 7=domingo)")
    if not (0 <= hora_inicio < 24 and 0 < hora_fin <= 24 and hora_fin > hora_inicio):
        raise HTTPException(400, "Horario inválido: 0 <= inicio < fin <= 24")

    rutina_id = _db.crear_rutina(
        chat_id,
        nombre,
        ",".join(str(d) for d in nums),
        hora_inicio,
        hora_fin,
        ocupacion=body.get("ocupacion") or None,
        deporte=body.get("deporte") or None,
    )
    return {"id": rutina_id}


@app.delete("/api/perfil/{perfil_id}/rutinas/{rutina_id}")
async def api_eliminar_rutina(perfil_id: int, rutina_id: int):
    """Borra una rutina del perfil."""
    chat_id = _chat_id_de_perfil(perfil_id)
    if chat_id is None:
        raise HTTPException(404, "Perfil no encontrado")
    if not any(r["id"] == rutina_id for r in _db.listar_rutinas(chat_id)):
        raise HTTPException(404, "Rutina no encontrada")
    _db.eliminar_rutina(rutina_id)
    return {"ok": True}


@app.get("/api/perfil/{perfil_id}/avisos")
async def api_get_avisos(perfil_id: int):
    """Hora de aviso diario del perfil ('HH:MM' o null si no hay)."""
    chat_id = _chat_id_de_perfil(perfil_id)
    if chat_id is None:
        raise HTTPException(404, "Perfil no encontrado")
    return {"hora": _db.obtener_hora_aviso(chat_id)}


@app.post("/api/perfil/{perfil_id}/avisos")
async def api_set_avisos(perfil_id: int, body: dict):
    """Configura la hora de aviso diario del perfil; hora=null la desactiva."""
    chat_id = _chat_id_de_perfil(perfil_id)
    if chat_id is None:
        raise HTTPException(404, "Perfil no encontrado")
    hora = body.get("hora")
    if hora is not None:
        hora = _validar_hora_aviso(str(hora))
        if hora is None:
            raise HTTPException(400, "Formato de hora inválido. Usa HH:MM (ej: 08:00).")
    _db.guardar_hora_aviso(chat_id, hora)
    return {"ok": True}


@app.post("/api/perfil/{perfil_id}/pronostico-dia")
async def api_pronostico_dia(perfil_id: int, body: dict):
    """Riesgo por cada ventana de las rutinas del día (weekday 1-7, 1=lunes).

    La ventana de cada rutina la define la propia rutina (hora_inicio +
    duración); el perfil aporta el resto de factores. Si el perfil no tiene
    ubicación, error claro.
    """
    from datetime import datetime

    from climasafeai.models.ensemble import predict_ensemble

    perfil = _db.obtener_perfil(perfil_id)
    if perfil is None:
        raise HTTPException(404, "Perfil no encontrado")
    if perfil.get("lat") is None or perfil.get("lon") is None:
        raise HTTPException(
            400,
            "El perfil no tiene ubicación (lat/lon). Guarda la ubicación antes de pedir el pronóstico del día.",
        )

    chat_id = _chat_id_de_perfil(perfil_id)
    weekday = body.get("weekday")
    if weekday is None:
        weekday = datetime.now().isoweekday()
    else:
        try:
            weekday = int(weekday)
        except (TypeError, ValueError):
            raise HTTPException(400, "weekday debe ser 1-7 (1=lunes, 7=domingo)")
        if not (1 <= weekday <= 7):
            raise HTTPException(400, "weekday debe estar entre 1 y 7 (1=lunes, 7=domingo)")

    ventanas = []
    for r in _db.rutinas_por_dia(chat_id, weekday):
        perfil_pred = _perfil_prediccion_desde_rutina(perfil, r)
        try:
            result = predict_ensemble(
                lat=perfil["lat"],
                lon=perfil["lon"],
                provincia=perfil.get("provincia") or "Madrid",
                perfil=perfil_pred,
            )
        except Exception as exc:
            ventanas.append(
                {
                    "rutina_id": r["id"],
                    "nombre": r["nombre"],
                    "hora_inicio": r["hora_inicio"],
                    "hora_fin": r["hora_fin"],
                    "error": str(exc),
                }
            )
            continue
        temps = _temps_en_ventana(
            result.get("weather", {}).get("perfil_horario") or [],
            {
                "hora_inicio": r["hora_inicio"],
                "duracion_actividad_h": r["hora_fin"] - r["hora_inicio"],
            },
        )
        ventanas.append(
            {
                "rutina_id": r["id"],
                "nombre": r["nombre"],
                "dias": r["dias"],
                "hora_inicio": r["hora_inicio"],
                "hora_fin": r["hora_fin"],
                "ocupacion": r.get("ocupacion"),
                "deporte": r.get("deporte"),
                "clase": result.get("clase_final_label", "SEGURO"),
                "prob_riesgo": round(
                    result.get("perfil", {}).get("calor", {}).get("prob_personalizada") or 0, 4
                ),
                "temp_media": round(sum(temps) / len(temps), 1) if temps else None,
            }
        )
    return {"weekday": weekday, "rutinas": ventanas}


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


def _calc_demografico(rangos: list, total: int) -> dict | None:
    if not rangos or total <= 0:
        return None
    total_peligro = sum(r.get("peligro", 0) for r in rangos)
    if total_peligro <= 0:
        return None
    contribuciones = []
    for r in rangos:
        pct_pob = r.get("n_personas", 0) / total * 100
        pct_riesgo = r.get("peligro", 0) / total_peligro * 100 if total_peligro else 0
        if pct_riesgo > pct_pob * 1.2:
            contribuciones.append(
                {
                    "rango": r["rango"],
                    "pct_poblacion": round(pct_pob, 1),
                    "pct_del_riesgo": round(pct_riesgo, 1),
                    "desproporcion": round(pct_riesgo / pct_pob, 2) if pct_pob > 0 else 0,
                }
            )
    return contribuciones[:5] if contribuciones else None


def _generar_resumen(
    pct_peligro: float,
    total_peligro: int,
    total_precaucion: int,
    total_seguros: int,
    factor_extra: float,
    factores_detalle: list,
    actividad: str,
) -> str:
    partes = []
    if pct_peligro > 15:
        partes.append(f"Riesgo alto: {pct_peligro}% del grupo en peligro")
    elif pct_peligro > 5:
        partes.append(
            f"Riesgo moderado: {pct_peligro}% en peligro, {total_precaucion} personas en precaución"
        )
    else:
        partes.append(f"Riesgo bajo: mayoría del grupo ({total_seguros} personas) en nivel seguro")

    if factor_extra > 1.1 and factores_detalle:
        top = max(factores_detalle, key=lambda f: f["multiplicador"])
        partes.append(
            f"Factor más influyente: {top['nombre']} (afecta al {top['pct']:.0f}% del grupo, ×{top['multiplicador']})"
        )

    if actividad:
        etiqueta_act = {
            "reposo": "reposo",
            "ligera": "ligera",
            "moderada": "moderada",
            "intensa": "intensa",
            "muy_intensa": "muy intensa",
        }.get(actividad, actividad)
        partes.append(f"Actividad: {etiqueta_act}")

    return " · ".join(partes) if partes else ""


@app.post("/api/contrafactuales-grupo")
async def api_contrafactuales_grupo(body: dict):
    """Simula cambios en parámetros del grupo y compara con la predicción
    original. Acepta el mismo body que POST /api/riesgo-colectivo (tipo=numero)."""
    cambios = body.get("cambios", {})
    escenario = body.get("escenario", "")
    body_base = {k: v for k, v in body.items() if k not in ("cambios", "escenario")}

    c_original = _calcular_riesgo_colectivo(body_base)
    body_mod = dict(body_base)
    body_mod.update(cambios)
    c_mod = _calcular_riesgo_colectivo(body_mod)

    orig_pct = c_original["pct_peligro"]
    mod_pct = c_mod["pct_peligro"]
    diff_pct = round(mod_pct - orig_pct, 1)
    diff_abs = c_mod["en_peligro"] - c_original["en_peligro"]

    return {
        "escenario": escenario,
        "original": {
            "total_peligro": c_original["en_peligro"],
            "total_precaucion": c_original["en_precaucion"],
            "total_seguros": c_original["seguros"],
            "pct_peligro": orig_pct,
            "clase": "PELIGRO" if orig_pct > 20 else ("PRECAUCION" if orig_pct > 5 else "SEGURO"),
            "factor_extra": c_original["factor_extra"],
        },
        "modificado": {
            "total_peligro": c_mod["en_peligro"],
            "total_precaucion": c_mod["en_precaucion"],
            "total_seguros": c_mod["seguros"],
            "pct_peligro": mod_pct,
            "clase": "PELIGRO" if mod_pct > 20 else ("PRECAUCION" if mod_pct > 5 else "SEGURO"),
            "factor_extra": c_mod["factor_extra"],
        },
        "diferencia": {
            "pct_peligro": diff_pct,
            "absoluta": diff_abs,
            "mejora": diff_abs < 0,
        },
    }


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
        "nota": "La 'clase_final_sistema' puede incluir un override por HI/UV que prevalece sobre la probabilidad personalizada."
        if result.get("override_fisico")
        else None,
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
        result = riesgo_zona_grid(
            lat=lat, lon=lon, radio_km=radio_km, perfil_id=perfil, target_date=date_obj
        )
        if "error" in result:
            return result
        return result
    except Exception as exc:
        return {"error": str(exc)}


def _riesgo_zona_resultado(body: dict) -> dict:
    """Núcleo compartido de /api/riesgo-zona (POST) y sus exportaciones."""
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
            lat=lat,
            lon=lon,
            radio_km=radio_km,
            perfil_id=perfil_id,
            target_date=date_obj,
            perfil=perfil,
        )
        if "error" in result:
            return result
        return result
    except Exception as e:
        return {"error": str(e)}


@app.post("/api/riesgo-zona")
async def api_riesgo_zona_post(body: dict):
    return _riesgo_zona_resultado(body)


@app.post("/api/riesgo-zona/export/geojson")
async def api_riesgo_zona_export_geojson(body: dict):
    """Descarga las celdas del grid como FeatureCollection GeoJSON (MAPA-001).

    Cada celda es un Polygon con la clase de riesgo (riesgo/riesgo_label) y su
    HI pico (hi_pico) en properties. Misma geometría que el overlay Leaflet.
    """
    from climasafeai.data.grid_risk import celdas_a_featurecollection

    result = _riesgo_zona_resultado(body)
    if "error" in result:
        raise HTTPException(400, result["error"])
    return JSONResponse(
        celdas_a_featurecollection(result["celdas"]),
        headers={"Content-Disposition": 'attachment; filename="riesgo_zona.geojson"'},
    )


@app.post("/api/riesgo-zona/export/png")
async def api_riesgo_zona_export_png(body: dict):
    """Descarga el overlay de riesgo del grid como PNG (MAPA-001).

    Render server-side con matplotlib (dependencia ya existente) usando el
    mismo código de color por clase que el overlay Leaflet.
    """
    from climasafeai.data.grid_risk import render_riesgo_png

    result = _riesgo_zona_resultado(body)
    if "error" in result:
        raise HTTPException(400, result["error"])
    png = render_riesgo_png(
        result["celdas"],
        stats=result.get("stats"),
        center=result.get("center"),
        resumen=result.get("resumen_horario"),
        perfil_label=result.get("perfil_label", ""),
    )
    return Response(
        content=png,
        media_type="image/png",
        headers={"Content-Disposition": 'attachment; filename="riesgo_zona.png"'},
    )


@app.post("/api/riesgo-volumen")
async def api_riesgo_volumen(body: dict):
    """Estima cuántas personas de un volumen dado podrían requerir
    atención médica por calor, combinando prevalencia poblacional de
    ECV con las condiciones climáticas previstas."""
    provincia = body.get("provincia", "Madrid")
    lat = body.get("lat")
    lon = body.get("lon")
    total_personas = int(body.get("total_personas", 1000))
    pct_mayores_50 = float(body.get("pct_mayores_50", 30))
    tipo_evento = body.get("tipo_evento", "general")
    hora_inicio = body.get("hora_inicio")
    duracion_h = body.get("duracion_h")
    target_date = body.get("fecha")
    date_obj = None
    if target_date:
        try:
            from datetime import date as date_type

            date_obj = date_type.fromisoformat(target_date)
        except ValueError:
            pass

    from climasafeai.data.weather_fetcher import fetch_weather_data
    from climasafeai.features.weather_indices import heat_index
    import numpy as np

    try:
        weather = fetch_weather_data(lat=lat, lon=lon, provincia=provincia, target_date=date_obj)
    except Exception as e:
        return {"error": f"Error fetching weather: {e}"}

    df_hora = weather.get("df_hora")
    hourly_data = None
    if df_hora is not None and not df_hora.empty:
        df = df_hora.copy()
        if "rh" in df.columns and "t2m_c" in df.columns:
            df["heat_index_c"] = heat_index(df["t2m_c"].values, df["rh"].values)
        hourly_data = df.to_dict("records")

    horas_actividad = []
    if hourly_data:
        import pandas as pd

        for row in hourly_data:
            dt = pd.to_datetime(row.get("datetime"))
            hi = row.get("heat_index_c")
            if hi is not None and not (isinstance(hi, float) and np.isnan(hi)):
                horas_actividad.append({"hora": dt.hour, "hi": float(hi)})

    if hora_inicio is not None and duracion_h is not None:
        h_ini = int(hora_inicio)
        h_fin = min(23, h_ini + max(1, int(duracion_h)))
        horas_filtradas = [h for h in horas_actividad if h_ini <= h["hora"] < h_fin]
    else:
        horas_filtradas = horas_actividad

    hi_peak = None
    if horas_filtradas:
        hi_peak = max(h["hi"] for h in horas_filtradas)
    elif horas_actividad:
        hi_peak = max(h["hi"] for h in horas_actividad)
    else:
        current = weather.get("current", {})
        t = current.get("t2m_c")
        rh = current.get("rh")
        if t is not None and rh is not None:
            hi_peak = float(heat_index(np.array([t]), np.array([rh]))[0])

    from climasafeai.models.volumen import estimar_afectados

    resultado = estimar_afectados(
        total_personas=total_personas,
        hi_peak=hi_peak,
        pct_mayores_50=pct_mayores_50,
        tipo_evento=tipo_evento,
    )
    resultado["weather"] = _get_weather_summary({"weather": weather})
    if target_date:
        resultado["target_date"] = target_date

    return resultado


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
                raise  # propagar para que el outer except lo capture
            except Exception:
                # Mensaje malformado (no JSON) — ignorar y seguir
                try:
                    await ws.send_json(
                        {"type": "bot", "text": " ✕ Mensaje no válido. Usa texto plano."}
                    )
                except Exception:
                    pass
                continue
            msg = data.get("text", "").strip()
            reply = await process_message(msg, session)
            await ws.send_json({"type": "bot", "text": reply})
    except WebSocketDisconnect:
        pass
