"""
agents.tools.prediction_mcp_tool — MCP server para predicción de riesgo.

Expone tools MCP para que un LLM (claude, etc.) pueda:
  - Predecir riesgo individual (con contrafactuales y recomendaciones)
  - Estimar volumen de afectados en un evento
  - Evaluar riesgo de una zona (grid)
  - Predecir riesgo colectivo (grupo demográfico)
  - Comparar curvas de riesgo por edad
  - Consultar estado del sistema

Uso standalone:
    uv run python -m agents.tools.prediction_mcp_tool

Uso programático:
    from agents.tools.prediction_mcp_tool import run_mcp_server
    run_mcp_server(port=8101)
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import socket
import sys
import subprocess
from pathlib import Path
from datetime import date as date_type, datetime, timedelta, timezone
from typing import Any, Optional

logger = logging.getLogger(__name__)


# ── Utilidades de red ────────────────────────────────────────────────


def _local_ip() -> str:
    """Devuelve la IP local de la máquina (la que usaría para salir a internet)."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("8.8.8.8", 80))
            return s.getsockname()[0]
    except Exception:
        return "127.0.0.1"


# ── Certificado SSL autofirmado (sin openssl) ────────────────────────

_SELF_SIGNED_DIR = Path(__file__).resolve().parent / ".mcp-certs"


def _ensure_self_signed_cert(host: str = "localhost") -> tuple[str, str]:
    """Genera un certificado SSL autofirmado con cryptography si no existe.
    Devuelve (cert_path, key_path)."""
    _SELF_SIGNED_DIR.mkdir(parents=True, exist_ok=True)
    safe_host = "localhost" if host in ("0.0.0.0", "127.0.0.1") else host.replace("*", "_").replace(".", "_")
    cert_file = _SELF_SIGNED_DIR / f"{safe_host}.cert.pem"
    key_file = _SELF_SIGNED_DIR / f"{safe_host}.key.pem"

    if cert_file.exists() and key_file.exists():
        return str(cert_file), str(key_file)

    from cryptography import x509
    from cryptography.x509.oid import NameOID
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import rsa

    key = rsa.generate_private_key(
        public_exponent=65537,
        key_size=2048,
    )

    subject = issuer = x509.Name([
        x509.NameAttribute(NameOID.COUNTRY_NAME, "ES"),
        x509.NameAttribute(NameOID.ORGANIZATION_NAME, "ClimaSafeAI"),
        x509.NameAttribute(NameOID.COMMON_NAME, host if host != "0.0.0.0" else "localhost"),
    ])

    san_names = ["localhost", "127.0.0.1"]
    if host not in ("0.0.0.0", "localhost", "127.0.0.1"):
        san_names.append(host)

    try:
        host_ip = socket.gethostbyname(socket.gethostname())
        if host_ip not in san_names:
            san_names.append(host_ip)
    except Exception:
        pass

    cert = (
        x509.CertificateBuilder()
        .subject_name(subject)
        .issuer_name(issuer)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(datetime.now(timezone.utc))
        .not_valid_after(datetime.now(timezone.utc) + timedelta(days=365 * 5))
        .add_extension(
            x509.SubjectAlternativeName([x509.DNSName(n) for n in san_names]),
            critical=False,
        )
        .sign(key, hashes.SHA256())
    )

    with open(key_file, "wb") as f:
        f.write(key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.TraditionalOpenSSL,
            encryption_algorithm=serialization.NoEncryption(),
        ))
    with open(cert_file, "wb") as f:
        f.write(cert.public_bytes(serialization.Encoding.PEM))

    os.chmod(key_file, 0o600)
    print(f"   Certificado SSL autofirmado generado: {cert_file}", file=sys.stderr, flush=True)
    return str(cert_file), str(key_file)


# ── Constantes reutilizadas (mismas que en chat/app.py) ──────────────

_FACTORES_COEF = {
    "grasa_alta": {"label": "Obesidad/grasa alta", "coef": 1.08},
    "cardiovascular": {"label": "Cardiovascular", "coef": 1.4},
    "diabetes": {"label": "Diabetes", "coef": 1.2},
    "respiratoria": {"label": "Respiratoria", "coef": 1.3},
    "mental": {"label": "Salud mental", "coef": 1.8},
    "no_aclimatados": {"label": "No aclimatados", "coef": 1.6},
}


# ── Helpers in-line (evitan importar chat.app, que arrastra FastAPI) ─


def _prevalencia(edad: float) -> dict[str, float]:
    e = max(18, min(90, edad))
    return {
        "grasa_alta": min(45, 20 + (e - 20) * 0.35),
        "cardiovascular": min(30, 2 + (e - 20) * 0.35),
        "diabetes": min(25, 1 + (e - 20) * 0.30),
        "respiratoria": min(12, 3 + (e - 20) * 0.12),
        "mental": min(15, 8 - abs(e - 45) * 0.15),
        "no_aclimatados": 40.0,
    }


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
            contribuciones.append({
                "rango": r["rango"],
                "pct_poblacion": round(pct_pob, 1),
                "pct_del_riesgo": round(pct_riesgo, 1),
                "desproporcion": round(pct_riesgo / pct_pob, 2) if pct_pob > 0 else 0,
            })
    return contribuciones[:5] if contribuciones else None


def _generar_resumen(
    pct_peligro: float, total_peligro: int, total_precaucion: int,
    total_seguros: int, factor_extra: float,
    factores_detalle: list, actividad: str,
) -> str:
    partes = []
    if pct_peligro > 15:
        partes.append(f"Riesgo alto: {pct_peligro}% del grupo en peligro")
    elif pct_peligro > 5:
        partes.append(f"Riesgo moderado: {pct_peligro}% en peligro, {total_precaucion} personas en precaución")
    else:
        partes.append(f"Riesgo bajo: mayoría del grupo ({total_seguros} personas) en nivel seguro")

    if factor_extra > 1.1 and factores_detalle:
        top = max(factores_detalle, key=lambda f: f["multiplicador"])
        partes.append(f"Factor más influyente: {top['nombre']} (afecta al {top['pct']:.0f}% del grupo, ×{top['multiplicador']})")

    if actividad:
        etiqueta_act = {"reposo": "reposo", "ligera": "ligera", "moderada": "moderada", "intensa": "intensa", "muy_intensa": "muy intensa"}.get(actividad, actividad)
        partes.append(f"Actividad: {etiqueta_act}")

    return " · ".join(partes) if partes else ""


# ── Funciones standalone ─────────────────────────────────────────────


def _try_prediction(
    lat: float, lon: float, provincia: str,
    perfil: dict, target_date: date_type | None = None,
) -> dict:
    from climasafeai.models.ensemble import predict_ensemble
    return predict_ensemble(lat=lat, lon=lon, provincia=provincia, perfil=perfil, target_date=target_date)


def _weather_for_date(lat: float, lon: float, provincia: str, target_date: date_type | None = None) -> dict:
    from climasafeai.data.weather_fetcher import fetch_weather_data
    return fetch_weather_data(lat=lat, lon=lon, provincia=provincia, target_date=target_date)


def _hi_peak_from_weather(
    weather: dict, hora_inicio: int | None = None, duracion_h: int | None = None,
) -> float | None:
    from climasafeai.features.weather_indices import heat_index
    import numpy as np
    import pandas as pd

    df_hora = weather.get("df_hora")
    if df_hora is not None and not df_hora.empty:
        df = df_hora.copy()
        if "rh" in df.columns and "t2m_c" in df.columns:
            df["heat_index_c"] = heat_index(df["t2m_c"].values, df["rh"].values)
        hourly = df.to_dict("records")
    else:
        return None

    horas = []
    for row in hourly:
        dt = pd.to_datetime(row.get("datetime"))
        hi = row.get("heat_index_c")
        if hi is not None and not (isinstance(hi, float) and np.isnan(hi)):
            horas.append({"hora": dt.hour, "hi": float(hi)})

    if hora_inicio is not None and duracion_h is not None:
        h_fin = min(23, int(hora_inicio) + max(1, int(duracion_h)))
        horas = [h for h in horas if int(hora_inicio) <= h["hora"] < h_fin]

    return max((h["hi"] for h in horas), default=None)


def _parse_date(fecha: str | None) -> date_type | None:
    if not fecha:
        return None
    try:
        return date_type.fromisoformat(fecha)
    except ValueError:
        return None


def predict_risk(
    lat: float, lon: float, provincia: str = "Madrid",
    edad: int = 40, sexo: str = "hombre",
    nivel_actividad: str = "ligera",
    hora_inicio: int = 10, duracion_h: float = 2.0,
    aclimatado: bool | None = None,
    peso: float | None = None, altura: float | None = None,
    grasa: float | None = None, entrenado: bool | None = None,
    ocupacion: str | None = None, deporte: str | None = None,
    comorbilidades: list[str] | None = None,
    medicacion: list[str] | None = None,
    fecha: str | None = None,
    incluir_contrafactuales: bool = True,
    incluir_recomendaciones: bool = True,
) -> dict:
    target_date = _parse_date(fecha)
    from climasafeai.features.personalizacion import riesgo_horario_acumulado, recomendar_horario, pico_riesgo_actividad

    perfil: dict[str, Any] = {
        "edad": edad,
        "sexo": sexo,
        "nivel_actividad": nivel_actividad,
        "hora_inicio": hora_inicio,
        "duracion_actividad_h": duracion_h,
    }
    if aclimatado is not None:
        perfil["aclimatado"] = aclimatado
    if peso is not None:
        perfil["peso"] = peso
    if altura is not None:
        perfil["altura"] = altura
    if grasa is not None:
        perfil["grasa_corporal"] = grasa
    if entrenado is not None:
        perfil["entrenado"] = entrenado
    if ocupacion is not None:
        perfil["ocupacion"] = ocupacion
    if deporte is not None:
        perfil["deporte"] = deporte
    if comorbilidades:
        perfil["comorbilidades"] = set(comorbilidades)
    if medicacion:
        perfil["medicacion"] = set(medicacion)

    result = _try_prediction(lat, lon, provincia, perfil, target_date)

    _ph = result.get("weather", {}).get("perfil_horario", [])
    if _ph:
        result["riesgo_horario"] = riesgo_horario_acumulado(_ph, perfil)
        result["riesgo_pico"] = pico_riesgo_actividad(result["riesgo_horario"], perfil)
        result["recomendacion_horario"] = recomendar_horario(_ph, perfil)

    if incluir_contrafactuales:
        try:
            from climasafeai.models.explicabilidad import generar_contrafactuales
            cfs = generar_contrafactuales(result)
            result["contrafactuales"] = cfs
        except Exception as e:
            result["contrafactuales"] = {"error": str(e)}

    if incluir_recomendaciones:
        try:
            from climasafeai.models.recomendaciones import generar_recomendaciones
            recs = generar_recomendaciones(perfil, result)
            result["recomendaciones"] = recs
        except Exception as e:
            result["recomendaciones"] = {"error": str(e)}

    _sanitize(result)
    return result


def predict_volume_risk(
    lat: float, lon: float,
    total_personas: int = 5000,
    pct_mayores_50: float = 30.0,
    tipo_evento: str = "deporte",
    hora_inicio: int | None = None,
    duracion_h: int | None = None,
    provincia: str = "Madrid",
    fecha: str | None = None,
) -> dict:
    from climasafeai.models.volumen import estimar_afectados

    target_date = _parse_date(fecha)
    weather = _weather_for_date(lat, lon, provincia, target_date)
    hi_peak = _hi_peak_from_weather(weather, hora_inicio, duracion_h)

    result = estimar_afectados(
        total_personas=total_personas,
        hi_peak=hi_peak,
        pct_mayores_50=pct_mayores_50,
        tipo_evento=tipo_evento,
    )
    result["hi_peak"] = hi_peak
    return result


def predict_zone_risk(
    lat: float, lon: float,
    radio_km: float = 5.0,
    perfil_id: str = "adulto",
    fecha: str | None = None,
    perfil: dict | None = None,
) -> dict:
    from climasafeai.data.grid_risk import riesgo_zona_grid

    target_date = _parse_date(fecha)
    return riesgo_zona_grid(
        lat=lat, lon=lon, radio_km=radio_km,
        perfil_id=perfil_id, target_date=target_date,
        perfil=perfil,
    )


def predict_group_risk(
    lat: float, lon: float,
    tipo: str = "numero",
    provincia: str = "Madrid",
    cantidad: int = 100,
    edad_min: int = 18,
    edad_max: int = 80,
    pct_hombres: float = 50.0,
    tipo_actividad: str = "deporte",
    actividad: str = "ligera",
    hora_inicio: int = 10,
    duracion: float = 2.0,
    aclimatado: str = "",
    ocupacion: str | None = None,
    deporte: str | None = None,
    fecha: str | None = None,
    tag: str | None = None,
) -> dict:
    from climasafeai.db.manager import DBManager

    target_date = _parse_date(fecha)
    db = DBManager()

    if tipo == "etiqueta":
        if not tag:
            return {"error": "tag requerido"}
        perfiles = db.buscar_por_tag(tag)
        if not perfiles:
            return {"error": f"No se encontraron perfiles con la etiqueta '{tag}'"}

        resultados = []
        for p in perfiles:
            try:
                perfil = {k: v for k, v in p.items() if k not in ("id", "alias", "tags", "created_at", "updated_at")}
                if hora_inicio is not None:
                    perfil["hora_inicio"] = float(hora_inicio)
                if duracion is not None:
                    perfil["duracion_actividad_h"] = float(duracion)
                if actividad:
                    perfil["nivel_actividad"] = actividad
                if ocupacion:
                    perfil["ocupacion"] = ocupacion
                if deporte:
                    perfil["deporte"] = deporte
                if aclimatado in ("si", "no"):
                    perfil["aclimatado"] = aclimatado == "si"

                pred = _try_prediction(lat, lon, provincia, perfil, target_date)
                pred["_alias"] = p.get("alias", f"ID {p['id']}")
                pred["_perfil_id"] = p["id"]
                resultados.append(pred)
            except Exception as e:
                resultados.append({"_alias": p.get("alias", f"ID {p['id']}"), "error": str(e)})

        en_peligro = sum(1 for r in resultados if r.get("clase_final_label") == "PELIGRO" or r.get("clase_final") == 2)
        en_precaucion = sum(1 for r in resultados if r.get("clase_final_label") == "PRECAUCION" or r.get("clase_final") == 1)

        return {
            "tipo": "etiqueta",
            "tag": tag,
            "total_perfiles": len(resultados),
            "en_peligro": en_peligro,
            "en_precaucion": en_precaucion,
            "resultados": resultados,
        }

    from climasafeai.features.personalizacion import riesgo_horario_acumulado, recomendar_horario
    from climasafeai.models.ensemble import predict_ensemble

    rangos_edad = [
        (18, 30), (30, 45), (45, 60), (60, 75), (75, 90)
    ]
    rangos_edad = [(a, b) for a, b in rangos_edad if a < edad_max and b > edad_min]
    if not rangos_edad:
        rangos_edad = [(edad_min, edad_max)]

    total_rango_pct = sum(min(b, edad_max) - max(a, edad_min) for a, b in rangos_edad)
    pcts: dict[str, float] = {k: 0.0 for k in _FACTORES_COEF}
    for a, b in rangos_edad:
        solapamiento = max(0, min(b, edad_max) - max(a, edad_min))
        if solapamiento <= 0:
            continue
        peso = solapamiento / total_rango_pct if total_rango_pct else 0
        edad_med = (max(a, edad_min) + min(b, edad_max)) / 2
        prev = _prevalencia(edad_med)
        for k in pcts:
            pcts[k] += prev[k] * peso

    factor_extra = 1.0
    factores_detalle: list[dict] = []
    for k, cfg in _FACTORES_COEF.items():
        pct = pcts[k]
        mult = 1.0 + (pct / 100.0) * (cfg["coef"] - 1.0) if pct > 0 else 1.0
        factor_extra *= mult
        if mult > 1.001:
            factores_detalle.append({
                "clave": k, "nombre": cfg["label"],
                "pct": round(pct, 1), "coef": cfg["coef"],
                "multiplicador": round(mult, 3),
            })
    factor_extra = min(factor_extra, 2.5)

    resultados_rangos: list[dict] = []
    total_seguros = 0
    total_precaucion = 0
    total_peligro = 0
    primer_pred = None

    for a, b in rangos_edad:
        n_rango = int(cantidad * (min(b, edad_max) - max(a, edad_min)) / (edad_max - edad_min))
        if n_rango <= 0:
            continue
        edad_med = (max(a, edad_min) + min(b, edad_max)) / 2
        pct_h = pct_hombres
        perfil_rango = {
            "edad": int(edad_med), "sexo": "hombre" if pct_h >= 50 else "mujer",
            "nivel_actividad": actividad, "hora_inicio": hora_inicio,
            "duracion_actividad_h": duracion, "aclimatado": aclimatado in ("si", None, ""),
        }
        if ocupacion:
            perfil_rango["ocupacion"] = ocupacion
        if deporte:
            perfil_rango["deporte"] = deporte
        try:
            pred = predict_ensemble(lat=lat, lon=lon, provincia=provincia, perfil=perfil_rango, target_date=target_date)
        except Exception:
            continue
        if primer_pred is None:
            primer_pred = pred
        clase = pred.get("clase_final")
        if clase == 2:
            n_peligro = n_rango
            n_precaucion = 0
            n_seguro = 0
        elif clase == 1:
            n_peligro = 0
            n_precaucion = n_rango
            n_seguro = 0
        else:
            n_peligro = 0
            n_precaucion = 0
            n_seguro = n_rango
        total_peligro += n_peligro
        total_precaucion += n_precaucion
        total_seguros += n_seguro
        resultados_rangos.append({
            "rango": f"{max(a, edad_min)}-{min(b, edad_max)}",
            "edad_media": int(edad_med), "n_personas": n_rango,
            "peligro": n_peligro, "precaucion": n_precaucion, "seguro": n_seguro,
        })

    total = total_seguros + total_precaucion + total_peligro
    pct_peligro = total_peligro / total * 100 if total else 0

    factores_activos = [f"{k}={pcts[k]:.0f}%" for k in sorted(pcts) if pcts[k] > 0]
    demografico = _calc_demografico(resultados_rangos, total)

    return {
        "total_personas": total,
        "seguros": total_seguros,
        "en_precaucion": total_precaucion,
        "en_peligro": total_peligro,
        "pct_peligro": pct_peligro,
        "clase": "PELIGRO" if pct_peligro > 20 else ("PRECAUCION" if pct_peligro > 5 else "SEGURO"),
        "factor_extra": round(factor_extra, 3),
        "factores_grupo": factores_activos,
        "factores_detalle": c["factores_detalle"],
        "demografico": demografico,
        "resumen": _generar_resumen(pct_peligro, total_peligro, total_precaucion, total_seguros, factor_extra, c["factores_detalle"], c["actividad"]),
    }


def predict_age_curves(
    lat: float, lon: float,
    provincia: str = "Madrid",
    edad: int = 40,
    sexo: str = "hombre",
    nivel_actividad: str = "ligera",
    hora_inicio: int = 10,
    duracion_h: float = 2.0,
    aclimatado: bool | None = None,
    fecha: str | None = None,
    edades: list[int] | None = None,
) -> dict:
    target_date = _parse_date(fecha)

    perfil: dict[str, Any] = {
        "edad": edad,
        "sexo": sexo,
        "nivel_actividad": nivel_actividad,
        "hora_inicio": hora_inicio,
        "duracion_actividad_h": duracion_h,
    }
    if aclimatado is not None:
        perfil["aclimatado"] = aclimatado

    from climasafeai.models.ensemble import predict_ensemble, perfil_horario_desde_df
    from climasafeai.features.personalizacion import riesgo_horario_acumulado, recomendar_horario

    weather = _weather_for_date(lat, lon, provincia, target_date)
    df_hora = weather.get("df_hora")
    if df_hora is None or df_hora.empty:
        return {"error": "No se pudieron obtener datos meteorológicos"}

    perfil_horario = perfil_horario_desde_df(df_hora)
    perfil["_perfil_horario"] = perfil_horario

    result = _try_prediction(lat, lon, provincia, perfil, target_date)

    ref_edades = edades or [25, 40, 55, 70, 85]
    curvas = []
    for e in ref_edades:
        p = dict(perfil)
        p["edad"] = e
        try:
            curva = riesgo_horario_acumulado(perfil_horario, p)
            curvas.append({"edad": e, "curva": curva})
        except Exception:
            curvas.append({"edad": e, "error": "fallo al computar curva"})

    return {
        "edad_referencia": edad,
        "clase_final": result.get("clase_final_label"),
        "curvas_edad": curvas,
        "recomendacion_horario": recomendar_horario(perfil_horario, perfil),
    }


def system_health() -> dict:
    info = {
        "proyecto": "ClimaSafeAI",
        "modulos_cargados": True,
    }
    try:
        from climasafeai.models.ensemble import PERS_THRESHOLD_PELIGRO
        info["umbral_peligro"] = PERS_THRESHOLD_PELIGRO
    except Exception:
        info["ensemble"] = "no disponible"

    try:
        from climasafeai.models.volumen import estimar_afectados
        info["volumen"] = "disponible"
    except Exception:
        info["volumen"] = "no disponible"

    try:
        from climasafeai.data.grid_risk import riesgo_zona_grid, PERFILES_DISPONIBLES
        info["perfiles_zona"] = list(PERFILES_DISPONIBLES.keys())
    except Exception:
        info["grid_risk"] = "no disponible"

    try:
        from climasafeai.models.ensemble import PERS_THRESHOLD_PELIGRO, _edad_a_estrato, _aplicar_factor_edad
        info["personalizacion"] = "disponible"
    except Exception:
        info["personalizacion"] = "no disponible"

    return info


def _sanitize(result: dict) -> None:
    result.pop("perfil_usuario", None)
    result.get("weather", {}).pop("df_hora", None)
    result.get("weather", {}).pop("df_features", None)
    for mod_name, mod_res in result.get("modelos", {}).items():
        if isinstance(mod_res, dict):
            mod_res.pop("_X", None)
    if "error" in result.get("modelos", {}).get("LSTM", {}):
        del result["modelos"]["LSTM"]["error"]


# ── MCP Server ───────────────────────────────────────────────────────

try:
    from mcp.server.fastmcp import FastMCP
    _mcp = FastMCP("ClimaSafeAI Predicción de Riesgo")

    @_mcp.tool()
    def predict_risk_mcp(
        lat: float,
        lon: float,
        provincia: str = "",
        edad: int = 40,
        sexo: str = "hombre",
        nivel_actividad: str = "ligera",
        hora_inicio: int = 10,
        duracion_h: float = 2.0,
        aclimatado: bool = False,
        peso: Optional[float] = None,
        altura: Optional[float] = None,
        comorbilidades: Optional[str] = None,
        medicacion: Optional[str] = None,
        fecha: Optional[str] = None,
    ) -> str:
        """Predice riesgo cardiovascular para 1 persona.

REGLAS ESTRICTAS (léelas todas antes de usar la herramienta):
1. Los datos fijos del perfil (edad, sexo, peso, altura, aclimatado, comorbilidades, medicación) los proporciona cargar_perfil_mcp. No los preguntes al usuario ni los inventes ni los confundas.
2. Los datos variables (hora_inicio, duracion_h, nivel_actividad, ubicación) DEBES preguntarlos al usuario explícitamente. NO los deduzcas de la conversación. NO asumas hora. NO asumas duración. NO asumas el nivel de actividad aunque el usuario diga "correr" o "andar". Pregunta textualmente hasta que el usuario responda cada uno.
3. ACLIMATADO se refiere exclusivamente a aclimatación al CALOR (vivir en clima cálido). No tiene nada que ver con estar acostumbrado a un deporte. No confundir.
4. Pregunta uno a uno. No asumas nada."""
        comorb_list = [c.strip() for c in comorbilidades.split(",")] if comorbilidades else None
        med_list = [m.strip() for m in medicacion.split(",")] if medicacion else None
        result = predict_risk(
            lat=lat, lon=lon, provincia=provincia,
            edad=edad, sexo=sexo, nivel_actividad=nivel_actividad,
            hora_inicio=hora_inicio, duracion_h=duracion_h,
            aclimatado=aclimatado, peso=peso, altura=altura,
            comorbilidades=comorb_list, medicacion=med_list,
            fecha=fecha,
        )
        _sanitize(result)
        return json.dumps(result, indent=2, default=str, ensure_ascii=False)

    @_mcp.tool()
    def listar_usuarios_mcp() -> str:
        """Lista todos los usuarios/perfiles guardados. Muestra alias, edad y sexo de cada uno. No tiene parámetros.

FLUJO CORRECTO:
1. Pregunta al usuario si tiene perfil guardado o si es nuevo.
2a. Si tiene perfil: pregúntale su alias, usa listar_usuarios_mcp para comprobar que existe, luego cargar_perfil_mcp.
2b. Si es nuevo: pregúntale TODOS los datos (edad, sexo, peso, altura, si está aclimatado al calor, comorbilidades, medicación) para llamar a predict_risk_mcp directamente. Ofrece guardar el perfil para futuras veces."""
        from climasafeai.db.manager import DBManager
        db = DBManager()
        perfiles = db.listar_perfiles()
        datos = []
        for p in perfiles:
            datos.append({
                "alias": p.get("alias") or f"ID {p['id']}",
                "id": p["id"],
                "edad": p.get("edad"),
                "sexo": p.get("sexo"),
                "provincia": p.get("provincia"),
                "tags": p.get("tags"),
            })
        return json.dumps(datos, indent=2, ensure_ascii=False, default=str)

    @_mcp.tool()
    def cargar_perfil_mcp(alias: str) -> str:
        """Carga un perfil de usuario por su alias exacto y devuelve sus datos estáticos: edad, sexo, peso, altura, aclimatado (al CALOR, no al deporte), comorbilidades, medicación, nivel_actividad_base.

REGLAS:
- Aclimatado se refiere exclusivamente a aclimatación al CALOR AMBIENTAL, no a estar acostumbrado a hacer ejercicio.
- nivel_actividad, hora_inicio y duracion del perfil son valores genéricos. NO los uses para la predicción. Pregunta al usuario hora_inicio, duracion_h y nivel_actividad concretos para esta salida.
- Después de cargar el perfil, pregunta los datos variables (hora, duración, actividad, ubicación) UNO A UNO."""
        from climasafeai.db.manager import DBManager
        db = DBManager()
        match = db.buscar_por_alias(alias)
        if not match:
            return json.dumps({"error": f"No se encontró perfil con alias '{alias}'"}, ensure_ascii=False)
        perfil = db.obtener_perfil(match["id"])
        if not perfil:
            return json.dumps({"error": f"Perfil ID {match['id']} no encontrado"}, ensure_ascii=False)
        return json.dumps(perfil, indent=2, ensure_ascii=False, default=str)

    _HAS_MCP = True
except ImportError:
    _HAS_MCP = False

def run_mcp_server(
    host: str = "0.0.0.0",
    port: int = 8101,
    ssl_keyfile: str | None = None,
    ssl_certfile: str | None = None,
    insecure: bool = False,
    stdio: bool = False,
) -> None:
    """Arranca el servidor MCP.

    Por defecto genera certificado autofirmado y sirve HTTPS (SSE).
    Con --insecure usa HTTP plano (SSE).
    Con --stdio usa transporte stdio (para Claude Desktop / Cursor).
    """
    if not _HAS_MCP:
        print("Error: mcp no está instalado.", file=sys.stderr)
        return

    if stdio:
        _mcp.run()
        return

    import uvicorn

    starlette_app = _mcp.streamable_http_app()

    if not insecure and not (ssl_keyfile and ssl_certfile):
        try:
            ssl_certfile, ssl_keyfile = _ensure_self_signed_cert(host)
        except Exception as e:
            print(f"   No se pudo generar certificado SSL ({e}), usando HTTP", file=sys.stderr, flush=True)

    proto = "https" if ssl_certfile else "http"
    display_host = _local_ip() if host == "0.0.0.0" else host
    banner = (
        "MCP Server — ClimaSafeAI Predicción de Riesgo\n"
        f"   Escuchando en {proto}://{display_host}:{port}/mcp\n"
        "   Usa Streamable HTTP\n"
    )
    print(banner, file=sys.stderr, flush=True)

    uvicorn_config: dict[str, Any] = {
        "app": starlette_app,
        "host": host,
        "port": port,
        "log_level": "info",
    }
    if ssl_keyfile and ssl_certfile:
        uvicorn_config["ssl_keyfile"] = ssl_keyfile
        uvicorn_config["ssl_certfile"] = ssl_certfile

    try:
        uvicorn.run(**uvicorn_config)
    except (KeyboardInterrupt, asyncio.CancelledError):
        print("\n   Servidor detenido.", file=sys.stderr, flush=True)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="ClimaSafeAI MCP Prediction Server")
    parser.add_argument("--host", default="0.0.0.0", help="Host (SSE mode, default 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8101, help="Puerto (SSE mode, default 8101)")
    parser.add_argument("--ssl-keyfile", help="Ruta a clave privada SSL")
    parser.add_argument("--ssl-certfile", help="Ruta a certificado SSL")
    parser.add_argument("--insecure", action="store_true", help="HTTP plano en vez de HTTPS (SSE mode)")
    parser.add_argument("--stdio", action="store_true", help="Usar transporte stdio (para Claude Desktop)")
    args = parser.parse_args()

    if args.stdio:
        run_mcp_server(stdio=True)
    else:
        run_mcp_server(
            host=args.host,
            port=args.port,
            ssl_keyfile=args.ssl_keyfile or os.environ.get("SSL_KEYFILE"),
            ssl_certfile=args.ssl_certfile or os.environ.get("SSL_CERTFILE"),
            insecure=args.insecure or os.environ.get("INSECURE") == "1",
        )


if __name__ == "__main__":
    main()
