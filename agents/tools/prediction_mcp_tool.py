"""
agents.tools.prediction_mcp_tool — MCP server para predicción de riesgo.

Expone tools MCP para que un LLM (claude, etc.) pueda:
  - Predecir riesgo individual (con contrafactuales y recomendaciones)
  - Dibujar la curva de riesgo por hora y devolverla como imagen PNG
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
import base64
import io
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


def _aplicar_deporte_a_perfil(perfil: dict) -> dict:
    """Si el perfil lleva deporte, la intensidad se deriva del MET del Compendium.

    ``nivel_actividad_de_deporte`` devuelve el nivel (reposo..muy_intensa) del
    MET publicado del deporte; ese nivel manda sobre cualquier nivel_actividad
    por defecto, igual que hace el bot en ``_perfil_prediccion_desde_rutina``.
    Si el deporte no está en la tabla se deja el nivel que ya traía el perfil.
    """
    deporte = perfil.get("deporte")
    if not deporte:
        return perfil
    from climasafeai.features.personalizacion import nivel_actividad_de_deporte

    nivel = nivel_actividad_de_deporte(deporte)
    if nivel:
        perfil["nivel_actividad"] = nivel
    return perfil


def predict_risk(
    lat: float, lon: float, provincia: str = "Madrid",
    edad: int = 40, sexo: str = "hombre",
    nivel_actividad: str = "ligera",
    hora_inicio: int = 10, duracion_h: float = 2.0,
    aclimatado: bool | None = None,
    grasa: float | None = None, entrenado: bool | None = None,
    ocupacion: str | None = None, deporte: str | None = None,
    comorbilidades: list[str] | None = None,
    medicacion: list[str] | None = None,
    fototipo: str | None = None,
    situacion_social: list[str] | None = None,
    falta_sueno: bool | None = None,
    enfermedad_reciente: bool | None = None,
    fiesta: bool | None = None,
    fecha: str | None = None,
    incluir_contrafactuales: bool = True,
    incluir_recomendaciones: bool = True,
) -> dict:
    target_date = _parse_date(fecha)
    from climasafeai.features.personalizacion import (
        riesgo_horario_acumulado, recomendar_horario, pico_riesgo_actividad,
    )

    perfil: dict[str, Any] = {
        "edad": edad,
        "sexo": sexo,
        "nivel_actividad": nivel_actividad,
        "hora_inicio": hora_inicio,
        "duracion_actividad_h": duracion_h,
    }
    if aclimatado is not None:
        perfil["aclimatado"] = aclimatado
    if grasa is not None:
        perfil["porcentaje_grasa"] = grasa
    if entrenado is not None:
        perfil["entrenado"] = entrenado
    if ocupacion is not None:
        perfil["ocupacion"] = ocupacion
    if deporte is not None:
        perfil["deporte"] = deporte
    if fototipo is not None:
        perfil["fototipo"] = fototipo
    if situacion_social:
        perfil["situacion_social"] = set(situacion_social)
    if falta_sueno is not None:
        perfil["falta_sueno"] = falta_sueno
    if enfermedad_reciente is not None:
        perfil["enfermedad_reciente"] = enfermedad_reciente
    if fiesta is not None:
        perfil["fiesta"] = fiesta
    if comorbilidades:
        perfil["comorbilidades"] = set(comorbilidades)
    if medicacion:
        perfil["farmacos"] = set(medicacion)

    _aplicar_deporte_a_perfil(perfil)

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
    if perfil:
        _aplicar_deporte_a_perfil(perfil)
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

                _aplicar_deporte_a_perfil(perfil)
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
        _aplicar_deporte_a_perfil(perfil_rango)
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
        "factores_detalle": factores_detalle,
        "demografico": demografico,
        "resumen": _generar_resumen(pct_peligro, total_peligro, total_precaucion, total_seguros, factor_extra, factores_detalle, actividad),
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

    perfil_horario = perfil_horario_desde_df(df_hora, target_date=weather.get("target_date"))
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


# ── Gráfica del riesgo por hora (PNG en memoria) ────────────────────────────


def _hi_a_nivel(hi: float) -> float:
    """Heat Index (°C) → peligrosidad ambiental 1-10.

    Misma escala tramo a tramo que ``hiToNivel`` de la web (chat/static/index.html),
    para que la imagen del MCP y la gráfica del navegador se lean igual.
    """
    if hi <= 15:
        return 1.0
    if hi <= 27:
        return 1 + (hi - 15) / 12
    if hi <= 32:
        return 2 + (hi - 27) / 5 * 2
    if hi <= 39:
        return 4 + (hi - 32) / 7 * 3
    if hi <= 46:
        return 7 + (hi - 39) / 7 * 2
    return min(10.0, 9 + (hi - 46) / 9)


def grafica_riesgo_horario_png(
    result: dict,
    hora_inicio: float | None = None,
    duracion_h: float | None = None,
    edad: int | None = None,
    fecha: str | None = None,
) -> bytes | None:
    """PNG de la curva de riesgo por hora a partir de un resultado de ``predict_risk``.

    NO calcula riesgo: consume ``result["riesgo_horario"]`` (curva personal 0-1) y
    ``result["weather"]["perfil_horario"]`` (HI por hora) que el pipeline ya dejó
    en el resultado. Devuelve ``None`` si no hay serie horaria que pintar.

    matplotlib se importa DENTRO de la función a propósito: el bloque MCP vive en
    un ``try/except ImportError`` y un import fallido a nivel de módulo apagaría
    todas las tools en silencio.
    """
    perfil_horario = (result.get("weather") or {}).get("perfil_horario") or []
    curva = result.get("riesgo_horario") or []
    puntos = sorted(
        (p for p in perfil_horario if p.get("HI") is not None),
        key=lambda p: p["hora"],
    )
    if not puntos or not curva:
        return None

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    horas = [int(p["hora"]) for p in puntos]
    niveles = [_hi_a_nivel(float(p["HI"])) for p in puntos]
    riesgo_por_hora = {int(c["hora"]): c["riesgo"] for c in curva}
    # NaN y no None en las horas sin curva: matplotlib deja hueco en la línea y
    # el array sigue siendo numérico.
    riesgos = [
        float(riesgo_por_hora[h]) * 100 if riesgo_por_hora.get(h) is not None else float("nan")
        for h in horas
    ]

    fig, ax = plt.subplots(figsize=(9.0, 4.2))
    ax2 = ax.twinx()

    # Ventana de actividad: sombreado gris de fondo.
    if hora_inicio is not None and duracion_h:
        ax.axvspan(float(hora_inicio), float(hora_inicio) + float(duracion_h),
                   color="#2c3e50", alpha=0.10, zorder=0,
                   label=f"Tu actividad ({int(hora_inicio)}:00-{int(float(hora_inicio) + float(duracion_h))}:00)")

    # Franja recomendada por el backend (recomendar_horario), en verde.
    rec = result.get("recomendacion_horario") or {}
    if rec.get("hora_inicio") is not None and rec.get("hora_fin") is not None:
        ax.axvspan(float(rec["hora_inicio"]), float(rec["hora_fin"]),
                   color="#27ae60", alpha=0.14, zorder=0,
                   label=f"Recomendado ({int(rec['hora_inicio'])}:00-{int(rec['hora_fin'])}:00)")

    ax.plot(horas, niveles, color="#2c3e50", linewidth=2.0, marker="o",
            markersize=3.5, label="Peligrosidad ambiental (HI)", zorder=3)
    etiqueta_riesgo = f"Tu riesgo ({edad} años)" if edad is not None else "Tu riesgo"
    ax2.plot(horas, riesgos, color="#e74c3c", linewidth=2.5,
             label=etiqueta_riesgo, zorder=4)

    # Pico de la curva personal, con etiqueta directa.
    validos = [(h, r) for h, r in zip(horas, riesgos) if r == r]  # descarta NaN
    if validos:
        hora_pico, pico = max(validos, key=lambda hr: hr[1])
        ax2.plot([hora_pico], [pico], marker="o", markersize=8,
                 color="#e74c3c", markeredgecolor="white", markeredgewidth=2, zorder=5)
        ax2.annotate(f"pico {pico:.0f}% a las {hora_pico}:00",
                     xy=(hora_pico, pico), xytext=(0, 12),
                     textcoords="offset points", ha="center",
                     fontsize=9, color="#c0392b")

    ax.set_xlabel("Hora del día", fontsize=10)
    ax.set_ylabel("Nivel de peligro (1-10)", fontsize=10, color="#2c3e50")
    ax2.set_ylabel("Tu riesgo (%)", fontsize=10, color="#c0392b")
    ax.set_ylim(0.5, 10.5)
    ax2.set_ylim(0, 100)
    ax.set_xlim(min(horas) - 0.5, max(horas) + 0.5)
    ax.set_xticks([h for h in horas if h % 2 == 0])
    ax.set_xticklabels([f"{h:02d}" for h in horas if h % 2 == 0], fontsize=9)
    ax.grid(axis="y", color="#000000", alpha=0.07)
    ax.set_axisbelow(True)
    for lado in ("top", "right", "left"):
        ax.spines[lado].set_visible(False)
    ax2.spines["top"].set_visible(False)

    provincia = (result.get("weather") or {}).get("provincia") or ""
    titulo = "Riesgo por hora"
    if provincia:
        titulo += f" — {provincia}"
    if fecha:
        titulo += f" ({fecha})"
    ax.set_title(titulo, fontsize=12, loc="left")

    manejadores = ax.get_legend_handles_labels()[0] + ax2.get_legend_handles_labels()[0]
    etiquetas = ax.get_legend_handles_labels()[1] + ax2.get_legend_handles_labels()[1]
    # Fondo semitransparente: la leyenda va arriba a la izquierda y ahí puede
    # cruzarse con las curvas cuando el riesgo ya es alto de mañana.
    ax.legend(manejadores, etiquetas, loc="upper left", fontsize=8, ncol=2,
              framealpha=0.85, edgecolor="none").set_zorder(10)

    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=100, bbox_inches="tight")
    plt.close(fig)
    return buffer.getvalue()


# ── Rutinas semanales: resolución de perfil/chat y perfil de predicción ─────


def _resolver_chat(
    alias: str | None = None,
    perfil_id: int | None = None,
    chat_id: str | None = None,
) -> tuple[str | None, dict | None]:
    """Resuelve el chat_id sobre el que viven las rutinas.

    Acepta el chat_id directamente o un perfil (alias/perfil_id) con chat de
    Telegram vinculado. Devuelve (chat_id, error); error no es None si falla.
    """
    from climasafeai.db.manager import DBManager

    db = DBManager()
    if chat_id:
        return str(chat_id), None

    if alias or perfil_id is not None:
        if perfil_id is not None:
            perfil = db.obtener_perfil(int(perfil_id))
        else:
            match = db.buscar_por_alias(alias)
            if not match:
                return None, {"error": f"No se encontró perfil con alias '{alias}'"}
            perfil = db.obtener_perfil(match["id"])
        if not perfil:
            return None, {"error": "Perfil no encontrado"}
        chat = perfil.get("telegram_chat_id")
        if not chat:
            return None, {
                "error": "El perfil no tiene un chat de Telegram vinculado: indica "
                "chat_id o vincula el perfil a un chat antes de usar rutinas"
            }
        return str(chat), None

    return None, {"error": "Indica uno de: chat_id, alias o perfil_id"}


def _resolver_perfil(
    alias: str | None = None,
    perfil_id: int | None = None,
    chat_id: str | None = None,
) -> tuple[dict | None, dict | None]:
    """Resuelve el perfil completo por alias, perfil_id o chat_id."""
    from climasafeai.db.manager import DBManager

    db = DBManager()
    pid: int | None = None
    if perfil_id is not None:
        pid = int(perfil_id)
    elif alias:
        match = db.buscar_por_alias(alias)
        if not match:
            return None, {"error": f"No se encontró perfil con alias '{alias}'"}
        pid = match["id"]
    elif chat_id:
        match = db.buscar_por_telegram(str(chat_id))
        if not match:
            return None, {"error": f"No hay perfil vinculado al chat '{chat_id}'"}
        pid = match["id"]
    else:
        return None, {"error": "Indica uno de: alias, perfil_id o chat_id"}

    perfil = db.obtener_perfil(pid) if pid is not None else None
    if not perfil:
        return None, {"error": "Perfil no encontrado"}
    return perfil, None


def _validar_hora_aviso(hora: str) -> str | None:
    """Valida 'HH:MM' y lo normaliza a 'HH:MM' con dos dígitos, o None."""
    import re

    m = re.fullmatch(r"(\d{1,2}):(\d{2})", hora.strip())
    if not m:
        return None
    hh, mm = int(m.group(1)), int(m.group(2))
    if hh > 23 or mm > 59:
        return None
    return f"{hh:02d}:{mm:02d}"


def _fmt_hora(h: float) -> str:
    """8.0 → '8:00', 18.5 → '18:30'."""
    hh = int(h)
    mm = int(round((h - hh) * 60))
    return f"{hh}:{mm:02d}"


def _validar_dias(dias: str) -> list[int] | None:
    """Valida '1,2,3,4,5' (1=lunes ... 7=domingo). Devuelve lista o None."""
    try:
        nums = [int(d) for d in dias.split(",") if d.strip()]
    except (ValueError, TypeError):
        return None
    if not nums or any(n < 1 or n > 7 for n in nums):
        return None
    return sorted(set(nums))


def _perfil_prediccion_desde_rutina(perfil: dict, rutina: dict) -> dict:
    """Perfil para predict_ensemble: datos del perfil + ventana de la rutina.

    La ventana a evaluar la define la rutina (hora_inicio + duración), no el
    perfil; igual que hace el bot en ``_perfil_prediccion_desde_rutina``. Si la
    rutina lleva deporte, la intensidad se deriva del MET del Compendium.
    """
    p: dict[str, Any] = {
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
    return _aplicar_deporte_a_perfil(p)


def _temps_en_ventana(perfil_horario: list[dict], perfil_usuario: dict) -> list[float]:
    """Temperaturas previstas en las horas en las que el usuario estará fuera."""
    inicio = perfil_usuario.get("hora_inicio")
    duracion = perfil_usuario.get("duracion_actividad_h")
    if inicio is not None and duracion is not None:
        en_ventana = [
            h["temp"] for h in perfil_horario
            if inicio <= h["hora"] < inicio + duracion and h.get("temp") is not None
        ]
        if en_ventana:
            return en_ventana
    return [h["temp"] for h in perfil_horario if h.get("temp") is not None]


# ── MCP Server ───────────────────────────────────────────────────────

try:
    from mcp.server.fastmcp import FastMCP
    from mcp.types import ImageContent
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
        grasa: Optional[float] = None,
        ocupacion: Optional[str] = None,
        entrenado: Optional[bool] = None,
        deporte: Optional[str] = None,
        comorbilidades: Optional[str] = None,
        medicacion: Optional[str] = None,
        fototipo: Optional[str] = None,
        situacion_social: Optional[str] = None,
        falta_sueno: Optional[bool] = None,
        enfermedad_reciente: Optional[bool] = None,
        fiesta: Optional[bool] = None,
        fecha: Optional[str] = None,
    ) -> str:
        """Predice riesgo cardiovascular para 1 persona.

hora_inicio, duracion_h, nivel_actividad y ubicación son de ESTA salida, no del
perfil. `aclimatado` es al CALOR ambiental, no a un deporte. Del cuerpo solo se usa
`grasa` (%): si el usuario no la sabe, omítela.

`ocupacion` describe ESTA salida, no el oficio del usuario: mándala SOLO si sale a
trabajar (reparto, mantenimiento, construccion, campo; oficina si es bajo techo).
Un peón de campo que sale a pasear el domingo NO lleva ocupacion — pesa hasta ×2.7.
`entrenado` es si está acostumbrado a ESA actividad: reduce a la mitad el extra de
esfuerzo. `deporte` es solo la etiqueta ("senderismo").

`situacion_social` separada por comas: vive_solo, no_sale, sin_aire_acondicionado,
vivienda_fria. `fototipo` escala Fitzpatrick 1-6."""
        comorb_list = [c.strip() for c in comorbilidades.split(",")] if comorbilidades else None
        med_list = [m.strip() for m in medicacion.split(",")] if medicacion else None
        sit_list = [s.strip() for s in situacion_social.split(",")] if situacion_social else None
        result = predict_risk(
            lat=lat, lon=lon, provincia=provincia,
            edad=edad, sexo=sexo, nivel_actividad=nivel_actividad,
            hora_inicio=hora_inicio, duracion_h=duracion_h,
            aclimatado=aclimatado, grasa=grasa,
            ocupacion=ocupacion, entrenado=entrenado, deporte=deporte,
            comorbilidades=comorb_list, medicacion=med_list,
            fototipo=fototipo, situacion_social=sit_list,
            falta_sueno=falta_sueno, enfermedad_reciente=enfermedad_reciente,
            fiesta=fiesta,
            fecha=fecha,
        )
        _sanitize(result)
        return json.dumps(result, indent=2, default=str, ensure_ascii=False)

    @_mcp.tool()
    def listar_usuarios_mcp() -> str:
        """Lista los perfiles guardados: alias, edad, sexo."""
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
        """Carga un perfil por su alias exacto. Su nivel_actividad, hora_inicio y duracion
son genéricos: no valen para la predicción, pide los de esta salida."""
        from climasafeai.db.manager import DBManager
        db = DBManager()
        match = db.buscar_por_alias(alias)
        if not match:
            return json.dumps({"error": f"No se encontró perfil con alias '{alias}'"}, ensure_ascii=False)
        perfil = db.obtener_perfil(match["id"])
        if not perfil:
            return json.dumps({"error": f"Perfil ID {match['id']} no encontrado"}, ensure_ascii=False)
        return json.dumps(perfil, indent=2, ensure_ascii=False, default=str)

    @_mcp.tool()
    def cargar_perfil_por_chat_id_mcp(chat_id: str) -> str:
        """Carga el perfil vinculado a un chat_id de Telegram, si lo hay."""
        from climasafeai.db.manager import DBManager
        db = DBManager()
        match = db.buscar_por_telegram(chat_id)
        if not match:
            return json.dumps({"encontrado": False, "mensaje": "No hay perfil vinculado a este chat. Pregunta al usuario si tiene un alias o si quiere crear un perfil nuevo."}, ensure_ascii=False)
        perfil = db.obtener_perfil(match["id"])
        if not perfil:
            return json.dumps({"encontrado": False, "mensaje": "Perfil no encontrado"}, ensure_ascii=False)
        perfil["encontrado"] = True
        return json.dumps(perfil, indent=2, ensure_ascii=False, default=str)

    @_mcp.tool()
    def vincular_chat_id_mcp(alias: str, chat_id: str) -> str:
        """Vincula un chat_id de Telegram a un perfil existente, buscado por alias."""
        from climasafeai.db.manager import DBManager
        db = DBManager()
        match = db.buscar_por_alias(alias)
        if not match:
            return json.dumps({"success": False, "error": f"No se encontró perfil con alias '{alias}'"}, ensure_ascii=False)
        db.actualizar_perfil(match["id"], {"telegram_chat_id": chat_id})
        return json.dumps({"success": True, "alias": alias, "chat_id": chat_id, "mensaje": f"Chat vinculado a '{alias}'"}, ensure_ascii=False)

    @_mcp.tool()
    def crear_perfil_mcp(alias: str, edad: int, sexo: str, grasa: Optional[float] = None, aclimatado: bool = False, comorbilidades: Optional[str] = None, medicacion: Optional[str] = None, nivel_actividad: Optional[str] = None, fototipo: Optional[str] = None, situacion_social: Optional[str] = None, chat_id: Optional[str] = None) -> str:
        """Crea un perfil y opcionalmente lo vincula a un chat de Telegram.

`grasa` es el % graso; si no lo sabe, omítela. `comorbilidades` y `medicacion` van
separadas por comas, p. ej.
"cardiovascular,diabetes" y "diureticos_asa,antipsicoticos".

`situacion_social` separada por comas: vive_solo, no_sale, sin_aire_acondicionado,
vivienda_fria. `fototipo` escala Fitzpatrick 1-6."""
        from climasafeai.db.manager import DBManager
        db = DBManager()
        exist = db.buscar_por_alias(alias)
        if exist:
            return json.dumps({"success": False, "error": f"Ya existe un perfil con alias '{alias}'"}, ensure_ascii=False)
        datos = {"alias": alias, "edad": edad, "sexo": sexo}
        # La tabla `perfiles` guarda porcentaje_grasa; no hay columna de peso ni altura.
        if grasa is not None:
            datos["porcentaje_grasa"] = grasa
        datos["aclimatado"] = aclimatado
        if comorbilidades:
            datos["comorbilidades"] = [c.strip() for c in comorbilidades.split(",")]
        if medicacion:
            datos["farmacos"] = [m.strip() for m in medicacion.split(",")]
        if nivel_actividad:
            datos["nivel_actividad"] = nivel_actividad
        if fototipo is not None:
            datos["fototipo"] = fototipo
        if situacion_social:
            datos["situacion_social"] = [s.strip() for s in situacion_social.split(",")]
        if chat_id:
            datos["telegram_chat_id"] = chat_id
        pid = db.crear_perfil(datos)
        return json.dumps({"success": True, "id": pid, "alias": alias, "mensaje": f"Perfil '{alias}' creado correctamente"}, indent=2, ensure_ascii=False)

    @_mcp.tool()
    def listar_rutinas_mcp(
        alias: Optional[str] = None,
        perfil_id: Optional[int] = None,
        chat_id: Optional[str] = None,
    ) -> str:
        """Lista las rutinas semanales de un perfil, por alias, perfil_id o chat_id.

Cada rutina trae: id, nombre, dias (1-7, 1=lunes), hora_inicio, hora_fin y
opcionalmente ocupacion o deporte."""
        chat, err = _resolver_chat(alias, perfil_id, chat_id)
        if err:
            return json.dumps(err, ensure_ascii=False)
        from climasafeai.db.manager import DBManager

        rutinas = DBManager().listar_rutinas(chat)
        for r in rutinas:
            r.pop("chat_id", None)
        if not rutinas:
            return json.dumps({"rutinas": [], "mensaje": "Este perfil/chat no tiene rutinas todavía"}, ensure_ascii=False)
        return json.dumps(rutinas, indent=2, ensure_ascii=False, default=str)

    @_mcp.tool()
    def crear_rutina_mcp(
        nombre: str,
        dias: str,
        hora_inicio: float,
        hora_fin: float,
        alias: Optional[str] = None,
        perfil_id: Optional[int] = None,
        chat_id: Optional[str] = None,
        ocupacion: Optional[str] = None,
        deporte: Optional[str] = None,
    ) -> str:
        """Crea una rutina semanal para un perfil/chat y devuelve su id.

`dias` es una cadena con los días 1-7 separados por coma (1=lunes, 7=domingo),
p. ej. "1,2,3,4,5". `hora_inicio` y `hora_fin` en formato 24h (8.5 = 8:30).
`deporte` solo si la rutina es un deporte conocido (correr, senderismo, futbol,
ciclismo, tenis...): su intensidad se deriva del MET del Compendium."""
        chat, err = _resolver_chat(alias, perfil_id, chat_id)
        if err:
            return json.dumps(err, ensure_ascii=False)
        nums = _validar_dias(dias)
        if nums is None:
            return json.dumps({"success": False, "error": f"dias inválido '{dias}': usa números 1-7 separados por coma (1=lunes)"}, ensure_ascii=False)
        if not (0 <= hora_inicio < 24 and 0 < hora_fin <= 24 and hora_fin > hora_inicio):
            return json.dumps({"success": False, "error": f"ventana inválida {hora_inicio}-{hora_fin}: hora_fin debe ser mayor que hora_inicio y ambas entre 0 y 24"}, ensure_ascii=False)
        from climasafeai.db.manager import DBManager

        rid = DBManager().crear_rutina(
            chat, nombre.strip(), ",".join(str(n) for n in nums),
            float(hora_inicio), float(hora_fin),
            ocupacion=ocupacion, deporte=deporte,
        )
        return json.dumps(
            {"success": True, "id": rid, "nombre": nombre.strip(), "dias": ",".join(str(n) for n in nums),
             "hora_inicio": hora_inicio, "hora_fin": hora_fin,
             "mensaje": f"Rutina '{nombre.strip()}' creada (id {rid})"},
            indent=2, ensure_ascii=False,
        )

    @_mcp.tool()
    def borrar_rutina_mcp(
        rutina_id: int,
        alias: Optional[str] = None,
        perfil_id: Optional[int] = None,
        chat_id: Optional[str] = None,
    ) -> str:
        """Borra una rutina propia por su id (los ids salen en listar_rutinas_mcp).

Hay que identificar al dueño con alias, perfil_id o chat_id: solo se borran
rutinas de ese perfil/chat."""
        chat, err = _resolver_chat(alias, perfil_id, chat_id)
        if err:
            return json.dumps(err, ensure_ascii=False)
        from climasafeai.db.manager import DBManager

        db = DBManager()
        rid = int(rutina_id)
        if not any(r["id"] == rid for r in db.listar_rutinas(chat)):
            return json.dumps(
                {"success": False, "error": f"No existe la rutina {rid} en este perfil/chat"},
                ensure_ascii=False,
            )
        db.eliminar_rutina(rid)
        return json.dumps({"success": True, "id": rid, "mensaje": "Rutina eliminada"}, ensure_ascii=False)

    @_mcp.tool()
    def configurar_hora_aviso_mcp(
        hora: Optional[str] = None,
        alias: Optional[str] = None,
        perfil_id: Optional[int] = None,
        chat_id: Optional[str] = None,
    ) -> str:
        """Configura o consulta la hora de aviso diario de un perfil/chat.

`hora` en formato HH:MM (p. ej. "08:00") configura el aviso; "off" lo desactiva;
si se omite, solo consulta la hora actual."""
        chat, err = _resolver_chat(alias, perfil_id, chat_id)
        if err:
            return json.dumps(err, ensure_ascii=False)
        from climasafeai.db.manager import DBManager

        db = DBManager()
        if hora is not None and hora.strip().lower() != "off":
            norm = _validar_hora_aviso(hora)
            if norm is None:
                return json.dumps({"success": False, "error": f"Hora inválida '{hora}': usa HH:MM entre 00:00 y 23:59, u 'off' para desactivar"}, ensure_ascii=False)
            db.guardar_hora_aviso(chat, norm)
            return json.dumps({"success": True, "chat_id": chat, "hora": norm, "mensaje": f"Aviso diario configurado a las {norm}"}, ensure_ascii=False)
        if hora is not None and hora.strip().lower() == "off":
            db.guardar_hora_aviso(chat, None)
            return json.dumps({"success": True, "chat_id": chat, "hora": None, "mensaje": "Aviso diario desactivado"}, ensure_ascii=False)
        actual = db.obtener_hora_aviso(chat)
        return json.dumps(
            {"chat_id": chat, "hora": actual,
             "mensaje": f"Aviso diario configurado a las {actual}" if actual else "No hay hora de aviso configurada"},
            ensure_ascii=False,
        )

    @_mcp.tool()
    def riesgo_rutinas_dia_mcp(
        alias: Optional[str] = None,
        perfil_id: Optional[int] = None,
        chat_id: Optional[str] = None,
        weekday: Optional[int] = None,
        fecha: Optional[str] = None,
    ) -> str:
        """Calcula el riesgo de cada rutina de un perfil para un día de la semana.

`weekday` usa 1=lunes ... 7=domingo; si se omite usa el día actual. Para cada
rutina del día calcula con predict_ensemble el riesgo de su ventana
(hora_inicio-hora_fin) y devuelve clase, probabilidad, temperatura media y
recomendación por ventana."""
        perfil, err = _resolver_perfil(alias, perfil_id, chat_id)
        if err:
            return json.dumps(err, ensure_ascii=False)
        if perfil.get("lat") is None or perfil.get("lon") is None:
            return json.dumps(
                {"error": "El perfil no tiene ubicación (lat/lon): configura lat, lon y provincia en el perfil antes de calcular el riesgo por rutinas"},
                ensure_ascii=False,
            )
        chat = perfil.get("telegram_chat_id")
        if not chat:
            return json.dumps({"error": "El perfil no tiene un chat de Telegram vinculado: no hay rutinas asociadas"}, ensure_ascii=False)
        if weekday is not None and int(weekday) not in range(1, 8):
            return json.dumps({"error": f"weekday inválido {weekday}: usa 1=lunes ... 7=domingo"}, ensure_ascii=False)
        wd = int(weekday) if weekday is not None else date_type.today().isoweekday()
        target_date = _parse_date(fecha)

        from climasafeai.db.manager import DBManager
        from climasafeai.models.recomendaciones import recomendacion_resumen

        rutinas = DBManager().rutinas_por_dia(chat, wd)
        ventanas = []
        for r in rutinas:
            perfil_pred = _perfil_prediccion_desde_rutina(perfil, r)
            try:
                pred = _try_prediction(
                    lat=perfil["lat"], lon=perfil["lon"],
                    provincia=perfil.get("provincia") or "Madrid",
                    perfil=perfil_pred, target_date=target_date,
                )
            except Exception as exc:
                ventanas.append({
                    "id": r["id"], "nombre": r["nombre"],
                    "ventana": f"{_fmt_hora(r['hora_inicio'])}-{_fmt_hora(r['hora_fin'])}",
                    "error": str(exc),
                })
                continue
            temps = _temps_en_ventana(pred.get("weather", {}).get("perfil_horario") or [], perfil_pred)
            ventanas.append({
                "id": r["id"],
                "nombre": r["nombre"],
                "dias": r["dias"],
                "hora_inicio": r["hora_inicio"],
                "hora_fin": r["hora_fin"],
                "ventana": f"{_fmt_hora(r['hora_inicio'])}-{_fmt_hora(r['hora_fin'])}",
                "deporte": r.get("deporte"),
                "ocupacion": r.get("ocupacion"),
                "nivel_actividad": perfil_pred.get("nivel_actividad"),
                "clase_final": pred.get("clase_final"),
                "clase_final_label": pred.get("clase_final_label"),
                "prob_calor": pred.get("perfil", {}).get("calor", {}).get("prob_personalizada"),
                "prob_frio": pred.get("perfil", {}).get("frio", {}).get("prob_personalizada"),
                "temp_media_ventana_c": round(sum(temps) / len(temps), 1) if temps else None,
                "recomendacion": recomendacion_resumen(pred),
            })

        return json.dumps(
            {
                "alias": perfil.get("alias"),
                "perfil_id": perfil["id"],
                "chat_id": chat,
                "weekday": wd,
                "fecha": target_date.isoformat() if target_date else "hoy",
                "provincia": perfil.get("provincia") or "Madrid",
                "num_rutinas": len(ventanas),
                "ventanas": ventanas,
            },
            indent=2, ensure_ascii=False, default=str,
        )

    @_mcp.tool(structured_output=False)
    def grafica_riesgo_horario_mcp(
        lat: float,
        lon: float,
        provincia: str = "",
        edad: int = 40,
        sexo: str = "hombre",
        nivel_actividad: str = "ligera",
        hora_inicio: int = 10,
        duracion_h: float = 2.0,
        aclimatado: bool = False,
        grasa: Optional[float] = None,
        ocupacion: Optional[str] = None,
        entrenado: Optional[bool] = None,
        deporte: Optional[str] = None,
        comorbilidades: Optional[str] = None,
        medicacion: Optional[str] = None,
        fototipo: Optional[str] = None,
        situacion_social: Optional[str] = None,
        falta_sueno: Optional[bool] = None,
        enfermedad_reciente: Optional[bool] = None,
        fiesta: Optional[bool] = None,
        fecha: Optional[str] = None,
    ) -> ImageContent | str:
        """Devuelve la curva de riesgo por hora como IMAGEN PNG, no como JSON.

Misma gráfica que enseña la web: eje izquierdo la peligrosidad ambiental 1-10
derivada del Heat Index, eje derecho tu riesgo personal en % hora a hora, con la
ventana de actividad sombreada, la franja recomendada en verde y el pico marcado.

Los parámetros son los mismos que `predict_risk_mcp`: úsala cuando el usuario
pida ver, dibujar o graficar el riesgo del día. Si quieres los números en vez
del dibujo, usa `predict_risk_mcp`."""
        comorb_list = [c.strip() for c in comorbilidades.split(",")] if comorbilidades else None
        med_list = [m.strip() for m in medicacion.split(",")] if medicacion else None
        sit_list = [s.strip() for s in situacion_social.split(",")] if situacion_social else None
        result = predict_risk(
            lat=lat, lon=lon, provincia=provincia,
            edad=edad, sexo=sexo, nivel_actividad=nivel_actividad,
            hora_inicio=hora_inicio, duracion_h=duracion_h,
            aclimatado=aclimatado, grasa=grasa,
            ocupacion=ocupacion, entrenado=entrenado, deporte=deporte,
            comorbilidades=comorb_list, medicacion=med_list,
            fototipo=fototipo, situacion_social=sit_list,
            falta_sueno=falta_sueno, enfermedad_reciente=enfermedad_reciente,
            fiesta=fiesta,
            fecha=fecha,
            incluir_contrafactuales=False,
            incluir_recomendaciones=False,
        )
        png = grafica_riesgo_horario_png(
            result, hora_inicio=hora_inicio, duracion_h=duracion_h,
            edad=edad, fecha=fecha,
        )
        if png is None:
            return (
                "No hay perfil horario para esta ubicación y fecha, así que no hay "
                "curva que dibujar. Usa predict_risk_mcp para el riesgo puntual."
            )
        return ImageContent(
            type="image",
            data=base64.b64encode(png).decode("ascii"),
            mimeType="image/png",
        )

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
