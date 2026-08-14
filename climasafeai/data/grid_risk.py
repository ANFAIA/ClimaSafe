import math
from datetime import date
from typing import Optional

import numpy as np
import pandas as pd

from climasafeai.data.weather_fetcher import (
    fetch_weather_data,
    fetch_hourly_forecast,
)
from climasafeai.features.weather_indices import heat_index

CLASES = ["SEGURO", "PRECAUCION", "PELIGRO"]

# Mismos colores que el overlay Leaflet (chat/static/index.html)
COLORES_RIESGO = {0: "#27ae60", 1: "#f39c12", 2: "#e74c3c"}
ETIQUETAS_RIESGO = ["Seguro", "Precaución", "Peligro"]

PERFILES_DISPONIBLES = {
    "vulnerable": {
        "label": "Más restrictivo",
        "desc": "Anciano, no aclimatado, comorbilidades",
        "hi_precaucion": 24,
        "hi_peligro": 32,
    },
    "mayor": {
        "label": "Adulto mayor (60+)",
        "desc": "Edad avanzada, riesgo elevado",
        "hi_precaucion": 27,
        "hi_peligro": 35,
    },
    "adulto": {
        "label": "Adulto (30-59)",
        "desc": "Población general activa",
        "hi_precaucion": 30,
        "hi_peligro": 39,
    },
    "joven": {
        "label": "Joven (18-29)",
        "desc": "Bajo riesgo base",
        "hi_precaucion": 32,
        "hi_peligro": 41,
    },
}


def _hi_a_probabilidad(hi: float) -> float:
    if hi <= 20:
        return 0.05
    if hi >= 50:
        return 0.95
    # Sigmoid suave centrado en ~32 °C
    x = (hi - 32) / 6
    return 1.0 / (1.0 + math.exp(-x))


def _generar_celdas(
    lat_center: float,
    lon_center: float,
    radio_km: float,
    paso_km: float = 1.0,
) -> list[dict]:
    celdas = []
    km_por_grado_lat = 111.0
    km_por_grado_lon = 111.0 * math.cos(math.radians(lat_center))

    paso_lat = paso_km / km_por_grado_lat
    paso_lon = paso_km / km_por_grado_lon
    radio_lat = radio_km / km_por_grado_lat
    radio_lon = radio_km / km_por_grado_lon

    lat = lat_center - radio_lat
    while lat <= lat_center + radio_lat + paso_lat / 2:
        lon = lon_center - radio_lon
        while lon <= lon_center + radio_lon + paso_lon / 2:
            dlat = (lat - lat_center) * km_por_grado_lat
            dlon = (lon - lon_center) * km_por_grado_lon
            dist = math.sqrt(dlat**2 + dlon**2)
            if dist <= radio_km:
                celdas.append({
                    "lat": round(lat, 5),
                    "lon": round(lon, 5),
                    "dist_km": round(dist, 2),
                })
            lon += paso_lon
        lat += paso_lat

    return celdas


def _clase_desde_hi(hi: float, perfil_cfg: dict) -> int:
    if hi >= perfil_cfg["hi_peligro"]:
        return 2
    if hi >= perfil_cfg["hi_precaucion"]:
        return 1
    return 0


def _prob_a_clase(prob: float) -> int:
    if prob >= 0.65:
        return 2
    if prob >= 0.40:
        return 1
    return 0


def riesgo_zona_grid(
    lat: float,
    lon: float,
    radio_km: float = 5,
    perfil_id: str = "vulnerable",
    target_date: Optional[date] = None,
    perfil: Optional[dict] = None,
) -> dict:
    perfil_cfg = PERFILES_DISPONIBLES.get(perfil_id, PERFILES_DISPONIBLES["vulnerable"])

    try:
        weather = fetch_weather_data(lat=lat, lon=lon, target_date=target_date)
    except Exception as e:
        return {"error": f"Error fetching weather: {e}"}

    df_hora = weather.get("df_hora")
    hourly_data = None
    if df_hora is not None and not df_hora.empty:
        if "datetime" in df_hora.columns and "t2m_c" in df_hora.columns:
            df = df_hora.copy()
            if "rh" in df.columns and "t2m_c" in df.columns:
                df["heat_index_c"] = heat_index(
                    df["t2m_c"].values, df["rh"].values
                )
            hourly_data = df.to_dict("records")

    horas = []
    if hourly_data:
        for row in hourly_data:
            dt = pd.to_datetime(row.get("datetime"))
            hi = row.get("heat_index_c")
            if hi is not None and not (isinstance(hi, float) and math.isnan(hi)):
                horas.append({"hora": dt.hour, "hi": float(hi)})

    # Filtrar horas según perfil del usuario (hora_inicio + duracion)
    if perfil:
        h_inicio = int(perfil.get("hora_inicio", 0))
        h_duracion = float(perfil.get("duracion_actividad_h", 2))
        h_fin = min(23, h_inicio + max(1, int(h_duracion)))
        horas_actividad = [h for h in horas if h_inicio <= h["hora"] < h_fin]
    else:
        horas_actividad = horas

    if horas_actividad:
        hi_peak = max(h["hi"] for h in horas_actividad)
        hi_min = min(h["hi"] for h in horas_actividad)
    elif horas:
        hi_peak = max(h["hi"] for h in horas)
        hi_min = min(h["hi"] for h in horas)
    else:
        current = weather.get("current", {})
        t = current.get("t2m_c")
        rh = current.get("rh")
        if t is not None and rh is not None:
            hi_peak = float(heat_index(np.array([t]), np.array([rh]))[0])
        else:
            hi_peak = None
        hi_min = hi_peak

    # Calcular clase base o personalizada
    prob_pers = None
    h_fin = None
    if perfil:
        h_inicio_p = int(perfil.get("hora_inicio", 0))
        h_duracion_p = float(perfil.get("duracion_actividad_h", 2))
        h_fin = min(23, h_inicio_p + max(1, int(h_duracion_p)))
    celdas = _generar_celdas(lat, lon, radio_km)

    celdas_resultado = []
    seguros = precaucion = peligro = 0
    for celda in celdas:
        if perfil and hi_peak is not None:
            cl = _prob_a_clase(prob_pers)
        elif hi_peak is not None:
            cl = _clase_desde_hi(hi_peak, perfil_cfg)
        else:
            cl = 0
        if cl == 2:
            peligro += 1
        elif cl == 1:
            precaucion += 1
        else:
            seguros += 1
        celdas_resultado.append({
            "lat": celda["lat"],
            "lon": celda["lon"],
            "hi": round(hi_peak, 1) if hi_peak is not None else None,
            "riesgo": cl,
            "riesgo_label": CLASES[cl],
        })

    perfil_label = "Personalizado" if perfil else perfil_cfg["label"]
    perfil_desc = f"Perfil del usuario" if perfil else perfil_cfg["desc"]

    return {
        "center": {"lat": round(lat, 5), "lon": round(lon, 5)},
        "perfil_usado": "personalizado" if perfil else perfil_id,
        "perfil_label": perfil_label,
        "perfil_desc": perfil_desc,
        "stats": {
            "total_celdas": len(celdas_resultado),
            "seguro": seguros,
            "precaucion": precaucion,
            "peligro": peligro,
            "pct_peligro": round(peligro / len(celdas_resultado) * 100, 1) if celdas_resultado else 0,
        },
        "celdas": celdas_resultado,
        "resumen_horario": {
            "hi_peak": round(hi_peak, 1) if hi_peak is not None else None,
            "hi_min": round(hi_min, 1) if hi_min is not None else None,
            "hora_pico": max(horas_actividad, key=lambda h: h["hi"])["hora"] if horas_actividad else None,
            "ventana_actividad": f"{int(perfil.get('hora_inicio',0))}:00-{h_fin}:00" if perfil and h_fin else "todo el día",
        },
        "prob_personalizada": round(prob_pers, 4) if prob_pers is not None else None,
        "target_date": weather.get("target_date"),
    }


def _paso_celdas(celdas: list[dict]) -> tuple[float, float]:
    """Separación real entre celdas consecutivas, en grados.

    La misma que deduce el frontend para pintar los rectángulos: la diferencia
    entre los lat/lon consecutivos de la cuadrícula (paso 1 km del grid).
    """
    paso_lat = paso_lon = 0.01
    lats = sorted({c["lat"] for c in celdas})
    lons = sorted({c["lon"] for c in celdas})
    if len(lats) > 1:
        paso_lat = min(b - a for a, b in zip(lats, lats[1:]))
    if len(lons) > 1:
        paso_lon = min(b - a for a, b in zip(lons, lons[1:]))
    if paso_lat <= 0.0001:
        paso_lat = 0.01
    if paso_lon <= 0.0001:
        paso_lon = 0.01
    return paso_lat, paso_lon


def celdas_a_featurecollection(celdas: list[dict]) -> dict:
    """Convierte las celdas del grid en una FeatureCollection GeoJSON.

    Cada celda es un Polygon con el mismo rectángulo que pinta el overlay
    Leaflet, con la clase de riesgo y el HI pico en properties. El orden de
    coordenadas es [lon, lat], como exige la especificación GeoJSON.
    """
    paso_lat, paso_lon = _paso_celdas(celdas)
    features = []
    for c in celdas:
        lon, lat = c["lon"], c["lat"]
        rect = [
            [lon - paso_lon / 2, lat - paso_lat / 2],
            [lon + paso_lon / 2, lat - paso_lat / 2],
            [lon + paso_lon / 2, lat + paso_lat / 2],
            [lon - paso_lon / 2, lat + paso_lat / 2],
            [lon - paso_lon / 2, lat - paso_lat / 2],
        ]
        features.append({
            "type": "Feature",
            "geometry": {"type": "Polygon", "coordinates": [rect]},
            "properties": {
                "riesgo": c["riesgo"],
                "riesgo_label": c["riesgo_label"],
                "hi_pico": c.get("hi"),
            },
        })
    return {"type": "FeatureCollection", "features": features}


def render_riesgo_png(
    celdas: list[dict],
    stats: Optional[dict] = None,
    center: Optional[dict] = None,
    resumen: Optional[dict] = None,
    perfil_label: str = "",
) -> bytes:
    """Renderiza el overlay de riesgo del grid en un PNG (bytes).

    Mismo color por clase que el mapa web; la cuadrícula se dibuja con el
    tamaño real de celda (paso 1 km) para que las celdas salgan cuadradas.
    """
    import io

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    paso_lat, paso_lon = _paso_celdas(celdas)
    fig, ax = plt.subplots(figsize=(8, 8))
    for c in celdas:
        ax.add_patch(Rectangle(
            (c["lon"] - paso_lon / 2, c["lat"] - paso_lat / 2),
            paso_lon, paso_lat,
            facecolor=COLORES_RIESGO.get(c["riesgo"], "#999999"),
            edgecolor="white", linewidth=0.3,
        ))
    if center and abs(center.get("lat", 0)) < 89.9:
        # Compensa la compresión de los grados de longitud con la latitud
        ax.set_aspect(1.0 / max(math.cos(math.radians(center["lat"])), 0.01))
    else:
        ax.set_aspect(1.0)
    ax.autoscale()
    ax.set_xlabel("Longitud")
    ax.set_ylabel("Latitud")

    titulo = "Riesgo por zona"
    if perfil_label:
        titulo += f" — {perfil_label}"
    if resumen and resumen.get("hi_peak") is not None:
        titulo += f" — HI pico {resumen['hi_peak']}°C"
    if stats:
        titulo += f" ({stats.get('total_celdas', '?')} celdas)"
    ax.set_title(titulo, fontsize=11)

    handles = [
        Rectangle((0, 0), 1, 1, facecolor=COLORES_RIESGO[i], label=ETIQUETAS_RIESGO[i])
        for i in (0, 1, 2)
    ]
    ax.legend(handles=handles, loc="lower right", fontsize=9)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    return buf.getvalue()
