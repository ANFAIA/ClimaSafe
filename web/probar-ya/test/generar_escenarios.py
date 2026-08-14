#!/usr/bin/env python3
"""Genera web/probar-ya/scenarios.json — 5 escenarios OFFLINE para la demo y el
test de paridad (WEB-012).

Cada escenario lleva las horas crudas que usaría `fetch_weather_data`
(histórico 14 días + día objetivo), el `current` (mediodía del día objetivo) y
un perfil de usuario. Con esto el pipeline JS y el Python pueden ejecutarse SIN
red sobre EXACTAMENTE los mismos datos.

Regenerar solo si cambian los escenarios o la fecha objetivo:
    uv run python web/probar-ya/test/generar_escenarios.py
"""
import json
import math
from datetime import date, timedelta
from pathlib import Path

TARGET_DATE = date(2026, 8, 14)
DIAS_HIST = 14
OUT = Path(__file__).resolve().parent.parent / "scenarios.json"

ESCENARIOS = [
    {
        "nombre": "dia_templado",
        "lat": 40.4168, "lon": -3.7038, "provincia": "Madrid",
        "t2m_c": 22.0, "rh": 55.0, "wind_speed_kmh": 10.0, "sp": 1013.0,
        "perfil": {
            "edad": 45, "sexo": "hombre", "porcentaje_grasa": 22.0,
            "aclimatado": True, "nivel_actividad": "moderada",
            "hora_inicio": 10, "duracion_actividad_h": 2.0,
            "entrenado": True, "deporte": "correr",
            "comorbilidades": [], "farmacos": [], "situacion_social": [],
            "fototipo": "3", "falta_sueno": False, "enfermedad_reciente": False,
            "fiesta": False, "ocupacion": None,
        },
    },
    {
        "nombre": "ola_calor_humeda",
        "lat": 37.3891, "lon": -5.9845, "provincia": "Sevilla",
        "t2m_c": 38.0, "rh": 62.0, "wind_speed_kmh": 8.0, "sp": 1008.0,
        "perfil": {
            "edad": 72, "sexo": "mujer", "porcentaje_grasa": 28.0,
            "aclimatado": False, "nivel_actividad": "moderada",
            "hora_inicio": 12, "duracion_actividad_h": 3.0,
            "entrenado": False, "deporte": "caminar",
            "comorbilidades": ["cardiovascular", "diabetes"],
            "farmacos": ["diureticos_asa"], "situacion_social": ["vive_solo"],
            "fototipo": "2", "falta_sueno": False, "enfermedad_reciente": False,
            "fiesta": False, "ocupacion": None,
        },
    },
    {
        "nombre": "ola_calor_seca",
        "lat": 37.8882, "lon": -4.7794, "provincia": "Córdoba",
        "t2m_c": 40.5, "rh": 22.0, "wind_speed_kmh": 16.0, "sp": 1005.0,
        "perfil": {
            "edad": 30, "sexo": "hombre", "porcentaje_grasa": 18.0,
            "aclimatado": True, "nivel_actividad": "intensa",
            "hora_inicio": 18, "duracion_actividad_h": 1.5,
            "entrenado": True, "deporte": "ciclismo",
            "comorbilidades": [], "farmacos": [], "situacion_social": [],
            "fototipo": "4", "falta_sueno": False, "enfermedad_reciente": False,
            "fiesta": False, "ocupacion": None,
        },
    },
    {
        "nombre": "dia_frio_humedo",
        "lat": 41.5034, "lon": -5.7443, "provincia": "Zamora",
        "t2m_c": 1.0, "rh": 85.0, "wind_speed_kmh": 24.0, "sp": 1002.0,
        "perfil": {
            "edad": 65, "sexo": "hombre", "porcentaje_grasa": 25.0,
            "aclimatado": True, "nivel_actividad": "ligera",
            "hora_inicio": 8, "duracion_actividad_h": 2.0,
            "entrenado": False, "deporte": "pasear",
            "comorbilidades": ["respiratoria"], "farmacos": [],
            "situacion_social": [],
            "fototipo": "3", "falta_sueno": False, "enfermedad_reciente": False,
            "fiesta": False, "ocupacion": None,
        },
    },
    {
        "nombre": "helada_seca",
        "lat": 40.6564, "lon": -4.6993, "provincia": "Ávila",
        "t2m_c": -4.0, "rh": 45.0, "wind_speed_kmh": 28.0, "sp": 1021.0,
        "perfil": {
            "edad": 80, "sexo": "mujer", "porcentaje_grasa": 24.0,
            "aclimatado": True, "nivel_actividad": "reposo",
            "hora_inicio": 9, "duracion_actividad_h": 1.0,
            "entrenado": False, "deporte": None,
            "comorbilidades": [], "farmacos": [],
            "situacion_social": ["vive_solo", "no_sale"],
            "fototipo": "3", "falta_sueno": False, "enfermedad_reciente": False,
            "fiesta": False, "ocupacion": None,
        },
    },
]


def _hora(fecha: date, h: int, esc: dict) -> dict:
    """Una fila horaria del día `fecha` (misma fórmula que el test JS)."""
    t2m = esc["t2m_c"] + 4.0 * math.sin(2 * math.pi * (h - 8) / 24)
    rh = min(100.0, max(5.0, esc["rh"] + 8.0 * math.sin(2 * math.pi * (h - 2) / 24)))
    wind = max(0.0, esc["wind_speed_kmh"] + 2.0 * math.cos(2 * math.pi * h / 24))
    sp = esc["sp"] + 3.0 * math.sin(2 * math.pi * h / 24)
    return {
        "datetime": f"{fecha.isoformat()}T{h:02d}:00",
        "t2m_c": round(t2m, 6),
        "rh": round(rh, 6),
        "wind_speed_kmh": round(wind, 6),
        "sp": round(sp, 6),
    }


def main() -> int:
    out = {"fecha_generacion": TARGET_DATE.isoformat(), "escenarios": []}
    for esc in ESCENARIOS:
        horas = []
        for d in range(DIAS_HIST, -1, -1):
            fecha = TARGET_DATE - timedelta(days=d)
            for h in range(24):
                horas.append(_hora(fecha, h, esc))
        target_rows = [r for r in horas if r["datetime"].startswith(TARGET_DATE.isoformat())]
        midday = target_rows[len(target_rows) // 2]
        out["escenarios"].append({
            "nombre": esc["nombre"],
            "lat": esc["lat"], "lon": esc["lon"],
            "provincia": esc["provincia"],
            "target_date": TARGET_DATE.isoformat(),
            "current": {
                "t2m_c": midday["t2m_c"],
                "rh": midday["rh"],
                "wind_speed_kmh": midday["wind_speed_kmh"],
                "sp": midday["sp"],
            },
            "perfil": esc["perfil"],
            "horas": horas,
        })
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"scenarios.json escrito: {OUT} ({OUT.stat().st_size / 1024:.0f} KB, "
          f"{len(out['escenarios'])} escenarios × {len(horas)} horas)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
