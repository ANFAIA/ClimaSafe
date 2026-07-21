"""
Análisis manual de casos sospechosos: muestra input completo + output + debug.
Busca contextos donde el resultado no tenga sentido.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from climasafeai.models.ensemble import predict_ensemble, fetch_weather_data

_WEATHER_CACHE: dict = {}
def _cached_fetch(**kw):
    key = (kw.get("lat"), kw.get("lon"), kw.get("provincia", ""))
    if key not in _WEATHER_CACHE:
        _WEATHER_CACHE[key] = fetch_weather_data(**kw)
    return _WEATHER_CACHE[key]

import climasafeai.models.ensemble as _emod
_emod.fetch_weather_data = _cached_fetch

# Casos diseñados para ser sospechosos
SUSPECT_CASES = [
    # (nombre, lat, lon, provincia, perfil, razon_por_que_es_sospechoso)
    {
        "name": "Sevilla 41C, perfil vacio",
        "lat": 37.38, "lon": -5.99, "prov": "Sevilla",
        "perfil": {},
        "sospecha": "HI pico 41C → debería ser PELIGRO aunque perfil vacio",
    },
    {
        "name": "Sevilla 41C, joven sano noche",
        "lat": 37.38, "lon": -5.99, "prov": "Sevilla",
        "perfil": {"edad": 25, "hora_inicio": 22, "duracion_actividad_h": 2, "nivel_actividad": "reposo"},
        "sospecha": "Actividad nocturna, HI ventana deberia ser < 27 → SEGURO o PRECAUCION, no PELIGRO",
    },
    {
        "name": "Sevilla 41C, trabajador dia completo",
        "lat": 37.38, "lon": -5.99, "prov": "Sevilla",
        "perfil": {
            "edad": 45, "sexo": "hombre", "nivel_actividad": "moderada",
            "hora_inicio": 8, "duracion_actividad_h": 10,
            "aclimatado": False,
            "situacion_ocupacional": ["esfuerzo_termico_laboral"],
        },
        "sospecha": "Trabajador 10h en Sevilla con HI 41C → PELIGRO con fatiga acumulada",
    },
    {
        "name": "A Coruña frio, perfil vacio",
        "lat": 43.36, "lon": -8.41, "prov": "A Coruña",
        "perfil": {},
        "sospecha": "Clima templado, HI<27, sin perfil → SEGURO",
    },
    {
        "name": "A Coruña, anciano vulnerable",
        "lat": 43.36, "lon": -8.41, "prov": "A Coruña",
        "perfil": {
            "edad": 80, "sexo": "mujer", "aclimatado": False,
            "comorbilidades": ["cardiovascular", "diabetes", "respiratoria"],
            "situacion_social": ["vive_solo", "no_sale"],
            "enfermedad_reciente": True,
        },
        "sospecha": "Clima templado pero perfil muy vulnerable → ¿debería subir a PRECAUCION?",
    },
    {
        "name": "Madrid HI 34C, correr 2h al mediodia",
        "lat": 40.41, "lon": -3.70, "prov": "Madrid",
        "perfil": {
            "edad": 35, "sexo": "hombre", "nivel_actividad": "intensa",
            "hora_inicio": 13, "duracion_actividad_h": 2,
            "aclimatado": False, "porcentaje_grasa": 25,
        },
        "sospecha": "Correr 2h al mediodia en HI 34C → deberia ser PRECAUCION+PELIGRO con fatiga",
    },
    {
        "name": "Sevilla 41C, correr al atardecer",
        "lat": 37.38, "lon": -5.99, "prov": "Sevilla",
        "perfil": {
            "edad": 30, "sexo": "mujer", "nivel_actividad": "intensa",
            "hora_inicio": 20, "duracion_actividad_h": 2,
            "aclimatado": True,
        },
        "sospecha": "Correr 20-22h en Sevilla, HI bajando pero aun 35C+ → PRECAUCION con fatiga?",
    },
    {
        "name": "Badajoz 39C, esprint 1h",
        "lat": 38.87, "lon": -6.97, "prov": "Badajoz",
        "perfil": {
            "edad": 28, "sexo": "hombre", "nivel_actividad": "muy_intensa",
            "hora_inicio": 15, "duracion_actividad_h": 1,
            "aclimatado": False,
        },
        "sospecha": "Esprint 1h a las 15h en HI 39C → PELIGRO con fatiga acumulada (umbral 1h para muy_intensa)",
    },
    {
        "name": "Barcelona humedo, paseo 3h",
        "lat": 41.38, "lon": 2.17, "prov": "Barcelona",
        "perfil": {
            "edad": 60, "sexo": "hombre", "nivel_actividad": "moderada",
            "hora_inicio": 11, "duracion_actividad_h": 3,
            "aclimatado": True, "porcentaje_grasa": 30,
            "comorbilidades": ["cardiovascular"],
        },
        "sospecha": "Barcelona humedo 3h paseo → PRECAUCION?",
    },
    {
        "name": "TODOS los factores juntos",
        "lat": 37.38, "lon": -5.99, "prov": "Sevilla",
        "perfil": {
            "edad": 85, "sexo": "mujer", "nivel_actividad": "muy_intensa",
            "hora_inicio": 14, "duracion_actividad_h": 4,
            "porcentaje_grasa": 40, "aclimatado": False, "fototipo": 1,
            "comorbilidades": ["cardiovascular", "diabetes", "respiratoria", "mental"],
            "farmacos": ["antipsicoticos", "diureticos_asa"],
            "situacion_social": ["vive_solo", "no_sale", "sin_aire_acondicionado", "encamado"],
            "situacion_ocupacional": ["esfuerzo_termico_laboral"],
            "enfermedad_reciente": True, "falta_sueno": True,
        },
        "sospecha": "Todos los factores + Sevilla 41C → PELIGRO con multiplicadores",
    },
    {
        "name": "León frio, perfil frio extremo",
        "lat": 42.60, "lon": -5.57, "prov": "León",
        "perfil": {
            "edad": 80, "sexo": "mujer", "aclimatado": False,
            "porcentaje_grasa": 10,
            "comorbilidades": ["cardiovascular", "respiratoria"],
            "situacion_social": ["vive_solo", "vivienda_fria"],
            "enfermedad_reciente": True,
        },
        "sospecha": "Leon frio + perfil vulnerable al frio → deberia subir riesgo frio",
    },
    {
        "name": "Perfil minimo: solo edad",
        "lat": 40.41, "lon": -3.70, "prov": "Madrid",
        "perfil": {"edad": 18},
        "sospecha": "Solo edad joven → joven tiene factor 0.6x, deberia bajar riesgo",
    },
    {
        "name": "Actividad 3am-5am en Sevilla",
        "lat": 37.38, "lon": -5.99, "prov": "Sevilla",
        "perfil": {
            "edad": 40, "nivel_actividad": "moderada",
            "hora_inicio": 3, "duracion_actividad_h": 2,
        },
        "sospecha": "Actividad 3-5am, HI minimo (~25C) → SEGURO, el override NO deberia forzar PRECAUCION",
    },
]


def analizar_caso(c):
    print(f"\n{'='*70}")
    print(f"CASO: {c['name']}")
    print(f"SOSPECHA: {c['sospecha']}")
    print(f"UBICACION: {c['prov']} ({c['lat']}, {c['lon']})")
    print(f"PERFIL: {json.dumps(c['perfil'], default=str)}")
    print(f"{'='*70}")

    r = predict_ensemble(lat=c["lat"], lon=c["lon"], provincia=c["prov"], perfil=c["perfil"])

    weather = r["weather"]
    current = weather.get("current", {})
    perfil_horario = weather.get("perfil_horario", [])

    # 1. Weather
    print(f"\n[WEATHER]")
    print(f"  Temp actual: {current.get('t2m_c', '?')}C")
    print(f"  HI actual: {current.get('heat_index_c', '?')}")
    print(f"  HI pico diario: {max((h['HI'] for h in perfil_horario), default='?'):.1f}C" if perfil_horario else "  HI pico: N/A")
    print(f"  HI min diario: {min((h['HI'] for h in perfil_horario), default='?'):.1f}C" if perfil_horario else "")
    print(f"  WC: {current.get('wind_chill_c', '?')}")

    # 2. HI ventana actividad
    perfil_input = c["perfil"]
    h_ini = perfil_input.get("hora_inicio")
    dur = perfil_input.get("duracion_actividad_h")
    if h_ini is not None and dur is not None and perfil_horario:
        fin = h_ini + dur
        ventana = [h for h in perfil_horario if h_ini <= h["hora"] < fin]
        if ventana:
            pico_v = max(h["HI"] for h in ventana)
            min_v = min(h["HI"] for h in ventana)
            print(f"\n[VENTANA ACTIVIDAD] {h_ini:.0f}:00-{fin:.0f}:00 ({dur}h)")
            print(f"  HI en ventana: {min_v:.1f}C - {pico_v:.1f}C")
            actividad = perfil_input.get("nivel_actividad", "reposo")
            print(f"  Actividad: {actividad}")
            print(f"  Umbral fatiga: ", end="")
            umbrales = {"muy_intensa": 1, "intensa": 2, "moderada": 3}
            if actividad in umbrales:
                print(f"{umbrales[actividad]}h (dur={dur}h → {'ACTIVA' if dur >= umbrales[actividad] else 'NO alcanza'})")
            else:
                print("N/A (solo esfuerzo)")

    # 3. Modelos
    print(f"\n[MODELOS]")
    for name, m in r["modelos"].items():
        if "error" in m:
            print(f"  {name}: ERROR {m['error']}")
            continue
        if "prob_riesgo" in m:
            print(f"  {name}: prob_riesgo={m['prob_riesgo']:.4f}, clase_threshold={m.get('clase_threshold','')}")
        if "calor" in m:
            h = m["calor"]
            print(f"  {name} calor: clase={h.get('clase','')}, thr={h.get('clase_threshold','')}, HI={h.get('heat_index_c','')}")
        if "frio" in m:
            f = m["frio"]
            print(f"  {name} frio: clase={f.get('clase','')}, thr={f.get('clase_threshold','')}, WC={f.get('wind_chill_c','')}")

    # 4. Personalizacion
    perfil_apl = r.get("perfil", {})
    print(f"\n[PERSONALIZACION]")
    for tipo in ("calor", "frio"):
        p = perfil_apl.get(tipo, {})
        print(f"  {tipo}: prob_pob={p.get('prob_poblacional',0):.4f} × factor_total={p.get('factor_total',1):.3f} "
              f"(cap={p.get('producto_bruto',1):.3f}) → prob_pers={p.get('prob_personalizada',0):.4f}")
        if p.get("factores"):
            for f in p["factores"]:
                print(f"    - {f['nombre']} ({f['categoria']}) ×{f['factor']}")

    # 5. Resultado
    clase = r["clase_final_label"]
    modelo_det = r["explicacion"]["modelo_determinante"]
    print(f"\n[RESULTADO]")
    print(f"  Clase final: {clase}")
    print(f"  Modelo determinante: {modelo_det}")
    if r["explicacion"].get("override"):
        print(f"  Override: {r['explicacion']['override']['razon']}")
    print(f"  Prob calor personalizada: {perfil_apl.get('calor',{}).get('prob_personalizada',0):.4f}")
    print(f"  Prob frio personalizada: {perfil_apl.get('frio',{}).get('prob_personalizada',0):.4f}")

    # 6. Juicio: ¿el resultado es sospechoso?
    print(f"\n[JUICIO]")
    problemas = []
    perfil_input = c["perfil"]
    h_ini = perfil_input.get("hora_inicio")
    dur = perfil_input.get("duracion_actividad_h")
    perfil_horario = weather.get("perfil_horario", [])
    if h_ini is not None and dur is not None and perfil_horario:
        fin = h_ini + dur
        ventana = [h for h in perfil_horario if h_ini <= h["hora"] < fin]
        if ventana:
            pico_v = max(h["HI"] for h in ventana)
            hi_current = current.get("heat_index_c")
            # Override usa HI_peak de ventana o HI_current?
            if r["explicacion"].get("override"):
                razon = r["explicacion"]["override"]["razon"]
                # Extraer HI_peak usado
                import re
                m = re.search(r'HI_peak=([\d.]+)C', razon)
                if m:
                    hi_usado = float(m.group(1))
                    # Diferencia con pico ventana
                    if abs(hi_usado - pico_v) > 1.0 and abs(hi_usado - (hi_current or 0)) > 1.0:
                        problemas.append(f"HI_peak en razón ({hi_usado:.1f}C) no coincide ni con ventana ({pico_v:.1f}C) ni con actual ({hi_current})")

        # Verificar fatiga acumulada
        actividad = perfil_input.get("nivel_actividad", "reposo")
        umbrales = {"muy_intensa": 1, "intensa": 2, "moderada": 3}
        if actividad in umbrales and pico_v >= 27 and dur >= umbrales[actividad]:
            tiene_fatiga = any(
                "fatiga" in str(f.get("nombre", "")).lower()
                for f in perfil_apl.get("calor", {}).get("factores", [])
            )
            if not tiene_fatiga:
                problemas.append(f"Fatiga NO activada: {actividad} {dur}h, HI pico {pico_v:.1f}C, umbral {umbrales[actividad]}h")

    if problemas:
        print(f"  ⚠️  PROBLEMAS DETECTADOS:")
        for p in problemas:
            print(f"    ❌ {p}")
    else:
        print(f"  ✓ Sin problemas aparentes")

    return r


if __name__ == "__main__":
    for c in SUSPECT_CASES:
        try:
            analizar_caso(c)
        except Exception as e:
            print(f"\n❌ ERROR en {c['name']}: {e}")
