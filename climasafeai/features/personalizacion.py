"""
climasafeai.features.personalizacion — modula el índice poblacional de riesgo
según el perfil individual.

El índice que dan la LSTM / los modelos ML es POBLACIONAL (0-1 por
provincia/día): no distingue individuos. En producción se personaliza con
factores multiplicativos documentados en literatura epidemiológica — no salen
del entrenamiento (MoMo no tiene atributos individuales), igual que el ×1.6 de
"sin aclimatación" (NIOSH) que el proyecto ya usa.

Fundamento, coeficientes y fuentes:
    documentacion/coeficientes_personalizacion_riesgo.md
    documentacion/papers/

Dos puntos de diseño clave:

  1. CALOR y FRÍO tienen tablas DISTINTAS. La obesidad es el ejemplo: en calor
     la grasa aísla y dificulta disipar (factor de riesgo débil, y solo en
     esfuerzo); en frío ese mismo aislamiento es neutro/protector. Por eso hay
     COEFS_CALOR y COEFS_FRIO separados y `tipo` decide cuál se aplica.

  2. Composición en ODDS, no multiplicación directa. El índice es una
     probabilidad; multiplicarla se sale de [0,1] (0.95×1.2=1.14). En odds el
     factor empuja donde hay margen y nunca se sale de escala. Estándar en
     epidemiología (los RR/OR publicados son razones de odds/hazard).
"""

from __future__ import annotations

import math

from climasafeai.db.manager import DBManager

_DB = DBManager()
_FACTORES_CACHE: dict | None = None


def _factores_implementados(tipo: str, categoria: str) -> dict:
    """Devuelve {clave: {coef, nombre, doi}} solo con implementado=true."""
    raw = _DB.obtener_factores(solo_implementados=True, tipo=tipo)
    items = raw.get(tipo, {}).get(categoria, [])
    return {i["clave"]: {"coef": i["coef"], "nombre": i["nombre"], "doi": i.get("doi")} for i in items}


# Cap del PRODUCTO de factores: los factores no son independientes (mayor +
# obeso + cardiópata solapan mecanismos), así que acumularlos sin límite
# genera riesgos individuales absurdos. Peca de precavido (lado correcto para
# un aviso), pero acotado.
CAP_FACTORES_DEFECTO: float = 3.0

_ACTIVIDADES_ESFUERZO = {"moderada", "intensa", "muy_intensa"}

# Porcentaje graso de referencia por edad y sexo (población española)
# Fuente: CUN-BAE (Clin Univ Navarra), ENPE, EXERNET multi-céntrico.
# Valores suavizados entre estudios; la grasa corporal aumenta hasta ~60 años
# y luego se estabiliza o reduce ligeramente.
_REF_GRASA_MUJER = [(18, 24.0), (30, 25.5), (40, 27.0), (50, 28.0), (60, 28.0), (70, 27.5), (80, 27.0), (100, 26.0)]
_REF_GRASA_HOMBRE = [(18, 16.0), (30, 18.5), (40, 20.5), (50, 22.5), (60, 23.5), (70, 24.0), (80, 24.0), (100, 23.5)]


def _grasa_referencia(edad: float, sexo: str = "hombre") -> float:
    tabla = _REF_GRASA_MUJER if sexo == "mujer" else _REF_GRASA_HOMBRE
    if edad <= tabla[0][0]:
        return tabla[0][1]
    if edad >= tabla[-1][0]:
        return tabla[-1][1]
    for i in range(len(tabla) - 1):
        x1, y1 = tabla[i]
        x2, y2 = tabla[i + 1]
        if x1 <= edad <= x2:
            t = (edad - x1) / (x2 - x1)
            return y1 + t * (y2 - y1)
    return tabla[-1][1]


def _factor_grasa_relativa(grasa: float, edad: float, sexo: str | None) -> float:
    """Devuelve factor 0.85–1.15 según desviación de la media poblacional por edad/sexo.

    El modelo MOMO se entrenó mayoritariamente con población >70a (grasa ~media
    de ese grupo). Para un joven, el factor de edad (×0.6 del ensemble) ya reduce
    el riesgo; este factor solo mide desviación intra-grupo de edad — no repite
    el ajuste entre grupos.
    """
    if sexo not in ("hombre", "mujer"):
        sexo = "mujer"
    ref = _grasa_referencia(edad, sexo)
    if ref <= 0:
        return 1.0
    ratio = grasa / ref
    return max(0.85, min(1.15, 1.0 + (ratio - 1.0) * 0.3))


def _factor_edad_calor(edad: int) -> float:
    if edad >= 85:
        return 2.0
    if edad >= 75:
        return 1.5
    if edad >= 65:
        return 1.2
    if edad >= 55:
        return 1.1
    if edad >= 45:
        return 1.05
    return 1.0


def _factor_edad_frio(edad: int) -> float:
    if edad >= 85:
        return 1.7
    if edad >= 75:
        return 1.4
    if edad >= 65:
        return 1.2
    return 1.0


_ACTIVIDAD_CALOR = {"reposo": 1.0, "ligera": 1.1, "moderada": 1.3, "intensa": 1.6, "muy_intensa": 2.0}
# En frío la actividad moderada GENERA calor (protectora); la intensa con
# sudor+viento empapa la ropa y acelera la pérdida (perjudicial).
_ACTIVIDAD_FRIO = {"reposo": 1.0, "ligera": 0.95, "moderada": 0.9, "intensa": 1.2, "muy_intensa": 1.2}

# Exposición laboral al calor — niveles progresivos
# Basado en: Annual Reviews 2024 (agriculture/construction highest HRI),
# CDC/NIOSH (delivery workers risk), Cal/OSHA HRI rates.
# Agricultura 35× más mortalidad que población general (Sokas 2023);
# Construcción 36% muertes laborales por calor con 6% fuerza laboral.
_OCUPACION_NIVELES: dict[str, tuple[float, str]] = {
    "oficina":      (1.00, "Trabajo sedentario (oficina, interior climatizado)"),
    "reparto":      (1.35, "Reparto / conducción (vehículo, carga ligera, descensos)"),
    "mantenimiento":(1.70, "Mantenimiento exterior / jardinería (continua, carga moderada)"),
    "construccion": (2.20, "Construcción / albañilería (carga pesada, PPE, sol directo)"),
    "campo":        (2.70, "Campo / agricultura (máxima exposición, pieza, sin sombra)"),
}

# Umbral de viento (km/h) a partir del cual el sudor en frío acelera la pérdida
# de calor (efecto wind-chill sobre ropa húmeda). Basado en NWS Wind Chill:
# a 0°C con viento de 20 km/h la sensación térmica equivale a -5°C.
_VIENTO_FRIO_UMBRAL = 10.0  # km/h, viento perceptible que empapa la ropa
_VIENTO_FRIO_MAX = 40.0     # km/h, saturación del factor


def _factor_viento_frio(viento_kmh: float | None, actividad: str) -> float:
    if viento_kmh is None:
        return 1.0
    if actividad not in ("intensa", "muy_intensa"):
        return 1.0
    if viento_kmh <= _VIENTO_FRIO_UMBRAL:
        return 1.0
    fraccion = min((viento_kmh - _VIENTO_FRIO_UMBRAL) / (_VIENTO_FRIO_MAX - _VIENTO_FRIO_UMBRAL), 1.0)
    return 1.0 + 0.5 * fraccion


def _factor_duracion_calor(horas: float) -> float:
    if horas > 4:
        return 1.4
    if horas > 2:
        return 1.25
    if horas > 1:
        return 1.1
    return 1.0


def _solapamiento_horas(inicio: float, duracion: float, ventana_inicio: float, ventana_fin: float) -> float:
    fin = inicio + duracion
    overlap_start = max(inicio, ventana_inicio)
    overlap_end = min(fin, ventana_fin)
    overlap_h = max(0.0, overlap_end - overlap_start)
    return overlap_h / duracion if duracion > 0 else 0.0


def _factor_hora_calor(hora_inicio: float | None, duracion: float | None) -> float:
    if hora_inicio is None or duracion is None:
        return 1.0
    overlap = _solapamiento_horas(hora_inicio, duracion, 12, 18)
    if overlap >= 0.75:
        return 1.3
    if overlap >= 0.5:
        return 1.2
    if overlap > 0.0:
        return 1.1
    return 1.0


def _factor_hora_frio(hora_inicio: float | None, duracion: float | None) -> float:
    if hora_inicio is None or duracion is None:
        return 1.0
    overlap = _solapamiento_horas(hora_inicio, duracion, 4, 8)
    if overlap >= 0.75:
        return 1.3
    if overlap >= 0.5:
        return 1.2
    if overlap > 0.0:
        return 1.1
    return 1.0


# Duración mínima de actividad en calor para que la fatiga acumulada
# sea relevante, según intensidad. A mayor intensidad, menos tiempo
# para que la producción metabólica de calor sature la termorregulación.
# Fuente: adaptado de umbrales NIOSH (ACGIH TLV para trabajo en calor).
_UMBRAL_DURACION_FATIGA = {
    "muy_intensa": 1,   # ej. esprint, fútbol competitivo
    "intensa": 2,       # ej. correr, ciclismo rápido
    "moderada": 3,      # ej. caminata rápida, bici suave
}

_FACTOR_FATIGA_BASE = {
    "muy_intensa": 1.35,
    "intensa": 1.2,
    "moderada": 1.1,
}


def _factor_fatiga_acumulada(perfil: dict) -> tuple[str, float] | None:
    hora_inicio = perfil.get("hora_inicio")
    duracion = perfil.get("duracion_actividad_h")
    actividad = perfil.get("nivel_actividad", "reposo")
    perfil_horario = perfil.get("_perfil_horario")
    if any(v is None for v in (hora_inicio, duracion, perfil_horario)):
        return None
    if actividad not in _ACTIVIDADES_ESFUERZO:
        return None
    umbral = _UMBRAL_DURACION_FATIGA.get(actividad, 4)
    if duracion < umbral:
        return None
    fin = hora_inicio + duracion
    window = [h for h in perfil_horario if hora_inicio <= h["hora"] < fin]
    if not window:
        return None
    peak = max(window, key=lambda x: x["HI"])
    peak_HI = peak["HI"]
    if peak_HI < 27:
        return None
    horas_hasta_pico = peak["hora"] - hora_inicio
    factor_base = _FACTOR_FATIGA_BASE.get(actividad, 1.1)
    # Bonus si el pico de calor cae avanzada la actividad
    fraccion_pico = horas_hasta_pico / duracion if duracion > 0 else 0
    bonus = 0.05 * max(0, fraccion_pico - 0.3)
    factor = min(factor_base + bonus, 1.5)
    label = (
        f"fatiga acumulada ({horas_hasta_pico:.0f}/{duracion:.0f}h de "
        f"{actividad}, HI pico {peak_HI:.1f}C)"
    )
    return (label, factor)


def _factores_calor(perfil: dict) -> list[tuple[str, str, float]]:
    """Devuelve [(nombre, categoria, valor), ...] de los factores activos."""
    f: list[tuple[str, str, float]] = []

    fisio_cal = _factores_implementados("calor", "fisiologico")
    if (sx := perfil.get("sexo")) in ("hombre", "mujer"):
        v = 0.96 if sx == "hombre" else 1.04
        f.append((f"sexo {sx}", "fisiologico", v))

    if (edad := perfil.get("edad")) is not None:
        v = _factor_edad_calor(edad)
        if v != 1.0:
            f.append((f"edad {int(edad)} años", "fisiologico", v))

    actividad = perfil.get("nivel_actividad", "reposo")
    v_act = _ACTIVIDAD_CALOR.get(actividad, 1.0)
    if v_act != 1.0:
        if perfil.get("entrenado") and actividad in _ACTIVIDADES_ESFUERZO:
            v_act = 1.0 + (v_act - 1.0) * 0.5
        label = f"actividad {actividad}"
        if perfil.get("entrenado") and actividad in _ACTIVIDADES_ESFUERZO:
            label += " (entrenado)"
        if dep := perfil.get("deporte"):
            label = f"{dep} ({label})"
        f.append((label, "fisiologico", round(v_act, 3)))

    if (horas := perfil.get("duracion_actividad_h")) is not None:
        v = _factor_duracion_calor(horas)
        if v != 1.0:
            f.append((f"duración {horas} h", "fisiologico", v))

    v = _factor_hora_calor(perfil.get("hora_inicio"), horas)
    if v != 1.0 and horas is not None:
        inicio = perfil["hora_inicio"]
        f.append((f"hora inicio {inicio:.0f}:00 (solapa pico calor)", "fisiologico", v))

    grasa = perfil.get("porcentaje_grasa")
    sexo = perfil.get("sexo")
    edad = perfil.get("edad")
    if grasa is not None and edad is not None:
        v = _factor_grasa_relativa(grasa, edad, sexo)
        if v != 1.0:
            ref = _grasa_referencia(edad, sexo or "mujer")
            direccion = "superior" if grasa > ref else "inferior"
            f.append((f"grasa corporal {grasa}% (ref. {ref:.0f}% para {int(edad)}a/{sexo or '?'}: {direccion})", "fisiologico", round(v, 3)))

    comorb_cal = _factores_implementados("calor", "comorbilidades")
    farmacos_cal = _factores_implementados("calor", "farmacos")
    social_cal = _factores_implementados("calor", "situacional")

    if perfil.get("aclimatado") is False:
        ac = fisio_cal.get("no_aclimatado")
        if ac:
            f.append(("no aclimatado", "fisiologico", ac["coef"]))

    if perfil.get("falta_sueno"):
        fs = fisio_cal.get("falta_sueno")
        if fs:
            f.append((fs["nombre"], "fisiologico", fs["coef"]))

    if perfil.get("enfermedad_reciente"):
        er = fisio_cal.get("enfermedad_reciente")
        if er:
            f.append((er["nombre"], "fisiologico", er["coef"]))

    comorb = perfil.get("comorbilidades", set())
    for k in comorb:
        if k in comorb_cal and k != "mental":
            f.append((comorb_cal[k]["nombre"], "medico", comorb_cal[k]["coef"]))

    # Salud mental y psicofármacos: un ÚNICO factor (el riesgo lo marca la
    # condición, no el fármaco aislado — ver wong-2024). Evita doble conteo.
    farmacos = perfil.get("farmacos", set())
    if "mental" in comorb or "antipsicoticos" in farmacos:
        ap = comorb_cal.get("mental") or farmacos_cal.get("antipsicoticos")
        if ap:
            f.append((ap["nombre"], "medico", ap["coef"]))
    for k in farmacos:
        if k in farmacos_cal and k != "antipsicoticos":
            f.append((farmacos_cal[k]["nombre"], "medico", farmacos_cal[k]["coef"]))

    fatiga = _factor_fatiga_acumulada(perfil)
    if fatiga:
        f.append((fatiga[0], "fisiologico", fatiga[1]))

    sociales = set(perfil.get("situacion_social", []))
    if perfil.get("fiesta"):
        f.append(("fiesta / consumo de alcohol reciente", "situacional", 1.8))
    elif "fiesta" in sociales:
        f.append(("fiesta / consumo de alcohol reciente", "situacional", 1.8))
    presentes = [(k, social_cal[k]) for k in sociales if k in social_cal and k != "fiesta"]
    if presentes:
        nombre, mejor = max(presentes, key=lambda kv: kv[1]["coef"])
        f.append((f"aislamiento/dependencia ({mejor['nombre']})", "situacional", mejor["coef"]))

    # Factor UV según fototipo (solo si hay índice UV disponible)
    uv_index = perfil.get("_uv_index")
    fototipo = perfil.get("fototipo")
    if uv_index is not None and fototipo is not None and uv_index > 3:
        foto_map = {"1": 1.3, "2": 1.2, "3": 1.1, "4": 1.0, "5": 0.9, "6": 0.85}
        uv_factor = foto_map.get(str(fototipo), 1.0)
        intensidad = uv_index / 11.0
        factor = 1.0 + (uv_factor - 1.0) * intensidad
        if factor > 1.0:
            f.append((f"UV {uv_index:.1f} + fototipo {fototipo}", "fisiologico", round(factor, 2)))

    # Exposición laboral al calor
    if (ocp := perfil.get("ocupacion")) and ocp in _OCUPACION_NIVELES:
        coef, label = _OCUPACION_NIVELES[ocp]
        if coef != 1.0:
            f.append((f"trabajo {label}", "ocupacional", coef))

    return f


def _factores_frio(perfil: dict) -> list[tuple[str, str, float]]:
    f: list[tuple[str, str, float]] = []

    fisio_frio = _factores_implementados("frio", "fisiologico")
    if (sx := perfil.get("sexo")) in ("hombre", "mujer"):
        v = 1.15 if sx == "hombre" else 0.87
        f.append((f"sexo {sx}", "fisiologico", v))

    if (edad := perfil.get("edad")) is not None:
        v = _factor_edad_frio(edad)
        if v != 1.0:
            f.append((f"edad {int(edad)} años", "fisiologico", v))

    actividad = perfil.get("nivel_actividad", "reposo")
    v_act = _ACTIVIDAD_FRIO.get(actividad, 1.0)
    v_viento = _factor_viento_frio(perfil.get("_wind_speed_kmh"), actividad)
    v = v_act * v_viento
    if v != 1.0:
        if perfil.get("entrenado") and actividad in _ACTIVIDADES_ESFUERZO:
            v_act = 1.0 + (v_act - 1.0) * 0.5
            v = v_act * v_viento
        label = f"actividad {actividad}"
        if perfil.get("entrenado") and actividad in _ACTIVIDADES_ESFUERZO:
            label += " (entrenado)"
        if dep := perfil.get("deporte"):
            label = f"{dep} ({label})"
        if v_viento > 1.0:
            label += f" + viento {perfil.get('_wind_speed_kmh', '?')} km/h"
        f.append((label, "fisiologico", round(v, 2)))

    v = _factor_hora_frio(perfil.get("hora_inicio"), perfil.get("duracion_actividad_h"))
    if v != 1.0:
        inicio = perfil["hora_inicio"]
        f.append((f"hora inicio {inicio:.0f}:00 (solapa amanecer)", "fisiologico", v))

    grasa = perfil.get("porcentaje_grasa")
    sexo = perfil.get("sexo")
    edad = perfil.get("edad")
    if grasa is not None and edad is not None:
        v = _factor_grasa_relativa(grasa, edad, sexo)
        if v != 1.0:
            ref = _grasa_referencia(edad, sexo or "mujer")
            direccion = "superior" if grasa > ref else "inferior"
            f.append((f"grasa corporal {grasa}% (ref. {ref:.0f}% para {int(edad)}a/{sexo or '?'}: {direccion})", "fisiologico", round(v, 3)))

    comorb_frio = _factores_implementados("frio", "comorbilidades")
    comorb = perfil.get("comorbilidades", set())
    for k in comorb:
        if k in comorb_frio:
            f.append((comorb_frio[k]["nombre"], "medico", comorb_frio[k]["coef"]))

    social_frio = _factores_implementados("frio", "situacional")
    social = set(perfil.get("situacion_social", []))
    if perfil.get("fiesta"):
        f.append(("fiesta / consumo de alcohol reciente", "situacional", 1.8))
    elif "fiesta" in social:
        f.append(("fiesta / consumo de alcohol reciente", "situacional", 1.8))
    presentes = [(k, social_frio[k]) for k in social if k in social_frio and k != "fiesta"]
    if presentes:
        nombre, mejor = max(presentes, key=lambda kv: kv[1]["coef"])
        f.append((f"aislamiento/vivienda fría ({mejor['nombre']})", "situacional", mejor["coef"]))

    return f


def personalizar_riesgo(
    indice: float,
    perfil: dict,
    tipo: str = "calor",
    cap_factores: float = CAP_FACTORES_DEFECTO,
) -> dict:
    """
    Modula el índice poblacional 0-1 con los factores del perfil individual.

    Parameters
    ----------
    indice : float
        Índice de peligrosidad POBLACIONAL 0-1 (p. ej. `indice_riesgo_softmax`
        de la LSTM, o cualquier probabilidad de riesgo de los modelos ML).
    perfil : dict
        Campos opcionales (ausente = neutro, no penaliza por falta de dato):
        edad:int, sexo:"hombre"|"mujer", porcentaje_grasa:float,
        nivel_actividad:"reposo"|"ligera"|"moderada"|"intensa"|"muy_intensa",
        duracion_actividad_h:float, hora_inicio:float (0-23), aclimatado:bool,
        comorbilidades:set{"cardiovascular","diabetes","respiratoria","mental"},
        farmacos:set{"antipsicoticos","diureticos_asa"},
        situacion_social:set{"vive_solo","encamado","no_sale","vivienda_fria"},
        alcohol_reciente:bool (añade "alcohol" a situacion_social),
        entrenado:bool (reduce a la mitad el factor de actividad si ≥moderada),
        ocupacional:set{"estres_termico_laboral","esfuerzo_termico_laboral"},
        _perfil_horario:list[{"hora":int, "HI":float}] (opcional, usado por _factor_fatiga_acumulada).
    tipo : "calor" | "frio"
        Qué tabla de coeficientes aplicar. Son DISTINTAS (la obesidad y la
        actividad se comportan al revés en frío).
    cap_factores : float
        Límite del producto de factores (los factores no son independientes).

    Returns
    -------
    dict con:
        indice_original, factor_total (ya capado), indice_personalizado,
        capado (bool), factores (lista de {nombre, categoria, factor}).
        El desglose es intencionado: un sistema de salud necesita explicar
        "índice 0.8 × 1.5 por edad × 1.4 por cardiopatía", no un número opaco.
    """
    if not 0.0 <= indice <= 1.0:
        raise ValueError(f"indice debe estar en [0, 1], no {indice}")
    if tipo not in ("calor", "frio"):
        raise ValueError(f"tipo debe ser 'calor' o 'frio', no {tipo!r}")

    perfil = dict(perfil)
    for alias_key, canonical_key in [("grasa_corporal", "porcentaje_grasa"), ("grasa", "porcentaje_grasa")]:
        if alias_key in perfil and canonical_key not in perfil:
            perfil[canonical_key] = perfil.pop(alias_key)

    factores = _factores_calor(perfil) if tipo == "calor" else _factores_frio(perfil)

    producto = 1.0
    for _, _, valor in factores:
        producto *= valor
    factor_total = min(producto, cap_factores)
    capado = producto > cap_factores

    # Composición en ODDS: odds' = odds * factor -> prob'
    if indice in (0.0, 1.0):
        # Odds 0 o infinito: el factor no puede moverlo, se queda igual.
        personalizado = indice
    else:
        odds = indice / (1.0 - indice)
        odds_ind = odds * factor_total
        personalizado = odds_ind / (1.0 + odds_ind)

    return {
        "indice_original": indice,
        "factor_total": round(factor_total, 3),
        "producto_bruto": round(producto, 3),
        "capado": capado,
        "indice_personalizado": round(personalizado, 4),
        "factores": [
            {"nombre": n, "categoria": c, "factor": v} for n, c, v in factores
        ],
    }


def _sigmoid_hi(hi: float) -> float:
    """Probabilidad base de riesgo por calor a partir del Heat Index (0-1).

    Sigmoide suave centrada en ~32°C (misma forma que grid_risk._hi_a_probabilidad).
    """
    if hi <= 20:
        return 0.05
    if hi >= 50:
        return 0.95
    return 1.0 / (1.0 + math.exp(-(hi - 32) / 6.0))


def _factor_suave(producto_bruto: float, techo: float = 8.0, escala: float = 5.0) -> float:
    """Compresión monótona y saturante del producto de factores.

    El cap DURO de ``personalizar_riesgo`` (3.0 por defecto) es correcto para la
    clasificación puntual, pero colapsa al mismo valor perfiles muy distintos
    (p. ej. un albañil de 57 con cardiopatía y otro de 32 sano, ambos con el
    factor ocupación ×2.2, saturan el cap). Para la CURVA horaria usamos en su
    lugar una saturación suave: a mayor riesgo, mayor factor, sin meseta, así
    cada perfil mantiene su propia línea. Los factores protectores (<1) se
    respetan tal cual.
    """
    if producto_bruto <= 1.0:
        return producto_bruto
    return 1.0 + (techo - 1.0) * (1.0 - math.exp(-(producto_bruto - 1.0) / escala))


def riesgo_horario_acumulado(
    perfil_horario: list[dict],
    perfil: dict,
    umbral: float = 28.0,
    decay: float = 0.85,
    k: float = 0.045,
) -> list[dict]:
    """Curva de riesgo por hora (0-1) para un perfil concreto.

    Combina dos cosas y por eso NO es una recta:
      1. Riesgo instantáneo de cada hora: el Heat Index de esa hora, llevado a
         probabilidad y personalizado con los factores del perfil (edad, sexo,
         salud, ocupación, aclimatación…). Por eso sube y baja con el calor.
      2. Carga térmica ACUMULADA: el calor es acumulativo. Se acumula el exceso
         de HI sobre `umbral` con una fugacidad `decay` (el cuerpo no se recupera
         del todo si no refresca), así que a igual HI la tarde pesa más que la
         mañana. El acumulado añade riesgo en espacio de odds (× (1 + k·carga)).

    Devuelve ``[{"hora": h, "riesgo": r, "hi": HI}, ...]`` ordenado por hora.
    """
    if not perfil_horario:
        return []

    # Factor de personalización (constante en el día) calculado una sola vez.
    # Usamos el producto BRUTO con saturación suave (no el cap duro), para que
    # perfiles distintos den curvas distintas y no una recta compartida.
    perfil_sin_horario = {kk: vv for kk, vv in perfil.items() if kk != "_perfil_horario"}
    factor = _factor_suave(personalizar_riesgo(0.5, perfil_sin_horario, tipo="calor")["producto_bruto"])

    horas = sorted(
        (h for h in perfil_horario if h.get("HI") is not None),
        key=lambda e: e["hora"],
    )
    carga = 0.0
    out = []
    for e in horas:
        hi = float(e["HI"])
        carga = max(0.0, decay * carga + (hi - umbral))
        base = _sigmoid_hi(hi)
        # Personalización + carga acumulada, ambas en espacio de odds.
        if 0.0 < base < 1.0:
            odds = (base / (1.0 - base)) * factor * (1.0 + k * carga)
            r = odds / (1.0 + odds)
        else:
            r = base
        out.append({
            "hora": int(e["hora"]),
            "riesgo": round(min(r, 0.99), 4),
            "hi": round(hi, 1),
            "temp": e.get("temp"),
        })
    return out


def pico_riesgo_actividad(curva: list[dict], perfil: dict) -> float | None:
    """Riesgo máximo de la curva durante la ventana de actividad del perfil.

    Es el número que representa mejor a una persona (su peor momento durante la
    jornada) y, al venir de la curva suavizada, diferencia perfiles distintos.
    """
    if not curva:
        return None
    h_ini = perfil.get("hora_inicio")
    dur = perfil.get("duracion_actividad_h")
    if h_ini is not None and dur:
        win = [c["riesgo"] for c in curva if h_ini <= c["hora"] < h_ini + dur]
        if win:
            return round(max(win), 4)
    return round(max(c["riesgo"] for c in curva), 4)


def recomendar_horario(
    perfil_horario: list[dict],
    perfil: dict,
    duracion_h: float | None = None,
) -> dict | None:
    """Recomienda la ventana de `duracion_h` horas con MENOR riesgo acumulado.

    Usa :func:`riesgo_horario_acumulado` y busca la ventana contigua de menor
    riesgo medio, dando prioridad a horas de luz razonables (6-21h). Devuelve
    ``{"hora_inicio", "hora_fin", "riesgo_medio", "riesgo_actual"}`` o None.
    """
    curva = riesgo_horario_acumulado(perfil_horario, perfil)
    if not curva:
        return None
    dur = int(round(duracion_h or perfil.get("duracion_actividad_h") or 2))
    dur = max(1, dur)
    por_hora = {c["hora"]: c["riesgo"] for c in curva}
    horas_dia = [h for h in range(6, 22) if h in por_hora]
    if len(horas_dia) < dur:
        horas_dia = sorted(por_hora)
    mejor = None
    for i in range(0, len(horas_dia) - dur + 1):
        tramo = horas_dia[i:i + dur]
        if tramo[-1] - tramo[0] != dur - 1:
            continue  # ventana no contigua
        medio = sum(por_hora[h] for h in tramo) / dur
        if mejor is None or medio < mejor["riesgo_medio"]:
            mejor = {"hora_inicio": tramo[0], "hora_fin": tramo[0] + dur, "riesgo_medio": round(medio, 4)}
    if mejor is None:
        return None
    h_ini_actual = perfil.get("hora_inicio")
    if h_ini_actual is not None:
        actual = [por_hora[h] for h in range(int(h_ini_actual), int(h_ini_actual) + dur) if h in por_hora]
        mejor["riesgo_actual"] = round(sum(actual) / len(actual), 4) if actual else None
    return mejor
