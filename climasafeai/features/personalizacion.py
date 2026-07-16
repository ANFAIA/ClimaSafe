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

# ---------------------------------------------------------------------------
# Coeficientes (ver documentacion/coeficientes_personalizacion_riesgo.md)
# Cada factor: categoría para el desglose transparente
#   fisiologico | medico | situacional
# ---------------------------------------------------------------------------

# Cap del PRODUCTO de factores: los factores no son independientes (mayor +
# obeso + cardiópata solapan mecanismos), así que acumularlos sin límite
# genera riesgos individuales absurdos. Peca de precavido (lado correcto para
# un aviso), pero acotado.
CAP_FACTORES_DEFECTO: float = 3.0

_ACTIVIDADES_ESFUERZO = {"moderada", "intensa", "muy_intensa"}


def _factor_edad_calor(edad: int) -> float:
    if edad >= 85:
        return 2.0
    if edad >= 75:
        return 1.5
    if edad >= 65:
        return 1.2
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


def _factor_fatiga_acumulada(perfil: dict) -> tuple[str, float] | None:
    hora_inicio = perfil.get("hora_inicio")
    duracion = perfil.get("duracion_actividad_h")
    actividad = perfil.get("nivel_actividad", "reposo")
    perfil_horario = perfil.get("_perfil_horario")
    if any(v is None for v in (hora_inicio, duracion, perfil_horario)):
        return None
    if actividad not in _ACTIVIDADES_ESFUERZO:
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
    if horas_hasta_pico < 4:
        return None
    label = f"fatiga acumulada ({horas_hasta_pico:.0f}h trabajo al llegar a {peak_HI:.1f}C)"
    factor = 1.3 if horas_hasta_pico >= 6 else 1.2
    return (label, factor)


# Social: valores TEMPLADOS respecto a los OR univariantes de Chicago (2.3-6.7),
# que se solapan entre sí — se toma el MÁXIMO de los presentes, no el producto,
# para no contar la misma vulnerabilidad varias veces.
_SOCIAL_CALOR = {"encamado": 2.0, "no_sale": 2.0, "vive_solo": 1.5, "vivienda_fria": 1.0}
_SOCIAL_FRIO = {"vivienda_fria": 1.5, "encamado": 1.5, "vive_solo": 1.3, "no_sale": 1.3}


def _factores_calor(perfil: dict) -> list[tuple[str, str, float]]:
    """Devuelve [(nombre, categoria, valor), ...] de los factores activos."""
    f: list[tuple[str, str, float]] = []

    if (edad := perfil.get("edad")) is not None:
        v = _factor_edad_calor(edad)
        if v != 1.0:
            f.append((f"edad {edad}", "fisiologico", v))

    if perfil.get("sexo") == "mujer":
        f.append(("sexo mujer", "fisiologico", 1.1))

    actividad = perfil.get("nivel_actividad", "reposo")
    if (v := _ACTIVIDAD_CALOR.get(actividad, 1.0)) != 1.0:
        f.append((f"actividad {actividad}", "fisiologico", v))

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
    if grasa is not None:
        umbral_alto = 25 if sexo == "hombre" else 32
        if grasa >= umbral_alto and actividad in _ACTIVIDADES_ESFUERZO:
            f.append((f"grasa corporal alta ({grasa}%, en esfuerzo)", "fisiologico", 1.2))

    if perfil.get("aclimatado") is False:
        f.append(("no aclimatado", "fisiologico", 1.6))

    comorb = perfil.get("comorbilidades", set())
    if "cardiovascular" in comorb:
        f.append(("cardiopatía/HTA", "medico", 1.4))
    if "diabetes" in comorb:
        f.append(("diabetes", "medico", 1.2))

    # Salud mental y psicofármacos: un ÚNICO factor (el riesgo lo marca la
    # condición, no el fármaco aislado — ver wong-2024). Evita doble conteo.
    farmacos = perfil.get("farmacos", set())
    if "mental" in comorb or "antipsicoticos" in farmacos:
        f.append(("salud mental / antipsicóticos", "medico", 1.8))
    if "diureticos_asa" in farmacos:
        f.append(("diuréticos de asa", "medico", 1.3))

    fatiga = _factor_fatiga_acumulada(perfil)
    if fatiga:
        f.append((fatiga[0], "fisiologico", fatiga[1]))

    social = perfil.get("situacion_social", set())
    presentes = [(k, _SOCIAL_CALOR[k]) for k in social if _SOCIAL_CALOR.get(k, 1.0) > 1.0]
    if presentes:
        nombre, valor = max(presentes, key=lambda kv: kv[1])
        f.append((f"aislamiento/dependencia ({nombre})", "situacional", valor))

    return f


def _factores_frio(perfil: dict) -> list[tuple[str, str, float]]:
    f: list[tuple[str, str, float]] = []

    if (edad := perfil.get("edad")) is not None:
        v = _factor_edad_frio(edad)
        if v != 1.0:
            f.append((f"edad {edad}", "fisiologico", v))

    if perfil.get("sexo") == "mujer":
        f.append(("sexo mujer", "fisiologico", 1.05))

    actividad = perfil.get("nivel_actividad", "reposo")
    if (v := _ACTIVIDAD_FRIO.get(actividad, 1.0)) != 1.0:
        f.append((f"actividad {actividad}", "fisiologico", v))

    v = _factor_hora_frio(perfil.get("hora_inicio"), perfil.get("duracion_actividad_h"))
    if v != 1.0:
        inicio = perfil["hora_inicio"]
        f.append((f"hora inicio {inicio:.0f}:00 (solapa amanecer)", "fisiologico", v))

    grasa = perfil.get("porcentaje_grasa")
    sexo = perfil.get("sexo")
    if grasa is not None:
        umbral_bajo = 12 if sexo == "hombre" else 20
        umbral_alto = 25 if sexo == "hombre" else 32
        if grasa < umbral_bajo:
            f.append((f"grasa corporal baja ({grasa}%)", "fisiologico", 1.3))
        elif grasa >= umbral_alto:
            f.append((f"grasa corporal alta ({grasa}%, protector)", "fisiologico", 0.9))

    comorb = perfil.get("comorbilidades", set())
    if "cardiovascular" in comorb:
        f.append(("cardiopatía/HTA", "medico", 1.5))
    if "respiratoria" in comorb:
        f.append(("enf. respiratoria", "medico", 1.4))

    social = perfil.get("situacion_social", set())
    presentes = [(k, _SOCIAL_FRIO[k]) for k in social if _SOCIAL_FRIO.get(k, 1.0) > 1.0]
    if presentes:
        nombre, valor = max(presentes, key=lambda kv: kv[1])
        f.append((f"aislamiento/vivienda fría ({nombre})", "situacional", valor))

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
