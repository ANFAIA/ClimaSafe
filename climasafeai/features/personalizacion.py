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

    # Obesidad: SOLO cuenta en esfuerzo (el OR alto es de estudios de esfuerzo;
    # en reposo/población general no hay señal — ver nota en la doc).
    imc = perfil.get("imc")
    if imc is not None:
        if imc >= 30 and actividad in _ACTIVIDADES_ESFUERZO:
            f.append(("obesidad (IMC≥30, en esfuerzo)", "fisiologico", 1.2))
        elif imc < 18.5:
            f.append(("fragilidad (IMC<18.5)", "fisiologico", 1.3))

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

    imc = perfil.get("imc")
    if imc is not None and imc < 18.5:
        f.append(("fragilidad (IMC<18.5)", "fisiologico", 1.3))
    # IMC≥30 en frío: neutro (grasa protectora) -> no se añade factor.

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
        edad:int, sexo:"hombre"|"mujer", imc:float,
        nivel_actividad:"reposo"|"ligera"|"moderada"|"intensa"|"muy_intensa",
        duracion_actividad_h:float, aclimatado:bool,
        comorbilidades:set{"cardiovascular","diabetes","respiratoria","mental"},
        farmacos:set{"antipsicoticos","diureticos_asa"},
        situacion_social:set{"vive_solo","encamado","no_sale","vivienda_fria"}.
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
        "indice_personalizado": round(personalizado, 4),
        "capado": capado,
        "factores": [
            {"nombre": n, "categoria": c, "factor": v} for n, c, v in factores
        ],
    }
