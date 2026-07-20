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

import json
import warnings
from pathlib import Path

# ── Factores desde JSON (data/factores_riesgo.json) ──────────────────────
# Los coeficientes simples (comorbilidades, fármacos, situación social) se
# leen de este archivo. Los factores complejos (edad, actividad, duración,
# hora, fatiga acumulada) siguen siendo funciones de código.
# El scout añade factores nuevos con implementado=false para revisión manual.

_FACTORES_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "factores_riesgo.json"
_FACTORES_CACHE: dict | None = None


def _cargar_factores() -> dict:
    global _FACTORES_CACHE
    if _FACTORES_CACHE is not None:
        return _FACTORES_CACHE
    with open(_FACTORES_PATH) as f:
        _FACTORES_CACHE = json.load(f)

    # Warning si hay factores pendientes de implementar
    pendientes = 0
    for tipo in ("calor", "frio"):
        for categoria, factores in _FACTORES_CACHE.get(tipo, {}).items():
            if not isinstance(factores, dict):
                continue
            for clave, info in factores.items():
                if isinstance(info, dict) and not info.get("implementado"):
                    pendientes += 1
    if pendientes:
        warnings.warn(
            f"⚠️  {pendientes} factor(es) de riesgo pendiente(s) de revisión. "
            f"Ejecuta: uv run python -m agents scout --review",
            stacklevel=2,
        )

    return _FACTORES_CACHE


def _factores_implementados(tipo: str, categoria: str) -> dict:
    """Devuelve {clave: {coef, nombre, doi}} solo con implementado=true."""
    data = _cargar_factores()
    factores = data.get(tipo, {}).get(categoria, {})
    return {k: v for k, v in factores.items() if v.get("implementado")}


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


def _factores_calor(perfil: dict) -> list[tuple[str, str, float]]:
    """Devuelve [(nombre, categoria, valor), ...] de los factores activos."""
    f: list[tuple[str, str, float]] = []

    if (edad := perfil.get("edad")) is not None:
        v = _factor_edad_calor(edad)
        if v != 1.0:
            f.append((f"edad {edad}", "fisiologico", v))

    fisio_cal = _factores_implementados("calor", "fisiologico")
    if perfil.get("sexo") == "mujer":
        sm = fisio_cal.get("sexo_mujer")
        if sm:
            f.append((sm["nombre"], "fisiologico", sm["coef"]))

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
            gae = fisio_cal.get("grasa_alta_esfuerzo")
            if gae:
                f.append((f"{gae['nombre']} ({grasa}%)", "fisiologico", gae["coef"]))

    comorb_cal = _factores_implementados("calor", "comorbilidades")
    farmacos_cal = _factores_implementados("calor", "farmacos")
    social_cal = _factores_implementados("calor", "situacional")

    if perfil.get("aclimatado") is False:
        ac = fisio_cal.get("no_aclimatado")
        if ac:
            f.append(("no aclimatado", "fisiologico", ac["coef"]))

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

    social = perfil.get("situacion_social", set())
    presentes = [(k, social_cal[k]) for k in social if k in social_cal]
    if presentes:
        nombre, mejor = max(presentes, key=lambda kv: kv[1]["coef"])
        f.append((f"aislamiento/dependencia ({mejor['nombre']})", "situacional", mejor["coef"]))

    return f


def _factores_frio(perfil: dict) -> list[tuple[str, str, float]]:
    f: list[tuple[str, str, float]] = []

    if (edad := perfil.get("edad")) is not None:
        v = _factor_edad_frio(edad)
        if v != 1.0:
            f.append((f"edad {edad}", "fisiologico", v))

    fisio_frio = _factores_implementados("frio", "fisiologico")
    if perfil.get("sexo") == "mujer":
        sm = fisio_frio.get("sexo_mujer")
        if sm:
            f.append((sm["nombre"], "fisiologico", sm["coef"]))

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
            gb = fisio_frio.get("grasa_baja")
            if gb:
                f.append((f"{gb['nombre']} ({grasa}%)", "fisiologico", gb["coef"]))
        elif grasa >= umbral_alto:
            gp = fisio_frio.get("grasa_alta_protector")
            if gp:
                f.append((f"{gp['nombre']} ({grasa}%)", "fisiologico", gp["coef"]))

    comorb_frio = _factores_implementados("frio", "comorbilidades")
    comorb = perfil.get("comorbilidades", set())
    for k in comorb:
        if k in comorb_frio:
            f.append((comorb_frio[k]["nombre"], "medico", comorb_frio[k]["coef"]))

    social_frio = _factores_implementados("frio", "situacional")
    social = perfil.get("situacion_social", set())
    presentes = [(k, social_frio[k]) for k in social if k in social_frio]
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
