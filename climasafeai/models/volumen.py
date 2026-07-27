"estimación epidemiológica macro para predicción por volumen"

PREVALENCIA_ECV_GENERAL = 0.04
PREVALENCIA_ECV_MAYORES_50 = 0.12

MULTIPLICADOR_EVENTO = {
    "general": 1.0,
    "concierto": 1.15,
    "festival": 1.25,
    "deporte": 1.35,
    "trabajo_exterior": 1.20,
}

INCERTIDUMBRE = 0.30


def _multiplicador_climatico(hi_peak: float) -> float:
    if hi_peak is None:
        return 1.0
    if hi_peak < 27:
        return 1.0
    if hi_peak < 32:
        return 1.0 + (hi_peak - 27) * 0.012
    if hi_peak < 39:
        return 1.06 + (hi_peak - 32) * 0.013
    if hi_peak < 45:
        return 1.15 + (hi_peak - 39) * 0.017
    return 1.25 + min(hi_peak - 45, 10) * 0.01


def _tasa_incidencia_directa(hi_peak: float, factor_evento: float) -> float:
    if hi_peak is None:
        return 0.0003
    if hi_peak < 27:
        return 0.0003
    if hi_peak < 32:
        return 0.0005
    if hi_peak < 39:
        return 0.0010
    if hi_peak < 45:
        return 0.0025
    return 0.0050


def _categoria_hi(hi_peak: float) -> str:
    if hi_peak is None:
        return "desconocido"
    if hi_peak < 27:
        return "NORMAL"
    if hi_peak < 32:
        return "PRECAUCION"
    if hi_peak < 39:
        return "PRECAUCION_EXTREMA"
    if hi_peak < 45:
        return "PELIGRO"
    return "PELIGRO_EXTREMO"


def estimar_afectados(
    total_personas: int,
    hi_peak: float | None = None,
    pct_mayores_50: float = 30.0,
    tipo_evento: str = "general",
) -> dict:
    if total_personas <= 0:
        return {"error": "total_personas debe ser > 0"}

    pct_mayores = max(0.0, min(100.0, pct_mayores_50)) / 100.0
    factor_evento = MULTIPLICADOR_EVENTO.get(tipo_evento, 1.0)

    prevalencia_ecv = (
        pct_mayores * PREVALENCIA_ECV_MAYORES_50
        + (1 - pct_mayores) * PREVALENCIA_ECV_GENERAL
    )

    mult = _multiplicador_climatico(hi_peak)
    tasa_incidencia_directa = _tasa_incidencia_directa(hi_peak, factor_evento)

    exceso_ecv = prevalencia_ecv * (mult - 1.0) if mult > 1.0 else 0.0
    exceso_directo = tasa_incidencia_directa * factor_evento
    tasa_total = exceso_ecv + exceso_directo

    estimacion = round(total_personas * tasa_total)
    rango_bajo = max(1, round(estimacion * (1 - INCERTIDUMBRE)))
    rango_alto = round(estimacion * (1 + INCERTIDUMBRE))

    categoria = _categoria_hi(hi_peak)

    return {
        "total_personas": total_personas,
        "estimacion_atencion_medica": estimacion,
        "rango_bajo": rango_bajo,
        "rango_alto": rango_alto,
        "pct_estimado": round(tasa_total * 100, 2),
        "clima": {
            "hi_peak": round(hi_peak, 1) if hi_peak is not None else None,
            "categoria": categoria,
        },
        "prevalencia_ecv_usada": round(prevalencia_ecv, 4),
        "multiplicador_climatico": round(mult, 3),
        "factor_evento": factor_evento,
        "tasa_incidencia_directa": round(tasa_incidencia_directa, 5),
        "exceso_ecv": round(exceso_ecv, 5),
        "exceso_directo": round(exceso_directo, 5),
        "mensaje": (
            f"De {total_personas} asistentes, ~{estimacion} "
            f"podrían requerir atención médica por calor"
        ),
        "mensaje_largo": (
            f"De {total_personas} asistentes, se estima que entre {rango_bajo} y "
            f"{rango_alto} podrían requerir atención médica por causas "
            f"relacionadas con el calor basado en la prevalencia de ECV "
            f"({prevalencia_ecv:.1%}) y las condiciones climáticas previstas "
            f"(HI pico {categoria} de {hi_peak:.0f}°C)"
        ) if hi_peak is not None else (
            f"De {total_personas} asistentes, se estima que entre {rango_bajo} y "
            f"{rango_alto} podrían requerir atención médica"
        ),
    }
