import json
from pathlib import Path


_RECOMENDACIONES_PATH = Path(__file__).resolve().parent.parent / "data" / "recomendaciones.json"


def _cargar_catalogo() -> dict:
    if _RECOMENDACIONES_PATH.exists():
        with open(_RECOMENDACIONES_PATH) as f:
            return json.load(f)
    return {}


def _riesgo_dominante(resultado: dict) -> str:
    modelos = resultado.get("modelos", {})
    clase_final = resultado.get("clase_final", 0)
    heat_clases = []
    cold_clases = []
    for nombre, res in modelos.items():
        if isinstance(res, dict) and "error" in res:
            continue
        if nombre == "LSTM":
            c = res.get("calor", {}).get("clase_threshold", 0)
            if c:
                heat_clases.append(c)
            c = res.get("frio", {}).get("clase_threshold", 0)
            if c:
                cold_clases.append(c)
        elif nombre == "Formula":
            c = res.get("calor", {}).get("clase", 0)
            if c:
                heat_clases.append(c)
            c = res.get("frio", {}).get("clase", 0)
            if c:
                cold_clases.append(c)
        elif "calor" in nombre.lower():
            c = res.get("clase_threshold", 0)
            if c:
                heat_clases.append(c)
        elif "frio" in nombre.lower():
            c = res.get("clase_threshold", 0)
            if c:
                cold_clases.append(c)
    max_heat = max(heat_clases) if heat_clases else 0
    max_cold = max(cold_clases) if cold_clases else 0
    if max_heat > max_cold:
        return "calor"
    if max_cold > max_heat:
        return "frio"
    return "ambos"


def _clasificar_clima(current: dict, resultado: dict, riesgo_dominante: str = "ambos") -> list[str]:
    etiquetas = []
    t = current.get("t2m_c")
    wc = resultado.get("modelos", {}).get("Formula", {}).get("frio", {}).get("wind_chill_c")
    hi = resultado.get("modelos", {}).get("Formula", {}).get("calor", {}).get("heat_index_c")
    uv = current.get("uv_index")

    if riesgo_dominante != "frio":
        if t is not None and t >= 35:
            etiquetas.append("calor_extremo")
        elif t is not None and t >= 30:
            etiquetas.append("calor_moderado")

    if riesgo_dominante != "calor":
        if wc is not None and wc <= -25:
            etiquetas.append("frio_extremo")
        elif wc is not None and wc <= 0:
            etiquetas.append("frio_moderado")

    if uv is not None and uv >= 8:
        etiquetas.append("uv_alto")
    elif uv is not None and uv >= 6:
        etiquetas.append("uv_alto")

    return etiquetas


def _nivel_actividad_segura(clase_final: int) -> str:
    if clase_final >= 2:
        return "reposo"
    if clase_final == 1:
        return "ligera"
    return ""


def generar_recomendaciones(perfil: dict, resultado: dict) -> list[str]:
    catalogo = _cargar_catalogo()
    if not catalogo:
        return ["No hay catalogo de recomendaciones disponible."]
    if not perfil:
        return []

    current = resultado.get("weather", {}).get("current", {})
    clase_final = resultado.get("clase_final", 0)
    riesgo_dom = _riesgo_dominante(resultado)
    recomendaciones = []

    clima_tags = _clasificar_clima(current, resultado, riesgo_dom)

    for tag in clima_tags:
        seccion = catalogo.get("clima", {}).get(tag)
        if seccion and "texto" in seccion:
            recomendaciones.append(seccion["texto"])

    fototipo = perfil.get("fototipo", "")
    if fototipo:
        seccion = catalogo.get("fototipo", {}).get(fototipo)
        if seccion and "texto" in seccion:
            recomendaciones.append(seccion["texto"])
    else:
        seccion = catalogo.get("fototipo", {}).get("desconocido")
        if seccion and "texto" in seccion:
            recomendaciones.append(seccion["texto"])

    actividad = perfil.get("nivel_actividad", "").lower()
    nivel_seguro = _nivel_actividad_segura(clase_final)
    if nivel_seguro == "reposo":
        recomendaciones.append("El nivel de riesgo es PELIGRO. No se recomienda realizar actividad fisica al aire libre. Busca un lugar fresco y permanece en reposo.")
    elif actividad:
        seccion = catalogo.get("actividad", {}).get(actividad)
        if seccion and "texto" in seccion:
            rec = seccion["texto"]
            if nivel_seguro == "ligera" and actividad in ("moderada", "intensa", "muy_intensa"):
                rec += " Dado el nivel de riesgo actual, considera reducir la intensidad de tu actividad."
            recomendaciones.append(rec)

    comorbilidades = perfil.get("comorbilidades", set())
    for comorb in comorbilidades:
        seccion = catalogo.get("comorbilidades", {}).get(comorb.lower())
        if seccion and "texto" in seccion:
            recomendaciones.append(seccion["texto"])

    farmacos = perfil.get("farmacos", set())
    for farmaco in farmacos:
        seccion = catalogo.get("farmacos", {}).get(farmaco.lower())
        if seccion and "texto" in seccion:
            recomendaciones.append(seccion["texto"])

    situacion = perfil.get("situacion_social", set())
    for sit in situacion:
        seccion = catalogo.get("situacion_social", {}).get(sit.lower())
        if seccion and "texto" in seccion:
            recomendaciones.append(seccion["texto"])

    generales = catalogo.get("generales", {})
    for key in ("hidratacion", "ropa", "horas_peligro", "comidas"):
        texto = generales.get(key)
        if texto:
            recomendaciones.append(texto)

    if perfil.get("aclimatado") is False:
        recomendaciones.append(
            "No estas aclimatado al clima local. Tu riesgo de golpe de calor o hipotermia "
            "es significativamente mayor. Limita la exposicion los primeros 3-5 dias y aumentala gradualmente."
        )

    if clase_final >= 1:
        if riesgo_dom != "frio":
            senal = generales.get("senal_alarma_calor")
            if senal:
                recomendaciones.append(senal)
        if riesgo_dom != "calor":
            senal = generales.get("senal_alarma_frio")
            if senal:
                recomendaciones.append(senal)

    if perfil.get("duracion_actividad_h") is not None and perfil.get("duracion_actividad_h", 0) > 2:
        recomendaciones.append(
            f"Tu actividad esta prevista para {perfil['duracion_actividad_h']} horas. "
            "Planifica pausas regulares y lleva suficiente agua (minimo 1 litro cada 2 horas)."
        )

    vistos = set()
    unicos = []
    for r in recomendaciones:
        if r not in vistos:
            vistos.add(r)
            unicos.append(r)

    return unicos
