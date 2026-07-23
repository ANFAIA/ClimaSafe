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


def _ventana_actividad(perfil: dict) -> tuple | None:
    inicio = perfil.get("hora_inicio")
    duracion = perfil.get("duracion_actividad_h")
    if inicio is not None and duracion is not None:
        return (inicio, inicio + duracion)
    if inicio is not None:
        return (inicio, inicio + 1)
    return None


def _en_horas_centrales(ventana: tuple | None) -> bool:
    if ventana is None:
        return True
    fin = ventana[1]
    return fin > 12 and ventana[0] < 18


def _actividad_label(perfil: dict) -> str:
    dep = perfil.get("deporte")
    niv = perfil.get("nivel_actividad", "")
    if dep:
        return dep
    if niv:
        return f"actividad {niv}"
    return "actividad"


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

    ventana = _ventana_actividad(perfil)
    en_horas_centrales = _en_horas_centrales(ventana)

    clima_tags = _clasificar_clima(current, resultado, riesgo_dom)

    for tag in clima_tags:
        seccion = catalogo.get("clima", {}).get(tag)
        if seccion and "texto" in seccion:
            recomendaciones.append(seccion["texto"])

    fototipo = perfil.get("fototipo", "")
    if fototipo:
        seccion = catalogo.get("fototipo", {}).get(fototipo)
        if seccion and "texto" in seccion:
            texto = seccion["texto"]
            if not en_horas_centrales and ventana:
                texto = texto.replace("Busca sombra en horas centrales del dia.", f"Tu actividad es a partir de las {ventana[0]:.0f}:00, fuera del pico UV. Aun asi, proteccion solar recomendada.")
                texto = texto.replace("Evita la exposicion directa entre las 12:00 y las 18:00.", f"Tu actividad empieza a las {ventana[0]:.0f}:00, fuera del horario de maximo UV, pero lleva proteccion.")
            recomendaciones.append(texto)
    else:
        seccion = catalogo.get("fototipo", {}).get("desconocido")
        if seccion and "texto" in seccion:
            recomendaciones.append(seccion["texto"])

    label_act = _actividad_label(perfil)
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
    for key in ("hidratacion", "ropa", "comidas"):
        texto = generales.get(key)
        if texto:
            recomendaciones.append(texto)

    if ventana:
        inicio_label = f"{ventana[0]:.0f}:00"
        fin_label = f"{ventana[1]:.0f}:00"
        if en_horas_centrales:
            recomendaciones.append(
                f"Tu actividad ({inicio_label}-{fin_label}) coincide con las horas de mayor riesgo. "
                "Toma precauciones extra."
            )
        else:
            recomendaciones.append(
                f"Tu actividad es en horario seguro ({inicio_label}-{fin_label}), "
                "fuera del pico de calor (12:00-18:00)."
            )
    else:
        horas = generales.get("horas_peligro")
        if horas:
            recomendaciones.append(horas)

    if perfil.get("fiesta") and generales.get("hidratacion"):
        recomendaciones.append(
            "Has indicado que tienes planes de ocio/fiesta. Si consumes alcohol, "
            "hazlo con moderacion: el alcohol acelera la deshidratacion y altera "
            "la percepcion del riesgo termico. Alterna con agua."
        )

    if not perfil.get("aclimatado"):
        recomendaciones.append(
            "No estas aclimatado al clima local. Tu riesgo de golpe de calor o hipotermia "
            "es significativamente mayor. Limita la exposicion los primeros 3-5 dias y aumentala gradualmente."
        )

    if perfil.get("falta_sueno"):
        recomendaciones.append(
            "Has indicado falta de sueno o mala noche. La fatiga empeora la tolerancia "
            "al calor y la capacidad de tomar decisiones. Extremar precauciones."
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
            f"Tu actividad esta prevista para {perfil['duracion_actividad_h']:.0f} horas. "
            "Planifica pausas regulares y lleva suficiente agua (minimo 1 litro cada 2 horas)."
        )

    vistos = set()
    unicos = []
    for r in recomendaciones:
        if r not in vistos:
            vistos.add(r)
            unicos.append(r)

    return unicos
