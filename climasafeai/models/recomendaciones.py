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


# P(riesgo) por debajo de la cual un canal (calor/frío) se considera
# irrelevante para la recomendación y no aporta consejos (BOT-011). Es más
# laxo que el t1 de clase (0.25): por debajo de 15% el canal no manda.
UMBRAL_CANAL_IRRELEVANTE = 0.15


def _canal_dominante(resultado: dict) -> str | None:
    """Canal que manda en la recomendación según la probabilidad personalizada.

    Devuelve 'calor', 'frio', 'ninguno' (ambos por debajo del umbral) o None
    si el resultado no trae la información de canales (dicts mínimos de test) —
    en ese caso quien llama degrada a la lógica por clima físico.
    """
    perfil = resultado.get("perfil") or {}
    prob_calor = (perfil.get("calor") or {}).get("prob_personalizada")
    prob_frio = (perfil.get("frio") or {}).get("prob_personalizada")
    if prob_calor is None or prob_frio is None:
        return None
    activo_calor = prob_calor >= UMBRAL_CANAL_IRRELEVANTE
    activo_frio = prob_frio >= UMBRAL_CANAL_IRRELEVANTE
    if not activo_calor and not activo_frio:
        return "ninguno"
    if activo_calor and not activo_frio:
        return "calor"
    if activo_frio and not activo_calor:
        return "frio"
    return "calor" if prob_calor >= prob_frio else "frio"


def _recomendacion_uv(uv, texto_canal: str | None = None) -> list[str]:
    """Partes de la recomendación según el índice UV; texto_canal se añade al final."""
    partes = ["Mantente hidratado"]
    if uv is not None and uv >= 8:
        partes.append("utiliza protector solar SPF 50+ (renueva cada 2 horas)")
    elif uv is not None and uv >= 6:
        partes.append("utiliza protector solar SPF 30+")
    elif uv is not None and uv >= 3:
        partes.append("lleva protección solar básica")
    if texto_canal:
        partes.append(texto_canal)
    return partes


def recomendacion_resumen(resultado: dict) -> str:
    """Recomendación de una línea adaptada al canal dominante (frío/calor/UV).

    El canal que manda es el de mayor probabilidad personalizada (BOT-011); si
    un canal queda por debajo de `UMBRAL_CANAL_IRRELEVANTE` no aporta consejos,
    así que un día de calor no recomienda abrigo ni un día de frío recomienda
    evitar las horas de más calor. La protección solar solo aparece cuando el
    índice UV lo justifica. Es el texto que cierra el parte final del bot.
    """
    w = resultado.get("weather", {})
    cur = w.get("current", {})
    t = cur.get("t2m_c")
    uv = w.get("uv_index")
    if uv is None:
        uv = cur.get("uv_index")
    formula = (resultado.get("modelos") or {}).get("Formula") or {}
    wc = formula.get("frio", {}).get("wind_chill_c")
    hi = formula.get("calor", {}).get("heat_index_c")

    # Riesgo PELIGRO: lo primero es no hacer la actividad
    if resultado.get("clase_final", 0) >= 2:
        return ("Riesgo alto: evita la actividad física al aire libre y "
                "permanece en un lugar fresco. Mantente hidratado.")

    canal = _canal_dominante(resultado)
    if canal is None:
        # Sin probabilidades de canal: se decide por el clima físico (dicts
        # mínimos / consumidores que no pasan `perfil`).
        frio = (wc is not None and wc <= 0) or (t is not None and t <= 10)
        calor = (hi is not None and hi >= 32) or (t is not None and t >= 30)
        if frio:
            canal = "frio"
        elif calor:
            canal = "calor"
        else:
            canal = "ninguno"

    if canal == "frio":
        frase = ("Mantente hidratado y abrígate con varias capas; "
                 "protege las extremidades del viento.")
        if uv is not None and uv >= 6:
            frase += " El índice UV sigue alto: usa protección solar."
        return frase

    if canal == "calor":
        partes = _recomendacion_uv(uv, "evita la exposición prolongada entre las horas de mayor calor")
        return ", ".join(partes[:-1]) + f" y {partes[-1]}."

    # canal == "ninguno": clima suave, sin consejos de canal
    partes = _recomendacion_uv(uv, "busca sombra si el sol aprieta")
    return ", ".join(partes[:-1]) + f" y {partes[-1]}."
