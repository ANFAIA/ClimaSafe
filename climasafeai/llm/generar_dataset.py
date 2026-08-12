#! /usr/bin/env python
"""
Genera dataset sintético para fine‑tuning de Qwen 2.5 ClimaSafeAI.

Crea pares (instrucción + perfil → respuesta ideal) en formato Alpaca JSONL,
usando el pipeline real de predicción para que las respuestas sean factuales.

Uso:
    python climasafeai/llm/generar_dataset.py \
        --output data/llm/train.jsonl \
        --num-ejemplos 150 \
        --val-split 0.1
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

import pandas as pd
from dotenv import load_dotenv


# ---------------------------------------------------------------------------
# Perfiles sintéticos (combinaciones sistemáticas)
# ---------------------------------------------------------------------------

EDADES = [25, 45, 65, 75, 85]
SEXOS = ["hombre", "mujer"]
GRASA = [None, 15, 25, 35]
ACLIMATADO = [True, False]
FOTOTIPO = ["II", "III", "IV"]
SITUACION_SOCIAL = [
    "",
    "vive_solo",
    "vive_solo,sin_aire_acondicionado",
    "no_sale",
]
# Las claves tienen que ser las que el modelo reconoce, no sinónimos en castellano.
# `_factores_implementados("calor", ...)` solo puntúa estas: cardiovascular x1.4
# (incluye HTA), diabetes x1.2, mental x1.8 y respiratoria x1.3. Poner "cardiopatia"
# o "hipertension" hacía que el ejemplo dijera que el usuario es cardiópata y que
# la respuesta no aplicara ningún factor por ello — enseñándole al modelo que da
# igual. La obesidad no va aquí: entra por `porcentaje_grasa`.
COMORBILIDADES = [
    "",
    "diabetes",
    "cardiovascular",
    "respiratoria",
    "mental",
    "cardiovascular,diabetes",
    "diabetes,respiratoria",
    "cardiovascular,mental",
]
# Igual con los fármacos: solo hay coeficiente para estos dos.
MEDICACION = [
    "",
    "diureticos_asa",
    "antipsicoticos",
    "antipsicoticos,diureticos_asa",
]
ACTIVIDADES = ["reposo", "ligera", "moderada", "intensa"]
DURACIONES = [0.5, 1.0, 2.0, 4.0, 6.0]
HORAS = [8, 10, 12, 14, 16, 18]

# Escenarios climáticos
# Los siete escenarios originales eran de calor peninsular en julio, y de ahí salía
# un dataset con 85 PELIGRO frente a 15 SEGURO. Se añaden sitios frescos (norte
# atlántico, montaña, Canarias) para que haya ejemplos de riesgo bajo y de frío.
ESCENARIOS = [
    # (lat, lon, provincia, descripción)
    (37.38, -5.99, "Sevilla", "calor extremo"),
    (41.65, -0.88, "Zaragoza", "calor extremo seco"),
    (37.18, -3.60, "Granada", "calor seco"),
    (40.41, -3.70, "Madrid", "calor moderado"),
    (39.47, -0.38, "Valencia", "calor humedo"),
    (43.26, -2.93, "Bilbao", "templado atlantico"),
    (43.36, -8.41, "Coruna", "templado humedo"),
    (42.29, -8.81, "Pontevedra", "templado atlantico"),
    (43.54, -5.66, "Asturias", "fresco costero"),
    (42.60, -5.57, "Leon", "frio de meseta"),
    (42.88, -2.68, "Vitoria", "fresco continental"),
    (42.51, 1.53, "Lleida", "montaña pirenaica"),
    (28.46, -16.25, "Tenerife", "subtropical estable"),
]

RANDOM_SEED = 42


# ---------------------------------------------------------------------------
# Generación de perfiles
# ---------------------------------------------------------------------------


def _variar_con_step(valor, step, p=0.3):
    """Con probabilidad p, varía valor ±step."""
    if random.random() < p:
        return valor + random.choice([-step, step])
    return valor


def generar_perfiles(num: int) -> list[dict]:
    """Genera N perfiles combinando sistemáticamente los parámetros."""
    random.seed(RANDOM_SEED)
    perfiles = []

    # Asegurar cobertura de todas las combinaciones principales
    combinaciones_base = []
    for edad in EDADES:
        for sexo in SEXOS:
            for aclim in ACLIMATADO:
                combinaciones_base.append({
                    "edad": edad,
                    "sexo": sexo,
                    "aclimatado": aclim,
                })

    # Repetir hasta tener suficientes
    while len(perfiles) < num:
        for base in combinaciones_base:
            if len(perfiles) >= num:
                break
            perfil = dict(base)
            # Un tercio de los perfiles va "limpio": sin patologías, sin fármacos y
            # sin nada de la noche anterior. Si todos van cargados, el producto de
            # factores choca con CAP_FACTORES_DEFECTO (x3.0) y el dataset le enseña
            # al modelo que perfiles muy distintos dan exactamente el mismo tope.
            suave = random.random() < 0.35

            perfil["grasa"] = random.choice(GRASA)
            perfil["fototipo"] = random.choice(FOTOTIPO)
            perfil["situacion_social"] = "" if suave else random.choice(SITUACION_SOCIAL)
            perfil["comorbilidades"] = "" if suave else random.choice(COMORBILIDADES)
            perfil["medicacion"] = "" if suave else random.choice(MEDICACION)
            if suave:
                # Un perfil suave tiene que serlo también aquí: la edad, la falta de
                # aclimatación, la intensidad y las horas seguidas ya se comen el tope
                # de x3.0 por sí solas, sin necesidad de ninguna patología.
                perfil["edad"] = random.choice([25, 45])
                perfil["aclimatado"] = True
                perfil["nivel_actividad"] = random.choice(["reposo", "ligera"])
                perfil["duracion_h"] = random.choice([0.5, 1.0, 2.0])
                perfil["hora_inicio"] = random.choice([8, 10, 18])
            else:
                perfil["nivel_actividad"] = random.choice(ACTIVIDADES)
                perfil["duracion_h"] = random.choice(DURACIONES)
                perfil["hora_inicio"] = random.choice(HORAS)
            perfil["entrenado"] = random.choice([True, False]) if random.random() < 0.4 else None
            # Cómo llega a la salida: fiesta x1.8, enfermedad x1.3, mala noche x1.2.
            perfil["fiesta"] = (not suave) and random.random() < 0.25
            perfil["falta_sueno"] = (not suave) and random.random() < 0.30
            perfil["enfermedad_reciente"] = (not suave) and random.random() < 0.20
            perfil["ocupacion"] = random.choice(["", "reparto", "construccion", "campo"]) if random.random() < 0.3 else None

            # Escenario climático
            esc = random.choice(ESCENARIOS)
            perfil["lat"] = _variar_con_step(esc[0], 0.05)
            perfil["lon"] = _variar_con_step(esc[1], 0.05)
            perfil["provincia"] = esc[2]
            perfil["_escenario"] = esc[3]

            perfiles.append(perfil)

    return perfiles[:num]


# ---------------------------------------------------------------------------
# Predicción (usa el pipeline real)
# ---------------------------------------------------------------------------


# Misma fuente (lat, lon) → mismo parte dentro de la generación. Sin esta caché,
# los 400 perfiles repetirían ~117 descargas de Open-Meteo+OpenUV. Mismo patrón
# que tests/manual_analysis.py. OpenUV tiene cupo diario (50 req) y sin caché el
# UV quedaría en None para casi todo el dataset.
_WEATHER_CACHE: dict[tuple[float, float], dict] = {}

# El UV horario NO puede ir en df_hora: las features alimentan modelos entrenados
# sin esa columna y rompen (XGBoost/RandomForest: "feature names unseen at fit").
# Se pide a Open-Meteo aparte, misma fuente que el resto del pipeline, sin cupo.
_UV_CACHE: dict[tuple[float, float], pd.DataFrame | None] = {}


def _uv_horario(lat: float, lon: float) -> pd.DataFrame | None:
    """UV horario de Open-Meteo para las próximas 48 h (fuente real, sin cupo)."""
    key = (lat, lon)
    if key not in _UV_CACHE:
        _UV_CACHE[key] = None
        try:
            import requests

            from climasafeai.data.weather_fetcher import OPENMETEO_BASE

            data = requests.get(
                OPENMETEO_BASE,
                params={
                    "latitude": lat,
                    "longitude": lon,
                    "hourly": "uv_index",
                    "timezone": "auto",
                    "forecast_hours": 48,
                },
                timeout=30,
            ).json()
            hourly = data.get("hourly", {})
            if hourly.get("time") and hourly.get("uv_index"):
                _UV_CACHE[key] = pd.DataFrame({
                    "datetime": pd.to_datetime(hourly["time"]),
                    "uv_index": hourly["uv_index"],
                })
        except Exception:
            _UV_CACHE[key] = None
    return _UV_CACHE[key]


def _perfil_para_modelo(perfil: dict) -> dict:
    """Traduce el perfil del generador a las claves que LEE el modelo.

    Ojo con los nombres: `farmacos` y `porcentaje_grasa`. Escribir `medicacion` o
    `grasa_corporal` no da error — el factor se salta en silencio y el riesgo sale
    por debajo del que toca. Ya pasó en el MCP y en el bot.
    """
    def _conjunto(valor):
        if not valor:
            return set()
        if isinstance(valor, (set, list, tuple)):
            return set(valor)
        return {x.strip() for x in str(valor).split(",") if x.strip()}

    p: dict[str, Any] = {
        "edad": perfil["edad"],
        "sexo": perfil["sexo"],
        "aclimatado": perfil["aclimatado"],
        "nivel_actividad": perfil["nivel_actividad"],
        "hora_inicio": perfil["hora_inicio"],
        "duracion_actividad_h": perfil["duracion_h"],
        "comorbilidades": _conjunto(perfil.get("comorbilidades")),
        "farmacos": _conjunto(perfil.get("medicacion")),
    }
    if perfil.get("grasa") is not None:
        p["porcentaje_grasa"] = perfil["grasa"]
    if perfil.get("fototipo"):
        p["fototipo"] = perfil["fototipo"]
    if perfil.get("situacion_social"):
        p["situacion_social"] = _conjunto(perfil["situacion_social"])
    if perfil.get("entrenado") is not None:
        p["entrenado"] = perfil["entrenado"]
    if perfil.get("ocupacion"):
        p["ocupacion"] = perfil["ocupacion"]
    # Cómo llega a la salida: fiesta x1.8, enfermedad reciente x1.3, mala noche x1.2
    for clave in ("fiesta", "falta_sueno", "enfermedad_reciente"):
        if perfil.get(clave):
            p[clave] = True
    return p


def predecir(perfil: dict, weather: dict | None = None) -> dict:
    """Ejecuta la predicción REAL de ClimaSafeAI. Si no puede, revienta.

    Antes esto tenía un `except ImportError` que caía en `_predecir_fake`, y como
    los tres imports que hacía no existían en ese módulo, el dataset ENTERO salía
    de la simulación: un riesgo que solo dependía de la edad. Sin error, sin aviso,
    y con pinta de bueno — 150 ejemplos para enseñarle al modelo una función de
    riesgo inventada. Por eso ahora no hay red: si el pipeline no va, se para la
    generación y se dice por qué.

    El UV se mete en el perfil ANTES de predecir (clave `_uv_index`, la que lee
    personalizar_riesgo): si no, el pipeline solo aplica el factor UV cuando
    OpenUV tiene cupo/caché, y 384 de 400 ejemplos saldrían con un input que
    dice UV pero un output que no lo usó — el mismo fallo que esta feature
    corrige. El parte del input se calcula del mismo `weather` que ve el pipeline.

    `weather` opcional: si se pasa (p. ej. construido desde el parquet de frío por
    `generar_dataset_frio._construir_weather`), se usa tal cual — sin cache ni
    descarga — y el ejemplo sale del pipeline sobre ESE weather. Sin él, se
    descarga con fetch_weather_data (comportamiento histórico del generador de
    calor).
    """
    from climasafeai.data.weather_fetcher import fetch_weather_data
    from climasafeai.models.ensemble import predict_ensemble

    inicio = int(perfil.get("hora_inicio") or 0)
    fin = inicio + max(1, round(float(perfil.get("duracion_h") or 1)))

    key = (perfil["lat"], perfil["lon"])
    if weather is None:
        if key not in _WEATHER_CACHE:
            _WEATHER_CACHE[key] = fetch_weather_data(
                lat=perfil["lat"], lon=perfil["lon"], provincia=perfil["provincia"],
            )
            _WEATHER_CACHE[key]["uv_horario"] = _uv_horario(*key)
        weather = _WEATHER_CACHE[key]

    perfil_modelo = _perfil_para_modelo(perfil)
    uv = _uv_de_la_ventana(weather, inicio, fin)
    if uv is not None:
        perfil_modelo["_uv_index"] = uv

    resultado = predict_ensemble(
        lat=perfil["lat"],
        lon=perfil["lon"],
        provincia=perfil["provincia"],
        perfil=perfil_modelo,
        weather=weather,
    )
    # `predict_ensemble` reconstruye su propio dict de weather (solo coge unas
    # claves) y el parte del input tiene que leer el mismo UV que se inyectó.
    resultado["weather"]["uv_horario"] = weather.get("uv_horario")
    # La clase final la decide el canal con más riesgo personalizado
    # (prob_pers = max(calor, frío) en predict_ensemble). Antes se reportaba
    # SIEMPRE el canal calor: en el dataset de calor (agosto) no se notaba,
    # pero en un día de frío decía "RIESGO: PELIGRO" con un índice de 0.03
    # (el de calor) — incoherente y el modelo lo aprendería. Se reporta el
    # canal que de verdad movió la clase.
    calor = (resultado.get("perfil") or {}).get("calor") or {}
    frio = (resultado.get("perfil") or {}).get("frio") or {}
    canal = frio if frio.get("prob_personalizada", 0.0) > calor.get("prob_personalizada", 0.0) else calor
    return {
        "clase": resultado.get("clase_final_label", "DESCONOCIDO"),
        "indice_personalizado": canal.get("prob_personalizada", 0.0),
        "indice_base": canal.get("prob_poblacional", 0.0),
        "factor_total": canal.get("factor_total", 1.0),
        "producto_bruto": canal.get("producto_bruto"),
        "capado": bool(canal.get("capado")),
        "factores": canal.get("factores") or [],
        "recomendaciones": resultado.get("recomendaciones") or [],
        "clima": _clima_de_la_ventana(resultado, perfil),
        "perfil": perfil,
    }


def _fila_ventana_df(weather: dict, inicio: int, fin: int) -> pd.DataFrame | None:
    """Filas de df_hora cuya hora está en la ventana [inicio, fin).

    Es el dato horario que consumen los modelos (df_features se construye sobre
    df_hora); las horas repiten por día y `perfil_horario` se queda solo con el
    HI máximo por hora, así que aquí se busca la humedad/viento/UV de la ventana.
    """
    df = weather.get("df_hora") if weather else None
    if df is None or getattr(df, "empty", True) or "datetime" not in df.columns:
        return None
    horas = pd.to_datetime(df["datetime"]).dt.hour
    return df.loc[(horas >= inicio) & (horas < fin)]


def _uv_de_la_ventana(weather: dict, inicio: int, fin: int) -> float | None:
    """UV máximo de la ventana de actividad, del Open-Meteo horario.

    `uv_horario` lo guarda `predecir` en el dict de weather (no puede ir en
    df_hora: rompería los modelos, que no vieron esa columna en fit). Es el
    mismo dato que se inyecta como `_uv_index` al pipeline, así el input y el
    output usan exactamente el mismo UV. Sin uv_horario, cae al `uv_index` del
    día (OpenUV), que es el que el pipeline usa en `_personalizar_si_hay`.
    """
    uv_horario = weather.get("uv_horario")
    if isinstance(uv_horario, pd.DataFrame) and not uv_horario.empty and "uv_index" in uv_horario.columns:
        horas = pd.to_datetime(uv_horario["datetime"]).dt.hour
        vals = pd.to_numeric(
            uv_horario.loc[(horas >= inicio) & (horas < fin), "uv_index"],
            errors="coerce",
        ).dropna()
        if len(vals) > 0:
            return round(float(vals.max()), 1)
    uv = weather.get("uv_index")
    if isinstance(uv, (int, float)):
        return round(float(uv), 1)
    return None


def _clima_de_la_ventana(resultado: dict, perfil: dict) -> dict:
    """El tiempo que de verdad se usó para calcular el riesgo.

    Sin esto el dataset era imposible de aprender: el `input` decía "Ubicación:
    Sevilla" y nada más, mientras que el `output` llevaba un índice calculado con
    el parte meteorológico del día en que se generó. El mismo input tenía
    respuestas distintas según el día — comprobado: un perfil que el 30-jul daba
    0.23 hoy da 0.24. El modelo solo podía memorizar "Sevilla en julio = calor",
    y evaluarlo medía la meteorología del día de generación, no lo que sabe.

    Se saca la ventana de actividad (hora_inicio + duración) del perfil horario,
    que es sobre la que se calcula el riesgo, no el `current` del momento.

    Los cinco campos son obligatorios para el dataset: la temperatura media y la
    máxima (máx SIEMPRE, aunque coincida con la media), la humedad, el viento y
    el UV de la ventana. Si `current` no trae humedad/viento se miran en la misma
    ventana de df_hora (el dato que consumen los modelos); el UV viene de
    `_uv_de_la_ventana`, el mismo valor que `predecir` inyecta al pipeline.
    """
    weather = resultado.get("weather") or {}
    current = weather.get("current") or {}
    horario = weather.get("perfil_horario") or []

    inicio = int(perfil.get("hora_inicio") or 0)
    fin = inicio + max(1, round(float(perfil.get("duracion_h") or 1)))
    temps = [h["temp"] for h in horario
             if isinstance(h.get("temp"), (int, float)) and inicio <= h.get("hora", -1) < fin]
    if not temps:
        temps = [t for t in (current.get("t2m_c"),) if isinstance(t, (int, float))]

    ventana = _fila_ventana_df(weather, inicio, fin)

    def _media_ventana(col):
        if ventana is None or col not in ventana.columns:
            return None
        vals = pd.to_numeric(ventana[col], errors="coerce").dropna()
        return round(float(vals.mean()), 1) if len(vals) > 0 else None

    rh = current.get("rh")
    if rh is None:
        rh = _media_ventana("rh")
    viento = current.get("wind_speed_kmh")
    if viento is None:
        viento = _media_ventana("wind_speed_kmh")

    return {
        "t_media": round(sum(temps) / len(temps), 1) if temps else None,
        "t_max": round(max(temps), 1) if temps else None,
        "rh": rh,
        "viento_kmh": viento,
        "uv": _uv_de_la_ventana(weather, inicio, fin),
    }


# ---------------------------------------------------------------------------
# Formato de respuesta (texto natural para el dataset)
# ---------------------------------------------------------------------------


def formatear_respuesta(perfil: dict, riesgo: dict) -> str:
    """Convierte el resultado de la predicción en texto tipo bot.

    Los factores y las recomendaciones salen del pipeline, no de una escalera de
    if/elif escrita a mano: si el dataset enseña recomendaciones inventadas, el
    modelo las repetirá con toda la seguridad del mundo.
    """
    lineas = [f"RIESGO: {riesgo.get('clase', 'DESCONOCIDO')}", ""]
    lineas.append(f"Índice personalizado: {riesgo.get('indice_personalizado', 0):.2f}")
    lineas.append(f"Índice poblacional: {riesgo.get('indice_base', 0):.2f}")
    # Sin esto, el 78 % de los ejemplos ponía "×3.00" y el modelo aprendería que ese
    # es el factor de medio mundo. 3.0 es CAP_FACTORES_DEFECTO, un techo: el producto
    # real de un perfil cargado pasa de ×100. Se muestran los dos números.
    factor = riesgo.get("factor_total", 1.0)
    bruto = riesgo.get("producto_bruto")
    if riesgo.get("capado") and isinstance(bruto, (int, float)):
        lineas.append(f"Factor total aplicado: ×{factor:.2f} "
                      f"(tope; el producto de sus factores da ×{bruto:.2f})")
    else:
        lineas.append(f"Factor total aplicado: ×{factor:.2f}")

    factores = riesgo.get("factores") or []
    if factores:
        lineas.append("")
        lineas.append("Factores activados:")
        for f in factores:
            nombre = f.get("nombre") if isinstance(f, dict) else str(f)
            coef = f.get("valor", f.get("coef")) if isinstance(f, dict) else None
            lineas.append(f"- {nombre}: ×{coef:.2f}" if isinstance(coef, (int, float))
                          else f"- {nombre}")

    recs = riesgo.get("recomendaciones") or []
    if recs:
        lineas.append("")
        lineas.append("Recomendaciones:")
        for r in recs[:4]:
            lineas.append(f"- {r}")

    return "\n".join(lineas)


def formatear_input(perfil: dict, clima: dict | None = None) -> str:
    """Convierte el perfil a texto legible para el prompt.

    `clima` es el parte con el que se calculó la respuesta. Va en el input a
    propósito: es la única forma de que el par (input → output) sea estable y de
    que la tarea sea aprendible. Ver `_clima_de_la_ventana`.
    """
    partes = [
        f"Edad: {perfil['edad']}",
        f"Sexo: {perfil['sexo']}",
    ]
    if perfil.get("grasa"):
        partes.append(f"Grasa corporal: {perfil['grasa']}%")
    partes.append(f"Aclimatado: {'sí' if perfil['aclimatado'] else 'no'}")
    if perfil.get("fototipo"):
        partes.append(f"Fototipo: {perfil['fototipo']}")
    if perfil.get("comorbilidades"):
        partes.append(f"Comorbilidades: {perfil['comorbilidades']}")
    if perfil.get("medicacion"):
        partes.append(f"Medicación: {perfil['medicacion']}")
    if perfil.get("nivel_actividad"):
        partes.append(f"Actividad: {perfil['nivel_actividad']}")
    if perfil.get("duracion_h"):
        partes.append(f"Duración: {perfil['duracion_h']}h")
    if perfil.get("hora_inicio") is not None:
        partes.append(f"Desde las: {perfil['hora_inicio']}:00")
    if perfil.get("provincia"):
        partes.append(f"Ubicación: {perfil['provincia']}")
    if perfil.get("situacion_social"):
        partes.append(f"Situación social: {perfil['situacion_social']}")
    if perfil.get("entrenado") is not None:
        partes.append(f"Entrenado: {'sí' if perfil['entrenado'] else 'no'}")
    if perfil.get("ocupacion"):
        partes.append(f"Ocupación: {perfil['ocupacion']}")
    llega = [etiqueta for clave, etiqueta in (
        ("fiesta", "fiesta o alcohol reciente"),
        ("falta_sueno", "ha dormido poco"),
        ("enfermedad_reciente", "enfermedad reciente"),
    ) if perfil.get(clave)]
    if llega:
        partes.append(f"Cómo llega: {', '.join(llega)}")

    if clima:
        meteo = []
        if clima.get("t_media") is not None:
            meteo.append(f"{clima['t_media']} °C de media")
        # La máxima va SIEMPRE, aunque coincida con la media: el criterio de la
        # feature pide los dos campos en cada ejemplo.
        if clima.get("t_max") is not None:
            meteo.append(f"máx {clima['t_max']} °C")
        if clima.get("rh") is not None:
            meteo.append(f"humedad {clima['rh']} %")
        if clima.get("viento_kmh") is not None:
            meteo.append(f"viento {clima['viento_kmh']} km/h")
        if isinstance(clima.get("uv"), (int, float)):
            # El índice UV viene con toda la precisión del float (7.7645). Enseñarle
            # esos decimales al modelo es enseñarle ruido: la escala UV es 0-11+.
            meteo.append(f"UV {clima['uv']:.1f}")
        if meteo:
            partes.append(f"Tiempo en esa franja: {', '.join(meteo)}")

    return ". ".join(partes) + "."


# ---------------------------------------------------------------------------
# Generación del dataset completo
# ---------------------------------------------------------------------------


INSTRUCCION = "Predice el riesgo térmico para este perfil y da recomendaciones."


def generar_dataset(num_ejemplos: int, equilibrar: bool = True) -> list[dict]:
    """Genera dataset completo en formato Alpaca.

    Con `equilibrar`, genera de más y va descartando ejemplos de la clase que ya
    tiene su cupo, hasta repartir `num_ejemplos` entre las clases que aparezcan.
    Sin esto el reparto lo decide el clima: la primera versión salió con 85 PELIGRO
    y 15 SEGURO, y un modelo entrenado ahí aprende a decir PELIGRO por defecto.
    """
    cupo = num_ejemplos  # sin equilibrar, el cupo por clase es el total
    if equilibrar:
        # Las clases del sistema son tres; se deja holgura para que no se quede
        # corto si un clima no da nunca cierta clase.
        cupo = max(1, round(num_ejemplos / 3 * 1.35))

    dataset: list[dict] = []
    por_clase: dict[str, int] = {}
    descartados = 0
    incompletos = 0
    fallidos: list[str] = []

    # Se piden más perfiles de los necesarios: al descartar por cupo se gastan.
    for perfil in generar_perfiles(num_ejemplos * 4 if equilibrar else num_ejemplos):
        if len(dataset) >= num_ejemplos:
            break
        # Un perfil que revienta se salta y se cuenta. NO se sustituye por datos
        # inventados: eso es lo que hacía el `_predecir_fake` que se ha quitado.
        # Pasa con partes meteorológicos incompletos, que dan un índice NaN.
        try:
            riesgo = predecir(perfil)
        except Exception as exc:
            fallidos.append(f"{perfil.get('provincia')}: {type(exc).__name__}: {exc}")
            continue
        # El input de cada ejemplo tiene que llevar el parte completo (media,
        # máxima, humedad, viento y UV); si falta alguno, el ejemplo no entra.
        # Mejor que un dataset con el hueco: el criterio de la feature lo exige.
        clima = riesgo.get("clima") or {}
        campos_parte = ("t_media", "t_max", "rh", "viento_kmh", "uv")
        if not all(clima.get(c) is not None for c in campos_parte):
            incompletos += 1
            continue
        clase = riesgo.get("clase", "DESCONOCIDO")
        if equilibrar and por_clase.get(clase, 0) >= cupo:
            descartados += 1
            continue
        por_clase[clase] = por_clase.get(clase, 0) + 1
        dataset.append({
            "instruction": INSTRUCCION,
            "input": formatear_input(perfil, riesgo.get("clima")),
            "output": formatear_respuesta(perfil, riesgo),
        })

    if equilibrar:
        reparto = " · ".join(f"{k} {v}" for k, v in sorted(por_clase.items()))
        print(f"  Reparto por clase: {reparto}  ({descartados} descartados por cupo)")
    if incompletos:
        print(f"  {incompletos} perfiles saltados por parte incompleto (sin media/máx/humedad/viento/UV)")
    if fallidos:
        print(f"  {len(fallidos)} perfiles saltados por error de predicción:")
        for linea in fallidos[:5]:
            print(f"    - {linea}")
        if len(fallidos) > 5:
            print(f"    ... y {len(fallidos) - 5} más")
    if len(dataset) < num_ejemplos:
        print(f"  AVISO: solo {len(dataset)} de {num_ejemplos} ejemplos. "
              "Sube --num-ejemplos o revisa los errores de arriba.")

    return dataset


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generar dataset sintético para fine-tuning")
    p.add_argument("-o", "--output", default="data/llm/train.jsonl",
                   help="Ruta del JSONL de salida")
    p.add_argument("-n", "--num-ejemplos", type=int, default=150,
                   help="Número de ejemplos a generar")
    p.add_argument("--val-split", type=float, default=0.1,
                   help="Fracción para validación (default: 0.1)")
    p.add_argument("--sin-equilibrar", action="store_true",
                   help="No equilibrar las clases: acepta el reparto que salga del clima")
    p.add_argument("--seed", type=int, default=RANDOM_SEED,
                   help="Semilla aleatoria")
    return p.parse_args(argv)


def main() -> None:
    args = parse_args()
    global RANDOM_SEED
    RANDOM_SEED = args.seed

    # Sin esto, download_openuv no encuentra OpenUV_API_KEY y el `uv_index` del
    # pipeline queda en None para casi todo el dataset (16/400 en la ronda
    # anterior). El UV de la ventana sale del Open-Meteo horario, pero el del día
    # (OpenUV) es el fallback, y cuanto más dato real, mejor.
    load_dotenv()

    print(f"Generando {args.num_ejemplos} ejemplos sintéticos...")
    dataset = generar_dataset(args.num_ejemplos, equilibrar=not args.sin_equilibrar)

    # Dividir train/val
    random.seed(RANDOM_SEED)
    indices = list(range(len(dataset)))
    random.shuffle(indices)
    val_n = int(len(dataset) * args.val_split)
    val_indices = set(indices[:val_n])
    train_indices = indices[val_n:]

    train = [dataset[i] for i in train_indices]
    val = [dataset[i] for i in val_indices]

    # Guardar
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    val_path = output_path.with_name("val.jsonl")

    with open(output_path, "w") as f:
        for ex in train:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
    with open(val_path, "w") as f:
        for ex in val:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    print(f"  Train: {len(train)} ejemplos → {output_path}")
    print(f"  Val:   {len(val)} ejemplos → {val_path}")

    # Mostrar ejemplo
    ex = dataset[0]
    print(f"\nEjemplo:\n  Input: {ex['input'][:100]}...\n  Output: {ex['output'][:200]}...")


if __name__ == "__main__":
    main()
