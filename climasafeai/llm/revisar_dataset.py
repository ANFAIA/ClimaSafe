#! /usr/bin/env python
"""
Control de calidad del dataset sintético de fine-tuning (LLM-015).

Detecta pares sospechosos en data/llm/*.jsonl antes de empaquetar el zip del
Colab: un LLM fine-tuneado aprende las alucinaciones que ve repetidas, así que
un par con la clase equivocada, sin el parte meteorológico o con un perfil
imposible se convierte en comportamiento aprendido.

Detectores (cada uno con su salida por par y agregada):

  a. Clase incoherente: re-ejecuta `predict_ensemble` (el pipeline real) sobre
     una muestra de pares — con weather sintético reproducible, SIN red — y
     compara la clase que afirma la respuesta (RIESGO: X) con la del pipeline.
     Ejecutar el ensemble es lento (XGBoost + RandomForest + LSTM por par), por
     eso se muestrea: `--muestra 50` por defecto. Si faltan los modelos
     entrenados (models/*.joblib), el detector entra en modo degradado y marca
     los pares como "no verificables" en vez de fallar.
  b. Inputs sin el campo "Tiempo en esa franja" (el parte meteorológico con el
     que se calculó la respuesta) o con el parte incompleto (sin máx o sin UV).
  c. Perfiles imposibles: edad fuera de 0-120, grasa fuera de 3-70, sexo no
     válido, o claves de comorbilidades/medicación/situación social/ocupación
     que `factores_riesgo.json` no reconoce.
  d. Duplicados casi idénticos: inputs con similitud de Jaccard sobre tokens
     (normalizados: minúsculas, espacios colapsados, números anonimizados)
     superior al umbral (0.9 por defecto).
  e. Desequilibrio de clases: distribución de la clase que afirma la respuesta
     (SEGURO/PRECAUCIÓN/PELIGRO); avisa si alguna clase queda por debajo del
     10 % del conjunto.

Uso:
    python climasafeai/llm/revisar_dataset.py \
        --train data/llm/train.jsonl --val data/llm/val.jsonl \
        --out /tmp/informe_qc.json --muestra 50
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[2]
FACTORES_PATH = ROOT / "data" / "factores_riesgo.json"
MODELS_DIR = ROOT / "models"

MARCA = "Tiempo en esa franja"
CLASES = ("SEGURO", "PRECAUCION", "PELIGRO")
SEXOS_VALIDOS = {"hombre", "mujer"}
RANGO_EDAD = (0, 120)
RANGO_GRASA = (3, 70)
UMBRAL_DESEQUILIBRIO = 0.10
UMBRAL_SIMILITUD = 0.90
SEED = 42

# El QC re-ejecuta el pipeline con un weather SINTÉTICO reproducible (el input
# del dataset ya lleva el parte con el que se calculó la respuesta). Para ello
# necesita lat/lon de la provincia: se combinan los escenarios del generador de
# calor con las coordenadas del generador de frío.
_COORDS: dict[str, tuple[float, float]] = {}


def _cargar_coords() -> dict[str, tuple[float, float]]:
    global _COORDS
    if _COORDS:
        return _COORDS
    try:
        from climasafeai.llm import generar_dataset as gd
        from climasafeai.llm import generar_dataset_frio as gf

        _COORDS = {esc[2]: (esc[0], esc[1]) for esc in gd.ESCENARIOS}
        _COORDS.update(gf.COORDS_PROVINCIA)
    except ImportError:
        _COORDS = {}
    return _COORDS


# ─────────────────────────────────────────────────────────────────────────────
# Carga y helpers básicos
# ─────────────────────────────────────────────────────────────────────────────


def _cargar_jsonl(path: str | Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(linea) for linea in f if linea.strip()]


def _clase_afirmada(output: str) -> str | None:
    """La clase que la respuesta afirma: 'RIESGO: PELIGRO' → 'PELIGRO'."""
    m = re.search(r"RIESGO:\s*(\w+)", output or "")
    return m.group(1) if m else None


def _normalizar(texto: str) -> str:
    """Minúsculas, espacios colapsados y números anonimizados.

    Los números se convierten a '#' a propósito: dos inputs con el mismo perfil
    pero el parte con 41.5 °C vs 42.1 °C son casi idénticos para el
    aprendizaje, y sin anonimizar '41.5' y '42.1' contarían como tokens
    distintos.
    """
    t = (texto or "").lower()
    t = re.sub(r"\s+", " ", t)
    t = re.sub(r"\d+(?:\.\d+)?", "#", t)
    return t.strip()


def _jaccard(a: str, b: str) -> float:
    ta = set(_normalizar(a).split())
    tb = set(_normalizar(b).split())
    if not ta and not tb:
        return 1.0
    inter = len(ta & tb)
    union = len(ta | tb)
    return inter / union if union else 0.0


def _claves_validas() -> dict[str, set[str]]:
    """Claves que el pipeline reconoce (por categoría de perfil).

    La fuente principal es `data/factores_riesgo.json`. Las ocupaciones no
    viven ahí: `personalizacion.py::_OCUPACION_NIVELES` define reparto/
    construccion/campo con coeficiente directo, y son las que usa el generador.
    """
    if not FACTORES_PATH.exists():
        return {}
    datos = json.loads(FACTORES_PATH.read_text())
    # categoría del JSON → campo del perfil que el QC valida
    por_categoria = {
        "comorbilidades": "comorbilidades",
        "farmacos": "medicacion",
        "situacional": "situacion_social",
        "ocupacional": "ocupacion",
    }
    claves: dict[str, set[str]] = {"comorbilidades": set(), "medicacion": set(),
                                   "situacion_social": set(), "ocupacion": set()}
    for canal in ("calor", "frio"):
        for cat, mapa in datos.get(canal, {}).items():
            campo = por_categoria.get(cat)
            if campo:
                claves[campo].update(mapa)
    claves["ocupacion"].update(OCUPACIONES_PIPELINE)
    return claves


# Ocupaciones con coeficiente directo en personalizacion.py::_OCUPACION_NIVELES
# (líneas 183-186). No están en factores_riesgo.json pero el pipeline las lee.
OCUPACIONES_PIPELINE = {"reparto", "construccion", "campo"}

_CLAVES = _claves_validas()


# ─────────────────────────────────────────────────────────────────────────────
# Detectores puros (sin pipeline)
# ─────────────────────────────────────────────────────────────────────────────


def detectar_sin_marca(ejemplos: list[dict]) -> list[int]:
    """Índices de pares cuyo input no lleva 'Tiempo en esa franja'."""
    return [i for i, e in enumerate(ejemplos)
            if MARCA not in (e.get("input") or "")]


def detectar_parte_incompleto(ejemplos: list[dict]) -> list[int]:
    """Índices con la marca pero sin máx o sin UV dentro del parte.

    El criterio del dataset (LLM-004) exige los cinco campos del parte: media,
    máxima, humedad, viento y UV. Un parte sin máx o sin UV es un input con
    menos información de la que el modelo verá en producción.
    """
    indices = []
    for i, e in enumerate(ejemplos):
        input_txt = e.get("input") or ""
        if MARCA not in input_txt:
            continue
        parte = input_txt.split(MARCA, 1)[1]
        if "máx" not in parte or re.search(r"\bUV\b", parte) is None:
            indices.append(i)
    return indices


_PATRONES: dict[str, tuple[str, Callable[[str], Any]]] = {
    "edad": (r"Edad:\s*(\d+)", int),
    "sexo": (r"Sexo:\s*(\w+)", str),
    "grasa": (r"Grasa corporal:\s*([\d.]+)%", float),
    "aclimatado": (r"Aclimatado:\s*(sí|si|no)", lambda s: s in ("sí", "si")),
    "fototipo": (r"Fototipo:\s*(\w+)", str),
    "comorbilidades": (r"Comorbilidades:\s*([^.]+)", str.strip),
    "medicacion": (r"Medicación:\s*([^.]+)", str.strip),
    "nivel_actividad": (r"Actividad:\s*(\w+)", str),
    "duracion_h": (r"Duración:\s*([\d.]+)h", float),
    "hora_inicio": (r"Desde las:\s*(\d+):00", int),
    "provincia": (r"Ubicación:\s*([A-Za-zÁÉÍÓÚáéíóúñÑüÜ]+)", str),
    "situacion_social": (r"Situación social:\s*([^.]+)", str.strip),
    "entrenado": (r"Entrenado:\s*(sí|si|no)", lambda s: s in ("sí", "si")),
    "ocupacion": (r"Ocupación:\s*(\w+)", str),
    "como_llega": (r"Cómo llega:\s*([^.]+)", str.strip),
}


def _separar_claves(valor) -> set[str]:
    if not valor:
        return set()
    if isinstance(valor, (set, list, tuple)):
        return set(valor)
    return {x.strip() for x in str(valor).split(",") if x.strip()}


def _problemas_perfil(input_txt: str) -> list[str]:
    """Problemas de un input: rango, sexo o claves que el pipeline no conoce."""
    problemas: list[str] = []
    perfil = _parsear_input(input_txt)[0]
    edad = perfil.get("edad")
    if edad is not None and not (RANGO_EDAD[0] <= edad <= RANGO_EDAD[1]):
        problemas.append(f"edad {edad} fuera de {RANGO_EDAD[0]}-{RANGO_EDAD[1]}")
    grasa = perfil.get("grasa")
    if grasa is not None and not (RANGO_GRASA[0] <= grasa <= RANGO_GRASA[1]):
        problemas.append(f"grasa {grasa}% fuera de {RANGO_GRASA[0]}-{RANGO_GRASA[1]}")
    sexo = perfil.get("sexo")
    if sexo and sexo.lower() not in SEXOS_VALIDOS:
        problemas.append(f"sexo '{sexo}' no válido")
    for campo, clave_cat in (("comorbilidades", "comorbilidades"),
                             ("medicacion", "medicacion"),
                             ("situacion_social", "situacion_social"),
                             ("ocupacion", "ocupacion")):
        valor = perfil.get(campo)
        if not valor:
            continue
        validas = _CLAVES.get(clave_cat, set())
        desconocidas = _separar_claves(valor) - validas
        if desconocidas:
            problemas.append(
                f"{campo} con claves no reconocidas: {', '.join(sorted(desconocidas))}"
            )
    return problemas


def detectar_perfiles_imposibles(ejemplos: list[dict]) -> list[dict]:
    """Pares cuyo input describe un perfil imposible o no reconocible."""
    hallazgos = []
    for i, e in enumerate(ejemplos):
        problemas = _problemas_perfil(e.get("input") or "")
        if problemas:
            hallazgos.append({"indice": i, "problemas": problemas})
    return hallazgos


def detectar_duplicados(
    ejemplos: list[dict], umbral: float = UMBRAL_SIMILITUD
) -> list[dict]:
    """Pares de inputs casi idénticos (Jaccard de tokens > umbral)."""
    inputs = [e.get("input") or "" for e in ejemplos]
    pares = []
    for i in range(len(inputs)):
        for j in range(i + 1, len(inputs)):
            sim = _jaccard(inputs[i], inputs[j])
            if sim > umbral:
                pares.append({"i": i, "j": j, "similitud": round(sim, 4)})
    return pares


def detectar_desequilibrio(ejemplos: list[dict]) -> dict:
    """Distribución de la clase afirmada y aviso si alguna cae < 10 %."""
    conteo: Counter = Counter(_clase_afirmada(e.get("output") or "")
                              for e in ejemplos)
    total = sum(conteo.values())
    bajo = []
    if total > 0:
        for clase in CLASES:
            frac = conteo.get(clase, 0) / total
            if frac < UMBRAL_DESEQUILIBRIO:
                bajo.append({"clase": clase, "n": conteo.get(clase, 0),
                             "fraccion": round(frac, 4)})
    return {
        "n": total,
        "distribucion": {k: v for k, v in sorted(conteo.items())},
        "umbral": UMBRAL_DESEQUILIBRIO,
        "desequilibrio": bool(bajo),
        "clases_bajo_umbral": bajo,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Detector de clase: re-ejecuta predict_ensemble sin red
# ─────────────────────────────────────────────────────────────────────────────


def _parsear_input(input_txt: str) -> tuple[dict, dict]:
    """Perfil + clima desde el texto del input (formato de formatear_input)."""
    perfil: dict = {}
    for campo, (patron, cast) in _PATRONES.items():
        m = re.search(patron, input_txt or "")
        if m:
            try:
                perfil[campo] = cast(m.group(1))
            except ValueError:
                continue

    clima: dict = {}
    # El parte va al final del input y los números llevan punto decimal, así
    # que NO se captura con [^.]+: se toma todo lo que sigue a la marca hasta
    # el final de la línea.
    m = re.search(r"Tiempo en esa franja:\s*(.+)$", input_txt or "")
    if m:
        parte = m.group(1)
        for campo, patron, cast in (
            # El signo negativo es obligatorio: los ejemplos de frío real
            # (parquet 2016-2026) llevan medias y máximas bajo cero, y sin él
            # el QC leía "-6.2" como media "6.2" y la máx negativa como None
            # (→ todo el canal frío quedaba "no verificable").
            ("t_media", r"(-?[\d.]+) °C de media", float),
            ("t_max", r"máx\s*(-?[\d.]+) °C", float),
            ("rh", r"humedad\s*([\d.]+) %", float),
            ("viento_kmh", r"viento\s*([\d.]+) km/h", float),
            ("uv", r"\bUV\s*(\d+(?:\.\d+)?)", float),
        ):
            m2 = re.search(patron, parte)
            if m2:
                clima[campo] = cast(m2.group(1))
    return perfil, clima


def _perfil_para_modelo(perfil: dict) -> dict:
    """Traduce el perfil parseado a las claves que LEE el pipeline (igual que
    generar_dataset._perfil_para_modelo)."""
    p = {
        "edad": perfil.get("edad"),
        "sexo": perfil.get("sexo"),
        "aclimatado": perfil.get("aclimatado"),
        "nivel_actividad": perfil.get("nivel_actividad"),
        "hora_inicio": perfil.get("hora_inicio"),
        "duracion_actividad_h": perfil.get("duracion_h"),
        "comorbilidades": _separar_claves(perfil.get("comorbilidades")),
        "farmacos": _separar_claves(perfil.get("medicacion")),
    }
    if perfil.get("grasa") is not None:
        p["porcentaje_grasa"] = perfil["grasa"]
    if perfil.get("fototipo"):
        p["fototipo"] = perfil["fototipo"]
    if perfil.get("situacion_social"):
        p["situacion_social"] = _separar_claves(perfil["situacion_social"])
    if perfil.get("entrenado") is not None:
        p["entrenado"] = perfil["entrenado"]
    if perfil.get("ocupacion"):
        p["ocupacion"] = perfil["ocupacion"]
    como_llega = perfil.get("como_llega") or ""
    for clave, etiqueta in (
        ("fiesta", "fiesta o alcohol reciente"),
        ("falta_sueno", "ha dormido poco"),
        ("enfermedad_reciente", "enfermedad reciente"),
    ):
        if etiqueta in como_llega:
            p[clave] = True
    return p


def _construir_weather(perfil: dict, clima: dict) -> dict | None:
    """Weather sintético reproducible desde el parte del input.

    El df_hora se monta alrededor de la ventana de actividad
    (hora_inicio + duración) con heat_index_c = t_max en esa ventana, para que
    `perfil_horario_desde_df` y los overrides físicos del ensemble (HI >= 39,
    HI >= 27 + UV > 3) vean exactamente el extremo que el parte declara. Las
    features de persistencia (horas sobre umbral, grados-día...) se DERIVAN de
    la temperatura del parte en vez de usar constantes. El df_features (el día
    que ven los modelos tabulares) se construye DIA_DELTA por debajo de la
    ventana: el parte declara la media/máx de la ventana de actividad, no del
    día (ver nota en _construir_weather). Sin t_media no hay weather que
    construir (señal de input roto).
    """
    import pandas as pd

    t_media = clima.get("t_media")
    if t_media is None:
        return None
    t_max = clima.get("t_max") or t_media
    rh = clima.get("rh") or 50.0
    viento = clima.get("viento_kmh") or 10.0
    inicio = int(perfil.get("hora_inicio") or 12)
    duracion = max(1, round(float(perfil.get("duracion_h") or 1)))

    # El día entero NO es la ventana de actividad. El parte declara la media y
    # la máxima de la VENTANA (2-6 h, el tramo que el perfil está expuesto), no
    # del día: medido con el forecast real (2026-08-19, 13 escenarios), la
    # ventana de tarde queda +3..+17 °C sobre la media diaria. Reconstruir el
    # día con la media de la ventana (como se hacía) convertía un día de 21 °C
    # con pico de 30.9 en un día tórrido de 30.9, el ensemble disparaba la prob
    # de calor (0.07 → 0.50) y el QC acusaba de PELIGRO a respuestas que el
    # pipeline real, con el weather completo, había dado SEGURO. El día
    # sintético se queda DIA_DELTA por debajo de la ventana (el pico de la
    # ventana es la máxima del día, no su media).
    DIA_DELTA = 8.0
    dia_t2m = t_media - DIA_DELTA
    dia_hi = t_max - DIA_DELTA

    # Persistencia derivada del parte: coherente con el día que describe.
    horas_sobre_umbral = max(0, int((t_max - 27.0) * 2)) if t_max > 27.0 else 0
    horas_bajo_umbral = max(0, int((5.0 - t_media) * 2)) if t_media < 5.0 else 0
    dias_consec_sobre = 2 if t_media >= 27.0 else (1 if t_media >= 24.0 else 0)
    grados_calor = max(0.0, t_media - 24.0) * 3.5
    grados_frio = max(0.0, 5.0 - t_media) * 3.5
    t_noche = min(t_media - 5.0, t_media * 0.8)

    horas = list(range(max(0, inicio - 2), inicio + duracion + 3))
    filas = []
    for h in horas:
        # Dentro de la ventana, el extremo que el parte declara (t_max): es lo
        # que ve perfil_horario y lo que activa los overrides físicos. Fuera de
        # la ventana, el día (dia_t2m).
        if inicio <= h < inicio + duracion:
            temp = t_max
        else:
            temp = dia_t2m
        # La hora puede pasar de 24 (ventana nocturna): se envuelve.
        hh = h % 24
        filas.append({
            "datetime": pd.Timestamp(f"2024-07-15 {hh:02d}:00"),
            "t2m_c": temp, "rh": rh, "wind_speed_kmh": viento,
            "sp": 101300.0,
            "heat_index_c": temp, "wbgt_c": temp - 1.0,
            "wind_chill_c": temp - 1.0,
        })
    df_hora = pd.DataFrame(filas)

    # df_features describe el DÍA (lo que consumen los modelos tabulares), así
    # que usa dia_t2m/dia_hi, no los valores de la ventana.
    df_features = pd.DataFrame([{
        "fecha": "2024-07-15", "datetime": "2024-07-15 14:00",
        "t2m_c": dia_t2m, "rh": rh, "wind_speed_kmh": viento, "sp": 101300.0,
        "heat_index_c": dia_hi, "wbgt_c": dia_hi - 1.0, "wind_chill_c": dia_hi - 1.0,
        "heat_index_mean": dia_hi - 1.0, "heat_index_std": 2.0,
        "heat_index_min": dia_hi - 3.0,
        "horas_sobre_umbral": horas_sobre_umbral,
        "wind_chill_mean": dia_hi - 1.0, "wind_chill_std": 1.0,
        "wind_chill_max": dia_hi, "horas_bajo_umbral": horas_bajo_umbral,
        "heat_index_c_lag1": dia_hi - 1.0, "heat_index_c_roll3": dia_hi,
        "heat_index_c_roll7": dia_hi - 1.0,
        "dias_consec_sobre_umbral": dias_consec_sobre,
        "grados_dia_calor_roll7": grados_calor,
        "grados_dia_calor_roll14": grados_calor * 1.5,
        "wind_chill_mean_roll3": dia_hi - 1.0,
        "wind_chill_mean_roll7": dia_hi - 1.0,
        "wind_chill_mean_roll14": dia_hi - 1.0,
        "grados_dia_frio_roll7": grados_frio,
        "grados_dia_frio_roll14": grados_frio * 1.5,
        "dias_consec_bajo_umbral": 1 if t_media < 5.0 else 0,
        "t2m_min_noche_lag1": t_noche, "t2m_min_noche_roll7": t_noche - 0.5,
        "dias_consec_wc_severo": 0, "horas_wc_severo_sum14": 0,
    }])

    lat, lon = _coordenadas(perfil.get("provincia"))
    return {
        "lat": lat,
        "lon": lon,
        "current": {"t2m_c": dia_t2m, "rh": rh,
                    "wind_speed_kmh": viento, "sp": 101300.0},
        "df_hora": df_hora,
        "df_features": df_features,
        "uv_index": clima.get("uv"),
        "target_date": "2024-07-15",
    }


def _coordenadas(provincia: str | None) -> tuple[float, float]:
    coords = _cargar_coords()
    if provincia and provincia in coords:
        return coords[provincia]
    return (40.4168, -3.7038)  # Madrid por defecto


def _pipeline_disponible() -> bool:
    """Sin modelos entrenados el detector de clase no puede verificar."""
    return (MODELS_DIR / "XGBoost_calor.joblib").exists() and (
        MODELS_DIR / "RandomForest_frio.joblib").exists()


def _distancia_clase(a: str, b: str) -> int:
    if a not in CLASES or b not in CLASES:
        return -1
    return abs(CLASES.index(a) - CLASES.index(b))


def _verificar_par(ejemplo: dict) -> dict | None:
    """Clase afirmada vs pipeline para un par. None si es coherente.

    Los pares con el parte incompleto (sin máx o sin UV) NO se verifican: sin
    esos campos la reconstrucción del weather es demasiado especulativa y la
    comparación de clase sería ruido. Ya quedan señalados por el detector de
    parte incompleto.
    """
    from climasafeai.models.ensemble import predict_ensemble

    perfil, clima = _parsear_input(ejemplo.get("input") or "")
    if clima.get("t_media") is None:
        return {"verificable": False, "razon": "input sin parte meteorológico"}
    if clima.get("t_max") is None or clima.get("uv") is None:
        return {"verificable": False,
                "razon": "parte incompleto (sin máx o sin UV): no comparable"}
    weather = _construir_weather(perfil, clima)
    if weather is None:
        return {"verificable": False, "razon": "input sin parte meteorológico"}
    try:
        resultado = predict_ensemble(
            lat=weather["lat"], lon=weather["lon"],
            provincia=perfil.get("provincia") or "Madrid",
            perfil=_perfil_para_modelo(perfil),
            weather=weather,
        )
    except Exception as exc:
        return {"verificable": False, "razon": f"{type(exc).__name__}: {exc}"}
    clase_pipeline = resultado.get("clase_final_label", "DESCONOCIDO")
    clase_afirmada = _clase_afirmada(ejemplo.get("output") or "")
    if clase_afirmada is None:
        return {"verificable": False, "razon": "output sin RIESGO: X"}
    distancia = _distancia_clase(clase_afirmada, clase_pipeline)
    if distancia == 0:
        return None  # coherente: no es un hallazgo
    return {
        "verificable": True,
        "clase_afirmada": clase_afirmada,
        "clase_pipeline": clase_pipeline,
        "gravedad": "critica" if distancia >= 2 else "menor",
    }


def verificar_clase(ejemplos: list[dict], muestra: int = 50) -> dict:
    """Re-ejecuta el pipeline sobre una muestra y compara clases."""
    if not _pipeline_disponible():
        return {
            "disponible": False,
            "n_verificados": 0,
            "no_verificables": len(ejemplos),
            "aviso": ("faltan models/*.joblib: pares marcados como "
                      "'no verificables', sin comparación de clase"),
            "incoherentes": [],
        }
    if muestra <= 0:
        return {"disponible": True, "n_verificados": 0, "incoherentes": [],
                "no_verificables": 0}
    rng = random.Random(SEED)
    indices = rng.sample(range(len(ejemplos)), min(muestra, len(ejemplos)))

    incoherentes = []
    no_verificables = 0
    n_verificados = 0
    for idx in indices:
        hallazgo = _verificar_par(ejemplos[idx])
        if hallazgo is None:
            n_verificados += 1
            continue
        if not hallazgo["verificable"]:
            no_verificables += 1
            continue
        n_verificados += 1
        incoherentes.append({"indice": idx, **hallazgo})

    return {
        "disponible": True,
        "muestra": len(indices),
        "n_verificados": n_verificados,
        "no_verificables": no_verificables,
        "incoherentes": incoherentes,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Orquestación y salida
# ─────────────────────────────────────────────────────────────────────────────


def revisar_fichero(
    ejemplos: list[dict], muestra: int = 50, umbral_sim: float = UMBRAL_SIMILITUD
) -> dict:
    sin_marca = detectar_sin_marca(ejemplos)
    incompleto = detectar_parte_incompleto(ejemplos)
    imposibles = detectar_perfiles_imposibles(ejemplos)
    duplicados = detectar_duplicados(ejemplos, umbral_sim)
    return {
        "n": len(ejemplos),
        "sin_marca": {"n": len(sin_marca), "indices": sin_marca},
        "parte_incompleto": {"n": len(incompleto), "indices": incompleto},
        "perfiles_imposibles": {"n": len(imposibles), "detalles": imposibles},
        "duplicados": {"n_pares": len(duplicados), "pares": duplicados},
        "desequilibrio": detectar_desequilibrio(ejemplos),
        "clase_pipeline": verificar_clase(ejemplos, muestra),
    }


def _resumen_fichero(resultado: dict) -> dict:
    return {
        "n": resultado["n"],
        "sin_marca": resultado["sin_marca"]["n"],
        "parte_incompleto": resultado["parte_incompleto"]["n"],
        "perfiles_imposibles": resultado["perfiles_imposibles"]["n"],
        "duplicados": resultado["duplicados"]["n_pares"],
        "desequilibrio": resultado["desequilibrio"]["desequilibrio"],
        "clase_pipeline": {
            "disponible": resultado["clase_pipeline"]["disponible"],
            "incoherentes": len(resultado["clase_pipeline"]["incoherentes"]),
            "n_verificados": resultado["clase_pipeline"]["n_verificados"],
        },
    }


def _print_resumen(ruta: str, res: dict) -> None:
    r = _resumen_fichero(res)
    print(f"  {ruta}  ({r['n']} pares)")
    print(f"    sin 'Tiempo en esa franja':        {r['sin_marca']}")
    print(f"    parte incompleto (sin máx/UV):     {r['parte_incompleto']}")
    print(f"    perfiles imposibles:               {r['perfiles_imposibles']}")
    print(f"    duplicados casi idénticos:         {r['duplicados']}")
    eq = res["desequilibrio"]
    print(f"    desequilibrio: {eq['desequilibrio']}  "
          f"(distribución {eq['distribucion']}, umbral < {eq['umbral']:.0%})")
    cp = res["clase_pipeline"]
    if not cp["disponible"]:
        print(f"    clase vs pipeline: NO VERIFICABLE — {cp['aviso']}")
    else:
        criticas = sum(1 for h in cp["incoherentes"] if h["gravedad"] == "critica")
        menores = len(cp["incoherentes"]) - criticas
        print(f"    clase vs pipeline: {len(cp['incoherentes'])} incoherentes "
              f"({criticas} críticas, {menores} menores) de "
              f"{cp['n_verificados']} verificados "
              f"({cp['no_verificables']} no verificables)")


def revisar(train_path: str, val_path: str, muestra: int = 50,
            umbral_sim: float = UMBRAL_SIMILITUD, out_path: str | None = None) -> dict:
    train = _cargar_jsonl(train_path)
    val = _cargar_jsonl(val_path)

    por_fichero = {
        "train": revisar_fichero(train, muestra, umbral_sim),
        "val": revisar_fichero(val, muestra, umbral_sim),
    }
    return {
        "ficheros": {"train": train_path, "val": val_path},
        "por_fichero": por_fichero,
        "resumen": {
            "train": _resumen_fichero(por_fichero["train"]),
            "val": _resumen_fichero(por_fichero["val"]),
        },
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="QC del dataset sintético de fine-tuning")
    p.add_argument("--train", default="data/llm/train.jsonl",
                   help="JSONL de entrenamiento")
    p.add_argument("--val", default="data/llm/val.jsonl",
                   help="JSONL de validación")
    p.add_argument("--out", default=None,
                   help="Ruta del informe JSON (por par y agregado)")
    p.add_argument("--muestra", type=int, default=50,
                   help="Pares a re-ejecutar con predict_ensemble (default: 50; "
                        "0 desactiva el detector de clase)")
    p.add_argument("--umbral-sim", type=float, default=UMBRAL_SIMILITUD,
                   help="Umbral de similitud para duplicados (default: 0.9)")
    return p.parse_args(argv)


def main() -> int:
    args = parse_args()
    resultados = revisar(args.train, args.val, args.muestra, args.umbral_sim,
                         args.out)
    print("QC del dataset sintético:")
    print("  train:", args.train)
    print("  val:  ", args.val)
    _print_resumen(args.train, resultados["por_fichero"]["train"])
    _print_resumen(args.val, resultados["por_fichero"]["val"])
    if args.out:
        with open(args.out, "w") as f:
            json.dump(resultados, f, ensure_ascii=False, indent=2)
        print(f"Informe JSON: {args.out}")
    # Código de salida: 0 aunque haya hallazgos — el QC informa, no bloquea.
    return 0


if __name__ == "__main__":
    sys.exit(main())
