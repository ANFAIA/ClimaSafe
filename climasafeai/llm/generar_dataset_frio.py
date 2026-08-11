#! /usr/bin/env python
"""
Genera ejemplos de FRÍO reales para el dataset de fine-tuning de Qwen 2.5
ClimaSafeAI (DATA-008).

El dataset actual (data/llm/train.jsonl + val.jsonl) es 100 % calor: se generó
en agosto con el forecast a 48 h de Open-Meteo. Esta feature añade ejemplos de
frío REALES desde data/processed/dataset_frio_labeled.parquet (2016-2026,
45 provincias, una fila por provincia y día con las features diarias con las
que se entrenaron los modelos).

Cada ejemplo sale del pipeline real, nunca de datos inventados: se construye el
dict `weather` que consume `predict_ensemble` a partir de la fila del parquet y
se ejecuta `predict_ensemble(lat, lon, provincia, perfil, weather)`. El input
lleva el parte completo (t_media, t_max, rh, viento, UV) con el que se calculó
la respuesta, igual que el generador de calor.

El UV no existe como dato histórico: OpenUV no da histórico (404 en
/v1/history) y el archivo de Open-Meteo devuelve uv_index a None. Se ESTIMA del
shortwave_radiation (W/m²) que sí sirve el archivo, con la regla de pulgar
documentada UVI ≈ GHI/100 (guía WHO "Global Solar UV Index"). En días fríos el
UV no es determinante del riesgo (los overrides de calor necesitan HI ≥ 27), así
que el sesgo de la estimación apenas mueve la clase; y el input y el pipeline
usan el MISMO valor, como exige el generador de calor.

Uso:
    python climasafeai/llm/generar_dataset_frio.py \
        --num-frio 180 --output data/llm/train.jsonl

Genera train.jsonl + val.jsonl mixtos (los ~300 de calor actuales + los fríos
nuevos). El JSONL completo (≈ 0.6 MB) no se commitea: si supera 1 MB se regenera
con el comando de arriba cuando LLM-006 lo necesite. Para una demo pequeña:
    python climasafeai/llm/generar_dataset_frio.py --num-frio 30 \
        --output data/llm/demo_frio_mixto.jsonl
"""

from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd
from dotenv import load_dotenv

from climasafeai.llm import generar_dataset as gd

# ---------------------------------------------------------------------------
# Provincias frías
# ---------------------------------------------------------------------------

# El criterio exige al menos León, Soria, Lleida y Girona; se añaden otras de la
# meseta norte y del Pirineo para que el reparto no sea monotemático.
PROVINCIAS_FRIO = [
    "León", "Soria", "Lleida", "Girona",
    "Ávila", "Burgos", "Huesca", "Teruel", "Zamora", "Segovia",
]

# Coordenadas por provincia (nombre tal cual en el parquet, con acentos).
# `_get_province_coords` de weather_fetcher normaliza sin acentos y no resuelve
# "León" (cae a Madrid), así que se fijan aquí las que usamos. Los nombres con
# acento SÍ los resuelve `_EMBEDDED_DEMOGRAPHICS`, que es el que alimenta la
# INE del LSTM.
COORDS_PROVINCIA: dict[str, tuple[float, float]] = {
    "León": (42.5987, -5.5665),
    "Soria": (41.7636, -2.4650),
    "Lleida": (41.6148, 0.6266),
    "Girona": (41.9794, 2.8214),
    "Ávila": (40.6564, -4.6993),
    "Burgos": (42.3439, -3.6969),
    "Huesca": (42.1398, -0.4089),
    "Teruel": (40.3457, -1.1065),
    "Zamora": (41.5034, -5.7443),
    "Segovia": (40.9429, -4.1088),
}

# Días de frío: noviembre-febrero con la hora de riesgo del día por debajo de 5 °C.
MESES_INVIERNO = {11, 12, 1, 2}
UMBRAL_T2M_FRIO_C = 5.0
# Criterio 4: un ejemplo de frío solo entra si la media de la ventana < 10 °C.
UMBRAL_T_MEDIA_FRIO_C = 10.0

DEFAULT_PARQUET = "data/processed/dataset_frio_labeled.parquet"
DEFAULT_CALOR_TRAIN = "data/llm/train.jsonl"
DEFAULT_CALOR_VAL = "data/llm/val.jsonl"

# Columnas de etiqueta/fuga que el parquet etiquetado trae y que los modelos NO
# pueden ver como feature (clase_riesgo_frio es el TARGET del modelo frío y
# process_input no lo dropea para esa clase — solo lo dropea el split de
# entrenamiento). defunciones_* es fuga directa para ambas clases.
_COLS_ETIQUETA = ["clase_riesgo_frio", "clase_riesgo_frio_label", "defunciones_atrib_def_temp"]
# Columnas auxiliares que `generar_frio` añade para filtrar; tampoco son features.
_COLS_HELPER = ["_dt", "_mes", "_fecha_date"]
# Columnas brutas horarias que consume el pipeline (misma forma que
# fetch_hourly_forecast) + los índices ya calculados que necesita
# perfil_horario_desde_df ANTES de procesar (build_sequence_24h los recalcula).
_COLS_DF_HORA = ["datetime", "t2m_c", "rh", "wind_speed_kmh", "sp",
                 "heat_index_c", "wbgt_c", "wind_chill_c"]
CAMPOS_PARTE = ("t_media", "t_max", "rh", "viento_kmh", "uv")


# ---------------------------------------------------------------------------
# UV histórico (estimado, documentado)
# ---------------------------------------------------------------------------

# Caché por (lat, lon, fecha): el archivo de Open-Meteo no tiene cupo, pero un
# mismo día/provincia se reutiliza con varios perfiles.
_UV_CACHE: dict[tuple[float, float, str], pd.DataFrame | None] = {}


def _uv_estimado_historico(lat: float, lon: float, fecha: date) -> pd.DataFrame | None:
    """UV horario estimado del archivo Open-Meteo para `fecha`.

    El archivo de Open-Meteo no sirve uv_index (devuelve None) y OpenUV no da
    histórico en el plan actual (404 en /v1/history). Se estima de la radiación
    de onda corta (`shortwave_radiation`, W/m²) que SÍ sirve el archivo, con la
    regla de pulgar documentada UVI ≈ GHI/100 (guía WHO "Global Solar UV
    Index"). Devuelve un DataFrame(datetime, uv_index) con la misma forma que
    `_uv_horario` del generador de calor, para que `_uv_de_la_ventana` y el
    input usen exactamente el mismo valor. Sin dato → None (el ejemplo se
    descarta por parte incompleto, como hace generar_dataset).
    """
    key = (lat, lon, fecha.isoformat())
    if key not in _UV_CACHE:
        _UV_CACHE[key] = None
        try:
            import requests

            from climasafeai.data.weather_fetcher import OPENMETEO_ARCHIVE

            data = requests.get(
                OPENMETEO_ARCHIVE,
                params={
                    "latitude": lat,
                    "longitude": lon,
                    "start_date": fecha.isoformat(),
                    "end_date": fecha.isoformat(),
                    "hourly": "shortwave_radiation",
                    "timezone": "auto",
                },
                timeout=30,
            ).json()
            hourly = data.get("hourly", {})
            sw = hourly.get("shortwave_radiation")
            if hourly.get("time") and sw:
                vals = [0.0 if v is None else max(float(v), 0.0) for v in sw]
                _UV_CACHE[key] = pd.DataFrame({
                    "datetime": pd.to_datetime(hourly["time"]),
                    "uv_index": [round(v / 100.0, 1) for v in vals],
                })
        except Exception:
            _UV_CACHE[key] = None
    return _UV_CACHE[key]


# ---------------------------------------------------------------------------
# Weather desde el parquet
# ---------------------------------------------------------------------------


def _construir_weather(parquet: pd.DataFrame, provincia: str, fecha: date) -> dict | None:
    """El dict `weather` que consume `predict_ensemble`, desde la fila del parquet.

    Mismas claves que monta `fetch_weather_data` para que el pipeline no note la
    diferencia: `df_features` (la fila diaria del parquet, sin etiquetas/fugas),
    `df_hora` (columnas brutas horarias + índices), `current` (la fila del día),
    `uv_index` + `uv_horario` (estimados) y `target_date`. Sin la fila o sin UV
    devuelve None: no hay weather que predecir.
    """
    mask = (parquet["provincia"] == provincia) & (parquet["_fecha_date"] == fecha)
    filas = parquet.loc[mask]
    if filas.empty:
        return None
    fila = filas.iloc[0]

    lat, lon = COORDS_PROVINCIA[provincia]
    uv_horario = _uv_estimado_historico(lat, lon, fecha)
    if uv_horario is None or uv_horario.empty:
        return None

    df_features = fila.to_frame().T.drop(
        columns=[c for c in _COLS_ETIQUETA + _COLS_HELPER if c in fila.index],
        errors="ignore",
    )
    df_hora = fila.to_frame().T[[c for c in _COLS_DF_HORA if c in fila.index]].copy()

    return {
        "lat": lat,
        "lon": lon,
        "current": {
            "t2m_c": float(fila["t2m_c"]),
            "rh": float(fila["rh"]),
            "wind_speed_kmh": float(fila["wind_speed_kmh"]),
            "sp": float(fila["sp"]),
        },
        "df_hora": df_hora,
        "df_features": df_features,
        "uv_index": float(uv_horario["uv_index"].max()),
        "uv_horario": uv_horario,
        "target_date": fecha.isoformat(),
    }


def _seleccionar_dias_frios(parquet: pd.DataFrame, num_dias: int, rng: random.Random) -> list[tuple[str, date]]:
    """Días invernales con t2m_c < 5 °C, repartidos entre las provincias.

    Hace round-robin entre provincias (las cuatro exigidas primero) para que el
    dataset cubra varias, no un puñado de días de una sola. Devuelve pares
    (provincia, fecha) únicos.
    """
    candidatas = parquet.loc[
        parquet["provincia"].isin(PROVINCIAS_FRIO)
        & parquet["_mes"].isin(MESES_INVIERNO)
        & (parquet["t2m_c"] < UMBRAL_T2M_FRIO_C),
        ["provincia", "_fecha_date"],
    ].drop_duplicates()

    agrupado = {p: g for p, g in candidatas.groupby("provincia")}
    orden = [p for p in PROVINCIAS_FRIO if p in agrupado]
    for p in orden:
        idx = list(agrupado[p].index)
        rng.shuffle(idx)
        agrupado[p] = agrupado[p].loc[idx].reset_index(drop=True)

    elegidos: list[tuple[str, date]] = []
    max_len = max(len(g) for g in agrupado.values()) if agrupado else 0
    for i in range(max_len):
        for p in orden:
            if len(elegidos) >= num_dias:
                return elegidos
            if i < len(agrupado[p]):
                elegidos.append((p, agrupado[p].iloc[i]["_fecha_date"]))
    return elegidos


# ---------------------------------------------------------------------------
# Generación de ejemplos de frío
# ---------------------------------------------------------------------------


def generar_frio(
    num_frio: int,
    parquet_path: str = DEFAULT_PARQUET,
    num_dias: int = 0,
    seed: int = gd.RANDOM_SEED,
    equilibrar: bool = True,
) -> list[dict]:
    """Genera `num_frio` ejemplos Alpaca de frío desde el parquet.

    Reutiliza del generador de calor los perfiles sintéticos
    (`gd.generar_perfiles`), el formateo (`gd.formatear_input/respuesta`) y el
    criterio de parte completo. Por día se cogen ~4 perfiles distintos (cada
    día/provincia da varios ejemplos, como el generador de calor con sus
    escenarios). Solo entran ejemplos con la media de la ventana < 10 °C
    (criterio 4); el resto se descarta y se cuenta.
    """
    parquet = pd.read_parquet(parquet_path)
    parquet["_dt"] = pd.to_datetime(parquet["datetime"])
    parquet["_mes"] = parquet["_dt"].dt.month
    parquet["_fecha_date"] = parquet["_dt"].dt.date

    if num_dias <= 0:
        num_dias = max(4, (num_frio + 3) // 4)
    rng = random.Random(seed)
    dias = _seleccionar_dias_frios(parquet, num_dias, rng)
    if not dias:
        print("  AVISO: no hay días invernales con t2m_c < 5 °C en las provincias configuradas.")
        return []

    # Equilibrio por clase: el generador de calor usa num/3·1.35 porque sus tres
    # clases aparecen. Aquí el canal frío solo da dos de forma natural (en un día
    # de frío real ni un perfil joven sano llega a SEGURO — el modelo considera
    # el frío riesgoso), así que el cupo se reparte entre dos con holgura; si no,
    # el cupo de tres caparía el conjunto en 2/3 del objetivo (~162 de 180).
    cupo = max(1, round(num_frio / 2 * 1.15)) if equilibrar else num_frio
    por_clase: dict[str, int] = {}
    dataset: list[dict] = []
    incompletos = 0
    no_frio = 0
    descartados = 0
    fallidos: list[str] = []

    # Pool de perfiles sintéticos variados: `generar_perfiles` re-sembra la
    # aleatoriedad en cada llamada, así que se pide el pool de una vez y se
    # reparte entre los días (cada uno con su lat/lon/provincia).
    pool_perfiles = gd.generar_perfiles(max(8, num_dias * 4))
    n_perfiles_dia = max(1, pool_perfiles.__len__() // len(dias)) if dias else 0

    for i, (provincia, fecha) in enumerate(dias):
        if len(dataset) >= num_frio:
            break
        weather = _construir_weather(parquet, provincia, fecha)
        if weather is None:
            incompletos += 1
            continue
        for j in range(n_perfiles_dia):
            if len(dataset) >= num_frio:
                break
            perfil = dict(pool_perfiles[(i * n_perfiles_dia + j) % len(pool_perfiles)])
            perfil["provincia"] = provincia
            perfil["lat"], perfil["lon"] = COORDS_PROVINCIA[provincia]
            try:
                riesgo = gd.predecir(perfil, weather=weather)
            except Exception as exc:
                fallidos.append(f"{provincia} {fecha}: {type(exc).__name__}: {exc}")
                continue
            clima = riesgo.get("clima") or {}
            if not all(clima.get(c) is not None for c in CAMPOS_PARTE):
                incompletos += 1
                continue
            if clima.get("t_media") is None or clima["t_media"] >= UMBRAL_T_MEDIA_FRIO_C:
                no_frio += 1
                continue
            clase = riesgo.get("clase", "DESCONOCIDO")
            if equilibrar and por_clase.get(clase, 0) >= cupo:
                descartados += 1
                continue
            por_clase[clase] = por_clase.get(clase, 0) + 1
            dataset.append({
                "instruction": gd.INSTRUCCION,
                "input": gd.formatear_input(perfil, clima),
                "output": gd.formatear_respuesta(perfil, riesgo),
                # Procedencia verificable en el JSONL (criterio 4): la fecha y la
                # provincia del día real del parquet. fine_tune solo lee
                # instruction/input/output; estas claves no entran al prompt.
                "fecha": fecha.isoformat(),
                "provincia": provincia,
            })

    reparto = " · ".join(f"{k} {v}" for k, v in sorted(por_clase.items()))
    print(f"  Frío por clase: {reparto}  ({descartados} descartados por cupo)")
    print(f"  {no_frio} perfiles con ventana >= {UMBRAL_T_MEDIA_FRIO_C} °C (no son frío, descartados)")
    if incompletos:
        print(f"  {incompletos} días/perfiles sin parte completo (sin media/máx/humedad/viento/UV)")
    if fallidos:
        print(f"  {len(fallidos)} perfiles saltados por error de predicción:")
        for linea in fallidos[:5]:
            print(f"    - {linea}")
        if len(fallidos) > 5:
            print(f"    ... y {len(fallidos) - 5} más")
    if len(dataset) < num_frio:
        print(f"  AVISO: solo {len(dataset)} de {num_frio} ejemplos de frío. "
              "Sube --num-frio/--num-dias o revisa los errores de arriba.")

    return dataset


# ---------------------------------------------------------------------------
# Mezcla calor + frío y reparto train/val
# ---------------------------------------------------------------------------


def _reparto_dataset(dataset: list[dict]) -> tuple[Counter, Counter]:
    """Reparto por canal (t_media < 10 vs >= 10) y por clase, desde el JSONL.

    El canal se lee del input ("X °C de media") y la clase del output
    ("RIESGO: X"), sin depender de metadatos que el JSONL no guarda.
    """
    canales: Counter[str] = Counter()
    clases: Counter[str] = Counter()
    for ex in dataset:
        m = re.search(r"([\d.]+) °C de media", ex.get("input", ""))
        t = float(m.group(1)) if m else None
        canales["frio" if (t is not None and t < UMBRAL_T_MEDIA_FRIO_C) else "calor"] += 1
        m2 = re.search(r"RIESGO: (\w+)", ex.get("output", ""))
        clases[m2.group(1) if m2 else "DESCONOCIDO"] += 1
    return canales, clases


def _cargar_calor(paths: list[str]) -> list[dict]:
    """Los ejemplos de calor actuales (train.jsonl + val.jsonl, 100 % calor)."""
    ejemplos: list[dict] = []
    for p in paths:
        if Path(p).exists():
            with open(p, encoding="utf-8") as f:
                ejemplos += [json.loads(line) for line in f if line.strip()]
    return ejemplos


def generar_dataset_mixto(
    num_frio: int,
    parquet_path: str = DEFAULT_PARQUET,
    calor_paths: list[str] | None = None,
    num_dias: int = 0,
    seed: int = gd.RANDOM_SEED,
    val_split: float = 0.1,
) -> tuple[list[dict], list[dict], dict[str, int]]:
    """Mezcla los ~300 de calor actuales con `num_frio` de frío y parte train/val.

    Devuelve (train, val, conteo) con train/val mixtos por construcción: se
    baraja el conjunto completo (calor + frío) con semilla y se parte. `conteo`
    lleva el número de ejemplos de cada fuente.
    """
    if calor_paths is None:
        calor_paths = [DEFAULT_CALOR_TRAIN, DEFAULT_CALOR_VAL]
    calor = _cargar_calor(calor_paths)
    frio = generar_frio(num_frio, parquet_path=parquet_path, num_dias=num_dias, seed=seed)

    dataset = calor + frio
    random.seed(seed)
    random.shuffle(dataset)
    val_n = int(len(dataset) * val_split)
    return dataset[val_n:], dataset[:val_n], {"calor": len(calor), "frio": len(frio)}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generar dataset mixto calor+frío para fine-tuning")
    p.add_argument("-o", "--output", default=DEFAULT_CALOR_TRAIN,
                   help="Ruta del JSONL de train de salida (el val se escribe como "
                        "val.jsonl al lado del output, misma convención que el generador de calor)")
    p.add_argument("--num-frio", type=int, default=180,
                   help="Número de ejemplos de frío a generar (default: 180)")
    p.add_argument("--num-dias", type=int, default=0,
                   help="Días distintos a muestrear del parquet (default: ceil(num_frio/4))")
    p.add_argument("--parquet", default=DEFAULT_PARQUET,
                   help="Parquet con los días de frío etiquetados")
    p.add_argument("--calor-train", default=DEFAULT_CALOR_TRAIN,
                   help="JSONL con los ejemplos de calor actuales (train)")
    p.add_argument("--calor-val", default=DEFAULT_CALOR_VAL,
                   help="JSONL con los ejemplos de calor actuales (val)")
    p.add_argument("--val-split", type=float, default=0.1,
                   help="Fracción para validación (default: 0.1)")
    p.add_argument("--seed", type=int, default=gd.RANDOM_SEED,
                   help="Semilla aleatoria")
    p.add_argument("--sin-equilibrar", action="store_true",
                   help="No equilibrar las clases de frío")
    return p.parse_args(argv)


def _guardar(path: Path, ejemplos: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for ex in ejemplos:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")


def main() -> None:
    args = parse_args()

    # El UV estimado no necesita clave, pero load_dotenv no estorba y mantiene
    # el mismo arranque que el generador de calor.
    load_dotenv()

    print(f"Generando {args.num_frio} ejemplos de frío desde {args.parquet}...")
    train, val, conteo = generar_dataset_mixto(
        num_frio=args.num_frio,
        parquet_path=args.parquet,
        calor_paths=[args.calor_train, args.calor_val],
        num_dias=args.num_dias,
        seed=args.seed,
        val_split=args.val_split,
    )

    output_path = Path(args.output)
    val_path = output_path.with_name("val.jsonl")
    _guardar(output_path, train)
    _guardar(val_path, val)

    print(f"\n  Mezcla: {conteo['calor']} calor + {conteo['frio']} frío")
    print(f"  Train: {len(train)} ejemplos → {output_path}")
    print(f"  Val:   {len(val)} ejemplos → {val_path}")
    for nombre, split in (("Train", train), ("Val", val)):
        canales, clases = _reparto_dataset(split)
        canal = " · ".join(f"{k} {v}" for k, v in sorted(canales.items()))
        clase = " · ".join(f"{k} {v}" for k, v in sorted(clases.items()))
        print(f"  {nombre} — canal: {canal}")
        print(f"  {nombre} — clase: {clase}")

    if train:
        ex = train[0]
        print(f"\nEjemplo:\n  Input: {ex['input'][:150]}...\n  Output: {ex['output'][:200]}...")


if __name__ == "__main__":
    main()
