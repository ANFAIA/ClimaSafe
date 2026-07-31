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


def predecir(perfil: dict) -> dict:
    """Ejecuta la predicción REAL de ClimaSafeAI. Si no puede, revienta.

    Antes esto tenía un `except ImportError` que caía en `_predecir_fake`, y como
    los tres imports que hacía no existían en ese módulo, el dataset ENTERO salía
    de la simulación: un riesgo que solo dependía de la edad. Sin error, sin aviso,
    y con pinta de bueno — 150 ejemplos para enseñarle al modelo una función de
    riesgo inventada. Por eso ahora no hay red: si el pipeline no va, se para la
    generación y se dice por qué.
    """
    from climasafeai.models.ensemble import predict_ensemble

    resultado = predict_ensemble(
        lat=perfil["lat"],
        lon=perfil["lon"],
        provincia=perfil["provincia"],
        perfil=_perfil_para_modelo(perfil),
    )
    calor = (resultado.get("perfil") or {}).get("calor") or {}
    return {
        "clase": resultado.get("clase_final_label", "DESCONOCIDO"),
        "indice_personalizado": calor.get("prob_personalizada", 0.0),
        "indice_base": calor.get("prob_poblacional", 0.0),
        "factor_total": calor.get("factor_total", 1.0),
        "producto_bruto": calor.get("producto_bruto"),
        "capado": bool(calor.get("capado")),
        "factores": calor.get("factores") or [],
        "recomendaciones": resultado.get("recomendaciones") or [],
        "perfil": perfil,
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


def formatear_input(perfil: dict) -> str:
    """Convierte el perfil a texto legible para el prompt."""
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
        clase = riesgo.get("clase", "DESCONOCIDO")
        if equilibrar and por_clase.get(clase, 0) >= cupo:
            descartados += 1
            continue
        por_clase[clase] = por_clase.get(clase, 0) + 1
        dataset.append({
            "instruction": INSTRUCCION,
            "input": formatear_input(perfil),
            "output": formatear_respuesta(perfil, riesgo),
        })

    if equilibrar:
        reparto = " · ".join(f"{k} {v}" for k, v in sorted(por_clase.items()))
        print(f"  Reparto por clase: {reparto}  ({descartados} descartados por cupo)")
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
