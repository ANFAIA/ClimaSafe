#! /usr/bin/env python
"""
Benchmark de modelos locales sobre data/llm/val.jsonl (LLM-003).

Mide sin juez LLM. Un modelo puntuando a otro mete su propio sesgo y no es
reproducible; aquí la respuesta de referencia sale del pipeline determinista,
así que se puede comparar carácter por carácter y número por número.

Métricas:
  clase      ¿acierta SEGURO / PRECAUCION / PELIGRO?
  formato    ¿trae las líneas que el bot espera (RIESGO, índice, factor)?
  inventadas ¿mete cifras que no están ni en la pregunta ni en la referencia?
  err_indice desviación absoluta media del índice personalizado
  latencia   segundos por respuesta (mediana y p95)

Uso:
    uv run python climasafeai/llm/benchmark.py --modelos ollama/qwen2.5:1.5b
    uv run python climasafeai/llm/benchmark.py --modelos ollama/qwen2.5:1.5b,ollama/gemma3:4b
    uv run python climasafeai/llm/benchmark.py --limite 20 --json informe.json
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import time
from pathlib import Path
from typing import Any

VAL_POR_DEFECTO = Path("data/llm/val.jsonl")

# El prompt tiene que pedir el mismo formato que el dataset enseña. Si se le pide
# "responde libremente" no se puede comparar nada y el benchmark mide redacción.
SYSTEM = """Eres ClimaSafeAI. Respondes el riesgo térmico de una persona con este formato EXACTO:

RIESGO: <SEGURO|PRECAUCION|PELIGRO>

Índice personalizado: <0.00>
Índice poblacional: <0.00>
Factor total aplicado: ×<0.00>

Factores activados:
- <factor>

Recomendaciones:
- <recomendación>

No inventes cifras que no estén en los datos que te dan."""

CLASES = ("SEGURO", "PRECAUCION", "PELIGRO")

_RE_CLASE = re.compile(r"RIESGO:\s*([A-ZÁÉÍÓÚÑ]+)", re.I)
_RE_INDICE = re.compile(r"[ÍI]ndice\s+personalizado:\s*([0-9]+[.,][0-9]+)", re.I)
_RE_NUMERO = re.compile(r"\d+(?:[.,]\d+)?")


def _clase(texto: str) -> str | None:
    m = _RE_CLASE.search(texto or "")
    if not m:
        return None
    clase = m.group(1).upper().replace("Ó", "O").replace("Á", "A")
    return clase if clase in CLASES else None


def _indice(texto: str) -> float | None:
    m = _RE_INDICE.search(texto or "")
    if not m:
        return None
    try:
        return float(m.group(1).replace(",", "."))
    except ValueError:
        return None


def _numeros(texto: str) -> set[str]:
    """Números normalizados: '0,240' y '0.24' cuentan como el mismo."""
    fuera = set()
    for n in _RE_NUMERO.findall(texto or ""):
        n = n.replace(",", ".")
        try:
            fuera.add(f"{float(n):g}")
        except ValueError:
            continue
    return fuera


def _formato_ok(texto: str) -> bool:
    t = (texto or "").lower()
    return all(marca in t for marca in ("riesgo:", "índice personalizado", "factor total"))


def evaluar_respuesta(respuesta: str, ejemplo: dict) -> dict[str, Any]:
    """Compara una respuesta con su referencia. Todo determinista."""
    referencia = ejemplo["output"]
    esperada = _clase(referencia)
    obtenida = _clase(respuesta)

    # Una cifra es "inventada" si no aparece ni en lo que se le dio ni en la
    # respuesta correcta. Es la métrica que más importa en un asistente de salud:
    # un modelo que se inventa un índice suena igual de convincente que uno que no.
    permitidas = _numeros(ejemplo["input"]) | _numeros(referencia)
    inventadas = _numeros(respuesta) - permitidas

    i_real, i_pred = _indice(referencia), _indice(respuesta)
    return {
        "clase_esperada": esperada,
        "clase_obtenida": obtenida,
        "clase_ok": bool(obtenida and obtenida == esperada),
        "formato_ok": _formato_ok(respuesta),
        "n_inventadas": len(inventadas),
        "inventadas": sorted(inventadas)[:6],
        "err_indice": abs(i_real - i_pred) if (i_real is not None and i_pred is not None) else None,
    }


def _tamano_ollama(modelo: str) -> str:
    """GB que ocupa el modelo en Ollama, para la columna de portabilidad."""
    if not modelo.startswith("ollama/"):
        return "-"
    try:
        import requests

        r = requests.get("http://localhost:11434/api/tags", timeout=5)
        corto = modelo.split("/", 1)[1]
        for m in r.json().get("models", []):
            if m.get("name") == corto:
                return f"{m.get('size', 0) / 1e9:.2f} GB"
    except Exception:
        pass
    return "?"


def evaluar_modelo(modelo: str, ejemplos: list[dict], verbose: bool = True) -> dict[str, Any]:
    from climasafeai.llm.rag_qwen import LLMConfig, _chat_litellm

    config = LLMConfig(model=modelo)
    filas, latencias, fallos = [], [], 0

    for i, ej in enumerate(ejemplos, 1):
        mensajes = [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": f"{ej['instruction']}\n\n{ej['input']}"},
        ]
        t0 = time.perf_counter()
        respuesta = _chat_litellm(mensajes, config)
        latencias.append(time.perf_counter() - t0)

        if not respuesta:
            fallos += 1
            continue
        filas.append(evaluar_respuesta(respuesta, ej))
        if verbose and i % 10 == 0:
            aciertos = sum(f["clase_ok"] for f in filas)
            print(f"    {i}/{len(ejemplos)} · clase {aciertos}/{len(filas)} · "
                  f"{statistics.median(latencias):.1f}s/resp", flush=True)

    n = len(filas)
    if n == 0:
        return {"modelo": modelo, "n": 0, "fallos": fallos,
                "error": "ninguna respuesta utilizable"}

    errores = [f["err_indice"] for f in filas if f["err_indice"] is not None]
    return {
        "modelo": modelo,
        "tamano": _tamano_ollama(modelo),
        "n": n,
        "fallos": fallos,
        "clase_acc": sum(f["clase_ok"] for f in filas) / n,
        "formato_acc": sum(f["formato_ok"] for f in filas) / n,
        "pct_con_inventadas": sum(f["n_inventadas"] > 0 for f in filas) / n,
        "inventadas_media": statistics.mean(f["n_inventadas"] for f in filas),
        "err_indice_medio": statistics.mean(errores) if errores else None,
        "sin_indice": sum(f["err_indice"] is None for f in filas) / n,
        "latencia_p50": statistics.median(latencias),
        "latencia_p95": sorted(latencias)[int(len(latencias) * 0.95) - 1] if latencias else None,
        "_filas": filas,
    }


def imprimir_tabla(informes: list[dict]) -> None:
    cab = (f"{'modelo':<26} {'tam':>8} {'clase':>7} {'formato':>8} "
           f"{'inventa':>8} {'err_idx':>8} {'p50 s':>7} {'p95 s':>7}")
    print("\n" + cab)
    print("-" * len(cab))
    for r in informes:
        if r.get("error"):
            print(f"{r['modelo']:<26} {'—':>8}  {r['error']}")
            continue
        err = f"{r['err_indice_medio']:.3f}" if r["err_indice_medio"] is not None else "—"
        print(f"{r['modelo']:<26} {r['tamano']:>8} {r['clase_acc']:>6.0%} "
              f"{r['formato_acc']:>7.0%} {r['pct_con_inventadas']:>7.0%} "
              f"{err:>8} {r['latencia_p50']:>7.2f} {r['latencia_p95'] or 0:>7.2f}")
    print("\nclase   = acierta SEGURO/PRECAUCION/PELIGRO")
    print("formato = trae las líneas que el bot espera")
    print("inventa = % de respuestas con al menos una cifra que no estaba en los datos")
    print("err_idx = desviación media del índice personalizado")


def cargar_val(ruta: Path, limite: int | None) -> list[dict]:
    ejemplos = [json.loads(linea) for linea in ruta.read_text(encoding="utf-8").splitlines() if linea.strip()]
    return ejemplos[:limite] if limite else ejemplos


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Benchmark de modelos locales sobre val.jsonl")
    p.add_argument("--modelos", default="ollama/qwen2.5:1.5b",
                   help="Modelos LiteLLM separados por comas")
    p.add_argument("--val", type=Path, default=VAL_POR_DEFECTO, help="JSONL de validación")
    p.add_argument("--limite", type=int, default=None, help="Usar solo los N primeros ejemplos")
    p.add_argument("--json", type=Path, default=None, help="Guardar el informe completo aquí")
    return p.parse_args(argv)


def main() -> None:
    args = parse_args()
    if not args.val.exists():
        raise SystemExit(f"No existe {args.val}. Genera el dataset con generar_dataset.py.")

    ejemplos = cargar_val(args.val, args.limite)
    modelos = [m.strip() for m in args.modelos.split(",") if m.strip()]
    print(f"{len(ejemplos)} ejemplos de {args.val} · {len(modelos)} modelo(s)")

    def _guardar(informes: list[dict]) -> None:
        if not args.json:
            return
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(
            json.dumps({"val": str(args.val), "n": len(ejemplos), "informes": informes},
                       ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    # Se guarda DESPUÉS DE CADA MODELO, no al final. Una tanda de tres modelos
    # sobre 100 ejemplos dura horas: si se corta a la mitad —timeout, batería,
    # un ctrl-c— guardar solo al final tira a la basura todo lo ya medido.
    informes = []
    for modelo in modelos:
        print(f"\n▶  {modelo}", flush=True)
        informes.append(evaluar_modelo(modelo, ejemplos))
        _guardar(informes)
        imprimir_tabla(informes)

    if args.json:
        print(f"\nInforme completo → {args.json}")


if __name__ == "__main__":
    main()
