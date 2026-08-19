#!/usr/bin/env python3
"""RAG-006 — Compara modelos de embeddings y solapamiento contra la línea base.

Línea base RAG-004 (k=5, all-MiniLM-L6-v2, sin solapamiento):
    factores   recall@5 = 0.780 (25 preguntas)
    documentos recall@5 = 0.325 (39 preguntas)

Mide, sobre data/rag/eval_set.json, el recall@k por canal de cuatro estados del
índice (en este orden):

    1. línea base          — el índice activo tal cual (antes de tocar nada).
    2. + solapamiento      — reindexa el índice activo con el solapamiento
                             indicado (--overlap): aísla el efecto del chunking.
    3. + modelo alternativo (sin solape) — indexa RAG_EMBEDDER_ALT en sus tablas
                             propias con overlap=0: aísla el efecto del modelo.
    4. + modelo alternativo (con solape) — lo mismo con el solapamiento.

Cada estado deja su número por escrito; la decisión de qué queda como índice
activo (modelo por defecto y si el solapamiento se mantiene) la toma quien lo
ejecuta, por los números, y se registra en
documentacion/rag_006_comparativa_embeddings.md.

El script MUTA la BD que se le pasa (--db): reindexa el índice activo y crea
las tablas del modelo alternativo. Pásale una copia si no quieres tocar la real.

Uso:
  .venv/bin/python scripts/comparar_modelos_rag.py [--k 5]
                        [--db data/climasafe.db] [--set data/rag/eval_set.json]
                        [--overlap 200] [--json]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from climasafeai.db.rag import RAG, RAG_EMBEDDER_ALT, RAG_EMBEDDER_DEFAULT

import evaluar_rag

RAIZ = Path(__file__).resolve().parents[1]
SET_POR_DEFECTO = RAIZ / "data" / "rag" / "eval_set.json"
DB_POR_DEFECTO = RAIZ / "data" / "climasafe.db"

CANALES = ("factores", "documentos")


def medir_canal(search_fn, preguntas: list[dict], canal: str, k: int) -> dict | None:
    """recall@k/precision@k de un canal para una función de búsqueda dada.

    ``search_fn(pregunta, k)`` devuelve el top-k recuperado para esa query.
    Reutiliza las métricas de evaluar_rag (misma definición que RAG-004) y
    agrega con la misma regla: media sobre preguntas con esperados en el canal.
    """
    con_esperados = []
    for p in preguntas:
        recuperados = search_fn(p["pregunta"], k=k)
        m = evaluar_rag.calcular_metricas(p[f"{canal}_esperados"], recuperados, k, canal)
        if m is not None:
            con_esperados.append(m)
    if not con_esperados:
        return None
    return {
        "n_preguntas": len(con_esperados),
        "recall": sum(m["recall"] for m in con_esperados) / len(con_esperados),
        "precision": sum(m["precision"] for m in con_esperados) / len(con_esperados),
    }


def _celda(m: dict | None) -> str:
    return "-" if m is None else f"{m['recall']:.3f}"


def formatear_tabla(filas: list[dict], k: int) -> str:
    lineas = [
        "=== RAG-006 · comparativa de modelos y solapamiento (retrieval, sin LLM) ===",
        f"k: {k}",
        f"{'config':<34}{'factores recall@k':>20}{'documentos recall@k':>20}",
    ]
    for f in filas:
        lineas.append(
            f"{f['nombre']:<34}{_celda(f['metricas']['factores']):>20}"
            f"{_celda(f['metricas']['documentos']):>20}"
        )
    return "\n".join(lineas)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compara modelos de embeddings y solapamiento contra la línea base RAG-004."
    )
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--db", type=Path, default=DB_POR_DEFECTO)
    parser.add_argument("--set", type=Path, default=SET_POR_DEFECTO)
    parser.add_argument("--overlap", type=int, default=200, help="solapamiento en caracteres")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    if args.k < 1:
        sys.exit("ERROR: k debe ser >= 1")
    if not args.db.exists():
        sys.exit(f"ERROR: no existe la BD {args.db}")
    preguntas = evaluar_rag.cargar_set(args.set)
    rag = RAG(args.db)

    filas = []

    # 1. Línea base: el índice activo tal cual, ANTES de tocar nada (la BD
    #    original de RAG-004 ya está indexada sin solapamiento y con el default).
    base = {
        "factores": medir_canal(rag.search_factores, preguntas, "factores", args.k),
        "documentos": medir_canal(rag.search_documentos, preguntas, "documentos", args.k),
    }
    filas.append(
        {
            "nombre": f"1. linea base ({RAG_EMBEDDER_DEFAULT}, sin solape)",
            "modelo": RAG_EMBEDDER_DEFAULT,
            "overlap": 0,
            "metricas": base,
        }
    )

    # Asegura el esquema (columnas hash/modelo y tablas por modelo alternativo).
    rag.initialize()

    # 2. + solapamiento: reindexa el índice activo (mismo modelo) con overlap.
    rag.resync_documentos(overlap=args.overlap)
    con_solape = {
        "factores": medir_canal(rag.search_factores, preguntas, "factores", args.k),
        "documentos": medir_canal(rag.search_documentos, preguntas, "documentos", args.k),
    }
    filas.append(
        {
            "nombre": f"2. + solape {args.overlap} ({RAG_EMBEDDER_DEFAULT})",
            "modelo": RAG_EMBEDDER_DEFAULT,
            "overlap": args.overlap,
            "metricas": con_solape,
        }
    )

    # 3. + modelo alternativo SIN solapamiento: aísla el efecto del modelo.
    rag.resync_factores_modelo(RAG_EMBEDDER_ALT)
    rag.resync_documentos_modelo(RAG_EMBEDDER_ALT, overlap=0)
    alt_sin = {
        "factores": medir_canal(
            lambda q, k=args.k: rag.search_factores_modelo(q, RAG_EMBEDDER_ALT, k),
            preguntas,
            "factores",
            args.k,
        ),
        "documentos": medir_canal(
            lambda q, k=args.k: rag.search_documentos_modelo(q, RAG_EMBEDDER_ALT, k),
            preguntas,
            "documentos",
            args.k,
        ),
    }
    filas.append(
        {
            "nombre": f"3. + modelo alt ({RAG_EMBEDDER_ALT}, sin solape)",
            "modelo": RAG_EMBEDDER_ALT,
            "overlap": 0,
            "metricas": alt_sin,
        }
    )

    # 4. + modelo alternativo CON solapamiento.
    rag.resync_documentos_modelo(RAG_EMBEDDER_ALT, overlap=args.overlap)
    alt_con = {
        "factores": medir_canal(
            lambda q, k=args.k: rag.search_factores_modelo(q, RAG_EMBEDDER_ALT, k),
            preguntas,
            "factores",
            args.k,
        ),
        "documentos": medir_canal(
            lambda q, k=args.k: rag.search_documentos_modelo(q, RAG_EMBEDDER_ALT, k),
            preguntas,
            "documentos",
            args.k,
        ),
    }
    filas.append(
        {
            "nombre": f"4. + modelo alt ({RAG_EMBEDDER_ALT}, solape {args.overlap})",
            "modelo": RAG_EMBEDDER_ALT,
            "overlap": args.overlap,
            "metricas": alt_con,
        }
    )

    if args.json:
        json.dump(
            {
                "k": args.k,
                "db": str(args.db),
                "set": str(args.set),
                "overlap": args.overlap,
                "linea_base_rag004": {
                    "factores": {"recall": 0.780},
                    "documentos": {"recall": 0.325},
                },
                "configuraciones": filas,
            },
            sys.stdout,
            ensure_ascii=False,
            indent=2,
        )
        return

    print(formatear_tabla(filas, args.k))
    print()
    print("Línea base RAG-004 (referencia): factores recall@5 = 0.780 · documentos 0.325")
    print("Decisión: en documentacion/rag_006_comparativa_embeddings.md")


if __name__ == "__main__":
    main()
