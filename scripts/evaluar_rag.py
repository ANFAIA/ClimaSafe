#!/usr/bin/env python3
"""RAG-004 — Evalúa el retrieval del RAG con recall@k y precision@k.

Lee el set etiquetado (data/rag/eval_set.json) y, para cada pregunta,
recupera el top-k de factores y de documentos con el RAG real (sqlite-vec
sobre data/climasafe.db) y compara contra lo etiquetado. Solo MIDE el estado
actual del retrieval: no toca el motor ni genera respuestas con LLM.

Definiciones (por canal — factores y documentos se evalúan por separado):

  recall@k    = |recuperados ∩ esperados| / |esperados|
    Fracción de ítems etiquetados que aparecen en el top-k. 1.0 = se
    recuperan todos los esperados; 0.0 = ninguno. Solo se calcula si la
    pregunta tiene esperados en ese canal (si no, la celda es "-" y la
    pregunta no cuenta en el agregado de ese canal).

  precision@k = |recuperados ∩ esperados| / k
    Fracción de los k slots del top-k ocupados por ítems etiquetados.
    Solo se calcula si la pregunta tiene esperados en ese canal.

  Agregado = media de recall (y de precision) por canal sobre las
  preguntas que tienen esperados en ese canal.

Claves de comparación:
  - factores:   "tipo/clave" (p. ej. "calor/diabetes"). El mismo clave
    existe en calor y en frío (cardiovascular, sexo_mujer, encamado...),
    así que el tipo forma parte de la clave para no contar un falso
    positivo entre tipos.
  - documentos: ruta relativa al repo (p. ej. "documentacion/riesgo/
    formulas_deterministas.md"). La BD guarda rutas absolutas; se
    normalizan recortando desde el componente "documentacion/" (si no
    aparece, se usa el nombre de fichero).

Uso:
  .venv/bin/python scripts/evaluar_rag.py [--k 5] [--db data/climasafe.db]
                        [--set data/rag/eval_set.json] [--json]

Reproducible y local: el embedder (all-MiniLM-L6-v2) está cacheado en
~/.cache/huggingface; si no estuviera, descárgalo con
  python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"
La BD data/climasafe.db se genera con el propio RAG (RAG.initialize/sync) y
no se versiona; si falta, el script avisa y sale sin tocar nada.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from climasafeai.db.rag import RAG

RAIZ = Path(__file__).resolve().parents[1]
SET_POR_DEFECTO = RAIZ / "data" / "rag" / "eval_set.json"
DB_POR_DEFECTO = RAIZ / "data" / "climasafe.db"

CANALES = ("factores", "documentos")


# ── Carga y validación del set ────────────────────────────────────────────


def cargar_set(ruta: str | Path) -> list[dict]:
    """Lee el JSON del set y devuelve la lista de preguntas."""
    ruta = Path(ruta)
    if not ruta.exists():
        sys.exit(f"ERROR: no existe el set de evaluación: {ruta}")
    with ruta.open(encoding="utf-8") as fh:
        doc = json.load(fh)
    preguntas = doc.get("preguntas") if isinstance(doc, dict) else None
    if not isinstance(preguntas, list) or not preguntas:
        sys.exit(f"ERROR: {ruta} no tiene una lista 'preguntas' no vacía")
    return preguntas


# ── Claves de comparación ────────────────────────────────────────────────


def normalizar_ruta(ruta: str) -> str:
    """Ruta absoluta de la BD → relativa al repo.

    La BD guarda rutas absolutas (p. ej.
    /home/.../documentacion/papers/foo.md); el set etiqueta rutas relativas
    estables (documentacion/papers/foo.md). Se recorta desde el componente
    "documentacion/"; si no aparece, queda el nombre de fichero."""
    idx = ruta.find("documentacion/")
    return ruta[idx:] if idx != -1 else Path(ruta).name


def clave_recuperado(res: dict, canal: str) -> str:
    """Clave de comparación de un ítem recuperado en un canal."""
    if canal == "factores":
        return f"{res['tipo']}/{res['clave']}"
    return normalizar_ruta(res["ruta"])


def calcular_metricas(
    esperados: list[str], recuperados: list[dict], k: int, canal: str
) -> dict | None:
    """recall@k y precision@k de una pregunta en un canal.

    Devuelve None si la pregunta no tiene esperados en ese canal: sin
    etiqueta no hay nada que medir y la pregunta no cuenta en el agregado.
    """
    if not esperados:
        return None
    recuperadas = {clave_recuperado(r, canal) for r in recuperados}
    aciertos = len({e for e in esperados if e in recuperadas})
    return {
        "recall": aciertos / len(esperados),
        "precision": aciertos / k,
        "esperados": len(esperados),
        "encontrados": aciertos,
    }


def _faltan(esperados: list[str], recuperados: list[dict], canal: str) -> list[str]:
    """Esperados que NO aparecen en el top-k (para el listado de fallos)."""
    recuperadas = {clave_recuperado(r, canal) for r in recuperados}
    return [e for e in esperados if e not in recuperadas]


# ── Evaluación ────────────────────────────────────────────────────────────


def evaluar_pregunta(rag: RAG, pregunta: dict, k: int) -> dict:
    """Recupera top-k por canal y calcula las métricas de una pregunta."""
    query = pregunta["pregunta"]
    recuperados = {
        "factores": rag.search_factores(query, k=k),
        "documentos": rag.search_documentos(query, k=k),
    }
    metricas = {
        canal: calcular_metricas(pregunta[f"{canal}_esperados"], recuperados[canal], k, canal)
        for canal in CANALES
    }
    faltan = {
        canal: _faltan(pregunta[f"{canal}_esperados"], recuperados[canal], canal)
        for canal in CANALES
    }
    return {
        "id": pregunta["id"],
        "pregunta": query,
        "metricas": metricas,
        "faltan": faltan,
    }


def agregar(resultados: list[dict]) -> dict:
    """Medias de recall y precision por canal sobre preguntas con esperados."""
    agregado = {}
    for canal in CANALES:
        con_esperados = [r["metricas"][canal] for r in resultados if r["metricas"][canal]]
        if not con_esperados:
            agregado[canal] = None
            continue
        n = len(con_esperados)
        agregado[canal] = {
            "n_preguntas": n,
            "recall": sum(m["recall"] for m in con_esperados) / n,
            "precision": sum(m["precision"] for m in con_esperados) / n,
        }
    return agregado


# ── Salida ────────────────────────────────────────────────────────────────


def _celda(metrica: dict | None) -> tuple[str, str]:
    if metrica is None:
        return "-", "-"
    return f"{metrica['recall']:.3f}", f"{metrica['precision']:.3f}"


def formatear_tabla(
    set_ruta: Path, db_ruta: Path, k: int, stats: dict, resultados: list[dict], agregado: dict
) -> str:
    lineas = [
        "=== RAG-004 · evaluación del retrieval (solo retrieval, sin LLM) ===",
        f"set: {set_ruta}",
        f"db:  {db_ruta}",
        f"k:   {k} · {len(resultados)} preguntas · "
        f"{stats['factores']['embedded']}/{stats['factores']['total']} factores indexados · "
        f"{stats['documentos']['fragmentos']} fragmentos de documentación",
        "",
        "POR PREGUNTA (canales con esperados; '-' = sin esperados en ese canal)",
        f"{'id':<16}{'pregunta':<46}{'fac recall@k':>13}{'fac prec@k':>11}"
        f"{'doc recall@k':>13}{'doc prec@k':>11}",
    ]
    for r in resultados:
        pregunta = r["pregunta"]
        if len(pregunta) > 45:
            pregunta = pregunta[:44] + "…"
        fr, fp = _celda(r["metricas"]["factores"])
        dr, dp = _celda(r["metricas"]["documentos"])
        lineas.append(
            f"{r['id']:<16}{pregunta:<46}{fr:>13}{fp:>11}{dr:>13}{dp:>11}"
        )

    lineas.append("")
    lineas.append(f"AGREGADO (k={k}) — media sobre preguntas con esperados en cada canal")
    for canal in CANALES:
        agg = agregado[canal]
        if agg is None:
            lineas.append(f"  {canal:<12} sin preguntas con esperados")
        else:
            lineas.append(
                f"  {canal:<12} {agg['n_preguntas']:>3} preguntas · "
                f"recall@{k} = {agg['recall']:.3f} · precision@{k} = {agg['precision']:.3f}"
            )

    fallos = [
        r for r in resultados if any(r["metricas"][c] and r["metricas"][c]["recall"] < 1.0 for c in CANALES)
    ]
    lineas.append("")
    lineas.append(f"FALLOS (recall < 1.0 en algún canal con esperados): {len(fallos)}/{len(resultados)}")
    for r in fallos:
        detalle = []
        for canal in CANALES:
            if not r["faltan"][canal]:
                continue
            faltan = [e.split("/")[-1] for e in r["faltan"][canal]]
            detalle.append(f"{canal}: faltan {faltan}")
        lineas.append(f"  {r['id']:<16} {'; '.join(detalle)}")

    return "\n".join(lineas)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evalúa el retrieval del RAG (recall@k y precision@k).")
    parser.add_argument("--k", type=int, default=5, help="top-k a recuperar (default: 5)")
    parser.add_argument("--db", type=Path, default=DB_POR_DEFECTO, help="ruta de la BD sqlite-vec")
    parser.add_argument("--set", type=Path, default=SET_POR_DEFECTO, help="ruta del JSON etiquetado")
    parser.add_argument(
        "--json", action="store_true", help="salida JSON (para consumo por agentes), no tabla"
    )
    args = parser.parse_args()

    if args.k < 1:
        sys.exit("ERROR: k debe ser >= 1")
    preguntas = cargar_set(args.set)

    if not args.db.exists():
        sys.exit(
            f"ERROR: no existe la BD {args.db}. Genérala con el propio RAG "
            "(RAG.initialize + sync_factores + sync_documentos) y vuelve a ejecutar."
        )
    rag = RAG(args.db)
    stats = rag.stats()

    resultados = [evaluar_pregunta(rag, p, args.k) for p in preguntas]
    agregado = agregar(resultados)

    if args.json:
        json.dump(
            {
                "set": str(args.set),
                "db": str(args.db),
                "k": args.k,
                "stats": stats,
                "resultados": resultados,
                "agregado": agregado,
            },
            sys.stdout,
            ensure_ascii=False,
            indent=2,
        )
        return

    print(formatear_tabla(args.set, args.db, args.k, stats, resultados, agregado))


if __name__ == "__main__":
    main()
