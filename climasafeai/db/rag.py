from __future__ import annotations

import os
import sqlite3
import struct
import textwrap
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

import sqlite_vec

EMBEDDING_DIM = 384
_embedder: Any = None
_llm_client: Any = None


SYSTEM_PROMPT = textwrap.dedent("""\
    Eres un asistente experto en factores de riesgo térmico (calor y frío)
    para ClimaSafeAI, un sistema de predicción de riesgo personalizado.

    Tu función es responder preguntas sobre factores de riesgo basándote
    EXCLUSIVAMENTE en el contexto proporcionado abajo (factores de riesgo
    recuperados de la base de conocimiento).

    NORMAS:
    - No inventes factores que no estén en el contexto.
    - Si el contexto no es suficiente para responder, dilo claramente.
    - Menciona los nombres concretos de los factores cuando sean relevantes.
    - Indica la categoría y el tipo (calor/frío) de cada factor que cites.
    - Sé conciso: responde directo, sin rodeos.
    - Si la pregunta no está relacionada con riesgo térmico, indica
      educadamente que solo respondes sobre factores de riesgo climático.
""")


def _get_embedder():
    global _embedder
    if _embedder is None:
        from sentence_transformers import SentenceTransformer
        _embedder = SentenceTransformer("all-MiniLM-L6-v2")
    return _embedder


def _get_llm_client() -> Any | None:
    """Cliente OpenAI-compatible para generar respuestas RAG.

    Proveedores (por orden de preferencia):
    1. GROQ_API_KEY + GROQ_BASE_URL (por defecto)
    2. OPENAI_API_KEY + OPENAI_BASE_URL
    3. GEMINI_API_KEY + GEMINI_BASE_URL
    """
    global _llm_client
    if _llm_client is not None:
        return _llm_client

    from openai import OpenAI

    configs = [
        ("GROQ_API_KEY", "GROQ_BASE_URL", "https://api.groq.com/openai/v1"),
        ("OPENAI_API_KEY", "OPENAI_BASE_URL", None),
        ("GEMINI_API_KEY", "GEMINI_BASE_URL", "https://generativelanguage.googleapis.com/v1beta/openai/"),
    ]

    for key_var, url_var, default_url in configs:
        api_key = os.getenv(key_var)
        if api_key:
            base_url = os.getenv(url_var, default_url) if default_url else os.getenv(url_var, "https://api.openai.com/v1")
            if base_url:
                _llm_client = OpenAI(api_key=api_key, base_url=base_url, timeout=30)
                return _llm_client
    return None


def _llm_model() -> str:
    return os.getenv("RAG_MODEL", "llama-3.3-70b-versatile")


def _factor_text(f: dict) -> str:
    parts = [f.get("nombre") or f["clave"], f"tipo: {f['tipo']}", f"categoría: {f['categoria']}"]
    if f.get("poblacion"):
        parts.append(f"población: {f['poblacion']}")
    return ". ".join(parts)


class RAG:
    def __init__(self, db_path: str | Path):
        self.db_path = Path(db_path)

    @contextmanager
    def _conn(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        conn.enable_load_extension(True)
        sqlite_vec.load(conn)
        conn.enable_load_extension(False)
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def initialize(self) -> None:
        with self._conn() as conn:
            conn.execute(f"""
                CREATE VIRTUAL TABLE IF NOT EXISTS factores_vec USING vec0(
                    embedding float[{EMBEDDING_DIM}] distance_metric=cosine
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS factores_vec_src (
                    vec_rowid INTEGER PRIMARY KEY,
                    factor_id INTEGER NOT NULL,
                    tipo TEXT NOT NULL,
                    categoria TEXT NOT NULL,
                    clave TEXT NOT NULL,
                    texto TEXT NOT NULL
                )
            """)
        self.sync_factores()

    def sync_factores(self) -> int:
        nuevas = 0
        with self._conn() as conn:
            rows = conn.execute("""
                SELECT f.id, f.tipo, f.categoria, f.clave, f.nombre, f.coef, f.poblacion
                FROM factores_riesgo f
                LEFT JOIN factores_vec_src s ON f.id = s.factor_id
                WHERE s.factor_id IS NULL
                ORDER BY f.id
            """).fetchall()

            model = _get_embedder()
            for r in rows:
                f = dict(r)
                texto = _factor_text(f)
                emb = model.encode(texto)
                emb_bytes = struct.pack(f"{len(emb)}f", *emb)
                cur = conn.execute(
                    "INSERT INTO factores_vec (embedding) VALUES (?)", (emb_bytes,)
                )
                vec_rowid = cur.lastrowid
                conn.execute(
                    "INSERT INTO factores_vec_src (vec_rowid, factor_id, tipo, categoria, clave, texto) VALUES (?, ?, ?, ?, ?, ?)",
                    (vec_rowid, f["id"], f["tipo"], f["categoria"], f["clave"], texto),
                )
                nuevas += 1
        return nuevas

    def resync_factores(self) -> int:
        with self._conn() as conn:
            conn.execute("DELETE FROM factores_vec")
            conn.execute("DELETE FROM factores_vec_src")
        return self.sync_factores()


    def search_factores(self, query: str, k: int = 5) -> list[dict]:
        emb = _get_embedder().encode(query)
        emb_bytes = struct.pack(f"{len(emb)}f", *emb)
        with self._conn() as conn:
            rows = conn.execute("""
                SELECT s.tipo, s.categoria, s.clave, s.texto, v.distance
                FROM factores_vec v
                JOIN factores_vec_src s ON v.rowid = s.vec_rowid
                WHERE v.embedding MATCH ? AND k=?
                ORDER BY v.distance
            """, (emb_bytes, k)).fetchall()
            return [dict(r) for r in rows]

    def stats(self) -> dict:
        with self._conn() as conn:
            total = conn.execute(
                "SELECT COUNT(*) as n FROM factores_vec_src"
            ).fetchone()["n"]
            total_factores = conn.execute(
                "SELECT COUNT(*) as n FROM factores_riesgo"
            ).fetchone()["n"]
            return {
                "embedded": total,
                "total_factores": total_factores,
                "pending": total_factores - total,
            }

    def ask(self, query: str, k: int = 5) -> dict:
        """Retrieve + Augment + Generate: RAG completo.

        1. Recupera los k factores más relevantes para la query.
        2. Construye un prompt con el contexto.
        3. Genera respuesta con LLM (Gemini vía API).
        4. Devuelve respuesta + fuentes.
        """
        results = self.search_factores(query, k=k)

        if not results:
            return {
                "answer": "No encontré factores de riesgo relevantes para tu consulta.",
                "sources": [],
            }

        client = _get_llm_client()
        if client is None:
            return {
                "answer": None,
                "sources": results,
                "error": "GEMINI_API_KEY no configurada — no se puede generar respuesta",
            }

        # Prompt de usuario: contexto recuperado + pregunta
        ctx = "\n".join(
            f"{i}. {r['texto']} (distancia: {r['distance']:.3f})"
            for i, r in enumerate(results, 1)
        )
        user_prompt = f"""Factores de riesgo recuperados:\n{ctx}\n\nPregunta: {query}\n\nResponde basándote exclusivamente en los factores de riesgo listados. Si no hay información suficiente, dilo. Menciona factores concretos cuando sea relevante."""

        try:
            resp = client.chat.completions.create(
                model=_llm_model(),
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.3,
                max_tokens=1024,
            )
            answer = resp.choices[0].message.content.strip()
        except Exception as e:
            return {
                "answer": None,
                "sources": results,
                "error": f"Error generando respuesta: {e}",
            }

        return {
            "answer": answer,
            "sources": results,
        }
