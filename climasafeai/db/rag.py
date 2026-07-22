from __future__ import annotations

import sqlite3
import struct
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

import sqlite_vec

EMBEDDING_DIM = 384
_embedder: Any = None


def _get_embedder():
    global _embedder
    if _embedder is None:
        from sentence_transformers import SentenceTransformer
        _embedder = SentenceTransformer("all-MiniLM-L6-v2")
    return _embedder


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
