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
DOCS_DIR = "documentacion"
DOCS_GLOB = "**/*.md"
DOCS_EXCLUDE = {"llm"}  # subcarpetas que excluir (aún no indexadas)
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
    if f.get("coef") is not None:
        parts.append(f"coeficiente: {f['coef']}")
    if f.get("poblacion"):
        parts.append(f"población: {f['poblacion']}")
    if f.get("doi"):
        parts.append(f"doi: {f['doi']}")
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
            conn.execute(f"""
                CREATE VIRTUAL TABLE IF NOT EXISTS docs_vec USING vec0(
                    embedding float[{EMBEDDING_DIM}] distance_metric=cosine
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS docs_vec_src (
                    vec_rowid INTEGER PRIMARY KEY,
                    ruta TEXT NOT NULL,
                    titulo TEXT NOT NULL,
                    seccion TEXT,
                    texto TEXT NOT NULL,
                    palabras INTEGER NOT NULL DEFAULT 0
                )
            """)
        self.sync_factores()
        self.sync_documentos()

    def sync_factores(self) -> int:
        nuevas = 0
        with self._conn() as conn:
            rows = conn.execute("""
                SELECT f.id, f.tipo, f.categoria, f.clave, f.nombre, f.coef, f.doi, f.poblacion
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


    # ── Indexado de documentos ──────────────────────────────────────

    def _chunks_desde_md(self, ruta: Path) -> list[dict[str, Any]]:
        """Divide un .md en fragmentos por secciones (##). Cada fragmento
        conserva el título del documento y el nombre de la sección."""
        texto = ruta.read_text(encoding="utf-8")
        titulo = ""
        for line in texto.splitlines():
            if line.startswith("# ") and not line.startswith("## "):
                titulo = line.lstrip("# ").strip()
                break
        secciones: list[dict[str, Any]] = []
        seccion_actual = "__intro__"
        parrafos: list[str] = []

        for line in texto.splitlines():
            if line.startswith("## "):
                if parrafos:
                    cuerpo = " ".join(p.strip() for p in parrafos if p.strip())
                    if len(cuerpo.split()) >= 10:
                        secciones.append({
                            "ruta": str(ruta),
                            "titulo": titulo.split(" — ")[0] if " — " in titulo else titulo or ruta.stem,
                            "seccion": seccion_actual,
                            "texto": cuerpo,
                            "palabras": len(cuerpo.split()),
                        })
                    parrafos = []
                seccion_actual = line.lstrip("## ").strip()
            else:
                parrafos.append(line)

        if parrafos:
            cuerpo = " ".join(p.strip() for p in parrafos if p.strip())
            if len(cuerpo.split()) >= 10:
                secciones.append({
                    "ruta": str(ruta),
                    "titulo": titulo.split(" — ")[0] if " — " in titulo else titulo or ruta.stem,
                    "seccion": seccion_actual,
                    "texto": cuerpo,
                    "palabras": len(cuerpo.split()),
                })
        return secciones

    def _documentos_nuevos(self, project_root: Path) -> list[dict[str, Any]]:
        """Escanea documentacion/ y devuelve los fragmentos no indexados."""
        docs_dir = project_root / DOCS_DIR
        if not docs_dir.is_dir():
            return []
        with self._conn() as conn:
            indexed = {
                row["ruta"] + "::" + (row["seccion"] or "")
                for row in conn.execute(
                    "SELECT DISTINCT ruta, COALESCE(seccion,'') as seccion FROM docs_vec_src"
                ).fetchall()
            }

        chunks_nuevos = []
        for md_file in sorted(docs_dir.rglob(DOCS_GLOB)):
            # Excluir subcarpetas que no tocan
            rel = md_file.relative_to(docs_dir)
            if any(part in DOCS_EXCLUDE for part in rel.parts):
                continue
            for chunk in self._chunks_desde_md(md_file):
                key = chunk["ruta"] + "::" + (chunk["seccion"] or "")
                if key not in indexed:
                    chunks_nuevos.append(chunk)
        return chunks_nuevos

    def sync_documentos(self, project_root: Path | None = None) -> int:
        """Indexa los fragmentos nuevos de documentacion/ en la tabla vectorial."""
        if project_root is None:
            project_root = Path(__file__).resolve().parents[2]  # rag.py → db/ → climasafeai/ → raíz
        chunks = self._documentos_nuevos(project_root)
        if not chunks:
            return 0
        model = _get_embedder()
        with self._conn() as conn:
            for c in chunks:
                emb = model.encode(c["texto"])
                emb_bytes = struct.pack(f"{len(emb)}f", *emb)
                cur = conn.execute(
                    "INSERT INTO docs_vec (embedding) VALUES (?)", (emb_bytes,)
                )
                conn.execute(
                    "INSERT INTO docs_vec_src (vec_rowid, ruta, titulo, seccion, texto, palabras) VALUES (?, ?, ?, ?, ?, ?)",
                    (cur.lastrowid, c["ruta"], c["titulo"], c["seccion"], c["texto"], c["palabras"]),
                )
        return len(chunks)

    def resync_documentos(self, project_root: Path | None = None) -> int:
        """Borra y reindexa todos los documentos desde cero."""
        with self._conn() as conn:
            conn.execute("DELETE FROM docs_vec")
            conn.execute("DELETE FROM docs_vec_src")
        return self.sync_documentos(project_root=project_root)

    # ── Búsquedas ───────────────────────────────────────────────────

    def search_documentos(self, query: str, k: int = 5) -> list[dict]:
        """Busca fragmentos de documentacion/ por similitud semántica."""
        emb = _get_embedder().encode(query)
        emb_bytes = struct.pack(f"{len(emb)}f", *emb)
        with self._conn() as conn:
            rows = conn.execute("""
                SELECT s.ruta, s.titulo, s.seccion, s.texto, s.palabras, v.distance
                FROM docs_vec v
                JOIN docs_vec_src s ON v.rowid = s.vec_rowid
                WHERE v.embedding MATCH ? AND k=?
                ORDER BY v.distance
            """, (emb_bytes, k)).fetchall()
            return [dict(r) for r in rows]

    def search_all(self, query: str, k: int = 5) -> dict:
        """Busca en factores y documentos, combina resultados."""
        return {
            "factores": self.search_factores(query, k=k),
            "documentos": self.search_documentos(query, k=k),
        }

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
            total_factores_emb = conn.execute(
                "SELECT COUNT(*) as n FROM factores_vec_src"
            ).fetchone()["n"]
            total_factores = conn.execute(
                "SELECT COUNT(*) as n FROM factores_riesgo"
            ).fetchone()["n"]
            total_docs = conn.execute(
                "SELECT COUNT(*) as n FROM docs_vec_src"
            ).fetchone()["n"]
            total_docs_palabras = conn.execute(
                "SELECT COALESCE(SUM(palabras), 0) as n FROM docs_vec_src"
            ).fetchone()["n"]
            return {
                "factores": {
                    "embedded": total_factores_emb,
                    "total": total_factores,
                    "pending": total_factores - total_factores_emb,
                },
                "documentos": {
                    "fragmentos": total_docs,
                    "palabras": total_docs_palabras,
                },
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
