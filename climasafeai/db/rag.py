from __future__ import annotations

import hashlib
import os
import sqlite3
import struct
import textwrap
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

import sqlite_vec

EMBEDDING_DIM = 512
# RAG-006: dimensión del modelo alternativo de embeddings. sqlite-vec fija la
# dimensión en el schema de la columna virtual, así que un modelo con dimensión
# distinta necesita su propia tabla (ver MODELOS_EMBEDDING).
EMBEDDING_DIM_ALT = 384
DOCS_DIR = "documentacion"
DOCS_GLOB = "**/*.md"
DOCS_EXCLUDE = {"llm"}  # subcarpetas que excluir (aún no indexadas)

# RAG-006: modelos de embeddings soportados. El índice activo (tablas sin
# sufijo) usa el modelo por defecto; los alternativos conviven en tablas
# ``_<slug>`` con su propia dimensión, para compararlos sin perder el índice
# anterior. El por defecto se eligió por los números de la comparativa
# (scripts/comparar_modelos_rag.py → documentacion/rag_006_comparativa_embeddings.md):
# distiluse-base-multilingual-cased-v2 gana a all-MiniLM-L6-v2 en ambos canales.
RAG_EMBEDDER_DEFAULT = "distiluse-base-multilingual-cased-v2"
RAG_EMBEDDER_ALT = "all-MiniLM-L6-v2"
MODELOS_EMBEDDING = {
    RAG_EMBEDDER_DEFAULT: {"dim": EMBEDDING_DIM, "slug": None},
    RAG_EMBEDDER_ALT: {"dim": EMBEDDING_DIM_ALT, "slug": "minilm"},
}

# RAG-006: solapamiento por defecto del chunking por secciones — caracteres de
# la sección anterior que se anteponen a cada sección. Se implementó y se midió
# contra la línea base de RAG-004: EMPEORA el recall de documentos (0.325→0.286
# con all-MiniLM; 0.611→0.526 con distiluse), así que se deja desactivado (0).
# El código queda disponible (configurable) por si se quisiera re-medir.
CHUNK_OVERLAP = 0

# RAG-003: de las colecciones de documentacion/, solo papers (literatura del
# dominio) y riesgo (coeficientes y fórmulas) son conocimiento recuperable en
# preguntas de usuario. La metodología ML (ml/, modelos/, arquitectura/) y las
# notas internas del proyecto (documentos raíz) se siguen indexando — para
# stats y reindexado — pero se filtran en la búsqueda.
DOCS_COLECCION_USUARIO = ("papers", "riesgo")

_embedders: dict[str, Any] = {}
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


def _get_embedder(modelo: str | None = None):
    """Devuelve el SentenceTransformer del modelo pedido (cacheado por modelo).

    ``modelo=None`` → el modelo por defecto (RAG_EMBEDDER_DEFAULT). La llamada
    sin argumentos sigue funcionando igual que antes de RAG-006.
    """
    global _embedders
    modelo = modelo or RAG_EMBEDDER_DEFAULT
    if modelo not in _embedders:
        from sentence_transformers import SentenceTransformer

        _embedders[modelo] = SentenceTransformer(modelo)
    return _embedders[modelo]


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
        (
            "GEMINI_API_KEY",
            "GEMINI_BASE_URL",
            "https://generativelanguage.googleapis.com/v1beta/openai/",
        ),
    ]

    for key_var, url_var, default_url in configs:
        api_key = os.getenv(key_var)
        if api_key:
            base_url = (
                os.getenv(url_var, default_url)
                if default_url
                else os.getenv(url_var, "https://api.openai.com/v1")
            )
            if base_url:
                _llm_client = OpenAI(api_key=api_key, base_url=base_url, timeout=30)
                return _llm_client
    return None


def _llm_model() -> str:
    return os.getenv("RAG_MODEL", "llama-3.3-70b-versatile")


def _hash_texto(texto: str) -> str:
    """Huella del contenido embebido. La detección de cambios se hace con esto
    y no con la clave (ruta::sección o factor_id): editar el cuerpo sin tocar
    el título dejaba el fragmento viejo indexado para siempre."""
    return hashlib.sha256(texto.encode("utf-8")).hexdigest()


def _coleccion_desde_ruta(ruta: str) -> str:
    """Clasifica un fragmento de documentacion/ en una colección.

    La colección sale de la carpeta siguiente a ``documentacion/``:
    - ``papers/**`` → papers (literatura del dominio)
    - ``riesgo/**`` → riesgo (coeficientes y fórmulas)
    - ``ml|modelos|arquitectura|llm/**`` → ml (metodología ML)
    - el resto (documentos raíz: PRD, roadmap, notas...) → interna

    Solo ``papers`` y ``riesgo`` se recuperan en preguntas de usuario
    (``DOCS_COLECCION_USUARIO``); el resto se indexa pero no se busca.
    """
    partes = Path(ruta).parts
    try:
        idx = next(i for i, p in enumerate(partes) if p == DOCS_DIR)
    except StopIteration:
        return "interna"
    primera = partes[idx + 1] if idx + 1 < len(partes) else ""
    if primera in DOCS_COLECCION_USUARIO:
        return primera
    if primera in ("ml", "modelos", "arquitectura", "llm"):
        return "ml"
    return "interna"


def _tablas_modelo(modelo: str, canal: str) -> tuple[str, str]:
    """Nombres (vectorial, src) de las tablas de un modelo y canal.

    El índice activo (modelo por defecto) no lleva sufijo: es el que usan
    search_factores/search_documentos y las bases creadas antes de RAG-006.
    Los modelos alternativos viven en tablas ``_<slug>`` con su propia
    dimensión, para compararlos sin tocar el índice anterior.
    """
    slug = MODELOS_EMBEDDING[modelo]["slug"]
    if slug is None:
        return f"{canal}_vec", f"{canal}_vec_src"
    return f"{canal}_vec_{slug}", f"{canal}_vec_{slug}_src"


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
                    texto TEXT NOT NULL,
                    hash TEXT,
                    modelo TEXT
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
                    palabras INTEGER NOT NULL DEFAULT 0,
                    hash TEXT,
                    coleccion TEXT,
                    modelo TEXT
                )
            """)
            # RAG-006: una tabla vectorial por modelo alternativo, cada una con
            # su propia dimensión (sqlite-vec la fija en el schema). Así se
            # comparan sin tocar el índice activo (tablas sin sufijo). La tabla
            # src acompaña la estructura de su canal (factores vs documentos).
            for modelo, info in MODELOS_EMBEDDING.items():
                if modelo == RAG_EMBEDDER_DEFAULT:
                    continue
                for canal in ("factores", "docs"):
                    tabla_vec, tabla_src = _tablas_modelo(modelo, canal)
                    conn.execute(
                        f"CREATE VIRTUAL TABLE IF NOT EXISTS {tabla_vec} USING vec0("
                        f"embedding float[{info['dim']}] distance_metric=cosine)"
                    )
                    if canal == "factores":
                        conn.execute(
                            f"CREATE TABLE IF NOT EXISTS {tabla_src} ("
                            "vec_rowid INTEGER PRIMARY KEY,"
                            "factor_id INTEGER NOT NULL,"
                            "tipo TEXT NOT NULL,"
                            "categoria TEXT NOT NULL,"
                            "clave TEXT NOT NULL,"
                            "texto TEXT NOT NULL,"
                            "hash TEXT,"
                            "modelo TEXT)"
                        )
                    else:
                        conn.execute(
                            f"CREATE TABLE IF NOT EXISTS {tabla_src} ("
                            "vec_rowid INTEGER PRIMARY KEY,"
                            "ruta TEXT NOT NULL,"
                            "titulo TEXT NOT NULL,"
                            "seccion TEXT,"
                            "texto TEXT NOT NULL,"
                            "palabras INTEGER NOT NULL DEFAULT 0,"
                            "hash TEXT,"
                            "coleccion TEXT,"
                            "modelo TEXT)"
                        )
            # Bases ya creadas antes de RAG-005 no tienen la columna. Sin hash
            # se comportan como antes (todo se ve como cambiado) hasta el
            # primer sync, que las rellena.
            for tabla in ("factores_vec_src", "docs_vec_src"):
                columnas = {r["name"] for r in conn.execute(f"PRAGMA table_info({tabla})")}
                if "hash" not in columnas:
                    conn.execute(f"ALTER TABLE {tabla} ADD COLUMN hash TEXT")
            # RAG-003: bases creadas antes no tienen la colección. Sin ella, la
            # búsqueda no sabe qué filtrar, así que se rellena desde la ruta.
            columnas = {r["name"] for r in conn.execute("PRAGMA table_info(docs_vec_src)")}
            if "coleccion" not in columnas:
                conn.execute("ALTER TABLE docs_vec_src ADD COLUMN coleccion TEXT")
            # RAG-006: bases creadas antes no tienen la columna modelo. Sin ella
            # el sync no sabría si la fila se embebió con el modelo actual, así
            # que se añade; el primer sync la rellena (y reembebe si procede).
            for tabla in ("factores_vec_src", "docs_vec_src"):
                columnas = {r["name"] for r in conn.execute(f"PRAGMA table_info({tabla})")}
                if "modelo" not in columnas:
                    conn.execute(f"ALTER TABLE {tabla} ADD COLUMN modelo TEXT")
            self._backfill_coleccion(conn)
        self.sync_factores()
        self.sync_documentos()

    def _backfill_coleccion(self, conn) -> None:
        """Rellena la colección de filas creadas antes de RAG-003 (o sin ella).

        La colección se deriva de la ruta: solo hace falta una vez, en la
        migración; las filas nuevas ya la traen del chunker."""
        for fila in conn.execute(
            "SELECT vec_rowid, ruta FROM docs_vec_src WHERE coleccion IS NULL"
        ).fetchall():
            conn.execute(
                "UPDATE docs_vec_src SET coleccion = ? WHERE vec_rowid = ?",
                (_coleccion_desde_ruta(fila["ruta"]), fila["vec_rowid"]),
            )

    def sync_factores(self, modelo: str | None = None) -> int:
        """Indexa los factores nuevos y REINDEXA los que cambiaron de contenido
        (nombre, coeficiente o DOI) o de modelo de embeddings: la comparación es
        por hash del texto + modelo con el que se embebió."""
        modelo = modelo or RAG_EMBEDDER_DEFAULT
        tabla_vec, tabla_src = _tablas_modelo(modelo, "factores")
        embedder = (
            _get_embedder() if modelo == RAG_EMBEDDER_DEFAULT else _get_embedder(modelo)
        )
        nuevas = 0
        with self._conn() as conn:
            rows = conn.execute(f"""
                SELECT f.id, f.tipo, f.categoria, f.clave, f.nombre, f.coef, f.doi, f.poblacion,
                       s.vec_rowid as _vec_rowid, s.hash as _hash, s.modelo as _modelo
                FROM factores_riesgo f
                LEFT JOIN {tabla_src} s ON f.id = s.factor_id
                ORDER BY f.id
            """).fetchall()

            for r in rows:
                f = dict(r)
                vec_rowid_previo = f.pop("_vec_rowid", None)
                hash_previo = f.pop("_hash", None)
                modelo_previo = f.pop("_modelo", None)
                texto = _factor_text(f)
                hash_actual = _hash_texto(texto)
                if (
                    vec_rowid_previo is not None
                    and hash_previo == hash_actual
                    and modelo_previo == modelo
                ):
                    continue

                # Cambió: fuera la fila vieja antes de insertar, o stats()
                # contaría el factor dos veces.
                if vec_rowid_previo is not None:
                    conn.execute(f"DELETE FROM {tabla_vec} WHERE rowid = ?", (vec_rowid_previo,))
                    conn.execute(
                        f"DELETE FROM {tabla_src} WHERE vec_rowid = ?", (vec_rowid_previo,)
                    )

                emb = embedder.encode(texto)
                emb_bytes = struct.pack(f"{len(emb)}f", *emb)
                cur = conn.execute(f"INSERT INTO {tabla_vec} (embedding) VALUES (?)", (emb_bytes,))
                vec_rowid = cur.lastrowid
                conn.execute(
                    f"INSERT INTO {tabla_src} (vec_rowid, factor_id, tipo, categoria, clave, texto, hash, modelo) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    (vec_rowid, f["id"], f["tipo"], f["categoria"], f["clave"], texto, hash_actual, modelo),
                )
                nuevas += 1
        return nuevas

    def resync_factores(self, modelo: str | None = None) -> int:
        modelo = modelo or RAG_EMBEDDER_DEFAULT
        tabla_vec, tabla_src = _tablas_modelo(modelo, "factores")
        with self._conn() as conn:
            conn.execute(f"DELETE FROM {tabla_vec}")
            conn.execute(f"DELETE FROM {tabla_src}")
        return self.sync_factores(modelo=modelo)

    def sync_factores_modelo(self, modelo: str) -> int:
        """Indexa los factores en el índice del modelo indicado."""
        return self.sync_factores(modelo=modelo)

    def resync_factores_modelo(self, modelo: str) -> int:
        """Borra y reindexa los factores en el índice del modelo indicado."""
        return self.resync_factores(modelo=modelo)

    # ── Indexado de documentos ──────────────────────────────────────

    def _chunks_desde_md(self, ruta: Path, overlap: int | None = None) -> list[dict[str, Any]]:
        """Divide un .md en fragmentos por secciones (##). Cada fragmento
        conserva el título del documento y el nombre de la sección.

        RAG-006: si ``overlap > 0``, cada sección lleva como prefijo las últimas
        ``overlap`` caracteres de la sección anterior (sin contar el prefijo que
        ella misma recibió). Así una pregunta que cae entre dos secciones
        recupera ambas: la anterior por su propio fragmento y la siguiente
        porque contiene su cola. La clave de dedup (ruta::seccion) no cambia;
        solo cambia el texto (y con él el hash), así que sync_documentos
        reindexa lo que haga falta.
        """
        if overlap is None:
            overlap = CHUNK_OVERLAP
        texto = ruta.read_text(encoding="utf-8")
        titulo = ""
        for line in texto.splitlines():
            if line.startswith("# ") and not line.startswith("## "):
                titulo = line.lstrip("# ").strip()
                break
        secciones: list[dict[str, Any]] = []
        seccion_actual = "__intro__"
        parrafos: list[str] = []
        cola_anterior = ""  # cola de la sección previa, sin su propio prefijo

        def _emitir() -> None:
            nonlocal cola_anterior
            cuerpo_puro = " ".join(p.strip() for p in parrafos if p.strip())
            if len(cuerpo_puro.split()) < 10:
                return
            cuerpo = (cola_anterior + " " + cuerpo_puro).strip() if cola_anterior else cuerpo_puro
            cola_anterior = cuerpo_puro[-overlap:] if overlap else ""
            secciones.append(
                {
                    "ruta": str(ruta),
                    "titulo": titulo.split(" — ")[0]
                    if " — " in titulo
                    else titulo or ruta.stem,
                    "seccion": seccion_actual,
                    "texto": cuerpo,
                    "palabras": len(cuerpo.split()),
                    "coleccion": _coleccion_desde_ruta(str(ruta)),
                }
            )

        for line in texto.splitlines():
            if line.startswith("## "):
                _emitir()
                parrafos = []
                seccion_actual = line.lstrip("## ").strip()
            else:
                parrafos.append(line)
        _emitir()
        return secciones

    def _documentos_nuevos(
        self,
        project_root: Path,
        tabla_src: str = "docs_vec_src",
        overlap: int | None = None,
        modelo: str = RAG_EMBEDDER_DEFAULT,
    ) -> list[dict[str, Any]]:
        """Escanea documentacion/ y devuelve los fragmentos nuevos y los que
        cambiaron de contenido o de modelo. Cada fragmento a reindexar lleva el
        ``_vec_rowid`` de su fila vieja para que sync la borre antes."""
        docs_dir = project_root / DOCS_DIR
        if not docs_dir.is_dir():
            return []
        with self._conn() as conn:
            indexed = {
                row["ruta"] + "::" + (row["seccion"] or ""): (
                    row["vec_rowid"],
                    row["hash"],
                    row["modelo"],
                )
                for row in conn.execute(
                    f"SELECT ruta, COALESCE(seccion,'') as seccion, vec_rowid, hash, modelo FROM {tabla_src}"
                ).fetchall()
            }

        chunks_nuevos = []
        for md_file in sorted(docs_dir.rglob(DOCS_GLOB)):
            # Excluir subcarpetas que no tocan
            rel = md_file.relative_to(docs_dir)
            if any(part in DOCS_EXCLUDE for part in rel.parts):
                continue
            for chunk in self._chunks_desde_md(md_file, overlap=overlap):
                key = chunk["ruta"] + "::" + (chunk["seccion"] or "")
                chunk["hash"] = _hash_texto(chunk["texto"])
                previo = indexed.get(key)
                if previo is None:
                    chunks_nuevos.append(chunk)
                elif previo[1] != chunk["hash"] or previo[2] != modelo:
                    chunk["_vec_rowid"] = previo[0]
                    chunks_nuevos.append(chunk)
        return chunks_nuevos

    def sync_documentos(
        self,
        project_root: Path | None = None,
        overlap: int | None = None,
        modelo: str | None = None,
    ) -> int:
        """Indexa los fragmentos nuevos de documentacion/ en la tabla vectorial.

        ``modelo``: embedder con el que embeder (por defecto el activo). El
        solapamiento del chunking se controla con ``overlap``.
        """
        if project_root is None:
            project_root = Path(__file__).resolve().parents[2]  # rag.py → db/ → climasafeai/ → raíz
        modelo = modelo or RAG_EMBEDDER_DEFAULT
        tabla_vec, tabla_src = _tablas_modelo(modelo, "docs")
        chunks = self._documentos_nuevos(
            project_root, tabla_src=tabla_src, overlap=overlap, modelo=modelo
        )
        if not chunks:
            return 0
        embedder = (
            _get_embedder() if modelo == RAG_EMBEDDER_DEFAULT else _get_embedder(modelo)
        )
        with self._conn() as conn:
            for c in chunks:
                # Fragmento que ya existía y cambió: fuera la fila vieja, o
                # stats() contaría dos veces la misma sección.
                vec_rowid_previo = c.get("_vec_rowid")
                if vec_rowid_previo is not None:
                    conn.execute(f"DELETE FROM {tabla_vec} WHERE rowid = ?", (vec_rowid_previo,))
                    conn.execute(
                        f"DELETE FROM {tabla_src} WHERE vec_rowid = ?", (vec_rowid_previo,)
                    )

                emb = embedder.encode(c["texto"])
                emb_bytes = struct.pack(f"{len(emb)}f", *emb)
                cur = conn.execute(f"INSERT INTO {tabla_vec} (embedding) VALUES (?)", (emb_bytes,))
                conn.execute(
                    f"INSERT INTO {tabla_src} (vec_rowid, ruta, titulo, seccion, texto, palabras, hash, coleccion, modelo) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        cur.lastrowid,
                        c["ruta"],
                        c["titulo"],
                        c["seccion"],
                        c["texto"],
                        c["palabras"],
                        c["hash"],
                        c["coleccion"],
                        modelo,
                    ),
                )
        return len(chunks)

    def resync_documentos(
        self,
        project_root: Path | None = None,
        overlap: int | None = None,
        modelo: str | None = None,
    ) -> int:
        """Borra y reindexa todos los documentos desde cero."""
        modelo = modelo or RAG_EMBEDDER_DEFAULT
        tabla_vec, tabla_src = _tablas_modelo(modelo, "docs")
        with self._conn() as conn:
            conn.execute(f"DELETE FROM {tabla_vec}")
            conn.execute(f"DELETE FROM {tabla_src}")
        return self.sync_documentos(project_root=project_root, overlap=overlap, modelo=modelo)

    def sync_documentos_modelo(
        self,
        modelo: str,
        project_root: Path | None = None,
        overlap: int | None = None,
    ) -> int:
        """Indexa los documentos en el índice del modelo indicado."""
        return self.sync_documentos(project_root=project_root, overlap=overlap, modelo=modelo)

    def resync_documentos_modelo(
        self,
        modelo: str,
        project_root: Path | None = None,
        overlap: int | None = None,
    ) -> int:
        """Borra y reindexa los documentos en el índice del modelo indicado."""
        return self.resync_documentos(project_root=project_root, overlap=overlap, modelo=modelo)

    # ── Búsquedas ───────────────────────────────────────────────────

    def search_documentos(self, query: str, k: int = 5) -> list[dict]:
        """Busca fragmentos de documentacion/ por similitud semántica (índice activo)."""
        return self._search_documentos_en(query, k=k, modelo=RAG_EMBEDDER_DEFAULT)

    def search_documentos_modelo(self, query: str, modelo: str, k: int = 5) -> list[dict]:
        """Busca fragmentos en el índice del modelo indicado."""
        return self._search_documentos_en(query, k=k, modelo=modelo)

    def _search_documentos_en(self, query: str, k: int, modelo: str) -> list[dict]:
        """Busca fragmentos de documentacion/ por similitud semántica.

        Solo devuelve las colecciones de conocimiento del dominio (papers y
        riesgo): la metodología ML y las notas internas del proyecto se
        indexan, pero no se recuperan en preguntas de usuario (RAG-003).
        """
        tabla_vec, tabla_src = _tablas_modelo(modelo, "docs")
        embedder = (
            _get_embedder() if modelo == RAG_EMBEDDER_DEFAULT else _get_embedder(modelo)
        )
        emb = embedder.encode(query)
        emb_bytes = struct.pack(f"{len(emb)}f", *emb)
        # Se piden k×3 candidatos y se filtra después: el índice vectorial
        # devuelve el top-k global y el filtro por colección se aplica sobre
        # ese top-k. Con solo k candidatos, una pregunta de usuario perdería
        # resultados válidos de papers/riesgo que quedasen por detrás de
        # fragmentos ml/internos más cercanos.
        with self._conn() as conn:
            rows = conn.execute(
                f"""
                SELECT s.ruta, s.titulo, s.seccion, s.texto, s.palabras, s.coleccion, v.distance
                FROM {tabla_vec} v
                JOIN {tabla_src} s ON v.rowid = s.vec_rowid
                WHERE v.embedding MATCH ? AND k=?
                ORDER BY v.distance
            """,
                (emb_bytes, k * 3),
            ).fetchall()
            return [dict(r) for r in rows if r["coleccion"] in DOCS_COLECCION_USUARIO][:k]

    def search_all(self, query: str, k: int = 5) -> dict:
        """Busca en factores y documentos, combina resultados."""
        return {
            "factores": self.search_factores(query, k=k),
            "documentos": self.search_documentos(query, k=k),
        }

    def search_factores(self, query: str, k: int = 5) -> list[dict]:
        """Busca factores por similitud semántica (índice activo)."""
        return self._search_factores_en(query, k=k, modelo=RAG_EMBEDDER_DEFAULT)

    def search_factores_modelo(self, query: str, modelo: str, k: int = 5) -> list[dict]:
        """Busca factores en el índice del modelo indicado."""
        return self._search_factores_en(query, k=k, modelo=modelo)

    def _search_factores_en(self, query: str, k: int, modelo: str) -> list[dict]:
        tabla_vec, tabla_src = _tablas_modelo(modelo, "factores")
        embedder = (
            _get_embedder() if modelo == RAG_EMBEDDER_DEFAULT else _get_embedder(modelo)
        )
        emb = embedder.encode(query)
        emb_bytes = struct.pack(f"{len(emb)}f", *emb)
        with self._conn() as conn:
            rows = conn.execute(
                f"""
                SELECT s.tipo, s.categoria, s.clave, s.texto, v.distance
                FROM {tabla_vec} v
                JOIN {tabla_src} s ON v.rowid = s.vec_rowid
                WHERE v.embedding MATCH ? AND k=?
                ORDER BY v.distance
            """,
                (emb_bytes, k),
            ).fetchall()
            return [dict(r) for r in rows]

    def stats(self) -> dict:
        with self._conn() as conn:
            total_factores_emb = conn.execute(
                "SELECT COUNT(*) as n FROM factores_vec_src"
            ).fetchone()["n"]
            total_factores = conn.execute("SELECT COUNT(*) as n FROM factores_riesgo").fetchone()[
                "n"
            ]
            total_docs = conn.execute("SELECT COUNT(*) as n FROM docs_vec_src").fetchone()["n"]
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
            f"{i}. {r['texto']} (distancia: {r['distance']:.3f})" for i, r in enumerate(results, 1)
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
