"""
climasafeai.ml.active_learner — Aprendizaje activo para el paper scout.

Clasifica papers como "aceptable" (contiene un factor de riesgo válido) o
"irrelevante" basado en el embedding del título+abstract, usando un
clasificador ligero (LogisticRegression) que mejora con cada approve/reject.

Flujo:
  1. ActiveLearner.predict(titulo, abstract) → (veredicto, confianza, info)
  2. Si confianza > 0.75, auto-filtra sin llamar al LLM
  3. Si confianza < 0.75, se abstiene → pasa al LLM
  4. Cada veredicto del LLM se almacena via store() / store_many()
  5. retrain() reentrena el clasificador con todos los ejemplos acumulados

La tabla ``scout_entrenamiento`` se crea automáticamente en SQLite.

Uso:
    from climasafeai.ml.active_learner import ActiveLearner

    al = ActiveLearner()
    al.ensure_table()

    # Añadir ejemplos
    al.store("title", "abstract", "aceptable")       # individual
    al.store_many([{...}])                            # batch

    # Entrenar
    al.retrain()

    # Predecir
    veredicto, confianza, info = al.predict("title", "abstract")
    # veredicto: "aceptable", "irrelevante", o None (incierto)

    # Estadísticas
    al.stats()

Entrenamiento inicial (27 factores + 15 negativos sintéticos):
    python3 -c \"from agents.paper_scout import _ACTIVE_LEARNER; print(_ACTIVE_LEARNER.stats())\"
"""

from __future__ import annotations

import sqlite3
import struct
import time
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

DB_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "climasafe.db"
EMBEDDING_DIM = 384
CONFIDENCE_THRESHOLD = 0.75
MIN_SAMPLES = 5

_embedder: Any = None


def _get_embedder():
    global _embedder
    if _embedder is None:
        from sentence_transformers import SentenceTransformer
        _embedder = SentenceTransformer("all-MiniLM-L6-v2")
    return _embedder


class ActiveLearner:
    def __init__(self):
        self._model: LogisticRegression | None = None
        self._scaler: StandardScaler | None = None
        self._classes: list[str] | None = None
        self._n_samples = 0
        self._fitted = False

    # ── Tabla ────────────────────────────────────────────────────────

    def ensure_table(self, conn: sqlite3.Connection | None = None) -> None:
        if conn is None:
            conn = sqlite3.connect(str(DB_PATH))
            own_conn = True
        else:
            own_conn = False
        conn.execute("""
            CREATE TABLE IF NOT EXISTS scout_entrenamiento (
                id           INTEGER PRIMARY KEY AUTOINCREMENT,
                titulo       TEXT NOT NULL,
                abstract     TEXT NOT NULL,
                embedding    BLOB,
                veredicto    TEXT NOT NULL CHECK(veredicto IN ('aceptable', 'irrelevante')),
                fuente       TEXT NOT NULL DEFAULT 'llm',
                created_at   TEXT NOT NULL DEFAULT (datetime('now'))
            )
        """)
        conn.commit()
        if own_conn:
            conn.close()

    # ── Embedding ────────────────────────────────────────────────────

    def embed(self, text: str) -> list[float]:
        return _get_embedder().encode(text).tolist()

    def embed_bytes(self, text: str) -> bytes:
        emb = self.embed(text)
        return struct.pack(f"{len(emb)}f", *emb)

    # ── Almacenar ejemplos ───────────────────────────────────────────

    def store(self, titulo: str, abstract: str, veredicto: str,
              fuente: str = "llm") -> None:
        conn = sqlite3.connect(str(DB_PATH))
        self.ensure_table(conn)
        emb_bytes = self.embed_bytes(f"{titulo}. {abstract}")
        conn.execute(
            "INSERT INTO scout_entrenamiento (titulo, abstract, embedding, veredicto, fuente) VALUES (?, ?, ?, ?, ?)",
            (titulo, abstract, emb_bytes, veredicto, fuente),
        )
        conn.commit()
        conn.close()

    def store_many(self, papers: list[dict]) -> int:
        conn = sqlite3.connect(str(DB_PATH))
        self.ensure_table(conn)
        count = 0
        for p in papers:
            emb_bytes = self.embed_bytes(f"{p['titulo']}. {p['abstract']}")
            conn.execute(
                "INSERT INTO scout_entrenamiento (titulo, abstract, embedding, veredicto, fuente) VALUES (?, ?, ?, ?, ?)",
                (p["titulo"], p["abstract"], emb_bytes, p["veredicto"], p.get("fuente", "llm")),
            )
            count += 1
        conn.commit()
        conn.close()
        return count

    def count_labels(self) -> int:
        conn = sqlite3.connect(str(DB_PATH))
        self.ensure_table(conn)
        n = conn.execute("SELECT COUNT(*) as n FROM scout_entrenamiento").fetchone()[0]
        conn.close()
        return n

    # ── Entrenar ─────────────────────────────────────────────────────

    def retrain(self) -> dict:
        conn = sqlite3.connect(str(DB_PATH))
        self.ensure_table(conn)
        rows = conn.execute(
            "SELECT embedding, veredicto FROM scout_entrenamiento WHERE embedding IS NOT NULL"
        ).fetchall()
        conn.close()

        if len(rows) < MIN_SAMPLES:
            self._fitted = False
            return {"fitted": False, "samples": len(rows), "min_required": MIN_SAMPLES}

        X_list, y_list = [], []
        for emb_bytes, veredicto in rows:
            emb = list(struct.unpack(f"{EMBEDDING_DIM}f", emb_bytes))
            X_list.append(emb)
            y_list.append(veredicto)

        X = np.array(X_list)
        self._scaler = StandardScaler()
        X_scaled = self._scaler.fit_transform(X)

        self._model = LogisticRegression(C=1.0, max_iter=1000, class_weight="balanced")
        self._model.fit(X_scaled, y_list)
        self._classes = self._model.classes_.tolist()
        self._n_samples = len(rows)
        self._fitted = True

        return {
            "fitted": True,
            "samples": len(rows),
            "classes": self._classes,
            "aceptables": y_list.count("aceptable"),
            "irrelevantes": y_list.count("irrelevante"),
        }

    # ── Clasificar ───────────────────────────────────────────────────

    def predict(self, titulo: str, abstract: str) -> tuple[str | None, float, dict]:
        """Returns (veredicto, confidence, info).

        veredicto is 'aceptable', 'irrelevante', or None if uncertain.
        """
        info = {"model_fitted": self._fitted, "n_samples": self._n_samples}

        if not self._fitted or self._model is None:
            return (None, 0.0, info)

        emb = self.embed(f"{titulo}. {abstract}")
        X = np.array([emb])
        X_scaled = self._scaler.transform(X)

        proba = self._model.predict_proba(X_scaled)[0]
        confidence = float(max(proba))
        idx = int(np.argmax(proba))
        label = str(self._classes[idx])

        info["confidence"] = confidence
        info["predicted_label"] = label

        if confidence < CONFIDENCE_THRESHOLD:
            return (None, confidence, info)

        return (label, confidence, info)

    # ── Métricas ─────────────────────────────────────────────────────

    def stats(self) -> dict:
        conn = sqlite3.connect(str(DB_PATH))
        self.ensure_table(conn)
        total = conn.execute("SELECT COUNT(*) as n FROM scout_entrenamiento").fetchone()[0]
        por_fuente = {
            r[0]: r[1]
            for r in conn.execute(
                "SELECT fuente, COUNT(*) as n FROM scout_entrenamiento GROUP BY fuente"
            ).fetchall()
        }
        por_veredicto = {
            r[0]: r[1]
            for r in conn.execute(
                "SELECT veredicto, COUNT(*) as n FROM scout_entrenamiento GROUP BY veredicto"
            ).fetchall()
        }
        conn.close()
        return {
            "total": total,
            "por_fuente": por_fuente,
            "por_veredicto": por_veredicto,
            "modelo_entrenado": self._fitted if hasattr(self, '_fitted') else False,
            "umbral_confianza": CONFIDENCE_THRESHOLD,
            "min_muestras": MIN_SAMPLES,
        }
