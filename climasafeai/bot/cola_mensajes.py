"""
climasafeai.bot.cola_mensajes — Cola de mensajes con N workers (MSG-003).

Cola persistente en sqlite (cero infraestructura: sin redis/rabbitmq) de la
que compiten N workers. Cada worker reserva un mensaje con una UPDATE atómica
(pendiente → enviando) y lo envía por el `MessageAdapter` de messaging.py
(MSG-001): la cola no sabe qué canal hay detrás.

Reserva atómica (la clave de la concurrencia):

    BEGIN IMMEDIATE;
    UPDATE mensajes_cola SET estado = 'enviando'
    WHERE id = (SELECT id FROM mensajes_cola
                WHERE estado = 'pendiente' ORDER BY id LIMIT 1)
    RETURNING id, destino, texto, kwargs;
    COMMIT;

`BEGIN IMMEDIATE` coge el lock de escritura antes de leer, así que la reserva
queda serializada: aunque N workers sean hilos o procesos distintos (cada uno
con su conexión), nunca reservan el mismo mensaje. Cada mensaje pasa por
pendiente → enviando → enviado (o fallido).

La conexión usa WAL + `synchronous=NORMAL`: con el journal por defecto cada
commit (reserva y marca) hacía un fsync y procesar 20 mensajes tardaba ~13 s.

Uso:

    from climasafeai.bot.cola_mensajes import ColaMensajes
    from climasafeai.bot.messaging import TelegramAdapter

    cola = ColaMensajes(adapter=TelegramAdapter())
    cola.encolar("123", "texto")
    await cola.procesar(n_workers=4)
"""

from __future__ import annotations

import asyncio
import json
import logging
import sqlite3
from pathlib import Path
from typing import Any

from climasafeai.bot.messaging import MessageAdapter

logger = logging.getLogger(__name__)

_DB_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "cola_mensajes.db"
_POLL_INTERVALO = 1.0


class ColaMensajes:
    """Cola persistente en sqlite de la que compiten N workers.

    `encolar`/`encolar_lote` meten mensajes; `procesar` lanza N workers que
    los reservan (UPDATE atómico) y los envían por el `MessageAdapter` de
    MSG-001. El fichero de la cola se inyecta en los tests; por defecto vive
    en `data/cola_mensajes.db`.
    """

    def __init__(self, adapter: MessageAdapter, db_path: str | Path | None = None) -> None:
        self.adapter = adapter
        self.db_path = Path(db_path) if db_path else _DB_PATH
        self._initialize()

    def _conn(self) -> sqlite3.Connection:
        c = sqlite3.connect(str(self.db_path), timeout=30)
        c.row_factory = sqlite3.Row
        # WAL + synchronous=NORMAL: los commits no hacen fsync por mensaje
        # (la reserva y la marca son 2 commits por mensaje).
        c.execute("PRAGMA journal_mode=WAL").fetchone()
        c.execute("PRAGMA synchronous=NORMAL").fetchone()
        return c

    def _initialize(self) -> None:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._conn() as c:
            c.execute(
                """
                CREATE TABLE IF NOT EXISTS mensajes_cola (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    destino TEXT NOT NULL,
                    texto TEXT NOT NULL,
                    kwargs TEXT,
                    estado TEXT NOT NULL DEFAULT 'pendiente'
                )
                """
            )

    # ── productor ──────────────────────────────────────────────────────────

    def encolar(self, destino: str, texto: str, **kwargs: Any) -> int:
        """Mete un mensaje en la cola y devuelve su id."""
        with self._conn() as c:
            cur = c.execute(
                "INSERT INTO mensajes_cola (destino, texto, kwargs, estado)"
                " VALUES (?, ?, ?, 'pendiente')",
                (destino, texto, json.dumps(kwargs) if kwargs else None),
            )
            lastrowid = cur.lastrowid
            assert lastrowid is not None  # tras un INSERT siempre existe
            return int(lastrowid)

    def encolar_lote(self, mensajes: list[tuple[str, str]]) -> int:
        """Encola varios mensajes (destino, texto) en una sola transacción."""
        with self._conn() as c:
            cur = c.executemany(
                "INSERT INTO mensajes_cola (destino, texto, estado) VALUES (?, ?, 'pendiente')",
                mensajes,
            )
            return cur.rowcount

    # ── consumidor ─────────────────────────────────────────────────────────

    def _reservar(self) -> dict | None:
        """Reserva el primer mensaje pendiente y lo devuelve (o None si no hay).

        `BEGIN IMMEDIATE` adquiere el lock de escritura antes de leer, así que
        dos workers con conexiones distintas nunca reclaman el mismo mensaje.
        Cada llamada usa su propia conexión y commitea al momento: el lock
        nunca queda retenido mientras se envía (el envío va fuera de la
        transacción, con el mensaje en 'enviando').
        """
        conn = self._conn()
        try:
            conn.execute("BEGIN IMMEDIATE")
            fila = conn.execute(
                """
                UPDATE mensajes_cola SET estado = 'enviando'
                WHERE id = (SELECT id FROM mensajes_cola
                            WHERE estado = 'pendiente' ORDER BY id LIMIT 1)
                RETURNING id, destino, texto, kwargs
                """
            ).fetchone()
            conn.commit()
            return dict(fila) if fila else None
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _marcar(self, mensaje_id: int, estado: str) -> None:
        with self._conn() as c:
            c.execute("UPDATE mensajes_cola SET estado = ? WHERE id = ?", (estado, mensaje_id))

    @staticmethod
    def _cargar_kwargs(raw: str | None) -> dict:
        return json.loads(raw) if raw else {}

    def _conteo_por_estado(self) -> dict[str, int]:
        """Mensajes por estado (pendiente/enviando/enviado/fallido)."""
        with self._conn() as c:
            filas = c.execute(
                "SELECT estado, COUNT(*) AS n FROM mensajes_cola GROUP BY estado"
            ).fetchall()
        return {f["estado"]: f["n"] for f in filas}

    async def worker(self, esperar: bool = False) -> int:
        """Un worker: reserva mensajes y los envía. Devuelve cuántos envió.

        Con `esperar=False` termina al vaciar la cola; con `esperar=True` se
        queda en bucle esperando mensajes nuevos (modo servicio). Cada reserva
        y cada marca usan su propia conexión, como haría un proceso separado.
        """
        enviados = 0
        while True:
            fila = self._reservar()
            if fila is None:
                if not esperar:
                    return enviados
                await asyncio.sleep(_POLL_INTERVALO)
                continue
            try:
                await self.adapter.send(
                    fila["destino"], fila["texto"], **self._cargar_kwargs(fila["kwargs"])
                )
                self._marcar(fila["id"], "enviado")
                enviados += 1
            except Exception:
                logger.exception("Mensaje %s fallido; queda marcado como fallido", fila["id"])
                self._marcar(fila["id"], "fallido")

    async def procesar(self, n_workers: int = 1, esperar: bool = False) -> list[int]:
        """Lanza N workers compitiendo por la misma cola y espera a que acaben.

        Devuelve cuántos mensajes envió cada worker.
        """
        return await asyncio.gather(*(self.worker(esperar=esperar) for _ in range(n_workers)))
