"""
climasafeai.db.manager — Gestión de la base de datos SQLite de perfiles.

Uso:
    from climasafeai.db.manager import DBManager

    db = DBManager()
    db.initialize()              # crea tablas si no existen
    pid = db.crear_perfil({...})
    perfil = db.obtener_perfil(pid)
    db.guardar_consulta(...)
"""

from __future__ import annotations

import base64
import hashlib
import json
import secrets
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

_DB_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "climasafe.db"
_SCHEMA_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "schema.sql"
_FACTORES_JSON = Path(__file__).resolve().parent.parent.parent / "data" / "factores_riesgo.json"


def nuevo_uid() -> str:
    """Identificador público opaco de un perfil: `usr_` + 16 bytes en base32.

    Sustituye a `alias` y `telegram_chat_id` como llave de acceso (MCP-003): el
    `id` autoincremental es secuencial y adivinable, y los otros dos los conoce
    cualquiera que hable con el usuario.
    """
    crudo = base64.b32encode(secrets.token_bytes(16)).decode("ascii").rstrip("=")
    return f"usr_{crudo.lower()}"


def hash_token_mcp(token: str) -> str:
    """sha256 del secreto del llamante. En BD solo se guarda el hash."""
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


class CampoDesconocidoError(ValueError):
    """El perfil trae campos que no existen en la tabla `perfiles`."""

    def __init__(self, campos: list[str]):
        self.campos = campos
        plural = "s" if len(campos) > 1 else ""
        super().__init__(
            f"campo{plural} desconocido{plural} en el perfil: {', '.join(campos)}. "
            "No existe{} en la tabla 'perfiles'.".format("n" if plural else "")
        )


class DBManager:
    def __init__(self, db_path: str | Path | None = None):
        self.db_path = Path(db_path) if db_path else _DB_PATH
        self._rag: Any = None

    @property
    def rag(self):
        if self._rag is None:
            try:
                from climasafeai.db.rag import RAG as _RAG
                self._rag = _RAG(self.db_path)
            except ImportError:
                pass
        return self._rag

    def init_rag(self) -> dict:
        if self.rag is None:
            return {"success": False, "error": "sqlite-vec no disponible (pip install '.[rag]')"}
        self.rag.initialize()
        stats = self.rag.stats()
        return {"success": True, "stats": stats}

    def sync_documentos(self) -> int:
        """Reindexa documentacion/ (solo fragmentos nuevos)."""
        if self.rag is None:
            return 0
        return self.rag.sync_documentos()

    def resync_documentos(self) -> int:
        """Borra y reindexa toda la documentacion/ desde cero."""
        if self.rag is None:
            return 0
        return self.rag.resync_documentos()

    def search_factores(self, query: str, k: int = 5) -> list[dict]:
        if self.rag is None:
            return []
        return self.rag.search_factores(query, k=k)

    def search_documentos(self, query: str, k: int = 5) -> list[dict]:
        """Búsqueda semántica sobre documentacion/."""
        if self.rag is None:
            return []
        return self.rag.search_documentos(query, k=k)

    def search_all(self, query: str, k: int = 5) -> dict:
        """Busca en factores y documentos simultáneamente."""
        if self.rag is None:
            return {"factores": [], "documentos": []}
        return self.rag.search_all(query, k=k)

    def ask_rag(self, query: str, k: int = 5) -> dict:
        """RAG completo: retrieve + generate."""
        if self.rag is None:
            return {"answer": None, "sources": [], "error": "RAG no disponible"}
        return self.rag.ask(query, k=k)

    # ── Conexión ────────────────────────────────────────────────────

    @contextmanager
    def conn(self) -> Iterator[sqlite3.Connection]:
        c = sqlite3.connect(str(self.db_path))
        c.row_factory = sqlite3.Row
        c.execute("PRAGMA journal_mode=WAL")
        c.execute("PRAGMA foreign_keys=ON")
        try:
            yield c
            c.commit()
        except Exception:
            c.rollback()
            raise
        finally:
            c.close()

    # ── Inicialización ──────────────────────────────────────────────

    def initialize(self) -> None:
        """Crea las tablas si no existen y migra si es necesario."""
        sql = _SCHEMA_PATH.read_text(encoding="utf-8")
        with self.conn() as c:
            c.executescript(sql)
        self._migrate()

    def _migrate(self) -> None:
        """Migraciones post-creación de schema."""
        with self.conn() as c:
            cols = [r["name"] for r in c.execute("PRAGMA table_info(perfiles)").fetchall()]
            if "fecha_nacimiento" not in cols:
                c.execute("ALTER TABLE perfiles ADD COLUMN fecha_nacimiento TEXT")
            if "tags" not in cols:
                c.execute("ALTER TABLE perfiles ADD COLUMN tags TEXT")
            if "telegram_chat_id" not in cols:
                c.execute("ALTER TABLE perfiles ADD COLUMN telegram_chat_id TEXT")
                c.execute("CREATE INDEX IF NOT EXISTS idx_perfiles_telegram ON perfiles(telegram_chat_id)")
            # MCP-003: identidad del llamante del MCP. `uid` no puede declararse
            # UNIQUE en un ALTER (SQLite no lo permite), así que la unicidad la
            # da el índice; se crea siempre porque en BD nuevas la columna ya
            # viene de schema.sql y el ALTER no llega a ejecutarse.
            if "uid" not in cols:
                c.execute("ALTER TABLE perfiles ADD COLUMN uid TEXT")
            if "mcp_token_hash" not in cols:
                c.execute("ALTER TABLE perfiles ADD COLUMN mcp_token_hash TEXT")
            if "rol" not in cols:
                c.execute("ALTER TABLE perfiles ADD COLUMN rol TEXT NOT NULL DEFAULT 'usuario'")
            # BOT-017: la última salida usada del chat, para ofrecer repetirla.
            # Se guarda como JSON en una columna TEXT: el perfil no necesita una
            # tabla propia para un único blob que se sobreescribe en cada /start.
            if "ultima_salida" not in cols:
                c.execute("ALTER TABLE perfiles ADD COLUMN ultima_salida TEXT")
            c.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_perfiles_uid ON perfiles(uid)")
            c.execute("CREATE INDEX IF NOT EXISTS idx_perfiles_mcp_token ON perfiles(mcp_token_hash)")
            # Backfill obligatorio: NULL no colisiona con UNIQUE en SQLite, así
            # que sin esto los perfiles previos quedarían indistinguibles por su
            # llave. Es idempotente (solo toca los NULL) y va en la misma
            # transacción que el ALTER.
            huerfanos = [r["id"] for r in c.execute(
                "SELECT id FROM perfiles WHERE uid IS NULL OR uid = ''"
            ).fetchall()]
            for pid in huerfanos:
                c.execute("UPDATE perfiles SET uid = ? WHERE id = ?", (nuevo_uid(), pid))
            c.execute("UPDATE perfiles SET rol = 'usuario' WHERE rol IS NULL OR rol = ''")

    def columnas_perfiles(self) -> set[str]:
        """Columnas reales de la tabla `perfiles`, leídas del esquema."""
        with self.conn() as c:
            return {r["name"] for r in c.execute("PRAGMA table_info(perfiles)").fetchall()}

    def _validar_campos_perfil(self, escalares: dict) -> None:
        """Rechaza claves que no son columnas, antes de que lo haga sqlite.

        Sin esto, un campo de más en el perfil llega al INSERT y sale como
        `sqlite3.OperationalError: table perfiles has no column named X`, que en la
        API se convierte en un 500 mudo. Escribir mal una clave ya ha costado dos
        veces en este proyecto (`medicacion` por `farmacos`, `grasa_corporal` por
        `porcentaje_grasa`), y las dos se descubrieron tarde porque nadie avisaba.
        """
        desconocidos = sorted(set(escalares) - self.columnas_perfiles())
        if desconocidos:
            raise CampoDesconocidoError(desconocidos)

    def tablas(self) -> list[str]:
        with self.conn() as c:
            rows = c.execute(
                "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
            ).fetchall()
            return [r["name"] for r in rows]

    # ── Perfiles ────────────────────────────────────────────────────

    def crear_perfil(self, datos: dict) -> int:
        """Inserta un perfil y devuelve su id.

        ``datos`` acepta los mismos campos que el frontend envía a
        ``/api/predict`` (edad, sexo, comorbilidades como lista, etc.).
        Los campos array (comorbilidades, farmacos, situacion_social,
        ocupacional) se insertan en sus tablas many-to-many.

        Si ``datos`` está vacío, crea un perfil vacío (solo timestamps).
        """
        array_fields = {"comorbilidades", "farmacos", "situacion_social", "ocupacional"}
        escalares = {k: v for k, v in datos.items() if k not in array_fields}

        self._validar_campos_perfil(escalares)

        # MCP-003: todo perfil nace con identificador público opaco.
        escalares.setdefault("uid", nuevo_uid())

        # Booleans: convertir True/False a 1/0
        for k in ("aclimatado", "falta_sueno", "enfermedad_reciente", "alcohol_reciente", "fiesta"):
            if k in escalares:
                escalares[k] = 1 if escalares[k] else 0

        # Entrenado: de booleano a texto "si"/"no"
        if "entrenado" in escalares:
            escalares["entrenado"] = "si" if escalares["entrenado"] else "no"

        with self.conn() as c:
            if not escalares:
                # Perfil vacío: insertar solo created_at
                cur = c.execute("INSERT INTO perfiles DEFAULT VALUES")
            else:
                incluir_aclimatado_ts = "aclimatado" in escalares
                columnas = ", ".join(escalares.keys())
                if incluir_aclimatado_ts:
                    columnas += ", aclimatado_actualizado_en"
                placeholders = ", ".join("?" for _ in escalares)
                if incluir_aclimatado_ts:
                    placeholders += ", datetime('now')"
                cur = c.execute(
                    f"INSERT INTO perfiles ({columnas}) VALUES ({placeholders})",
                    list(escalares.values()),
                )
            pid = cur.lastrowid

            # Arrays
            for campo in array_fields:
                vals = datos.get(campo)
                if not vals:
                    continue
                tabla = f"perfil_{campo}"
                c.executemany(
                    f"INSERT OR IGNORE INTO {tabla} (perfil_id, clave) VALUES (?, ?)",
                    [(pid, v) for v in vals],
                )

        return pid

    def obtener_perfil(self, perfil_id: int) -> dict | None:
        """Devuelve el perfil completo (escalares + arrays) o None."""
        with self.conn() as c:
            row = c.execute(
                "SELECT * FROM perfiles WHERE id = ?", (perfil_id,)
            ).fetchone()
            if row is None:
                return None

            perfil = dict(row)
            # El hash del token MCP es una credencial: no sale del getter
            # genérico, que alimenta la web, el bot y las tools (MCP-003).
            perfil.pop("mcp_token_hash", None)
            # Booleans: convertir 1/0 a True/False
            for k in ("aclimatado", "falta_sueno", "enfermedad_reciente", "alcohol_reciente", "fiesta"):
                if perfil.get(k) is not None:
                    perfil[k] = bool(perfil[k])

            # Entrenado: de texto "si"/"no" a booleano
            if "entrenado" in perfil:
                perfil["entrenado"] = perfil["entrenado"] == "si"

            # BOT-017: la última salida se guarda como JSON; se devuelve como
            # dict. JSON roto o NULL se degrada a None (mejor perder el atajo
            # que romper el perfil por un blob corrupto).
            if perfil.get("ultima_salida"):
                try:
                    perfil["ultima_salida"] = json.loads(perfil["ultima_salida"])
                except (ValueError, TypeError):
                    perfil.pop("ultima_salida", None)

            # Arrays
            for campo, tabla in (
                ("comorbilidades", "perfil_comorbilidades"),
                ("farmacos", "perfil_farmacos"),
                ("situacion_social", "perfil_situacion_social"),
                ("ocupacional", "perfil_ocupacional"),
            ):
                rows = c.execute(
                    f"SELECT clave FROM {tabla} WHERE perfil_id = ?", (perfil_id,)
                ).fetchall()
                perfil[campo] = [r["clave"] for r in rows]

            return perfil

    def listar_perfiles(self) -> list[dict]:
        """Todos los perfiles (sin arrays, solo cabecera)."""
        with self.conn() as c:
            rows = c.execute(
                "SELECT id, uid, rol, alias, edad, sexo, lat, lon, provincia, tags, telegram_chat_id, created_at, updated_at FROM perfiles ORDER BY updated_at DESC"
            ).fetchall()
            return [dict(r) for r in rows]

    def actualizar_perfil(self, perfil_id: int, datos: dict) -> bool:
        """Actualiza campos de un perfil. Reemplaza arrays completamente."""
        array_fields = {"comorbilidades", "farmacos", "situacion_social", "ocupacional"}
        escalares = {k: v for k, v in datos.items() if k not in array_fields}

        self._validar_campos_perfil(escalares)

        nuevo_aclimatado = None
        if "aclimatado" in escalares:
            nuevo_aclimatado = escalares["aclimatado"]
            escalares["aclimatado"] = 1 if escalares["aclimatado"] else 0

        for k in ("falta_sueno", "enfermedad_reciente", "alcohol_reciente", "fiesta"):
            if k in escalares:
                escalares[k] = 1 if escalares[k] else 0

        if "entrenado" in escalares:
            escalares["entrenado"] = "si" if escalares["entrenado"] else "no"

        # BOT-017: la última salida es un dict y se guarda serializada en la
        # columna TEXT; `_validar_campos_perfil` ya la acepta como columna.
        if isinstance(escalares.get("ultima_salida"), dict):
            escalares["ultima_salida"] = json.dumps(escalares["ultima_salida"], ensure_ascii=False)

        with self.conn() as c:
            if escalares:
                escalares["updated_at"] = "datetime('now')"
                # Si cambió aclimatado, también actualizar aclimatado_actualizado_en
                if nuevo_aclimatado is not None:
                    escalares["aclimatado_actualizado_en"] = "datetime('now')"

                set_clause = ", ".join(
                    f"{k} = ?" if k != "updated_at" and k != "aclimatado_actualizado_en" else f"{k} = datetime('now')"
                    for k in escalares
                )
                vals = [v for k, v in escalares.items() if k != "updated_at" and k != "aclimatado_actualizado_en"]
                vals.append(perfil_id)
                cur = c.execute(
                    f"UPDATE perfiles SET {set_clause} WHERE id = ?", vals
                )
                if cur.rowcount == 0:
                    return False

            for campo in array_fields:
                if campo not in datos:
                    continue
                tabla = f"perfil_{campo}"
                c.execute(f"DELETE FROM {tabla} WHERE perfil_id = ?", (perfil_id,))
                vals = datos.get(campo)
                if vals:
                    c.executemany(
                        f"INSERT INTO {tabla} (perfil_id, clave) VALUES (?, ?)",
                        [(perfil_id, v) for v in vals],
                    )
            return True

    def eliminar_perfil(self, perfil_id: int) -> bool:
        with self.conn() as c:
            cur = c.execute("DELETE FROM perfiles WHERE id = ?", (perfil_id,))
            return cur.rowcount > 0

    # ── Rutinas semanales (BOT-007) ─────────────────────────────────

    def crear_rutina(
        self,
        chat_id: str,
        nombre: str,
        dias: str,
        hora_inicio: float,
        hora_fin: float,
        ocupacion: str | None = None,
        deporte: str | None = None,
    ) -> int:
        """Inserta una rutina semanal del chat y devuelve su id.

        ``dias`` es una cadena coma-separada con los días de la semana en
        formato 1-7 (1=lunes, 7=domingo), ej: "1,2,3,4,5".
        """
        with self.conn() as c:
            cur = c.execute(
                """INSERT INTO rutinas (chat_id, nombre, dias, hora_inicio, hora_fin, ocupacion, deporte)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (str(chat_id), nombre, dias, hora_inicio, hora_fin, ocupacion, deporte),
            )
            return cur.lastrowid

    def listar_rutinas(self, chat_id: str) -> list[dict]:
        """Todas las rutinas de un chat, ordenadas por hora de inicio."""
        with self.conn() as c:
            rows = c.execute(
                "SELECT * FROM rutinas WHERE chat_id = ? ORDER BY hora_inicio, id",
                (str(chat_id),),
            ).fetchall()
            return [dict(r) for r in rows]

    def rutinas_por_dia(self, chat_id: str, weekday: int) -> list[dict]:
        """Rutinas de un chat que caen en un día concreto (1=lunes ... 7=domingo)."""
        resultado = []
        for r in self.listar_rutinas(chat_id):
            dias = [int(d) for d in r["dias"].split(",") if d.strip()]
            if weekday in dias:
                resultado.append(r)
        return resultado

    def eliminar_rutina(self, rutina_id: int) -> bool:
        with self.conn() as c:
            cur = c.execute("DELETE FROM rutinas WHERE id = ?", (rutina_id,))
            return cur.rowcount > 0

    # ── Hora de aviso diario (BOT-007) ───────────────────────────────

    def obtener_hora_aviso(self, chat_id: str) -> str | None:
        """Hora de aviso configurada ('HH:MM') o None si no hay."""
        with self.conn() as c:
            row = c.execute(
                "SELECT hora FROM avisos_config WHERE chat_id = ?", (str(chat_id),)
            ).fetchone()
            return row["hora"] if row else None

    def guardar_hora_aviso(self, chat_id: str, hora: str | None) -> None:
        """Configura la hora de aviso de un chat; ``hora=None`` la desactiva."""
        with self.conn() as c:
            if hora is None:
                c.execute("DELETE FROM avisos_config WHERE chat_id = ?", (str(chat_id),))
                return
            c.execute(
                """INSERT INTO avisos_config (chat_id, hora, updated_at)
                   VALUES (?, ?, datetime('now'))
                   ON CONFLICT(chat_id) DO UPDATE SET
                       hora = excluded.hora, updated_at = datetime('now')""",
                (str(chat_id), hora),
            )

    def chats_con_aviso(self) -> list[dict]:
        """Chats con hora de aviso configurada (chat_id, hora)."""
        with self.conn() as c:
            rows = c.execute("SELECT chat_id, hora FROM avisos_config ORDER BY chat_id").fetchall()
            return [dict(r) for r in rows]

    # ── Tags disponibles ────────────────────────────────────────────

    def listar_tags_disponibles(self) -> list[dict]:
        with self.conn() as c:
            return c.execute(
                "SELECT id, nombre FROM tags_disponibles ORDER BY nombre"
            ).fetchall()

    def crear_tag_disponible(self, nombre: str) -> int:
        with self.conn() as c:
            c.execute("INSERT OR IGNORE INTO tags_disponibles (nombre) VALUES (?)", (nombre,))
            row = c.execute("SELECT id FROM tags_disponibles WHERE nombre = ?", (nombre,)).fetchone()
            return row["id"] if row else -1

    def eliminar_tag_disponible(self, tag_id: int) -> bool:
        with self.conn() as c:
            c.execute("DELETE FROM tags_disponibles WHERE id = ?", (tag_id,))
            return c.rowcount > 0

    def buscar_por_alias(self, alias: str) -> dict | None:
        """Busca un perfil por alias exacto."""
        with self.conn() as c:
            row = c.execute(
                "SELECT id, alias, updated_at FROM perfiles WHERE alias = ?", (alias,)
            ).fetchone()
            return dict(row) if row else None

    def buscar_por_telegram(self, chat_id: str) -> dict | None:
        """Busca un perfil vinculado a un chat_id de Telegram."""
        with self.conn() as c:
            row = c.execute(
                "SELECT id, alias, telegram_chat_id, updated_at FROM perfiles WHERE telegram_chat_id = ?",
                (chat_id,),
            ).fetchone()
            if row is None:
                return None
            return dict(row)

    # ── Identidad MCP (MCP-003) ─────────────────────────────────────

    def buscar_por_uid(self, uid: str) -> dict | None:
        """Busca un perfil por su identificador público opaco."""
        with self.conn() as c:
            row = c.execute(
                "SELECT id, uid, alias, rol FROM perfiles WHERE uid = ?", (uid,)
            ).fetchone()
            return dict(row) if row else None

    def buscar_por_token_mcp(self, token: str) -> dict | None:
        """Perfil completo del dueño de un token MCP, o None si no lo hay.

        Único sitio que compara el secreto contra `mcp_token_hash`; el hash no
        sale nunca de aquí.
        """
        if not token:
            return None
        with self.conn() as c:
            row = c.execute(
                "SELECT id FROM perfiles WHERE mcp_token_hash = ?",
                (hash_token_mcp(token),),
            ).fetchone()
        return self.obtener_perfil(row["id"]) if row else None

    def emitir_token_mcp(self, perfil_id: int, rol: str | None = None) -> str:
        """Emite un secreto MCP para un perfil y guarda solo su hash.

        Devuelve el secreto en claro **una sola vez**: no hay forma de volver a
        leerlo. Reemitir invalida el anterior.
        """
        token = secrets.token_urlsafe(32)
        with self.conn() as c:
            if rol:
                cur = c.execute(
                    "UPDATE perfiles SET mcp_token_hash = ?, rol = ? WHERE id = ?",
                    (hash_token_mcp(token), rol, perfil_id),
                )
            else:
                cur = c.execute(
                    "UPDATE perfiles SET mcp_token_hash = ? WHERE id = ?",
                    (hash_token_mcp(token), perfil_id),
                )
            if cur.rowcount == 0:
                raise ValueError(f"No existe el perfil {perfil_id}")
        return token

    def buscar_por_tag(self, tag: str) -> list[dict]:
        """Busca perfiles que contengan una etiqueta (tags separados por coma).
        Incluye arrays (comorbilidades, fármacos, etc.) y conversión de booleanos
        igual que obtener_perfil()."""
        with self.conn() as c:
            rows = c.execute(
                "SELECT * FROM perfiles WHERE tags IS NOT NULL AND tags != ''"
            ).fetchall()
            result = []
            for r in rows:
                r = dict(r)
                r.pop("mcp_token_hash", None)  # credencial, nunca sale (MCP-003)
                tags = [t.strip() for t in (r.get("tags") or "").split(",") if t.strip()]
                if tag not in tags:
                    continue
                # Booleans
                for k in ("aclimatado", "falta_sueno", "enfermedad_reciente", "alcohol_reciente", "fiesta"):
                    if r.get(k) is not None:
                        r[k] = bool(r[k])
                if "entrenado" in r:
                    r["entrenado"] = r["entrenado"] == "si"
                # Arrays M2M
                pid = r.get("id")
                if pid:
                    for campo, tabla in (
                        ("comorbilidades", "perfil_comorbilidades"),
                        ("farmacos", "perfil_farmacos"),
                        ("situacion_social", "perfil_situacion_social"),
                        ("ocupacional", "perfil_ocupacional"),
                    ):
                        rows_m2m = c.execute(
                            f"SELECT clave FROM {tabla} WHERE perfil_id = ?", (pid,)
                        ).fetchall()
                        r[campo] = [rr["clave"] for rr in rows_m2m]
                result.append(r)
            return result

    # ── Factores de riesgo ──────────────────────────────────────────

    def obtener_factores(self, solo_implementados: bool = True, tipo: str | None = None) -> dict:
        """Devuelve factores agrupados por tipo/categoria (misma estructura que antes daba el JSON)."""
        where = ["1=1"]
        params: list = []
        if solo_implementados:
            where.append("implementado = 1")
        if tipo:
            where.append("tipo = ?")
            params.append(tipo)
        sql = f"""
            SELECT tipo, categoria, clave, nombre, coef, doi, calidad, poblacion, implementado
            FROM factores_riesgo
            WHERE {' AND '.join(where)}
            ORDER BY tipo, categoria, clave
        """
        with self.conn() as c:
            rows = c.execute(sql, params).fetchall()
        result: dict = {}
        for r in rows:
            r = dict(r)
            t = r.pop("tipo")
            cat = r.pop("categoria")
            result.setdefault(t, {}).setdefault(cat, []).append(r)
        return result

    def sugerir_factor(self, tipo: str, categoria: str, clave: str, nombre: str,
                       coef: float, doi: str | None = None, calidad: str = "baja",
                       poblacion: str | None = None) -> dict:
        """Inserta un factor con implementado=0. No sobreescribe si ya existe implementado."""
        with self.conn() as c:
            exist = c.execute(
                "SELECT implementado FROM factores_riesgo WHERE tipo=? AND categoria=? AND clave=?",
                (tipo, categoria, clave),
            ).fetchone()
            if exist and exist["implementado"]:
                return {"success": False, "error": f"'{clave}' ya existe y está implementado"}

            c.execute(
                """INSERT INTO factores_riesgo (tipo, categoria, clave, nombre, coef, doi, calidad, poblacion, implementado)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, 0)
                   ON CONFLICT(tipo, categoria, clave) DO UPDATE SET
                       nombre=excluded.nombre, coef=excluded.coef, doi=excluded.doi,
                       calidad=excluded.calidad, poblacion=excluded.poblacion, updated_at=datetime('now')""",
                (tipo, categoria, clave, nombre, coef, doi, calidad, poblacion),
            )
        return {"success": True}

    def aprobar_factor(self, tipo: str, categoria: str, clave: str) -> dict:
        """Marca implementado=1."""
        with self.conn() as c:
            cur = c.execute(
                "UPDATE factores_riesgo SET implementado=1, updated_at=datetime('now') WHERE tipo=? AND categoria=? AND clave=?",
                (tipo, categoria, clave),
            )
            if cur.rowcount == 0:
                return {"success": False, "error": f"'{clave}' no encontrado en {tipo}/{categoria}"}
        return {"success": True}

    def actualizar_factor(self, tipo: str, categoria: str, clave: str,
                          **kwargs: Any) -> dict:
        """Actualiza campos de un factor (coef, nombre, doi, calidad, poblacion)."""
        permitidos = {"coef", "nombre", "doi", "calidad", "poblacion"}
        cambios = {k: v for k, v in kwargs.items() if k in permitidos}
        if not cambios:
            return {"success": False, "error": "No hay campos válidos para actualizar"}
        cambios["updated_at"] = "datetime('now')"
        set_clause = ", ".join(
            f"{k} = ?" if k != "updated_at" else f"{k} = datetime('now')"
            for k in cambios
        )
        vals = [v for k, v in cambios.items() if k != "updated_at"]
        vals.extend([tipo, categoria, clave])
        with self.conn() as c:
            cur = c.execute(
                f"UPDATE factores_riesgo SET {set_clause} WHERE tipo=? AND categoria=? AND clave=?",
                vals,
            )
            if cur.rowcount == 0:
                return {"success": False, "error": f"'{clave}' no encontrado en {tipo}/{categoria}"}
        return {"success": True}

    def rechazar_factor(self, tipo: str, categoria: str, clave: str) -> dict:
        """Elimina un factor."""
        with self.conn() as c:
            cur = c.execute(
                "DELETE FROM factores_riesgo WHERE tipo=? AND categoria=? AND clave=?",
                (tipo, categoria, clave),
            )
            if cur.rowcount == 0:
                return {"success": False, "error": f"'{clave}' no encontrado"}
        return {"success": True}

    def factores_pendientes(self) -> list[dict]:
        """Factores con implementado=0."""
        with self.conn() as c:
            rows = c.execute(
                "SELECT tipo, categoria, clave, nombre, coef, doi, calidad FROM factores_riesgo WHERE implementado=0"
            ).fetchall()
            return [dict(r) for r in rows]

    # ── Historial ───────────────────────────────────────────────────

    def guardar_consulta(self, perfil_id: int | None = None, provincia: str = "Madrid",
                         lat: float | None = None, lon: float | None = None,
                         tipo_riesgo: str | None = None,
                         indice_original: float | None = None,
                         indice_personalizado: float | None = None,
                         clase_final: str | None = None) -> int:
        with self.conn() as c:
            cur = c.execute(
                """INSERT INTO historial_consultas
                   (perfil_id, provincia, lat, lon, tipo_riesgo, indice_original, indice_personalizado, clase_final)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (perfil_id, provincia, lat, lon, tipo_riesgo, indice_original, indice_personalizado, clase_final),
            )
            return cur.lastrowid

    def ultimas_consultas(self, limite: int = 20) -> list[dict]:
        with self.conn() as c:
            rows = c.execute(
                """SELECT h.*, p.alias
                   FROM historial_consultas h
                   LEFT JOIN perfiles p ON h.perfil_id = p.id
                   ORDER BY h.created_at DESC LIMIT ?""",
                (limite,),
            ).fetchall()
            return [dict(r) for r in rows]

    # ── Auto-aclimatación ────────────────────────────────────────────

    DIAS_ACLIMATACION = 14
    ACLIMATACION_EVIDENCIA = (
        "Karlsen et al. 2015 (DOI: 10.1111/sms.12449, Scandinavian Journal of "
        "Medicine and Science in Sports). Ciclistas entrenados en campamento de "
        "calor natural (34°C). Adaptaciones sudoromotoras y hematológicas en 5-6 "
        "días. Mejora de rendimiento progresiva durante 14 días completos. "
        "Conclusión: aclimatación completa a los 14 días en el mismo entorno."
    )

    def perfiles_para_aclimatar(self, dias: int | None = None) -> list[dict]:
        """Devuelve perfiles con aclimatado=false que ya deberían estar aclimatados
        según el tiempo transcurrido desde que se marcaron como no aclimatados.

        La evidencia (Karlsen 2015, DOI: 10.1111/sms.12449) muestra que las
        adaptaciones fisiológicas al calor se completan en ~14 días.
        """
        umbral = dias or self.DIAS_ACLIMATACION
        with self.conn() as c:
            rows = c.execute("""
                SELECT id, alias, edad, sexo, aclimatado_actualizado_en,
                       datetime('now') as ahora
                FROM perfiles
                WHERE aclimatado = 0
                  AND aclimatado_actualizado_en IS NOT NULL
                  AND julianday('now') - julianday(aclimatado_actualizado_en) >= ?
                ORDER BY aclimatado_actualizado_en ASC
            """, (umbral,)).fetchall()
            return [dict(r) for r in rows]

    def auto_aclimatar(self, perfil_id: int | None = None,
                       dias: int | None = None) -> dict:
        """Marca como aclimatados los perfiles que cumplan el criterio temporal.

        Args:
            perfil_id: si se pasa, solo ese perfil. Si None, todos los que cumplan.
            dias: días mínimos desde que se marcó no_aclimatado (def: 14).

        Returns:
            dict con perfiles_aclimatados, fallos.
        """
        candidatos = self.perfiles_para_aclimatar(dias=dias)
        if perfil_id is not None:
            candidatos = [c for c in candidatos if c["id"] == perfil_id]

        aclimatados = 0
        for c in candidatos:
            ok = self.actualizar_perfil(c["id"], {"aclimatado": True})
            if ok:
                aclimatados += 1

        return {
            "aclimatados": aclimatados,
            "total_candidatos": len(candidatos),
            "dias_umbral": dias or self.DIAS_ACLIMATACION,
            "evidencia": self.ACLIMATACION_EVIDENCIA,
        }

    # ── Migración desde JSON ─────────────────────────────────────────

    def migrar_desde_json(self, factores_json: str | Path | None = None) -> dict:
        """Vuelca el contenido de ``factores_riesgo.json`` a SQLite.

        Es seguro ejecutar varias veces (ON CONFLICT actualiza).
        Devuelve conteo de filas insertadas/actualizadas.
        """
        path = Path(factores_json) if factores_json else _FACTORES_JSON
        if not path.exists():
            return {"error": f"No se encuentra {path}", "insertados": 0}
        import json
        data = json.loads(path.read_text(encoding="utf-8"))
        version = data.get("version", 1)
        cap = data.get("cap_factores", 3.0)
        total = 0
        with self.conn() as c:
            # Guardar metadatos en una tabla propia
            c.execute(
                "CREATE TABLE IF NOT EXISTS _meta (key TEXT PRIMARY KEY, value TEXT)"
            )
            c.execute(
                "INSERT OR REPLACE INTO _meta (key, value) VALUES (?, ?)",
                ("factores_version", str(version)),
            )
            c.execute(
                "INSERT OR REPLACE INTO _meta (key, value) VALUES (?, ?)",
                ("cap_factores", str(cap)),
            )
            for tipo in ("calor", "frio"):
                seccion = data.get(tipo, {})
                for categoria, factores in seccion.items():
                    if not isinstance(factores, dict):
                        continue
                    for clave, info in factores.items():
                        if not isinstance(info, dict):
                            continue
                        c.execute(
                            """INSERT INTO factores_riesgo
                               (tipo, categoria, clave, nombre, coef, doi, calidad, implementado)
                               VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                               ON CONFLICT(tipo, categoria, clave) DO UPDATE SET
                                   nombre=excluded.nombre, coef=excluded.coef, doi=excluded.doi,
                                   calidad=excluded.calidad, implementado=excluded.implementado,
                                   updated_at=datetime('now')""",
                            (
                                tipo,
                                categoria,
                                clave,
                                info.get("nombre", clave),
                                info["coef"],
                                info.get("doi"),
                                info.get("calidad", "baja"),
                                1 if info.get("implementado") else 0,
                            ),
                        )
                        total += 1
        return {"insertados": total, "version": version, "cap_factores": cap}
