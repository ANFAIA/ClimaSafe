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

import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

_DB_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "climasafe.db"
_SCHEMA_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "schema.sql"
_FACTORES_JSON = Path(__file__).resolve().parent.parent.parent / "data" / "factores_riesgo.json"


class DBManager:
    def __init__(self, db_path: str | Path | None = None):
        self.db_path = Path(db_path) if db_path else _DB_PATH

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
        """Crea las tablas si no existen."""
        sql = _SCHEMA_PATH.read_text(encoding="utf-8")
        with self.conn() as c:
            c.executescript(sql)

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

        # Booleans: convertir True/False a 1/0
        for k in ("aclimatado", "falta_sueno", "enfermedad_reciente", "alcohol_reciente"):
            if k in escalares:
                escalares[k] = 1 if escalares[k] else 0

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
            # Booleans: convertir 1/0 a True/False
            for k in ("aclimatado", "falta_sueno", "enfermedad_reciente", "alcohol_reciente"):
                if perfil.get(k) is not None:
                    perfil[k] = bool(perfil[k])

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
                "SELECT id, alias, edad, sexo, created_at, updated_at FROM perfiles ORDER BY updated_at DESC"
            ).fetchall()
            return [dict(r) for r in rows]

    def actualizar_perfil(self, perfil_id: int, datos: dict) -> bool:
        """Actualiza campos de un perfil. Reemplaza arrays completamente."""
        array_fields = {"comorbilidades", "farmacos", "situacion_social", "ocupacional"}
        escalares = {k: v for k, v in datos.items() if k not in array_fields}

        nuevo_aclimatado = None
        if "aclimatado" in escalares:
            nuevo_aclimatado = escalares["aclimatado"]
            escalares["aclimatado"] = 1 if escalares["aclimatado"] else 0

        for k in ("falta_sueno", "enfermedad_reciente", "alcohol_reciente"):
            if k in escalares:
                escalares[k] = 1 if escalares[k] else 0

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
