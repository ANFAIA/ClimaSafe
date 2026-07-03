"""agents.tools.sqlite_tool — sqlite3 de la librería estándar, sin dependencias nuevas."""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from agents.tools.registry import register_tool


@register_tool("sqlite")
class SQLiteTool:
    @staticmethod
    @contextmanager
    def connect(db_path: Path) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(db_path)
        try:
            yield conn
        finally:
            conn.close()

    @staticmethod
    def query(db_path: Path, sql: str, params: tuple = ()) -> list[tuple]:
        with SQLiteTool.connect(db_path) as conn:
            cursor = conn.execute(sql, params)
            return cursor.fetchall()

    @staticmethod
    def list_tables(db_path: Path) -> list[str]:
        rows = SQLiteTool.query(db_path, "SELECT name FROM sqlite_master WHERE type='table'")
        return [row[0] for row in rows]

    @staticmethod
    def table_schema(db_path: Path, table: str) -> list[dict[str, Any]]:
        # PRAGMA no admite parámetros ligados (?) en sqlite3 — se valida el
        # nombre de tabla contra la lista real de tablas antes de interpolar,
        # para no abrir la puerta a inyección SQL vía `table`.
        if table not in SQLiteTool.list_tables(db_path):
            raise ValueError(f"La tabla '{table}' no existe en {db_path}.")
        rows = SQLiteTool.query(db_path, f"PRAGMA table_info({table})")
        columns = ["cid", "name", "type", "notnull", "default_value", "pk"]
        return [dict(zip(columns, row)) for row in rows]
