"""
agents.tools.duckdb_tool — Consultas SQL sobre CSV/Parquet/JSON con DuckDB.

`duckdb` solo se instala si el proyecto activó `use_duckdb=true` en
copier.yml, así que el import es perezoso (dentro del método, no a nivel de
módulo) para que importar `agents` no falle en proyectos sin ese extra.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from agents.exceptions import MissingDependencyError
from agents.tools.registry import register_tool


@register_tool("duckdb")
class DuckDBTool:
    @staticmethod
    def _connect():
        try:
            import duckdb
        except ImportError as exc:
            raise MissingDependencyError(
                "DuckDB no está instalado. Actívalo con use_duckdb=true en copier.yml, "
                "o instálalo manualmente: uv add duckdb"
            ) from exc
        return duckdb.connect()

    @staticmethod
    def query(sql: str) -> Any:
        """
        Ejecuta `sql` y devuelve un DataFrame de pandas.
        Ejemplo: DuckDBTool.query("SELECT * FROM read_csv('data/raw/dataset.csv') LIMIT 10")
        """
        conn = DuckDBTool._connect()
        try:
            return conn.execute(sql).df()
        finally:
            conn.close()

    @staticmethod
    def describe_file(path: Path) -> Any:
        """Perfil rápido de un CSV/Parquet vía DESCRIBE de DuckDB (tipos inferidos, no estadísticas)."""
        reader = "read_parquet" if path.suffix.lower() == ".parquet" else "read_csv_auto"
        return DuckDBTool.query(f"DESCRIBE SELECT * FROM {reader}('{path.as_posix()}')")
