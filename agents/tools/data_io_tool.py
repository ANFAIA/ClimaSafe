"""
agents.tools.data_io_tool — Lectura/escritura de CSV, JSON y Parquet.

pandas, numpy y scikit-learn están en las dependencias base del template
(`[project].dependencies` en pyproject.toml), así que se importan
directamente sin guardas. Parquet, en cambio, necesita `pyarrow`, que solo
se instala si el usuario activó `use_duckdb` en `copier.yml` — por eso ese
caso sí se protege con un mensaje claro en vez de un ImportError crudo.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from agents.exceptions import MissingDependencyError
from agents.tools.registry import register_tool


@register_tool("data_io")
class DataIOTool:
    @staticmethod
    def read_csv(path: Path, **kwargs: Any) -> pd.DataFrame:
        return pd.read_csv(path, **kwargs)

    @staticmethod
    def read_json(path: Path) -> Any:
        return json.loads(Path(path).read_text(encoding="utf-8"))

    @staticmethod
    def write_json(path: Path, data: Any, *, indent: int = 2) -> None:
        Path(path).write_text(json.dumps(data, indent=indent, ensure_ascii=False), encoding="utf-8")

    @staticmethod
    def read_parquet(path: Path, **kwargs: Any) -> pd.DataFrame:
        try:
            return pd.read_parquet(path, **kwargs)
        except ImportError as exc:
            raise MissingDependencyError(
                "Leer Parquet requiere 'pyarrow'. Este extra se instala con "
                "use_duckdb=true en copier.yml, o añádelo manualmente: "
                "uv add pyarrow"
            ) from exc

    @staticmethod
    def infer_reader(path: Path):
        """Devuelve la función de lectura adecuada según la extensión del archivo."""
        suffix = Path(path).suffix.lower()
        readers = {
            ".csv": DataIOTool.read_csv,
            ".json": DataIOTool.read_json,
            ".parquet": DataIOTool.read_parquet,
        }
        if suffix not in readers:
            raise ValueError(f"Extensión no soportada: '{suffix}' (esperaba .csv, .json o .parquet)")
        return readers[suffix]
