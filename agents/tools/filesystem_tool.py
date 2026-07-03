"""
agents.tools.filesystem_tool — Operaciones de archivo con la raíz del
proyecto como límite de seguridad.

Todos los métodos rechazan rutas que intenten escapar de `root` (p. ej. vía
`../../etc/passwd`), para que un agente que reciba una ruta mal formada no
pueda tocar nada fuera del proyecto.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


class PathEscapesRootError(Exception):
    pass


@dataclass
class FilesystemTool:
    root: Path

    def _resolve(self, relative_or_absolute: str | Path) -> Path:
        candidate = Path(relative_or_absolute)
        full = candidate if candidate.is_absolute() else self.root / candidate
        full = full.resolve()
        try:
            full.relative_to(self.root.resolve())
        except ValueError:
            raise PathEscapesRootError(
                f"'{relative_or_absolute}' queda fuera de la raíz del proyecto ({self.root})."
            ) from None
        return full

    def read_text(self, path: str | Path, *, encoding: str = "utf-8") -> str:
        return self._resolve(path).read_text(encoding=encoding)

    def write_text(self, path: str | Path, content: str, *, encoding: str = "utf-8") -> Path:
        target = self._resolve(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding=encoding)
        return target

    def exists(self, path: str | Path) -> bool:
        return self._resolve(path).exists()

    def glob(self, pattern: str) -> list[Path]:
        return sorted(self.root.glob(pattern))

    def list_dir(self, path: str | Path = ".") -> list[Path]:
        target = self._resolve(path)
        if not target.is_dir():
            return []
        return sorted(target.iterdir())

    def find_files(self, *, extensions: tuple[str, ...], within: str | Path = ".") -> list[Path]:
        base = self._resolve(within)
        return sorted(
            p for p in base.rglob("*")
            if p.is_file() and p.suffix.lower() in extensions and "__pycache__" not in p.parts
        )

    def ensure_dir(self, path: str | Path) -> Path:
        target = self._resolve(path)
        target.mkdir(parents=True, exist_ok=True)
        return target
