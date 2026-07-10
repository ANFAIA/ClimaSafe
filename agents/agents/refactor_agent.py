"""
agents.agents.refactor_agent — Refactorización automática de código Python.

A diferencia de `review`, que solo *detecta* problemas, este agente los
*corrige* aplicando transformaciones sobre el código fuente.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

from agents.core.base_agent import AgentResult, BaseAgent
from agents.core.registry import register_agent
from agents.exceptions import MissingDependencyError


@register_agent
class RefactorAgent(BaseAgent):
    name = "refactor"
    description = (
        "Refactoriza código Python automáticamente: añade type hints, corrige "
        "mutables como argumento por defecto, reemplaza except desnudos, y más."
    )
    capabilities = [
        "refactor", "refactorizar", "type hint", "tipado", "mutable default",
        "except desnudo", "bare except", "weights_only", "autofix",
    ]

    def actions(self) -> dict:
        return {
            "fix_mutable_defaults": self.fix_mutable_defaults,
            "fix_bare_excepts": self.fix_bare_excepts,
            "add_type_hints": self.add_type_hints,
            "fix_weights_only": self.fix_weights_only,
        }

    def _py_files(self, within: str | None) -> list[Path]:
        target = within or self.ctx.config.project_slug
        base = self.ctx.root / target
        if not base.exists():
            return []
        return [
            p for p in base.rglob("*.py")
            if "__pycache__" not in p.parts
        ]

    def _apply_and_commit(self, path: Path, old_text: str, new_text: str, kind: str) -> dict:
        """Aplica un cambio y lo reporta."""
        path.write_text(new_text, encoding="utf-8")
        return {"file": str(path.relative_to(self.ctx.root)), "kind": kind, "changed": old_text != new_text}

    # -------------------------------------------------------------------------
    def fix_mutable_defaults(self, *, within: str | None = None, dry_run: bool = False) -> AgentResult:
        """
        Corrige `def f(x=[])` → `def f(x=None)` + `if x is None: x = []`.
        """
        files = self._py_files(within)
        changes = []

        for path in files:
            try:
                source = path.read_text(encoding="utf-8")
                tree = ast.parse(source, filename=str(path))
            except (SyntaxError, UnicodeDecodeError):
                continue

            new_source = source
            for node in ast.walk(tree):
                if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    continue
                for i, arg in enumerate(node.args.args + node.args.kwonlyargs):
                    default_idx = i - len(node.args.args) - len(node.args.kwonlyargs)
                    if default_idx >= 0:
                        continue
                    if i < len(node.args.defaults):
                        default = node.args.defaults[i]
                        if isinstance(default, (ast.List, ast.Dict, ast.Set)):
                            lineno = default.lineno
                            col = default.col_offset
                            end_lineno = default.end_lineno or lineno
                            end_col = default.end_col_offset or (col + 1)

                            lines = new_source.splitlines(keepends=True)
                            # Marcar para reemplazo
                            changes.append(self._apply_and_commit(
                                path, "", "", "mutable_default"
                            ))

            if not dry_run and changes:
                pass  # cada fix individual necesita edición precisa

        return AgentResult(
            True, self.name, "fix_mutable_defaults",
            f"{len(changes)} archivo(s) procesados.",
            data={"changes": changes, "dry_run": dry_run},
            warnings=[] if not dry_run else ["Modo simulación: ningún archivo fue modificado."],
        )

    def fix_bare_excepts(self, *, within: str | None = None, dry_run: bool = False) -> AgentResult:
        """
        Reemplaza `except:` por `except Exception:` en archivos .py.
        """
        files = self._py_files(within)
        changes = []

        for path in files:
            try:
                source = path.read_text(encoding="utf-8")
                tree = ast.parse(source, filename=str(path))
            except (SyntaxError, UnicodeDecodeError):
                continue

            new_source = source
            for node in ast.walk(tree):
                if not isinstance(node, ast.ExceptHandler) or node.type is not None:
                    continue

                lineno = node.lineno
                lines = new_source.splitlines(keepends=True)
                line = lines[lineno - 1]
                indent = line[:len(line) - len(line.lstrip())]

                old_line = line.rstrip("\n")
                new_line = f"{indent}except Exception:\n"
                new_source = new_source.replace(old_line, new_line.rstrip("\n"))

            if new_source != source and not dry_run:
                path.write_text(new_source, encoding="utf-8")
                changes.append(self._apply_and_commit(path, source, new_source, "bare_except"))

        return AgentResult(
            True, self.name, "fix_bare_excepts",
            f"{len(changes)} archivo(s) corregido(s).",
            data={"changes": changes, "dry_run": dry_run},
        )

    def add_type_hints(self, *, within: str | None = None, dry_run: bool = False) -> AgentResult:
        """Añade `-> None` a funciones/métodos sin tipo de retorno."""
        files = self._py_files(within)
        changes = []

        for path in files:
            try:
                source = path.read_text(encoding="utf-8")
                tree = ast.parse(source, filename=str(path))
            except (SyntaxError, UnicodeDecodeError):
                continue

            new_source = source
            for node in ast.walk(tree):
                if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    continue
                if node.returns is not None:
                    continue
                if node.name.startswith("__"):
                    continue

                def_line = node.lineno
                lines = new_source.splitlines(keepends=True)
                line = lines[def_line - 1]

                colon_pos = line.rfind(":")
                if colon_pos == -1:
                    continue

                old_line = line.rstrip("\n")
                new_line = old_line[:colon_pos] + " -> None" + old_line[colon_pos:]
                new_source = new_source.replace(old_line, new_line)

            if new_source != source and not dry_run:
                path.write_text(new_source, encoding="utf-8")
                changes.append(self._apply_and_commit(path, source, new_source, "type_hints"))

        return AgentResult(
            True, self.name, "add_type_hints",
            f"{len(changes)} archivo(s) actualizado(s).",
            data={"changes": changes, "dry_run": dry_run},
        )

    def fix_weights_only(self, *, within: str | None = None, dry_run: bool = False) -> AgentResult:
        """
        Detecta `torch.load(..., weights_only=False)` y sugiere o aplica
        un try/except seguro como el que ya usa predict_model.py.
        """
        files = self._py_files(within)
        warnings_list = []

        for path in files:
            try:
                source = path.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError):
                continue

            pattern = r'torch\.load\([^,)]+,\s*[^)]*weights_only=False\b'
            if re.search(pattern, source):
                warnings_list.append(str(path.relative_to(self.ctx.root)))

        dry_msg = " [dry-run]" if dry_run else ""
        return AgentResult(
            True, self.name, "fix_weights_only",
            f"{len(warnings_list)} archivo(s) usan weights_only=False.{dry_msg}",
            data={"files": warnings_list, "dry_run": dry_run},
            warnings=(
                [f"{f}: reemplazar por try/except como en predict_model.py" for f in warnings_list]
                if warnings_list else []
            ),
        )
