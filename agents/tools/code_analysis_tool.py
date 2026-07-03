"""
agents.tools.code_analysis_tool — Análisis estático con `ast` (librería
estándar). No sustituye a ruff/pylint (que ya están en el proyecto vía
`make lint`) — se enfoca en señales que un linter genérico no da porque
requieren contexto: funciones largas, demasiados argumentos, funciones
duplicadas o casi duplicadas entre archivos.
"""

from __future__ import annotations

import ast
import hashlib
from dataclasses import dataclass
from pathlib import Path


@dataclass
class CodeSmell:
    file: str
    line: int
    kind: str
    message: str


@dataclass
class FunctionInfo:
    file: str
    name: str
    line: int
    n_lines: int
    n_args: int
    body_hash: str  # hash de la estructura del cuerpo, ignorando nombres — para detectar duplicados
    has_substantive_logic: bool  # False para funciones "de forma" (p. ej. `return {...}` sin control de flujo)


class CodeAnalysisTool:
    MAX_FUNCTION_LINES = 60
    MAX_ARGS = 6

    @staticmethod
    def parse(path: Path) -> ast.Module | None:
        try:
            source = path.read_text(encoding="utf-8")
            return ast.parse(source, filename=str(path))
        except (SyntaxError, UnicodeDecodeError, OSError):
            return None

    @staticmethod
    def _structural_hash(node: ast.AST) -> str:
        """
        Hash del 'esqueleto' de un nodo AST: tipos de nodo en orden, sin
        nombres de variables ni literales. Dos funciones con este mismo hash
        tienen la misma estructura de control aunque usen nombres distintos
        — un indicio razonable de duplicación, no una prueba matemática.
        """
        skeleton = "|".join(type(n).__name__ for n in ast.walk(node))
        return hashlib.sha256(skeleton.encode()).hexdigest()[:16]

    @staticmethod
    def _has_substantive_logic(node: ast.AST) -> bool:
        """
        False solo para el patrón exacto `def f(...): return {...}` (una
        única sentencia, un `return` de un diccionario literal) — el patrón
        de los métodos `actions()` de este propio sistema de agentes, cuyo
        AST coincide con el de cualquier otro dispatcher del mismo tamaño
        sin que eso sea duplicación de lógica real.

        Importante: esto NO excluye funciones sin llamadas ni control de
        flujo en general (p. ej. dos funciones que solo hacen sumas y
        asignaciones sí deben poder detectarse como duplicadas — eso es
        duplicación real). Solo se excluye el caso muy específico de
        "una sentencia, devuelve un dict".
        """
        body = node.body
        if body and isinstance(body[0], ast.Expr) and isinstance(getattr(body[0], "value", None), ast.Constant):
            body = body[1:]  # ignora el docstring de la función, si lo tiene
        is_trivial_dict_return = (
            len(body) == 1 and isinstance(body[0], ast.Return) and isinstance(body[0].value, ast.Dict)
        )
        return not is_trivial_dict_return

    @classmethod
    def analyze_file(cls, path: Path) -> tuple[list[CodeSmell], list[FunctionInfo]]:
        tree = cls.parse(path)
        if tree is None:
            return [], []

        smells: list[CodeSmell] = []
        functions: list[FunctionInfo] = []
        rel_name = str(path)

        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                n_lines = (node.end_lineno or node.lineno) - node.lineno + 1
                n_args = len(node.args.args) + len(node.args.kwonlyargs)

                functions.append(
                    FunctionInfo(
                        file=rel_name, name=node.name, line=node.lineno,
                        n_lines=n_lines, n_args=n_args,
                        body_hash=cls._structural_hash(node),
                        has_substantive_logic=cls._has_substantive_logic(node),
                    )
                )

                if n_lines > cls.MAX_FUNCTION_LINES:
                    smells.append(CodeSmell(
                        rel_name, node.lineno, "long_function",
                        f"'{node.name}' tiene {n_lines} líneas (umbral {cls.MAX_FUNCTION_LINES}). "
                        f"Considera dividirla en funciones más pequeñas."
                    ))
                if n_args > cls.MAX_ARGS:
                    smells.append(CodeSmell(
                        rel_name, node.lineno, "too_many_args",
                        f"'{node.name}' tiene {n_args} argumentos (umbral {cls.MAX_ARGS}). "
                        f"Considera agrupar en un dataclass o dict de config."
                    ))

            elif isinstance(node, ast.ExceptHandler) and node.type is None:
                smells.append(CodeSmell(
                    rel_name, node.lineno, "bare_except",
                    "except desnudo: captura hasta KeyboardInterrupt/SystemExit. "
                    "Especifica el tipo de excepción esperado."
                ))

        return smells, functions

    @classmethod
    def find_duplicates(cls, functions: list[FunctionInfo], *, min_lines: int = 5) -> list[list[FunctionInfo]]:
        """
        Agrupa funciones con el mismo `body_hash` (misma estructura AST).
        Se ignoran funciones muy cortas (`min_lines`, coinciden por
        trivialidad — p. ej. dos `__init__` de una línea) y funciones sin
        `has_substantive_logic` (dispatchers tipo `actions()` que devuelven
        un dict — su AST coincide por construcción con cualquier otro
        dispatcher del mismo tamaño, sin que eso sea duplicación real).
        """
        groups: dict[str, list[FunctionInfo]] = {}
        for fn in functions:
            if fn.n_lines < min_lines or not fn.has_substantive_logic:
                continue
            groups.setdefault(fn.body_hash, []).append(fn)
        return [group for group in groups.values() if len(group) > 1]
