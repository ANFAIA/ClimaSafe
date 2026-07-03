"""
agents.tools.process_tool — Ejecución segura de comandos externos.

Todas las herramientas que envuelven un binario de sistema (git, docker...)
pasan por aquí en vez de llamar a `subprocess` directamente, para que las
reglas de seguridad (`shell=False`, timeout, captura de stderr) se apliquen
en un único sitio.
"""

from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

from agents.exceptions import MissingDependencyError, ToolExecutionError

DEFAULT_TIMEOUT_SECONDS = 60


@dataclass
class ProcessResult:
    command: list[str]
    returncode: int
    stdout: str
    stderr: str

    @property
    def ok(self) -> bool:
        return self.returncode == 0


def require_binary(binary: str) -> None:
    """Lanza MissingDependencyError con un mensaje útil si el binario no está en PATH."""
    if shutil.which(binary) is None:
        raise MissingDependencyError(
            f"'{binary}' no está instalado o no está en el PATH. "
            f"Este agente necesita el binario '{binary}' disponible en el sistema."
        )


def run_command(
    args: list[str],
    *,
    cwd: Path | None = None,
    timeout: int = DEFAULT_TIMEOUT_SECONDS,
    check: bool = False,
) -> ProcessResult:
    """
    Ejecuta `args` (nunca a través de una shell — evita inyección de comandos)
    y devuelve un `ProcessResult`. Si `check=True`, lanza `ToolExecutionError`
    cuando el returncode no es 0.
    """
    if not args:
        raise ValueError("run_command requiere una lista de argumentos no vacía.")

    require_binary(args[0])

    try:
        completed = subprocess.run(
            args,
            cwd=str(cwd) if cwd else None,
            capture_output=True,
            text=True,
            timeout=timeout,
            shell=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise ToolExecutionError(
            f"El comando '{' '.join(args)}' superó el timeout de {timeout}s.",
        ) from exc

    result = ProcessResult(
        command=args,
        returncode=completed.returncode,
        stdout=completed.stdout,
        stderr=completed.stderr,
    )

    if check and not result.ok:
        raise ToolExecutionError(
            f"El comando '{' '.join(args)}' falló (exit {result.returncode}): {result.stderr.strip()}",
            returncode=result.returncode,
            stderr=result.stderr,
        )

    return result
