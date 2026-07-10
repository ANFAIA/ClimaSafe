"""
agents.audit — Registro de auditoría de todas las ejecuciones de agentes.

Cada vez que se ejecuta una acción a través de `BaseAgent.run()` (es decir,
por cualquier vía oficial: CLI, `Orchestrator`, `GStack`, `delegate_to`),
se añade una línea JSON a `agents/workspace/audit/audit.jsonl` con quién
hizo qué, si salió bien y cuánto tardó.

Para qué sirve
--------------
Es la materia prima para mejorar el equipo de agentes con datos en vez de
impresiones: el agente `audit` (agents/agents/audit_agent.py) lee este log y
responde a "¿qué agente falla más?", "¿qué acción es la más lenta?",
"¿qué agentes no se usan nunca?".

Decisiones de diseño
--------------------
- JSONL append-only, sin base de datos: legible con `cat`, `jq` o pandas,
  y sin dependencias nuevas (filosofía del README).
- La auditoría NUNCA rompe la acción auditada: cualquier error al escribir
  el log se traga en silencio. Un disco lleno no debe convertir un commit
  que funcionó en un fallo.
- No se guardan los `kwargs` de la acción: pueden contener rutas privadas,
  mensajes largos o (peor) secretos pasados por error. Solo los NOMBRES de
  los argumentos, que bastan para auditar el uso.
- Las llamadas directas a métodos (`GitAgent().suggest_commit_message()`)
  NO pasan por `run()` y por tanto no quedan auditadas — documentado aquí a
  propósito: si quieres auditoría completa, invoca siempre vía `run()`,
  `delegate_to()` o la CLI.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from agents.context import SharedContext

AUDIT_FILENAME = "audit.jsonl"


def audit_log_path(ctx: "SharedContext") -> Path:
    """Ruta del log de auditoría (crea `agents/workspace/audit/` si no existe)."""
    return ctx.agent_workspace("audit") / AUDIT_FILENAME


def record(
    ctx: "SharedContext",
    *,
    agent: str,
    action: str,
    success: bool,
    duration_ms: float,
    message: str,
    warnings: int = 0,
    kwarg_names: list[str] | None = None,
    error: str | None = None,
) -> None:
    """Añade una entrada al log. Nunca lanza: la auditoría no rompe lo auditado."""
    try:
        entry: dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "agent": agent,
            "action": action,
            "success": success,
            "duration_ms": round(duration_ms, 1),
            "message": message[:300],
            "warnings": warnings,
            "kwarg_names": kwarg_names or [],
        }
        if error:
            entry["error"] = error[:300]
        with open(audit_log_path(ctx), "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except Exception:  # noqa: BLE001 — ver docstring del módulo
        pass


def read_entries(ctx: "SharedContext", last: int | None = None) -> list[dict]:
    """
    Lee las entradas del log (las `last` más recientes, o todas).
    Las líneas corruptas (proceso interrumpido a mitad de escritura) se
    ignoran en vez de tumbar la lectura completa.
    """
    path = audit_log_path(ctx)
    if not path.exists():
        return []
    entries: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            entries.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return entries[-last:] if last else entries
