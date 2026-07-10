"""
agents.agents.audit_agent — Auditor del equipo de agentes.

Lee `agents/workspace/audit/audit.jsonl` (que escribe `BaseAgent.run()` en
cada ejecución, ver agents/audit.py) y lo convierte en respuestas accionables:

- `report`               → uso, tasa de éxito y duración por agente/acción.
- `failures`             → los fallos recientes, con su mensaje.
- `suggest_improvements` → heurísticas deterministas sobre el log: qué
                           acción falla demasiado, qué agente no se usa
                           nunca, qué acción es sospechosamente lenta.

Es la pieza de "mejorar el equipo con datos": después de una temporada de
uso, `suggest_improvements` te dice dónde invertir (arreglar, documentar,
retirar) sin depender de la memoria de nadie.

Límites (ver su contrato en agents/contracts.py): mide y propone, no
arregla. Y solo ve lo que pasó por `run()` — las llamadas directas a
métodos no se auditan (documentado en agents/audit.py).
"""

from __future__ import annotations

from collections import defaultdict

from agents import audit
from agents.core.base_agent import AgentResult, BaseAgent
from agents.core.registry import agent_registry, register_agent

# Umbrales de las heurísticas de suggest_improvements. Ajustables aquí,
# en un solo sitio, si tu proyecto tiene otra tolerancia.
MIN_RUNS_TO_JUDGE = 3        # menos ejecuciones que esto = no hay datos para juzgar
FAILURE_RATE_THRESHOLD = 0.5
SLOW_ACTION_MS = 30_000
NOISY_WARNINGS_RATIO = 0.5


@register_agent
class AuditAgent(BaseAgent):
    name = "audit"
    description = (
        "Audita al resto de agentes con el log de ejecuciones: uso, tasa de éxito, "
        "duración, fallos recientes y sugerencias de mejora."
    )
    capabilities = [
        "auditoria", "auditoría", "audit", "auditar",
        "rendimiento de agentes", "historial de ejecuciones",
    ]

    def actions(self) -> dict:
        return {
            "report": self.report,
            "failures": self.failures,
            "suggest_improvements": self.suggest_improvements,
        }

    # ── helpers ───────────────────────────────────────────────────────────

    def _aggregate(self, last: int) -> dict[str, dict]:
        """Agrega el log por 'agente.acción' → runs/ok/fail/avg_ms/warnings."""
        stats: dict[str, dict] = defaultdict(
            lambda: {"runs": 0, "ok": 0, "fail": 0, "total_ms": 0.0, "with_warnings": 0}
        )
        for entry in audit.read_entries(self.ctx, last=last):
            key = f"{entry.get('agent', '?')}.{entry.get('action', '?')}"
            s = stats[key]
            s["runs"] += 1
            s["ok" if entry.get("success") else "fail"] += 1
            s["total_ms"] += float(entry.get("duration_ms", 0))
            if entry.get("warnings", 0):
                s["with_warnings"] += 1
        return stats

    # ── acciones ──────────────────────────────────────────────────────────

    def report(self, last: int = 500) -> AgentResult:
        """Informe de uso por agente/acción sobre las últimas `last` ejecuciones."""
        stats = self._aggregate(last)
        if not stats:
            return AgentResult(
                True, self.name, "report",
                "El log de auditoría está vacío — todavía no se ha ejecutado ninguna acción "
                "vía run()/CLI/pipeline. Usa el sistema y vuelve.",
                data=[],
            )

        rows = []
        for key, s in sorted(stats.items()):
            rows.append({
                "accion": key,
                "runs": s["runs"],
                "exito": f"{s['ok'] / s['runs']:.0%}",
                "media_ms": round(s["total_ms"] / s["runs"], 1),
                "con_warnings": s["with_warnings"],
            })
        total = sum(s["runs"] for s in stats.values())
        total_fail = sum(s["fail"] for s in stats.values())
        return AgentResult(
            True, self.name, "report",
            f"{total} ejecuciones auditadas ({total_fail} fallidas) en {len(stats)} acciones distintas.",
            data=rows,
        )

    def failures(self, last: int = 100) -> AgentResult:
        """Los fallos más recientes con su mensaje — por dónde empezar a mejorar."""
        failed = [
            {
                "timestamp": e.get("timestamp"),
                "accion": f"{e.get('agent')}.{e.get('action')}",
                "message": e.get("message"),
                "error": e.get("error"),
            }
            for e in audit.read_entries(self.ctx, last=last)
            if not e.get("success")
        ]
        return AgentResult(
            True, self.name, "failures",
            f"{len(failed)} fallo(s) en las últimas {last} ejecuciones."
            + (" Nada que arreglar por aquí." if not failed else ""),
            data=failed,
        )

    def suggest_improvements(self, last: int = 500) -> AgentResult:
        """
        Sugerencias deterministas a partir del log. Cada una dice el síntoma
        y la acción concreta que tomar — el humano decide, este agente no toca nada.
        """
        stats = self._aggregate(last)
        suggestions: list[str] = []

        for key, s in sorted(stats.items()):
            if s["runs"] < MIN_RUNS_TO_JUDGE:
                continue
            rate_fail = s["fail"] / s["runs"]
            if rate_fail >= FAILURE_RATE_THRESHOLD:
                suggestions.append(
                    f"'{key}' falla el {rate_fail:.0%} de las veces ({s['fail']}/{s['runs']}). "
                    f"Revisa sus fallos con `run audit failures` — o su contrato: puede que se le "
                    f"esté pidiendo algo fuera de su rol."
                )
            avg = s["total_ms"] / s["runs"]
            if avg >= SLOW_ACTION_MS:
                suggestions.append(
                    f"'{key}' tarda {avg / 1000:.1f}s de media ({s['runs']} runs). Candidata a "
                    f"cachear resultados o reducir el alcance por defecto."
                )
            if s["with_warnings"] / s["runs"] >= NOISY_WARNINGS_RATIO:
                suggestions.append(
                    f"'{key}' devuelve warnings en el {s['with_warnings'] / s['runs']:.0%} de sus runs — "
                    f"o el aviso es esperado (súbelo a la documentación) o hay un límite del agente "
                    f"que conviene arreglar."
                )

        agent_registry.discover()
        used_agents = {key.split(".")[0] for key in stats}
        never_used = sorted(set(agent_registry.all()) - used_agents - {self.name})
        if never_used and stats:
            suggestions.append(
                f"Agentes sin ninguna ejecución auditada: {never_used}. Si llevan tiempo sin uso: "
                f"mejora sus keywords de ruteo, documenta cuándo usarlos, o plantéate retirarlos."
            )

        return AgentResult(
            True, self.name, "suggest_improvements",
            f"{len(suggestions)} sugerencia(s) a partir de {sum(s['runs'] for s in stats.values())} "
            f"ejecuciones auditadas."
            + (" El equipo está sano — sigue usándolo para acumular datos." if not suggestions else ""),
            data=suggestions,
        )
