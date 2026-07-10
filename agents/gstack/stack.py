"""
gstack.stack — Núcleo de la stack: cola de operaciones + ejecución secuencial.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

from agents.context import get_context
from agents.orchestrator import Orchestrator
from agents.core.base_agent import AgentResult


@dataclass
class StackStep:
    agent: str
    action: str
    kwargs: dict[str, Any] = field(default_factory=dict)
    auto_commit_message: str | None = None
    run_if: Callable[[list[AgentResult]], bool] | None = None
    result_key: str | None = None


@dataclass
class StackResult:
    success: bool
    steps: list[StackStep]
    results: list[AgentResult]
    failed_at: int | None = None
    step_outputs: dict[str, AgentResult] = field(default_factory=dict)

    @property
    def summary(self) -> str:
        total = len(self.steps)
        ok = sum(1 for r in self.results if r.success)
        skipped = sum(1 for r in self.results if not r.success and r.action == "__skipped__")
        lines = [f"Stack: {ok}/{total} pasos completados ({skipped} omitidos)"]
        for i, (step, result) in enumerate(zip(self.steps, self.results)):
            if result.action == "__skipped__":
                status = "SKIP"
            else:
                status = "OK" if result.success else "FAIL"
            lines.append(f"  [{i}] {step.agent}.{step.action} → {status}: {result.message}")
            if result.warnings:
                for w in result.warnings:
                    lines.append(f"       ⚠ {w}")
        if self.failed_at is not None:
            lines.append(f"  Detenido en paso {self.failed_at}")
        return "\n".join(lines)


class GStack:
    """
    Pila de operaciones que se ejecutan secuencialmente, con paso de
    resultados entre pasos y branching condicional.

    Puedes acceder al resultado de un paso anterior con ``prev(key)`` dentro
    del kwargs del siguiente paso. Los pasos también pueden declarar un
    ``run_if`` como predicado sobre resultados anteriores.

    Parámetros
    ----------
    auto_commit : bool
        Si True, hace `git add -A && git commit` entre cada paso.
    commit_on_error : bool
        Si True y un paso falla, hace commit de lo acumulado antes de detenerse.
    log_events : bool
        Si True, escribe cada acción a ``agents/workspace/events.jsonl``.
    """

    def __init__(self, auto_commit: bool = False, commit_on_error: bool = True, log_events: bool = True,
                 context=None):
        self._steps: list[StackStep] = []
        self.auto_commit = auto_commit
        self.commit_on_error = commit_on_error
        self.log_events = log_events
        # `context` explícito para quien orquesta sobre otro proyecto (o un
        # test con contexto propio); sin él, el del proceso, como siempre.
        self._ctx = context or get_context()
        self._orch = Orchestrator(context=self._ctx)

    def push(self, agent: str, action: str, **kwargs) -> GStack:
        """
        Añade un paso al final de la stack.

        kwargs especiales (no se pasan al agente):
            auto_commit_message : str — sobreescribe el mensaje de commit auto
            result_key         : str — clave para acceder al resultado luego
            run_if             : callable — ``run_if=lambda results: results[-1].success``
        """
        msg = kwargs.pop("auto_commit_message", None)
        key = kwargs.pop("result_key", None)
        predicate = kwargs.pop("run_if", None)
        self._steps.append(StackStep(
            agent=agent, action=action, kwargs=kwargs,
            auto_commit_message=msg, result_key=key, run_if=predicate,
        ))
        return self

    def insert(self, index: int, agent: str, action: str, **kwargs) -> GStack:
        """Inserta un paso en una posición concreta."""
        msg = kwargs.pop("auto_commit_message", None)
        key = kwargs.pop("result_key", None)
        predicate = kwargs.pop("run_if", None)
        self._steps.insert(index, StackStep(
            agent=agent, action=action, kwargs=kwargs,
            auto_commit_message=msg, result_key=key, run_if=predicate,
        ))
        return self

    def run(self) -> StackResult:
        if not self._steps:
            return StackResult(success=True, steps=[], results=[])

        results: list[AgentResult] = []
        step_outputs: dict[str, AgentResult] = {}

        for i, step in enumerate(self._steps):
            if step.run_if is not None and not step.run_if(results):
                skipped = AgentResult(True, step.agent, "__skipped__", f"Paso omitido por run_if", data=None)
                results.append(skipped)
                self._log_event(step, skipped, i)
                continue

            resolved = self._resolve_kwargs(step.kwargs, step_outputs)
            result = self._orch.run(step.agent, step.action, **resolved)
            results.append(result)

            if step.result_key:
                step_outputs[step.result_key] = result

            self._log_event(step, result, i)

            if not result.success:
                if self.commit_on_error:
                    self._commit_step(i, step, success=False)
                return StackResult(
                    success=False, steps=self._steps, results=results,
                    failed_at=i, step_outputs=step_outputs,
                )

            if self.auto_commit:
                self._commit_step(i, step, success=True)

        return StackResult(success=True, steps=self._steps, results=results, step_outputs=step_outputs)

    def _resolve_kwargs(self, kwargs: dict, outputs: dict[str, AgentResult]) -> dict:
        """Resuelve referencias ``prev(key).data`` en kwargs."""
        import re
        resolved = {}
        for k, v in kwargs.items():
            if isinstance(v, str) and v.startswith("prev("):
                match = re.match(r"prev\((\w+)\)(?:\.(\w+))?", v)
                if match:
                    key = match.group(1)
                    attr = match.group(2)
                    prev_result = outputs.get(key)
                    if prev_result is not None:
                        resolved[k] = prev_result.data if attr == "data" else prev_result
                    else:
                        resolved[k] = v
                else:
                    resolved[k] = v
            else:
                resolved[k] = v
        return resolved

    def _log_event(self, step: StackStep, result: AgentResult, index: int) -> None:
        if not self.log_events:
            return
        try:
            log_dir = self._ctx.agent_workspace("gstack")
            log_path = log_dir / "events.jsonl"
            entry = {
                "timestamp": datetime.utcnow().isoformat(),
                "index": index,
                "agent": step.agent,
                "action": step.action,
                "success": result.success,
                "message": result.message,
                "warnings": result.warnings,
            }
            with open(log_path, "a") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        except Exception:
            pass

    def _commit_step(self, index: int, step: StackStep, success: bool) -> None:
        try:
            from agents.tools.git_tool import GitTool
            git = GitTool(repo_root=self._ctx.root)
            if not git.is_repo():
                return
            changed = git.changed_files(staged=False)
            if not changed:
                return
            git.add("-A")
            if step.auto_commit_message:
                msg = step.auto_commit_message
            else:
                prefix = "wip" if not success else step.action.replace("_", " ")
                msg = f"{prefix}: auto-commit tras paso {index} ({step.agent}.{step.action})"
            git.commit(msg)
        except Exception:
            pass

    @property
    def event_log_path(self) -> Path | None:
        """Ruta al archivo de eventos, o None si log_events=False."""
        if not self.log_events:
            return None
        return self._ctx.agent_workspace("gstack") / "events.jsonl"

    def __len__(self) -> int:
        return len(self._steps)

    def __bool__(self) -> bool:
        return len(self._steps) > 0
