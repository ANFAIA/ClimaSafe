"""
agents.agents.plan_agent — Jefe de proyecto: encargo → preguntas → delegación.

El flujo de trabajo que implementa (pensado para que el humano solo tenga
que describir, responder, y verificar):

    1. `intake(brief)`   — Descompone el encargo en pasos, asigna cada paso
                           al agente responsable (ruteo del Orchestrator) y
                           detecta TODOS los argumentos que faltan. Devuelve
                           las preguntas y guarda la orden de trabajo en
                           `agents/workspace/plan/orden-<id>.json`.
    2. (humano)          — Revisa el plan (el JSON es editable a mano: puedes
                           cambiar agente/acción de un paso, borrar pasos,
                           reordenar) y responde las preguntas con `answer`.
    3. `execute(order)`  — Se NIEGA a ejecutar si quedan preguntas sin
                           responder. Si está completa, delega cada paso vía
                           GStack (cada ejecución queda auditada) y devuelve
                           el resumen + qué verificar.
    4. `status()`        — Estado de todas las órdenes de trabajo.

Límites (ver su contrato en agents/contracts.py):
- No ejecuta ninguna acción de dominio él mismo: siempre delega en el
  agente dueño del recurso.
- No inventa argumentos: lo que falta se pregunta (AgentResult.needs).
- La descomposición del brief es heurística (frases separadas por saltos de
  línea, ';' o conectores tipo "y luego"). Para encargos complejos, escribe
  el brief con un paso por línea — o edita el JSON de la orden antes de
  ejecutar. Un agente de codificación (Claude, etc.) también puede escribir
  la orden JSON directamente y saltarse el intake.
"""

from __future__ import annotations

import inspect
import json
import re
from datetime import datetime
from pathlib import Path

from agents.core.base_agent import AgentResult, BaseAgent
from agents.core.registry import register_agent

# Conectores que separan pasos dentro de una misma línea del brief.
_STEP_SPLIT = re.compile(
    r"[\n;]+|(?:,\s+)?\by\s+(?:luego|después|despues|entonces)\b|\bdespués de eso\b",
    re.IGNORECASE,
)


@register_agent
class PlanAgent(BaseAgent):
    name = "plan"
    description = (
        "Jefe de proyecto: convierte un encargo en una orden de trabajo, "
        "pregunta lo que falte, delega cada paso al agente dueño y resume qué verificar."
    )
    capabilities = [
        "planificar", "plan de trabajo", "orden de trabajo", "workorder",
        "encargo", "delegar", "brief", "organiza el trabajo",
    ]

    def actions(self) -> dict:
        return {
            "intake": self.intake,
            "answer": self.answer,
            "execute": self.execute,
            "status": self.status,
        }

    # ── helpers internos ──────────────────────────────────────────────────

    def _orders_dir(self) -> Path:
        return self.ctx.agent_workspace("plan")

    def _order_path(self, order_id: str) -> Path:
        return self._orders_dir() / f"orden-{order_id}.json"

    def _load_order(self, order_id: str) -> dict | None:
        path = self._order_path(order_id)
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8"))

    def _save_order(self, order: dict) -> None:
        self._order_path(order["id"]).write_text(
            json.dumps(order, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    @staticmethod
    def _required_params(method) -> list[str]:
        """Parámetros sin valor por defecto de una acción (los que hay que preguntar)."""
        required = []
        for pname, param in inspect.signature(method).parameters.items():
            if param.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
                continue
            if param.default is inspect.Parameter.empty:
                required.append(pname)
        return required

    @staticmethod
    def _questions(order: dict) -> list[str]:
        """Preguntas pendientes de una orden: una por argumento faltante + pasos sin asignar."""
        questions = []
        for i, step in enumerate(order["steps"]):
            for param in step["missing"]:
                questions.append(
                    f"Paso {i} ({step['agent']}.{step['action']}): falta '{param}'. "
                    f"Responde con: run plan answer --order {order['id']} --step{i}-{param.replace('_', '-')} VALOR"
                )
        for fragment in order.get("unmatched", []):
            questions.append(
                f"No sé qué agente debe hacer: '{fragment}'. Añade el paso a mano en "
                f"{order.get('_file', 'la orden JSON')} (agente + acción), o quítalo del encargo."
            )
        return questions

    # ── acciones ──────────────────────────────────────────────────────────

    def intake(self, brief: str) -> AgentResult:
        """
        Descompone `brief` en pasos, asigna agentes y devuelve las preguntas
        necesarias para poder ejecutar sin inventar nada.
        """
        from agents.orchestrator import Orchestrator

        fragments = [f.strip(" .,") for f in _STEP_SPLIT.split(brief) if f and f.strip(" .,")]
        if not fragments:
            return AgentResult(
                False, self.name, "intake",
                "El encargo está vacío. Describe qué quieres hacer (un paso por línea funciona mejor).",
                needs=["el encargo (brief) con al menos un paso"],
            )

        orch = Orchestrator(context=self.ctx)
        steps: list[dict] = []
        unmatched: list[str] = []

        for fragment in fragments:
            decision = orch.select_agent(fragment)
            if decision.agent_name is None or decision.agent_name == self.name:
                unmatched.append(fragment)
                continue
            agent = orch._get_instance(decision.agent_name)  # noqa: SLF001 — colaboración interna del sistema
            action = agent.best_action(fragment)
            if action is None:
                unmatched.append(fragment)
                continue
            missing = self._required_params(agent.actions()[action])
            steps.append({
                "fragment": fragment,
                "agent": decision.agent_name,
                "action": action,
                "confidence": round(decision.confidence, 2),
                "kwargs": {},
                "missing": missing,
            })

        order_id = datetime.now().strftime("%Y%m%d-%H%M%S")
        order = {
            "id": order_id,
            "created": datetime.now().isoformat(timespec="seconds"),
            "brief": brief,
            "status": "borrador",
            "steps": steps,
            "unmatched": unmatched,
        }
        self._save_order(order)
        order["_file"] = str(self._order_path(order_id))

        questions = self._questions(order)
        plan_lines = [
            f"  [{i}] {s['agent']}.{s['action']}  (confianza {s['confidence']}) ← '{s['fragment']}'"
            for i, s in enumerate(steps)
        ]
        summary = (
            f"Orden de trabajo {order_id} creada con {len(steps)} paso(s):\n"
            + "\n".join(plan_lines)
            + (f"\n  Sin asignar: {unmatched}" if unmatched else "")
            + (
                f"\n\nAntes de ejecutar necesito {len(questions)} respuesta(s) — no invento valores."
                if questions
                else f"\n\nLista para ejecutar: run plan execute --order {order_id}"
            )
            + f"\nRevisa/edita el plan en {order['_file']}"
        )
        return AgentResult(
            True, self.name, "intake", summary,
            data=order, needs=questions,
        )

    def answer(self, order: str, **answers) -> AgentResult:
        """
        Responde preguntas de una orden. Claves aceptadas:
          step0_filename=...  → argumento 'filename' del paso 0
          filename=...        → mismo argumento en TODOS los pasos que lo pidan
        """
        wo = self._load_order(order)
        if wo is None:
            return AgentResult(False, self.name, "answer", f"No existe la orden '{order}'.")
        if wo["status"] in ("completado", "fallido"):
            return AgentResult(False, self.name, "answer", f"La orden {order} ya se ejecutó ({wo['status']}).")

        applied = []
        for key, value in answers.items():
            match = re.match(r"step(\d+)_(.+)", key)
            targets = []
            if match:
                idx, param = int(match.group(1)), match.group(2)
                if idx < len(wo["steps"]):
                    targets = [(idx, param)]
            else:
                targets = [(i, key) for i, s in enumerate(wo["steps"]) if key in s["missing"]]
            for idx, param in targets:
                step = wo["steps"][idx]
                step["kwargs"][param] = value
                if param in step["missing"]:
                    step["missing"].remove(param)
                applied.append(f"paso {idx}: {param}={value}")

        wo["status"] = "borrador"
        self._save_order(wo)
        pending = self._questions(wo)
        message = (
            (f"Aplicado: {', '.join(applied)}. " if applied else "Ninguna respuesta coincidió con lo pedido. ")
            + (
                f"Quedan {len(pending)} pregunta(s)."
                if pending
                else f"Todo respondido — ejecuta con: run plan execute --order {order}"
            )
        )
        return AgentResult(bool(applied), self.name, "answer", message, data=wo, needs=pending)

    def execute(self, order: str, auto_commit: bool = False) -> AgentResult:
        """
        Ejecuta una orden completa delegando cada paso vía GStack.
        Se niega si quedan preguntas: dirigir es del humano, adivinar no es de nadie.
        """
        wo = self._load_order(order)
        if wo is None:
            return AgentResult(False, self.name, "execute", f"No existe la orden '{order}'.")

        pending = self._questions(wo)
        if pending:
            return AgentResult(
                False, self.name, "execute",
                f"La orden {order} tiene {len(pending)} pregunta(s) sin responder — no ejecuto con huecos.",
                needs=pending,
            )
        if not wo["steps"]:
            return AgentResult(False, self.name, "execute", f"La orden {order} no tiene pasos.")

        from agents.gstack.stack import GStack

        stack = GStack(auto_commit=auto_commit, context=self.ctx)
        for step in wo["steps"]:
            stack.push(step["agent"], step["action"], **step["kwargs"])
        result = stack.run()

        wo["status"] = "completado" if result.success else "fallido"
        wo["executed"] = datetime.now().isoformat(timespec="seconds")
        wo["results"] = [
            {"agent": r.agent, "action": r.action, "success": r.success, "message": r.message}
            for r in result.results
        ]
        self._save_order(wo)

        to_verify = [
            f"  [{i}] {r.agent}.{r.action}: {r.message}" for i, r in enumerate(result.results)
        ]
        message = (
            f"Orden {order} {wo['status']}.\n{result.summary}\n\n"
            "Para el humano — verifica:\n" + "\n".join(to_verify)
            + "\n(auditoría completa: run audit report)"
        )
        return AgentResult(result.success, self.name, "execute", message, data=wo)

    def status(self, order: str | None = None) -> AgentResult:
        """Estado de una orden concreta, o listado de todas."""
        if order is not None:
            wo = self._load_order(order)
            if wo is None:
                return AgentResult(False, self.name, "status", f"No existe la orden '{order}'.")
            return AgentResult(
                True, self.name, "status",
                f"Orden {order}: {wo['status']} ({len(wo['steps'])} pasos, "
                f"{len(self._questions(wo))} preguntas pendientes)",
                data=wo, needs=self._questions(wo),
            )

        orders = sorted(self._orders_dir().glob("orden-*.json"))
        listing = []
        for path in orders:
            try:
                wo = json.loads(path.read_text(encoding="utf-8"))
                listing.append({
                    "id": wo["id"], "status": wo["status"],
                    "steps": len(wo["steps"]), "brief": wo["brief"][:80],
                })
            except (json.JSONDecodeError, KeyError):
                continue
        return AgentResult(
            True, self.name, "status",
            f"{len(listing)} orden(es) de trabajo en {self._orders_dir()}",
            data=listing,
        )
