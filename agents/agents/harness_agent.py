"""
agents.agents.harness_agent — Dueño mecánico del arnés.

El arnés (ver AGENTS.md) tiene dos capas: los agentes markdown de
`.opencode/agents/` razonan (qué feature toca, cómo implementarla) y este
agente ejecuta. Todo lo que es determinista —leer el backlog, cambiar un
estado, escribir el histórico, ejecutar la puerta— vive aquí, en Python, y no
en un prompt: un LLM editando JSON a mano se equivoca, `json.dump` no.

La regla del arnés deja de ser una instrucción y pasa a ser código:
`finish()` REHÚSA cerrar una feature si `./init.sh` no pasa en verde. No hay
forma de saltársela pidiéndoselo amablemente al modelo.
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Any

from agents.core.base_agent import AgentResult, BaseAgent
from agents.core.registry import register_agent
from agents.tools.process_tool import run_command

VALID_STATUS = ("pending", "in_progress", "done", "blocked")
REQUIRED_FIELDS = ("id", "title", "description", "acceptance_criteria", "status")

#: Rechazos seguidos del reviewer antes de bloquear la feature y escalar.
#: Tres es suficiente para corregir un despiste; a partir de ahí el problema
#: casi nunca es el código, sino el criterio o cómo está planteada la feature.
MAX_REVIEW_ROUNDS = 3

_RECHAZOS = ("rechazado", "rechaza", "rejected", "fail", "ko")


def _es_rechazo(verdict: str) -> bool:
    return verdict.strip().lower() in _RECHAZOS

CURRENT_TEMPLATE = """# Tarea actual

**Feature:** {fid}
**Estado:** {status}
**Iniciada:** {started}
**Responsable:** {owner}

## Objetivo

{description}

## Criterios de aceptación

{criteria}

## Bitácora

{log}

## Bloqueos

{blockers}
"""

IDLE_CURRENT = """# Tarea actual

> Estado vivo de la ejecución en curso. Es la memoria **fuera** de la ventana de
> contexto: cualquier agente que arranque de cero lee este fichero y sabe dónde
> está el trabajo sin releer el proyecto entero.

**Feature:** _(ninguna)_
**Estado:** idle
**Iniciada:** —
**Responsable:** —

## Objetivo

_(sin trabajo en curso)_

## Criterios de aceptación

_(copiar aquí los `acceptance_criteria` de la feature al empezar)_

## Bitácora

_(una línea por paso: qué se hizo, qué fichero se tocó, qué verificó)_

## Bloqueos

_(qué impide avanzar y qué se necesita para desbloquearlo — vacío si nada)_
"""


@register_agent
class HarnessAgent(BaseAgent):
    name = "harness"
    description = (
        "Dueño del arnés: lee y actualiza featureslist.json y progress/, y "
        "ejecuta la puerta init.sh. No cierra una feature si la puerta no pasa."
    )
    # Ojo: "feature"/"features" NO van aquí — son del agente `data`
    # (feature engineering). Un keyword, un dueño.
    capabilities = [
        "arnes", "arnés", "harness", "backlog",
        "tarea pendiente", "siguiente tarea", "progreso", "progress",
        "criterios de aceptacion", "criterios de aceptación", "puerta", "gate",
    ]

    def actions(self) -> dict:
        return {
            "status": self.status,
            "next": self.next,
            "start": self.start,
            "finish": self.finish,
            "block": self.block,
            "record": self.record,
            "gate": self.gate,
            "add": self.add,
        }

    # -- rutas ---------------------------------------------------------------
    @property
    def _backlog_file(self) -> Path:
        return self.ctx.root / "featureslist.json"

    @property
    def _progress_dir(self) -> Path:
        return self.ctx.root / "progress"

    @property
    def _current_file(self) -> Path:
        return self._progress_dir / "current.md"

    @property
    def _history_file(self) -> Path:
        return self._progress_dir / "history.md"

    # -- backlog -------------------------------------------------------------
    def _load(self) -> tuple[dict | None, str]:
        """Devuelve (documento, error). Si error != "", el documento es None."""
        if not self._backlog_file.exists():
            return None, f"No existe {self._backlog_file.name}. El arnés está incompleto."
        try:
            doc = json.loads(self._backlog_file.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            return None, f"featureslist.json no es JSON válido: {exc}"
        if not isinstance(doc, dict) or not isinstance(doc.get("features"), list):
            return None, "featureslist.json debe ser un objeto con la clave 'features' (lista)."
        return doc, ""

    def _save(self, doc: dict) -> None:
        self._backlog_file.write_text(
            json.dumps(doc, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )

    @staticmethod
    def _find(doc: dict, feature_id: str) -> dict | None:
        for feat in doc["features"]:
            if isinstance(feat, dict) and feat.get("id") == feature_id:
                return feat
        return None

    @staticmethod
    def _eligible(doc: dict) -> list[dict]:
        """Pendientes cuyas dependencias están todas en done, en orden de backlog."""
        done = {f["id"] for f in doc["features"] if f.get("status") == "done"}
        return [
            f
            for f in doc["features"]
            if f.get("status") == "pending"
            and all(dep in done for dep in f.get("depends_on", []))
        ]

    def _fail(self, action: str, message: str, **kw: Any) -> AgentResult:
        return AgentResult(success=False, agent=self.name, action=action, message=message, **kw)

    # -- acciones ------------------------------------------------------------
    def status(self) -> AgentResult:
        """Estado del backlog y de la tarea en curso."""
        doc, error = self._load()
        if doc is None:
            return self._fail("status", error)

        features = doc["features"]
        counts = {status: 0 for status in VALID_STATUS}
        for feat in features:
            counts[feat.get("status", "pending")] = counts.get(feat.get("status", "pending"), 0) + 1

        running = [f["id"] for f in features if f.get("status") == "in_progress"]
        warnings = []
        if len(running) > 1:
            warnings.append(
                f"{len(running)} features in_progress a la vez ({', '.join(running)}). "
                f"El arnés espera una: cierra o bloquea las demás."
            )

        eligible = self._eligible(doc)
        return AgentResult(
            success=True,
            agent=self.name,
            action="status",
            message=(
                f"{len(features)} features · {counts['pending']} pending · "
                f"{counts['in_progress']} in_progress · {counts['done']} done · "
                f"{counts['blocked']} blocked"
            ),
            data={
                "counts": counts,
                "in_progress": running,
                "eligible": [f["id"] for f in eligible],
                "features": [
                    {"id": f.get("id"), "title": f.get("title"), "status": f.get("status")}
                    for f in features
                ],
            },
            warnings=warnings,
        )

    def next(self) -> AgentResult:
        """La feature que toca: la in_progress si la hay, si no la primera elegible."""
        doc, error = self._load()
        if doc is None:
            return self._fail("next", error)

        running = [f for f in doc["features"] if f.get("status") == "in_progress"]
        if running:
            feat = running[0]
            return AgentResult(
                success=True, agent=self.name, action="next",
                message=f"Retoma {feat['id']} — {feat['title']} (ya estaba in_progress).",
                data=feat,
            )

        eligible = self._eligible(doc)
        if not eligible:
            blocked = [f["id"] for f in doc["features"] if f.get("status") == "blocked"]
            pending = [f["id"] for f in doc["features"] if f.get("status") == "pending"]
            if pending:
                return self._fail(
                    "next",
                    "Hay features pendientes pero ninguna tiene sus dependencias en done. "
                    "Revisa depends_on o desbloquea lo que falte.",
                    data={"pending": pending, "blocked": blocked},
                )
            return AgentResult(
                success=True, agent=self.name, action="next",
                message="Sin trabajo pendiente: el backlog está cerrado.",
                data={"blocked": blocked},
            )

        feat = eligible[0]
        return AgentResult(
            success=True, agent=self.name, action="next",
            message=f"Siguiente: {feat['id']} — {feat['title']}",
            data=feat,
        )

    def start(self, *, id: str = "", owner: str = "implementer") -> AgentResult:
        """Abre una feature: status in_progress y progress/current.md rellenado."""
        if not id:
            return self._fail("start", "Falta el id de la feature.",
                              needs=["¿Qué feature abro? Usa el id de featureslist.json (ej. DATA-001)."])

        doc, error = self._load()
        if doc is None:
            return self._fail("start", error)

        feat = self._find(doc, id)
        if feat is None:
            return self._fail("start", f"No existe la feature '{id}' en el backlog.")

        running = [f["id"] for f in doc["features"] if f.get("status") == "in_progress"]
        if running and running != [id]:
            return self._fail(
                "start",
                f"Ya hay trabajo abierto: {', '.join(running)}. Ciérralo o bloquéalo antes de abrir '{id}'.",
                data={"in_progress": running},
            )

        done = {f["id"] for f in doc["features"] if f.get("status") == "done"}
        missing_deps = [dep for dep in feat.get("depends_on", []) if dep not in done]
        if missing_deps:
            return self._fail(
                "start",
                f"'{id}' depende de {', '.join(missing_deps)}, que no están en done.",
                data={"missing_deps": missing_deps},
            )

        feat["status"] = "in_progress"
        feat["started"] = date.today().isoformat()
        # Abrir una feature reinicia el contador de rondas: si venía bloqueada
        # por agotar el bucle, se reabre con las tres rondas enteras — el
        # humano ya intervino, no tiene sentido heredar el castigo anterior.
        feat["review_rounds"] = 0
        feat.pop("blocked_reason", None)
        self._save(doc)

        self._progress_dir.mkdir(parents=True, exist_ok=True)
        criteria = "\n".join(f"- [ ] {c}" for c in feat.get("acceptance_criteria", []))
        self._current_file.write_text(
            CURRENT_TEMPLATE.format(
                fid=feat["id"],
                status="in_progress",
                started=feat["started"],
                owner=owner,
                description=feat.get("description", ""),
                criteria=criteria or "_(sin criterios definidos)_",
                log="_(pendiente)_",
                blockers="_(ninguno)_",
            ),
            encoding="utf-8",
        )

        return AgentResult(
            success=True, agent=self.name, action="start",
            message=f"{id} abierta (in_progress) y volcada en progress/current.md.",
            data={"id": id, "criteria": feat.get("acceptance_criteria", [])},
        )

    def gate(self, *, quick: bool = False) -> AgentResult:
        """Ejecuta ./init.sh y devuelve el veredicto estructurado."""
        script = self.ctx.root / "init.sh"
        if not script.exists():
            return self._fail("gate", "No existe init.sh: el arnés no tiene puerta.")

        args = ["bash", str(script), "--json"]
        if quick:
            args.append("--quick")
        proc = run_command(args, cwd=self.ctx.root, timeout=900)

        try:
            report = json.loads(proc.stdout)
        except json.JSONDecodeError:
            return self._fail(
                "gate",
                f"init.sh no devolvió JSON (exit {proc.returncode}).",
                data={"stdout": proc.stdout[-2000:], "stderr": proc.stderr[-2000:]},
            )

        failed = [c for c in report.get("checks", []) if c.get("status") == "fail"]
        return AgentResult(
            success=bool(report.get("ready")),
            agent=self.name,
            action="gate",
            message=(
                "ENTORNO LISTO — se puede trabajar."
                if report.get("ready")
                else f"ENTORNO BLOQUEADO — {len(failed)} check(s) fallando."
            ),
            data=report,
            warnings=[f"{c['check']}: {c['detail']}" for c in failed],
        )

    def finish(self, *, id: str = "", evidence: str = "", changes: str = "",
               decisions: str = "", pending: str = "") -> AgentResult:
        """
        Cierra una feature. REHÚSA si ./init.sh no pasa en verde: es la regla
        del arnés, y aquí es código, no una instrucción que se pueda ignorar.
        """
        if not id:
            return self._fail("finish", "Falta el id de la feature.",
                              needs=["¿Qué feature cierro? Usa su id de featureslist.json."])

        doc, error = self._load()
        if doc is None:
            return self._fail("finish", error)

        feat = self._find(doc, id)
        if feat is None:
            return self._fail("finish", f"No existe la feature '{id}' en el backlog.")
        if feat.get("status") == "done":
            return self._fail("finish", f"'{id}' ya está en done.")

        gate = self.gate()
        if not gate.success:
            return self._fail(
                "finish",
                f"NO se cierra '{id}': la puerta no pasa. {gate.message}",
                data=gate.data,
                warnings=gate.warnings,
            )

        if not evidence:
            return self._fail(
                "finish",
                f"'{id}' no se cierra sin evidencia.",
                needs=[
                    "Pega la salida real del comando que demuestra cada criterio "
                    "(pytest, make check, ./init.sh). Una afirmación no es evidencia."
                ],
            )

        feat["status"] = "done"
        feat["closed"] = date.today().isoformat()
        self._save(doc)

        gate_line = gate.data.get("checks", []) if isinstance(gate.data, dict) else []
        pytest_line = next(
            (c["detail"] for c in gate_line if c.get("check") == "pytest"), "init.sh en verde"
        )

        entry = (
            f"\n## {id} — {feat.get('title', '')}\n\n"
            f"- **Cerrada:** {feat['closed']}\n"
            f"- **Verificación:** ./init.sh en verde · {pytest_line}\n"
            f"- **Cambios:** {changes or '_(no indicados)_'}\n"
            f"- **Decisiones:** {decisions or '_(ninguna reseñable)_'}\n"
            f"- **Pendiente:** {pending or '_(nada)_'}\n\n"
            f"<details><summary>Evidencia</summary>\n\n```\n{evidence.strip()}\n```\n\n</details>\n"
        )
        self._progress_dir.mkdir(parents=True, exist_ok=True)
        with self._history_file.open("a", encoding="utf-8") as fh:
            fh.write(entry)

        self._current_file.write_text(IDLE_CURRENT, encoding="utf-8")

        return AgentResult(
            success=True, agent=self.name, action="finish",
            message=f"{id} cerrada. Histórico actualizado y current.md en idle.",
            data={"id": id, "closed": feat["closed"]},
        )

    def block(self, *, id: str = "", reason: str = "") -> AgentResult:
        """Marca una feature como bloqueada, con el motivo."""
        if not id or not reason:
            missing = []
            if not id:
                missing.append("¿Qué feature bloqueo? (id de featureslist.json)")
            if not reason:
                missing.append("¿Por qué se bloquea? Sin motivo no sirve de nada.")
            return self._fail("block", "Faltan datos para bloquear.", needs=missing)

        doc, error = self._load()
        if doc is None:
            return self._fail("block", error)

        feat = self._find(doc, id)
        if feat is None:
            return self._fail("block", f"No existe la feature '{id}' en el backlog.")

        feat["status"] = "blocked"
        feat["blocked_reason"] = reason
        self._save(doc)
        return AgentResult(
            success=True, agent=self.name, action="block",
            message=f"{id} bloqueada: {reason}",
            data={"id": id, "reason": reason},
        )

    def record(self, *, agent: str = "", id: str = "", content: str = "",
               verdict: str = "ok") -> AgentResult:
        """Guarda el informe de un subagente en progress/<agente>-<ID>.md."""
        if not agent or not id or not content:
            missing = []
            if not agent:
                missing.append("¿Qué subagente escribe? (explorer, implementer, reviewer)")
            if not id:
                missing.append("¿Sobre qué feature? (id de featureslist.json)")
            if not content:
                missing.append("¿Qué contenido? El informe no puede ir vacío.")
            return self._fail("record", "Faltan datos para guardar el informe.", needs=missing)

        self._progress_dir.mkdir(parents=True, exist_ok=True)
        path = self._progress_dir / f"{agent}-{id}.md"
        header = (
            f"# {agent} · {id}\n\n"
            f"- **Fecha:** {date.today().isoformat()}\n"
            f"- **Veredicto:** {verdict}\n\n"
        )
        path.write_text(header + content.strip() + "\n", encoding="utf-8")

        # El bucle implementer <-> reviewer es un patrón evaluador-optimizador,
        # y esos bucles necesitan tope: sin él, un reviewer exigente y un
        # implementer que no acierta queman contexto para siempre y nadie se
        # entera de cuántas vueltas llevan. Al agotarse, la feature se bloquea
        # sola y se escala al humano — en código, no confiando en que el líder
        # lleve la cuenta.
        rounds = None
        if agent == "reviewer" and _es_rechazo(verdict):
            doc, error = self._load()
            if doc is None:
                return self._fail("record", error)
            feat = self._find(doc, id)
            if feat is not None:
                rounds = int(feat.get("review_rounds", 0)) + 1
                feat["review_rounds"] = rounds
                if rounds >= MAX_REVIEW_ROUNDS:
                    feat["status"] = "blocked"
                    feat["blocked_reason"] = (
                        f"El reviewer rechazó {rounds} veces seguidas: el bucle se agotó."
                    )
                self._save(doc)

                if rounds >= MAX_REVIEW_ROUNDS:
                    return self._fail(
                        "record",
                        f"Informe guardado, pero '{id}' se bloquea: {rounds} rechazos seguidos.",
                        data={"path": str(path.relative_to(self.ctx.root)),
                              "verdict": verdict, "review_rounds": rounds},
                        needs=[
                            f"El reviewer ha rechazado '{id}' {rounds} veces. Repetir la misma "
                            f"iteración no lo va a arreglar: lee progress/reviewer-{id}.md y "
                            f"decide si el criterio es correcto, si la feature está mal "
                            f"planteada o si hace falta partirla en varias."
                        ],
                    )

        return AgentResult(
            success=True, agent=self.name, action="record",
            message=(
                f"Informe guardado en progress/{path.name}"
                + (f" · ronda de revisión {rounds}/{MAX_REVIEW_ROUNDS}" if rounds else "")
            ),
            warnings=(
                [f"Van {rounds} rechazos de {MAX_REVIEW_ROUNDS}: a la siguiente se bloquea."]
                if rounds and rounds == MAX_REVIEW_ROUNDS - 1 else []
            ),
            data={"path": str(path.relative_to(self.ctx.root)), "verdict": verdict,
                  "review_rounds": rounds},
        )

    def add(self, *, id: str = "", title: str = "", description: str = "",
            criteria: str = "", depends_on: str = "") -> AgentResult:
        """Añade una feature al backlog. `criteria` y `depends_on` van separados por `;`."""
        missing = []
        if not id:
            missing.append("¿Qué id le pongo? (ej. API-002)")
        if not title:
            missing.append("¿Cuál es el título de la feature?")
        if not criteria:
            missing.append("¿Cuáles son los criterios de aceptación? Sepáralos con ';'.")
        if missing:
            return self._fail("add", "Faltan datos para añadir la feature.", needs=missing)

        doc, error = self._load()
        if doc is None:
            return self._fail("add", error)
        if self._find(doc, id) is not None:
            return self._fail("add", f"Ya existe una feature con id '{id}'.")

        feature = {
            "id": id,
            "title": title,
            "description": description or title,
            "acceptance_criteria": [c.strip() for c in criteria.split(";") if c.strip()],
            "status": "pending",
            "depends_on": [d.strip() for d in depends_on.split(";") if d.strip()],
        }
        unknown = [d for d in feature["depends_on"] if self._find(doc, d) is None]
        if unknown:
            return self._fail("add", f"depends_on apunta a features que no existen: {', '.join(unknown)}.")

        doc["features"].append(feature)
        self._save(doc)
        return AgentResult(
            success=True, agent=self.name, action="add",
            message=f"{id} añadida al backlog ({len(feature['acceptance_criteria'])} criterios).",
            data=feature,
        )
