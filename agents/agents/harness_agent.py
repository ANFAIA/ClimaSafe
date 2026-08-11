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
import re
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

INDEX_CURRENT_HEADER = """# Tareas actuales

> {n} features in_progress a la vez, como mucho una por dueño. Este fichero es
> estado **derivado** de `featureslist.json`: lo regenera `harness` en cada
> `start` / `finish` / `block`, así que nadie pisa el trabajo de otro. El
> detalle de cada tarea (criterios, bitácora, bloqueos) vive en su propio
> fichero, el de la última columna.

| Feature | Dueño | Iniciada | Detalle |
|---------|-------|----------|---------|
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

    # -- dueños --------------------------------------------------------------
    # El candado del arnés es por dueño, no global: dos asistentes trabajando en
    # paralelo (uno por feature) no se bloquean. Las features sin campo `owner`
    # comparten un mismo dueño implícito —el legado—, así que quien no usa
    # `--owner` sigue viendo exactamente el comportamiento de antes: una sola
    # feature abierta a la vez.
    @staticmethod
    def _norm_owner(owner: str | None) -> str:
        return (owner or "").strip().lower()

    @classmethod
    def _owner_slug(cls, owner: str | None) -> str:
        """Nombre de fichero seguro para un dueño. '' si no tiene dueño explícito."""
        return re.sub(r"[^a-z0-9_-]+", "-", cls._norm_owner(owner)).strip("-")

    def _detail_file(self, feat: dict) -> Path:
        """
        Fichero de detalle de una feature en curso. Lo nombra su dueño si lo
        tiene (`current-claude.md`), y si no su propio id (`current-data-004.md`):
        una feature abierta antes de ARNES-013 —sin campo `owner`— también
        conserva su objetivo y sus criterios cuando el índice se activa.
        """
        slug = self._owner_slug(feat.get("owner")) or self._owner_slug(feat.get("id"))
        return self._progress_dir / f"current-{slug}.md"

    # -- progress/ derivado --------------------------------------------------
    @staticmethod
    def _render_current(feat: dict) -> str:
        criteria = "\n".join(f"- [ ] {c}" for c in feat.get("acceptance_criteria", []))
        return CURRENT_TEMPLATE.format(
            fid=feat.get("id", ""),
            status=feat.get("status", ""),
            started=feat.get("started", "—"),
            owner=feat.get("owner") or "implementer",
            description=feat.get("description", ""),
            criteria=criteria or "_(sin criterios definidos)_",
            log="_(pendiente)_",
            blockers="_(ninguno)_",
        )

    def _refresh_current(self, doc: dict) -> None:
        """
        Reescribe `progress/current.md` a partir del backlog. Es estado
        derivado: no depende de quién llamó, así que cerrar una feature nunca
        borra el `current.md` de otro dueño.

        0 in_progress → idle · 1 → la plantilla de siempre · 2 o más → un
        índice que apunta al `current-<dueño>.md` de cada uno.
        """
        self._progress_dir.mkdir(parents=True, exist_ok=True)
        running = [f for f in doc["features"] if f.get("status") == "in_progress"]

        # Cada tarea en curso tiene su fichero de detalle: con dueño explícito
        # siempre, y sin dueño en cuanto se activa el índice —si no, el
        # objetivo y los criterios de una feature legada desaparecerían de
        # progress/ al abrir otro asistente la suya—. Si ya existe no se
        # sobrescribe: la bitácora que haya escrito su dueño no se toca.
        for feat in running:
            path = self._detail_file(feat)
            if not path.exists() and (feat.get("owner") or len(running) > 1):
                path.write_text(self._render_current(feat), encoding="utf-8")

        if not running:
            self._current_file.write_text(IDLE_CURRENT, encoding="utf-8")
            return
        if len(running) == 1:
            self._current_file.write_text(self._render_current(running[0]), encoding="utf-8")
            return

        rows = [
            f"| {feat.get('id', '')} | {feat.get('owner') or '_(sin dueño)_'} | "
            f"{feat.get('started', '—')} | `progress/{self._detail_file(feat).name}` |"
            for feat in running
        ]
        self._current_file.write_text(
            INDEX_CURRENT_HEADER.format(n=len(running)) + "\n".join(rows) + "\n",
            encoding="utf-8",
        )

    def _release_detail_file(self, feat: dict) -> None:
        """
        Retira el fichero de detalle de esta feature al cerrarla o bloquearla.
        Es por feature, así que nunca alcanza al de otra: el candado ya impide
        que un dueño tenga dos abiertas, y las que no tienen dueño se nombran
        por su id.
        """
        self._detail_file(feat).unlink(missing_ok=True)

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

        open_feats = [f for f in features if f.get("status") == "in_progress"]
        running = [f["id"] for f in open_feats]
        # varias in_progress a la vez es normal si son de dueños distintos; lo
        # que el arnés no admite es que un mismo dueño tenga dos abiertas
        por_dueño: dict[str, list[str]] = {}
        for feat in open_feats:
            por_dueño.setdefault(self._norm_owner(feat.get("owner")), []).append(feat["id"])
        warnings = [
            f"{len(ids)} features in_progress del mismo dueño "
            f"{repr(dueño) if dueño else '(sin dueño)'}: {', '.join(ids)}. "
            f"El arnés espera una por dueño: cierra o bloquea las demás."
            for dueño, ids in por_dueño.items()
            if len(ids) > 1
        ]

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

    def start(self, *, id: str = "", owner: str = "") -> AgentResult:
        """
        Abre una feature: status in_progress y progress/ regenerado.

        `owner` identifica a quién abre la tarea, y el candado es por dueño:
        dos asistentes en paralelo pueden tener una feature abierta cada uno,
        pero ninguno puede abrir dos. Sin `--owner` el comportamiento es el de
        siempre (una sola feature abierta entre todas las que no tienen dueño).
        """
        if not id:
            return self._fail("start", "Falta el id de la feature.",
                              needs=["¿Qué feature abro? Usa el id de featureslist.json (ej. DATA-001)."])

        doc, error = self._load()
        if doc is None:
            return self._fail("start", error)

        feat = self._find(doc, id)
        if feat is None:
            return self._fail("start", f"No existe la feature '{id}' en el backlog.")

        mine = [
            f["id"]
            for f in doc["features"]
            if f.get("status") == "in_progress"
            and f["id"] != id
            and self._norm_owner(f.get("owner")) == self._norm_owner(owner)
        ]
        if mine:
            quien = f"'{owner.strip()}'" if owner.strip() else "sin dueño (legado)"
            return self._fail(
                "start",
                f"Ya hay trabajo abierto de {quien}: {', '.join(mine)}. "
                f"Ciérralo o bloquéalo antes de abrir '{id}'.",
                data={"in_progress": mine, "owner": owner.strip()},
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
        # sin --owner la feature no gana el campo: se queda con el dueño
        # implícito (el legado) y el backlog sigue siendo el de antes
        if owner.strip():
            feat["owner"] = owner.strip()
        # Abrir una feature reinicia el contador de rondas: si venía bloqueada
        # por agotar el bucle, se reabre con las tres rondas enteras — el
        # humano ya intervino, no tiene sentido heredar el castigo anterior.
        feat["review_rounds"] = 0
        feat.pop("blocked_reason", None)
        self._save(doc)

        self._progress_dir.mkdir(parents=True, exist_ok=True)
        # con dueño explícito, la ficha propia se escribe siempre (y se
        # refresca al reabrir); sin dueño la crea `_refresh_current` solo si
        # hace falta el índice, para no cambiarle nada al flujo de siempre
        detail = self._detail_file(feat) if feat.get("owner") else None
        if detail is not None:
            detail.write_text(self._render_current(feat), encoding="utf-8")
        self._refresh_current(doc)

        destino = f"progress/{detail.name} y progress/current.md" if detail else "progress/current.md"
        return AgentResult(
            success=True, agent=self.name, action="start",
            message=f"{id} abierta (in_progress) y volcada en {destino}.",
            data={"id": id, "owner": feat.get("owner", ""),
                  "criteria": feat.get("acceptance_criteria", [])},
        )

    def gate(self, *, quick: bool = False) -> AgentResult:
        """Ejecuta ./init.sh y devuelve el veredicto estructurado."""
        script = self.ctx.root / "init.sh"
        if not script.exists():
            return self._fail("gate", "No existe init.sh: el arnés no tiene puerta.")

        args = ["bash", str(script), "--json"]
        if quick:
            args.append("--quick")
        # La suite completa (pytest tests/ + agents/tests/) tarda ~25 min; un
        # timeout menor (900s) mataba gate/finish y ninguna feature se podía cerrar.
        proc = run_command(args, cwd=self.ctx.root, timeout=3600)

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
               decisions: str = "", pending: str = "", commit: bool = False) -> AgentResult:
        """
        Cierra una feature. REHÚSA si ./init.sh no pasa en verde: es la regla
        del arnés, y aquí es código, no una instrucción que se pueda ignorar.

        Al cerrar, encadena el flujo de release ligero de GIT-001: sube un
        punto de versión patch en pyproject.toml/README.md (delegando en
        `DocumentationAgent.bump_version`) y propone el mensaje de commit del
        cierre (delegando en `GitAgent`).

        El commit automático (ARNES-014) SOLO se intenta cuando el lider pasa
        el flag explícito `commit=True` (CLI: `--commit`). Sin el flag, `finish`
        se comporta como antes de ARNES-014: propone el mensaje en
        `data.commit_suggestion` y NO commitea — ningún otro agente ni asistente
        commitea por su cuenta. Con el flag, commitea acotado a las rutas de
        `--changes` más los ficheros del propio cierre, con mensaje Conventional
        Commits sin línea de co-autoría; si no queda nada que commitear, si
        `--changes` viene vacío o trae rutas inexistentes, o si el árbol trae
        cambios ajenos al ticket, avisa en `warnings` y NO commitea. Ninguno de
        los encadenados bloquea el cierre: si fallan, se avisa en `warnings` y
        la feature queda cerrada igualmente.
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

        # progress/ es estado derivado: se retira solo la ficha de ESTA feature
        # y se regenera current.md desde el backlog, así que si otro asistente
        # sigue trabajando su tarea no se pierde
        self._release_detail_file(feat)
        self._refresh_current(doc)

        release = self._release_on_close(feat, changes=changes, commit=commit)
        message = f"{id} cerrada. Histórico actualizado y current.md regenerado."
        if "version_bump" in release["data"]:
            message += f" README bumped a {release['data']['version_bump']['new_version']}."
        auto = release["data"].get("auto_commit")
        if auto and auto.get("success"):
            message += f" Commit automático creado: {auto['message']}."

        return AgentResult(
            success=True, agent=self.name, action="finish",
            message=message,
            data={"id": id, "closed": feat["closed"], **release["data"]},
            warnings=release["warnings"],
        )

    @staticmethod
    def _next_patch_version(version: str | None) -> str | None:
        """Sube un punto de patch: '0.1.0' -> '0.1.1'. None si no es semver."""
        if not version:
            return None
        match = re.match(r"^(\d+)\.(\d+)\.(\d+)$", version.strip())
        if not match:
            return None
        major, minor, patch = (int(g) for g in match.groups())
        return f"{major}.{minor}.{patch + 1}"

    def _release_on_close(self, feat: dict, changes: str = "", commit: bool = False) -> dict:
        """
        Flujo de cierre de GIT-001: bump de versión patch (README incluido,
        vía `DocumentationAgent.bump_version`) y propuesta de mensaje de
        commit (vía `GitAgent`, con el id de la feature como subject). El
        commit automático de ARNES-014 solo se intenta con el flag explícito
        `commit=True` (el que pasa el lider): sin flag se propone el mensaje
        y no se commitea. Si no queda nada que commitear, si `--changes` viene
        vacío o trae rutas inexistentes, se avisa en `warnings` y no se
        commitea. Los cambios ajenos al ticket se avisan pero no impiden el
        commit: este queda acotado a las rutas del ticket y del cierre.

        Nunca bloquea el cierre: si algo falla (sin versión parseable, sin
        cambios que resumir, sin repo git) se avisa en `warnings` y se
        devuelve lo que sí se pudo hacer.
        """
        warnings: list[str] = []
        data: dict[str, Any] = {}

        current = None
        pyproject = self.ctx.pyproject_file
        if pyproject.exists():
            match = re.search(r'^version = "([^"]+)"', pyproject.read_text(encoding="utf-8"), re.MULTILINE)
            current = match.group(1) if match else None
        next_version = self._next_patch_version(current)
        if next_version is None:
            warnings.append(
                "No se pudo subir la versión: pyproject.toml no tiene 'version = \"X.Y.Z\"' "
                "parseable — el README no subió de punto."
            )
        else:
            from agents.agents.documentation_agent import DocumentationAgent

            doc_agent = DocumentationAgent(context=self.ctx)
            bump = doc_agent.run("bump_version", new_version=next_version)
            warnings.extend(bump.warnings)
            if bump.success:
                data["version_bump"] = {
                    "new_version": next_version,
                    "changed_files": bump.data["changed_files"],
                }
            else:
                warnings.append(f"El bump a '{next_version}' no se aplicó: {bump.message}")

        from agents.agents.git_agent import GitAgent

        git_agent = GitAgent(context=self.ctx)
        suggestion = git_agent.run("suggest_commit_message", hint=f"cierra {feat.get('id', '')}")
        if suggestion.success:
            data["commit_suggestion"] = suggestion.data["suggested_message"]
            data["suggested_changed_files"] = suggestion.data["changed_files"]
            warnings.extend(suggestion.warnings)
        else:
            warnings.append(f"No se generó propuesta de commit: {suggestion.message}")

        if commit:
            auto = self._auto_commit(feat, changes)
            warnings.extend(auto["warnings"])
            data.update(auto["data"])

        return {"warnings": warnings, "data": data}

    @staticmethod
    def _parse_changes(changes: str) -> list[str]:
        """Rutas del ticket pasadas en `--changes`, separadas por ';' (como criteria/depends_on)."""
        return [c.strip() for c in changes.split(";") if c.strip()]

    def _closure_paths(self) -> list[str]:
        """Rutas relativas que el propio cierre toca: solo existen si hay algo que commitear."""
        paths = ["featureslist.json", "progress/"]
        for f in (self.ctx.pyproject_file, self.ctx.readme_file):
            if f.exists():
                paths.append(str(f.relative_to(self.ctx.root)))
        return paths

    def _auto_commit(self, feat: dict, changes: str) -> dict:
        """
        ARNES-014 — commit automático del cierre. Solo se llama cuando el
        lider cierra con el flag explícito (`finish(..., commit=True)`): sin
        ese flag el cierre propone el mensaje y NO commitea.

        El commit se acota a las rutas de `--changes` (separadas por ';') más
        los ficheros del propio cierre (`featureslist.json`, `progress/`,
        `pyproject.toml`, `README.md`). Si `--changes` viene vacío o trae
        rutas que no existen, no se commitea nada y se avisa. Si el árbol trae
        cambios ajenos al ticket (trabajo de otro dueño o de otro ticket), se
        avisa en `warnings` y el commit continúa acotado: solo entran las
        rutas del ticket y del cierre, lo ajeno queda sin tocar. Si tras el
        filtrado no queda nada staged, no se commitea y se avisa.

        El mensaje es Conventional Commits con el id de la feature como
        subject y sin línea de co-autoría: se genera con
        `GitAgent.suggest_commit_message` una vez las rutas están en el índice,
        así el mensaje describe exactamente lo que se commitea. Delega en
        `GitTool` (git add/commit): este agente no inventa una capa git nueva.
        Los fallos del commit nunca bloquean el cierre: van a `warnings`.
        """
        warnings: list[str] = []
        data: dict[str, Any] = {}

        rutas = self._parse_changes(changes)
        if not rutas:
            warnings.append(
                "finish sin --changes: no hay rutas del ticket que commitear — "
                "no se commitea nada."
            )
            return {"warnings": warnings, "data": data}

        missing = [r for r in rutas if not (self.ctx.root / r).exists()]
        if missing:
            warnings.append(
                f"--changes trae rutas que no existen ({', '.join(missing)}) — "
                "no se commitea nada."
            )
            return {"warnings": warnings, "data": data}

        from agents.agents.git_agent import GitAgent

        git_agent = GitAgent(context=self.ctx)
        if not git_agent.git.is_repo():
            warnings.append("No es un repositorio git — no se commitea nada.")
            return {"warnings": warnings, "data": data}

        stage_paths = set(rutas) | set(self._closure_paths())
        allowed = set(stage_paths)
        # El log de auditoría (`agents/workspace/audit/audit.jsonl`) lo escribe
        # el propio cierre: cada `run` de los agentes delegados queda auditado.
        # No entra en el commit (es un log de trabajo, gitignored en el repo
        # real), pero su presencia no es trabajo ajeno que deba manchar el
        # commit ni el aviso de acumulación.
        audit_dir = self.ctx.workspace_dir / "audit"
        if audit_dir.exists():
            allowed.add(str(audit_dir.relative_to(self.ctx.root)) + "/")

        def _in_allowed(path: str) -> bool:
            if path in allowed:
                return True
            # "progress/" cubre todo lo que cuelga de él; "mi_paquete/" (dir
            # untracked agrupado por git) cubre cualquier fichero declarado
            # dentro — p. ej. "mi_paquete/nuevo.py"
            for p in allowed:
                base = p if p.endswith("/") else p + "/"
                if path.startswith(base) or base.startswith(path.rstrip("/") + "/"):
                    return True
            return False

        # Cambios ajenos al ticket: se avisan pero NO paran el commit. El
        # `git add` y el `git commit -- <paths>` están acotados a las rutas
        # del ticket + cierre, así que lo ajeno jamás entra en el commit
        # aunque el árbol esté sucio (p. ej. otra sesión trabajando en
        # paralelo). Parar bloqueaba el cierre automático siempre que hubiera
        # cualquier cambio fuera del ticket.
        ajenas = [p for _, p in git_agent.git.status_porcelain(all_untracked=True) if not _in_allowed(p)]
        if ajenas:
            warnings.append(
                f"Cambios fuera del ticket ({', '.join(ajenas[:5])}"
                f"{'…' if len(ajenas) > 5 else ''}) — trabajo de otro dueño o "
                "de otro ticket: se dejan fuera del commit del cierre (el "
                "commit solo incluye las rutas de este ticket)."
            )

        stage = git_agent.git.add(*stage_paths)
        if not stage.ok:
            warnings.append(f"'git add' falló: {stage.stderr.strip()} — no se commitea nada.")
            return {"warnings": warnings, "data": data}

        # Tras filtrar por las rutas del ticket, ¿queda algo staged? Si no
        # (p. ej. las rutas existen pero ya estaban commiteadas), no hay nada
        # que commitear: se avisa y se cierra igual.
        staged = git_agent.git.staged_files(*stage_paths)
        if not staged:
            warnings.append(
                "Tras el filtrado no queda nada staged — no se commitea nada."
            )
            return {"warnings": warnings, "data": data}

        suggestion = git_agent.run("suggest_commit_message", hint=f"cierra {feat.get('id', '')}")
        if not suggestion.success:
            warnings.append(f"No se generó mensaje de commit: {suggestion.message} — no se commitea nada.")
            return {"warnings": warnings, "data": data}
        message = suggestion.data["suggested_message"]

        # commit acotado por pathspec: aunque el índice arrastrara algo staged
        # de antes, el commit solo puede contener las rutas autorizadas
        commit_result = git_agent.git.commit(message, *stage_paths)
        if not commit_result.ok:
            warnings.append(f"'git commit' falló: {commit_result.stderr.strip()} — no se commiteó.")
            return {"warnings": warnings, "data": data}

        proc = run_command(["git", "show", "--name-only", "--format=", "HEAD"], cwd=self.ctx.root)
        files = [f for f in proc.stdout.splitlines() if f.strip()]
        data["auto_commit"] = {"success": True, "message": message, "files": files}
        return {"warnings": warnings, "data": data}

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
        self._release_detail_file(feat)
        self._refresh_current(doc)
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
