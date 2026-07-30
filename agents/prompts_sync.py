"""
agents.prompts_sync — Mantiene los prompts al día con el código.

Cada `agents/prompts/<agente>_agent.md` tiene dos partes:

- **La prosa**, escrita a mano: el criterio del agente, qué matizar al
  reportar, qué no dar por cierto. Eso no se genera — es lo valioso.
- **Un bloque autogenerado** con sus acciones reales y sus límites, derivado
  de `actions()` y de `agents/contracts.py`.

El bloque existe por dos razones concretas:

1. **Autosuficiencia.** Un prompt sin acciones obliga al asistente a gastar un
   `describe <agente>` extra antes de poder hacer nada. La tabla va incluida.
2. **Deriva.** Las mismas reglas estaban escritas en `contracts.py` y
   reescritas a mano en la prosa de cada prompt. Dos fuentes de verdad
   divergen siempre: el agente `doc` estuvo registrado sin contrato y nadie se
   enteró hasta que falló un test.

Uso:
    uv run python -m agents.prompts_sync            # comprueba (no escribe)
    uv run python -m agents.prompts_sync --write    # regenera los bloques
"""

from __future__ import annotations

import argparse
import inspect
import sys
from pathlib import Path

from agents.context import SharedContext, get_context
from agents.contracts import contract_for
from agents.core.registry import agent_registry

BEGIN = "<!-- BEGIN AUTOGEN — lo regenera `make prompts-sync`; no lo edites a mano -->"
END = "<!-- END AUTOGEN -->"

#: Los agentes del arnés viven en `.opencode/agents/` como única fuente. Cada
#: asistente los quiere en su propio sitio y formato, así que se espejan desde
#: ahí en vez de mantener N copias a mano.
HARNESS_AGENTS = ("lider", "explorer", "implementer", "reviewer")

CLAUDE_TOOLS = {
    # El explorer es de solo lectura: dárselo por escrito al asistente es más
    # fiable que pedírselo por favor en el prompt.
    "explorer": "Read, Grep, Glob, Bash",
}


def _signature_args(method) -> str:
    """Argumentos de una acción, marcando cuáles son obligatorios."""
    try:
        sig = inspect.signature(method)
    except (TypeError, ValueError):
        return "—"

    required, optional = [], []
    for name, param in sig.parameters.items():
        if name == "self" or param.kind in (param.VAR_POSITIONAL, param.VAR_KEYWORD):
            continue
        if param.default is inspect.Parameter.empty:
            required.append(f"`--{name}`")
        else:
            optional.append(f"`--{name}`")

    partes = []
    if required:
        partes.append(", ".join(required) + " (obligatorio)")
    if optional:
        partes.append(", ".join(optional))
    return " · ".join(partes) if partes else "—"


def render_block(agent_name: str, agent) -> str:
    """El bloque autogenerado de un agente."""
    lineas = [BEGIN, "", "## Acciones", "", "| Acción | Argumentos |", "|--------|------------|"]

    acciones = agent.actions()
    for nombre in acciones:
        lineas.append(f"| `run {agent_name} {nombre}` | {_signature_args(acciones[nombre])} |")
    if not acciones:
        lineas.append("| _(sin acciones)_ | — |")

    contrato = contract_for(agent_name)
    if contrato is not None:
        lineas += ["", "## Límites", "", f"**Rol.** {contrato.role}", ""]
        if contrato.cannot:
            lineas.append("**No hace:**")
            lineas += [f"- {item}" for item in contrato.cannot]
            lineas.append("")
        if contrato.needs:
            lineas.append("**Necesita que le den:** " + "; ".join(contrato.needs))
            lineas.append("")
        if contrato.owns:
            lineas.append("**Escribe en (nadie más toca esto):** " + ", ".join(contrato.owns))
            lineas.append("")
        if contrato.collaborates:
            lineas.append("**Se apoya en:** " + ", ".join(contrato.collaborates))
            lineas.append("")

    lineas.append(END)
    return "\n".join(lineas).rstrip() + "\n"


def _splice(texto: str, bloque: str) -> str:
    """Sustituye el bloque autogenerado, o lo añade al final si no existía."""
    inicio = texto.find(BEGIN)
    fin = texto.find(END)
    if inicio != -1 and fin != -1 and fin > inicio:
        return texto[:inicio] + bloque + texto[fin + len(END):].lstrip("\n")
    return texto.rstrip() + "\n\n" + bloque


def _first_paragraph(texto: str) -> str:
    """La primera línea de prosa del prompt, para la `description` del subagente."""
    for linea in texto.splitlines():
        limpia = linea.strip()
        if limpia and not limpia.startswith("#"):
            return limpia.rstrip(".")
    return "Agente del arnés"


def render_claude_agent(nombre: str, cuerpo: str) -> str:
    """
    El mismo agente, en el formato que espera Claude Code.

    Claude Code lee `.claude/agents/<n>.md` con frontmatter YAML; opencode lee
    `.opencode/agents/<n>.md` sin él. Es el mismo prompt: se espeja para que
    los dos asistentes se comporten igual, en vez de mantener dos versiones
    que acaban divergiendo.
    """
    descripcion = _first_paragraph(cuerpo)
    lineas = ["---", f"name: {nombre}", f"description: {descripcion}"]
    if nombre in CLAUDE_TOOLS:
        lineas.append(f"tools: {CLAUDE_TOOLS[nombre]}")
    lineas += ["---", "", "<!-- Generado desde .opencode/agents/" + nombre + ".md por"
               " `make assistants-sync`. Edita el original, no este fichero. -->", "", cuerpo.strip(), ""]
    return "\n".join(lineas)


def sync_assistants(*, write: bool, context: SharedContext | None = None) -> list[str]:
    """
    Espeja los agentes del arnés de `.opencode/agents/` a `.claude/agents/`.

    Devuelve la lista de los que estaban desincronizados.
    """
    ctx = context or get_context()
    origen = ctx.root / ".opencode" / "agents"
    destino = ctx.root / ".claude" / "agents"

    desincronizados: list[str] = []
    for nombre in HARNESS_AGENTS:
        fuente = origen / f"{nombre}.md"
        if not fuente.exists():
            continue
        esperado = render_claude_agent(nombre, fuente.read_text(encoding="utf-8"))
        objetivo = destino / f"{nombre}.md"
        actual = objetivo.read_text(encoding="utf-8") if objetivo.exists() else None
        if actual != esperado:
            desincronizados.append(nombre)
            if write:
                destino.mkdir(parents=True, exist_ok=True)
                objetivo.write_text(esperado, encoding="utf-8")

    return desincronizados


def sync(*, write: bool, context: SharedContext | None = None) -> tuple[list[str], list[str]]:
    """
    Devuelve (desincronizados, sin_prompt).

    Con `write=True` además los deja escritos. `context` se puede pasar
    explícitamente — igual que a cualquier agente — para que los tests
    trabajen sobre un proyecto temporal y no sobre el repo real.
    """
    ctx = context or get_context()
    agent_registry.discover()
    prompts_dir = ctx.root / "agents" / "prompts"

    desincronizados: list[str] = []
    sin_prompt: list[str] = []

    for nombre in sorted(agent_registry.all()):
        ruta = prompts_dir / f"{nombre}_agent.md"
        if not ruta.exists():
            sin_prompt.append(nombre)
            continue

        agente = agent_registry.get(nombre)(context=ctx)
        bloque = render_block(nombre, agente)
        actual = ruta.read_text(encoding="utf-8")
        nuevo = _splice(actual, bloque)
        if nuevo != actual:
            desincronizados.append(nombre)
            if write:
                ruta.write_text(nuevo, encoding="utf-8")

    return desincronizados, sin_prompt


def main() -> int:
    parser = argparse.ArgumentParser(prog="python -m agents.prompts_sync")
    parser.add_argument("--write", action="store_true", help="Escribe los bloques (si no, solo comprueba)")
    args = parser.parse_args()

    desincronizados, sin_prompt = sync(write=args.write)
    asistentes = sync_assistants(write=args.write)

    if asistentes:
        if args.write:
            print(f"✔ {len(asistentes)} subagente(s) espejados a .claude/agents/: {', '.join(asistentes)}")
        else:
            print(f"✘ {len(asistentes)} subagente(s) desincronizados en .claude/agents/: {', '.join(asistentes)}")
            print("  Ejecuta 'make assistants-sync' para regenerarlos.")

    if sin_prompt:
        print(f"✘ {len(sin_prompt)} agente(s) registrados sin prompt en agents/prompts/:")
        for nombre in sin_prompt:
            print(f"    {nombre}  → falta agents/prompts/{nombre}_agent.md")

    if args.write:
        if desincronizados:
            print(f"✔ {len(desincronizados)} prompt(s) actualizados: {', '.join(desincronizados)}")
        elif not asistentes:
            print("✔ Todo estaba ya al día.")
        return 1 if sin_prompt else 0

    if desincronizados:
        print(f"✘ {len(desincronizados)} prompt(s) desincronizados del código: {', '.join(desincronizados)}")
        print("  Ejecuta 'make prompts-sync' para regenerarlos.")

    if not desincronizados and not sin_prompt and not asistentes:
        print("✔ Prompts y subagentes sincronizados con el código y con los contratos.")
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
