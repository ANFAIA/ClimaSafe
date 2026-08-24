"""
agents.subagent — Delegación a subagentes y carga de definiciones markdown.

Los subagentes se definen en .opencode/agents/*.md (reutilizando las que ya
viven ahí). Cada subagente arranca con contexto vacío (no hereda el historial
del padre), siguiendo el principio de AGENTS.md:
  "Al lanzar un subagente, no le heredes contexto."

Decisión justificada (criterio 2 de ARNES-007):
  - Heredar historial duplicaría tokens (coste lineal) y contaminaría al
    subagente con decisiones del padre que no le incumben.
  - Contexto vacío = subagente enfocado, coste predecible, resultado limpio.
  - El padre resume lo que el subagente necesita en el prompt de delegación,
    no le vuelca la conversación entera.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


_AGENTS_DIR = Path(__file__).parent.parent / ".opencode" / "agents"

# Agentes que el loop ya conoce (no se cargan de markdown)
_BUILTIN_AGENTS: frozenset[str] = frozenset({"harness", "lider"})


@dataclass
class SubagentConfig:
    """Configuración de un subagente cargada desde markdown."""
    name: str
    system_prompt: str
    allowed_tools: set[str]
    source_file: Path


def load_subagent_config(agent_name: str) -> SubagentConfig | None:
    """
    Carga la configuración de un subagente desde .opencode/agents/<name>.md.

    Extrae:
      - El contenido del markdown como system prompt
      - Un catálogo de tools a partir de las secciones "## Herramientas"
        o del contenido del fichero

    Devuelve None si el agente no existe.
    """
    md_path = _AGENTS_DIR / f"{agent_name}.md"
    if not md_path.exists():
        return None

    content = md_path.read_text(encoding="utf-8")

    # Extract system prompt: everything before the first ## section,
    # or the full content if no sections
    system_prompt = _extract_system_prompt(content)

    # Extract tool catalog from the markdown
    allowed_tools = _extract_tools(content, agent_name)

    return SubagentConfig(
        name=agent_name,
        system_prompt=system_prompt,
        allowed_tools=allowed_tools,
        source_file=md_path,
    )


def _extract_system_prompt(content: str) -> str:
    """
    Extrae el system prompt del contenido markdown.

    Estrategia: todo el contenido hasta la primera sección ## que contenga
    'herramientas' o 'tools' (excluida), o las primeras ~20 líneas si no
    hay sección de tools.
    """
    lines = content.split("\n")
    prompt_lines: list[str] = []
    for line in lines:
        # Stop at tool section
        if re.match(r"^##\s+(herramientas|tools|comandos)", line, re.IGNORECASE):
            break
        prompt_lines.append(line)

    # If we didn't find a tools section, take first 20 lines as prompt
    if len(prompt_lines) == len(lines):
        prompt_lines = lines[:20]

    return "\n".join(prompt_lines).strip()


def _extract_tools(content: str, agent_name: str) -> set[str]:
    """
    Extrae las tools permitidas del contenido markdown.

    Mira secciones como '## Herramientas' o '## Tools' y extrae nombres
    de comandos/tools mencionados. También mira el catálogo de seguridad
    del agente como fallback.
    """
    tools: set[str] = set()

    # Look for tool sections
    tool_section_pattern = re.compile(
        r"^##\s+(herramientas|tools|comandos)", re.IGNORECASE
    )
    in_tool_section = False
    for line in content.split("\n"):
        if tool_section_pattern.match(line):
            in_tool_section = True
            continue
        if in_tool_section:
            if line.startswith("##"):
                break
            # Extract tool names from code blocks or backtick-wrapped words
            for match in re.finditer(r"`(\w+)`", line):
                tools.add(match.group(1))

    # If no tools found in markdown, use the security policy as fallback
    if not tools:
        from agents.security import AGENT_POLICIES
        policy = AGENT_POLICIES.get(agent_name)
        if policy:
            tools = set(policy.allowed_tools)

    return tools


def get_available_subagents() -> dict[str, SubagentConfig]:
    """Devuelve todos los subagentes disponibles cargados de markdown."""
    configs: dict[str, SubagentConfig] = {}
    if not _AGENTS_DIR.exists():
        return configs

    for md_file in sorted(_AGENTS_DIR.glob("*.md")):
        name = md_file.stem
        if name in _BUILTIN_AGENTS:
            continue
        config = load_subagent_config(name)
        if config:
            configs[name] = config

    return configs
