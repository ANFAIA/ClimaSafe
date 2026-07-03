"""
agents.cli — CLI para usar el sistema de agentes desde línea de comandos.

Uso:
    uv run python -m agents list
    uv run python -m agents describe git
    uv run python -m agents run git suggest_commit_message
    uv run python -m agents run data eda_report --filename dataset.csv --target-col target
    uv run python -m agents ask "genera el changelog desde el último tag"
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any

from agents.context import get_context
from agents.core.base_agent import AgentResult
from agents.core.registry import agent_registry
from agents.orchestrator import Orchestrator


def _print_result(result: AgentResult) -> None:
    status = "✔" if result.success else "✘"
    print(f"{status} [{result.agent}.{result.action}] {result.message}")
    for warning in result.warnings:
        print(f"  ⚠ {warning}")
    if result.data is not None:
        try:
            print(json.dumps(result.data, indent=2, ensure_ascii=False, default=str))
        except TypeError:
            print(result.data)


def _parse_kwargs(pairs: list[str]) -> dict[str, Any]:
    """Convierte ['--target-col', 'target', '--max-count', '10'] en {'target_col': 'target', 'max_count': 10}."""
    kwargs: dict[str, Any] = {}
    key = None
    for token in pairs:
        if token.startswith("--"):
            key = token[2:].replace("-", "_")
            kwargs[key] = True  # flag booleano por defecto, se sobreescribe si sigue un valor
        elif key is not None:
            value: Any = token
            if value.lower() in ("true", "false"):
                value = value.lower() == "true"
            elif value.replace(".", "", 1).lstrip("-").isdigit():
                value = float(value) if "." in value else int(value)
            kwargs[key] = value
            key = None
    return kwargs


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m agents", description="Sistema de agentes del proyecto.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("list", help="Lista todos los agentes registrados.")

    describe_p = subparsers.add_parser("describe", help="Detalle de un agente: acciones y capacidades.")
    describe_p.add_argument("agent_name")

    run_p = subparsers.add_parser("run", help="Ejecuta una acción concreta de un agente.")
    run_p.add_argument("agent_name")
    run_p.add_argument("action")
    run_p.add_argument("kwargs", nargs=argparse.REMAINDER, help="--kwarg valor --otro-kwarg valor")

    ask_p = subparsers.add_parser("ask", help="Rutea una petición en lenguaje natural al agente más relevante.")
    ask_p.add_argument("query")

    subparsers.add_parser("tools", help="Lista las herramientas registradas.")

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    orchestrator = Orchestrator(context=get_context())

    if args.command == "list":
        for info in orchestrator.list_agents():
            print(f"- {info['name']}: {info['description']}")
        return 0

    if args.command == "describe":
        agent_registry.discover()
        if args.agent_name not in agent_registry.all():
            print(f"No existe el agente '{args.agent_name}'. Disponibles: {sorted(agent_registry.all())}")
            return 1
        agent = orchestrator._get_instance(args.agent_name)  # noqa: SLF001 — CLI interna, acceso intencional
        print(json.dumps(agent.describe(), indent=2, ensure_ascii=False))
        return 0

    if args.command == "run":
        kwargs = _parse_kwargs(args.kwargs)
        result = orchestrator.run(args.agent_name, args.action, **kwargs)
        _print_result(result)
        return 0 if result.success else 1

    if args.command == "ask":
        result = orchestrator.dispatch(args.query)
        _print_result(result)
        return 0 if result.success else 1

    if args.command == "tools":
        from agents.tools.registry import tool_registry
        # Las herramientas se registran al importarse; importar los agentes
        # (que a su vez importan sus herramientas) las descubre todas.
        agent_registry.discover()
        for name in sorted(tool_registry.all()):
            print(f"- {name}")
        return 0

    parser.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
