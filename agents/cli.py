"""
agents.cli — CLI para usar el sistema de agentes desde línea de comandos.

Uso:
    uv run python -m agents list
    uv run python -m agents describe git
    uv run python -m agents run git suggest_commit_message
    uv run python -m agents run data eda_report --filename dataset.csv --target-col target
    uv run python -m agents ask "genera el changelog desde el último tag"
    uv run python -m agents scout
    uv run python -m agents scout --dry-run
    uv run python -m agents scout --query transformers
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

    pipeline_p = subparsers.add_parser("pipeline", help="Ejecuta un pipeline predefinido (develop|fix|release|cycle|analyze|data).")
    pipeline_p.add_argument("name", choices=["develop", "fix", "release", "cycle", "analyze", "data"])
    pipeline_p.add_argument("params", nargs=argparse.REMAINDER, help="--version 1.0.0 --filename data.csv")

    doctor_p = subparsers.add_parser("doctor", help="Diagnóstico completo del proyecto: entorno, código, tests, datos.")
    doctor_p.add_argument("--fix", action="store_true", help="Intenta corregir problemas automáticamente (usa auto_fix pipeline)")

    plan_p = subparsers.add_parser(
        "plan",
        help="Describe un encargo: el agente 'plan' lo descompone, pregunta lo que falte y delega. "
             "(atajo de `run plan intake --brief ...`; responde/ejecuta con `run plan answer/execute`)",
    )
    plan_p.add_argument("brief", help="El encargo en lenguaje natural (un paso por línea o separado por ';')")

    audit_p = subparsers.add_parser("audit", help="Audita al equipo de agentes con el log de ejecuciones.")
    audit_p.add_argument(
        "what", nargs="?", default="report", choices=["report", "failures", "suggest"],
        help="report (uso y tasas de éxito) | failures (fallos recientes) | suggest (mejoras propuestas)",
    )

    subparsers.add_parser("tools", help="Lista las herramientas registradas.")

    scout_p = subparsers.add_parser("scout", help="Busca papers en arXiv/OpenAlex y los clasifica.")
    scout_p.add_argument("--dry-run", action="store_true", help="Solo mostrar resultados, no guardar archivos")
    scout_p.add_argument("--query", action="append", help="Filtrar por substring en query o target (puede repetirse)")

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

    if args.command == "pipeline":
        from agents.gstack.pipelines import run_pipeline
        pipe_kwargs = _parse_kwargs(args.params)
        result = run_pipeline(args.name, **pipe_kwargs)
        print(result.summary)
        return 0 if result.success else 1

    if args.command == "doctor":
        if args.fix:
            from agents.gstack.pipelines import auto_fix
            result = auto_fix(auto_commit=True)
        else:
            from agents.gstack.pipelines import auto_analyze
            result = auto_analyze()
        print(result.summary)
        return 0 if result.success else 1

    if args.command == "plan":
        result = orchestrator.run("plan", "intake", brief=args.brief)
        _print_result(result)
        for question in result.needs:
            print(f"  ? {question}")
        return 0 if result.success else 1

    if args.command == "audit":
        action = {"report": "report", "failures": "failures", "suggest": "suggest_improvements"}[args.what]
        result = orchestrator.run("audit", action)
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

    if args.command == "scout":
        from agents.paper_scout import scout_run, scout_summary
        result = scout_run(dry_run=args.dry_run, queries=args.query)
        print(scout_summary(result))
        return 0 if not result.errors else 1

    parser.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
