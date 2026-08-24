"""
agents.loop — Bucle de agente propio en Python sobre LiteLLM.

Arquitectura:
  - REPL loop (outer): lee entrada del usuario, imprime respuesta, repite.
  - Agent loop (inner): dado un prompt, llama al modelo con tool calls,
    ejecuta tools (via security gateway) y repite hasta que el modelo
    diga que ha terminado (sin más tool calls).

Tools se auto-descubren de agents/tools/ (via tool_registry) y de
agents/agents/ (como wrappers). El catálogo es cerrado y por agente
(ARNES-011 gateway en cada ejecución).

LiteLLM como proveedor: abstrae Ollama, Groq, OpenAI y otros bajo la
misma API. No reescribimos la capa de proveedor — usamos litellm.completion()
directamente (capítulo 03 del tutorial cubierto).
"""

from __future__ import annotations

import importlib
import inspect
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import litellm

from agents.security import (
    ToolCall,
    approve_tool_call,
    register_default_policies,
)

# ---------------------------------------------------------------------------
# Tool discovery: escanea agents/tools/ (via tool_registry) y agents/agents/
# ---------------------------------------------------------------------------

_TOOLS_DIR = Path(__file__).parent / "tools"
_AGENTS_DIR = Path(__file__).parent / "agents"
_discovered_tools: dict[str, dict[str, Any]] = {}


def _discover_tools() -> dict[str, dict[str, Any]]:
    """
    Auto-descubre herramientas de dos fuentes:
    1. agents/tools/*.py via tool_registry (las tools registradas con @register_tool)
    2. agents/agents/*.py (los 26 agentes como wrappers)

    Cada tool descubierta declara:
      - name, description, input_schema, execute
    """
    global _discovered_tools
    if _discovered_tools:
        return _discovered_tools

    # --- Fuente 1: tools del tool_registry ---
    _discover_from_registry()

    # --- Fuente 2: agentes como tools ---
    _discover_agent_tools()

    return _discovered_tools


def _discover_from_registry() -> None:
    """
    Importa todos los *_tool.py de agents/tools/ y descubre tools de dos
    formas: las registradas con @register_tool en tool_registry, y las
    clases dataclass que existen en los módulos pero no se registraron.
    """
    from agents.tools.registry import tool_registry

    loaded_mods: dict[str, Any] = {}
    for py_file in sorted(_TOOLS_DIR.glob("*_tool.py")):
        mod_name = f"agents.tools.{py_file.stem}"
        try:
            loaded_mods[mod_name] = importlib.import_module(mod_name)
        except Exception:
            continue

    # --- Fase 1: tools del tool_registry (@register_tool) ---
    for name, tool_cls in tool_registry.all().items():
        if name in _discovered_tools:
            continue
        _discovered_tools[name] = _wrap_tool_class(tool_cls, name)

    # --- Fase 2: clases no registradas pero con interfaz de tool ---
    # Muchas tools son dataclasses sin @register_tool (filesystem, code_analysis, etc.)
    # Las detectamos por tener al menos un método público y un docstring.
    for mod_name, mod in loaded_mods.items():
        for attr_name in dir(mod):
            if attr_name.startswith("_"):
                continue
            attr = getattr(mod, attr_name)
            if not inspect.isclass(attr):
                continue
            # Saltar si ya está registrada
            if any(attr is v for v in tool_registry.all().values()):
                continue
            # Detectar si parece una tool: tiene __init__ con params y métodos públicos
            try:
                has_methods = any(
                    callable(getattr(attr, m))
                    for m in dir(attr)
                    if not m.startswith("_") and m not in ("name", "description")
                )
            except (ValueError, TypeError):
                continue
            if not has_methods:
                continue

            # Derivar nombre: "FilesystemTool" → "filesystem", "GitTool" → "git"
            base_name = attr_name.replace("Tool", "").lower()
            if base_name in _discovered_tools:
                continue
            _discovered_tools[base_name] = _wrap_tool_class(attr, base_name)

    # --- Fase 3: herramientas basadas en funciones (process_tool.run_command, etc.) ---
    _discover_function_tools(loaded_mods)


def _discover_function_tools(loaded_mods: dict[str, Any]) -> None:
    """
    Detecta módulos con funciones principales tipo tool (process_tool.run_command).
    Si el módulo tiene una función 'run_*' o similar y no hay tool con ese nombre,
    la registra como tool.
    """
    # Mapeo explícito: nombre_en_policy → (módulo, función)
    FUNCTION_TOOL_MAP = {
        "process": ("agents.tools.process_tool", "run_command"),
    }

    for tool_name, (mod_path, func_name) in FUNCTION_TOOL_MAP.items():
        if tool_name in _discovered_tools:
            continue
        try:
            mod = importlib.import_module(mod_path)
        except Exception:
            continue
        func = getattr(mod, func_name, None)
        if func is None or not callable(func):
            continue

        _discovered_tools[tool_name] = {
            "name": tool_name,
            "description": (func.__doc__ or f"Tool: {tool_name}").strip().split("\n")[0],
            "input_schema": {"type": "object", "properties": {"args": {"type": "array"}}},
            "execute": func,
        }


def _wrap_tool_class(tool_cls: Any, name: str) -> dict[str, Any]:
    """Envuelve una clase tool en el formato {name, description, input_schema, execute}."""
    description = ""
    if hasattr(tool_cls, "description") and tool_cls.description:
        description = tool_cls.description
    elif hasattr(tool_cls, "__doc__") and tool_cls.__doc__:
        description = tool_cls.__doc__.strip().split("\n")[0]

    input_schema: dict[str, Any] = {"type": "object", "properties": {}}
    if inspect.isclass(tool_cls):
        try:
            sig = inspect.signature(tool_cls)
            props: dict[str, Any] = {}
            required: list[str] = []
            for pname, param in sig.parameters.items():
                if pname == "self":
                    continue
                annotation = param.annotation
                json_type = "string"
                if annotation in (int, float):
                    json_type = "number"
                elif annotation is bool:
                    json_type = "boolean"
                elif annotation is list:
                    json_type = "array"
                props[pname] = {"type": json_type}
                if param.default is inspect.Parameter.empty:
                    required.append(pname)
            if props:
                input_schema = {"type": "object", "properties": props}
                if required:
                    input_schema["required"] = required
        except (ValueError, TypeError):
            pass

    def execute(**kwargs: Any) -> Any:
        try:
            instance = tool_cls(**kwargs) if kwargs else tool_cls()
        except TypeError:
            instance = tool_cls()
        if hasattr(instance, "__dict__"):
            return {"tool": name, "instance": str(instance)}
        return {"tool": name}

    return {
        "name": name,
        "description": description or f"Tool: {name}",
        "input_schema": input_schema,
        "execute": execute,
    }


def _discover_agent_tools() -> None:
    """
    Registra los agentes Python de agents/agents/ como tools.
    Cada agente se expone como tool con su nombre y una execute que llama
    a agent.run(action, **kwargs).
    """
    for py_file in sorted(_AGENTS_DIR.glob("*_agent.py")):
        if py_file.name.startswith("_"):
            continue
        mod_name = f"agents.agents.{py_file.stem}"
        try:
            mod = importlib.import_module(mod_name)
        except Exception:
            continue

        for attr_name in dir(mod):
            attr = getattr(mod, attr_name)
            if (
                inspect.isclass(attr)
                and hasattr(attr, "name")
                and hasattr(attr, "actions")
                and attr_name != "BaseAgent"
            ):
                agent_name = getattr(attr, "name", py_file.stem.replace("_agent", ""))
                if agent_name in _discovered_tools:
                    continue

                def _make_execute(cls: type) -> Any:
                    def execute(**kwargs: Any) -> dict[str, Any]:
                        instance = cls()
                        action = kwargs.pop("action", None)
                        if action is None:
                            return {"success": False, "message": "Falta 'action'"}
                        result = instance.run(action, **kwargs)
                        return {
                            "success": result.success,
                            "message": result.message,
                            "data": result.data,
                        }
                    return execute

                _discovered_tools[agent_name] = {
                    "name": agent_name,
                    "description": getattr(attr, "description", f"Agente {agent_name}"),
                    "input_schema": {
                        "type": "object",
                        "properties": {
                            "action": {
                                "type": "string",
                                "description": "Acción del agente a ejecutar",
                            },
                        },
                        "required": ["action"],
                    },
                    "execute": _make_execute(attr),
                }


# ---------------------------------------------------------------------------
# Closed tool catalog per agent type
# ---------------------------------------------------------------------------

AGENT_TOOL_CATALOGS: dict[str, set[str]] = {}


def _build_catalogs() -> dict[str, set[str]]:
    """
    Construye el catálogo cerrado por tipo de agente.
    Usa las políticas de seguridad existentes (ARNES-011) como fuente
    de verdad para qué tools puede usar cada agente.

    El mapping: policy.allowed_tools contiene nombres genéricos como "git",
    "process", "filesystem". El catálogo se construye intersectando esos
    nombres con las tools realmente descubiertas, más el propio agente.
    """
    global AGENT_TOOL_CATALOGS
    if AGENT_TOOL_CATALOGS:
        return AGENT_TOOL_CATALOGS

    register_default_policies()

    from agents.security import AGENT_POLICIES

    for agent_name, policy in AGENT_POLICIES.items():
        allowed: set[str] = set()
        for tool_name in _discovered_tools:
            # Incluir si el nombre coincide con un allowed_tools del policy
            if tool_name in policy.allowed_tools:
                allowed.add(tool_name)
        # El propio agente siempre puede invocarse a sí mismo
        if agent_name in _discovered_tools:
            allowed.add(agent_name)
        AGENT_TOOL_CATALOGS[agent_name] = allowed

    return AGENT_TOOL_CATALOGS


def get_tool_catalog(agent_type: str) -> set[str]:
    """Devuelve el catálogo cerrado de tools para un tipo de agente."""
    _discover_tools()
    _build_catalogs()
    return set(AGENT_TOOL_CATALOGS.get(agent_type, set()))


# ---------------------------------------------------------------------------
# Tool execution via security gateway (ARNES-011)
# ---------------------------------------------------------------------------

def execute_tool(agent_type: str, tool_name: str, args: dict[str, Any]) -> dict[str, Any]:
    """
    Ejecuta una tool pasando por el gateway de seguridad.
    Ninguna tool se ejecuta sin pasar por approve_tool_call().
    """
    _discover_tools()
    _build_catalogs()

    # Verificar catálogo cerrado
    catalog = AGENT_TOOL_CATALOGS.get(agent_type, set())
    if tool_name not in catalog:
        return {
            "error": f"Tool '{tool_name}' no está en el catálogo de '{agent_type}': {sorted(catalog)}",
            "blocked": True,
            "reason": "catalog",
        }

    # Gateway de seguridad
    call = ToolCall(agent=agent_type, tool=tool_name, args=args)
    approval = approve_tool_call(call)

    if not approval.approved:
        return {
            "error": f"Gateway bloqueó '{tool_name}': {approval.reason}",
            "blocked": True,
            "reason": approval.reason,
        }

    # Ejecutar
    tool_def = _discovered_tools.get(tool_name)
    if tool_def is None:
        return {"error": f"Tool '{tool_name}' no encontrada tras aprobación", "blocked": False}

    try:
        result = tool_def["execute"](**args)
        return {"result": result, "blocked": False}
    except Exception as exc:
        return {"error": f"Excepción en '{tool_name}': {exc}", "blocked": False}


# ---------------------------------------------------------------------------
# LLM interaction via LiteLLM
# ---------------------------------------------------------------------------

def _build_tools_for_model(agent_type: str) -> list[dict[str, Any]]:
    """
    Convierte el catálogo de tools al formato OpenAI function calling
    que LiteLLM entiende.
    """
    _discover_tools()
    _build_catalogs()

    catalog = AGENT_TOOL_CATALOGS.get(agent_type, set())
    model_tools = []

    for tool_name in sorted(catalog):
        tool_def = _discovered_tools.get(tool_name)
        if tool_def is None:
            continue

        schema = tool_def.get("input_schema", {"type": "object", "properties": {}})
        model_tools.append({
            "type": "function",
            "function": {
                "name": tool_name,
                "description": tool_def.get("description", ""),
                "parameters": schema,
            },
        })

    return model_tools


def _call_llm(
    messages: list[dict[str, Any]],
    model: str,
    tools: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Llamada a LiteLLM con soporte para tool calling."""
    kwargs: dict[str, Any] = {
        "model": model,
        "messages": messages,
    }
    if tools:
        kwargs["tools"] = tools
    response = litellm.completion(**kwargs)
    return response.choices[0].message.model_dump()


# ---------------------------------------------------------------------------
# Agent inner loop
# ---------------------------------------------------------------------------

@dataclass
class LoopResult:
    """Resultado de una ejecución del agente loop."""
    success: bool
    messages: list[dict[str, Any]]
    tool_calls: list[dict[str, Any]]
    final_response: str


def agent_loop(
    prompt: str,
    agent_type: str,
    model: str = "ollama/llama3.2",
    max_iterations: int = 10,
    system_prompt: str | None = None,
) -> LoopResult:
    """
    Bucle interno del agente: evalúa con llamadas al modelo y ejecución
    de tools hasta que el modelo diga que ha terminado.

    Pasos:
    1. Envía prompt + tools al modelo
    2. Si el modelo devuelve tool calls → ejecutarlas (via gateway)
    3. Añadir resultado al historial y repetir
    4. Si no hay tool calls → el modelo terminó, devolver respuesta
    """
    _discover_tools()
    _build_catalogs()

    tools = _build_tools_for_model(agent_type)

    messages: list[dict[str, Any]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    all_tool_calls: list[dict[str, Any]] = []

    for iteration in range(max_iterations):
        response = _call_llm(messages, model, tools if tools else None)

        # Si no hay tool calls, el modelo terminó
        if not response.get("tool_calls"):
            messages.append(response)
            return LoopResult(
                success=True,
                messages=messages,
                tool_calls=all_tool_calls,
                final_response=response.get("content", ""),
            )

        # Ejecutar cada tool call
        messages.append(response)

        for tc in response["tool_calls"]:
            func_name = tc["function"]["name"]
            try:
                func_args = json.loads(tc["function"]["arguments"])
            except json.JSONDecodeError:
                func_args = {}

            result = execute_tool(agent_type, func_name, func_args)
            all_tool_calls.append({
                "tool": func_name,
                "args": func_args,
                "result": result,
                "iteration": iteration,
            })

            # Añadir resultado al historial para el modelo
            tool_result_content = json.dumps(result, ensure_ascii=False, default=str)
            messages.append({
                "role": "tool",
                "tool_call_id": tc.get("id", f"call_{iteration}"),
                "content": tool_result_content,
            })

    return LoopResult(
        success=False,
        messages=messages,
        tool_calls=all_tool_calls,
        final_response="Max iterations reached",
    )


# ---------------------------------------------------------------------------
# REPL loop (outer)
# ---------------------------------------------------------------------------

def repl_loop(
    agent_type: str = "implementer",
    model: str = "ollama/llama3.2",
    system_prompt: str | None = None,
) -> None:
    """
    REPL loop: lee entrada del usuario, evalúa con el agente loop,
    imprime la respuesta, repite. Ctrl+C para salir.
    """
    print(f"=== ClimaSafeAI Agent Loop ({agent_type}) ===")
    print(f"Modelo: {model}")
    print(f"Tools disponibles: {sorted(get_tool_catalog(agent_type))}")
    print("Escribe tu pregunta o 'quit' para salir.\n")

    while True:
        try:
            user_input = input(">>> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n¡Hasta luego!")
            break

        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit", "q"):
            print("¡Hasta luego!")
            break

        result = agent_loop(
            prompt=user_input,
            agent_type=agent_type,
            model=model,
            system_prompt=system_prompt,
        )

        print(f"\n[Respuesta] {result.final_response}")
        if result.tool_calls:
            print(f"[Tools ejecutadas: {len(result.tool_calls)}]")
            for tc in result.tool_calls:
                print(f"  → {tc['tool']}({json.dumps(tc['args'], ensure_ascii=False)[:100]})")
                if tc["result"].get("blocked"):
                    print(f"    ✗ BLOQUEADO: {tc['result'].get('error', tc['result'].get('reason'))}")
                else:
                    print(f"    ✓ OK")
        print()
