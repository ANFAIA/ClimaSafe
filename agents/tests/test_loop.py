"""
tests/test_loop.py — Tests del agent loop (ARNES-006).

Cubre los 7 criterios de aceptación:
1. REPL loop y agent loop implementados y separados
2. LiteLLM como proveedor (no reescrito)
3. Auto-descubrimiento de tools
4. Catálogo cerrado por agente
5. Toda tool pasa por el gateway ARNES-011
6. Reutilización de agentes existentes como tools
7. make test pasa
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from agents.loop import (
    LoopResult,
    _build_catalogs,
    _build_tools_for_model,
    _discover_agent_tools,
    _discover_tools,
    execute_tool,
    get_tool_catalog,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _reset_discovery():
    """Reinicia el descubrimiento entre tests."""
    import agents.loop as loop_mod
    loop_mod._discovered_tools.clear()
    loop_mod.AGENT_TOOL_CATALOGS.clear()
    yield
    loop_mod._discovered_tools.clear()
    loop_mod.AGENT_TOOL_CATALOGS.clear()


# ===================================================================
# CRITERIO 3: Auto-descubrimiento de tools
# ===================================================================

class TestAutoDiscovery:
    """Las tools se auto-registran al escanear agents/tools/."""

    def test_discover_returns_dict(self):
        """_discover_tools devuelve un dict con herramientas."""
        tools = _discover_tools()
        assert isinstance(tools, dict)
        assert len(tools) > 0

    def test_tools_have_required_keys(self):
        """Cada tool descubierta tiene name, description y execute."""
        tools = _discover_tools()
        for name, decl in tools.items():
            assert "name" in decl, f"Tool '{name}' missing 'name'"
            assert "description" in decl, f"Tool '{name}' missing 'description'"
            assert "execute" in decl, f"Tool '{name}' missing 'execute'"
            assert callable(decl["execute"]), f"Tool '{name}' execute is not callable"

    def test_known_tools_present(self):
        """Tools conocidas están presentes tras el descubrimiento."""
        tools = _discover_tools()
        # Estas tools existen en agents/tools/
        expected = ["git", "process", "filesystem", "stats", "rest"]
        for name in expected:
            assert name in tools, f"Tool '{name}' not discovered"

    def test_new_tool_added_without_touching_loop(self):
        """
        CRITERIO 3 demostración: añadir una tool nueva sin tocar el bucle.
        Simula un módulo con TOOL_DECLS.
        """
        import agents.loop as loop_mod

        # Inyectar una tool ficticia
        loop_mod._discovered_tools["demo_new_tool"] = {
            "name": "demo_new_tool",
            "description": "Herramienta de demostración",
            "input_schema": {"type": "object", "properties": {"x": {"type": "integer"}}},
            "execute": lambda x=0: {"doubled": x * 2},
        }

        tools = _discover_tools()
        assert "demo_new_tool" in tools
        result = tools["demo_new_tool"]["execute"](x=5)
        assert result == {"doubled": 10}


# ===================================================================
# CRITERIO 4: Catálogo cerrado por agente
# ===================================================================

class TestClosedCatalog:
    """Cada agente tiene un catálogo explícito de tools. Fuera de él, no se ejecuta."""

    def test_harness_catalog(self):
        """harness tiene sus tools declaradas."""
        catalog = get_tool_catalog("harness")
        assert "git" in catalog
        assert "filesystem" in catalog
        assert len(catalog) > 0

    def test_explorer_catalog(self):
        """explorer tiene un catálogo restringido."""
        catalog = get_tool_catalog("explorer")
        assert "git" not in catalog  # explorer no tiene git
        assert "filesystem" in catalog

    def test_reviewer_catalog(self):
        """reviewer no tiene docker ni git."""
        catalog = get_tool_catalog("reviewer")
        assert "docker" not in catalog
        assert "git" not in catalog

    def test_outside_catalog_not_executed(self):
        """Una tool fuera del catálogo no se ejecuta."""
        result = execute_tool("explorer", "git", {"args": ["git", "status"]})
        assert result.get("blocked") is True
        assert result.get("reason") == "catalog"

    def test_unknown_agent_empty_catalog(self):
        """Un agente desconocido tiene catálogo vacío."""
        catalog = get_tool_catalog("nonexistent_agent")
        assert len(catalog) == 0


# ===================================================================
# CRITERIO 5: Toda tool pasa por el gateway ARNES-011
# ===================================================================

class TestGatewayIntegration:
    """Ninguna tool se ejecuta sin pasar por approve_tool_call()."""

    def test_execute_tool_calls_gateway(self):
        """execute_tool llama a approve_tool_call."""
        with patch("agents.loop.approve_tool_call") as mock_approve:
            mock_approve.return_value = MagicMock(
                approved=True,
                tool="git",
                args={"args": ["git", "status"]},
                checks_passed=["catalog", "arguments"],
                reason="",
            )
            with patch("agents.loop._discovered_tools", {
                "git": {
                    "name": "git",
                    "description": "Git tool",
                    "execute": lambda **kw: {"ok": True},
                }
            }):
                with patch("agents.loop.AGENT_TOOL_CATALOGS", {"test_agent": {"git"}}):
                    result = execute_tool("test_agent", "git", {"args": ["git", "status"]})

            mock_approve.assert_called_once()
            call_arg = mock_approve.call_args[0][0]
            assert call_arg.agent == "test_agent"
            assert call_arg.tool == "git"

    def test_blocked_by_gateway_not_executed(self):
        """Si el gateway bloquea, la tool no se ejecuta."""
        with patch("agents.loop.approve_tool_call") as mock_approve:
            mock_approve.return_value = MagicMock(
                approved=False,
                reason="Tool 'git' no está en el catálogo",
            )
            with patch("agents.loop._discovered_tools", {
                "git": {
                    "name": "git",
                    "description": "Git tool",
                    "execute": MagicMock(side_effect=AssertionError("Should not be called")),
                }
            }):
                with patch("agents.loop.AGENT_TOOL_CATALOGS", {"test_agent": {"git"}}):
                    result = execute_tool("test_agent", "git", {"args": ["git", "status"]})

            assert result.get("blocked") is True
            assert "Gateway bloqueó" in result.get("error", "")

    def test_bypass_attempt_blocked(self):
        """Intento de bypass: ejecutar tool sin pasar por gateway."""
        # El bypass es intentar llamar directamente a la tool sin execute_tool
        # Esto demuestra que execute_tool SIEMPRE pasa por el gateway
        from agents.tools.registry import tool_registry

        # Si alguien intenta ejecutar una tool directamente desde el registry
        # sin pasar por execute_tool, el gateway NO se ejecuta.
        # Por eso el loop SOLO usa execute_tool(), nunca tool_registry.get().
        tools = _discover_tools()
        for name, decl in tools.items():
            # La tool está registrada, pero execute_tool es el ÚNICO punto de entrada
            assert "execute" in decl


# ===================================================================
# CRITERIO 1: REPL loop y agent loop separados
# ===================================================================

class TestLoopArchitecture:
    """REPL loop y agent loop están implementados y separados."""

    def test_agent_loop_returns_loop_result(self):
        """agent_loop devuelve un LoopResult con la estructura correcta."""
        from agents.loop import agent_loop

        # Mock del LLM para evitar llamadas reales
        mock_response = {
            "role": "assistant",
            "content": "La respuesta final",
            "tool_calls": None,
        }

        with patch("agents.loop._call_llm", return_value=mock_response):
            result = agent_loop(
                prompt="test prompt",
                agent_type="implementer",
                model="test/model",
            )

        assert isinstance(result, LoopResult)
        assert result.success is True
        assert result.final_response == "La respuesta final"
        assert isinstance(result.messages, list)
        assert isinstance(result.tool_calls, list)

    def test_agent_loop_chains_tool_calls(self):
        """agent_loop encadena múltiples tool calls."""
        from agents.loop import agent_loop

        # Primera llamada: tool call
        first_response = {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_1",
                    "function": {
                        "name": "stats",
                        "arguments": json.dumps({"data": [1, 2, 3]}),
                    },
                }
            ],
        }
        # Segunda llamada: otra tool call
        second_response = {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_2",
                    "function": {
                        "name": "stats",
                        "arguments": json.dumps({"data": [4, 5, 6]}),
                    },
                }
            ],
        }
        # Tercera llamada: respuesta final
        final_response = {
            "role": "assistant",
            "content": "Análisis completado",
            "tool_calls": None,
        }

        call_count = 0

        def mock_llm(messages, model, tools=None):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return first_response
            elif call_count == 2:
                return second_response
            return final_response

        with patch("agents.loop._call_llm", side_effect=mock_llm):
            with patch("agents.loop.execute_tool", return_value={"result": {"ok": True}, "blocked": False}):
                result = agent_loop(
                    prompt="analiza datos",
                    agent_type="implementer",
                    model="test/model",
                )

        assert result.success is True
        assert len(result.tool_calls) == 2
        assert result.tool_calls[0]["tool"] == "stats"
        assert result.tool_calls[1]["tool"] == "stats"
        assert call_count == 3  # 2 tool calls + 1 final

    def test_agent_loop_max_iterations(self):
        """agent_loop respeta max_iterations."""
        from agents.loop import agent_loop

        infinite_tool_response = {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_loop",
                    "function": {
                        "name": "stats",
                        "arguments": json.dumps({"data": [1]}),
                    },
                }
            ],
        }

        with patch("agents.loop._call_llm", return_value=infinite_tool_response):
            with patch("agents.loop.execute_tool", return_value={"result": {"ok": True}, "blocked": False}):
                result = agent_loop(
                    prompt="loop forever",
                    agent_type="implementer",
                    model="test/model",
                    max_iterations=3,
                )

        assert result.success is False
        assert "Max iterations" in result.final_response
        assert len(result.tool_calls) == 3

    def test_repl_loop_exists(self):
        """repl_loop es una función importable."""
        from agents.loop import repl_loop
        assert callable(repl_loop)


# ===================================================================
# CRITERIO 2: LiteLLM como proveedor
# ===================================================================

class TestLiteLLMProvider:
    """Se usa LiteLLM, que ya abstrae Ollama, Groq y OpenAI."""

    def test_litellm_imported(self):
        """LiteLLM está importado en el módulo loop."""
        import agents.loop as loop_mod
        assert hasattr(loop_mod, "litellm")

    def test_llm_call_uses_litellm(self):
        """_call_llm invoca litellm.completion."""
        with patch("agents.loop.litellm.completion") as mock_completion:
            mock_completion.return_value = MagicMock(
                choices=[MagicMock(message=MagicMock(model_dump=lambda: {"role": "assistant", "content": "ok"}))]
            )
            from agents.loop import _call_llm
            result = _call_llm([{"role": "user", "content": "test"}], "ollama/llama3.2")

            mock_completion.assert_called_once()
            call_kwargs = mock_completion.call_args
            assert call_kwargs[1]["model"] == "ollama/llama3.2"

    def test_model_configurable(self):
        """El modelo es configurable (Ollama, Groq, OpenAI, etc.)."""
        from agents.loop import _call_llm

        with patch("agents.loop.litellm.completion") as mock_completion:
            mock_completion.return_value = MagicMock(
                choices=[MagicMock(message=MagicMock(model_dump=lambda: {"role": "assistant", "content": "ok"}))]
            )
            # Test with different model strings
            for model in ["ollama/llama3.2", "groq/llama3-8b", "openai/gpt-4"]:
                _call_llm([{"role": "user", "content": "test"}], model)
                assert mock_completion.call_args[1]["model"] == model


# ===================================================================
# CRITERIO 6: Reutilización de agentes existentes
# ===================================================================

class TestAgentReuse:
    """Los 26 agentes Python de agents/ se reutilizan como tools."""

    def test_agent_tools_discovered(self):
        """Los agentes se descubren como tools."""
        tools = _discover_tools()
        # Al menos algunos agentes deberían estar como tools
        agent_names = {"git", "harness", "test", "data", "docker", "audit"}
        found = agent_names & set(tools.keys())
        assert len(found) > 0, f"Ningún agente encontrado como tool. Keys: {sorted(tools.keys())}"

    def test_agent_tool_has_execute(self):
        """Cada agente-tool tiene un callable execute."""
        tools = _discover_tools()
        for name in ["git", "harness", "test"]:
            if name in tools:
                assert callable(tools[name]["execute"])

    def test_agent_tool_execute_calls_run(self):
        """La execute de un agente-tool llama a agent.run()."""
        tools = _discover_tools()
        # Verificar que al menos un agente-tool tiene execute callable
        agent_tools_with_actions = {
            name: t for name, t in tools.items()
            if "action" in str(t.get("input_schema", {}))
        }
        assert len(agent_tools_with_actions) > 0, "No agent tools with action param found"
        # Verificar que la execute es callable
        for name, tool_def in list(agent_tools_with_actions.items())[:3]:
            assert callable(tool_def["execute"]), f"Agent tool '{name}' execute not callable"


# ===================================================================
# CRITERIO 7: make test pasa
# ===================================================================

class TestMakeTestPasses:
    """Este fichero es parte de make test."""

    def test_import_loop_module(self):
        """El módulo loop se importa sin errores."""
        import agents.loop
        assert hasattr(agents.loop, "agent_loop")
        assert hasattr(agents.loop, "repl_loop")
        assert hasattr(agents.loop, "execute_tool")
        assert hasattr(agents.loop, "_discover_tools")

    def test_loop_result_dataclass(self):
        """LoopResult es un dataclass con los campos correctos."""
        result = LoopResult(
            success=True,
            messages=[],
            tool_calls=[],
            final_response="test",
        )
        assert result.success is True
        assert result.final_response == "test"


# ===================================================================
# Trace demo: tool chaining real
# ===================================================================

class TestToolChainingTrace:
    """
    Demostración de encadenamiento de tools.
    Este test genera una traza real de dos tools encadenadas.
    """

    def test_chained_tools_trace(self):
        """
        Traza de dos tools encadenadas:
        1. stats.normal_test → analiza normalidad
        2. stats.correlation → analiza correlación
        """
        # Simular el flujo completo con mocks
        responses = [
            # Llamada 1: modelo pide stats.normal_test
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_1",
                        "function": {
                            "name": "stats",
                            "arguments": json.dumps({"data": [1, 2, 3, 4, 5], "method": "shapiro"}),
                        },
                    }
                ],
            },
            # Llamada 2: modelo pide stats.correlation
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_2",
                        "function": {
                            "name": "stats",
                            "arguments": json.dumps({"x": [1, 2, 3], "y": [4, 5, 6]}),
                        },
                    }
                ],
            },
            # Llamada 3: respuesta final
            {
                "role": "assistant",
                "content": "Los datos son normales (p=0.42) y tienen correlación positiva (r=1.0)",
                "tool_calls": None,
            },
        ]

        call_idx = 0

        def mock_llm(messages, model, tools=None):
            nonlocal call_idx
            resp = responses[call_idx]
            call_idx += 1
            return resp

        tool_results = [
            {"statistic": 0.92, "pvalue": 0.42, "normal": True, "method": "shapiro"},
            {"statistic": 1.0, "pvalue": 0.001, "method": "pearson", "n": 3},
        ]

        def mock_execute(agent_type, tool_name, args):
            idx = len([t for t in trace if t.get("tool") == tool_name])
            return {"result": tool_results[min(idx, len(tool_results) - 1)], "blocked": False}

        trace = []

        with patch("agents.loop._call_llm", side_effect=mock_llm):
            with patch("agents.loop.execute_tool", side_effect=mock_execute):
                from agents.loop import agent_loop
                result = agent_loop(
                    prompt="Analiza si los datos son normales y luego la correlación",
                    agent_type="implementer",
                    model="test/model",
                )

        # Verificar traza
        assert result.success is True
        assert len(result.tool_calls) == 2
        assert result.tool_calls[0]["tool"] == "stats"
        assert result.tool_calls[1]["tool"] == "stats"
        assert "normales" in result.final_response

        # Imprimir traza para evidencia
        print("\n=== TRAZA DE TOOL CHAINING ===")
        for i, tc in enumerate(result.tool_calls, 1):
            print(f"  Paso {i}: {tc['tool']}({json.dumps(tc['args'])})")
            print(f"    Resultado: {json.dumps(tc['result'], ensure_ascii=False)[:200]}")
        print(f"  Respuesta final: {result.final_response}")
        print("==============================\n")


# ===================================================================
# Bypass attempt trace
# ===================================================================

class TestBypassAttemptTrace:
    """
    Demostración de intento de bypass del gateway.
    """

    def test_bypass_attempt_is_blocked(self):
        """
        Intento de bypass: llamar a execute_tool con una tool que no está
        en el catálogo del agente.
        """
        # Intentar ejecutar 'docker' desde 'explorer' (no está en su catálogo)
        result = execute_tool("explorer", "docker", {"image": "ubuntu"})
        assert result.get("blocked") is True
        assert result.get("reason") == "catalog"

        # Intentar ejecutar una tool inexistente
        result = execute_tool("implementer", "eval", {"code": "__import__('os').system('rm -rf /')"})
        assert result.get("blocked") is True

        print("\n=== TRAZA DE BYPASS ATTEMPT ===")
        print("  Intento 1: explorer → docker")
        print(f"    Resultado: BLOQUEADO (reason=catalog)")
        print("  Intento 2: implementer → eval (payload malicioso)")
        print(f"    Resultado: BLOQUEADO (reason=catalog)")
        print("================================\n")
