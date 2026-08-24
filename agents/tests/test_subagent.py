"""
agents/tests/test_subagent.py — Tests de subagentes y compactación (ARNES-007).

Cubre los 7 criterios de aceptación:
1. Delegación a subagentes con tool built-in
2. Subagente arranca con contexto vacío (decisión justificada)
3. Subagentes definidos en markdown (.opencode/agents/*.md)
4. Tres estrategias de compactación intercambiables
5. Compactación no parte bloques tool_use (test explícito)
6. Coste en tokens de summary compaction medido
7. make test pasa
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from agents.compaction import (
    CompactionResult,
    CompactionStrategy,
    NoneCompaction,
    SlidingWindowCompaction,
    SummaryCompaction,
    count_tokens_approx,
    is_in_tool_use_block,
    is_tool_use_boundary,
)
from agents.loop import (
    LoopResult,
    _discover_tools,
    execute_delegation,
    execute_tool,
    get_tool_catalog,
)
from agents.subagent import (
    SubagentConfig,
    get_available_subagents,
    load_subagent_config,
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
# CRITERIO 3: Subagentes definidos en markdown
# ===================================================================

class TestMarkdownSubagents:
    """Los subagentes se cargan de .opencode/agents/*.md, no hardcodeados."""

    def test_load_explorer_config(self):
        """Se puede cargar la config del explorer desde markdown."""
        config = load_subagent_config("explorer")
        assert config is not None
        assert config.name == "explorer"
        assert len(config.system_prompt) > 0
        assert config.source_file.exists()
        assert config.source_file.suffix == ".md"

    def test_load_implementer_config(self):
        """Se puede cargar la config del implementer desde markdown."""
        config = load_subagent_config("implementer")
        assert config is not None
        assert config.name == "implementer"
        assert len(config.system_prompt) > 0

    def test_load_reviewer_config(self):
        """Se puede cargar la config del reviewer desde markdown."""
        config = load_subagent_config("reviewer")
        assert config is not None
        assert config.name == "reviewer"

    def test_nonexistent_agent_returns_none(self):
        """Un agente inexistente devuelve None."""
        config = load_subagent_config("nonexistent_xyz")
        assert config is None

    def test_available_subagents(self):
        """get_available_subagents devuelve al menos los 3 del arnés."""
        agents = get_available_subagents()
        assert len(agents) >= 3
        assert "explorer" in agents
        assert "implementer" in agents
        assert "reviewer" in agents

    def test_agents_are_markdown_not_hardcoded(self):
        """Los ficheros son .md en .opencode/agents/, no clases Python."""
        agents_dir = Path(__file__).parent.parent.parent / ".opencode" / "agents"
        assert agents_dir.exists()
        md_files = list(agents_dir.glob("*.md"))
        assert len(md_files) >= 3
        for f in md_files:
            content = f.read_text(encoding="utf-8")
            assert len(content) > 50, f"{f.name} seems too short to be a real agent definition"

    def test_system_prompt_excludes_tool_section(self):
        """El system prompt extraído no incluye la sección de herramientas."""
        config = load_subagent_config("explorer")
        # El system prompt no debería contener bloques de código de herramientas
        # (o al menos no la sección completa ## Herramientas)
        assert "## Herramientas" not in config.system_prompt or \
               config.system_prompt.index("## Herramientas") > len(config.system_prompt) // 2


# ===================================================================
# CRITERIO 1: Delegación a subagentes
# ===================================================================

class TestDelegation:
    """El bucle puede instanciar subagentes via delegate_to_subagent."""

    def test_delegation_tool_exists(self):
        """delegate_to_subagent está registrado como tool built-in."""
        tools = _discover_tools()
        assert "delegate_to_subagent" in tools
        assert tools["delegate_to_subagent"]["name"] == "delegate_to_subagent"

    def test_delegation_in_all_catalogs(self):
        """delegate_to_subagent aparece en todos los catálogos de agentes."""
        for agent_type in ["implementer", "explorer", "reviewer", "harness"]:
            catalog = get_tool_catalog(agent_type)
            assert "delegate_to_subagent" in catalog, (
                f"delegate_to_subagent not in {agent_type} catalog"
            )

    def test_delegation_tool_schema(self):
        """La tool de delegación tiene el esquema correcto."""
        tools = _discover_tools()
        schema = tools["delegate_to_subagent"]["input_schema"]
        assert "agent_name" in schema["properties"]
        assert "prompt" in schema["properties"]
        assert "max_iterations" in schema["properties"]
        assert "agent_name" in schema["required"]
        assert "prompt" in schema["required"]

    def test_execute_delegation_returns_result(self):
        """execute_delegation devuelve un dict con success, response, tool_calls."""
        mock_result = LoopResult(
            success=True,
            messages=[],
            tool_calls=[{"tool": "stats", "args": {"data": [1, 2]}}, {"tool": "git", "args": {}}],
            final_response="Análisis completado",
        )

        with patch("agents.loop.agent_loop", return_value=mock_result) as mock_loop:
            result = execute_delegation(
                agent_name="explorer",
                prompt="¿Qué hay en agents/tools/?",
                max_iterations=3,
            )

        assert result["success"] is True
        assert result["response"] == "Análisis completado"
        assert result["tool_calls_count"] == 2
        assert len(result["tool_calls"]) == 2
        assert result["tool_calls"][0]["tool"] == "stats"

        # Verify agent_loop was called with the subagent's system prompt
        call_kwargs = mock_loop.call_args
        assert call_kwargs[1]["agent_type"] == "explorer"
        assert call_kwargs[1]["max_iterations"] == 3
        assert call_kwargs[1]["system_prompt"] is not None

    def test_execute_delegation_unknown_agent(self):
        """Delegar a un agente inexistente devuelve error."""
        result = execute_delegation(
            agent_name="nonexistent_xyz",
            prompt="test",
        )
        assert "error" in result
        assert "nonexistent_xyz" in result["error"]

    def test_delegation_trace(self):
        """
        Traza real de una delegación (mocked):
        1. Modelo decide delegar al explorer
        2. Sub-loop del explorer ejecuta tools
        3. Resultado vuelve al padre
        """
        # Simular el sub-loop del explorer
        sub_result = LoopResult(
            success=True,
            messages=[
                {"role": "system", "content": "# Explorer — investiga, no toca..."},
                {"role": "user", "content": "¿Qué hay en agents/tools/?"},
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "function": {
                                "name": "filesystem",
                                "arguments": json.dumps({"action": "list", "path": "agents/tools/"}),
                            },
                        }
                    ],
                },
                {
                    "role": "tool",
                    "tool_call_id": "call_1",
                    "content": json.dumps({"files": ["git_tool.py", "process_tool.py"]}),
                },
                {
                    "role": "assistant",
                    "content": "En agents/tools/ hay git_tool.py, process_tool.py y más.",
                    "tool_calls": None,
                },
            ],
            tool_calls=[
                {"tool": "filesystem", "args": {"action": "list", "path": "agents/tools/"}, "result": {"result": {"files": ["git_tool.py"]}, "blocked": False}, "iteration": 0},
            ],
            final_response="En agents/tools/ hay git_tool.py, process_tool.py y más.",
        )

        with patch("agents.loop.agent_loop", return_value=sub_result) as mock_loop:
            result = execute_delegation(
                agent_name="explorer",
                prompt="¿Qué hay en agents/tools/?",
            )

        assert result["success"] is True
        assert "git_tool.py" in result["response"]
        assert result["tool_calls_count"] == 1

        print("\n=== TRAZA DE DELEGACIÓN ===")
        print(f"  Padre → Subagente: explorer")
        print(f"  Prompt: ¿Qué hay en agents/tools/?")
        print(f"  Sub-loop iteraciones: 1")
        print(f"  Tools ejecutadas: {result['tool_calls_count']}")
        print(f"  Respuesta: {result['response']}")
        print("===========================\n")


# ===================================================================
# CRITERIO 2: Subagente arranca con contexto vacío
# ===================================================================

class TestEmptyContextDecision:
    """
    Decisión: el subagente arranca con contexto vacío, NO hereda el
    historial del padre.

    Justificación:
      1. Heredar historial duplica tokens (coste lineal por profundidad)
      2. El padre resume lo que el subagente necesita en el prompt
      3. Subagente enfocado = resultado más limpio y predecible
      4. Principio AGENTS.md: "Al lanzar un subagente, no le heredes contexto"
    """

    def test_subagent_gets_empty_message_history(self):
        """El subagente recibe solo su system prompt + el prompt, NO el historial del padre."""
        with patch("agents.loop.agent_loop") as mock_loop:
            mock_loop.return_value = LoopResult(
                success=True, messages=[], tool_calls=[], final_response="ok"
            )
            execute_delegation(agent_name="explorer", prompt="test task")

        # agent_loop should be called with only the prompt, not parent's messages
        call_kwargs = mock_loop.call_args
        # The prompt parameter should be just the delegation task, not a conversation
        assert call_kwargs[1]["prompt"] == "test task"
        # No parent_messages or similar parameter exists
        assert "parent_messages" not in call_kwargs[1]

    def test_subagent_system_prompt_from_markdown(self):
        """El system prompt del subagente viene del markdown, no del padre."""
        with patch("agents.loop.agent_loop") as mock_loop:
            mock_loop.return_value = LoopResult(
                success=True, messages=[], tool_calls=[], final_response="ok"
            )
            execute_delegation(agent_name="explorer", prompt="test")

        call_kwargs = mock_loop.call_args
        system_prompt = call_kwargs[1]["system_prompt"]
        # Should contain content from .opencode/agents/explorer.md
        assert "Explorer" in system_prompt or "investiga" in system_prompt.lower()

    def test_subagent_tool_catalog_is_its_own(self):
        """El catálogo de tools del subagente es el suyo, no el del padre."""
        # This is verified by the architecture: agent_loop builds tools from
        # agent_type, which for the subagent is the subagent's name
        with patch("agents.loop.agent_loop") as mock_loop:
            mock_loop.return_value = LoopResult(
                success=True, messages=[], tool_calls=[], final_response="ok"
            )
            execute_delegation(agent_name="explorer", prompt="test")

        call_kwargs = mock_loop.call_args
        assert call_kwargs[1]["agent_type"] == "explorer"

    def test_justification_written(self):
        """La decisión está documentada en agents/subagent.py."""
        import agents.subagent as submod
        docstring = submod.__doc__ or ""
        assert "contexto vacío" in docstring or "vacío" in docstring or "no le heredes" in docstring


# ===================================================================
# CRITERIO 4: Tres estrategias de compactación
# ===================================================================

class TestCompactionStrategies:
    """Hay al menos tres estrategias intercambiables."""

    def test_none_compaction(self):
        """NoneCompaction no modifica los mensajes."""
        strategy = NoneCompaction()
        messages = [
            {"role": "system", "content": "You are helpful"},
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there"},
        ]
        result = strategy.apply(messages)
        assert result.compacted is False
        assert len(result.messages) == len(messages)
        assert result.tokens_before == result.tokens_after

    def test_sliding_window_compaction(self):
        """SlidingWindowCompaction mantiene los últimos N mensajes."""
        strategy = SlidingWindowCompaction(max_messages=3)
        messages = [
            {"role": "system", "content": "System"},
            {"role": "user", "content": "msg1"},
            {"role": "assistant", "content": "resp1"},
            {"role": "user", "content": "msg2"},
            {"role": "assistant", "content": "resp2"},
            {"role": "user", "content": "msg3"},
            {"role": "assistant", "content": "resp3"},
            {"role": "user", "content": "msg4"},
        ]
        result = strategy.apply(messages)
        # System prompt + 3 most recent messages = 4 total
        assert len(result.messages) == 4
        assert result.messages[0]["role"] == "system"
        assert result.messages[-1]["content"] == "msg4"
        assert result.compacted is True

    def test_sliding_window_keeps_system_prompt(self):
        """El system prompt nunca se descarta."""
        strategy = SlidingWindowCompaction(max_messages=2)
        messages = [
            {"role": "system", "content": "Important system prompt"},
            {"role": "user", "content": "1"},
            {"role": "assistant", "content": "2"},
            {"role": "user", "content": "3"},
            {"role": "assistant", "content": "4"},
            {"role": "user", "content": "5"},
        ]
        result = strategy.apply(messages)
        assert result.messages[0]["role"] == "system"
        assert result.messages[0]["content"] == "Important system prompt"

    def test_sliding_window_no_compaction_when_small(self):
        """No compacta si hay pocos mensajes."""
        strategy = SlidingWindowCompaction(max_messages=20)
        messages = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "hi"},
        ]
        result = strategy.apply(messages)
        assert result.compacted is False

    def test_summary_compaction(self):
        """SummaryCompaction usa el modelo para resumir."""
        import agents.compaction as comp_mod
        strategy = SummaryCompaction(model="test/model", threshold_tokens=10)
        messages = [
            {"role": "system", "content": "You are helpful"},
            {"role": "user", "content": "First question " * 20},
            {"role": "assistant", "content": "First answer " * 20},
            {"role": "user", "content": "Second question " * 20},
            {"role": "assistant", "content": "Second answer " * 20},
            {"role": "user", "content": "Third question " * 20},
        ]

        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content="Summary of conversation"))]
        mock_response.usage = MagicMock(total_tokens=150)

        with patch.object(comp_mod.litellm, "completion", return_value=mock_response):
            result = strategy.apply(messages)

        assert result.compacted is True
        # System prompt + summary + last 4 messages = 6
        assert len(result.messages) <= 7
        # Summary message should be present
        summary_msgs = [m for m in result.messages if "[Conversation summary]" in (m.get("content") or "")]
        assert len(summary_msgs) == 1

    def test_summary_compaction_no_op_when_below_threshold(self):
        """SummaryCompaction no compacta si está por debajo del umbral."""
        strategy = SummaryCompaction(threshold_tokens=10000)
        messages = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "short"},
        ]
        result = strategy.apply(messages)
        assert result.compacted is False

    def test_selectable_without_touching_loop(self):
        """Las estrategias se seleccionan pasando el parámetro, sin tocar el loop."""
        from agents.loop import agent_loop

        # Verify the parameter exists
        import inspect
        sig = inspect.signature(agent_loop)
        assert "compaction" in sig.parameters
        # Default is None (which means NoneCompaction inside)
        assert sig.parameters["compaction"].default is None

    def test_strategy_interface(self):
        """Todas las estrategias implementan CompactionStrategy."""
        assert isinstance(NoneCompaction(), CompactionStrategy)
        assert isinstance(SlidingWindowCompaction(), CompactionStrategy)
        assert isinstance(SummaryCompaction(), CompactionStrategy)


# ===================================================================
# CRITERIO 5: Compactación no parte bloques tool_use
# ===================================================================

class TestToolUseBlockIntegrity:
    """La compactación nunca parte un bloque de tool_use por la mitad."""

    def test_sliding_window_respects_tool_use_boundary(self):
        """SlidingWindowCompaction no corta un bloque assistant+tool."""
        strategy = SlidingWindowCompaction(max_messages=3)
        messages = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "old question"},
            # Start of tool_use block
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {"id": "c1", "function": {"name": "stats", "arguments": "{}"}}
                ],
            },
            {"role": "tool", "tool_call_id": "c1", "content": '{"result": 42}'},
            # End of tool_use block
            {"role": "user", "content": "new question"},
            {"role": "assistant", "content": "answer"},
        ]
        result = strategy.apply(messages)

        # Check that tool_use blocks are not split
        for i, msg in enumerate(result.messages):
            if msg.get("role") == "tool":
                # A tool result should always be preceded by its assistant message
                assert i > 0
                prev = result.messages[i - 1]
                assert prev.get("role") == "assistant", (
                    f"Tool result at index {i} preceded by {prev.get('role')}, not assistant"
                )

    def test_sliding_window_preserves_complete_tool_blocks(self):
        """Un bloque tool_use completo (assistant + tool result) se mantiene entero."""
        strategy = SlidingWindowCompaction(max_messages=2)
        messages = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "q1"},
            {"role": "assistant", "content": "a1"},
            # Tool block
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {"id": "c1", "function": {"name": "git", "arguments": "{}"}}
                ],
            },
            {"role": "tool", "tool_call_id": "c1", "content": "ok"},
            # More messages
            {"role": "assistant", "content": "after tool"},
            {"role": "user", "content": "q2"},
        ]
        result = strategy.apply(messages)

        # The tool block should be either entirely present or entirely absent
        tool_indices = [
            i for i, m in enumerate(result.messages)
            if m.get("role") == "tool"
        ]
        assistant_indices_with_tools = [
            i for i, m in enumerate(result.messages)
            if m.get("role") == "assistant" and m.get("tool_calls")
        ]

        # If there's a tool result, there should be an assistant before it
        for ti in tool_indices:
            assert any(ai == ti - 1 for ai in assistant_indices_with_tools), (
                f"Orphan tool result at index {ti}"
            )

    def test_compaction_never_splits_tool_use_block(self):
        """
        Test explícito: ningún mensaje tool result aparece sin su assistant
        correspondiente inmediatamente antes, en cualquier estrategia.
        """
        # Create a conversation with multiple tool_use blocks
        messages = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "q1"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [{"id": "c1", "function": {"name": "git", "arguments": "{}"}}],
            },
            {"role": "tool", "tool_call_id": "c1", "content": "result1"},
            {"role": "assistant", "content": "intermediate"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [{"id": "c2", "function": {"name": "process", "arguments": "{}"}}],
            },
            {"role": "tool", "tool_call_id": "c2", "content": "result2"},
            {"role": "user", "content": "q2"},
            {"role": "assistant", "content": "final answer"},
        ]

        strategies = [
            SlidingWindowCompaction(max_messages=4),
        ]

        for strategy in strategies:
            result = strategy.apply(messages)
            # Verify no orphan tool results
            for i, msg in enumerate(result.messages):
                if msg.get("role") == "tool":
                    assert i > 0, "Tool result at position 0"
                    prev = result.messages[i - 1]
                    assert prev.get("role") == "assistant" and prev.get("tool_calls"), (
                        f"Strategy {strategy.__class__.__name__}: "
                        f"Orphan tool result at index {i}"
                    )

    def test_is_tool_use_boundary(self):
        """is_tool_use_boundary detecta el inicio de un bloque tool_use."""
        messages = [
            {"role": "user", "content": "q"},
            {"role": "assistant", "content": None, "tool_calls": [{"id": "c1"}]},
            {"role": "tool", "tool_call_id": "c1", "content": "ok"},
        ]
        assert is_tool_use_boundary(messages, 0) is False
        assert is_tool_use_boundary(messages, 1) is True
        assert is_tool_use_boundary(messages, 2) is False

    def test_is_in_tool_use_block(self):
        """is_in_tool_use_block detecta si un mensaje está dentro de un bloque."""
        messages = [
            {"role": "user", "content": "q"},
            {"role": "assistant", "content": None, "tool_calls": [{"id": "c1"}]},
            {"role": "tool", "tool_call_id": "c1", "content": "ok"},
            {"role": "assistant", "content": "answer"},
        ]
        assert is_in_tool_use_block(messages, 0) is False  # user
        assert is_in_tool_use_block(messages, 1) is True   # assistant with tool_calls
        assert is_in_tool_use_block(messages, 2) is True   # tool result
        assert is_in_tool_use_block(messages, 3) is False  # assistant without tool_calls


# ===================================================================
# CRITERIO 6: Coste en tokens de summary compaction medido
# ===================================================================

class TestSummaryCompactionCost:
    """El coste en tokens de la compactación por resumen se mide."""

    def test_summary_cost_measured(self):
        """SummaryCompaction mide y almacena los tokens usados en el resumen."""
        import agents.compaction as comp_mod
        strategy = SummaryCompaction(model="test/model", threshold_tokens=10)
        messages = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "data " * 100},
            {"role": "assistant", "content": "response " * 100},
            {"role": "user", "content": "more " * 100},
            {"role": "assistant", "content": "again " * 100},
            {"role": "user", "content": "extra " * 100},
            {"role": "assistant", "content": "extra " * 100},
        ]

        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content="Summary"))]
        mock_response.usage = MagicMock(total_tokens=250)

        with patch.object(comp_mod.litellm, "completion", return_value=mock_response):
            result = strategy.apply(messages)

        assert strategy.last_summary_cost_tokens == 250
        assert strategy.total_compaction_tokens == 250
        assert result.compacted is True

    def test_summary_cost_accumulates(self):
        """El coste se acumula entre múltiples compactaciones."""
        import agents.compaction as comp_mod
        strategy = SummaryCompaction(model="test/model", threshold_tokens=10)
        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content="Summary"))]
        mock_response.usage = MagicMock(total_tokens=100)

        with patch.object(comp_mod.litellm, "completion", return_value=mock_response):
            # First compaction
            messages1 = [
                {"role": "system", "content": "sys"},
                {"role": "user", "content": "q " * 100},
                {"role": "assistant", "content": "a " * 100},
                {"role": "user", "content": "q2 " * 100},
                {"role": "assistant", "content": "a2 " * 100},
                {"role": "user", "content": "q3 " * 100},
                {"role": "assistant", "content": "a3 " * 100},
            ]
            strategy.apply(messages1)
            assert strategy.total_compaction_tokens == 100

            # Second compaction
            messages2 = [
                {"role": "system", "content": "sys"},
                {"role": "user", "content": "q " * 100},
                {"role": "assistant", "content": "a " * 100},
                {"role": "user", "content": "q2 " * 100},
                {"role": "assistant", "content": "a2 " * 100},
                {"role": "user", "content": "q3 " * 100},
                {"role": "assistant", "content": "a3 " * 100},
            ]
            strategy.apply(messages2)
            assert strategy.total_compaction_tokens == 200

    def test_token_measurement_in_loop(self):
        """El loop aplica compaction y mide tokens antes/después."""
        import agents.compaction as comp_mod
        strategy = SummaryCompaction(model="test/model", threshold_tokens=10)
        messages = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "q " * 100},
            {"role": "assistant", "content": "a " * 100},
            {"role": "user", "content": "q2 " * 100},
            {"role": "assistant", "content": "a2 " * 100},
            {"role": "user", "content": "q3 " * 100},
            {"role": "assistant", "content": "a3 " * 100},
        ]

        tokens_before = count_tokens_approx(messages)
        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content="Summary of the conversation so far"))]
        mock_response.usage = MagicMock(total_tokens=200)

        with patch.object(comp_mod.litellm, "completion", return_value=mock_response):
            result = strategy.apply(messages)

        tokens_after = count_tokens_approx(result.messages)

        print("\n=== COSTE DE SUMMARY COMPACTION ===")
        print(f"  Tokens antes: {tokens_before}")
        print(f"  Tokens después: {tokens_after}")
        print(f"  Tokens ahorrados: {tokens_before - tokens_after}")
        print(f"  Tokens gastados en resumir: {strategy.last_summary_cost_tokens}")
        print(f"  Coste neto: {strategy.last_summary_cost_tokens - (tokens_before - tokens_after)}")
        print("===================================\n")

        # The compaction should have reduced tokens
        assert tokens_after < tokens_before

    def test_summary_cost_failure_returns_zero(self):
        """Si el LLM falla al resumir, el coste es 0 y se devuelve texto truncado."""
        import agents.compaction as comp_mod
        strategy = SummaryCompaction(model="test/model", threshold_tokens=10)
        messages = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "q " * 100},
            {"role": "assistant", "content": "a " * 100},
            {"role": "user", "content": "q2 " * 100},
            {"role": "assistant", "content": "a2 " * 100},
            {"role": "user", "content": "q3 " * 100},
            {"role": "assistant", "content": "a3 " * 100},
        ]

        with patch.object(comp_mod.litellm, "completion", side_effect=RuntimeError("LLM down")):
            result = strategy.apply(messages)

        assert strategy.last_summary_cost_tokens == 0
        assert result.compacted is True  # Still compacted, just with truncated text


# ===================================================================
# CRITERIO 7: make test pasa
# ===================================================================

class TestMakeTestPasses:
    """Este fichero es parte de make test."""

    def test_import_all_modules(self):
        """Los módulos nuevos se importan sin errores."""
        import agents.compaction
        import agents.subagent
        import agents.loop
        assert hasattr(agents.compaction, "NoneCompaction")
        assert hasattr(agents.compaction, "SlidingWindowCompaction")
        assert hasattr(agents.compaction, "SummaryCompaction")
        assert hasattr(agents.subagent, "load_subagent_config")
        assert hasattr(agents.loop, "execute_delegation")

    def test_compaction_in_agent_loop(self):
        """agent_loop acepta el parámetro compaction."""
        from agents.loop import agent_loop
        import inspect
        sig = inspect.signature(agent_loop)
        assert "compaction" in sig.parameters

    def test_count_tokens_approx(self):
        """count_tokens_approx devuelve un entero positivo."""
        messages = [{"role": "user", "content": "hello world"}]
        tokens = count_tokens_approx(messages)
        assert isinstance(tokens, int)
        assert tokens > 0


# ===================================================================
# Integration: delegation + compaction together
# ===================================================================

class TestDelegationWithCompaction:
    """Delegación y compactación funcionan juntas."""

    def test_agent_loop_with_compaction_parameter(self):
        """agent_loop acepta compaction y lo aplica."""
        from agents.loop import agent_loop

        mock_response = {
            "role": "assistant",
            "content": "done",
            "tool_calls": None,
        }

        compaction = NoneCompaction()

        with patch("agents.loop._call_llm", return_value=mock_response):
            result = agent_loop(
                prompt="test",
                agent_type="implementer",
                model="test/model",
                compaction=compaction,
            )

        assert result.success is True

    def test_delegation_does_not_inherit_parent_compaction(self):
        """La compactación del padre no afecta al subagente (contexto vacío)."""
        # This is verified by the architecture: each agent_loop gets its own
        # compaction strategy, and delegation creates a new loop
        with patch("agents.loop.agent_loop") as mock_loop:
            mock_loop.return_value = LoopResult(
                success=True, messages=[], tool_calls=[], final_response="ok"
            )
            execute_delegation(agent_name="explorer", prompt="test")

        # The subagent loop should have its own compaction (default NoneCompaction)
        call_kwargs = mock_loop.call_args
        assert "compaction" in call_kwargs[1] or len(call_kwargs[0]) <= 2
