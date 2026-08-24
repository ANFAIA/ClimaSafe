"""
agents.compaction — Estrategias de compactación de contexto para el agent loop.

Tres estrategias intercambiables, seleccionables sin tocar el bucle:
  1. NoneCompaction: no hace nada (passthrough)
  2. SlidingWindowCompaction: mantiene system prompt + últimos N mensajes
  3. SummaryCompaction: usa el propio modelo para resumir mensajes antiguos

Ninguna estrategia parte un bloque de tool_use por la mitad.
"""

from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import litellm

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Token counting helpers
# ---------------------------------------------------------------------------

def count_tokens_approx(messages: list[dict[str, Any]]) -> int:
    """Estimación aproximada de tokens: 1 token ≈ 4 caracteres (heurística estándar)."""
    total_chars = 0
    for msg in messages:
        content = msg.get("content") or ""
        if isinstance(content, str):
            total_chars += len(content)
        elif isinstance(content, list):
            # content can be a list of content blocks (text, image, etc.)
            for block in content:
                if isinstance(block, dict) and "text" in block:
                    total_chars += len(block["text"])
        # Also count tool_calls arguments
        for tc in msg.get("tool_calls") or []:
            args = tc.get("function", {}).get("arguments", "")
            if isinstance(args, str):
                total_chars += len(args)
    return total_chars // 4


def is_tool_use_boundary(messages: list[dict[str, Any]], index: int) -> bool:
    """
    Devuelve True si el mensaje en `index` es un tool_use boundary:
    el inicio de un bloque assistant+tool que no debe partirse.

    Un bloque tool_use es: assistant con tool_calls seguido de N tool results.
    El boundary es el assistant message que inicia el bloque.
    """
    if index < 0 or index >= len(messages):
        return False
    msg = messages[index]
    # Un assistant message con tool_calls es el inicio de un bloque
    if msg.get("role") == "assistant" and msg.get("tool_calls"):
        return True
    return False


def is_in_tool_use_block(messages: list[dict[str, Any]], index: int) -> bool:
    """
    Devuelve True si el mensaje en `index` está dentro de un bloque tool_use
    (assistant con tool_calls + sus tool results consecutivos).
    """
    # Walk backwards from index to find if we're inside a tool_use block
    for i in range(index, -1, -1):
        msg = messages[i]
        if msg.get("role") == "assistant" and msg.get("tool_calls"):
            # We're inside or at the start of a tool_use block
            return True
        if msg.get("role") == "tool":
            # A tool result — continue looking backwards
            continue
        # Any other role breaks the block
        break
    return False


# ---------------------------------------------------------------------------
# Compaction strategy interface
# ---------------------------------------------------------------------------

@dataclass
class CompactionResult:
    """Resultado de una compactación."""
    messages: list[dict[str, Any]]
    tokens_before: int
    tokens_after: int
    compacted: bool


class CompactionStrategy(ABC):
    """Interfaz base para estrategias de compactación."""

    @abstractmethod
    def apply(self, messages: list[dict[str, Any]]) -> CompactionResult:
        """Aplica la compactación y devuelve los mensajes resultantes."""
        ...


# ---------------------------------------------------------------------------
# Strategy 1: No-op
# ---------------------------------------------------------------------------

class NoneCompaction(CompactionStrategy):
    """No hace nada. Los mensajes pasan tal cual."""

    def apply(self, messages: list[dict[str, Any]]) -> CompactionResult:
        tokens = count_tokens_approx(messages)
        return CompactionResult(
            messages=messages,
            tokens_before=tokens,
            tokens_after=tokens,
            compacted=False,
        )


# ---------------------------------------------------------------------------
# Strategy 2: Sliding window
# ---------------------------------------------------------------------------

class SlidingWindowCompaction(CompactionStrategy):
    """
    Mantiene el system prompt + los últimos N mensajes.
    Respeta los bloques tool_use: nunca corta un bloque assistant+tool por la mitad.

    Parámetros:
      - max_messages: máximo de mensajes a mantener (sin contar system prompt)
    """

    def __init__(self, max_messages: int = 20) -> None:
        self.max_messages = max_messages

    def apply(self, messages: list[dict[str, Any]]) -> CompactionResult:
        tokens_before = count_tokens_approx(messages)

        if len(messages) <= self.max_messages + 1:  # +1 for system prompt
            return CompactionResult(
                messages=messages,
                tokens_before=tokens_before,
                tokens_after=tokens_before,
                compacted=False,
            )

        # Separate system prompt from the rest
        system_msgs: list[dict[str, Any]] = []
        other_msgs: list[dict[str, Any]] = []
        for msg in messages:
            if msg.get("role") == "system":
                system_msgs.append(msg)
            else:
                other_msgs.append(msg)

        # If we need to trim, find a safe cut point that doesn't break tool_use blocks
        if len(other_msgs) <= self.max_messages:
            result = system_msgs + other_msgs
        else:
            # Start from the end and walk backwards, keeping max_messages
            # but skipping any messages that are part of an incomplete tool_use block
            keep_from = len(other_msgs) - self.max_messages

            # Adjust keep_from to avoid splitting a tool_use block
            # If keep_from lands inside a tool_use block, move it to before the block
            while keep_from > 0 and is_in_tool_use_block(other_msgs, keep_from):
                keep_from -= 1

            result = system_msgs + other_msgs[keep_from:]

        tokens_after = count_tokens_approx(result)
        return CompactionResult(
            messages=result,
            tokens_before=tokens_before,
            tokens_after=tokens_after,
            compacted=(tokens_after < tokens_before),
        )


# ---------------------------------------------------------------------------
# Strategy 3: Summary with the model itself
# ---------------------------------------------------------------------------

_SUMMARY_SYSTEM_PROMPT = (
    "You are a conversation summarizer. Compress the following conversation "
    "into a concise summary that preserves the key facts, decisions, tool "
    "results, and current state. Be factual, not creative. Include all tool "
    "outputs verbatim or near-verbatim since they contain important data."
)


class SummaryCompaction(CompactionStrategy):
    """
    Usa el propio modelo para resumir mensajes antiguos.

    Parámetros:
      - model: modelo a usar para el resumen (mismo que el del loop o uno más barato)
      - threshold_tokens: si los tokens superan este umbral, compacta
      - max_summary_tokens: tokens máximos del resumen generado
    """

    def __init__(
        self,
        model: str = "ollama/llama3.2",
        threshold_tokens: int = 4000,
        max_summary_tokens: int = 1000,
    ) -> None:
        self.model = model
        self.threshold_tokens = threshold_tokens
        self.max_summary_tokens = max_summary_tokens
        self.last_summary_cost_tokens: int = 0
        self.total_compaction_tokens: int = 0

    def apply(self, messages: list[dict[str, Any]]) -> CompactionResult:
        tokens_before = count_tokens_approx(messages)

        if tokens_before <= self.threshold_tokens:
            return CompactionResult(
                messages=messages,
                tokens_before=tokens_before,
                tokens_after=tokens_before,
                compacted=False,
            )

        # Separate system prompt and recent messages to keep
        system_msgs: list[dict[str, Any]] = []
        other_msgs: list[dict[str, Any]] = []
        for msg in messages:
            if msg.get("role") == "system":
                system_msgs.append(msg)
            else:
                other_msgs.append(msg)

        if len(other_msgs) <= 4:
            return CompactionResult(
                messages=messages,
                tokens_before=tokens_before,
                tokens_after=tokens_before,
                compacted=False,
            )

        # Keep last 4 messages intact, summarize the rest
        to_summarize = other_msgs[:-4]
        to_keep = other_msgs[-4:]

        # Build the conversation text for summarization
        conversation_text = self._format_for_summary(to_summarize)

        # Call the model to summarize
        summary, summary_tokens = self._summarize(conversation_text)
        self.last_summary_cost_tokens = summary_tokens
        self.total_compaction_tokens += summary_tokens

        # Build the compacted message list
        summary_msg: dict[str, Any] = {
            "role": "user",
            "content": f"[Conversation summary]\n{summary}",
        }

        result = system_msgs + [summary_msg] + to_keep
        tokens_after = count_tokens_approx(result)

        return CompactionResult(
            messages=result,
            tokens_before=tokens_before,
            tokens_after=tokens_after,
            compacted=True,
        )

    def _format_for_summary(self, messages: list[dict[str, Any]]) -> str:
        """Formatea mensajes para el prompt de resumen."""
        parts: list[str] = []
        for msg in messages:
            role = msg.get("role", "unknown")
            content = msg.get("content") or ""
            if isinstance(content, list):
                # Handle content blocks
                text_parts = []
                for block in content:
                    if isinstance(block, dict) and "text" in block:
                        text_parts.append(block["text"])
                content = " ".join(text_parts)

            if msg.get("tool_calls"):
                for tc in msg["tool_calls"]:
                    func = tc.get("function", {})
                    parts.append(
                        f"[{role}] tool_call: {func.get('name', '?')}"
                        f"({func.get('arguments', '')})"
                    )
            elif role == "tool":
                parts.append(f"[tool_result] {content[:500]}")
            else:
                parts.append(f"[{role}] {content}")

        return "\n".join(parts)

    def _summarize(self, conversation_text: str) -> tuple[str, int]:
        """Llama al modelo para generar un resumen. Devuelve (resumen, tokens_usados)."""
        messages = [
            {"role": "system", "content": _SUMMARY_SYSTEM_PROMPT},
            {"role": "user", "content": conversation_text},
        ]

        try:
            response = litellm.completion(
                model=self.model,
                messages=messages,
                max_tokens=self.max_summary_tokens,
            )
            summary = response.choices[0].message.content or ""
            usage = getattr(response, "usage", None)
            total_tokens = 0
            if usage:
                total_tokens = getattr(usage, "total_tokens", 0) or 0
            return summary.strip(), total_tokens
        except Exception as exc:
            logger.warning("Summary compaction failed: %s", exc)
            return conversation_text[:self.max_summary_tokens * 4], 0
