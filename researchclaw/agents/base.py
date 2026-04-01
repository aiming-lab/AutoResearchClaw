"""Base classes for multi-agent subsystems.

Provides ``BaseAgent`` (individual agent) and ``AgentOrchestrator``
(coordinator for multi-agent workflows).  Both use the existing
``LLMClient`` for model calls and follow the same structural-typing
conventions as ``CodeAgent``.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Protocol

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# LLM protocol (structural typing — no import dependency on llm.client)
# ---------------------------------------------------------------------------


class _LLMResponseLike(Protocol):  # pragma: no cover
    content: str
    model: str
    prompt_tokens: int
    completion_tokens: int


class _LLMClientLike(Protocol):  # pragma: no cover
    def chat(
        self,
        messages: list[dict[str, str]],
        *,
        system: str | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
        json_mode: bool = False,
    ) -> Any: ...


# ---------------------------------------------------------------------------
# Agent result
# ---------------------------------------------------------------------------


@dataclass
class AgentStepResult:
    """Output from a single agent step."""

    success: bool
    data: dict[str, Any] = field(default_factory=dict)
    error: str = ""
    llm_calls: int = 0
    token_usage: int = 0


# ---------------------------------------------------------------------------
# Base agent
# ---------------------------------------------------------------------------


class BaseAgent:
    """Base class for all sub-agents in a multi-agent system.

    Subclasses must implement ``execute(context) -> AgentStepResult``.
    """

    name: str = "base"

    def __init__(self, llm: _LLMClientLike) -> None:
        self._llm = llm
        self._calls = 0
        self._tokens = 0
        self.logger = logging.getLogger(f"{__name__}.{self.name}")

    # -- LLM helpers -------------------------------------------------------

    def _chat(
        self,
        system: str,
        user: str,
        *,
        max_tokens: int = 4096,
        temperature: float = 0.4,
        json_mode: bool = False,
    ) -> str:
        """Send a chat message and return the content string."""
        self._calls += 1
        resp = self._llm.chat(
            [{"role": "user", "content": user}],
            system=system,
            max_tokens=max_tokens,
            temperature=temperature,
            json_mode=json_mode,
        )
        self._tokens += getattr(resp, "total_tokens", 0)
        return resp.content

    def _chat_json(
        self,
        system: str,
        user: str,
        *,
        max_tokens: int = 4096,
        temperature: float = 0.3,
    ) -> dict[str, Any]:
        """Send a chat message expecting JSON output.  Falls back to regex extraction."""
        raw = self._chat(
            system, user,
            max_tokens=max_tokens,
            temperature=temperature,
            json_mode=True,
        )
        return self._parse_json(raw) or {}

    # -- JSON parsing (3-tier, matching CodeAgent convention) ---------------

    @staticmethod
    def _parse_json(text: str) -> dict[str, Any] | None:
        """Try to extract JSON from text using three strategies.

        Always returns a ``dict`` or ``None`` — lists and other JSON
        primitives are discarded so callers can safely use ``.get()``.
        """
        def _as_dict(val: Any) -> dict[str, Any] | None:
            return val if isinstance(val, dict) else None

        # 1. Direct parse
        try:
            return _as_dict(json.loads(text))
        except (json.JSONDecodeError, ValueError):
            pass
        # 2. Fenced code block
        m = re.search(r"```(?:json)?\s*\n(.*?)```", text, re.DOTALL)
        if m:
            try:
                return _as_dict(json.loads(m.group(1)))
            except (json.JSONDecodeError, ValueError):
                pass
        # 3. First balanced { ... } block (BUG-DA6-07: use non-greedy brace matching)
        depth = 0
        start_idx = -1
        for i, ch in enumerate(text):
            if ch == "{":
                if depth == 0:
                    start_idx = i
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0 and start_idx >= 0:
                    candidate = text[start_idx : i + 1]
                    try:
                        return _as_dict(json.loads(candidate))
                    except (json.JSONDecodeError, ValueError):
                        start_idx = -1  # try next top-level block
        return None

    # -- Subclass API ------------------------------------------------------

    def execute(self, context: dict[str, Any]) -> AgentStepResult:
        """Execute the agent's task.  Must be overridden."""
        raise NotImplementedError

    def _make_result(
        self, success: bool, data: dict[str, Any] | None = None, error: str = "",
    ) -> AgentStepResult:
        # BUG-DA6-01: Return per-call delta, then reset counters to avoid
        # double-counting when the same agent instance is reused across retries.
        calls, tokens = self._calls, self._tokens
        self._calls = 0
        self._tokens = 0
        return AgentStepResult(
            success=success,
            data=data or {},
            error=error,
            llm_calls=calls,
            token_usage=tokens,
        )


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


class AgentOrchestrator:
    """Coordinates a sequence of agents with optional retry loops.

    Subclasses implement ``orchestrate(context) -> dict`` which defines the
    specific workflow (sequential, branching, iterative, etc.).
    """

    def __init__(self, llm: _LLMClientLike, *, max_iterations: int = 3) -> None:
        self._llm = llm
        self.max_iterations = max_iterations
        self.logger = logging.getLogger(f"{__name__}.orchestrator")
        self.total_llm_calls = 0
        self.total_tokens = 0

    def _accumulate(self, result: AgentStepResult) -> None:
        """Track cumulative LLM usage."""
        self.total_llm_calls += result.llm_calls
        self.total_tokens += result.token_usage

    def orchestrate(self, context: dict[str, Any]) -> dict[str, Any]:
        """Run the multi-agent workflow.  Must be overridden."""
        raise NotImplementedError
