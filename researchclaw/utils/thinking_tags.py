"""Strip reasoning artifacts from LLM output before they leak into papers.

Handles ALL known thinking/reasoning formats:
- ``<think>...</think>`` -- DeepSeek-R1, QwQ, Gemini 2.5 format
- ``[thinking] ...`` -- Claude Code / ACP output format (bracket-style)
- Insight blocks -- Claude Code explanatory mode decorators
- ``[plan] ...`` -- Claude Code plan mode markers
- ``[tool] ...`` -- ACP tool invocation output
- ``[client] ...``, ``[acpx] ...``, ``[done] ...`` -- acpx metadata

Without this stripping, these artifacts contaminate:
- Paper drafts (LaTeX / Markdown)
- Generated experiment code
- YAML/JSON responses (search plans, experiment plans)
- Citation references

Usage::

    from researchclaw.utils.thinking_tags import strip_thinking_tags

    clean = strip_thinking_tags(raw_llm_output)
"""

from __future__ import annotations

import re

# ---------------------------------------------------------------------------
# Pattern 1: XML-style <think>...</think> (DeepSeek-R1, QwQ, Gemini)
# ---------------------------------------------------------------------------

_THINK_BLOCK_RE = re.compile(
    r"<think>.*?</think>",
    re.DOTALL | re.IGNORECASE,
)
_THINK_UNCLOSED_RE = re.compile(
    r"<think>.*",
    re.DOTALL | re.IGNORECASE,
)
_THINK_STRAY_CLOSE_RE = re.compile(
    r"</think>",
    re.IGNORECASE,
)

# ---------------------------------------------------------------------------
# Pattern 2: [thinking] blocks (Claude Code / ACP)
# ---------------------------------------------------------------------------

_BRACKET_THINKING_RE = re.compile(
    r"\[thinking\].*?(?=\n\n(?!\[thinking\])|\n(?:#{1,3}\s)|\n```|\Z)",
    re.DOTALL | re.IGNORECASE,
)

# ---------------------------------------------------------------------------
# Pattern 3: Insight blocks (Claude Code explanatory style)
# ---------------------------------------------------------------------------

_INSIGHT_BLOCK_RE = re.compile(
    r"`[*\u2605]\s*Insight[^`]*`\s*\n.*?`[\u2500-]+`",
    re.DOTALL,
)
_INSIGHT_ASCII_RE = re.compile(
    r"`\*\s*Insight[-]+`\s*\n.*?`[-]+`",
    re.DOTALL,
)

# ---------------------------------------------------------------------------
# Pattern 4: [plan] blocks (Claude Code plan mode)
# ---------------------------------------------------------------------------

_PLAN_BLOCK_RE = re.compile(
    r"\[plan\].*?(?=\n\n|\Z)",
    re.DOTALL,
)

# ---------------------------------------------------------------------------
# Pattern 5: ACP/acpx metadata lines
# ---------------------------------------------------------------------------

_ACPX_LINE_RE = re.compile(
    r"^\[(client|acpx|tool|done)\](?!\().*$",
    re.MULTILINE | re.IGNORECASE,
)


def strip_thinking_tags(text: str) -> str:
    """Remove all reasoning artifacts from LLM output.

    Handles XML <think> tags, bracket [thinking] blocks, insight
    decorators, plan markers, and acpx metadata.

    Returns cleaned text suitable for paper drafts, code, or YAML/JSON.
    """
    if not text:
        return text

    result = text

    # Phase 1: XML <think>...</think> blocks
    if "think" in result.lower():
        result = _THINK_BLOCK_RE.sub("", result)
        result = _THINK_UNCLOSED_RE.sub("", result)
        result = _THINK_STRAY_CLOSE_RE.sub("", result)

    # Phase 2: [thinking] blocks (ACP/Claude Code)
    if "[thinking]" in result.lower():
        result = _BRACKET_THINKING_RE.sub("", result)
        result = re.sub(
            r"^\[thinking\].*$", "", result,
            flags=re.MULTILINE | re.IGNORECASE,
        )

    # Phase 3: Insight blocks
    result = _INSIGHT_BLOCK_RE.sub("", result)
    result = _INSIGHT_ASCII_RE.sub("", result)

    # Phase 4: [plan] blocks
    if "[plan]" in result.lower():
        result = _PLAN_BLOCK_RE.sub("", result)

    # Phase 5: acpx metadata lines
    result = _ACPX_LINE_RE.sub("", result)

    # Phase 6: Clean up artifacts
    result = re.sub(r"^`[\u2500-]+`\s*$", "", result, flags=re.MULTILINE)
    result = re.sub(r"^`[-]{20,}`\s*$", "", result, flags=re.MULTILINE)

    # Collapse excessive blank lines
    result = re.sub(r"\n{3,}", "\n\n", result)

    return result.strip()
