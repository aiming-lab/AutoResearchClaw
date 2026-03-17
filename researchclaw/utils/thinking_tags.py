"""Utility to strip ``<think>`` / ``</think>`` reasoning tags from LLM output.

Many reasoning-capable models (DeepSeek-R1, QwQ, etc.) wrap their internal
chain-of-thought in ``<think>…</think>`` blocks.  These must **never** leak
into research papers, generated scripts, or pipeline artifacts.

Usage::

    from researchclaw.utils.thinking_tags import strip_thinking_tags

    clean = strip_thinking_tags(raw_llm_output)
"""

from __future__ import annotations

import re

# Regex patterns:
#   1) Multi-line <think>…</think> blocks (greedy within block, DOTALL).
#   2) Unclosed <think> at the start of response (model started thinking
#      but response was truncated before closing tag).
#   3) Stray closing </think> without an opener.

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


def strip_thinking_tags(text: str) -> str:
    """Remove ``<think>…</think>`` blocks and stray tags from *text*.

    Returns the cleaned text with leading/trailing whitespace trimmed where
    the tags were removed, but internal content is preserved verbatim.
    """
    if not text or ("think" not in text.lower()):
        return text

    # Phase 1: Remove complete <think>…</think> blocks
    result = _THINK_BLOCK_RE.sub("", text)

    # Phase 2: Remove unclosed <think> (truncated response)
    result = _THINK_UNCLOSED_RE.sub("", result)

    # Phase 3: Remove stray </think>
    result = _THINK_STRAY_CLOSE_RE.sub("", result)

    # Clean up excessive blank lines left behind
    result = re.sub(r"\n{3,}", "\n\n", result)

    return result.strip()
