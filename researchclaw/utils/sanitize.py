"""Input sanitization utilities for untrusted LLM-generated values."""

from __future__ import annotations

import re


def sanitize_figure_id(raw_id: str, *, fallback: str = "figure") -> str:
    """Sanitize a figure ID for safe use in file paths and Docker names.

    Strips path separators, dotdot sequences, and shell metacharacters.
    Returns *fallback* if the sanitized result is empty.

    >>> sanitize_figure_id("../../etc/evil")
    'etc_evil'
    >>> sanitize_figure_id("fig test (v2)")
    'fig_test_v2'
    >>> sanitize_figure_id("")
    'figure'
    """
    # Replace path separators and dangerous sequences
    cleaned = raw_id.replace("..", "").replace("/", "_").replace("\\", "_")
    # Keep only safe characters: alphanumeric, hyphen, underscore, dot
    cleaned = re.sub(r"[^a-zA-Z0-9_.-]", "_", cleaned)
    # Collapse multiple underscores
    cleaned = re.sub(r"_+", "_", cleaned).strip("_.")
    return cleaned or fallback
