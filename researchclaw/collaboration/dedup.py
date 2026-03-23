"""Cross-instance deduplication for shared artifacts."""

from __future__ import annotations

import hashlib
import logging
from typing import Any

logger = logging.getLogger(__name__)


def content_hash(content: Any) -> str:
    """Compute a content hash for deduplication.

    Args:
        content: Content to hash (str, dict, or list).

    Returns:
        Hex digest string.
    """
    if isinstance(content, (dict, list)):
        import json
        text = json.dumps(content, sort_keys=True, default=str)
    else:
        text = str(content)

    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def deduplicate_artifacts(
    artifacts: list[dict[str, Any]],
    key: str = "content",
) -> list[dict[str, Any]]:
    """Remove duplicate artifacts based on content hash.

    Args:
        artifacts: List of artifact dicts.
        key: Dict key containing the content to compare.

    Returns:
        Deduplicated list, preserving first occurrence.
    """
    seen: set[str] = set()
    unique: list[dict[str, Any]] = []

    for artifact in artifacts:
        h = content_hash(artifact.get(key, ""))
        if h not in seen:
            seen.add(h)
            unique.append(artifact)

    removed = len(artifacts) - len(unique)
    if removed > 0:
        logger.info("Deduplication removed %d artifacts", removed)

    return unique
