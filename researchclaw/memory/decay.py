"""Time-decay and confidence scoring for memory entries."""

from __future__ import annotations

import math
from datetime import datetime, timezone


def time_decay_weight(
    created_at: datetime,
    half_life_days: float = 90.0,
    max_age_days: float = 365.0,
    *,
    now: datetime | None = None,
) -> float:
    """Compute exponential decay weight based on entry age.

    Args:
        created_at: When the memory was created.
        half_life_days: Half-life in days (weight = 0.5 after this many days).
        max_age_days: Entries older than this get weight 0.0.
        now: Current time (defaults to UTC now).

    Returns:
        Weight in [0.0, 1.0].
    """
    if now is None:
        now = datetime.now(timezone.utc)
    if created_at.tzinfo is None:
        created_at = created_at.replace(tzinfo=timezone.utc)
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)

    age_seconds = (now - created_at).total_seconds()
    age_days = age_seconds / 86400.0

    if age_days < 0:
        return 1.0
    if age_days > max_age_days:
        return 0.0

    return math.exp(-age_days * math.log(2) / half_life_days)


def confidence_update(
    current: float,
    delta: float,
    floor: float = 0.0,
    ceiling: float = 1.0,
) -> float:
    """Update confidence score with clamping.

    Args:
        current: Current confidence value.
        delta: Change amount (positive for success, negative for failure).
        floor: Minimum allowed value.
        ceiling: Maximum allowed value.

    Returns:
        Updated confidence clamped to [floor, ceiling].
    """
    return max(floor, min(ceiling, current + delta))
