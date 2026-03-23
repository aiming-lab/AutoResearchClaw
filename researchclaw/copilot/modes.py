"""Research mode definitions for Co-Pilot."""

from __future__ import annotations

from enum import Enum


class ResearchMode(Enum):
    """Pipeline execution modes."""

    CO_PILOT = "co-pilot"        # Pause at every stage for feedback
    AUTO_PILOT = "auto-pilot"    # Pause only at gate stages
    ZERO_TOUCH = "zero-touch"    # Fully automatic, no pauses
