"""MetaClaw session lifecycle management for AutoResearchClaw.

Manages MetaClaw proxy session headers and lifecycle signals
to enable proper skill evolution and RL training data collection.
"""

from __future__ import annotations

import logging
import uuid

logger = logging.getLogger(__name__)


class MetaClawSession:
    """Manages a MetaClaw session spanning an AutoResearchClaw pipeline run."""

    def __init__(self, run_id: str) -> None:
        self.session_id = f"arc-{run_id}"
        self._active = True
        logger.info("MetaClaw session started: %s", self.session_id)

    def get_headers(self, stage_name: str = "") -> dict[str, str]:
        """Return HTTP headers for MetaClaw proxy requests.

        Args:
            stage_name: Current pipeline stage (for logging/tracking).

        Returns:
            Dict of headers to include in LLM API requests.
        """
        headers: dict[str, str] = {
            "X-Session-Id": self.session_id,
            "X-Turn-Type": "main",
        }
        if stage_name:
            headers["X-AutoRC-Stage"] = stage_name
        return headers

    def end(self) -> dict[str, str]:
        """Return headers that signal session completion.

        Call this when the pipeline run finishes to trigger
        MetaClaw's post-session processing (skill evolution, etc.).
        """
        self._active = False
        logger.info("MetaClaw session ended: %s", self.session_id)
        return {
            "X-Session-Id": self.session_id,
            "X-Session-Done": "true",
            "X-Turn-Type": "main",
        }

    @property
    def is_active(self) -> bool:
        return self._active
