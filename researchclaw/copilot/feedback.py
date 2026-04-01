"""User feedback processing for Co-Pilot mode."""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


FEEDBACK_ACTIONS = frozenset({
    "approve",   # Continue to next stage
    "modify",    # Apply modifications and continue
    "retry",     # Re-run current stage
    "skip",      # Skip current stage
    "discuss",   # Enter discussion mode (future)
    "branch",    # Create exploration branch
    "rollback",  # Roll back to a previous stage
})


@dataclass(frozen=True)
class Feedback:
    """Structured user feedback for a pipeline stage."""

    action: str
    stage: int
    message: str = ""
    modifications: dict[str, Any] | None = None
    branch_name: str = ""
    rollback_to: int | None = None
    timestamp: str = ""


class FeedbackHandler:
    """Handle feedback input/output for Co-Pilot mode."""

    def __init__(self, run_dir: Path):
        self.run_dir = run_dir

    def write_feedback_request(
        self,
        stage: int,
        stage_name: str,
        summary: str,
        options: list[str] | None = None,
    ) -> Path:
        """Write a feedback request file for external consumers."""
        from datetime import datetime, timezone

        request = {
            "stage": stage,
            "stage_name": stage_name,
            "summary": summary,
            "options": options or list(FEEDBACK_ACTIONS),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "status": "waiting",
        }
        request_path = self.run_dir / "copilot_feedback_request.json"
        request_path.write_text(
            json.dumps(request, indent=2), encoding="utf-8"
        )
        return request_path

    def read_feedback_response(self) -> Feedback | None:
        """Read feedback response from file (written by UI or CLI)."""
        response_path = self.run_dir / "copilot_feedback_response.json"
        if not response_path.exists():
            return None
        try:
            data = json.loads(response_path.read_text(encoding="utf-8"))
            action = data.get("action", "approve")
            if action not in FEEDBACK_ACTIONS:
                logger.warning("Invalid feedback action: %s", action)
                return None
            return Feedback(
                action=action,
                stage=int(data.get("stage", 0)),
                message=str(data.get("message", "")),
                modifications=data.get("modifications"),
                branch_name=str(data.get("branch_name", "")),
                rollback_to=(
                    int(data["rollback_to"])
                    if data.get("rollback_to") is not None
                    else None
                ),
                timestamp=str(data.get("timestamp", "")),
            )
        except (json.JSONDecodeError, TypeError, ValueError) as exc:
            logger.warning("Failed to parse feedback response: %s", exc)
            return None

    def wait_for_feedback(
        self,
        stage: int,
        timeout_sec: int = 3600,
        poll_interval_sec: float = 1.0,
    ) -> Feedback | None:
        """Wait for feedback by polling the response file."""
        response_path = self.run_dir / "copilot_feedback_response.json"

        # Clear any stale response
        if response_path.exists():
            response_path.unlink()

        deadline = time.monotonic() + timeout_sec
        while time.monotonic() < deadline:
            feedback = self.read_feedback_response()
            if feedback is not None and feedback.stage == stage:
                # Clean up response file
                if response_path.exists():
                    response_path.unlink()
                return feedback
            time.sleep(poll_interval_sec)

        logger.info("Feedback timeout for stage %d after %ds", stage, timeout_sec)
        return None

    def clear_request(self) -> None:
        """Clear the feedback request file."""
        request_path = self.run_dir / "copilot_feedback_request.json"
        if request_path.exists():
            request_path.unlink()
