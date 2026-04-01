"""Co-Pilot controller — orchestrates pause/feedback/branch logic."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from researchclaw.config import CoPilotConfig
from researchclaw.copilot.branching import BranchManager
from researchclaw.copilot.feedback import Feedback, FeedbackHandler
from researchclaw.copilot.modes import ResearchMode
from researchclaw.pipeline.stages import GATE_STAGES

logger = logging.getLogger(__name__)


class CoPilotController:
    """Control Co-Pilot mode during pipeline execution."""

    def __init__(
        self,
        config: CoPilotConfig,
        run_dir: Path,
    ):
        self.config = config
        self.mode = ResearchMode(config.mode)
        self.run_dir = run_dir
        self.feedback_handler = FeedbackHandler(run_dir)
        self.branch_manager = BranchManager(
            run_dir, max_branches=config.max_branches
        )

    def should_pause(self, stage_num: int, is_gate: bool) -> bool:
        """Determine if the pipeline should pause at this stage."""
        if self.mode == ResearchMode.ZERO_TOUCH:
            return False
        if self.mode == ResearchMode.AUTO_PILOT:
            return is_gate and self.config.pause_at_gates
        # CO_PILOT mode
        if self.config.pause_at_every_stage:
            return True
        return is_gate

    def present_stage_result(
        self,
        stage_num: int,
        stage_name: str,
        artifacts: list[str],
        status: str,
        error: str | None = None,
    ) -> str:
        """Format stage result summary for user review."""
        lines = [
            f"Stage {stage_num}: {stage_name}",
            f"Status: {status}",
        ]

        if error:
            lines.append(f"Error: {error}")

        if artifacts:
            lines.append(f"Artifacts: {', '.join(artifacts)}")

        lines.extend([
            "",
            "Available actions: approve, modify, retry, skip, branch, rollback",
        ])

        return "\n".join(lines)

    def request_feedback(
        self,
        stage_num: int,
        stage_name: str,
        summary: str,
    ) -> Feedback | None:
        """Request and wait for user feedback."""
        self.feedback_handler.write_feedback_request(
            stage=stage_num,
            stage_name=stage_name,
            summary=summary,
        )

        logger.info(
            "Co-Pilot: waiting for feedback on stage %d (%s)",
            stage_num,
            stage_name,
        )

        feedback = self.feedback_handler.wait_for_feedback(
            stage=stage_num,
            timeout_sec=self.config.feedback_timeout_sec,
        )

        self.feedback_handler.clear_request()
        return feedback

    def handle_feedback(
        self,
        feedback: Feedback,
    ) -> dict[str, Any]:
        """Process user feedback and return action instructions."""
        result: dict[str, Any] = {
            "action": feedback.action,
            "stage": feedback.stage,
        }

        if feedback.action == "approve":
            result["instruction"] = "continue"

        elif feedback.action == "modify":
            result["instruction"] = "apply_modifications"
            result["modifications"] = feedback.modifications or {}
            result["message"] = feedback.message

        elif feedback.action == "retry":
            result["instruction"] = "rerun_stage"

        elif feedback.action == "skip":
            result["instruction"] = "skip_stage"

        elif feedback.action == "branch":
            if self.config.allow_branching:
                branch_name = feedback.branch_name or f"branch_{feedback.stage}"
                try:
                    branch_path = self.branch_manager.create_branch(
                        branch_name, feedback.stage
                    )
                    result["instruction"] = "branch_created"
                    result["branch_name"] = branch_name
                    result["branch_path"] = branch_path
                except ValueError as exc:
                    result["instruction"] = "branch_failed"
                    result["error"] = str(exc)
            else:
                result["instruction"] = "branching_disabled"

        elif feedback.action == "rollback":
            result["instruction"] = "rollback"
            result["rollback_to"] = feedback.rollback_to

        else:
            result["instruction"] = "continue"

        return result

    @classmethod
    def from_config(
        cls,
        config: CoPilotConfig,
        run_dir: Path,
    ) -> CoPilotController | None:
        """Create a controller, or None if mode is zero-touch."""
        mode = ResearchMode(config.mode)
        if mode == ResearchMode.ZERO_TOUCH:
            return None
        return cls(config, run_dir)
