"""Tests for researchclaw.copilot — Interactive Co-Pilot Mode (Agent D2).

30+ tests covering modes, feedback, branching, and controller.
"""

from __future__ import annotations

import json
import shutil
import time
from datetime import date, timedelta
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from researchclaw.copilot.modes import ResearchMode
from researchclaw.copilot.feedback import (
    FEEDBACK_ACTIONS,
    Feedback,
    FeedbackHandler,
)
from researchclaw.copilot.branching import BranchManager
from researchclaw.copilot.controller import CoPilotController
from researchclaw.config import CoPilotConfig


# ===================================================================
# ResearchMode tests
# ===================================================================


class TestResearchMode:
    def test_all_modes(self):
        assert ResearchMode.CO_PILOT.value == "co-pilot"
        assert ResearchMode.AUTO_PILOT.value == "auto-pilot"
        assert ResearchMode.ZERO_TOUCH.value == "zero-touch"

    def test_from_value(self):
        assert ResearchMode("co-pilot") == ResearchMode.CO_PILOT
        assert ResearchMode("auto-pilot") == ResearchMode.AUTO_PILOT
        assert ResearchMode("zero-touch") == ResearchMode.ZERO_TOUCH

    def test_invalid_mode_raises(self):
        with pytest.raises(ValueError):
            ResearchMode("invalid")

    def test_mode_count(self):
        assert len(ResearchMode) == 3


# ===================================================================
# Feedback tests
# ===================================================================


class TestFeedback:
    def test_feedback_actions_defined(self):
        expected = {"approve", "modify", "retry", "skip", "discuss", "branch", "rollback"}
        assert FEEDBACK_ACTIONS == expected

    def test_feedback_frozen(self):
        fb = Feedback(action="approve", stage=5)
        with pytest.raises(AttributeError):
            fb.action = "retry"  # type: ignore[misc]

    def test_feedback_defaults(self):
        fb = Feedback(action="approve", stage=1)
        assert fb.message == ""
        assert fb.modifications is None
        assert fb.branch_name == ""
        assert fb.rollback_to is None

    def test_feedback_with_modifications(self):
        fb = Feedback(
            action="modify",
            stage=5,
            message="Update hypothesis",
            modifications={"hypothesis": "new hypothesis"},
        )
        assert fb.modifications == {"hypothesis": "new hypothesis"}


# ===================================================================
# FeedbackHandler tests
# ===================================================================


class TestFeedbackHandler:
    def test_write_feedback_request(self, tmp_path: Path):
        handler = FeedbackHandler(tmp_path)
        request_path = handler.write_feedback_request(
            stage=5,
            stage_name="LITERATURE_SCREEN",
            summary="10 papers screened",
        )
        assert request_path.exists()
        data = json.loads(request_path.read_text(encoding="utf-8"))
        assert data["stage"] == 5
        assert data["stage_name"] == "LITERATURE_SCREEN"
        assert data["status"] == "waiting"
        assert isinstance(data["options"], list)

    def test_read_feedback_response_valid(self, tmp_path: Path):
        handler = FeedbackHandler(tmp_path)
        response = {
            "action": "approve",
            "stage": 5,
            "message": "Looks good",
        }
        resp_path = tmp_path / "copilot_feedback_response.json"
        resp_path.write_text(json.dumps(response), encoding="utf-8")
        fb = handler.read_feedback_response()
        assert fb is not None
        assert fb.action == "approve"
        assert fb.stage == 5
        assert fb.message == "Looks good"

    def test_read_feedback_response_invalid_action(self, tmp_path: Path):
        handler = FeedbackHandler(tmp_path)
        response = {"action": "invalid_action", "stage": 5}
        resp_path = tmp_path / "copilot_feedback_response.json"
        resp_path.write_text(json.dumps(response), encoding="utf-8")
        fb = handler.read_feedback_response()
        assert fb is None

    def test_read_feedback_response_missing(self, tmp_path: Path):
        handler = FeedbackHandler(tmp_path)
        assert handler.read_feedback_response() is None

    def test_read_feedback_response_malformed(self, tmp_path: Path):
        handler = FeedbackHandler(tmp_path)
        resp_path = tmp_path / "copilot_feedback_response.json"
        resp_path.write_text("{invalid json", encoding="utf-8")
        assert handler.read_feedback_response() is None

    def test_read_feedback_response_with_rollback(self, tmp_path: Path):
        handler = FeedbackHandler(tmp_path)
        response = {
            "action": "rollback",
            "stage": 15,
            "rollback_to": 8,
        }
        resp_path = tmp_path / "copilot_feedback_response.json"
        resp_path.write_text(json.dumps(response), encoding="utf-8")
        fb = handler.read_feedback_response()
        assert fb is not None
        assert fb.action == "rollback"
        assert fb.rollback_to == 8

    def test_read_feedback_response_branch(self, tmp_path: Path):
        handler = FeedbackHandler(tmp_path)
        response = {
            "action": "branch",
            "stage": 9,
            "branch_name": "alt_experiment",
        }
        resp_path = tmp_path / "copilot_feedback_response.json"
        resp_path.write_text(json.dumps(response), encoding="utf-8")
        fb = handler.read_feedback_response()
        assert fb is not None
        assert fb.branch_name == "alt_experiment"

    def test_clear_request(self, tmp_path: Path):
        handler = FeedbackHandler(tmp_path)
        handler.write_feedback_request(1, "TOPIC_INIT", "Done")
        handler.clear_request()
        assert not (tmp_path / "copilot_feedback_request.json").exists()

    def test_clear_request_no_file(self, tmp_path: Path):
        handler = FeedbackHandler(tmp_path)
        handler.clear_request()  # should not raise

    def test_wait_for_feedback_timeout(self, tmp_path: Path):
        handler = FeedbackHandler(tmp_path)
        result = handler.wait_for_feedback(stage=1, timeout_sec=0, poll_interval_sec=0.01)
        assert result is None

    def test_wait_for_feedback_finds_response(self, tmp_path: Path):
        handler = FeedbackHandler(tmp_path)
        # Pre-clear any stale response (wait_for_feedback clears first)
        # Then write a response matching stage
        response = {"action": "approve", "stage": 5}
        resp_path = tmp_path / "copilot_feedback_response.json"

        def write_response():
            """Simulate delayed response writing."""
            time.sleep(0.05)
            resp_path.write_text(json.dumps(response), encoding="utf-8")

        import threading
        t = threading.Thread(target=write_response)
        t.start()
        fb = handler.wait_for_feedback(stage=5, timeout_sec=2, poll_interval_sec=0.02)
        t.join()
        assert fb is not None
        assert fb.action == "approve"


# ===================================================================
# BranchManager tests
# ===================================================================


class TestBranchManager:
    def test_create_branch(self, tmp_path: Path):
        # Create stage dirs
        (tmp_path / "stage-01").mkdir()
        (tmp_path / "stage-01" / "output.json").write_text("{}")
        (tmp_path / "stage-02").mkdir()
        (tmp_path / "stage-02" / "result.txt").write_text("ok")

        bm = BranchManager(tmp_path, max_branches=3)
        branch_path = bm.create_branch("exp_alt", from_stage=2)

        assert Path(branch_path).exists()
        assert (Path(branch_path) / "stage-01" / "output.json").exists()
        assert (Path(branch_path) / "stage-02" / "result.txt").exists()
        assert (Path(branch_path) / "branch_meta.json").exists()

        meta = json.loads(
            (Path(branch_path) / "branch_meta.json").read_text(encoding="utf-8")
        )
        assert meta["name"] == "exp_alt"
        assert meta["from_stage"] == 2

    def test_create_branch_max_reached(self, tmp_path: Path):
        bm = BranchManager(tmp_path, max_branches=1)
        bm.create_branch("b1", from_stage=1)
        with pytest.raises(ValueError, match="Maximum branches"):
            bm.create_branch("b2", from_stage=1)

    def test_create_branch_duplicate_name(self, tmp_path: Path):
        bm = BranchManager(tmp_path, max_branches=5)
        bm.create_branch("dup", from_stage=1)
        with pytest.raises(ValueError, match="already exists"):
            bm.create_branch("dup", from_stage=1)

    def test_list_branches_empty(self, tmp_path: Path):
        bm = BranchManager(tmp_path)
        assert bm.list_branches() == []

    def test_list_branches(self, tmp_path: Path):
        bm = BranchManager(tmp_path, max_branches=5)
        bm.create_branch("alpha", from_stage=1)
        bm.create_branch("beta", from_stage=2)
        branches = bm.list_branches()
        assert len(branches) == 2
        names = {b["name"] for b in branches}
        assert names == {"alpha", "beta"}

    def test_switch_branch(self, tmp_path: Path):
        bm = BranchManager(tmp_path, max_branches=3)
        bm.create_branch("test_branch", from_stage=1)
        path = bm.switch_branch("test_branch")
        assert path.exists()

    def test_switch_branch_nonexistent(self, tmp_path: Path):
        bm = BranchManager(tmp_path)
        with pytest.raises(ValueError, match="does not exist"):
            bm.switch_branch("nonexistent")

    def test_delete_branch(self, tmp_path: Path):
        bm = BranchManager(tmp_path, max_branches=3)
        bm.create_branch("doomed", from_stage=1)
        assert len(bm.list_branches()) == 1
        bm.delete_branch("doomed")
        assert len(bm.list_branches()) == 0

    def test_delete_branch_nonexistent(self, tmp_path: Path):
        bm = BranchManager(tmp_path)
        with pytest.raises(ValueError, match="does not exist"):
            bm.delete_branch("ghost")

    def test_compare_branches(self, tmp_path: Path):
        bm = BranchManager(tmp_path, max_branches=5)
        (tmp_path / "stage-01").mkdir()
        (tmp_path / "stage-02").mkdir()
        bm.create_branch("a", from_stage=2)
        bm.create_branch("b", from_stage=1)
        result = bm.compare_branches("a", "b")
        assert result["branch_a"] == "a"
        assert result["stages_a"] == 2
        assert result["stages_b"] == 1

    def test_compare_branches_nonexistent(self, tmp_path: Path):
        bm = BranchManager(tmp_path, max_branches=3)
        bm.create_branch("real", from_stage=1)
        result = bm.compare_branches("real", "fake")
        assert "error" in result

    def test_count_stages(self, tmp_path: Path):
        (tmp_path / "stage-01").mkdir()
        (tmp_path / "stage-02").mkdir()
        (tmp_path / "other_dir").mkdir()
        assert BranchManager._count_stages(tmp_path) == 2


# ===================================================================
# CoPilotController tests
# ===================================================================


class TestCoPilotController:
    def _make_config(self, **overrides) -> CoPilotConfig:
        defaults = {
            "mode": "co-pilot",
            "pause_at_gates": True,
            "pause_at_every_stage": False,
            "feedback_timeout_sec": 3600,
            "allow_branching": True,
            "max_branches": 3,
        }
        defaults.update(overrides)
        return CoPilotConfig(**defaults)

    def test_should_pause_zero_touch(self, tmp_path: Path):
        config = self._make_config(mode="zero-touch")
        ctrl = CoPilotController(config, tmp_path)
        assert ctrl.should_pause(5, is_gate=True) is False
        assert ctrl.should_pause(1, is_gate=False) is False

    def test_should_pause_auto_pilot_gate(self, tmp_path: Path):
        config = self._make_config(mode="auto-pilot")
        ctrl = CoPilotController(config, tmp_path)
        assert ctrl.should_pause(5, is_gate=True) is True
        assert ctrl.should_pause(1, is_gate=False) is False

    def test_should_pause_auto_pilot_gates_disabled(self, tmp_path: Path):
        config = self._make_config(mode="auto-pilot", pause_at_gates=False)
        ctrl = CoPilotController(config, tmp_path)
        assert ctrl.should_pause(5, is_gate=True) is False

    def test_should_pause_copilot_every_stage(self, tmp_path: Path):
        config = self._make_config(mode="co-pilot", pause_at_every_stage=True)
        ctrl = CoPilotController(config, tmp_path)
        assert ctrl.should_pause(1, is_gate=False) is True
        assert ctrl.should_pause(5, is_gate=True) is True

    def test_should_pause_copilot_gates_only(self, tmp_path: Path):
        config = self._make_config(mode="co-pilot", pause_at_every_stage=False)
        ctrl = CoPilotController(config, tmp_path)
        assert ctrl.should_pause(5, is_gate=True) is True
        assert ctrl.should_pause(1, is_gate=False) is False

    def test_present_stage_result(self, tmp_path: Path):
        config = self._make_config()
        ctrl = CoPilotController(config, tmp_path)
        summary = ctrl.present_stage_result(
            stage_num=5,
            stage_name="LITERATURE_SCREEN",
            artifacts=["screen_report.json"],
            status="done",
        )
        assert "Stage 5: LITERATURE_SCREEN" in summary
        assert "Status: done" in summary
        assert "screen_report.json" in summary

    def test_present_stage_result_with_error(self, tmp_path: Path):
        config = self._make_config()
        ctrl = CoPilotController(config, tmp_path)
        summary = ctrl.present_stage_result(
            stage_num=12,
            stage_name="EXPERIMENT_RUN",
            artifacts=[],
            status="failed",
            error="CUDA out of memory",
        )
        assert "Error: CUDA out of memory" in summary

    def test_handle_feedback_approve(self, tmp_path: Path):
        config = self._make_config()
        ctrl = CoPilotController(config, tmp_path)
        fb = Feedback(action="approve", stage=5)
        result = ctrl.handle_feedback(fb)
        assert result["instruction"] == "continue"

    def test_handle_feedback_modify(self, tmp_path: Path):
        config = self._make_config()
        ctrl = CoPilotController(config, tmp_path)
        fb = Feedback(
            action="modify",
            stage=5,
            message="Change approach",
            modifications={"key": "value"},
        )
        result = ctrl.handle_feedback(fb)
        assert result["instruction"] == "apply_modifications"
        assert result["modifications"] == {"key": "value"}

    def test_handle_feedback_retry(self, tmp_path: Path):
        config = self._make_config()
        ctrl = CoPilotController(config, tmp_path)
        fb = Feedback(action="retry", stage=12)
        result = ctrl.handle_feedback(fb)
        assert result["instruction"] == "rerun_stage"

    def test_handle_feedback_skip(self, tmp_path: Path):
        config = self._make_config()
        ctrl = CoPilotController(config, tmp_path)
        fb = Feedback(action="skip", stage=21)
        result = ctrl.handle_feedback(fb)
        assert result["instruction"] == "skip_stage"

    def test_handle_feedback_branch(self, tmp_path: Path):
        config = self._make_config(allow_branching=True)
        ctrl = CoPilotController(config, tmp_path)
        fb = Feedback(action="branch", stage=9, branch_name="alt_design")
        result = ctrl.handle_feedback(fb)
        assert result["instruction"] == "branch_created"
        assert result["branch_name"] == "alt_design"

    def test_handle_feedback_branch_disabled(self, tmp_path: Path):
        config = self._make_config(allow_branching=False)
        ctrl = CoPilotController(config, tmp_path)
        fb = Feedback(action="branch", stage=9)
        result = ctrl.handle_feedback(fb)
        assert result["instruction"] == "branching_disabled"

    def test_handle_feedback_branch_max_reached(self, tmp_path: Path):
        config = self._make_config(allow_branching=True, max_branches=1)
        ctrl = CoPilotController(config, tmp_path)
        # Create first branch
        fb1 = Feedback(action="branch", stage=1, branch_name="b1")
        ctrl.handle_feedback(fb1)
        # Second branch should fail
        fb2 = Feedback(action="branch", stage=2, branch_name="b2")
        result = ctrl.handle_feedback(fb2)
        assert result["instruction"] == "branch_failed"

    def test_handle_feedback_rollback(self, tmp_path: Path):
        config = self._make_config()
        ctrl = CoPilotController(config, tmp_path)
        fb = Feedback(action="rollback", stage=15, rollback_to=8)
        result = ctrl.handle_feedback(fb)
        assert result["instruction"] == "rollback"
        assert result["rollback_to"] == 8

    def test_handle_feedback_unknown_action(self, tmp_path: Path):
        config = self._make_config()
        ctrl = CoPilotController(config, tmp_path)
        # Construct with a technically valid action but unhandled by match
        fb = Feedback(action="discuss", stage=1)
        result = ctrl.handle_feedback(fb)
        assert result["instruction"] == "continue"

    def test_from_config_zero_touch_returns_none(self, tmp_path: Path):
        config = self._make_config(mode="zero-touch")
        ctrl = CoPilotController.from_config(config, tmp_path)
        assert ctrl is None

    def test_from_config_copilot_returns_controller(self, tmp_path: Path):
        config = self._make_config(mode="co-pilot")
        ctrl = CoPilotController.from_config(config, tmp_path)
        assert ctrl is not None
        assert isinstance(ctrl, CoPilotController)

    def test_from_config_auto_pilot_returns_controller(self, tmp_path: Path):
        config = self._make_config(mode="auto-pilot")
        ctrl = CoPilotController.from_config(config, tmp_path)
        assert ctrl is not None

    def test_handle_feedback_branch_default_name(self, tmp_path: Path):
        config = self._make_config(allow_branching=True)
        ctrl = CoPilotController(config, tmp_path)
        fb = Feedback(action="branch", stage=9)  # no branch_name
        result = ctrl.handle_feedback(fb)
        assert result["instruction"] == "branch_created"
        assert result["branch_name"] == "branch_9"
