"""Tests for robotics & control domain adapter.

Covers adapter dispatch, prompt block generation, and integration
with the existing domain detection and profile system.
"""

from __future__ import annotations

import pytest

from researchclaw.domains.detector import (
    get_profile,
    _keyword_detect,
    _profile_cache,
)
from researchclaw.domains.prompt_adapter import (
    MLPromptAdapter,
    GenericPromptAdapter,
    get_adapter,
)


# ---------------------------------------------------------------------------
# Profile sanity
# ---------------------------------------------------------------------------


class TestRoboticsProfile:
    def setup_method(self):
        _profile_cache.clear()

    def test_profile_exists(self):
        profile = get_profile("robotics_control")
        assert profile is not None
        assert profile.domain_id == "robotics_control"

    def test_profile_fields(self):
        profile = get_profile("robotics_control")
        assert profile is not None
        assert profile.experiment_paradigm == "comparison"
        assert "gymnasium" in profile.core_libraries
        assert "stable-baselines3" in profile.core_libraries
        assert profile.gpu_required is True

    def test_profile_baselines(self):
        profile = get_profile("robotics_control")
        assert profile is not None
        baselines = profile.standard_baselines
        assert any("PPO" in b for b in baselines)
        assert any("SAC" in b for b in baselines)


# ---------------------------------------------------------------------------
# Keyword detection
# ---------------------------------------------------------------------------


class TestRoboticsKeywordDetection:
    def test_robot_keyword(self):
        assert _keyword_detect("robot manipulation task") == "robotics_control"

    def test_mujoco(self):
        assert _keyword_detect("locomotion in MuJoCo") == "robotics_control"

    def test_pybullet(self):
        assert _keyword_detect("grasping policy with PyBullet") == "robotics_control"


# ---------------------------------------------------------------------------
# Adapter dispatch
# ---------------------------------------------------------------------------


class TestRoboticsAdapter:
    def test_gets_robotics_adapter(self):
        profile = get_profile("robotics_control")
        if profile is None:
            pytest.skip("robotics_control profile not found")
        adapter = get_adapter(profile)
        assert not isinstance(adapter, MLPromptAdapter)
        # Before this contribution it would fall back to GenericPromptAdapter
        from researchclaw.domains.adapters.robotics import (
            RoboticsPromptAdapter,
        )
        assert isinstance(adapter, RoboticsPromptAdapter)

    def test_code_generation_blocks_nonempty(self):
        profile = get_profile("robotics_control")
        if profile is None:
            pytest.skip("robotics_control profile not found")
        adapter = get_adapter(profile)
        blocks = adapter.get_code_generation_blocks({})
        assert blocks.code_generation_hints
        assert blocks.dataset_guidance
        assert blocks.output_format_guidance

    def test_experiment_design_mentions_baselines(self):
        profile = get_profile("robotics_control")
        if profile is None:
            pytest.skip("robotics_control profile not found")
        adapter = get_adapter(profile)
        blocks = adapter.get_experiment_design_blocks({})
        assert "PPO" in blocks.experiment_design_context
        assert "SAC" in blocks.experiment_design_context

    def test_result_analysis_mentions_return(self):
        profile = get_profile("robotics_control")
        if profile is None:
            pytest.skip("robotics_control profile not found")
        adapter = get_adapter(profile)
        blocks = adapter.get_result_analysis_blocks({})
        assert "return" in blocks.result_analysis_hints.lower()

    def test_blueprint_context(self):
        profile = get_profile("robotics_control")
        if profile is None:
            pytest.skip("robotics_control profile not found")
        adapter = get_adapter(profile)
        ctx = adapter.get_blueprint_context()
        if profile.typical_file_structure:
            assert "agent.py" in ctx or "train.py" in ctx
