"""Tests for PRM quality gate module."""

from unittest.mock import patch, MagicMock

from researchclaw.metaclaw_bridge.prm_gate import (
    ResearchPRMGate,
    _GATE_INSTRUCTIONS,
)


def test_gate_instructions_cover_expected_stages():
    """PRM gate instructions should cover key gate stages."""
    assert 5 in _GATE_INSTRUCTIONS
    assert 9 in _GATE_INSTRUCTIONS
    assert 15 in _GATE_INSTRUCTIONS
    assert 20 in _GATE_INSTRUCTIONS


def test_should_gate():
    gate = ResearchPRMGate(
        api_base="http://test",
        api_key="test",
    )
    assert gate.should_gate(5) is True
    assert gate.should_gate(9) is True
    assert gate.should_gate(15) is True
    assert gate.should_gate(20) is True
    assert gate.should_gate(1) is False
    assert gate.should_gate(10) is False


def test_from_bridge_config_disabled():
    """Should return None when PRM is not enabled."""
    config = MagicMock()
    config.enabled = False
    assert ResearchPRMGate.from_bridge_config(config) is None


def test_from_bridge_config_enabled():
    """Should create a gate when properly configured."""
    config = MagicMock()
    config.enabled = True
    config.api_base = "http://test"
    config.api_key = "test-key"
    config.api_key_env = ""
    config.model = "gpt-5.4"
    config.votes = 3
    config.temperature = 0.6

    gate = ResearchPRMGate.from_bridge_config(config)
    assert gate is not None
    assert gate.api_base == "http://test"
    assert gate.votes == 3


@patch("researchclaw.metaclaw_bridge.prm_gate._single_judge_call")
def test_evaluate_stage_majority_pass(mock_call):
    """Should return 1.0 when majority votes pass."""
    mock_call.side_effect = [1.0, 1.0, -1.0]
    gate = ResearchPRMGate(
        api_base="http://test",
        api_key="test",
        votes=3,
    )
    score = gate.evaluate_stage(20, "This is a good paper.")
    assert score == 1.0


@patch("researchclaw.metaclaw_bridge.prm_gate._single_judge_call")
def test_evaluate_stage_majority_fail(mock_call):
    """Should return -1.0 when majority votes fail."""
    mock_call.side_effect = [-1.0, -1.0, 1.0]
    gate = ResearchPRMGate(
        api_base="http://test",
        api_key="test",
        votes=3,
    )
    score = gate.evaluate_stage(20, "This paper has critical issues.")
    assert score == -1.0


@patch("researchclaw.metaclaw_bridge.prm_gate._single_judge_call")
def test_evaluate_stage_all_failed(mock_call):
    """Should return 0.0 when all judge calls fail."""
    mock_call.side_effect = [None, None, None]
    gate = ResearchPRMGate(
        api_base="http://test",
        api_key="test",
        votes=3,
    )
    score = gate.evaluate_stage(20, "test")
    assert score == 0.0
