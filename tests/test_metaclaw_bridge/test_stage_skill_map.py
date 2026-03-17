"""Tests for stage-skill mapping module."""

from researchclaw.metaclaw_bridge.stage_skill_map import (
    STAGE_SKILL_MAP,
    LESSON_CATEGORY_TO_SKILL_CATEGORY,
    get_stage_config,
)


def test_all_23_stages_mapped():
    """All 23 pipeline stages should have a mapping entry."""
    expected_stages = [
        "topic_init", "problem_decompose", "search_strategy",
        "literature_collect", "literature_screen", "knowledge_extract",
        "synthesis", "hypothesis_gen", "experiment_design",
        "code_generation", "resource_planning", "experiment_run",
        "iterative_refine", "result_analysis", "research_decision",
        "paper_outline", "paper_draft", "peer_review",
        "paper_revision", "quality_gate", "knowledge_archive",
        "export_publish", "citation_verify",
    ]
    for stage in expected_stages:
        assert stage in STAGE_SKILL_MAP, f"Missing mapping for {stage}"


def test_stage_config_has_required_keys():
    """Each stage config should have task_type, skills, and top_k."""
    for stage_name, config in STAGE_SKILL_MAP.items():
        assert "task_type" in config, f"{stage_name} missing task_type"
        assert "skills" in config, f"{stage_name} missing skills"
        assert "top_k" in config, f"{stage_name} missing top_k"
        assert isinstance(config["skills"], list)
        assert isinstance(config["top_k"], int)
        assert config["top_k"] > 0


def test_get_stage_config_known():
    cfg = get_stage_config("hypothesis_gen")
    assert cfg["task_type"] == "research"
    assert "hypothesis-formulation" in cfg["skills"]


def test_get_stage_config_unknown_returns_default():
    cfg = get_stage_config("nonexistent_stage")
    assert cfg["task_type"] == "research"
    assert cfg["top_k"] == 4


def test_lesson_category_mapping_complete():
    """All lesson categories should map to a skill category."""
    expected = ["system", "experiment", "writing", "analysis", "literature", "pipeline"]
    for cat in expected:
        assert cat in LESSON_CATEGORY_TO_SKILL_CATEGORY
