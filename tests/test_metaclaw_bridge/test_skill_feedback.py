"""Tests for skill feedback tracking module."""

from pathlib import Path

from researchclaw.metaclaw_bridge.skill_feedback import (
    SkillEffectivenessRecord,
    SkillFeedbackStore,
    record_stage_skills,
)


def test_append_and_load(tmp_path):
    store = SkillFeedbackStore(tmp_path / "feedback.jsonl")
    rec = SkillEffectivenessRecord(
        skill_name="hypothesis-formulation",
        stage_name="hypothesis_gen",
        run_id="test-001",
        stage_success=True,
        timestamp="2026-03-15T00:00:00+00:00",
    )
    store.append(rec)

    loaded = store.load_all()
    assert len(loaded) == 1
    assert loaded[0].skill_name == "hypothesis-formulation"
    assert loaded[0].stage_success is True


def test_append_many(tmp_path):
    store = SkillFeedbackStore(tmp_path / "feedback.jsonl")
    records = [
        SkillEffectivenessRecord("skill-a", "stage-1", "run-1", True, "2026-01-01"),
        SkillEffectivenessRecord("skill-b", "stage-2", "run-1", False, "2026-01-01"),
    ]
    store.append_many(records)
    assert len(store.load_all()) == 2


def test_compute_stats(tmp_path):
    store = SkillFeedbackStore(tmp_path / "feedback.jsonl")
    records = [
        SkillEffectivenessRecord("skill-a", "s1", "r1", True, "t1"),
        SkillEffectivenessRecord("skill-a", "s2", "r1", False, "t1"),
        SkillEffectivenessRecord("skill-a", "s3", "r2", True, "t2"),
        SkillEffectivenessRecord("skill-b", "s1", "r1", False, "t1"),
    ]
    store.append_many(records)

    stats = store.compute_skill_stats()
    assert stats["skill-a"]["total"] == 3
    assert stats["skill-a"]["successes"] == 2
    assert abs(stats["skill-a"]["success_rate"] - 2 / 3) < 0.01
    assert stats["skill-b"]["total"] == 1
    assert stats["skill-b"]["success_rate"] == 0.0


def test_record_stage_skills(tmp_path):
    store = SkillFeedbackStore(tmp_path / "feedback.jsonl")
    record_stage_skills(
        store,
        stage_name="hypothesis_gen",
        run_id="test-002",
        stage_success=True,
        active_skills=["hypothesis-formulation", "research-gap-identification"],
    )
    loaded = store.load_all()
    assert len(loaded) == 2
    names = {r.skill_name for r in loaded}
    assert names == {"hypothesis-formulation", "research-gap-identification"}


def test_empty_store(tmp_path):
    store = SkillFeedbackStore(tmp_path / "nonexistent.jsonl")
    assert store.load_all() == []
    assert store.compute_skill_stats() == {}
