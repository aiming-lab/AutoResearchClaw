"""Tests for lesson-to-skill conversion module."""

import json
import tempfile
from pathlib import Path

from researchclaw.metaclaw_bridge.lesson_to_skill import (
    _format_lessons,
    _list_existing_skill_names,
    _parse_skills_response,
    _write_skill,
)
from researchclaw.evolution import LessonEntry


def _make_lesson(stage: str = "experiment_run", severity: str = "error") -> LessonEntry:
    return LessonEntry(
        stage_name=stage,
        stage_num=12,
        category="experiment",
        severity=severity,
        description="Metric NaN detected in loss computation",
        timestamp="2026-03-15T00:00:00+00:00",
        run_id="test-001",
    )


def test_format_lessons():
    lessons = [_make_lesson(), _make_lesson("code_generation")]
    text = _format_lessons(lessons)
    assert "experiment_run" in text
    assert "code_generation" in text
    assert "NaN" in text


def test_list_existing_skills(tmp_path):
    (tmp_path / "skill-a").mkdir()
    (tmp_path / "skill-b").mkdir()
    (tmp_path / "not-a-skill.txt").write_text("x")
    names = _list_existing_skill_names(tmp_path)
    assert "skill-a" in names
    assert "skill-b" in names
    assert "not-a-skill.txt" not in names


def test_list_existing_skills_missing_dir():
    names = _list_existing_skill_names(Path("/nonexistent/dir"))
    assert names == []


def test_parse_skills_response_valid():
    response = json.dumps([
        {
            "name": "arc-fix-nan",
            "description": "Prevent NaN in loss",
            "category": "coding",
            "content": "# Fix NaN\n1. Check inputs\n2. Use grad clipping",
        }
    ])
    parsed = _parse_skills_response(response)
    assert len(parsed) == 1
    assert parsed[0]["name"] == "arc-fix-nan"


def test_parse_skills_response_with_code_fence():
    response = "```json\n" + json.dumps([
        {
            "name": "arc-test",
            "description": "test",
            "category": "coding",
            "content": "test content",
        }
    ]) + "\n```"
    parsed = _parse_skills_response(response)
    assert len(parsed) == 1


def test_parse_skills_response_invalid():
    assert _parse_skills_response("not json") == []
    assert _parse_skills_response("[]") == []


def test_write_skill(tmp_path):
    skill = {
        "name": "arc-test-skill",
        "description": "A test skill",
        "category": "coding",
        "content": "# Test\n1. Do something",
    }
    path = _write_skill(tmp_path, skill)
    assert path is not None
    assert path.exists()
    content = path.read_text()
    assert "name: arc-test-skill" in content
    assert "category: coding" in content
    assert "# Test" in content
