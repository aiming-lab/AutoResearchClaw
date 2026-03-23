"""Convert AutoResearchClaw failure lessons into MetaClaw skills.

Analyses high-severity lessons from the evolution store and uses an LLM
to generate actionable MetaClaw skill files that prevent future recurrence.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import TYPE_CHECKING

from researchclaw.metaclaw_bridge.stage_skill_map import (
    LESSON_CATEGORY_TO_SKILL_CATEGORY,
)

if TYPE_CHECKING:
    from researchclaw.evolution import LessonEntry
    from researchclaw.llm.client import LLMClient

logger = logging.getLogger(__name__)

_SEVERITY_ORDER = {"info": 0, "warning": 1, "error": 2, "critical": 3}

_CONVERSION_PROMPT_SYSTEM = """\
You are a skill designer for an AI agent system. Your job is to convert
failure lessons from an automated research pipeline into reusable skill
guides that help the agent avoid the same mistakes in the future.

Each skill must include:
- A descriptive name (lowercase-hyphenated, prefixed with "arc-")
- A one-line description of when to use the skill
- A category from: {categories}
- Markdown content with numbered steps and an anti-pattern section

Output a JSON array of skill objects. Each object has:
  "name": "arc-<slug>",
  "description": "<when to use>",
  "category": "<category>",
  "content": "<markdown body>"
"""

_CONVERSION_PROMPT_USER = """\
The following failure lessons were extracted from automated research runs.
Please generate {max_skills} reusable skills to address these failures.

## Failure Lessons

{lessons_text}

## Existing Skills (do not duplicate)

{existing_skills}

Return ONLY a JSON array. No extra text.
"""


def _format_lessons(lessons: list[LessonEntry]) -> str:
    """Format lessons into a text block for the LLM prompt."""
    parts: list[str] = []
    for i, lesson in enumerate(lessons, 1):
        parts.append(
            f"{i}. [{lesson.severity}] [{lesson.category}] "
            f"Stage {lesson.stage_name}: {lesson.description}"
        )
    return "\n".join(parts)


def _list_existing_skill_names(skills_dir: Path) -> list[str]:
    """List all existing skill names in the MetaClaw skills directory."""
    try:
        if not skills_dir.exists():
            return []
        return [d.name for d in skills_dir.iterdir() if d.is_dir()]
    except OSError:
        return []


def _parse_skills_response(text: str) -> list[dict[str, str]]:
    """Parse the LLM response into a list of skill dicts."""
    # Strip markdown code fences if present
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```\w*\n?", "", text)
        text = re.sub(r"\n?```\s*$", "", text)
    try:
        data = json.loads(text)
        # Handle both bare array and {"skills": [...]} wrapper
        if isinstance(data, dict):
            for key in ("skills", "results", "data"):
                if key in data and isinstance(data[key], list):
                    data = data[key]
                    break
        if isinstance(data, list):
            return [
                s
                for s in data
                if isinstance(s, dict)
                and all(k in s for k in ("name", "description", "category", "content"))
            ]
    except json.JSONDecodeError:
        logger.warning("Failed to parse skill evolution response as JSON")
    return []


def _write_skill(skills_dir: Path, skill: dict[str, str]) -> Path | None:
    """Write a single skill to disk as a SKILL.md file."""
    name = skill["name"]
    # Sanitize name
    name = re.sub(r"[^a-z0-9-]", "-", name.lower()).strip("-")
    if not name:
        return None

    skill_dir = skills_dir / name
    skill_dir.mkdir(parents=True, exist_ok=True)
    skill_path = skill_dir / "SKILL.md"

    content = f"---\nname: {name}\n"
    content += f"description: {skill['description']}\n"
    content += "metadata:\n"
    content += f"  category: {skill['category']}\n"
    content += f"---\n{skill['content']}\n"

    skill_path.write_text(content, encoding="utf-8")
    logger.info("Created new MetaClaw skill: %s", name)
    return skill_path


def _severity_at_least(severity: str, min_severity: str) -> bool:
    """Check if severity meets or exceeds the minimum threshold."""
    return _SEVERITY_ORDER.get(severity, 0) >= _SEVERITY_ORDER.get(min_severity, 0)


def convert_lessons_to_skills(
    lessons: list[LessonEntry],
    llm: LLMClient,
    skills_dir: str | Path,
    *,
    min_severity: str = "warning",
    max_skills: int = 3,
) -> list[str]:
    """Convert failure lessons into MetaClaw skills.

    Args:
        lessons: Lessons to convert (will be filtered by severity).
        llm: LLM client for generating skills.
        skills_dir: Path to MetaClaw skills directory.
        min_severity: Minimum severity to include ("info", "warning", "error", "critical").
        max_skills: Maximum number of skills to generate.

    Returns:
        List of created skill names.
    """
    if not lessons:
        return []

    # Filter by severity threshold (>= min_severity)
    filtered = [
        l for l in lessons
        if _severity_at_least(getattr(l, "severity", ""), min_severity)
    ]
    if not filtered:
        logger.info(
            "No lessons at severity >= %s (total lessons: %d)", min_severity, len(lessons)
        )
        return []

    logger.info(
        "Converting %d lessons (severity >= %s) to skills", len(filtered), min_severity
    )

    skills_path = Path(skills_dir).expanduser()
    skills_path.mkdir(parents=True, exist_ok=True)

    categories = ", ".join(sorted(set(LESSON_CATEGORY_TO_SKILL_CATEGORY.values())))
    existing = _list_existing_skill_names(skills_path)

    system = _CONVERSION_PROMPT_SYSTEM.format(categories=categories)
    user = _CONVERSION_PROMPT_USER.format(
        max_skills=max_skills,
        lessons_text=_format_lessons(filtered),
        existing_skills=", ".join(existing[:50]) if existing else "(none)",
    )

    try:
        resp = llm.chat(
            [{"role": "user", "content": user}],
            system=system,
            json_mode=True,
            max_tokens=3000,
        )
    except Exception:
        logger.warning("LLM call for lesson-to-skill conversion failed", exc_info=True)
        return []

    parsed = _parse_skills_response(resp.content)
    if not parsed:
        logger.warning("No valid skills parsed from LLM response")
        return []

    created: list[str] = []
    for skill in parsed[:max_skills]:
        # Map category using our mapping if needed
        if skill["category"] not in LESSON_CATEGORY_TO_SKILL_CATEGORY.values():
            lesson_cat = skill.get("category", "pipeline")
            skill["category"] = LESSON_CATEGORY_TO_SKILL_CATEGORY.get(
                lesson_cat, "research"
            )
        path = _write_skill(skills_path, skill)
        if path is not None:
            created.append(skill["name"])

    return created
