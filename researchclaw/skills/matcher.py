"""Skill-to-stage matching engine."""

from __future__ import annotations

import logging
import re

from researchclaw.skills.schema import STAGE_NAME_TO_NUMBER, Skill

logger = logging.getLogger(__name__)


def _tokenize(text: str) -> set[str]:
    """Extract lowercase tokens from text."""
    return set(re.findall(r"[a-z0-9_]+", text.lower()))


def _resolve_stage(stage: int | str) -> int:
    """Convert a stage name to its number, or pass through an int."""
    if isinstance(stage, int):
        return stage
    return STAGE_NAME_TO_NUMBER.get(stage, -1)


def match_skills(
    skills: list[Skill],
    context: str,
    stage: int | str,
    top_k: int = 3,
    *,
    fallback_matching: bool = True,
) -> list[Skill]:
    """Match skills to the current context and stage.

    Scoring:
    - Stage applicability (must match, or empty = all stages)
    - Keyword overlap with context
    - Description-based fallback at 0.5x discount (for skills without trigger_keywords)
    - Priority (lower = higher priority)

    Args:
        skills: Available skills to match against.
        context: Current task context text.
        stage: Current pipeline stage number or name.
        top_k: Maximum number of skills to return.
        fallback_matching: Enable description-based matching for skills
            without trigger_keywords.

    Returns:
        List of matched skills sorted by relevance.
    """
    stage_num = _resolve_stage(stage)
    context_tokens = _tokenize(context)
    scored: list[tuple[float, Skill]] = []

    for skill in skills:
        # Filter by stage applicability
        if skill.applicable_stages and stage_num not in skill.applicable_stages:
            continue

        # Keyword matching score
        keyword_score = 0.0
        has_keywords = bool(skill.trigger_keywords)
        for kw in skill.trigger_keywords:
            kw_tokens = _tokenize(kw)
            if kw_tokens & context_tokens:
                keyword_score += 1.0

        # Description-based fallback for external skills without keywords
        if keyword_score == 0.0 and not has_keywords and fallback_matching:
            desc_tokens = _tokenize(skill.description)
            overlap = len(desc_tokens & context_tokens)
            if overlap > 0:
                keyword_score = overlap * 0.5  # 0.5x discount
                max_possible = max(len(desc_tokens), 1)
                normalized_kw = keyword_score / max_possible
            else:
                continue
        elif keyword_score == 0.0:
            continue
        else:
            max_possible = max(len(skill.trigger_keywords), 1)
            normalized_kw = keyword_score / max_possible

        # Priority adjustment (priority 1 → boost 0.5, priority 10 → boost 0.0)
        priority_boost = (10 - skill.priority) / 20.0

        total_score = normalized_kw + priority_boost
        scored.append((total_score, skill))

    scored.sort(key=lambda x: (-x[0], x[1].priority))
    return [skill for _, skill in scored[:top_k]]


def format_skills_for_prompt(skills: list[Skill], max_chars: int = 4000) -> str:
    """Format matched skills as prompt injection text.

    Uses ``skill.body`` as primary content.  Truncates long bodies
    (common with external skills) to ``max_chars / len(skills)`` per skill.

    Args:
        skills: List of matched skills.
        max_chars: Maximum character limit.

    Returns:
        Formatted string for LLM prompt injection.
    """
    if not skills:
        return ""

    per_skill_budget = max_chars // max(len(skills), 1)
    parts: list[str] = []
    total_len = 0

    for skill in skills:
        content = skill.body or skill.prompt_template
        # Truncate long bodies
        if len(content) > per_skill_budget:
            content = content[:per_skill_budget - 20] + "\n\n[... truncated]"

        section = f"### {skill.name} ({skill.category})\n{content}"
        if skill.code_template:
            section += f"\n**Code Template:**\n```python\n{skill.code_template}\n```"
        if skill.references:
            section += "\n**References:** " + "; ".join(skill.references)

        if total_len + len(section) > max_chars:
            break
        parts.append(section)
        total_len += len(section)

    return "\n\n".join(parts)
