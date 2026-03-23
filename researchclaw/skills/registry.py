"""Skill registry — central hub for loading and querying skills."""

from __future__ import annotations

import logging
from pathlib import Path

from researchclaw.skills.loader import load_skills_from_directory
from researchclaw.skills.matcher import format_skills_for_prompt, match_skills
from researchclaw.skills.schema import Skill

logger = logging.getLogger(__name__)

# Default builtin directory relative to this file
_BUILTIN_DIR = Path(__file__).parent / "builtin"


class SkillRegistry:
    """Central registry for managing and querying skills.

    Loads builtin skills on init, then optionally loads custom and
    external skills from user-specified directories.
    """

    def __init__(
        self,
        builtin_dir: str | Path = "",
        custom_dirs: tuple[str, ...] | list[str] = (),
        external_dirs: tuple[str, ...] | list[str] = (),
        auto_match: bool = True,
        max_skills_per_stage: int = 3,
        fallback_matching: bool = True,
    ) -> None:
        self._skills: dict[str, Skill] = {}
        self._auto_match = auto_match
        self._max_skills = max_skills_per_stage
        self._fallback_matching = fallback_matching

        # Load builtin skills
        builtin = Path(builtin_dir) if builtin_dir else _BUILTIN_DIR
        self._load_from_dir(builtin)

        # Load custom skills
        for d in custom_dirs:
            self._load_from_dir(Path(d))

        # Load external skills (same mechanism)
        for d in external_dirs:
            self._load_from_dir(Path(d))

    def _load_from_dir(self, directory: Path) -> None:
        """Load skills from a directory and register them."""
        skills = load_skills_from_directory(directory)
        for skill in skills:
            self.register(skill)

    def register(self, skill: Skill) -> None:
        """Register a skill. Overwrites existing skill with same name.

        Args:
            skill: The skill to register.
        """
        self._skills[skill.name] = skill
        logger.debug("Registered skill: %s", skill.name)

    def unregister(self, skill_id: str) -> bool:
        """Remove a skill from the registry.

        Args:
            skill_id: The name/ID of the skill to remove.

        Returns:
            True if skill was found and removed.
        """
        if skill_id in self._skills:
            del self._skills[skill_id]
            return True
        return False

    def get(self, skill_id: str) -> Skill | None:
        """Get a skill by name/ID."""
        return self._skills.get(skill_id)

    def list_all(self) -> list[Skill]:
        """Return all registered skills."""
        return list(self._skills.values())

    def list_by_category(self, category: str) -> list[Skill]:
        """Return skills filtered by category."""
        return [s for s in self._skills.values() if s.category == category]

    def list_by_stage(self, stage: int) -> list[Skill]:
        """Return skills applicable to a specific stage."""
        return [
            s for s in self._skills.values()
            if not s.applicable_stages or stage in s.applicable_stages
        ]

    def match(
        self,
        context: str,
        stage: int | str,
        top_k: int | None = None,
    ) -> list[Skill]:
        """Match skills to current context and stage.

        Args:
            context: Task context text.
            stage: Current pipeline stage number or name.
            top_k: Max results (defaults to max_skills_per_stage).

        Returns:
            List of matched skills sorted by relevance.
        """
        k = top_k or self._max_skills
        return match_skills(
            list(self._skills.values()),
            context,
            stage,
            top_k=k,
            fallback_matching=self._fallback_matching,
        )

    def export_for_prompt(
        self,
        skills: list[Skill],
        max_chars: int = 4000,
    ) -> str:
        """Format matched skills as prompt injection text.

        Args:
            skills: List of matched skills.
            max_chars: Character limit.

        Returns:
            Formatted prompt text.
        """
        return format_skills_for_prompt(skills, max_chars=max_chars)

    def count(self) -> int:
        """Return total number of registered skills."""
        return len(self._skills)
