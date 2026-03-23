"""Skill file loader — supports YAML, JSON, and SKILL.md (agentskills.io)."""

from __future__ import annotations

import json
import logging
from pathlib import Path
import yaml

from researchclaw.skills.schema import Skill

logger = logging.getLogger(__name__)


# ── SKILL.md loader ──────────────────────────────────────────────────


def load_skill_from_skillmd(path: Path) -> Skill | None:
    """Load a skill from a ``SKILL.md`` file (agentskills.io format).

    Expected layout::

        ---
        name: kebab-case-id
        description: one-liner
        metadata:
          category: domain
          trigger-keywords: "kw1,kw2"
        ---

        Markdown body here ...

    Args:
        path: Path to the SKILL.md file.

    Returns:
        Parsed :class:`Skill`, or *None* on failure.
    """
    try:
        text = path.read_text(encoding="utf-8")
    except Exception as exc:
        logger.warning("Failed to read SKILL.md at %s: %s", path, exc)
        return None

    # Split on YAML frontmatter markers
    parts = text.split("---", 2)
    if len(parts) < 3:
        logger.warning("SKILL.md missing frontmatter delimiters: %s", path)
        return None

    try:
        header = yaml.safe_load(parts[1])
    except Exception as exc:
        logger.warning("Invalid YAML frontmatter in %s: %s", path, exc)
        return None

    if not isinstance(header, dict):
        logger.warning("Frontmatter is not a dict in %s", path)
        return None

    name = str(header.get("name", ""))
    if not name:
        logger.warning("SKILL.md missing 'name' field: %s", path)
        return None

    description = str(header.get("description", ""))
    body = parts[2].strip()

    # Build metadata — flatten nested 'metadata' dict from frontmatter
    metadata: dict[str, str] = {}
    raw_meta = header.get("metadata")
    if isinstance(raw_meta, dict):
        for k, v in raw_meta.items():
            metadata[str(k)] = str(v)

    # Also pull top-level keys that map to metadata
    for key in ("category", "license", "compatibility", "version", "author"):
        if key in header and key not in metadata:
            metadata[key] = str(header[key])

    skill_license = str(header.get("license", ""))
    compatibility = str(header.get("compatibility", ""))

    return Skill(
        name=name,
        description=description,
        body=body,
        license=skill_license,
        compatibility=compatibility,
        metadata=metadata,
        source_dir=path.parent,
        source_format="skillmd",
    )


def load_skillmd_from_directory(directory: Path) -> list[Skill]:
    """Scan *directory* for ``*/SKILL.md`` sub-directories.

    Each immediate sub-directory containing a ``SKILL.md`` file is
    treated as a single skill.
    """
    skills: list[Skill] = []
    if not directory.exists():
        return skills

    for skill_md in sorted(directory.rglob("SKILL.md")):
        skill = load_skill_from_skillmd(skill_md)
        if skill:
            skills.append(skill)

    return skills


# ── Legacy YAML / JSON loader ────────────────────────────────────────


def load_skill_file(path: Path) -> Skill | None:
    """Load a single skill from a YAML or JSON file.

    Args:
        path: Path to the skill file.

    Returns:
        Parsed Skill object, or None if loading fails.
    """
    try:
        text = path.read_text(encoding="utf-8")
        if path.suffix in (".yaml", ".yml"):
            data = yaml.safe_load(text)
        elif path.suffix == ".json":
            data = json.loads(text)
        else:
            logger.warning("Unsupported skill file format: %s", path)
            return None

        if not isinstance(data, dict):
            logger.warning("Skill file is not a dict: %s", path)
            return None

        skill = Skill.from_dict(data)
        if not skill.name:
            logger.warning("Skill missing name/id: %s", path)
            return None

        return skill
    except Exception as exc:
        logger.warning("Failed to load skill from %s: %s", path, exc)
        return None


def load_skills_from_directory(directory: Path) -> list[Skill]:
    """Recursively load all skills from a directory.

    Supports both ``SKILL.md`` (agentskills.io) and legacy YAML/JSON.
    When both formats exist for the same skill name, SKILL.md wins.

    Args:
        directory: Root directory to scan.

    Returns:
        List of successfully loaded Skill objects.
    """
    skills_by_name: dict[str, Skill] = {}
    if not directory.exists():
        return []

    # 1. Load SKILL.md files first (higher priority)
    for skill in load_skillmd_from_directory(directory):
        skills_by_name[skill.name] = skill

    # 2. Load legacy YAML/JSON (only if no SKILL.md with same name)
    for pattern in ("*.yaml", "*.yml", "*.json"):
        for path in sorted(directory.rglob(pattern)):
            if path.name == "__init__.py":
                continue
            skill = load_skill_file(path)
            if skill and skill.name not in skills_by_name:
                skills_by_name[skill.name] = skill

    skills = list(skills_by_name.values())
    logger.info("Loaded %d skills from %s", len(skills), directory)
    return skills
