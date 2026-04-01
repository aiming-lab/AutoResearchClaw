"""Configuration for the MetaClaw integration bridge."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class PRMConfig:
    """PRM (Process Reward Model) quality gate settings."""

    enabled: bool = False
    api_base: str = ""
    api_key_env: str = ""
    api_key: str = ""
    model: str = "gpt-5.4"
    votes: int = 3
    temperature: float = 0.6
    gate_stages: tuple[int, ...] = (5, 9, 15, 20)


@dataclass(frozen=True)
class LessonToSkillConfig:
    """Settings for converting AutoResearchClaw lessons into MetaClaw skills."""

    enabled: bool = True
    min_severity: str = "error"
    max_skills_per_run: int = 3


@dataclass(frozen=True)
class MetaClawBridgeConfig:
    """Top-level MetaClaw bridge configuration."""

    enabled: bool = False
    proxy_url: str = "http://localhost:30000"
    skills_dir: str = "~/.metaclaw/skills"
    fallback_url: str = ""  # Direct LLM URL if MetaClaw proxy is down
    fallback_api_key: str = ""
    prm: PRMConfig = field(default_factory=PRMConfig)
    lesson_to_skill: LessonToSkillConfig = field(default_factory=LessonToSkillConfig)
