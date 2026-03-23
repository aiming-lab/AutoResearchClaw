"""Tests for the dynamic skills library.

Covers:
- Skill schema (agentskills.io data model)
- YAML skill loading (legacy)
- SKILL.md loading (agentskills.io)
- Skill registry (register, query, external dirs)
- Keyword matching + description fallback
- Stage filtering (int + string)
- Prompt formatting
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from researchclaw.skills.schema import Skill, STAGE_NAME_TO_NUMBER
from researchclaw.skills.loader import (
    load_skill_file,
    load_skill_from_skillmd,
    load_skillmd_from_directory,
    load_skills_from_directory,
)
from researchclaw.skills.registry import SkillRegistry
from researchclaw.skills.matcher import (
    match_skills,
    format_skills_for_prompt,
    _tokenize,
    _resolve_stage,
)


# ── Fixtures ─────────────────────────────────────────────────────────


@pytest.fixture
def sample_skill() -> Skill:
    return Skill(
        name="test-skill-1",
        description="A test skill for unit testing",
        body="## Test Skill\nDo the thing.",
        metadata={
            "category": "tooling",
            "trigger-keywords": "training,pytorch,gpu",
            "applicable-stages": "10,12",
            "priority": "5",
            "version": "1.0",
            "code-template": "print('hello')",
            "references": "Test Paper 2024",
        },
    )


@pytest.fixture
def skill_yaml_dir(tmp_path: Path) -> Path:
    d = tmp_path / "skills"
    d.mkdir()
    skill_data = {
        "id": "yaml-skill-1",
        "name": "YAML Test Skill",
        "category": "experiment",
        "description": "Loaded from YAML",
        "trigger_keywords": ["review", "literature"],
        "applicable_stages": [3, 4, 5],
        "prompt_template": "Do literature review",
        "version": "1.0",
        "priority": 3,
    }
    import yaml
    (d / "test_skill.yaml").write_text(yaml.dump(skill_data), encoding="utf-8")
    return d


@pytest.fixture
def skill_json_dir(tmp_path: Path) -> Path:
    d = tmp_path / "json_skills"
    d.mkdir()
    skill_data = {
        "id": "json-skill-1",
        "name": "JSON Test Skill",
        "category": "writing",
        "description": "Loaded from JSON",
        "trigger_keywords": ["paper", "writing"],
        "applicable_stages": [17],
        "prompt_template": "Write well",
        "version": "1.0",
        "priority": 4,
    }
    (d / "test_skill.json").write_text(
        json.dumps(skill_data), encoding="utf-8"
    )
    return d


@pytest.fixture
def skillmd_dir(tmp_path: Path) -> Path:
    """Create a directory with SKILL.md files for testing."""
    d = tmp_path / "skillmd_skills"
    d.mkdir()
    # Skill with full metadata
    s1 = d / "test-skill-md"
    s1.mkdir()
    (s1 / "SKILL.md").write_text(
        "---\n"
        "name: test-skill-md\n"
        "description: A test skill from SKILL.md\n"
        "metadata:\n"
        "  category: domain\n"
        "  trigger-keywords: \"nlp,transformer,bert\"\n"
        "  applicable-stages: \"9,10\"\n"
        "  priority: \"2\"\n"
        "---\n\n"
        "## NLP Skill\nDo NLP things.\n",
        encoding="utf-8",
    )
    # Skill with minimal metadata (no trigger-keywords)
    s2 = d / "minimal-skill"
    s2.mkdir()
    (s2 / "SKILL.md").write_text(
        "---\n"
        "name: minimal-skill\n"
        "description: A minimal skill for testing description-based matching\n"
        "---\n\n"
        "## Minimal\nJust a body.\n",
        encoding="utf-8",
    )
    return d


@pytest.fixture
def external_skillmd_dir(tmp_path: Path) -> Path:
    """Simulates an external skill directory (like Collider-Agent)."""
    d = tmp_path / "external"
    d.mkdir()
    s = d / "hep-feynrules"
    s.mkdir()
    (s / "SKILL.md").write_text(
        "---\n"
        "name: hep-feynrules\n"
        "description: Generate FeynRules model files for BSM physics\n"
        "metadata:\n"
        "  category: domain\n"
        "  applicable-stages: \"10\"\n"
        "---\n\n"
        "## FeynRules Model Generation\n"
        "Build BSM model files for MadGraph.\n",
        encoding="utf-8",
    )
    return d


# ── Skill Schema ─────────────────────────────────────────────────────


class TestSkillSchema:
    def test_create_skill(self, sample_skill: Skill) -> None:
        assert sample_skill.name == "test-skill-1"
        assert sample_skill.id == "test-skill-1"  # backward compat
        assert sample_skill.category == "tooling"
        assert len(sample_skill.trigger_keywords) == 3

    def test_to_dict(self, sample_skill: Skill) -> None:
        d = sample_skill.to_dict()
        assert d["id"] == "test-skill-1"
        assert d["applicable_stages"] == [10, 12]
        assert d["code_template"] == "print('hello')"

    def test_from_dict(self) -> None:
        data = {
            "id": "from-dict",
            "name": "From Dict",
            "category": "domain",
            "description": "Created from dict",
            "trigger_keywords": ["test"],
            "applicable_stages": [1],
            "prompt_template": "test prompt",
        }
        skill = Skill.from_dict(data)
        assert skill.name == "from-dict"
        assert skill.priority == 5  # default

    def test_from_dict_defaults(self) -> None:
        skill = Skill.from_dict({})
        assert skill.name == ""
        assert skill.version == "1.0"
        assert skill.code_template is None

    def test_roundtrip(self, sample_skill: Skill) -> None:
        d = sample_skill.to_dict()
        restored = Skill.from_dict(d)
        assert restored.name == sample_skill.name
        assert restored.applicable_stages == sample_skill.applicable_stages

    def test_stage_name_to_number(self) -> None:
        assert STAGE_NAME_TO_NUMBER["code_generation"] == 10
        assert STAGE_NAME_TO_NUMBER["paper_draft"] == 17
        assert len(STAGE_NAME_TO_NUMBER) == 23

    def test_prompt_template_alias(self, sample_skill: Skill) -> None:
        assert sample_skill.prompt_template == sample_skill.body


# ── Skill Loader ─────────────────────────────────────────────────────


class TestSkillLoader:
    def test_load_yaml(self, skill_yaml_dir: Path) -> None:
        skill = load_skill_file(skill_yaml_dir / "test_skill.yaml")
        assert skill is not None
        assert skill.name == "yaml-skill-1"
        assert skill.category == "experiment"

    def test_load_json(self, skill_json_dir: Path) -> None:
        skill = load_skill_file(skill_json_dir / "test_skill.json")
        assert skill is not None
        assert skill.name == "json-skill-1"

    def test_load_nonexistent(self, tmp_path: Path) -> None:
        skill = load_skill_file(tmp_path / "nope.yaml")
        assert skill is None

    def test_load_invalid_yaml(self, tmp_path: Path) -> None:
        bad = tmp_path / "bad.yaml"
        bad.write_text("not: [valid: yaml: {", encoding="utf-8")
        skill = load_skill_file(bad)
        assert skill is None

    def test_load_unsupported_format(self, tmp_path: Path) -> None:
        txt = tmp_path / "skill.txt"
        txt.write_text("id: test", encoding="utf-8")
        skill = load_skill_file(txt)
        assert skill is None

    def test_load_directory(self, skill_yaml_dir: Path) -> None:
        skills = load_skills_from_directory(skill_yaml_dir)
        assert len(skills) == 1

    def test_load_empty_directory(self, tmp_path: Path) -> None:
        empty = tmp_path / "empty"
        empty.mkdir()
        skills = load_skills_from_directory(empty)
        assert skills == []

    def test_load_missing_directory(self, tmp_path: Path) -> None:
        skills = load_skills_from_directory(tmp_path / "nonexistent")
        assert skills == []


class TestSkillMdLoader:
    def test_load_skillmd(self, skillmd_dir: Path) -> None:
        skill = load_skill_from_skillmd(skillmd_dir / "test-skill-md" / "SKILL.md")
        assert skill is not None
        assert skill.name == "test-skill-md"
        assert skill.category == "domain"
        assert "nlp" in skill.trigger_keywords
        assert skill.applicable_stages == [9, 10]
        assert skill.priority == 2
        assert "NLP Skill" in skill.body
        assert skill.source_format == "skillmd"

    def test_load_skillmd_minimal(self, skillmd_dir: Path) -> None:
        skill = load_skill_from_skillmd(skillmd_dir / "minimal-skill" / "SKILL.md")
        assert skill is not None
        assert skill.name == "minimal-skill"
        assert skill.trigger_keywords == []
        assert skill.applicable_stages == []
        assert skill.priority == 5  # default

    def test_load_skillmd_missing(self, tmp_path: Path) -> None:
        skill = load_skill_from_skillmd(tmp_path / "nope" / "SKILL.md")
        assert skill is None

    def test_load_skillmd_no_frontmatter(self, tmp_path: Path) -> None:
        d = tmp_path / "bad-skill"
        d.mkdir()
        (d / "SKILL.md").write_text("No frontmatter here", encoding="utf-8")
        skill = load_skill_from_skillmd(d / "SKILL.md")
        assert skill is None

    def test_load_skillmd_directory(self, skillmd_dir: Path) -> None:
        skills = load_skillmd_from_directory(skillmd_dir)
        assert len(skills) == 2
        names = {s.name for s in skills}
        assert "test-skill-md" in names
        assert "minimal-skill" in names

    def test_skillmd_wins_over_yaml(self, tmp_path: Path) -> None:
        """When both SKILL.md and YAML exist for the same name, SKILL.md wins."""
        d = tmp_path / "mixed"
        d.mkdir()
        # YAML file
        import yaml
        (d / "test-skill-md.yaml").write_text(
            yaml.dump({
                "id": "test-skill-md",
                "name": "test-skill-md",
                "category": "tooling",
                "description": "From YAML",
                "trigger_keywords": ["x"],
                "applicable_stages": [1],
                "prompt_template": "yaml body",
            }),
            encoding="utf-8",
        )
        # SKILL.md file
        sd = d / "test-skill-md"
        sd.mkdir()
        (sd / "SKILL.md").write_text(
            "---\nname: test-skill-md\ndescription: From SKILL.md\n---\n\nskillmd body\n",
            encoding="utf-8",
        )
        skills = load_skills_from_directory(d)
        matched = [s for s in skills if s.name == "test-skill-md"]
        assert len(matched) == 1
        assert matched[0].source_format == "skillmd"
        assert "From SKILL.md" in matched[0].description


# ── Matcher ──────────────────────────────────────────────────────────


class TestMatcher:
    def test_tokenize(self) -> None:
        tokens = _tokenize("PyTorch Training GPU")
        assert "pytorch" in tokens
        assert "training" in tokens
        assert "gpu" in tokens

    def test_match_by_keyword(self, sample_skill: Skill) -> None:
        matched = match_skills(
            [sample_skill],
            context="training a pytorch model on gpu",
            stage=10,
        )
        assert len(matched) == 1
        assert matched[0].name == "test-skill-1"

    def test_match_filters_by_stage(self, sample_skill: Skill) -> None:
        matched = match_skills(
            [sample_skill],
            context="training pytorch gpu",
            stage=1,  # not in applicable_stages
        )
        assert len(matched) == 0

    def test_match_empty_context(self, sample_skill: Skill) -> None:
        matched = match_skills([sample_skill], context="", stage=10)
        assert len(matched) == 0

    def test_match_no_keyword_overlap(self, sample_skill: Skill) -> None:
        matched = match_skills(
            [sample_skill],
            context="linguistics morphology",
            stage=10,
        )
        assert len(matched) == 0

    def test_match_respects_top_k(self) -> None:
        skills = [
            Skill(
                name=f"skill-{i}",
                description="test",
                body="test",
                metadata={
                    "category": "tooling",
                    "trigger-keywords": "training",
                    "applicable-stages": "10",
                    "priority": str(i),
                },
            )
            for i in range(10)
        ]
        matched = match_skills(skills, context="training", stage=10, top_k=3)
        assert len(matched) == 3

    def test_match_priority_ordering(self) -> None:
        high = Skill(
            name="high", description="t", body="t",
            metadata={
                "trigger-keywords": "training",
                "applicable-stages": "10",
                "priority": "1",
            },
        )
        low = Skill(
            name="low", description="t", body="t",
            metadata={
                "trigger-keywords": "training",
                "applicable-stages": "10",
                "priority": "9",
            },
        )
        matched = match_skills([low, high], context="training", stage=10)
        assert matched[0].name == "high"

    def test_match_string_stage(self, sample_skill: Skill) -> None:
        """String stage names should be resolved via STAGE_NAME_TO_NUMBER."""
        matched = match_skills(
            [sample_skill],
            context="training pytorch gpu",
            stage="code_generation",  # resolves to 10
        )
        assert len(matched) == 1
        assert matched[0].name == "test-skill-1"

    def test_match_string_stage_mismatch(self, sample_skill: Skill) -> None:
        matched = match_skills(
            [sample_skill],
            context="training pytorch gpu",
            stage="paper_draft",  # resolves to 17, not in [10, 12]
        )
        assert len(matched) == 0

    def test_resolve_stage(self) -> None:
        assert _resolve_stage(10) == 10
        assert _resolve_stage("code_generation") == 10
        assert _resolve_stage("unknown_stage") == -1

    def test_match_description_fallback(self) -> None:
        """Skills without trigger_keywords should match via description."""
        external_skill = Skill(
            name="ext-skill",
            description="Generate FeynRules model files for BSM physics",
            body="Do feynrules things.",
            metadata={"applicable-stages": "10"},
        )
        matched = match_skills(
            [external_skill],
            context="feynrules model generation",
            stage=10,
            fallback_matching=True,
        )
        assert len(matched) == 1
        assert matched[0].name == "ext-skill"

    def test_match_description_fallback_disabled(self) -> None:
        external_skill = Skill(
            name="ext-skill",
            description="Generate FeynRules model files for BSM physics",
            body="Do feynrules things.",
            metadata={"applicable-stages": "10"},
        )
        matched = match_skills(
            [external_skill],
            context="feynrules model generation",
            stage=10,
            fallback_matching=False,
        )
        assert len(matched) == 0


class TestFormatSkills:
    def test_format_single_skill(self, sample_skill: Skill) -> None:
        text = format_skills_for_prompt([sample_skill])
        assert "test-skill-1" in text
        assert "tooling" in text

    def test_format_empty(self) -> None:
        assert format_skills_for_prompt([]) == ""

    def test_format_includes_code_template(self, sample_skill: Skill) -> None:
        text = format_skills_for_prompt([sample_skill])
        assert "print('hello')" in text

    def test_format_includes_references(self, sample_skill: Skill) -> None:
        text = format_skills_for_prompt([sample_skill])
        assert "Test Paper 2024" in text

    def test_format_respects_max_chars(self) -> None:
        skills = [
            Skill(
                name=f"s{i}", description="t", body="x" * 500,
                metadata={
                    "category": "tooling",
                    "trigger-keywords": "t",
                },
            )
            for i in range(10)
        ]
        text = format_skills_for_prompt(skills, max_chars=1000)
        assert len(text) <= 1500  # some slack for headers


# ── Registry ─────────────────────────────────────────────────────────


class TestSkillRegistry:
    def test_registry_loads_builtins(self) -> None:
        registry = SkillRegistry()
        assert registry.count() >= 12  # builtin skills (SKILL.md format)

    def test_builtin_skillmd_count(self) -> None:
        """All builtin skills should load from SKILL.md."""
        registry = SkillRegistry()
        assert registry.count() == 16

    def test_register_custom(self, sample_skill: Skill) -> None:
        registry = SkillRegistry()
        initial = registry.count()
        registry.register(sample_skill)
        assert registry.count() == initial + 1

    def test_get_skill(self, sample_skill: Skill) -> None:
        registry = SkillRegistry()
        registry.register(sample_skill)
        got = registry.get("test-skill-1")
        assert got is not None
        assert got.name == "test-skill-1"

    def test_get_nonexistent(self) -> None:
        registry = SkillRegistry()
        assert registry.get("nonexistent") is None

    def test_unregister(self, sample_skill: Skill) -> None:
        registry = SkillRegistry()
        registry.register(sample_skill)
        assert registry.unregister("test-skill-1")
        assert registry.get("test-skill-1") is None

    def test_unregister_nonexistent(self) -> None:
        registry = SkillRegistry()
        assert not registry.unregister("nope")

    def test_list_by_category(self) -> None:
        registry = SkillRegistry()
        tooling = registry.list_by_category("tooling")
        assert len(tooling) > 0
        assert all(s.category == "tooling" for s in tooling)

    def test_list_by_stage(self) -> None:
        registry = SkillRegistry()
        stage_10 = registry.list_by_stage(10)
        assert len(stage_10) > 0

    def test_match(self) -> None:
        registry = SkillRegistry()
        matched = registry.match("pytorch training classification cifar", stage=10)
        assert len(matched) > 0

    def test_match_string_stage(self) -> None:
        registry = SkillRegistry()
        matched = registry.match(
            "pytorch training classification",
            stage="code_generation",
        )
        assert len(matched) > 0

    def test_export_for_prompt(self) -> None:
        registry = SkillRegistry()
        matched = registry.match("pytorch training", stage=10, top_k=2)
        text = registry.export_for_prompt(matched)
        assert len(text) > 0

    def test_custom_dir_loading(self, skill_yaml_dir: Path) -> None:
        registry = SkillRegistry(custom_dirs=[str(skill_yaml_dir)])
        skill = registry.get("yaml-skill-1")
        assert skill is not None

    def test_registry_external_dirs(self, external_skillmd_dir: Path) -> None:
        registry = SkillRegistry(external_dirs=[str(external_skillmd_dir)])
        assert registry.count() == 17  # 16 builtin + 1 external
        skill = registry.get("hep-feynrules")
        assert skill is not None
        assert skill.category == "domain"

    def test_registry_external_match_fallback(
        self, external_skillmd_dir: Path
    ) -> None:
        """External skills without trigger_keywords should match via description."""
        registry = SkillRegistry(
            external_dirs=[str(external_skillmd_dir)],
            fallback_matching=True,
        )
        matched = registry.match("feynrules model generation", stage=10, top_k=10)
        names = [s.name for s in matched]
        assert "hep-feynrules" in names
