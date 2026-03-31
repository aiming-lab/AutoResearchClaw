"""Tests for the A-Evolve structured evolution module."""

from __future__ import annotations

import json
from pathlib import Path
from unittest import mock

import pytest

from researchclaw.evolution import LessonEntry
from researchclaw.evolution_aevolve import (
    Mutation,
    Observation,
    evolve,
    gate,
    observe,
    reload,
    run_aevolve_cycle,
)


class FakeLLMResponse:
    def __init__(self, content: str, model: str = "fake"):
        self.content = content
        self.model = model
        self.finish_reason = "stop"


class FakeLLMClient:
    def __init__(self, responses: list[str]):
        self._responses = list(responses)
        self._call_idx = 0

    def chat(self, messages, *, system=None, json_mode=False, max_tokens=None, **kw):
        idx = min(self._call_idx, len(self._responses) - 1)
        self._call_idx += 1
        return FakeLLMResponse(self._responses[idx])


def _make_lesson(
    stage: str = "experiment_run",
    stage_num: int = 12,
    category: str = "experiment",
    severity: str = "error",
    desc: str = "Sandbox timeout after 300s",
) -> LessonEntry:
    return LessonEntry(
        stage_name=stage,
        stage_num=stage_num,
        category=category,
        severity=severity,
        description=desc,
        timestamp="2026-03-30T12:00:00+00:00",
        run_id="test-run",
    )


def _make_stage_result(stage_num: int, status: str, error: str = ""):
    return mock.MagicMock(stage=stage_num, status=status, error=error)


# ---------------------------------------------------------------------------
# observe()
# ---------------------------------------------------------------------------


class TestObserve:
    def test_observe_returns_observations(self):
        llm = FakeLLMClient([
            json.dumps([{
                "obs_id": "OBS-1",
                "category": "timeout",
                "root_cause": "Sandbox killed after 300s",
                "affected_stages": ["experiment_run"],
                "frequency": 3,
                "severity": "blocking",
                "description": "Experiment execution timeout",
            }])
        ])
        lessons = [_make_lesson()]
        results = [_make_stage_result(12, "failed", "timeout")]

        obs = observe(lessons, results, llm)
        assert len(obs) == 1
        assert obs[0].obs_id == "OBS-1"
        assert obs[0].category == "timeout"
        assert obs[0].severity == "blocking"

    def test_observe_skips_when_no_failures(self):
        llm = FakeLLMClient(["should not be called"])
        results = [_make_stage_result(1, "done")]

        obs = observe([], results, llm)
        assert obs == []

    def test_observe_handles_llm_failure(self):
        llm = mock.MagicMock()
        llm.chat.side_effect = RuntimeError("LLM down")

        obs = observe([_make_lesson()], [_make_stage_result(12, "failed", "err")], llm)
        assert obs == []


# ---------------------------------------------------------------------------
# evolve()
# ---------------------------------------------------------------------------


class TestEvolve:
    def test_evolve_proposes_skill_mutation(self, tmp_path: Path):
        llm = FakeLLMClient([
            json.dumps([{
                "mutation_type": "skill",
                "name": "arc-aevolve-sandbox-timeout",
                "description": "Handle sandbox timeouts gracefully",
                "content": "# Sandbox Timeout Handler\n\n1. Check timeout config...",
                "target_observation": "OBS-1",
            }])
        ])
        obs = [Observation(
            obs_id="OBS-1", category="timeout",
            root_cause="Sandbox killed", affected_stages=["experiment_run"],
            frequency=3, severity="blocking",
            description="Experiment execution timeout",
        )]

        mutations = evolve(obs, llm, tmp_path)
        assert len(mutations) == 1
        assert mutations[0].mutation_type == "skill"
        assert mutations[0].name == "arc-aevolve-sandbox-timeout"

    def test_evolve_caps_at_3_mutations(self, tmp_path: Path):
        llm = FakeLLMClient([
            json.dumps([
                {"mutation_type": "skill", "name": f"arc-aevolve-s{i}",
                 "description": "d", "content": "c", "target_observation": "OBS-1"}
                for i in range(5)
            ])
        ])
        obs = [Observation(
            obs_id="OBS-1", category="timeout",
            root_cause="x", affected_stages=["s"],
            frequency=5, severity="blocking", description="d",
        )]

        mutations = evolve(obs, llm, tmp_path)
        assert len(mutations) == 3  # capped

    def test_evolve_returns_empty_on_no_observations(self, tmp_path: Path):
        llm = FakeLLMClient(["should not be called"])
        assert evolve([], llm, tmp_path) == []


# ---------------------------------------------------------------------------
# gate()
# ---------------------------------------------------------------------------


class TestGate:
    def test_gate_accepts_good_mutation(self):
        llm = FakeLLMClient([
            json.dumps([{
                "name": "arc-aevolve-fix",
                "passed": True,
                "reason": "Targeted and specific",
            }])
        ])
        mutations = [Mutation(
            mutation_type="skill", name="arc-aevolve-fix",
            description="d", content="c", target_observation="OBS-1",
        )]

        result = gate(mutations, llm)
        assert result[0].gate_passed is True

    def test_gate_rejects_bad_mutation(self):
        llm = FakeLLMClient([
            json.dumps([{
                "name": "arc-aevolve-bad",
                "passed": False,
                "reason": "Too broad, could cause regressions",
            }])
        ])
        mutations = [Mutation(
            mutation_type="skill", name="arc-aevolve-bad",
            description="d", content="c", target_observation="OBS-1",
        )]

        result = gate(mutations, llm)
        assert result[0].gate_passed is False
        assert "broad" in result[0].gate_reason

    def test_gate_skips_none_mutations(self):
        llm = FakeLLMClient(["should not be called"])
        mutations = [Mutation(
            mutation_type="none", name="skip",
            description="d", content="", target_observation="OBS-1",
        )]

        result = gate(mutations, llm)
        assert result[0].gate_passed is True


# ---------------------------------------------------------------------------
# reload()
# ---------------------------------------------------------------------------


class TestReload:
    def test_reload_writes_skill(self, tmp_path: Path):
        skills_dir = tmp_path / "skills"
        run_dir = tmp_path / "run"
        run_dir.mkdir()

        mutations = [Mutation(
            mutation_type="skill", name="arc-aevolve-fix-timeout",
            description="Handle timeouts", content="# Fix\n\n1. Check config",
            target_observation="OBS-1", gate_passed=True,
        )]
        obs = [Observation(
            obs_id="OBS-1", category="timeout",
            root_cause="x", affected_stages=["s"],
            frequency=3, severity="blocking", description="d",
        )]

        created = reload(mutations, obs, skills_dir, run_dir)
        assert "arc-aevolve-fix-timeout" in created
        skill_file = skills_dir / "arc-aevolve-fix-timeout" / "SKILL.md"
        assert skill_file.exists()
        content = skill_file.read_text(encoding="utf-8")
        assert "Handle timeouts" in content
        assert "a-evolve" in content

    def test_reload_writes_prompt_patch(self, tmp_path: Path):
        run_dir = tmp_path / "run"
        run_dir.mkdir()

        mutations = [Mutation(
            mutation_type="prompt_patch", name="patch-clarity",
            description="Improve prompt clarity",
            content="When generating experiment code, always include error handling.",
            target_observation="OBS-2", gate_passed=True,
        )]

        reload(mutations, [], tmp_path / "skills", run_dir)
        patches_file = run_dir / "evolution" / "prompt_patches.md"
        assert patches_file.exists()
        assert "error handling" in patches_file.read_text(encoding="utf-8")

    def test_reload_writes_knowledge_entry(self, tmp_path: Path):
        run_dir = tmp_path / "run"
        run_dir.mkdir()

        mutations = [Mutation(
            mutation_type="knowledge_entry", name="know-batch-size",
            description="Batch size insight",
            content="Batch size > 64 causes OOM on 8GB GPUs",
            target_observation="OBS-3", gate_passed=True,
        )]

        reload(mutations, [], tmp_path / "skills", run_dir)
        kb_file = run_dir / "evolution" / "knowledge_entries.jsonl"
        assert kb_file.exists()
        entry = json.loads(kb_file.read_text(encoding="utf-8").strip())
        assert "OOM" in entry["content"]

    def test_reload_skips_rejected_mutations(self, tmp_path: Path):
        skills_dir = tmp_path / "skills"
        run_dir = tmp_path / "run"
        run_dir.mkdir()

        mutations = [Mutation(
            mutation_type="skill", name="arc-aevolve-rejected",
            description="d", content="c",
            target_observation="OBS-1", gate_passed=False,
        )]

        created = reload(mutations, [], skills_dir, run_dir)
        assert created == []
        assert not (skills_dir / "arc-aevolve-rejected").exists()

    def test_reload_writes_log(self, tmp_path: Path):
        run_dir = tmp_path / "run"
        run_dir.mkdir()

        reload([], [], tmp_path / "skills", run_dir)
        log_file = run_dir / "evolution" / "aevolve_log.json"
        assert log_file.exists()
        log_data = json.loads(log_file.read_text(encoding="utf-8"))
        assert "observations" in log_data
        assert "mutations" in log_data


# ---------------------------------------------------------------------------
# run_aevolve_cycle() (integration)
# ---------------------------------------------------------------------------


class TestRunAEvolveCycle:
    def test_full_cycle(self, tmp_path: Path):
        skills_dir = tmp_path / "skills"
        run_dir = tmp_path / "run"
        run_dir.mkdir()

        # LLM responses: observe → evolve → gate
        llm = FakeLLMClient([
            # observe
            json.dumps([{
                "obs_id": "OBS-1", "category": "timeout",
                "root_cause": "Sandbox killed",
                "affected_stages": ["experiment_run"],
                "frequency": 3, "severity": "blocking",
                "description": "Experiment timeout",
            }]),
            # evolve
            json.dumps([{
                "mutation_type": "skill",
                "name": "arc-aevolve-timeout-fix",
                "description": "Handle timeouts",
                "content": "# Fix\n1. Increase timeout",
                "target_observation": "OBS-1",
            }]),
            # gate
            json.dumps([{
                "name": "arc-aevolve-timeout-fix",
                "passed": True,
                "reason": "Specific and testable",
            }]),
        ])

        lessons = [_make_lesson()]
        results = [_make_stage_result(12, "failed", "timeout")]

        created = run_aevolve_cycle(lessons, results, llm, skills_dir, run_dir)
        assert len(created) == 1
        assert "arc-aevolve-timeout-fix" in created
        assert (skills_dir / "arc-aevolve-timeout-fix" / "SKILL.md").exists()
        assert (run_dir / "evolution" / "aevolve_log.json").exists()

    def test_cycle_skips_when_no_issues(self, tmp_path: Path):
        llm = FakeLLMClient(["should not be called"])
        results = [_make_stage_result(1, "done")]

        created = run_aevolve_cycle([], results, llm, tmp_path, tmp_path)
        assert created == []


# ---------------------------------------------------------------------------
# _load_project_skills()
# ---------------------------------------------------------------------------


class TestLoadProjectSkills:
    def test_load_project_skills_finds_skill(self, tmp_path: Path, monkeypatch):
        from researchclaw import evolution

        # Create a fake .claude/skills/test-skill/SKILL.md
        skill_dir = tmp_path / ".claude" / "skills" / "test-skill"
        skill_dir.mkdir(parents=True)
        (skill_dir / "SKILL.md").write_text("# Test Skill\nDo something.")

        # Also create the researchclaw skill (should be skipped)
        rc_dir = tmp_path / ".claude" / "skills" / "researchclaw"
        rc_dir.mkdir(parents=True)
        (rc_dir / "SKILL.md").write_text("# CLI Skill\nShould be skipped.")

        # Patch the root detection
        monkeypatch.setattr(
            evolution, "_PROJECT_SKILLS_DIRS", (".claude/skills",)
        )
        # Patch Path(__file__).parent.parent to our tmp_path
        original_func = evolution._load_project_skills

        def patched():
            from pathlib import Path as _P
            skills: list[str] = []
            for rel_dir in evolution._PROJECT_SKILLS_DIRS:
                sd = tmp_path / rel_dir
                if not sd.is_dir():
                    continue
                for sub in sorted(sd.iterdir()):
                    if not sub.is_dir() or sub.name == "researchclaw":
                        continue
                    sf = sub / "SKILL.md"
                    if sf.is_file():
                        try:
                            text = sf.read_text(encoding="utf-8").strip()
                            if text:
                                skills.append(text)
                        except OSError:
                            continue
            return skills

        monkeypatch.setattr(evolution, "_load_project_skills", patched)

        result = evolution._load_project_skills()
        assert len(result) == 1
        assert "Test Skill" in result[0]
