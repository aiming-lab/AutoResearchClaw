"""Tests for the advanced multi-phase code generation agent (F-02)."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

from researchclaw.llm.client import LLMResponse
from researchclaw.pipeline.code_agent import (
    CodeAgent,
    CodeAgentConfig,
    CodeAgentResult,
    SolutionNode,
    _SimpleResult,
)
from researchclaw.prompts import PromptManager


# ---------------------------------------------------------------------------
# Test fixtures
# ---------------------------------------------------------------------------


class FakeLLM:
    """Fake LLM client that returns configurable responses."""

    def __init__(self, responses: list[str] | None = None):
        self.calls: list[dict[str, Any]] = []
        self._responses = list(responses or [])
        self._call_idx = 0

    def chat(self, messages: list[dict], **kwargs: Any) -> LLMResponse:
        self.calls.append({"messages": messages, **kwargs})
        if self._responses:
            text = self._responses[min(self._call_idx, len(self._responses) - 1)]
        else:
            text = '```filename:main.py\nprint("hello")\n```'
        self._call_idx += 1
        return LLMResponse(content=text, model="fake-model")


@dataclass
class FakeSandboxResult:
    returncode: int = 0
    stdout: str = "primary_metric: 0.95"
    stderr: str = ""
    elapsed_sec: float = 1.0
    metrics: dict[str, object] = field(default_factory=dict)
    timed_out: bool = False


class FakeSandbox:
    """Fake sandbox for testing."""

    def __init__(self, results: list[FakeSandboxResult] | None = None):
        self.runs: list[Path] = []
        self._results = list(results or [FakeSandboxResult()])
        self._run_idx = 0

    def run_project(
        self, project_dir: Path, *, entry_point: str = "main.py",
        timeout_sec: int = 300,
    ) -> FakeSandboxResult:
        self.runs.append(project_dir)
        result = self._results[min(self._run_idx, len(self._results) - 1)]
        self._run_idx += 1
        return result


@pytest.fixture()
def stage_dir(tmp_path: Path) -> Path:
    d = tmp_path / "stage-10"
    d.mkdir()
    return d


@pytest.fixture()
def pm() -> PromptManager:
    return PromptManager()


# ---------------------------------------------------------------------------
# CodeAgentConfig tests
# ---------------------------------------------------------------------------


class TestCodeAgentConfig:
    def test_default_values(self) -> None:
        cfg = CodeAgentConfig()
        assert cfg.enabled is True
        assert cfg.architecture_planning is True
        assert cfg.exec_fix_max_iterations == 3
        assert cfg.tree_search_enabled is False
        assert cfg.review_max_rounds == 2

    def test_custom_values(self) -> None:
        cfg = CodeAgentConfig(
            enabled=False,
            exec_fix_max_iterations=5,
            tree_search_enabled=True,
            tree_search_candidates=5,
        )
        assert cfg.enabled is False
        assert cfg.exec_fix_max_iterations == 5
        assert cfg.tree_search_enabled is True
        assert cfg.tree_search_candidates == 5


# ---------------------------------------------------------------------------
# Phase 1: Architecture Planning
# ---------------------------------------------------------------------------


class TestPhase1Architecture:
    def test_architecture_planning_produces_spec(
        self, stage_dir: Path, pm: PromptManager,
    ) -> None:
        arch_yaml = (
            "```yaml\nfiles:\n  - name: main.py\n    purpose: entry point\n"
            "  - name: models.py\n    purpose: models\n```"
        )
        code = '```filename:main.py\nprint("metric: 1.0")\n```'
        # reviewer approves immediately
        review = '{"verdict": "APPROVE", "score": 8, "critical_issues": []}'
        llm = FakeLLM(responses=[arch_yaml, code, review])

        agent = CodeAgent(
            llm=llm, prompts=pm,
            config=CodeAgentConfig(architecture_planning=True),
            stage_dir=stage_dir,
        )
        result = agent.generate(
            topic="test topic", exp_plan="objectives: test",
            metric="accuracy", pkg_hint="numpy, torch",
        )

        assert result.architecture_spec
        assert "main.py" in result.architecture_spec
        assert result.files
        assert result.total_llm_calls >= 2  # arch + codegen + review

    def test_architecture_planning_disabled(
        self, stage_dir: Path, pm: PromptManager,
    ) -> None:
        code = '```filename:main.py\nprint("metric: 1.0")\n```'
        review = '{"verdict": "APPROVE", "score": 9, "critical_issues": []}'
        llm = FakeLLM(responses=[code, review])

        agent = CodeAgent(
            llm=llm, prompts=pm,
            config=CodeAgentConfig(architecture_planning=False),
            stage_dir=stage_dir,
        )
        result = agent.generate(
            topic="test", exp_plan="plan", metric="m", pkg_hint="",
        )

        assert result.architecture_spec == ""
        assert result.files
        # First call should be code_generation, not the architecture planning prompt
        first_call_user = llm.calls[0]["messages"][0]["content"]
        # The architecture planning prompt has "Design the architecture" phrasing
        assert "design the architecture for an experiment" not in first_call_user.lower()


# ---------------------------------------------------------------------------
# Phase 2: Execution-in-the-Loop
# ---------------------------------------------------------------------------


class TestPhase2ExecFix:
    def test_exec_fix_loop_fixes_crashing_code(
        self, stage_dir: Path, pm: PromptManager,
    ) -> None:
        # Initial code crashes, then fix succeeds
        initial_code = '```filename:main.py\nraise RuntimeError("bug")\n```'
        fixed_code = '```filename:main.py\nprint("metric: 1.0")\n```'
        review = '{"verdict": "APPROVE", "score": 8, "critical_issues": []}'
        llm = FakeLLM(responses=[
            initial_code,  # phase 2: initial generation (no arch)
            fixed_code,    # phase 2: exec-fix iteration
            review,        # phase 4: review
        ])

        sandbox_results = [
            FakeSandboxResult(returncode=1, stderr="RuntimeError: bug"),
            FakeSandboxResult(returncode=0, stdout="metric: 1.0"),
        ]
        fake_sandbox = FakeSandbox(results=sandbox_results)

        agent = CodeAgent(
            llm=llm, prompts=pm,
            config=CodeAgentConfig(
                architecture_planning=False,
                exec_fix_max_iterations=3,
            ),
            stage_dir=stage_dir,
            sandbox_factory=lambda cfg, wd: fake_sandbox,
            experiment_config=None,
        )
        result = agent.generate(
            topic="test", exp_plan="plan", metric="metric", pkg_hint="",
        )

        assert result.files
        assert result.total_sandbox_runs >= 1

    def test_exec_fix_skipped_without_sandbox(
        self, stage_dir: Path, pm: PromptManager,
    ) -> None:
        code = '```filename:main.py\nprint("m: 1")\n```'
        review = '{"verdict": "APPROVE", "score": 9, "critical_issues": []}'
        llm = FakeLLM(responses=[code, review])

        agent = CodeAgent(
            llm=llm, prompts=pm,
            config=CodeAgentConfig(architecture_planning=False),
            stage_dir=stage_dir,
            sandbox_factory=None,
        )
        result = agent.generate(
            topic="t", exp_plan="p", metric="m", pkg_hint="",
        )

        assert result.total_sandbox_runs == 0
        assert result.files

    def test_exec_fix_max_iterations_respected(
        self, stage_dir: Path, pm: PromptManager,
    ) -> None:
        code = '```filename:main.py\nraise RuntimeError("persistent")\n```'
        review = '{"verdict": "APPROVE", "score": 5, "critical_issues": []}'
        llm = FakeLLM(responses=[code, code, code, code, review])

        always_crash = FakeSandbox(
            results=[FakeSandboxResult(returncode=1, stderr="RuntimeError")]
        )

        agent = CodeAgent(
            llm=llm, prompts=pm,
            config=CodeAgentConfig(
                architecture_planning=False,
                exec_fix_max_iterations=2,
            ),
            stage_dir=stage_dir,
            sandbox_factory=lambda cfg, wd: always_crash,
            experiment_config=None,
        )
        result = agent.generate(
            topic="t", exp_plan="p", metric="m", pkg_hint="",
        )

        # Should have exactly 2 sandbox runs (max iterations)
        assert result.total_sandbox_runs == 2


# ---------------------------------------------------------------------------
# Phase 3: Solution Tree Search
# ---------------------------------------------------------------------------


class TestPhase3TreeSearch:
    def test_tree_search_generates_multiple_candidates(
        self, stage_dir: Path, pm: PromptManager,
    ) -> None:
        code_a = '```filename:main.py\nprint("metric: 0.5")\n```'
        code_b = '```filename:main.py\nprint("metric: 0.9")\n```'
        review = '{"verdict": "APPROVE", "score": 9, "critical_issues": []}'
        llm = FakeLLM(responses=[code_a, code_b, review])

        sandbox = FakeSandbox(results=[
            FakeSandboxResult(returncode=0, stdout="metric: 0.5",
                              metrics={"metric": 0.5}),
            FakeSandboxResult(returncode=0, stdout="metric: 0.9",
                              metrics={"metric": 0.9}),
        ])

        agent = CodeAgent(
            llm=llm, prompts=pm,
            config=CodeAgentConfig(
                architecture_planning=False,
                tree_search_enabled=True,
                tree_search_candidates=2,
                tree_search_max_depth=1,
            ),
            stage_dir=stage_dir,
            sandbox_factory=lambda cfg, wd: sandbox,
            experiment_config=None,
        )
        result = agent.generate(
            topic="t", exp_plan="p", metric="metric", pkg_hint="",
        )

        assert result.tree_nodes_explored >= 2
        assert result.files

    def test_tree_search_fixes_crashing_candidates(
        self, stage_dir: Path, pm: PromptManager,
    ) -> None:
        crash_code = '```filename:main.py\nraise ValueError("x")\n```'
        fixed_code = '```filename:main.py\nprint("metric: 1.0")\n```'
        review = '{"verdict": "APPROVE", "score": 8, "critical_issues": []}'
        llm = FakeLLM(responses=[
            crash_code,    # candidate 0
            crash_code,    # candidate 1
            fixed_code,    # fix for candidate 0
            fixed_code,    # fix for candidate 1
            review,        # review
        ])

        results_seq = [
            FakeSandboxResult(returncode=1, stderr="ValueError: x"),
            FakeSandboxResult(returncode=1, stderr="ValueError: x"),
            FakeSandboxResult(returncode=0, stdout="metric: 1.0"),
            FakeSandboxResult(returncode=0, stdout="metric: 1.0"),
        ]
        sandbox = FakeSandbox(results=results_seq)

        agent = CodeAgent(
            llm=llm, prompts=pm,
            config=CodeAgentConfig(
                architecture_planning=False,
                tree_search_enabled=True,
                tree_search_candidates=2,
                tree_search_max_depth=2,
            ),
            stage_dir=stage_dir,
            sandbox_factory=lambda cfg, wd: sandbox,
            experiment_config=None,
        )
        result = agent.generate(
            topic="t", exp_plan="p", metric="metric", pkg_hint="",
        )

        assert result.tree_nodes_explored >= 2


# ---------------------------------------------------------------------------
# Phase 4: Multi-Agent Review
# ---------------------------------------------------------------------------


class TestPhase4Review:
    def test_review_approves_on_first_round(
        self, stage_dir: Path, pm: PromptManager,
    ) -> None:
        code = '```filename:main.py\nprint("m: 1")\n```'
        review = '{"verdict": "APPROVE", "score": 9, "critical_issues": []}'
        llm = FakeLLM(responses=[code, review])

        agent = CodeAgent(
            llm=llm, prompts=pm,
            config=CodeAgentConfig(
                architecture_planning=False,
                review_max_rounds=2,
            ),
            stage_dir=stage_dir,
        )
        result = agent.generate(
            topic="t", exp_plan="p", metric="m", pkg_hint="",
        )

        assert result.review_rounds == 1

    def test_review_triggers_fix_on_critical_issues(
        self, stage_dir: Path, pm: PromptManager,
    ) -> None:
        code = '```filename:main.py\nprint("m: 1")\n```'
        review1 = json.dumps({
            "verdict": "REVISE",
            "score": 3,
            "critical_issues": ["Missing seed handling", "Wrong metric name"],
            "suggestions": [],
        })
        fixed = '```filename:main.py\nimport random\nrandom.seed(42)\nprint("m: 1")\n```'
        review2 = '{"verdict": "APPROVE", "score": 8, "critical_issues": []}'
        llm = FakeLLM(responses=[code, review1, fixed, review2])

        agent = CodeAgent(
            llm=llm, prompts=pm,
            config=CodeAgentConfig(
                architecture_planning=False,
                review_max_rounds=3,
                hard_validation=False,  # Test focuses on review, not validation
            ),
            stage_dir=stage_dir,
        )
        result = agent.generate(
            topic="t", exp_plan="p", metric="m", pkg_hint="",
        )

        assert result.review_rounds == 2
        assert result.total_llm_calls == 4  # codegen + review1 + fix + review2

    def test_review_disabled(
        self, stage_dir: Path, pm: PromptManager,
    ) -> None:
        code = '```filename:main.py\nprint("m: 1")\n```'
        llm = FakeLLM(responses=[code])

        agent = CodeAgent(
            llm=llm, prompts=pm,
            config=CodeAgentConfig(
                architecture_planning=False,
                review_max_rounds=0,
                hard_validation=False,  # Test focuses on review, not validation
            ),
            stage_dir=stage_dir,
        )
        result = agent.generate(
            topic="t", exp_plan="p", metric="m", pkg_hint="",
        )

        assert result.review_rounds == 0
        assert result.total_llm_calls == 1  # only codegen


# ---------------------------------------------------------------------------
# Full pipeline tests
# ---------------------------------------------------------------------------


class TestFullPipeline:
    def test_all_phases_end_to_end(
        self, stage_dir: Path, pm: PromptManager,
    ) -> None:
        arch = "```yaml\nfiles:\n  - name: main.py\n```"
        code = '```filename:main.py\nprint("acc: 0.9")\n```'
        review = '{"verdict": "APPROVE", "score": 9, "critical_issues": []}'
        llm = FakeLLM(responses=[arch, code, review])

        sandbox = FakeSandbox(results=[
            FakeSandboxResult(returncode=0, stdout="acc: 0.9"),
        ])

        agent = CodeAgent(
            llm=llm, prompts=pm,
            config=CodeAgentConfig(
                architecture_planning=True,
                exec_fix_max_iterations=2,
                review_max_rounds=1,
            ),
            stage_dir=stage_dir,
            sandbox_factory=lambda cfg, wd: sandbox,
            experiment_config=None,
        )
        result = agent.generate(
            topic="image classification", exp_plan="test plan",
            metric="accuracy", pkg_hint="torch",
        )

        assert result.architecture_spec
        assert "main.py" in result.files
        assert result.total_llm_calls >= 3  # arch + code + review
        assert result.total_sandbox_runs >= 1
        assert result.review_rounds == 1
        assert result.validation_log

    def test_agent_writes_attempt_directories(
        self, stage_dir: Path, pm: PromptManager,
    ) -> None:
        code = '```filename:main.py\nprint("x: 1")\n```'
        review = '{"verdict": "APPROVE", "score": 9, "critical_issues": []}'
        llm = FakeLLM(responses=[code, review])

        sandbox = FakeSandbox()

        agent = CodeAgent(
            llm=llm, prompts=pm,
            config=CodeAgentConfig(architecture_planning=False),
            stage_dir=stage_dir,
            sandbox_factory=lambda cfg, wd: sandbox,
            experiment_config=None,
        )
        result = agent.generate(
            topic="t", exp_plan="p", metric="x", pkg_hint="",
        )

        attempt_dir = stage_dir / "agent_runs" / "attempt_001"
        assert attempt_dir.exists()
        assert (attempt_dir / "main.py").exists()


# ---------------------------------------------------------------------------
# SolutionNode and scoring
# ---------------------------------------------------------------------------


class TestSolutionNodeScoring:
    def test_score_running_node(self) -> None:
        node = SolutionNode(
            node_id="test",
            files={"main.py": "x"},
            runs_ok=True,
            stdout="lots of output " * 20,
            metrics={"metric": 0.95},
        )
        score = CodeAgent._score_node(node, "metric")
        assert score >= 2.0  # runs_ok(1.0) + output(0.3) + metrics(0.5) + key(0.5)

    def test_score_crashing_node(self) -> None:
        node = SolutionNode(
            node_id="test",
            files={"main.py": "x"},
            runs_ok=False,
            stderr="Error: something broke",
        )
        score = CodeAgent._score_node(node, "metric")
        assert score == 0.0  # no runs_ok, error penalty, max(0)

    def test_score_partial_output(self) -> None:
        node = SolutionNode(
            node_id="test",
            files={"main.py": "x"},
            runs_ok=True,
            stdout="short",
            metrics={},
        )
        score = CodeAgent._score_node(node, "metric")
        assert score == 1.0  # only runs_ok


# ---------------------------------------------------------------------------
# Helper methods
# ---------------------------------------------------------------------------


class TestHelpers:
    def test_format_files(self) -> None:
        files = {"main.py": "print(1)", "utils.py": "x = 2"}
        formatted = CodeAgent._format_files(files)
        assert "```filename:main.py" in formatted
        assert "```filename:utils.py" in formatted
        assert "print(1)" in formatted

    def test_parse_json_direct(self) -> None:
        result = CodeAgent._parse_json('{"score": 5}')
        assert result == {"score": 5}

    def test_parse_json_fenced(self) -> None:
        text = 'Some text\n```json\n{"verdict": "APPROVE"}\n```\nmore text'
        result = CodeAgent._parse_json(text)
        assert result == {"verdict": "APPROVE"}

    def test_parse_json_embedded(self) -> None:
        text = 'The review is: {"score": 7, "verdict": "REVISE"} end'
        result = CodeAgent._parse_json(text)
        assert result is not None
        assert result["score"] == 7

    def test_parse_json_invalid(self) -> None:
        result = CodeAgent._parse_json("not json at all")
        assert result is None

    def test_simple_result_defaults(self) -> None:
        r = _SimpleResult()
        assert r.returncode == 1
        assert r.stdout == ""
        assert r.timed_out is False


# ---------------------------------------------------------------------------
# Config integration test
# ---------------------------------------------------------------------------


class TestConfigIntegration:
    def test_code_agent_config_in_experiment_config(self) -> None:
        from researchclaw.config import CodeAgentConfig, ExperimentConfig

        exp = ExperimentConfig()
        assert hasattr(exp, "code_agent")
        assert isinstance(exp.code_agent, CodeAgentConfig)
        assert exp.code_agent.enabled is True

    def test_code_agent_config_from_dict(self, tmp_path: Path) -> None:
        from researchclaw.config import RCConfig

        data = {
            "project": {"name": "test", "mode": "docs-first"},
            "research": {
                "topic": "test",
                "domains": ["ml"],
                "daily_paper_count": 1,
                "quality_threshold": 7.0,
            },
            "runtime": {"timezone": "UTC"},
            "notifications": {
                "channel": "local",
                "on_stage_start": True,
                "on_stage_fail": False,
                "on_gate_required": True,
            },
            "knowledge_base": {
                "backend": "markdown",
                "root": str(tmp_path / "kb"),
            },
            "openclaw_bridge": {"use_memory": True, "use_message": True},
            "llm": {
                "provider": "openai-compatible",
                "base_url": "http://localhost:1234/v1",
                "api_key_env": "TEST",
                "api_key": "test-key",
                "primary_model": "test",
                "fallback_models": [],
            },
            "experiment": {
                "mode": "sandbox",
                "code_agent": {
                    "enabled": False,
                    "tree_search_enabled": True,
                    "tree_search_candidates": 5,
                },
            },
        }
        cfg = RCConfig.from_dict(data, project_root=tmp_path, check_paths=False)
        assert cfg.experiment.code_agent.enabled is False
        assert cfg.experiment.code_agent.tree_search_enabled is True
        assert cfg.experiment.code_agent.tree_search_candidates == 5


# ---------------------------------------------------------------------------
# Prompts integration test
# ---------------------------------------------------------------------------


class TestPromptsIntegration:
    def test_architecture_planning_prompt_exists(self, pm: PromptManager) -> None:
        sp = pm.sub_prompt(
            "architecture_planning",
            topic="image classification",
            exp_plan="test plan",
            metric="accuracy",
        )
        assert "architect" in sp.system.lower()
        assert "accuracy" in sp.user
        assert "image classification" in sp.user

    def test_code_exec_fix_prompt_exists(self, pm: PromptManager) -> None:
        sp = pm.sub_prompt(
            "code_exec_fix",
            stderr="ImportError: no module named foo",
            stdout_tail="loading data...",
            returncode="1",
            files_context="```filename:main.py\nimport foo\n```",
        )
        assert "debug" in sp.system.lower() or "fix" in sp.system.lower()
        assert "ImportError" in sp.user

    def test_code_reviewer_prompt_exists(self, pm: PromptManager) -> None:
        sp = pm.sub_prompt(
            "code_reviewer",
            topic="RL",
            exp_plan="test plan",
            metric="reward",
            files_context="```filename:main.py\nprint('hi')\n```",
        )
        assert "review" in sp.system.lower()
        assert "reward" in sp.user
        assert "APPROVE" in sp.user or "REVISE" in sp.user
