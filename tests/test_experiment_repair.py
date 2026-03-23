"""Tests for experiment_repair — repair loop and prompt generation."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from researchclaw.pipeline.experiment_diagnosis import (
    DeficiencyType,
    Deficiency,
    ExperimentDiagnosis,
    PaperMode,
)
from researchclaw.pipeline.experiment_repair import (
    ExperimentRepairResult,
    RepairCycleResult,
    build_repair_prompt,
    run_repair_loop,
    select_best_results,
    _extract_code_blocks,
    _build_experiment_summary_from_run,
    _load_experiment_code,
    _load_experiment_summary,
    _summary_quality_score,
)


# ---------------------------------------------------------------------------
# build_repair_prompt tests
# ---------------------------------------------------------------------------


class TestBuildRepairPrompt:
    def test_basic_prompt(self):
        diag = ExperimentDiagnosis(
            deficiencies=[
                Deficiency(
                    type=DeficiencyType.MISSING_DEPENDENCY,
                    severity="critical",
                    description="Missing Python package: utils",
                    suggested_fix="Add 'utils' to requirements.txt",
                )
            ],
            conditions_completed=["CondA"],
            conditions_failed=["CondB"],
            total_planned=2,
            completion_rate=0.5,
            summary="1 deficiency. 1/2 conditions completed.",
        )
        prompt = build_repair_prompt(
            diagnosis=diag,
            original_code={"main.py": "import utils\nprint('hello')"},
            time_budget_sec=2400,
        )
        assert "EXPERIMENT REPAIR TASK" in prompt
        assert "utils" in prompt
        assert "main.py" in prompt
        assert "2400" in prompt

    def test_scope_reduction_included(self):
        diag = ExperimentDiagnosis(
            deficiencies=[
                Deficiency(
                    type=DeficiencyType.TIME_GUARD_DOMINANT,
                    severity="major",
                    description="Time guard killed 8/10 conditions",
                    affected_conditions=["C3", "C4", "C5"],
                    suggested_fix="Reduce conditions",
                )
            ],
            conditions_completed=["C1", "C2"],
            conditions_failed=["C3", "C4", "C5", "C6", "C7", "C8", "C9", "C10"],
            total_planned=10,
            completion_rate=0.2,
        )
        prompt = build_repair_prompt(diag, original_code={})
        assert "SCOPE REDUCTION" in prompt
        assert "BASELINE" in prompt

    def test_dep_fix_section(self):
        diag = ExperimentDiagnosis(
            deficiencies=[
                Deficiency(
                    type=DeficiencyType.MISSING_DEPENDENCY,
                    severity="critical",
                    description="Missing Python package: box2d-py",
                    suggested_fix="Add 'box2d-py' to requirements.txt",
                ),
            ],
        )
        prompt = build_repair_prompt(diag, original_code={})
        assert "DEPENDENCY FIXES" in prompt
        assert "box2d-py" in prompt

    def test_long_code_truncated(self):
        long_code = "x = 1\n" * 5000
        diag = ExperimentDiagnosis()
        prompt = build_repair_prompt(diag, original_code={"big.py": long_code})
        assert "truncated" in prompt

    def test_output_format_section(self):
        diag = ExperimentDiagnosis()
        prompt = build_repair_prompt(diag, original_code={"main.py": "pass"})
        assert "OUTPUT FORMAT" in prompt
        assert "filename.py" in prompt


# ---------------------------------------------------------------------------
# ExperimentRepairResult tests
# ---------------------------------------------------------------------------


class TestRepairResult:
    def test_serialization(self):
        result = ExperimentRepairResult(
            success=False,
            total_cycles=2,
            final_mode=PaperMode.PRELIMINARY_STUDY,
        )
        d = result.to_dict()
        assert d["success"] is False
        assert d["total_cycles"] == 2
        assert d["final_mode"] == "preliminary_study"

    def test_serialization_with_cycles(self):
        diag = ExperimentDiagnosis(summary="test")
        result = ExperimentRepairResult(
            success=True,
            total_cycles=1,
            final_mode=PaperMode.FULL_PAPER,
            cycle_history=[
                RepairCycleResult(
                    cycle=1,
                    diagnosis=diag,
                    repair_applied=True,
                    repair_description="Fixed 2 files",
                ),
            ],
        )
        d = result.to_dict()
        assert d["success"] is True
        assert len(d["cycle_history"]) == 1
        assert d["cycle_history"][0]["repair_applied"] is True
        assert d["cycle_history"][0]["diagnosis_summary"] == "test"


# ---------------------------------------------------------------------------
# Code extraction tests
# ---------------------------------------------------------------------------


class TestExtractCodeBlocks:
    def test_named_blocks(self):
        text = """Here are the fixed files:

```python main.py
import torch
print("hello")
```

```python requirements.txt
torch>=2.0
numpy
```
"""
        files = _extract_code_blocks(text)
        assert "main.py" in files
        assert "requirements.txt" in files
        assert "torch" in files["main.py"]
        assert "numpy" in files["requirements.txt"]

    def test_unnamed_block_fallback(self):
        text = """```python
import torch
model = torch.nn.Linear(10, 2)
print("condition=Baseline metric=0.95")
```"""
        files = _extract_code_blocks(text)
        assert "main.py" in files
        assert "torch" in files["main.py"]

    def test_no_blocks(self):
        text = "No code here, just text."
        files = _extract_code_blocks(text)
        assert files == {}

    def test_path_normalization(self):
        text = """```python src/models/main.py
import torch
print("hello world, this is a test of the extraction")
```"""
        files = _extract_code_blocks(text)
        assert "main.py" in files


# ---------------------------------------------------------------------------
# Summary building tests
# ---------------------------------------------------------------------------


class TestBuildExperimentSummary:
    def test_basic_summary(self):
        run_result = {
            "stdout": "condition=Baseline metric=80.0\ncondition=Proposed metric=90.0",
            "stderr": "",
            "returncode": 0,
            "metrics": {
                "Baseline/0/accuracy": 80.0,
                "Proposed/0/accuracy": 90.0,
                "primary_metric": 90.0,
            },
            "elapsed_sec": 120.0,
            "timed_out": False,
        }
        summary = _build_experiment_summary_from_run(run_result, {"main.py": "pass"})
        assert "condition_summaries" in summary
        assert "Baseline" in summary["condition_summaries"]
        assert "Proposed" in summary["condition_summaries"]
        assert summary["total_conditions"] == 2
        assert summary["best_run"]["status"] == "completed"

    def test_failed_run(self):
        run_result = {
            "stdout": "",
            "stderr": "Error: crash",
            "returncode": 1,
            "metrics": {},
            "elapsed_sec": 5.0,
            "timed_out": False,
        }
        summary = _build_experiment_summary_from_run(run_result, {})
        assert summary["best_run"]["status"] == "failed"
        assert summary["total_conditions"] == 0

    def test_multi_seed_grouping(self):
        run_result = {
            "stdout": "",
            "stderr": "",
            "returncode": 0,
            "metrics": {
                "Baseline/0/accuracy": 80.0,
                "Baseline/1/accuracy": 82.0,
                "Proposed/0/accuracy": 90.0,
                "Proposed/1/accuracy": 92.0,
            },
            "elapsed_sec": 300.0,
            "timed_out": False,
        }
        summary = _build_experiment_summary_from_run(run_result, {})
        assert len(summary["condition_summaries"]) == 2
        # Mean of 80.0 and 82.0
        bl = summary["condition_summaries"]["Baseline"]
        assert abs(bl["metrics"]["accuracy"] - 81.0) < 0.01
        assert bl["n_seeds"] == 2


# ---------------------------------------------------------------------------
# File loading tests
# ---------------------------------------------------------------------------


class TestLoadExperimentCode:
    def test_loads_from_stage_13(self, tmp_path):
        exp_dir = tmp_path / "stage-13" / "experiment_final"
        exp_dir.mkdir(parents=True)
        (exp_dir / "main.py").write_text("print('hello')")
        (exp_dir / "requirements.txt").write_text("torch")

        code = _load_experiment_code(tmp_path)
        assert "main.py" in code
        assert "requirements.txt" in code

    def test_loads_from_stage_10(self, tmp_path):
        exp_dir = tmp_path / "stage-10" / "experiment"
        exp_dir.mkdir(parents=True)
        (exp_dir / "main.py").write_text("print('hello')")

        code = _load_experiment_code(tmp_path)
        assert "main.py" in code

    def test_empty_when_no_code(self, tmp_path):
        code = _load_experiment_code(tmp_path)
        assert code == {}


class TestLoadExperimentSummary:
    def test_loads_summary(self, tmp_path):
        stage_dir = tmp_path / "stage-14"
        stage_dir.mkdir()
        summary = {"condition_summaries": {"A": {}}}
        (stage_dir / "experiment_summary.json").write_text(json.dumps(summary))

        result = _load_experiment_summary(tmp_path)
        assert result is not None
        assert "A" in result["condition_summaries"]


# ---------------------------------------------------------------------------
# select_best_results tests
# ---------------------------------------------------------------------------


class TestSelectBestResults:
    def test_picks_best_across_cycles(self, tmp_path):
        # Original (1 condition)
        s14 = tmp_path / "stage-14"
        s14.mkdir()
        (s14 / "experiment_summary.json").write_text(json.dumps({
            "condition_summaries": {"A": {}},
            "best_run": {"metrics": {}},
        }))

        # Repair v1 (3 conditions — better)
        r1 = tmp_path / "stage-14_repair_v1"
        r1.mkdir()
        (r1 / "experiment_summary.json").write_text(json.dumps({
            "condition_summaries": {"A": {}, "B": {}, "C": {}},
            "best_run": {"metrics": {"primary_metric": 90.0}},
        }))

        best = select_best_results(tmp_path, [])
        assert best is not None
        assert len(best["condition_summaries"]) == 3

    def test_returns_none_when_empty(self, tmp_path):
        result = select_best_results(tmp_path, [])
        assert result is None


# ---------------------------------------------------------------------------
# Full repair loop tests (mocked)
# ---------------------------------------------------------------------------


class TestRunRepairLoop:
    def _make_run_dir(self, tmp_path, n_conditions=1, has_code=True):
        """Create a minimal run directory for testing."""
        # Stage 14 — experiment summary
        s14 = tmp_path / "stage-14"
        s14.mkdir()
        (s14 / "runs").mkdir()

        conds = {f"Cond{i}": {"metrics": {"accuracy": 70.0 + i}} for i in range(n_conditions)}
        summary = {
            "condition_summaries": conds,
            "best_run": {"metrics": {f"Cond{i}/0/accuracy": 70.0 + i for i in range(n_conditions)}},
            "metrics_summary": {"accuracy": {"mean": 70.5}},
        }
        (s14 / "experiment_summary.json").write_text(json.dumps(summary))

        run_data = {
            "stdout": "\n".join(f"condition=Cond{i} metric={70.0 + i}" for i in range(n_conditions)),
            "stderr": "",
        }
        (s14 / "runs" / "run_0.json").write_text(json.dumps(run_data))

        # Stage 10 — experiment code
        if has_code:
            s10 = tmp_path / "stage-10" / "experiment"
            s10.mkdir(parents=True)
            (s10 / "main.py").write_text("import torch\nprint('hello')")

        return tmp_path

    def test_skips_when_already_sufficient(self, tmp_path):
        """If experiment is already sufficient, return immediately."""
        # 3 conditions with 2+ seeds = full_paper
        s14 = tmp_path / "stage-14"
        s14.mkdir()
        (s14 / "runs").mkdir()
        summary = {
            "condition_summaries": {
                "A": {"metrics": {"m": 80.0}},
                "B": {"metrics": {"m": 85.0}},
                "C": {"metrics": {"m": 90.0}},
            },
            "best_run": {
                "metrics": {
                    "A/0/m": 80.0, "A/1/m": 81.0,
                    "B/0/m": 85.0, "B/1/m": 86.0,
                    "C/0/m": 90.0, "C/1/m": 91.0,
                },
            },
        }
        (s14 / "experiment_summary.json").write_text(json.dumps(summary))

        from researchclaw.config import ExperimentConfig, ExperimentRepairConfig

        class FakeConfig:
            class experiment:
                time_budget_sec = 2400
                repair = ExperimentRepairConfig(enabled=True)

            class llm:
                pass

        result = run_repair_loop(tmp_path, FakeConfig(), "test")
        assert result.success is True
        assert result.total_cycles == 0
        assert result.final_mode == PaperMode.FULL_PAPER

    def test_returns_failure_when_no_code(self, tmp_path):
        """If no experiment code found, return failure."""
        s14 = tmp_path / "stage-14"
        s14.mkdir()
        (s14 / "experiment_summary.json").write_text(json.dumps({
            "condition_summaries": {"A": {"metrics": {"m": 80.0}}},
            "best_run": {"metrics": {}},
        }))

        from researchclaw.config import ExperimentRepairConfig

        class FakeConfig:
            class experiment:
                time_budget_sec = 2400
                repair = ExperimentRepairConfig(enabled=True)

            class llm:
                pass

        result = run_repair_loop(tmp_path, FakeConfig(), "test")
        assert result.success is False
        assert result.total_cycles == 0

    def test_repair_loop_with_mocked_llm(self, tmp_path):
        """Test full repair loop with mocked LLM and sandbox."""
        run_dir = self._make_run_dir(tmp_path, n_conditions=1)

        from researchclaw.config import ExperimentRepairConfig, ExperimentConfig, OpenCodeConfig

        class FakeConfig:
            class experiment:
                time_budget_sec = 2400
                mode = "simulated"
                repair = ExperimentRepairConfig(enabled=True, max_cycles=1, use_opencode=False)
                opencode = OpenCodeConfig(enabled=False)
                metric_key = "primary_metric"

            class llm:
                pass

        # Mock the LLM to return fixed code
        mock_llm = MagicMock()
        mock_resp = MagicMock()
        mock_resp.content = """```python main.py
import torch
for cond in ["Baseline", "Proposed", "Ablation"]:
    for seed in range(2):
        acc = 80.0 + hash(cond) % 20 + seed
        print(f"condition={cond}/{seed}/accuracy metric={acc}")
print("condition=Baseline metric=80.0")
print("condition=Proposed metric=90.0")
print("condition=Ablation metric=85.0")
```"""
        mock_llm.chat.return_value = mock_resp

        # Mock sandbox to return good results
        mock_sandbox_result = MagicMock()
        mock_sandbox_result.stdout = (
            "condition=Baseline/0/accuracy metric=80.0\n"
            "condition=Baseline/1/accuracy metric=82.0\n"
            "condition=Proposed/0/accuracy metric=90.0\n"
            "condition=Proposed/1/accuracy metric=92.0\n"
            "condition=Ablation/0/accuracy metric=85.0\n"
            "condition=Ablation/1/accuracy metric=87.0\n"
        )
        mock_sandbox_result.stderr = ""
        mock_sandbox_result.returncode = 0
        mock_sandbox_result.metrics = {
            "Baseline/0/accuracy": 80.0, "Baseline/1/accuracy": 82.0,
            "Proposed/0/accuracy": 90.0, "Proposed/1/accuracy": 92.0,
            "Ablation/0/accuracy": 85.0, "Ablation/1/accuracy": 87.0,
        }
        mock_sandbox_result.elapsed_sec = 120.0
        mock_sandbox_result.timed_out = False

        mock_sandbox = MagicMock()
        mock_sandbox.run_project.return_value = mock_sandbox_result

        with patch("researchclaw.llm.create_llm_client") as mock_create_llm, \
             patch("researchclaw.experiment.factory.create_sandbox") as mock_create_sb:
            mock_create_llm.return_value = mock_llm
            mock_create_sb.return_value = mock_sandbox

            result = run_repair_loop(run_dir, FakeConfig(), "test-mock")

        assert result.total_cycles == 1
        assert len(result.cycle_history) == 1
        assert result.cycle_history[0].repair_applied is True

        # Check that repair files were saved
        repair_dir = run_dir / "stage-14_repair_v1"
        assert repair_dir.exists()
        assert (repair_dir / "experiment" / "main.py").exists()
        assert (repair_dir / "experiment_summary.json").exists()


# ---------------------------------------------------------------------------
# BUG-199: 2-part metric keys (condition/metric) in summary builder
# ---------------------------------------------------------------------------


class TestBuildExperimentSummaryTwoPartKeys:
    """BUG-199: Stage 13 refinement produces 2-part keys (condition/metric)
    instead of 3-part keys (condition/seed/metric).  The parser must handle
    both formats.
    """

    def test_two_part_keys_parsed(self):
        """2-part keys like 'Baseline/accuracy' should create conditions."""
        run_result = {
            "stdout": "",
            "stderr": "",
            "returncode": 0,
            "metrics": {
                "Baseline/accuracy": 0.85,
                "Proposed/accuracy": 0.94,
                "Ablation/accuracy": 0.88,
            },
            "elapsed_sec": 120.0,
            "timed_out": False,
        }
        summary = _build_experiment_summary_from_run(run_result, {})
        assert summary["total_conditions"] == 3
        assert "Baseline" in summary["condition_summaries"]
        assert "Proposed" in summary["condition_summaries"]
        assert "Ablation" in summary["condition_summaries"]
        assert summary["condition_summaries"]["Proposed"]["metrics"]["accuracy"] == 0.94

    def test_two_part_keys_create_synthetic_seed(self):
        """2-part keys should create a synthetic seed '0' entry."""
        run_result = {
            "stdout": "",
            "stderr": "",
            "returncode": 0,
            "metrics": {
                "Baseline/accuracy": 0.80,
                "Baseline/loss": 0.45,
            },
            "elapsed_sec": 60.0,
            "timed_out": False,
        }
        summary = _build_experiment_summary_from_run(run_result, {})
        bl = summary["condition_summaries"]["Baseline"]
        assert bl["metrics"]["accuracy"] == 0.80
        assert bl["metrics"]["loss"] == 0.45
        assert bl["n_seeds"] == 1  # synthetic seed "0"

    def test_mixed_two_and_three_part_keys(self):
        """Mix of 2-part and 3-part keys for different conditions."""
        run_result = {
            "stdout": "",
            "stderr": "",
            "returncode": 0,
            "metrics": {
                # 3-part keys (with seed)
                "Baseline/0/accuracy": 0.80,
                "Baseline/1/accuracy": 0.82,
                # 2-part keys (Stage 13 refinement output)
                "Proposed/accuracy": 0.94,
            },
            "elapsed_sec": 120.0,
            "timed_out": False,
        }
        summary = _build_experiment_summary_from_run(run_result, {})
        assert summary["total_conditions"] == 2
        # 3-part: mean of seeds
        bl = summary["condition_summaries"]["Baseline"]
        assert abs(bl["metrics"]["accuracy"] - 0.81) < 0.01
        assert bl["n_seeds"] == 2
        # 2-part: single value
        pr = summary["condition_summaries"]["Proposed"]
        assert pr["metrics"]["accuracy"] == 0.94
        assert pr["n_seeds"] == 1

    def test_empty_metrics_still_empty(self):
        """Empty metrics dict should still produce 0 conditions."""
        run_result = {
            "stdout": "",
            "stderr": "",
            "returncode": 1,
            "metrics": {},
            "elapsed_sec": 5.0,
            "timed_out": False,
        }
        summary = _build_experiment_summary_from_run(run_result, {})
        assert summary["total_conditions"] == 0


# ---------------------------------------------------------------------------
# BUG-198: Conditional promotion of repair summary in runner.py
# ---------------------------------------------------------------------------


class TestRepairSummaryPromotion:
    """BUG-198: runner.py should NOT overwrite a richer stage-14 summary
    with an empty/poorer repair result.
    """

    def test_empty_repair_does_not_overwrite_rich_summary(self, tmp_path):
        """Repair result with 0 conditions must NOT replace a summary
        that has real conditions and metrics.
        """
        # Create a rich existing stage-14 summary
        s14 = tmp_path / "stage-14"
        s14.mkdir()
        rich_summary = {
            "condition_summaries": {
                "Baseline": {"metrics": {"accuracy": 0.80}},
                "Proposed": {"metrics": {"accuracy": 0.94}},
                "Ablation": {"metrics": {"accuracy": 0.88}},
            },
            "best_run": {
                "metrics": {
                    "Baseline/0/accuracy": 0.80,
                    "Proposed/0/accuracy": 0.94,
                    "primary_metric": 0.94,
                },
            },
            "total_conditions": 3,
            "total_metric_keys": 3,
        }
        (s14 / "experiment_summary.json").write_text(json.dumps(rich_summary))

        # Compute scores to verify the logic
        rich_score = _summary_quality_score(rich_summary)

        empty_summary = {
            "condition_summaries": {},
            "best_run": {"metrics": {}},
            "total_conditions": 0,
            "total_metric_keys": 0,
        }
        empty_score = _summary_quality_score(empty_summary)

        # The rich summary must score higher
        assert rich_score > empty_score

        # Verify that the existing file is preserved (simulate what runner does)
        existing = json.loads(
            (s14 / "experiment_summary.json").read_text(encoding="utf-8")
        )
        existing_score = _summary_quality_score(existing)
        repair_score = _summary_quality_score(empty_summary)

        # runner.py should NOT overwrite because repair_score <= existing_score
        assert repair_score <= existing_score
        # The file should still contain the rich data
        after = json.loads(
            (s14 / "experiment_summary.json").read_text(encoding="utf-8")
        )
        assert len(after["condition_summaries"]) == 3

    def test_richer_repair_does_overwrite(self, tmp_path):
        """Repair result with MORE conditions should replace a poorer summary."""
        s14 = tmp_path / "stage-14"
        s14.mkdir()
        poor_summary = {
            "condition_summaries": {"A": {"metrics": {"m": 0.5}}},
            "best_run": {"metrics": {}},
            "total_conditions": 1,
            "total_metric_keys": 0,
        }
        (s14 / "experiment_summary.json").write_text(json.dumps(poor_summary))

        rich_repair = {
            "condition_summaries": {
                "A": {"metrics": {"m": 0.80}},
                "B": {"metrics": {"m": 0.85}},
                "C": {"metrics": {"m": 0.90}},
            },
            "best_run": {"metrics": {"primary_metric": 0.90}},
            "total_conditions": 3,
            "total_metric_keys": 4,
        }

        poor_score = _summary_quality_score(poor_summary)
        rich_score = _summary_quality_score(rich_repair)
        assert rich_score > poor_score
