"""Tests for experiment_diagnosis — failure analysis agent."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from researchclaw.pipeline.experiment_diagnosis import (
    DeficiencyType,
    ExperimentDiagnosis,
    ExperimentQualityAssessment,
    PaperMode,
    assess_experiment_quality,
    diagnose_experiment,
)

ARTIFACTS = Path(__file__).resolve().parent.parent / "artifacts"


# ---------------------------------------------------------------------------
# Unit tests — individual checks
# ---------------------------------------------------------------------------


class TestMissingDependency:
    def test_detects_module_not_found(self):
        diag = diagnose_experiment(
            experiment_summary={"condition_summaries": {}, "best_run": {"metrics": {}}},
            stdout="",
            stderr="ModuleNotFoundError: No module named 'utils'",
        )
        types = {d.type for d in diag.deficiencies}
        assert DeficiencyType.MISSING_DEPENDENCY in types

    def test_detects_box2d(self):
        diag = diagnose_experiment(
            experiment_summary={"condition_summaries": {}, "best_run": {"metrics": {}}},
            stdout="BOX2D_WARNING: Box2D/LunarLander-v3 not available; skipping",
            stderr="",
        )
        types = {d.type for d in diag.deficiencies}
        assert DeficiencyType.MISSING_DEPENDENCY in types


class TestPermissionError:
    def test_detects_hf_permission(self):
        diag = diagnose_experiment(
            experiment_summary={"condition_summaries": {}, "best_run": {"metrics": {}}},
            stdout="",
            stderr="PermissionError: Cannot download huggingface model",
        )
        types = {d.type for d in diag.deficiencies}
        assert DeficiencyType.PERMISSION_ERROR in types


class TestTimeGuard:
    def test_detects_dominant_time_guard(self):
        summary = {
            "condition_summaries": {"CondA": {"metrics": {"metric": 80.0}}},
            "best_run": {"metrics": {}},
        }
        plan = {"conditions": [{"name": "CondA"}, {"name": "CondB"}, {"name": "CondC"}, {"name": "CondD"}]}
        diag = diagnose_experiment(
            experiment_summary=summary,
            experiment_plan=plan,
            stdout="TIME_GUARD: skipping CondB\nTIME_GUARD: skipping CondC\nTIME_GUARD: skipping CondD",
        )
        types = {d.type for d in diag.deficiencies}
        assert DeficiencyType.TIME_GUARD_DOMINANT in types

    def test_no_time_guard_if_most_complete(self):
        summary = {
            "condition_summaries": {
                "A": {"metrics": {"metric": 1.0}},
                "B": {"metrics": {"metric": 2.0}},
                "C": {"metrics": {"metric": 3.0}},
            },
            "best_run": {"metrics": {}},
        }
        plan = {"conditions": [{"name": "A"}, {"name": "B"}, {"name": "C"}, {"name": "D"}]}
        diag = diagnose_experiment(experiment_summary=summary, experiment_plan=plan)
        types = {d.type for d in diag.deficiencies}
        # 1/4 skipped = 25%, below 50% threshold
        assert DeficiencyType.TIME_GUARD_DOMINANT not in types


class TestSyntheticData:
    def test_detects_synthetic_fallback(self):
        diag = diagnose_experiment(
            experiment_summary={"condition_summaries": {}, "best_run": {"metrics": {}}},
            stdout="[data] WARNING: Alpaca load failed ... using synthetic data.",
        )
        types = {d.type for d in diag.deficiencies}
        assert DeficiencyType.SYNTHETIC_DATA_FALLBACK in types


class TestGPUOOM:
    def test_detects_oom(self):
        diag = diagnose_experiment(
            experiment_summary={"condition_summaries": {}, "best_run": {"metrics": {}}},
            stderr="RuntimeError: CUDA out of memory. Tried to allocate 2.00 GiB",
        )
        types = {d.type for d in diag.deficiencies}
        assert DeficiencyType.GPU_OOM in types


class TestIdenticalConditions:
    def test_detects_from_ablation_warnings(self):
        summary = {
            "condition_summaries": {"A": {"metrics": {"m": 1}}, "B": {"metrics": {"m": 1}}},
            "best_run": {"metrics": {}},
            "ablation_warnings": [
                "ABLATION FAILURE: Conditions 'A' and 'B' produce identical outputs across all 1 metrics."
            ],
        }
        diag = diagnose_experiment(experiment_summary=summary)
        types = {d.type for d in diag.deficiencies}
        assert DeficiencyType.IDENTICAL_CONDITIONS in types


class TestCodeCrash:
    def test_detects_traceback(self):
        diag = diagnose_experiment(
            experiment_summary={"condition_summaries": {}, "best_run": {"metrics": {}}},
            stderr=(
                "Traceback (most recent call last):\n"
                "  File 'main.py', line 42, in main\n"
                "    result = train(model)\n"
                "TypeError: train() missing argument 'data'\n"
            ),
        )
        types = {d.type for d in diag.deficiencies}
        assert DeficiencyType.CODE_CRASH in types


# ---------------------------------------------------------------------------
# Quality assessment
# ---------------------------------------------------------------------------


class TestQualityAssessment:
    def test_full_paper_mode(self):
        summary = {
            "condition_summaries": {
                "A": {"metrics": {"metric": 80.0}},
                "B": {"metrics": {"metric": 85.0}},
                "C": {"metrics": {"metric": 90.0}},
            },
            "best_run": {
                "metrics": {
                    "A/0/m": 80.0, "A/1/m": 81.0,
                    "B/0/m": 85.0, "B/1/m": 86.0,
                    "C/0/m": 90.0, "C/1/m": 91.0,
                },
            },
        }
        qa = assess_experiment_quality(summary)
        assert qa.mode == PaperMode.FULL_PAPER
        assert qa.sufficient

    def test_preliminary_study_mode(self):
        summary = {
            "condition_summaries": {
                "A": {"metrics": {"metric": 80.0}},
                "B": {"metrics": {"metric": 85.0}},
            },
            "best_run": {"metrics": {"A/0/m": 80.0, "B/0/m": 85.0}},
        }
        qa = assess_experiment_quality(summary)
        assert qa.mode == PaperMode.PRELIMINARY_STUDY
        assert not qa.sufficient

    def test_technical_report_no_conditions(self):
        summary = {
            "condition_summaries": {},
            "best_run": {"metrics": {}},
        }
        qa = assess_experiment_quality(summary)
        assert qa.mode == PaperMode.TECHNICAL_REPORT
        assert not qa.sufficient

    def test_technical_report_synthetic_data(self):
        summary = {
            "condition_summaries": {"A": {"metrics": {"metric": 80.0}}},
            "best_run": {"metrics": {}, "stdout": "using synthetic data"},
        }
        qa = assess_experiment_quality(summary)
        assert qa.mode == PaperMode.TECHNICAL_REPORT


# ---------------------------------------------------------------------------
# Repair prompt generation
# ---------------------------------------------------------------------------


class TestRepairPrompt:
    def test_generates_prompt(self):
        diag = diagnose_experiment(
            experiment_summary={"condition_summaries": {}, "best_run": {"metrics": {}}},
            stderr="ModuleNotFoundError: No module named 'special_lib'",
        )
        prompt = diag.to_repair_prompt()
        assert "special_lib" in prompt
        assert "DIAGNOSIS" in prompt
        assert "CRITICAL" in prompt

    def test_serialization(self):
        diag = diagnose_experiment(
            experiment_summary={"condition_summaries": {"A": {"metrics": {"m": 1}}}, "best_run": {"metrics": {}}},
        )
        d = diag.to_dict()
        assert isinstance(d, dict)
        assert "deficiencies" in d
        assert "conditions_completed" in d


# ---------------------------------------------------------------------------
# Integration — real artifacts
# ---------------------------------------------------------------------------


class TestRealArtifacts:
    def _load(self, run_id: str) -> tuple[dict, dict | None]:
        pattern = f"rc-*-{run_id}"
        matches = sorted(ARTIFACTS.glob(pattern))
        if not matches:
            pytest.skip(f"Artifact {run_id} not found")
        base = matches[0]
        summary_path = base / "stage-14" / "experiment_summary.json"
        ref_path = base / "stage-13" / "refinement_log.json"
        if not summary_path.exists():
            pytest.skip(f"No experiment_summary for {run_id}")
        summary = json.loads(summary_path.read_text())
        ref_log = json.loads(ref_path.read_text()) if ref_path.exists() else None
        return summary, ref_log

    def test_run_e57360_diagnosis(self):
        """Run 38 — 3/8 conditions completed, Box2D missing."""
        summary, ref_log = self._load("e57360")
        qa = assess_experiment_quality(summary, ref_log)
        # Should identify issues and NOT rate as full_paper
        assert qa.mode != PaperMode.FULL_PAPER or len(qa.deficiencies) > 0

    def test_run_8b4a1b_diagnosis(self):
        """Run 8b4a1b — all NaN, permission errors."""
        summary, ref_log = self._load("8b4a1b")
        qa = assess_experiment_quality(summary, ref_log)
        # Should be technical_report or preliminary_study at best
        assert qa.mode in (PaperMode.TECHNICAL_REPORT, PaperMode.PRELIMINARY_STUDY)

class TestDatasetNotFoundError:
    """BUG-203: HuggingFace DatasetNotFoundError should be caught."""

    def test_detects_hf_dataset_not_found(self):
        stderr = (
            "Traceback (most recent call last):\n"
            "  File \"/workspace/setup.py\", line 11, in main\n"
            "datasets.exceptions.DatasetNotFoundError: "
            "Dataset 'cifar10_corrupted' doesn't exist on the Hub or cannot be accessed.\n"
        )
        diag = diagnose_experiment(
            experiment_summary={"condition_summaries": {}, "best_run": {"metrics": {}}},
            stderr=stderr,
        )
        ds_issues = [d for d in diag.deficiencies if d.type == DeficiencyType.DATASET_UNAVAILABLE]
        assert len(ds_issues) >= 1
        assert "HuggingFace" in ds_issues[0].description
        # Should NOT also appear as a generic CODE_CRASH
        crashes = [d for d in diag.deficiencies if d.type == DeficiencyType.CODE_CRASH]
        assert not any("DatasetNotFoundError" in c.description for c in crashes)

    def test_suggested_fix_mentions_precached(self):
        stderr = (
            "DatasetNotFoundError: Dataset 'imagenet_v2' "
            "doesn't exist on the Hub or cannot be accessed.\n"
        )
        diag = diagnose_experiment(
            experiment_summary={"condition_summaries": {}, "best_run": {"metrics": {}}},
            stderr=stderr,
        )
        ds_issues = [d for d in diag.deficiencies if d.type == DeficiencyType.DATASET_UNAVAILABLE]
        assert any("/opt/datasets" in d.suggested_fix for d in ds_issues)


class TestNearRandomAccuracy:
    """BUG-204: Detect near-random accuracy in experiment results."""

    def test_detects_near_random_cifar10(self):
        """8.91% accuracy on CIFAR-10 should be flagged."""
        diag = diagnose_experiment(
            experiment_summary={
                "condition_summaries": {"cond_a": {"metrics": {"top1_accuracy": 8.91}}},
                "metrics_summary": {"top1_accuracy": {"min": 8.42, "max": 8.91, "mean": 8.67}},
                "best_run": {"metrics": {}},
            },
        )
        hp_issues = [d for d in diag.deficiencies if d.type == DeficiencyType.HYPERPARAMETER_ISSUE]
        assert any("random chance" in d.description for d in hp_issues)

    def test_normal_accuracy_not_flagged(self):
        """73% accuracy should NOT be flagged."""
        diag = diagnose_experiment(
            experiment_summary={
                "condition_summaries": {"baseline": {"metrics": {"accuracy": 73.07}}},
                "metrics_summary": {"accuracy": {"min": 68.0, "max": 73.07, "mean": 70.5}},
                "best_run": {"metrics": {}},
            },
        )
        hp_issues = [d for d in diag.deficiencies if d.type == DeficiencyType.HYPERPARAMETER_ISSUE]
        assert not any("random chance" in d.description for d in hp_issues)

    def test_zero_accuracy_not_flagged(self):
        """0% accuracy (no data) should NOT be flagged by this check."""
        diag = diagnose_experiment(
            experiment_summary={
                "condition_summaries": {},
                "metrics_summary": {},
                "best_run": {"metrics": {}},
            },
        )
        hp_issues = [d for d in diag.deficiencies if d.type == DeficiencyType.HYPERPARAMETER_ISSUE]
        assert not any("random chance" in d.description for d in hp_issues)


class TestRealArtifactsContinued(TestRealArtifacts):
    """Continuation of real artifact tests (after TestDatasetNotFoundError)."""

    def test_run_acbdfa_diagnosis(self):
        """Run acbdfa — 2 architectures, S4D nearly random."""
        summary, ref_log = self._load("acbdfa")
        diag = diagnose_experiment(
            experiment_summary=summary,
            refinement_log=ref_log,
            stdout=summary.get("best_run", {}).get("stdout", ""),
            stderr=summary.get("best_run", {}).get("stderr", ""),
        )
        assert diag.completion_rate > 0
