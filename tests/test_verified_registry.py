"""Tests for VerifiedRegistry — ground truth number whitelist."""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from researchclaw.pipeline.verified_registry import (
    ConditionResult,
    VerifiedRegistry,
    _is_finite,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

ARTIFACTS = Path(__file__).resolve().parent.parent / "artifacts"


def _load_experiment_summary(run_id: str) -> dict:
    """Load experiment_summary.json for a given run."""
    pattern = f"rc-*-{run_id}"
    matches = sorted(ARTIFACTS.glob(pattern))
    if not matches:
        pytest.skip(f"Artifact {run_id} not found")
    summary_path = matches[0] / "stage-14" / "experiment_summary.json"
    if not summary_path.exists():
        pytest.skip(f"No experiment_summary for {run_id}")
    return json.loads(summary_path.read_text())


def _load_refinement_log(run_id: str) -> dict | None:
    pattern = f"rc-*-{run_id}"
    matches = sorted(ARTIFACTS.glob(pattern))
    if not matches:
        return None
    log_path = matches[0] / "stage-13" / "refinement_log.json"
    if not log_path.exists():
        return None
    return json.loads(log_path.read_text())


# ---------------------------------------------------------------------------
# Unit tests — ConditionResult
# ---------------------------------------------------------------------------


class TestConditionResult:
    def test_compute_stats_multiple_seeds(self):
        cr = ConditionResult(name="test", per_seed_values={0: 10.0, 1: 20.0, 2: 30.0})
        cr.compute_stats()
        assert cr.n_seeds == 3
        assert cr.mean == pytest.approx(20.0)
        assert cr.std == pytest.approx(10.0)

    def test_compute_stats_single_seed(self):
        cr = ConditionResult(name="test", per_seed_values={0: 42.0})
        cr.compute_stats()
        assert cr.n_seeds == 1
        assert cr.mean == pytest.approx(42.0)
        assert cr.std == 0.0

    def test_compute_stats_with_nan(self):
        cr = ConditionResult(
            name="test", per_seed_values={0: 10.0, 1: float("nan"), 2: 30.0}
        )
        cr.compute_stats()
        assert cr.n_seeds == 2  # NaN excluded
        assert cr.mean == pytest.approx(20.0)

    def test_compute_stats_empty(self):
        cr = ConditionResult(name="test")
        cr.compute_stats()
        assert cr.n_seeds == 0
        assert cr.mean is None


# ---------------------------------------------------------------------------
# Unit tests — VerifiedRegistry core operations
# ---------------------------------------------------------------------------


class TestVerifiedRegistryCore:
    def test_add_value(self):
        reg = VerifiedRegistry()
        reg.add_value(74.28, "test_source")
        assert reg.is_verified(74.28)
        # Rounding variant
        assert reg.is_verified(74.3, tolerance=0.01)

    def test_percentage_conversion(self):
        """Value in [0,1] should also register value*100."""
        reg = VerifiedRegistry()
        reg.add_value(0.7428, "accuracy_fraction")
        assert reg.is_verified(0.7428)
        assert reg.is_verified(74.28)  # ×100 variant

    def test_reverse_percentage(self):
        """Value > 1 should also register value/100."""
        reg = VerifiedRegistry()
        reg.add_value(74.28, "accuracy_percent")
        assert reg.is_verified(74.28)
        assert reg.is_verified(0.7428)  # ÷100 variant

    def test_tolerance_matching(self):
        reg = VerifiedRegistry()
        reg.add_value(92.14, "test")
        # Within 1% tolerance
        assert reg.is_verified(92.14)
        assert reg.is_verified(92.0, tolerance=0.01)  # 0.15% off
        # Outside tolerance
        assert not reg.is_verified(95.0, tolerance=0.01)

    def test_zero_handling(self):
        reg = VerifiedRegistry()
        reg.add_value(0.0, "zero_metric")
        assert reg.is_verified(0.0)
        assert reg.is_verified(1e-8)  # Very close to zero
        assert not reg.is_verified(0.01)  # Not close enough

    def test_negative_values(self):
        reg = VerifiedRegistry()
        reg.add_value(-459.6, "bad_return")
        assert reg.is_verified(-459.6)
        assert reg.is_verified(-460.0, tolerance=0.01)

    def test_nan_inf_rejected(self):
        reg = VerifiedRegistry()
        reg.add_value(float("nan"), "nan_metric")
        reg.add_value(float("inf"), "inf_metric")
        assert not reg.is_verified(float("nan"))
        assert not reg.is_verified(float("inf"))
        assert len(reg.values) == 0

    def test_lookup(self):
        reg = VerifiedRegistry()
        reg.add_value(42.0, "the_answer")
        assert reg.lookup(42.0) == "the_answer"
        assert reg.lookup(999.0) is None

    def test_verify_condition(self):
        reg = VerifiedRegistry()
        reg.condition_names = {"DQN", "DQN+Abstraction"}
        assert reg.verify_condition("DQN")
        assert not reg.verify_condition("PPO")


# ---------------------------------------------------------------------------
# Unit tests — from_experiment (synthetic data)
# ---------------------------------------------------------------------------


class TestFromExperiment:
    def _make_summary(self) -> dict:
        return {
            "metrics_summary": {
                "CondA/0/metric": {"min": 80.0, "max": 80.0, "mean": 80.0, "count": 1},
                "CondA/1/metric": {"min": 85.0, "max": 85.0, "mean": 85.0, "count": 1},
                "CondB/0/metric": {"min": 70.0, "max": 70.0, "mean": 70.0, "count": 1},
                "primary_metric": {"min": 82.5, "max": 82.5, "mean": 82.5, "count": 1},
            },
            "best_run": {
                "metrics": {
                    "CondA/0/metric": 80.0,
                    "CondA/1/metric": 85.0,
                    "CondB/0/metric": 70.0,
                    "primary_metric": 82.5,
                    "primary_metric_std": 3.5355,
                    "total_elapsed_seconds": 1500.0,
                },
            },
            "condition_summaries": {
                "CondA": {"metrics": {"metric": 82.5}},
                "CondB": {"metrics": {"metric": 70.0}},
            },
            "condition_metrics": {
                "CondA": {"metrics": {"metric": 82.5}},
                "CondB": {"metrics": {"metric": 70.0}},
            },
            "total_conditions": 2,
        }

    def test_conditions_extracted(self):
        reg = VerifiedRegistry.from_experiment(self._make_summary())
        assert "CondA" in reg.condition_names
        assert "CondB" in reg.condition_names
        assert len(reg.condition_names) == 2

    def test_per_seed_values(self):
        reg = VerifiedRegistry.from_experiment(self._make_summary())
        assert reg.conditions["CondA"].per_seed_values == {0: 80.0, 1: 85.0}
        assert reg.conditions["CondB"].per_seed_values == {0: 70.0}

    def test_condition_stats(self):
        reg = VerifiedRegistry.from_experiment(self._make_summary())
        cond_a = reg.conditions["CondA"]
        assert cond_a.n_seeds == 2
        assert cond_a.mean == pytest.approx(82.5)
        assert cond_a.std == pytest.approx(3.5355, rel=0.01)

    def test_primary_metric(self):
        reg = VerifiedRegistry.from_experiment(self._make_summary())
        assert reg.primary_metric == pytest.approx(82.5)
        assert reg.primary_metric_std == pytest.approx(3.5355)

    def test_all_values_registered(self):
        reg = VerifiedRegistry.from_experiment(self._make_summary())
        # Core values must be verified
        assert reg.is_verified(80.0)
        assert reg.is_verified(85.0)
        assert reg.is_verified(70.0)
        assert reg.is_verified(82.5)
        assert reg.is_verified(3.5355, tolerance=0.01)

    def test_pairwise_differences(self):
        reg = VerifiedRegistry.from_experiment(self._make_summary())
        diff = 82.5 - 70.0  # CondA.mean - CondB.mean
        assert reg.is_verified(diff)
        assert reg.is_verified(abs(diff))

    def test_fabricated_number_rejected(self):
        reg = VerifiedRegistry.from_experiment(self._make_summary())
        assert not reg.is_verified(99.99)
        assert not reg.is_verified(60.51)

    def test_infra_keys_excluded(self):
        reg = VerifiedRegistry.from_experiment(self._make_summary())
        # total_elapsed_seconds goes to training_config, not values
        assert 1500.0 not in reg.values
        assert reg.training_config.get("total_elapsed_seconds") == 1500.0

    def test_with_refinement_log(self):
        summary = self._make_summary()
        ref_log = {
            "best_metric": 82.5,
            "best_version": "experiment_v1/",
            "iterations": [
                {
                    "version_dir": "experiment_v1/",
                    "metric": 82.5,
                    "sandbox": {"metrics": {"CondA/0/metric": 80.0}},
                }
            ],
        }
        reg = VerifiedRegistry.from_experiment(summary, ref_log)
        assert reg.is_verified(82.5)


# ---------------------------------------------------------------------------
# Integration tests — real artifact data
# ---------------------------------------------------------------------------


class TestRealArtifacts:
    """Test against actual pipeline output.  Skipped if artifacts not present."""

    def test_run_e57360_rl_exploration(self):
        """Run 38 (RL LACE) — 3 conditions, CartPole + Acrobot."""
        summary = _load_experiment_summary("e57360")
        ref_log = _load_refinement_log("e57360")
        reg = VerifiedRegistry.from_experiment(summary, ref_log)

        # Conditions that actually ran
        assert reg.verify_condition("DQN")
        assert reg.verify_condition("DQN+Abstraction")
        assert reg.verify_condition("DQN+RawCount")

        # Conditions that did NOT run (paper fabricated these)
        assert not reg.verify_condition("PPO")
        assert not reg.verify_condition("PPO+Abstraction")
        assert not reg.verify_condition("DQN+Autoencoder")

        # Real primary metric
        assert reg.is_verified(278.9333)
        assert reg.is_verified(146.4139, tolerance=0.01)

        # Fabricated number from paper (0.0 primary metric) — should NOT verify
        # unless 0.0 happens to be in the data for another reason
        # The paper claimed primary_metric=0.0 which is fabricated
        assert reg.primary_metric == pytest.approx(278.9333)

    def test_run_acbdfa_cnn_vs_ssm(self):
        """Run acbdfa (CTS) — ResNet vs S4D on CIFAR-100."""
        summary = _load_experiment_summary("acbdfa")
        reg = VerifiedRegistry.from_experiment(summary)

        # Real values from experiment
        assert reg.is_verified(69.99)
        assert reg.is_verified(69.93)
        assert reg.is_verified(58.66)
        assert reg.is_verified(2.75)

        # Primary metric
        assert reg.is_verified(66.1933, tolerance=0.01)

    def test_run_85fefc_contrastive_kd(self):
        """Run 85fefc (CRAFT) — contrastive KD."""
        summary = _load_experiment_summary("85fefc")
        ref_log = _load_refinement_log("85fefc")
        reg = VerifiedRegistry.from_experiment(summary, ref_log)

        # Should have conditions
        assert len(reg.condition_names) > 0

        # Primary metric should be registered
        assert reg.primary_metric is not None

    def test_run_8b4a1b_gard_lora(self):
        """Run 8b4a1b (GARD) — experiment failed, very few values."""
        summary = _load_experiment_summary("8b4a1b")
        reg = VerifiedRegistry.from_experiment(summary)

        # With empty metrics, registry should be sparse
        best_metrics = summary.get("best_run", {}).get("metrics", {})
        if not best_metrics:
            assert len(reg.values) == 0


# ---------------------------------------------------------------------------
# Unit tests — from_run_dir (merges multiple sources)
# ---------------------------------------------------------------------------


class TestFromRunDir:
    def _write_summary(self, path: Path, data: dict) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    def test_from_run_dir_merges_multiple_stage14(self, tmp_path: Path) -> None:
        """Two stage-14 dirs with different values → both present."""
        run_dir = tmp_path / "run"
        run_dir.mkdir()
        # Stage-14 with CondA
        self._write_summary(
            run_dir / "stage-14" / "experiment_summary.json",
            {
                "best_run": {"metrics": {"CondA/0/metric": 80.0}},
                "condition_summaries": {"CondA": {"metrics": {"metric": 80.0}}},
                "metrics_summary": {},
            },
        )
        # Stage-14-v2 with CondB
        self._write_summary(
            run_dir / "stage-14-v2" / "experiment_summary.json",
            {
                "best_run": {"metrics": {"CondB/0/metric": 90.0}},
                "condition_summaries": {"CondB": {"metrics": {"metric": 90.0}}},
                "metrics_summary": {},
            },
        )
        reg = VerifiedRegistry.from_run_dir(run_dir)
        assert "CondA" in reg.condition_names
        assert "CondB" in reg.condition_names
        assert reg.is_verified(80.0)
        assert reg.is_verified(90.0)

    def test_from_run_dir_includes_best(self, tmp_path: Path) -> None:
        """experiment_summary_best.json values merged."""
        run_dir = tmp_path / "run"
        run_dir.mkdir()
        # Only best summary at root level
        self._write_summary(
            run_dir / "experiment_summary_best.json",
            {
                "best_run": {"metrics": {"primary_metric": 0.95}},
                "condition_summaries": {"Proposed": {"metrics": {"acc": 0.95}}},
                "metrics_summary": {"acc": {"mean": 0.95, "min": 0.95, "max": 0.95}},
            },
        )
        reg = VerifiedRegistry.from_run_dir(run_dir)
        assert reg.is_verified(0.95)
        assert reg.is_verified(95.0)  # percentage variant
        assert "Proposed" in reg.condition_names

    def test_from_run_dir_empty_dir(self, tmp_path: Path) -> None:
        """Empty run dir → empty registry, no crash."""
        run_dir = tmp_path / "empty_run"
        run_dir.mkdir()
        reg = VerifiedRegistry.from_run_dir(run_dir)
        assert len(reg.values) == 0
        assert len(reg.condition_names) == 0

    # -----------------------------------------------------------------------
    # BUG-222: best_only mode — REFINE bypass prevention
    # -----------------------------------------------------------------------

    def test_best_only_uses_experiment_summary_best(self, tmp_path: Path) -> None:
        """best_only=True should use ONLY experiment_summary_best.json."""
        run_dir = tmp_path / "run"
        run_dir.mkdir()
        # v1 (best): FeatureKD 74.52%
        self._write_summary(
            run_dir / "experiment_summary_best.json",
            {
                "best_run": {"metrics": {"FeatureKD/0/metric": 0.7452}},
                "condition_summaries": {"FeatureKD": {"metrics": {"metric": 0.7452}}},
                "metrics_summary": {"metric": {"mean": 0.7452}},
            },
        )
        # v3 (regressed): FeatureKD 69.30%
        self._write_summary(
            run_dir / "stage-14" / "experiment_summary.json",
            {
                "best_run": {"metrics": {"FeatureKD/0/metric": 0.6930}},
                "condition_summaries": {"FeatureKD": {"metrics": {"metric": 0.6930}}},
                "metrics_summary": {"metric": {"mean": 0.6930}},
            },
        )

        reg = VerifiedRegistry.from_run_dir(run_dir, best_only=True)
        # Should ONLY have v1 (best) data
        assert reg.is_verified(0.7452)
        assert reg.is_verified(74.52)  # percentage variant
        # Should NOT have v3 (regressed) data
        assert not reg.is_verified(0.6930)
        assert not reg.is_verified(69.30)

    def test_best_only_excludes_refinement_log(self, tmp_path: Path) -> None:
        """best_only=True should NOT merge refinement_log.json sandbox data."""
        run_dir = tmp_path / "run"
        run_dir.mkdir()
        # Best summary
        self._write_summary(
            run_dir / "experiment_summary_best.json",
            {
                "best_run": {"metrics": {"primary_metric": 0.7452}},
                "condition_summaries": {"FeatureKD": {"metrics": {"metric": 0.7452}}},
                "metrics_summary": {"metric": {"mean": 0.7452}},
            },
        )
        # Refinement log with sandbox metrics from regressed iteration
        rl_dir = run_dir / "stage-13"
        rl_dir.mkdir(parents=True)
        (rl_dir / "refinement_log.json").write_text(json.dumps({
            "iterations": [
                {"sandbox": {"metrics": {"primary_metric": 0.6930, "best_metric": 0.6930}}}
            ]
        }), encoding="utf-8")

        reg = VerifiedRegistry.from_run_dir(run_dir, best_only=True)
        assert reg.is_verified(0.7452)
        assert not reg.is_verified(0.6930), "Refinement log sandbox values should NOT be in best_only registry"

    def test_best_only_falls_back_to_stage14(self, tmp_path: Path) -> None:
        """best_only=True without best.json falls back to stage-14/ (non-versioned)."""
        run_dir = tmp_path / "run"
        run_dir.mkdir()
        self._write_summary(
            run_dir / "stage-14" / "experiment_summary.json",
            {
                "best_run": {"metrics": {"metric": 0.85}},
                "condition_summaries": {"Baseline": {"metrics": {"metric": 0.85}}},
                "metrics_summary": {"metric": {"mean": 0.85}},
            },
        )
        reg = VerifiedRegistry.from_run_dir(run_dir, best_only=True)
        assert reg.is_verified(0.85)
        assert "Baseline" in reg.condition_names

    def test_default_mode_still_merges_all(self, tmp_path: Path) -> None:
        """Default (best_only=False) preserves backward-compat merging."""
        run_dir = tmp_path / "run"
        run_dir.mkdir()
        self._write_summary(
            run_dir / "experiment_summary_best.json",
            {
                "best_run": {"metrics": {"FeatureKD/0/metric": 0.7452}},
                "condition_summaries": {"FeatureKD": {"metrics": {"metric": 0.7452}}},
                "metrics_summary": {},
            },
        )
        self._write_summary(
            run_dir / "stage-14" / "experiment_summary.json",
            {
                "best_run": {"metrics": {"FeatureKD/0/metric": 0.6930}},
                "condition_summaries": {"FeatureKD": {"metrics": {"metric": 0.6930}}},
                "metrics_summary": {},
            },
        )
        reg = VerifiedRegistry.from_run_dir(run_dir, best_only=False)
        # Both should be present in non-best_only mode
        assert reg.is_verified(0.7452)
        assert reg.is_verified(0.6930)
