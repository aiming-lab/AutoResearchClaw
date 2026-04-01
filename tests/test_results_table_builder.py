"""Tests for results_table_builder — pre-built LaTeX tables."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from researchclaw.pipeline.verified_registry import VerifiedRegistry
from researchclaw.templates.results_table_builder import (
    LatexTable,
    build_condition_whitelist,
    build_results_tables,
)

ARTIFACTS = Path(__file__).resolve().parent.parent / "artifacts"


def _make_registry(
    conditions: dict[str, dict[int, float]],
    primary_metric: float | None = None,
) -> VerifiedRegistry:
    """Create a registry from simple condition → {seed: value} mapping."""
    summary = {"best_run": {"metrics": {}}, "condition_summaries": {}, "metrics_summary": {}}
    for cond_name, seeds in conditions.items():
        for seed_idx, value in seeds.items():
            key = f"{cond_name}/{seed_idx}/metric"
            summary["best_run"]["metrics"][key] = value
        cond_metric = sum(seeds.values()) / len(seeds) if seeds else 0
        summary["condition_summaries"][cond_name] = {"metrics": {"metric": cond_metric}}
    if primary_metric is not None:
        summary["best_run"]["metrics"]["primary_metric"] = primary_metric
    return VerifiedRegistry.from_experiment(summary)


class TestBuildResultsTables:
    def test_basic_table(self):
        reg = _make_registry(
            {
                "Baseline": {0: 80.0, 1: 82.0, 2: 81.0},
                "Proposed": {0: 85.0, 1: 87.0, 2: 86.0},
            },
            primary_metric=86.0,
        )
        tables = build_results_tables(reg, metric_name="Accuracy (\\%)")
        assert len(tables) == 2  # main + per-seed
        main = tables[0]
        assert main.label == "tab:main_results"
        assert "AUTO-GENERATED" in main.latex_code
        assert "\\begin{table}" in main.latex_code
        assert "Baseline" in main.latex_code
        assert "Proposed" in main.latex_code
        assert main.n_conditions == 2

    def test_best_is_bolded(self):
        reg = _make_registry(
            {
                "Baseline": {0: 70.0, 1: 72.0},
                "Proposed": {0: 85.0, 1: 87.0},
            }
        )
        tables = build_results_tables(reg, metric_direction="maximize")
        main = tables[0]
        # Proposed should be bold (higher metric)
        assert "\\textbf" in main.latex_code

    def test_single_seed_marker(self):
        reg = _make_registry(
            {
                "Baseline": {0: 80.0, 1: 82.0},
                "Proposed": {0: 90.0},  # Single seed
            }
        )
        tables = build_results_tables(reg)
        main = tables[0]
        assert "\\ddagger" in main.latex_code  # Single-seed footnote

    def test_no_conditions(self):
        reg = VerifiedRegistry()
        tables = build_results_tables(reg)
        assert len(tables) == 0

    def test_all_single_seed_no_per_seed_table(self):
        reg = _make_registry(
            {
                "A": {0: 80.0},
                "B": {0: 70.0},
            }
        )
        tables = build_results_tables(reg)
        # Only 1 table (main), no per-seed table (all single seed)
        assert len(tables) == 1

    def test_per_seed_table_structure(self):
        reg = _make_registry(
            {
                "DQN": {0: 156.1, 1: 105.5, 2: 356.7},
                "DQN+Abstraction": {0: 98.1, 1: 456.7, 2: 282.0},
            }
        )
        tables = build_results_tables(reg)
        assert len(tables) == 2
        seed_table = tables[1]
        assert seed_table.label == "tab:per_seed"
        assert "156.10" in seed_table.latex_code or "156.1" in seed_table.latex_code
        assert "Seed 0" in seed_table.latex_code

    def test_two_column_uses_table_star(self):
        reg = _make_registry({"A": {0: 80.0, 1: 82.0}})
        tables = build_results_tables(reg, two_column=True)
        assert "\\begin{table*}" in tables[0].latex_code

    def test_verified_values_populated(self):
        reg = _make_registry(
            {"A": {0: 80.0, 1: 82.0}, "B": {0: 70.0, 1: 72.0}}
        )
        tables = build_results_tables(reg)
        main = tables[0]
        assert 81.0 in main.verified_values or any(
            abs(v - 81.0) < 0.01 for v in main.verified_values
        )

    def test_special_chars_escaped(self):
        reg = _make_registry({"DQN+Raw_Count": {0: 80.0, 1: 82.0}})
        tables = build_results_tables(reg)
        assert "DQN+Raw\\_Count" in tables[0].latex_code

    def test_minimize_direction(self):
        reg = _make_registry(
            {
                "Baseline": {0: 20.0, 1: 22.0},
                "Proposed": {0: 10.0, 1: 12.0},
            }
        )
        tables = build_results_tables(reg, metric_direction="minimize")
        # Proposed (lower) should be bold
        lines = tables[0].latex_code.split("\n")
        proposed_line = [l for l in lines if "Proposed" in l][0]
        assert "\\textbf" in proposed_line


class TestConditionWhitelist:
    def test_basic(self):
        reg = _make_registry(
            {
                "DQN": {0: 206.1, 1: 105.5, 2: 356.7},
                "DQN+Abstraction": {0: 278.93},
            }
        )
        wl = build_condition_whitelist(reg)
        assert "DQN" in wl
        assert "DQN+Abstraction" in wl
        assert "3 seed(s)" in wl
        assert "1 seed(s)" in wl

    def test_empty_registry(self):
        reg = VerifiedRegistry()
        wl = build_condition_whitelist(reg)
        assert "no conditions completed" in wl


class TestRealArtifacts:
    def _load(self, run_id: str) -> VerifiedRegistry:
        pattern = f"rc-*-{run_id}"
        matches = sorted(ARTIFACTS.glob(pattern))
        if not matches:
            pytest.skip(f"Artifact {run_id} not found")
        summary_path = matches[0] / "stage-14" / "experiment_summary.json"
        ref_path = matches[0] / "stage-13" / "refinement_log.json"
        if not summary_path.exists():
            pytest.skip(f"No experiment_summary for {run_id}")
        summary = json.loads(summary_path.read_text())
        ref_log = None
        if ref_path.exists():
            ref_log = json.loads(ref_path.read_text())
        return VerifiedRegistry.from_experiment(summary, ref_log)

    def test_run_e57360_rl_tables(self):
        reg = self._load("e57360")
        tables = build_results_tables(reg, metric_name="Return")
        assert len(tables) >= 1
        main = tables[0]
        # Should NOT contain PPO (never ran)
        assert "PPO" not in main.latex_code
        # Should contain DQN
        assert "DQN" in main.latex_code

    def test_run_acbdfa_tables(self):
        reg = self._load("acbdfa")
        tables = build_results_tables(reg, metric_name="Top-1 Accuracy (\\%)")
        assert len(tables) >= 1
