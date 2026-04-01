"""Tests for the universal metric parser."""

from __future__ import annotations

import json
import math
import pytest
from pathlib import Path

from researchclaw.experiment.metrics import (
    ExperimentResults,
    MetricType,
    UniversalMetricParser,
)


@pytest.fixture
def parser():
    return UniversalMetricParser()


@pytest.fixture
def tmp_run_dir(tmp_path):
    return tmp_path


# ---------------------------------------------------------------------------
# JSON parsing tests
# ---------------------------------------------------------------------------


class TestJSONParsing:
    def test_parse_comparison_results(self, parser, tmp_run_dir):
        data = {
            "experiment_type": "comparison",
            "conditions": {
                "proposed_method": {
                    "seed_42": {"accuracy": 0.95, "f1": 0.93},
                    "seed_123": {"accuracy": 0.94, "f1": 0.92},
                },
                "baseline": {
                    "seed_42": {"accuracy": 0.88, "f1": 0.85},
                },
            },
            "metadata": {
                "domain": "ml_vision",
                "total_runtime_sec": 120.5,
            },
        }
        (tmp_run_dir / "results.json").write_text(json.dumps(data))

        result = parser.parse(tmp_run_dir)
        assert result.source == "json"
        assert result.experiment_type == "comparison"
        assert result.domain == "ml_vision"
        assert "proposed_method" in result.conditions
        flat = result.to_flat_metrics()
        assert "proposed_method/accuracy" in flat

    def test_parse_convergence_results(self, parser, tmp_run_dir):
        data = {
            "experiment_type": "convergence",
            "convergence": {
                "euler": [
                    {"h": 0.1, "error": 0.05},
                    {"h": 0.05, "error": 0.012},
                    {"h": 0.025, "error": 0.003},
                ],
                "rk4": [
                    {"h": 0.1, "error": 0.001},
                    {"h": 0.05, "error": 6.25e-5},
                    {"h": 0.025, "error": 3.9e-6},
                ],
            },
        }
        (tmp_run_dir / "results.json").write_text(json.dumps(data))

        result = parser.parse(tmp_run_dir)
        assert result.source == "json"
        assert "euler" in result.convergence
        assert len(result.convergence["euler"]) == 3
        flat = result.to_flat_metrics()
        assert "euler/error" in flat  # last point

    def test_parse_regression_table(self, parser, tmp_run_dir):
        data = {
            "experiment_type": "progressive_spec",
            "regression_table": {
                "spec_1_ols": {"coeff": 0.15, "se": 0.03, "p": 0.001, "n": 5000, "r2": 0.12},
                "spec_2_fe": {"coeff": 0.11, "se": 0.02, "p": 0.001, "n": 5000, "r2": 0.35},
            },
        }
        (tmp_run_dir / "results.json").write_text(json.dumps(data))

        result = parser.parse(tmp_run_dir)
        assert result.source == "json"
        assert "spec_1_ols" in result.regression_table
        flat = result.to_flat_metrics()
        assert "spec_1_ols/coeff" in flat
        assert flat["spec_1_ols/coeff"] == 0.15

    def test_parse_top_level_scalars(self, parser, tmp_run_dir):
        data = {"accuracy": 0.95, "loss": 0.32}
        (tmp_run_dir / "results.json").write_text(json.dumps(data))

        result = parser.parse(tmp_run_dir)
        assert result.scalars["accuracy"] == 0.95
        assert result.scalars["loss"] == 0.32

    def test_skip_nan_inf(self, parser, tmp_run_dir):
        data = {
            "conditions": {
                "method": {
                    "seed_1": {"accuracy": float("nan"), "f1": 0.9},
                },
            },
        }
        (tmp_run_dir / "results.json").write_text(json.dumps(data))

        result = parser.parse(tmp_run_dir)
        flat = result.to_flat_metrics()
        # NaN should be excluded
        for k, v in flat.items():
            assert math.isfinite(v), f"Non-finite value: {k}={v}"

    def test_invalid_json_falls_through(self, parser, tmp_run_dir):
        (tmp_run_dir / "results.json").write_text("not valid json{{{")
        result = parser.parse(tmp_run_dir, stdout="metric_a: 0.5")
        # Should fallback to stdout
        assert result.source == "stdout"


# ---------------------------------------------------------------------------
# CSV parsing tests
# ---------------------------------------------------------------------------


class TestCSVParsing:
    def test_parse_condition_csv(self, parser, tmp_run_dir):
        csv_content = "condition,seed,metric,value\nmethod_a,42,accuracy,0.95\nmethod_b,42,accuracy,0.88\n"
        (tmp_run_dir / "results.csv").write_text(csv_content)

        result = parser.parse(tmp_run_dir)
        assert result.source == "csv"
        assert "method_a/accuracy" in result.scalars
        assert result.scalars["method_a/accuracy"] == 0.95

    def test_parse_convergence_csv(self, parser, tmp_run_dir):
        csv_content = "method,h,error\neuler,0.1,0.05\neuler,0.05,0.012\nrk4,0.1,0.001\n"
        (tmp_run_dir / "results.csv").write_text(csv_content)

        result = parser.parse(tmp_run_dir)
        assert result.source == "csv"
        assert "euler" in result.convergence
        assert len(result.convergence["euler"]) == 2

    def test_csv_skip_invalid(self, parser, tmp_run_dir):
        csv_content = "condition,metric,value\nmethod,accuracy,not_a_number\n"
        (tmp_run_dir / "results.csv").write_text(csv_content)

        result = parser.parse(tmp_run_dir)
        assert result.source == "csv"
        assert len(result.scalars) == 0


# ---------------------------------------------------------------------------
# stdout fallback tests
# ---------------------------------------------------------------------------


class TestStdoutParsing:
    def test_parse_plain_metrics(self, parser, tmp_run_dir):
        result = parser.parse(tmp_run_dir, stdout="accuracy: 0.95\nloss: 0.32\n")
        assert result.source == "stdout"
        assert result.scalars["accuracy"] == 0.95
        assert result.scalars["loss"] == 0.32

    def test_parse_condition_metrics(self, parser, tmp_run_dir):
        stdout = "condition=method_a accuracy: 0.95\ncondition=method_b accuracy: 0.88\n"
        result = parser.parse(tmp_run_dir, stdout=stdout)
        assert result.source == "stdout"
        assert "method_a/accuracy" in result.scalars

    def test_fallback_to_stdout_log(self, parser, tmp_run_dir):
        (tmp_run_dir / "stdout.log").write_text("metric_x: 1.5\n")
        result = parser.parse(tmp_run_dir)
        assert result.source == "stdout"
        assert result.scalars.get("metric_x") == 1.5


# ---------------------------------------------------------------------------
# ExperimentResults tests
# ---------------------------------------------------------------------------


class TestExperimentResults:
    def test_to_flat_metrics_empty(self):
        result = ExperimentResults()
        assert result.to_flat_metrics() == {}

    def test_to_flat_metrics_scalars(self):
        result = ExperimentResults(scalars={"a": 1.0, "b": 2.0})
        flat = result.to_flat_metrics()
        assert flat["a"] == 1.0
        assert flat["b"] == 2.0

    def test_to_flat_metrics_conditions(self):
        result = ExperimentResults(
            conditions={
                "method": {"seed_1": {"acc": 0.9}, "seed_2": {"acc": 0.91}},
            }
        )
        flat = result.to_flat_metrics()
        assert "method/acc" in flat

    def test_to_flat_metrics_convergence(self):
        result = ExperimentResults(
            convergence={
                "euler": [
                    {"h": 0.1, "error": 0.05},
                    {"h": 0.05, "error": 0.01},
                ],
            }
        )
        flat = result.to_flat_metrics()
        assert "euler/error" in flat
        assert flat["euler/error"] == 0.01  # last point

    def test_to_flat_metrics_regression(self):
        result = ExperimentResults(
            regression_table={
                "ols": {"coeff": 0.5, "se": 0.1},
            }
        )
        flat = result.to_flat_metrics()
        assert flat["ols/coeff"] == 0.5


# ---------------------------------------------------------------------------
# Priority tests (JSON > CSV > stdout)
# ---------------------------------------------------------------------------


class TestParsePriority:
    def test_json_takes_priority_over_csv(self, parser, tmp_run_dir):
        (tmp_run_dir / "results.json").write_text('{"from_json": 1.0}')
        (tmp_run_dir / "results.csv").write_text("condition,metric,value\ncsv,m,2.0\n")

        result = parser.parse(tmp_run_dir)
        assert result.source == "json"

    def test_csv_takes_priority_over_stdout(self, parser, tmp_run_dir):
        (tmp_run_dir / "results.csv").write_text("condition,metric,value\ncsv,m,2.0\n")

        result = parser.parse(tmp_run_dir, stdout="stdout_metric: 3.0")
        assert result.source == "csv"

    def test_empty_json_falls_to_csv(self, parser, tmp_run_dir):
        (tmp_run_dir / "results.json").write_text("{}")
        (tmp_run_dir / "results.csv").write_text("condition,metric,value\ncsv,m,2.0\n")

        result = parser.parse(tmp_run_dir)
        assert result.source == "csv"


# ---------------------------------------------------------------------------
# MetricType enum tests
# ---------------------------------------------------------------------------


class TestMetricType:
    def test_values(self):
        assert MetricType.SCALAR.value == "scalar"
        assert MetricType.TABLE.value == "table"
        assert MetricType.CONVERGENCE.value == "convergence"
        assert MetricType.STRUCTURED.value == "structured"
