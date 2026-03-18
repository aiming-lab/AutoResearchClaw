"""Tests for the convergence study evaluator."""

from __future__ import annotations

import math
import pytest

from researchclaw.experiment.evaluators.convergence import (
    ConvergenceReport,
    ConvergenceResult,
    analyze_convergence,
    compute_convergence_order,
)


# ---------------------------------------------------------------------------
# compute_convergence_order tests
# ---------------------------------------------------------------------------


class TestComputeConvergenceOrder:
    def test_second_order(self):
        """h, h/2, h/4, h/8 with error ~ h^2."""
        hs = [0.1, 0.05, 0.025, 0.0125]
        errors = [h**2 for h in hs]
        order, r2 = compute_convergence_order(hs, errors)
        assert abs(order - 2.0) < 0.1
        assert r2 > 0.99

    def test_fourth_order(self):
        """Error ~ h^4."""
        hs = [0.1, 0.05, 0.025, 0.0125]
        errors = [h**4 for h in hs]
        order, r2 = compute_convergence_order(hs, errors)
        assert abs(order - 4.0) < 0.1
        assert r2 > 0.99

    def test_first_order(self):
        """Error ~ h."""
        hs = [0.1, 0.05, 0.025, 0.0125]
        errors = [h for h in hs]
        order, r2 = compute_convergence_order(hs, errors)
        assert abs(order - 1.0) < 0.1

    def test_too_few_points(self):
        order, r2 = compute_convergence_order([0.1], [0.01])
        assert order == 0.0
        assert r2 == 0.0

    def test_empty_input(self):
        order, r2 = compute_convergence_order([], [])
        assert order == 0.0

    def test_filters_invalid(self):
        hs = [0.1, 0.0, 0.025, -0.01]  # 0 and negative should be filtered
        errors = [0.01, 0.0, 0.001, 0.0001]
        order, r2 = compute_convergence_order(hs, errors)
        # Should still work with valid points
        assert order > 0


# ---------------------------------------------------------------------------
# analyze_convergence tests
# ---------------------------------------------------------------------------


class TestAnalyzeConvergence:
    def test_single_method(self):
        data = {
            "euler": [
                {"h": 0.1, "error": 0.1},
                {"h": 0.05, "error": 0.05},
                {"h": 0.025, "error": 0.025},
            ]
        }
        report = analyze_convergence(data)
        assert len(report.methods) == 1
        assert report.methods[0].method == "euler"
        assert abs(report.methods[0].convergence_order - 1.0) < 0.2
        assert report.best_method == "euler"

    def test_multiple_methods(self):
        data = {
            "euler": [
                {"h": 0.1, "error": 0.1},
                {"h": 0.05, "error": 0.05},
                {"h": 0.025, "error": 0.025},
            ],
            "rk4": [
                {"h": 0.1, "error": 1e-4},
                {"h": 0.05, "error": 6.25e-6},
                {"h": 0.025, "error": 3.9e-7},
            ],
        }
        report = analyze_convergence(data)
        assert len(report.methods) == 2
        # RK4 should have higher order
        orders = {r.method: r.convergence_order for r in report.methods}
        assert orders["rk4"] > orders["euler"]
        assert report.best_method == "rk4"

    def test_expected_orders(self):
        data = {
            "euler": [
                {"h": 0.1, "error": 0.1},
                {"h": 0.05, "error": 0.05},
                {"h": 0.025, "error": 0.025},
            ],
        }
        report = analyze_convergence(data, expected_orders={"euler": 1.0})
        assert report.methods[0].expected_order == 1.0
        assert report.methods[0].order_matches_expected is True

    def test_non_converging(self):
        data = {
            "bad_method": [
                {"h": 0.1, "error": 0.5},
                {"h": 0.05, "error": 0.6},  # error increases
                {"h": 0.025, "error": 0.7},
            ],
        }
        report = analyze_convergence(data)
        # Negative or very low order indicates no convergence
        assert not report.methods[0].is_converging

    def test_summary_string(self):
        data = {
            "method_a": [
                {"h": 0.1, "error": 0.01},
                {"h": 0.05, "error": 0.0025},
            ],
        }
        report = analyze_convergence(data)
        assert report.summary  # should not be empty
        assert "method_a" in report.summary

    def test_l2_error_key(self):
        """Should handle l2_error as the error key."""
        data = {
            "fem": [
                {"h": 0.1, "l2_error": 0.01},
                {"h": 0.05, "l2_error": 0.0025},
                {"h": 0.025, "l2_error": 0.000625},
            ],
        }
        report = analyze_convergence(data)
        assert abs(report.methods[0].convergence_order - 2.0) < 0.2

    def test_empty_data(self):
        report = analyze_convergence({})
        assert len(report.methods) == 0
        assert report.best_method == ""
