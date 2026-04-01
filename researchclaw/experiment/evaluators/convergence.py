"""Convergence study evaluator for physics/math domains.

Analyzes convergence data (error vs grid size/timestep) to determine
convergence order and quality of numerical methods.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class ConvergenceResult:
    """Result of convergence analysis for one method."""
    method: str
    convergence_order: float = 0.0
    r_squared: float = 0.0
    points: list[dict[str, float]] = field(default_factory=list)
    is_converging: bool = False
    expected_order: float | None = None  # if known
    order_matches_expected: bool = False


@dataclass
class ConvergenceReport:
    """Full convergence analysis report."""
    methods: list[ConvergenceResult] = field(default_factory=list)
    best_method: str = ""
    summary: str = ""


def compute_convergence_order(
    h_values: list[float],
    errors: list[float],
) -> tuple[float, float]:
    """Compute convergence order via log-log linear regression.

    Parameters
    ----------
    h_values : list[float]
        Grid sizes / timesteps (must be decreasing).
    errors : list[float]
        Error norms corresponding to each h value.

    Returns
    -------
    order : float
        Estimated convergence order (slope in log-log space).
    r_squared : float
        R² of the log-log fit.
    """
    if len(h_values) < 2 or len(errors) < 2:
        return 0.0, 0.0

    # Filter out non-positive values
    valid = [
        (h, e) for h, e in zip(h_values, errors)
        if h > 0 and e > 0 and math.isfinite(h) and math.isfinite(e)
    ]
    if len(valid) < 2:
        return 0.0, 0.0

    hs, es = zip(*valid)
    log_h = np.log(np.array(hs, dtype=np.float64))
    log_e = np.log(np.array(es, dtype=np.float64))

    # Linear regression: log(e) = p * log(h) + C
    n = len(log_h)
    sum_x = np.sum(log_h)
    sum_y = np.sum(log_e)
    sum_xy = np.sum(log_h * log_e)
    sum_x2 = np.sum(log_h ** 2)

    denom = n * sum_x2 - sum_x ** 2
    if abs(denom) < 1e-15:
        return 0.0, 0.0

    slope = (n * sum_xy - sum_x * sum_y) / denom
    intercept = (sum_y - slope * sum_x) / n

    # R²
    y_pred = slope * log_h + intercept
    ss_res = np.sum((log_e - y_pred) ** 2)
    ss_tot = np.sum((log_e - np.mean(log_e)) ** 2)
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 1e-15 else 0.0

    return float(slope), float(r_squared)


def analyze_convergence(
    convergence_data: dict[str, list[dict[str, float]]],
    expected_orders: dict[str, float] | None = None,
) -> ConvergenceReport:
    """Analyze convergence data for multiple methods.

    Parameters
    ----------
    convergence_data : dict
        Maps method name to list of {"h": ..., "error": ...} dicts.
    expected_orders : dict, optional
        Known convergence orders per method (for validation).

    Returns
    -------
    ConvergenceReport
    """
    results: list[ConvergenceResult] = []

    for method, points in convergence_data.items():
        if not points:
            continue

        # Sort by h (descending — coarsest first)
        sorted_pts = sorted(points, key=lambda p: p.get("h", 0), reverse=True)

        h_vals = [p["h"] for p in sorted_pts if "h" in p]
        # Try "error", "l2_error", "linf_error"
        error_key = "error"
        for key in ("error", "l2_error", "linf_error"):
            if key in sorted_pts[0]:
                error_key = key
                break

        errors = [p.get(error_key, 0) for p in sorted_pts]

        order, r2 = compute_convergence_order(h_vals, errors)

        expected = None
        matches = False
        if expected_orders and method in expected_orders:
            expected = expected_orders[method]
            matches = abs(order - expected) < 0.5  # within half an order

        is_converging = order > 0.5 and r2 > 0.8

        results.append(ConvergenceResult(
            method=method,
            convergence_order=order,
            r_squared=r2,
            points=sorted_pts,
            is_converging=is_converging,
            expected_order=expected,
            order_matches_expected=matches,
        ))

    # Find best method (highest convergence order)
    best = ""
    if results:
        best_result = max(results, key=lambda r: r.convergence_order)
        best = best_result.method

    # Generate summary
    summary_lines = []
    for r in results:
        line = f"{r.method}: order={r.convergence_order:.2f} (R²={r.r_squared:.3f})"
        if r.expected_order is not None:
            status = "✓" if r.order_matches_expected else "✗"
            line += f" [expected={r.expected_order:.1f} {status}]"
        summary_lines.append(line)

    return ConvergenceReport(
        methods=results,
        best_method=best,
        summary="\n".join(summary_lines),
    )
