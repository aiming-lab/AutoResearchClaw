"""Metric aggregation and computation for the dashboard."""

from __future__ import annotations

from typing import Any


def aggregate_metrics(runs: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate metrics across multiple runs for overview display."""
    total = len(runs)
    active = sum(1 for r in runs if r.get("is_active"))
    completed = sum(1 for r in runs if r.get("status") == "completed")
    failed = sum(1 for r in runs if r.get("status") == "failed")

    avg_stages = 0.0
    if total > 0:
        avg_stages = sum(r.get("current_stage", 0) for r in runs) / total

    return {
        "total_runs": total,
        "active_runs": active,
        "completed_runs": completed,
        "failed_runs": failed,
        "average_stage": round(avg_stages, 1),
    }


def extract_training_curve(metrics: dict[str, Any]) -> list[dict[str, float]]:
    """Extract training curve data points from experiment metrics."""
    curve: list[dict[str, float]] = []

    training_log = metrics.get("training_log", [])
    if isinstance(training_log, list):
        for entry in training_log:
            if isinstance(entry, dict):
                point = {}
                for key in ("epoch", "step", "loss", "accuracy", "lr"):
                    if key in entry:
                        try:
                            point[key] = float(entry[key])
                        except (ValueError, TypeError):
                            pass
                if point:
                    curve.append(point)

    return curve
