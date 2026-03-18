"""Universal metric parser — supports JSON, CSV, and stdout regex formats.

Parse priority:
  1. ``results.json`` — structured JSON output (recommended for all domains)
  2. ``results.csv`` — tabular output
  3. stdout regex — backward-compatible with existing ``metric: value`` format

This module extends (not replaces) the existing ``sandbox.parse_metrics``
function. The existing stdout parser remains the fallback.
"""

from __future__ import annotations

import csv
import json
import logging
import math
from dataclasses import dataclass, field
from enum import Enum
from io import StringIO
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class MetricType(str, Enum):
    SCALAR = "scalar"
    TABLE = "table"
    CONVERGENCE = "convergence"
    LEARNING_CURVE = "learning_curve"
    CONFUSION_MATRIX = "confusion"
    STRUCTURED = "structured"
    PARETO = "pareto"


@dataclass
class ExperimentResults:
    """Unified experiment results container.

    Works for all domains — ML scalar metrics, physics convergence data,
    economics regression tables, etc.
    """

    # Flat scalar metrics (backward-compatible with existing pipeline)
    scalars: dict[str, float] = field(default_factory=dict)

    # Per-condition results (new universal format)
    conditions: dict[str, dict[str, Any]] = field(default_factory=dict)

    # Convergence data (for physics/math domains)
    convergence: dict[str, list[dict[str, float]]] = field(default_factory=dict)

    # Regression tables (for economics)
    regression_table: dict[str, dict[str, Any]] = field(default_factory=dict)

    # Full structured data (raw JSON)
    structured: dict[str, Any] = field(default_factory=dict)

    # Metadata
    experiment_type: str = ""
    domain: str = ""
    total_runtime_sec: float = 0.0
    source: str = ""  # "json" | "csv" | "stdout"

    def to_flat_metrics(self) -> dict[str, float]:
        """Convert to flat metric dict for backward compatibility.

        The existing pipeline expects dict[str, float] from parse_metrics().
        This method flattens all result types into that format.
        """
        metrics: dict[str, float] = dict(self.scalars)

        # Flatten conditions
        for cond_name, seeds in self.conditions.items():
            if isinstance(seeds, dict):
                for seed_or_metric, value in seeds.items():
                    if isinstance(value, dict):
                        for metric_name, metric_val in value.items():
                            if isinstance(metric_val, (int, float)) and math.isfinite(metric_val):
                                metrics[f"{cond_name}/{metric_name}"] = float(metric_val)
                    elif isinstance(value, (int, float)) and math.isfinite(value):
                        metrics[f"{cond_name}/{seed_or_metric}"] = float(value)

        # Flatten convergence (take final/best error per method)
        for method, points in self.convergence.items():
            if points:
                last = points[-1]
                for key, val in last.items():
                    if key != "h" and isinstance(val, (int, float)) and math.isfinite(val):
                        metrics[f"{method}/{key}"] = float(val)

        # Flatten regression table
        for spec, coeffs in self.regression_table.items():
            if isinstance(coeffs, dict):
                for key, val in coeffs.items():
                    if isinstance(val, (int, float)) and math.isfinite(val):
                        metrics[f"{spec}/{key}"] = float(val)

        return metrics


class UniversalMetricParser:
    """Parse experiment results from multiple output formats.

    Usage::

        parser = UniversalMetricParser()
        results = parser.parse(run_dir)
        flat = results.to_flat_metrics()  # backward-compatible
    """

    def parse(self, run_dir: Path, stdout: str = "") -> ExperimentResults:
        """Parse experiment results from a run directory.

        Tries formats in order: JSON → CSV → stdout regex.
        """
        # 1. Try JSON
        results_json = run_dir / "results.json"
        if results_json.exists():
            try:
                result = self._parse_json(results_json)
                if result.scalars or result.conditions or result.convergence or result.regression_table:
                    logger.info("Parsed results from results.json")
                    return result
            except Exception:
                logger.warning("Failed to parse results.json", exc_info=True)

        # 2. Try CSV
        results_csv = run_dir / "results.csv"
        if results_csv.exists():
            try:
                result = self._parse_csv(results_csv)
                if result.source == "csv":
                    logger.info("Parsed results from results.csv")
                    return result
            except Exception:
                logger.warning("Failed to parse results.csv", exc_info=True)

        # 3. Fallback: stdout regex (existing behavior)
        if stdout:
            return self._parse_stdout(stdout)

        # Try reading stdout.log from run_dir
        stdout_log = run_dir / "stdout.log"
        if stdout_log.exists():
            try:
                stdout_text = stdout_log.read_text(encoding="utf-8", errors="replace")
                return self._parse_stdout(stdout_text)
            except Exception:
                logger.warning("Failed to read stdout.log", exc_info=True)

        return ExperimentResults(source="none")

    def _parse_json(self, path: Path) -> ExperimentResults:
        """Parse structured JSON results."""
        with path.open(encoding="utf-8") as fh:
            data = json.load(fh)

        if not isinstance(data, dict):
            return ExperimentResults(source="json")

        result = ExperimentResults(
            source="json",
            experiment_type=data.get("experiment_type", ""),
            structured=data,
        )

        # Extract metadata
        meta = data.get("metadata", {})
        if isinstance(meta, dict):
            result.domain = meta.get("domain", "")
            result.total_runtime_sec = float(meta.get("total_runtime_sec", 0))

        # Extract conditions (comparison experiments)
        conditions = data.get("conditions", {})
        if isinstance(conditions, dict):
            result.conditions = conditions
            # Also extract scalar metrics for backward compatibility
            for cond_name, seeds in conditions.items():
                if isinstance(seeds, dict):
                    for seed_key, metrics in seeds.items():
                        if isinstance(metrics, dict):
                            for metric_name, val in metrics.items():
                                if isinstance(val, (int, float)) and math.isfinite(val):
                                    result.scalars[f"{cond_name}/{metric_name}"] = float(val)
                                    result.scalars[metric_name] = float(val)
                        elif isinstance(metrics, (int, float)) and math.isfinite(metrics):
                            result.scalars[f"{cond_name}/{seed_key}"] = float(metrics)

        # Extract convergence data
        convergence = data.get("convergence", {})
        if isinstance(convergence, dict):
            result.convergence = convergence

        # Extract regression table
        reg_table = data.get("regression_table", {})
        if isinstance(reg_table, dict):
            result.regression_table = reg_table

        # Top-level scalar metrics
        for key, val in data.items():
            if key not in ("conditions", "convergence", "regression_table", "metadata", "experiment_type"):
                if isinstance(val, (int, float)) and math.isfinite(val):
                    result.scalars[key] = float(val)

        return result

    def _parse_csv(self, path: Path) -> ExperimentResults:
        """Parse CSV results (one row per condition/seed/metric)."""
        text = path.read_text(encoding="utf-8", errors="replace")
        reader = csv.DictReader(StringIO(text))

        result = ExperimentResults(source="csv")
        rows_processed = 0

        for row in reader:
            rows_processed += 1
            # Expected columns: condition, seed, metric, value
            # Or: method, h, error (for convergence)
            cond = row.get("condition", row.get("method", ""))
            metric = row.get("metric", "")
            value_str = row.get("value", row.get("error", ""))

            try:
                val = float(value_str)
            except (ValueError, TypeError):
                continue

            if not math.isfinite(val):
                continue

            if metric:
                key = f"{cond}/{metric}" if cond else metric
                result.scalars[key] = val
            elif cond:
                # Convergence-style: method, h, error
                h_str = row.get("h", "")
                try:
                    h = float(h_str)
                except (ValueError, TypeError):
                    continue
                if cond not in result.convergence:
                    result.convergence[cond] = []
                result.convergence[cond].append({"h": h, "error": val})

        # Mark as CSV source if we processed any rows (even if no valid data)
        if rows_processed == 0:
            result.source = "none"

        return result

    def _parse_stdout(self, stdout: str) -> ExperimentResults:
        """Parse stdout using the existing regex-based parser.

        Delegates to ``sandbox.parse_metrics`` for backward compatibility.
        """
        from researchclaw.experiment.sandbox import parse_metrics

        metrics = parse_metrics(stdout)
        return ExperimentResults(
            scalars={k: float(v) for k, v in metrics.items() if isinstance(v, (int, float))},
            source="stdout",
        )
