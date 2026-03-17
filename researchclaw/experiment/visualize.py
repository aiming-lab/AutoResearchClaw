"""Experiment result visualization.

Generates publication-quality charts from experiment run data:
- Condition comparison (grouped bar chart with CI error bars)
- Metric heatmap (condition × metric matrix)
- Metric trajectory (line chart with markers)
- Ablation delta chart (horizontal bar showing delta from baseline)
- Pipeline execution timeline
- Iteration score history

Uses Paul Tol colorblind-safe palette and academic styling.
"""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

try:
    import matplotlib

    matplotlib.use("Agg")  # Non-interactive backend
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    from matplotlib.patches import FancyBboxPatch
    import numpy as np

    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

# Paul Tol "bright" palette — colorblind-safe, publication-ready
_PAUL_TOL_BRIGHT = [
    "#4477AA",  # blue
    "#EE6677",  # red/pink
    "#228833",  # green
    "#CCBB44",  # yellow
    "#66CCEE",  # cyan
    "#AA3377",  # purple
    "#BBBBBB",  # grey
]

# Extended palette for many conditions (Tol muted + bright merged)
_PAUL_TOL_EXTENDED = [
    "#4477AA", "#EE6677", "#228833", "#CCBB44", "#66CCEE",
    "#AA3377", "#332288", "#88CCEE", "#44AA99", "#117733",
    "#999933", "#CC6677", "#882255", "#661100", "#6699CC",
]

# Metrics to exclude from comparison charts (timing, meta, non-scientific)
_EXCLUDED_METRICS: set[str] = {
    "time_budget_sec", "elapsed_sec", "elapsed_time", "execution_time",
    "wall_time", "runtime_sec", "total_time", "timeout",
    "seed", "seed_count", "n_seeds", "num_seeds",
    "success_rate", "num_conditions", "total_conditions",
    "calibration_iterations",
}

# Prefixes that indicate meta/timing metrics
_EXCLUDED_PREFIXES: tuple[str, ...] = ("time_", "runtime_", "elapsed_", "wall_")


def _is_excluded_metric(name: str) -> bool:
    """Return True if *name* is a timing/meta metric that shouldn't be charted."""
    low = name.lower()
    if low in _EXCLUDED_METRICS:
        return True
    return any(low.startswith(p) for p in _EXCLUDED_PREFIXES)


def _shorten_label(name: str, max_len: int = 22) -> str:
    """Shorten a metric label for chart readability."""
    if len(name) <= max_len:
        return name
    return name[: max_len - 1] + "\u2026"


def _format_cond_name(name: str) -> str:
    """Format condition name for display: underscores → spaces, title case."""
    return name.replace("_", " ").title()


def _ensure_dir(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _setup_academic_style() -> None:
    """Apply academic styling via rcParams."""
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 13,
        "axes.titleweight": "bold",
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "grid.linestyle": "--",
        "axes.axisbelow": True,
    })


# ---------------------------------------------------------------------------
# 1. Condition comparison — grouped bar with CI error bars
# ---------------------------------------------------------------------------


def plot_condition_comparison(
    condition_summaries: dict[str, dict[str, Any]],
    output_path: Path,
    *,
    metric_key: str = "primary_metric",
    title: str = "",
) -> Path | None:
    """Bar chart comparing conditions with mean +/- 95% CI error bars.

    Uses Paul Tol colorblind-safe palette with gradient shading.
    """
    if not HAS_MATPLOTLIB or not condition_summaries:
        return None

    _setup_academic_style()

    names: list[str] = []
    means: list[float] = []
    ci_low: list[float] = []
    ci_high: list[float] = []

    for cond, info in condition_summaries.items():
        m = info.get("metrics", {})
        mean_val = m.get(f"{metric_key}_mean") or m.get(metric_key)
        if mean_val is None:
            continue
        fmean = float(mean_val)
        names.append(_format_cond_name(cond))
        means.append(fmean)
        ci_low.append(float(info.get("ci95_low", fmean)))
        ci_high.append(float(info.get("ci95_high", fmean)))

    if not names:
        return None

    yerr_lo = [max(0, m - lo) for m, lo in zip(means, ci_low)]
    yerr_hi = [max(0, hi - m) for m, hi in zip(means, ci_high)]

    n = len(names)
    colors = [_PAUL_TOL_EXTENDED[i % len(_PAUL_TOL_EXTENDED)] for i in range(n)]

    fig, ax = plt.subplots(figsize=(max(7, n * 1.2), 5))
    x = np.arange(n)
    bars = ax.bar(
        x, means, color=colors, alpha=0.88,
        edgecolor="white", linewidth=0.8, width=0.7,
    )
    ax.errorbar(
        x, means, yerr=[yerr_lo, yerr_hi],
        fmt="none", ecolor="#333333", capsize=5, capthick=1.5, linewidth=1.5,
    )

    # Value labels above bars
    y_max = max(m + h for m, h in zip(means, yerr_hi)) if yerr_hi else max(means)
    offset = y_max * 0.025
    for i, m in enumerate(means):
        ax.text(
            i, m + yerr_hi[i] + offset, f"{m:.3f}",
            ha="center", va="bottom", fontsize=9, fontweight="bold", color="#333",
        )

    ax.set_xlabel("Method / Condition")
    metric_label = metric_key.replace("_", " ").title()
    ax.set_ylabel(metric_label)
    ax.set_title(title or f"{metric_label} Comparison (Mean \u00b1 95% CI)")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=30, ha="right")
    ax.set_ylim(bottom=0)
    fig.tight_layout()
    fig.savefig(_ensure_dir(output_path), dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved condition comparison: %s", output_path)
    return output_path


# ---------------------------------------------------------------------------
# 2. Metric heatmap — condition × metric matrix
# ---------------------------------------------------------------------------


def plot_metric_heatmap(
    condition_summaries: dict[str, dict[str, Any]],
    output_path: Path,
    *,
    title: str = "Performance Heatmap (Per-Condition Metrics)",
    max_metrics: int = 12,
) -> Path | None:
    """Heatmap of normalized metric values across conditions.

    Shows per-condition performance normalized to [0, 1] per metric column.
    """
    if not HAS_MATPLOTLIB or not condition_summaries:
        return None

    _setup_academic_style()

    # Collect all metric keys across conditions
    all_metric_keys: set[str] = set()
    for info in condition_summaries.values():
        m = info.get("metrics", {})
        for k in m:
            if not _is_excluded_metric(k) and not k.endswith("_std"):
                all_metric_keys.add(k)

    # Filter to _mean variants or raw metrics, deduplicate
    clean_keys: list[str] = []
    for k in sorted(all_metric_keys):
        base = k.replace("_mean", "")
        if base not in [ck.replace("_mean", "") for ck in clean_keys]:
            clean_keys.append(k)

    if len(clean_keys) < 2:
        return None

    clean_keys = clean_keys[:max_metrics]
    cond_names = list(condition_summaries.keys())

    # Build matrix
    data = np.zeros((len(cond_names), len(clean_keys)))
    for i, cond in enumerate(cond_names):
        m = condition_summaries[cond].get("metrics", {})
        for j, mk in enumerate(clean_keys):
            val = m.get(mk, 0)
            try:
                data[i, j] = float(val)
            except (ValueError, TypeError):
                data[i, j] = 0

    # Normalize per column (min-max)
    for j in range(data.shape[1]):
        col = data[:, j]
        lo, hi = col.min(), col.max()
        if hi > lo:
            data[:, j] = (col - lo) / (hi - lo)
        else:
            data[:, j] = 0.5

    fig, ax = plt.subplots(
        figsize=(max(6, len(clean_keys) * 1.3), max(4, len(cond_names) * 0.6))
    )
    im = ax.imshow(data, cmap="YlGnBu", aspect="auto", vmin=0, vmax=1)

    # Annotate cells
    for i in range(len(cond_names)):
        for j in range(len(clean_keys)):
            val = data[i, j]
            color = "white" if val > 0.6 else "#333"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    fontsize=8, color=color, fontweight="bold")

    ax.set_xticks(range(len(clean_keys)))
    ax.set_xticklabels(
        [_shorten_label(k.replace("_mean", "").replace("_", " ")) for k in clean_keys],
        rotation=35, ha="right", fontsize=9,
    )
    ax.set_yticks(range(len(cond_names)))
    ax.set_yticklabels([_format_cond_name(c) for c in cond_names], fontsize=9)
    ax.set_title(title)

    cbar = fig.colorbar(im, ax=ax, shrink=0.8, label="Normalized Score")
    fig.tight_layout()
    fig.savefig(_ensure_dir(output_path), dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved metric heatmap: %s", output_path)
    return output_path


# ---------------------------------------------------------------------------
# 3. Ablation delta chart — horizontal bars showing improvement over baseline
# ---------------------------------------------------------------------------


def plot_ablation_deltas(
    condition_summaries: dict[str, dict[str, Any]],
    output_path: Path,
    *,
    metric_key: str = "primary_metric",
    baseline_name: str = "",
    title: str = "",
    higher_is_better: bool = True,
) -> Path | None:
    """Horizontal bar chart showing delta from baseline for each ablation.

    Bars go left (worse) or right (better) from zero.
    """
    if not HAS_MATPLOTLIB or not condition_summaries:
        return None

    _setup_academic_style()

    # Find baseline
    cond_keys = list(condition_summaries.keys())
    if baseline_name:
        base_key = baseline_name
    else:
        # Heuristic: pick "baseline", "heuristic_baseline", or first condition
        for candidate in ["baseline", "heuristic_baseline", "random_baseline"]:
            if candidate in cond_keys:
                base_key = candidate
                break
        else:
            base_key = cond_keys[0]

    base_info = condition_summaries.get(base_key, {})
    base_m = base_info.get("metrics", {})
    base_val = float(base_m.get(f"{metric_key}_mean") or base_m.get(metric_key, 0))

    if base_val == 0:
        return None

    names: list[str] = []
    deltas: list[float] = []
    for cond, info in condition_summaries.items():
        if cond == base_key:
            continue
        m = info.get("metrics", {})
        val = m.get(f"{metric_key}_mean") or m.get(metric_key)
        if val is None:
            continue
        fval = float(val)
        pct = ((fval - base_val) / abs(base_val)) * 100
        names.append(_format_cond_name(cond))
        deltas.append(pct)

    if not names:
        return None

    # Sort by delta
    pairs = sorted(zip(deltas, names), reverse=True)
    deltas, names = zip(*pairs)
    deltas = list(deltas)
    names = list(names)

    fig, ax = plt.subplots(figsize=(8, max(4, len(names) * 0.5)))
    y = np.arange(len(names))
    bar_colors = []
    for d in deltas:
        if higher_is_better:
            bar_colors.append("#228833" if d > 0 else "#EE6677")
        else:
            bar_colors.append("#228833" if d < 0 else "#EE6677")

    ax.barh(y, deltas, color=bar_colors, alpha=0.85, edgecolor="white", height=0.6)
    ax.axvline(x=0, color="#333", linewidth=1, linestyle="-")

    # Value labels
    for i, d in enumerate(deltas):
        ha = "left" if d >= 0 else "right"
        offset = 0.5 if d >= 0 else -0.5
        ax.text(d + offset, i, f"{d:+.1f}%", ha=ha, va="center", fontsize=9, fontweight="bold")

    ax.set_yticks(y)
    ax.set_yticklabels(names)
    ax.set_xlabel(f"\u0394 {metric_key.replace('_', ' ').title()} vs. Baseline (%)")
    ax.set_title(title or f"Ablation Analysis (Baseline: {_format_cond_name(base_key)})")
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(_ensure_dir(output_path), dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved ablation deltas: %s", output_path)
    return output_path


# ---------------------------------------------------------------------------
# 4. Metric trajectory — line chart across refinement iterations
# ---------------------------------------------------------------------------


def plot_metric_trajectory(
    runs: list[dict[str, Any]],
    metric_key: str,
    output_path: Path,
    *,
    title: str = "",
) -> Path | None:
    """Plot metric values across runs as a styled line chart with markers."""
    if not HAS_MATPLOTLIB or not runs:
        return None

    _setup_academic_style()

    values: list[float] = []
    labels: list[str] = []
    for i, r in enumerate(runs):
        m = r.get("metrics") or r.get("key_metrics") or {}
        if isinstance(m, dict) and metric_key in m:
            try:
                values.append(float(m[metric_key]))
                labels.append(r.get("run_id", f"Iter {i + 1}"))
            except (ValueError, TypeError):
                continue

    if not values:
        return None

    fig, ax = plt.subplots(figsize=(max(6, len(values) * 1.5), 4.5))
    x = range(len(values))
    ax.plot(
        x, values, "o-", color=_PAUL_TOL_BRIGHT[0],
        linewidth=2.5, markersize=8, markerfacecolor="white",
        markeredgewidth=2, markeredgecolor=_PAUL_TOL_BRIGHT[0],
    )

    # Fill area under curve
    ax.fill_between(x, values, alpha=0.08, color=_PAUL_TOL_BRIGHT[0])

    # Value annotations
    for i, v in enumerate(values):
        ax.annotate(
            f"{v:.4f}", (i, v), textcoords="offset points",
            xytext=(0, 12), ha="center", fontsize=9, fontweight="bold",
        )

    metric_label = metric_key.replace("_", " ").title()
    ax.set_xlabel("Refinement Iteration")
    ax.set_ylabel(metric_label)
    ax.set_title(title or f"{metric_label} Across Iterations")
    ax.set_xticks(list(x))
    ax.set_xticklabels(
        [_shorten_label(lb, 15) for lb in labels],
        rotation=30, ha="right",
    )
    fig.tight_layout()
    fig.savefig(_ensure_dir(output_path), dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved metric trajectory: %s", output_path)
    return output_path


# ---------------------------------------------------------------------------
# 5. Experiment comparison — multi-metric bar chart
# ---------------------------------------------------------------------------


def plot_experiment_comparison(
    metrics_summary: dict[str, dict[str, float]],
    output_path: Path,
    *,
    title: str = "Experiment Results Comparison",
) -> Path | None:
    """Grouped bar chart comparing mean/min/max across metrics."""
    if not HAS_MATPLOTLIB or not metrics_summary:
        return None

    _setup_academic_style()

    filtered = {k: v for k, v in metrics_summary.items() if not _is_excluded_metric(k)}
    if not filtered:
        return None

    # Limit to top 12 metrics
    if len(filtered) > 12:
        top = sorted(filtered.items(), key=lambda kv: abs(kv[1].get("mean", 0)), reverse=True)[:12]
        filtered = dict(top)

    names = list(filtered.keys())
    means = [filtered[n].get("mean", 0) for n in names]
    mins = [filtered[n].get("min", 0) for n in names]
    maxs = [filtered[n].get("max", 0) for n in names]

    fig, ax = plt.subplots(figsize=(max(7, len(names) * 1.3), 5))
    x = np.arange(len(names))
    width = 0.6

    bars = ax.bar(
        x, means, width=width, color=_PAUL_TOL_BRIGHT[0], alpha=0.88,
        edgecolor="white", linewidth=0.8, label="Mean",
    )

    # Min-max range as thin lines
    for i, (lo, hi) in enumerate(zip(mins, maxs)):
        ax.plot([i, i], [lo, hi], color="#333", linewidth=2, solid_capstyle="round")
        ax.plot(i, lo, "_", color="#333", markersize=8, markeredgewidth=2)
        ax.plot(i, hi, "_", color="#333", markersize=8, markeredgewidth=2)

    ax.set_xlabel("Metric")
    ax.set_ylabel("Value")
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(
        [_shorten_label(n.replace("_", " ")) for n in names],
        rotation=35, ha="right",
    )
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(_ensure_dir(output_path), dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved experiment comparison: %s", output_path)
    return output_path


# ---------------------------------------------------------------------------
# 6. Pipeline execution timeline
# ---------------------------------------------------------------------------


def plot_pipeline_timeline(
    stage_results: list[dict[str, Any]],
    output_path: Path,
    *,
    title: str = "Pipeline Execution Timeline",
) -> Path | None:
    """Horizontal bar chart showing execution time per stage."""
    if not HAS_MATPLOTLIB or not stage_results:
        return None

    _setup_academic_style()

    labels: list[str] = []
    durations: list[float] = []
    colors: list[str] = []

    for r in stage_results:
        name = r.get("stage_name", r.get("stage", "?"))
        elapsed = r.get("elapsed_sec", 0)
        status = r.get("status", "done")
        labels.append(str(name))
        durations.append(float(elapsed) if elapsed else 1.0)
        colors.append("#228833" if status == "done" else "#EE6677")

    if not labels:
        return None

    fig, ax = plt.subplots(figsize=(10, max(4, len(labels) * 0.35)))
    y = range(len(labels))
    ax.barh(list(y), durations, color=colors, alpha=0.85, edgecolor="white", height=0.6)
    ax.set_yticks(list(y))
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("Time (seconds)")
    ax.set_title(title)
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(_ensure_dir(output_path), dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved pipeline timeline: %s", output_path)
    return output_path


# ---------------------------------------------------------------------------
# 7. Iteration score history
# ---------------------------------------------------------------------------


def plot_iteration_scores(
    scores: list[float | None],
    output_path: Path,
    *,
    threshold: float = 7.0,
    title: str = "Quality Score by Iteration",
) -> Path | None:
    """Line chart of quality scores across iterations."""
    if not HAS_MATPLOTLIB or not scores:
        return None

    _setup_academic_style()

    valid = [(i + 1, s) for i, s in enumerate(scores) if s is not None]
    if not valid:
        return None

    iters, vals = zip(*valid)

    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.plot(
        iters, vals, "o-", color=_PAUL_TOL_BRIGHT[5],
        linewidth=2.5, markersize=9, markerfacecolor="white",
        markeredgewidth=2, markeredgecolor=_PAUL_TOL_BRIGHT[5],
    )
    ax.axhline(
        y=threshold, color=_PAUL_TOL_BRIGHT[1], linestyle="--",
        alpha=0.7, linewidth=1.5, label=f"Threshold ({threshold})",
    )
    ax.fill_between(iters, vals, alpha=0.06, color=_PAUL_TOL_BRIGHT[5])
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Quality Score")
    ax.set_title(title)
    ax.set_ylim(0, 10.5)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(_ensure_dir(output_path), dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved iteration scores: %s", output_path)
    return output_path


# ---------------------------------------------------------------------------
# 8. All-in-one: generate all charts from run directory
# ---------------------------------------------------------------------------


def generate_all_charts(
    run_dir: Path,
    output_dir: Path | None = None,
    *,
    metric_key: str = "val_loss",
    metric_direction: str = "minimize",
) -> list[Path]:
    """Scan run_dir and generate all applicable charts.

    Returns list of generated image paths.
    """
    if not HAS_MATPLOTLIB:
        logger.warning("matplotlib not available — skipping chart generation")
        return []

    if output_dir is None:
        output_dir = run_dir / "charts"
    output_dir.mkdir(parents=True, exist_ok=True)

    generated: list[Path] = []

    # Collect experiment runs
    runs: list[dict[str, Any]] = []
    for stage_subdir in sorted(run_dir.glob("stage-*/runs")):
        for run_file in sorted(stage_subdir.glob("*.json")):
            try:
                data = json.loads(run_file.read_text(encoding="utf-8"))
                if isinstance(data, dict):
                    runs.append(data)
            except (json.JSONDecodeError, OSError):
                continue

    # 1. Metric trajectory
    path = plot_metric_trajectory(
        runs, metric_key, output_dir / "metric_trajectory.png"
    )
    if path:
        generated.append(path)

    # 2. Load experiment summary for condition-based charts
    summary_path = run_dir / "stage-14" / "experiment_summary.json"
    if summary_path.exists():
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))

            cs = summary.get("condition_summaries", {})
            if cs:
                # 2a. Condition comparison (bar chart with CI)
                path = plot_condition_comparison(
                    cs, output_dir / "method_comparison.png",
                    metric_key=metric_key,
                )
                if path:
                    generated.append(path)

                # 2b. Ablation delta chart (horizontal bars)
                path = plot_ablation_deltas(
                    cs, output_dir / "ablation_analysis.png",
                    metric_key=metric_key,
                    higher_is_better=(metric_direction != "minimize"),
                )
                if path:
                    generated.append(path)

                # 2c. Metric heatmap (condition × metric)
                path = plot_metric_heatmap(
                    cs, output_dir / "metric_heatmap.png",
                )
                if path:
                    generated.append(path)

            # 2d. Raw metrics comparison (fallback, limited)
            ms = summary.get("metrics_summary", {})
            if ms:
                ms = {k: v for k, v in ms.items() if not _is_excluded_metric(k)}
                if ms:
                    path = plot_experiment_comparison(
                        ms, output_dir / "experiment_comparison.png"
                    )
                    if path:
                        generated.append(path)
        except (json.JSONDecodeError, OSError):
            pass

    # 3. Iteration scores
    iter_path = run_dir / "iteration_summary.json"
    if iter_path.exists():
        try:
            iter_data = json.loads(iter_path.read_text(encoding="utf-8"))
            scores = iter_data.get("iteration_scores", [])
            threshold = iter_data.get("quality_threshold", 7.0)
            path = plot_iteration_scores(
                scores, output_dir / "iteration_scores.png", threshold=threshold
            )
            if path:
                generated.append(path)
        except (json.JSONDecodeError, OSError):
            pass

    logger.info("Generated %d chart(s) in %s", len(generated), output_dir)
    return generated
