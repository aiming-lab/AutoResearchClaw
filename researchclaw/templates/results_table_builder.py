"""Pre-build LaTeX results tables from experiment data.

Generates verified, ready-to-embed LaTeX tables directly from
``experiment_summary.json``.  The LLM receives these tables as
verbatim blocks and is instructed NOT to modify the numbers.

This removes the LLM from the number-generation loop entirely.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any

from researchclaw.pipeline.verified_registry import ConditionResult, VerifiedRegistry

logger = logging.getLogger(__name__)


@dataclass
class LatexTable:
    """A single pre-built LaTeX table."""

    label: str  # e.g. "tab:main_results"
    caption: str
    latex_code: str  # Complete \begin{table}...\end{table}
    verified_values: set[float] = field(default_factory=set)
    n_conditions: int = 0
    n_total_seeds: int = 0


def build_results_tables(
    registry: VerifiedRegistry,
    *,
    metric_name: str = "Metric",
    metric_direction: str = "maximize",
    two_column: bool = False,
) -> list[LatexTable]:
    """Generate LaTeX tables from a VerifiedRegistry.

    Parameters
    ----------
    registry:
        The verified registry built from experiment data.
    metric_name:
        Human-readable name for the primary metric column.
    metric_direction:
        ``"maximize"`` or ``"minimize"`` — determines which result is bolded.
    two_column:
        If True, use ``table*`` environment (for 2-column formats like ICML).

    Returns
    -------
    list[LatexTable]
        One or more tables.  Usually just one main results table.
    """
    tables: list[LatexTable] = []

    # --- Main results table ---
    conditions = _get_reportable_conditions(registry)
    if not conditions:
        logger.warning("No reportable conditions — skipping table generation")
        return tables

    main_table = _build_main_table(
        conditions,
        metric_name=metric_name,
        metric_direction=metric_direction,
        two_column=two_column,
    )
    tables.append(main_table)

    # --- Per-seed breakdown table (if seeds > 1 for any condition) ---
    has_multi_seed = any(c.n_seeds >= 2 for c in conditions)
    if has_multi_seed:
        seed_table = _build_per_seed_table(
            conditions,
            metric_name=metric_name,
            two_column=two_column,
        )
        tables.append(seed_table)

    return tables


def _get_reportable_conditions(registry: VerifiedRegistry) -> list[ConditionResult]:
    """Filter conditions to only those with at least 1 valid seed."""
    results = []
    for cond in registry.conditions.values():
        if cond.n_seeds >= 1 and cond.mean is not None and math.isfinite(cond.mean):
            results.append(cond)
    # Sort alphabetically for consistency
    results.sort(key=lambda c: c.name)
    return results


def _build_main_table(
    conditions: list[ConditionResult],
    *,
    metric_name: str,
    metric_direction: str,
    two_column: bool,
) -> LatexTable:
    """Build the main results table with mean ± std per condition."""
    verified: set[float] = set()

    # Find best condition for bolding
    best_idx = _find_best(conditions, metric_direction)

    # Build rows
    rows: list[str] = []
    for i, cond in enumerate(conditions):
        mean_str = _fmt(cond.mean)
        if cond.mean is not None:
            verified.add(round(cond.mean, 4))

        if cond.std is not None and cond.std > 0 and cond.n_seeds >= 2:
            std_str = _fmt(cond.std)
            val_str = f"{mean_str} $\\pm$ {std_str}"
            verified.add(round(cond.std, 4))
        elif cond.n_seeds == 1:
            val_str = f"{mean_str}$^{{\\ddagger}}$"
        else:
            val_str = mean_str

        if i == best_idx:
            val_str = f"\\textbf{{{val_str}}}"

        n_str = str(cond.n_seeds)
        name_escaped = _escape_latex(cond.name)
        rows.append(f"{name_escaped} & {val_str} & {n_str} \\\\")

    # Compose table
    table_env = "table*" if two_column else "table"
    col_spec = "l c r"

    body = "\n".join(rows)
    note_lines = []
    if any(c.n_seeds == 1 for c in conditions):
        note_lines.append(
            "$^{\\ddagger}$Single seed; no standard deviation available."
        )

    notes = "\n".join(note_lines)
    if notes:
        notes = f"\n\\vspace{{2pt}}\\par\\footnotesize {notes}\n"

    latex = (
        f"\\begin{{{table_env}}}[htbp]\n"
        f"\\centering\n"
        f"\\caption{{Experimental results. "
        f"{len(conditions)} conditions evaluated.}}\n"
        f"\\label{{tab:main_results}}\n"
        f"% AUTO-GENERATED FROM EXPERIMENT DATA — DO NOT MODIFY NUMBERS\n"
        f"\\begin{{tabular}}{{{col_spec}}}\n"
        f"\\toprule\n"
        f"Method & {metric_name} & $n$ \\\\\n"
        f"\\midrule\n"
        f"{body}\n"
        f"\\bottomrule\n"
        f"\\end{{tabular}}{notes}\n"
        f"\\end{{{table_env}}}"
    )

    return LatexTable(
        label="tab:main_results",
        caption=f"Experimental results. {len(conditions)} conditions evaluated.",
        latex_code=latex,
        verified_values=verified,
        n_conditions=len(conditions),
        n_total_seeds=sum(c.n_seeds for c in conditions),
    )


def _build_per_seed_table(
    conditions: list[ConditionResult],
    *,
    metric_name: str,
    two_column: bool,
) -> LatexTable:
    """Build per-seed breakdown table."""
    verified: set[float] = set()

    # Determine max seeds across conditions
    max_seeds = max(c.n_seeds for c in conditions)

    # Build header
    seed_cols = " & ".join(f"Seed {i}" for i in range(max_seeds))
    col_spec = "l " + " ".join("r" for _ in range(max_seeds)) + " r"

    # Build rows
    rows: list[str] = []
    for cond in conditions:
        name_escaped = _escape_latex(cond.name)
        cells = []
        for seed_idx in range(max_seeds):
            val = cond.per_seed_values.get(seed_idx)
            if val is not None and math.isfinite(val):
                cells.append(_fmt(val))
                verified.add(round(val, 4))
            else:
                cells.append("---")
        mean_str = _fmt(cond.mean) if cond.mean is not None else "---"
        cells_str = " & ".join(cells)
        rows.append(f"{name_escaped} & {cells_str} & {mean_str} \\\\")

    body = "\n".join(rows)
    table_env = "table*" if two_column else "table"

    latex = (
        f"\\begin{{{table_env}}}[htbp]\n"
        f"\\centering\n"
        f"\\caption{{Per-seed results breakdown.}}\n"
        f"\\label{{tab:per_seed}}\n"
        f"% AUTO-GENERATED FROM EXPERIMENT DATA — DO NOT MODIFY NUMBERS\n"
        f"\\begin{{tabular}}{{{col_spec}}}\n"
        f"\\toprule\n"
        f"Method & {seed_cols} & Mean \\\\\n"
        f"\\midrule\n"
        f"{body}\n"
        f"\\bottomrule\n"
        f"\\end{{tabular}}\n"
        f"\\end{{{table_env}}}"
    )

    return LatexTable(
        label="tab:per_seed",
        caption="Per-seed results breakdown.",
        latex_code=latex,
        verified_values=verified,
        n_conditions=len(conditions),
        n_total_seeds=sum(c.n_seeds for c in conditions),
    )


def build_condition_whitelist(registry: VerifiedRegistry) -> str:
    """Generate a human-readable condition whitelist for the LLM prompt.

    Example output::

        CONDITION WHITELIST (you may ONLY discuss these conditions):
        - DQN (3 seeds, mean=206.10)
        - DQN+Abstraction (3 seeds, mean=278.93)
        - DQN+RawCount (3 seeds, mean=180.80)
    """
    lines = ["CONDITION WHITELIST (you may ONLY discuss these conditions):"]
    for cond in sorted(registry.conditions.values(), key=lambda c: c.name):
        if cond.n_seeds == 0 or cond.mean is None or not math.isfinite(cond.mean):
            continue
        mean_str = f"{cond.mean:.4f}"
        lines.append(f"- {cond.name} ({cond.n_seeds} seed(s), mean={mean_str})")

    if len(lines) == 1:
        lines.append("- (no conditions completed)")

    return "\n".join(lines)


def _find_best(conditions: list[ConditionResult], direction: str) -> int | None:
    """Return index of best condition, or None if empty."""
    if not conditions:
        return None
    best_idx = 0
    for i, c in enumerate(conditions):
        if c.mean is None:
            continue
        if conditions[best_idx].mean is None:
            best_idx = i
            continue
        if direction == "maximize" and c.mean > conditions[best_idx].mean:
            best_idx = i
        elif direction == "minimize" and c.mean < conditions[best_idx].mean:
            best_idx = i
    return best_idx


def _fmt(value: float | None) -> str:
    """Format a number for LaTeX tables with sig-fig-aware rounding."""
    if value is None or not math.isfinite(value):
        return "---"
    # Sig-fig-aware formatting (same approach as BUG-83 fix)
    av = abs(value)
    if av >= 100:
        return f"{value:.2f}"
    elif av >= 1:
        return f"{value:.4f}"
    elif av >= 0.001:
        return f"{value:.4f}"
    elif av > 0:
        # Very small values: use 2 significant figures
        import decimal
        d = decimal.Decimal(str(value)).normalize()
        # Count leading zeros after decimal point
        exp = d.adjusted()
        sig_digits = max(2, -exp + 1)
        return f"{value:.{sig_digits}f}"
    else:
        return "0.0000"


def _escape_latex(text: str) -> str:
    """Escape special LaTeX characters in condition names."""
    # Backslash must be first to avoid double-escaping
    replacements = [
        ("\\", "\\textbackslash{}"),
        ("&", "\\&"),
        ("%", "\\%"),
        ("#", "\\#"),
        ("_", "\\_"),
        ("$", "\\$"),
        ("{", "\\{"),
        ("}", "\\}"),
        ("~", "\\textasciitilde{}"),
        ("^", "\\textasciicircum{}"),
    ]
    for old, new in replacements:
        text = text.replace(old, new)
    return text
