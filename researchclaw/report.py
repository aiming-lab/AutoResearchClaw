"""Generate human-readable run reports from pipeline artifacts."""

# pyright: basic
from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def generate_report(run_dir: Path) -> str:
    """Generate a Markdown report from a pipeline run directory.

    Args:
        run_dir: Path to the run artifacts directory (e.g., artifacts/rc-xxx/)

    Returns:
        Markdown string with the report content.

    Raises:
        FileNotFoundError: If run_dir doesn't exist.
        ValueError: If run_dir has no pipeline_summary.json.
    """
    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")

    summary_path = run_dir / "pipeline_summary.json"
    if not summary_path.exists():
        raise ValueError(f"No pipeline_summary.json found in {run_dir}")

    loaded = json.loads(summary_path.read_text(encoding="utf-8"))
    summary = loaded if isinstance(loaded, dict) else {}

    sections = []
    sections.append(_header(summary, run_dir))
    sections.append(_paper_section(run_dir))
    sections.append(_experiment_section(run_dir))
    sections.append(_citation_section(run_dir))
    sections.append(_warnings_section(summary))

    return "\n\n".join(section for section in sections if section)


def _header(summary: dict[str, Any], run_dir: Path) -> str:
    run_id = summary.get("run_id", "unknown")
    stages_done = summary.get("stages_done", 0)
    stages_total = summary.get("stages_executed", 0)
    status = summary.get("final_status", "unknown")
    generated = summary.get("generated", "unknown")

    status_icon = "✅" if status == "done" else "❌" if status == "failed" else "⚠️"

    lines = [
        "# ResearchClaw Run Report",
        "",
        f"**Run ID**: {run_id}",
        f"**Date**: {generated}",
        f"**Status**: {status_icon} {status} ({stages_done}/{stages_total} stages done)",
        f"**Artifacts**: `{run_dir}`",
    ]
    return "\n".join(lines)


def _paper_section(run_dir: Path) -> str:
    lines = ["## Paper"]

    draft_path = run_dir / "stage-17" / "paper_draft.md"
    if draft_path.exists():
        text = draft_path.read_text(encoding="utf-8")
        word_count = len(text.split())
        lines.append(
            f"- Draft: `{draft_path.relative_to(run_dir)}` (~{word_count} words)"
        )
    else:
        lines.append("- Draft: not generated")

    final_path = run_dir / "stage-22" / "paper_final.md"
    if final_path.exists():
        lines.append(f"- Final: `{final_path.relative_to(run_dir)}`")

    tex_path = run_dir / "stage-22" / "paper.tex"
    if tex_path.exists():
        lines.append(f"- LaTeX: `{tex_path.relative_to(run_dir)}`")

    rev_path = run_dir / "stage-19" / "paper_revised.md"
    if rev_path.exists():
        lines.append(f"- Revised: `{rev_path.relative_to(run_dir)}`")

    return "\n".join(lines)


def _experiment_section(run_dir: Path) -> str:
    lines = ["## Experiments"]

    code_path = run_dir / "stage-10" / "experiment_code.py"
    if code_path.exists():
        lines.append(f"- Code: `{code_path.relative_to(run_dir)}`")

    results_path = run_dir / "stage-12" / "experiment_results.json"
    if results_path.exists():
        try:
            loaded = json.loads(results_path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                data = loaded
                runs_default: list[Any] = []
                iterations = data.get("iterations", data.get("runs", runs_default))
                if isinstance(iterations, list):
                    lines.append(f"- Runs: {len(iterations)} iterations")
                best = data.get("best_metric") or data.get("best_result")
                if best is not None:
                    lines.append(f"- Best metric: {best}")
        except (json.JSONDecodeError, TypeError):
            lines.append("- Results: present (parse error)")
    else:
        lines.append("- Results: not available")

    # BUG-215: Also search stage-14* versioned dirs when stage-14/ is missing.
    analysis_path = run_dir / "stage-14" / "analysis.md"
    if not analysis_path.exists():
        for _s14 in sorted(run_dir.glob("stage-14*"), reverse=True):
            _alt = _s14 / "analysis.md"
            if _alt.exists():
                analysis_path = _alt
                break
    if analysis_path.exists():
        lines.append(f"- Analysis: `{analysis_path.relative_to(run_dir)}`")

    return "\n".join(lines)


def _citation_section(run_dir: Path) -> str:
    lines = ["## Citations"]

    bib_path = run_dir / "stage-22" / "references.bib"
    if not bib_path.exists():
        bib_path = run_dir / "stage-04" / "references.bib"

    if bib_path.exists():
        text = bib_path.read_text(encoding="utf-8")
        entries = re.findall(r"@\w+\{", text)
        lines.append(f"- References: {len(entries)} BibTeX entries")
    else:
        lines.append("- References: not available")

    verify_path = run_dir / "stage-23" / "verification_report.json"
    if verify_path.exists():
        try:
            loaded = json.loads(verify_path.read_text(encoding="utf-8"))
            vdata = loaded if isinstance(loaded, dict) else {}
            total = int(vdata.get("total_references", 0))
            verified = int(vdata.get("verified_count", 0))
            suspicious = int(vdata.get("suspicious_count", 0))
            hallucinated = int(vdata.get("hallucinated_count", 0))
            pct = f"{verified / total * 100:.1f}%" if total > 0 else "N/A"
            lines.append(f"- Verified: {verified}/{total} ({pct})")
            if suspicious:
                lines.append(f"- Suspicious: {suspicious}")
            if hallucinated:
                lines.append(f"- Hallucinated: {hallucinated}")
        except (json.JSONDecodeError, TypeError, ZeroDivisionError):
            lines.append("- Verification: present (parse error)")
    else:
        lines.append("- Verification: not run")

    return "\n".join(lines)


def _warnings_section(summary: dict[str, Any]) -> str:
    warnings: list[str] = []

    stages_failed = summary.get("stages_failed", 0)
    if stages_failed:
        warnings.append(f"- ⚠️ {stages_failed} stage(s) failed during execution")

    content_metrics = summary.get("content_metrics", {})
    if isinstance(content_metrics, dict):
        template_ratio = content_metrics.get("template_ratio")
        if isinstance(template_ratio, (int, float)) and template_ratio > 0.1:
            warnings.append(
                f"- ⚠️ Template content detected: {template_ratio:.1%} of paper may be template text"
            )

        degraded = content_metrics.get("degraded_sources", [])
        if isinstance(degraded, list) and degraded:
            warnings.append(f"- ⚠️ Degraded sources: {', '.join(degraded)}")

    if not warnings:
        return ""

    return "## Warnings\n" + "\n".join(warnings)


def print_report(run_dir: Path) -> None:
    print(generate_report(run_dir))


def write_report(run_dir: Path, output_path: Path) -> None:
    report = generate_report(run_dir)
    _ = output_path.write_text(report, encoding="utf-8")
