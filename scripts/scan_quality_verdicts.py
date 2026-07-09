#!/usr/bin/env python3
"""Scan historical run outputs for Stage 20 quality verdict behavior.

This is a read-only diagnostic helper for deciding whether release_check.py's
degraded exit code is reachable with real Stage 20 verdict strings.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DEFAULT_THRESHOLD = 5.0
FAIL_VERDICTS = {"revise", "reject", "failed", "fail"}


@dataclass(frozen=True)
class RunQualityRow:
    run_dir: Path
    score: float | None
    threshold: float
    threshold_source: str
    verdict: str
    summary_degraded: bool
    has_signal: bool
    low_score: bool
    fail_verdict: bool

    @property
    def real_degraded(self) -> bool:
        return self.has_signal or self.summary_degraded


def read_json(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def float_value(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value.strip())
        except ValueError:
            return None
    return None


def quality_score(quality: dict[str, Any]) -> float | None:
    for key in ("score_1_to_10", "score", "quality_score", "overall_score"):
        score = float_value(quality.get(key))
        if score is not None:
            return score
    return None


def threshold_for_run(run_dir: Path, default_threshold: float) -> tuple[float, str]:
    signal = read_json(run_dir / "degradation_signal.json")
    threshold = float_value(signal.get("threshold"))
    if threshold is not None:
        return threshold, "degradation_signal.json"
    return default_threshold, "default"


def iter_quality_rows(roots: list[Path], default_threshold: float) -> list[RunQualityRow]:
    rows: list[RunQualityRow] = []
    seen: set[Path] = set()
    for root in roots:
        if not root.exists():
            continue
        for qpath in sorted(root.rglob("stage-20/quality_report.json")):
            qpath = qpath.resolve()
            if qpath in seen:
                continue
            seen.add(qpath)
            run_dir = qpath.parent.parent
            quality = read_json(qpath)
            summary = read_json(run_dir / "pipeline_summary.json")
            score = quality_score(quality)
            threshold, threshold_source = threshold_for_run(run_dir, default_threshold)
            verdict = str(quality.get("verdict", "")).strip().lower()
            has_signal = (run_dir / "degradation_signal.json").exists()
            low_score = score is not None and score < threshold
            rows.append(
                RunQualityRow(
                    run_dir=run_dir,
                    score=score,
                    threshold=threshold,
                    threshold_source=threshold_source,
                    verdict=verdict,
                    summary_degraded=bool(summary.get("degraded")),
                    has_signal=has_signal,
                    low_score=low_score,
                    fail_verdict=verdict in FAIL_VERDICTS,
                )
            )
    return rows


def print_table(rows: list[RunQualityRow], limit: int) -> None:
    print("run_dir\tscore\tthreshold\tthreshold_source\tverdict\tsummary_degraded\thas_signal\tlow_score\tfail_verdict")
    shown = rows if limit <= 0 else rows[:limit]
    for row in shown:
        print(
            "\t".join(
                [
                    str(row.run_dir),
                    "" if row.score is None else f"{row.score:g}",
                    f"{row.threshold:g}",
                    row.threshold_source,
                    row.verdict,
                    str(row.summary_degraded),
                    str(row.has_signal),
                    str(row.low_score),
                    str(row.fail_verdict),
                ]
            )
        )
    if limit > 0 and len(rows) > limit:
        print(f"... truncated {len(rows) - limit} rows; rerun with --limit 0 to show all")


def print_summary(rows: list[RunQualityRow]) -> None:
    degraded = [row for row in rows if row.real_degraded]
    print()
    print(f"quality reports found: {len(rows)}")
    print(f"release-check degraded cohort (root signal exists or summary.degraded=true): {len(degraded)}")
    print()
    print("verdict distribution in release-check degraded cohort:")
    verdict_counts = Counter(row.verdict or "<empty>" for row in degraded)
    if verdict_counts:
        for verdict, count in verdict_counts.most_common():
            print(f"  {verdict!r}: {count}")
    else:
        print("  <none>")
    fail_count = sum(1 for row in degraded if row.fail_verdict)
    ratio = (fail_count / len(degraded)) if degraded else 0.0
    print()
    print(
        "degraded rows with verdict in "
        f"{sorted(FAIL_VERDICTS)}: {fail_count}/{len(degraded)} ({ratio:.1%})"
    )
    if not degraded:
        print("conclusion: no release-check degraded samples found; exit 2 reachability cannot be inferred.")
    elif ratio >= 0.8:
        print("conclusion: exit 2 is likely mostly unreachable unless verdict handling changes.")
    elif ratio <= 0.2:
        print("conclusion: exit 2 appears reachable for most degraded-only runs.")
    else:
        print("conclusion: exit 2 is verdict-sensitive; normalize verdicts or use objective degradation signals.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("roots", nargs="+", type=Path, help="Artifact root(s) or run directory parents to scan.")
    parser.add_argument(
        "--default-threshold",
        type=float,
        default=DEFAULT_THRESHOLD,
        help="Fallback threshold when degradation_signal.json has no threshold.",
    )
    parser.add_argument("--limit", type=int, default=200, help="Maximum table rows to print; use 0 for all.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rows = iter_quality_rows(args.roots, args.default_threshold)
    print_table(rows, args.limit)
    print_summary(rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
