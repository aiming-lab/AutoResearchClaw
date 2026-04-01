"""Tests for paper_verifier — post-generation fabrication detection."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from researchclaw.pipeline.paper_verifier import (
    VerificationResult,
    verify_paper,
)
from researchclaw.pipeline.verified_registry import VerifiedRegistry

ARTIFACTS = Path(__file__).resolve().parent.parent / "artifacts"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_registry(**kwargs) -> VerifiedRegistry:
    summary = {"best_run": {"metrics": {}}, "condition_summaries": {}, "metrics_summary": {}}
    conditions = kwargs.get("conditions", {})
    for cond_name, seeds in conditions.items():
        for seed_idx, value in seeds.items():
            summary["best_run"]["metrics"][f"{cond_name}/{seed_idx}/metric"] = value
        mean_val = sum(seeds.values()) / len(seeds)
        summary["condition_summaries"][cond_name] = {"metrics": {"metric": mean_val}}
    pm = kwargs.get("primary_metric")
    if pm is not None:
        summary["best_run"]["metrics"]["primary_metric"] = pm
    return VerifiedRegistry.from_experiment(summary)


# ---------------------------------------------------------------------------
# Unit tests — clean paper
# ---------------------------------------------------------------------------


class TestCleanPaper:
    def test_all_numbers_verified_passes(self):
        reg = _make_registry(
            conditions={"Baseline": {0: 80.0, 1: 82.0}, "Proposed": {0: 90.0, 1: 92.0}},
            primary_metric=91.0,
        )
        tex = r"""
\section{Results}
Our proposed method achieves 91.0000 on the primary metric,
compared to 81.0000 for the baseline.
\begin{table}[htbp]
\centering
\begin{tabular}{lcc}
\toprule
Method & Metric & $n$ \\
\midrule
Baseline & 81.0000 $\pm$ 1.4142 & 2 \\
Proposed & 91.0000 $\pm$ 1.4142 & 2 \\
\bottomrule
\end{tabular}
\end{table}
"""
        result = verify_paper(tex, reg)
        assert result.severity == "PASS"
        assert result.strict_violations == 0

    def test_common_constants_allowed(self):
        reg = _make_registry(conditions={"A": {0: 80.0}})
        tex = r"""
\section{Experimental Setup}
We use a batch size of 64 and train for 100 epochs
with a learning rate of 0.001.
"""
        result = verify_paper(tex, reg)
        assert result.severity == "PASS"

    def test_year_numbers_allowed(self):
        reg = _make_registry(conditions={"A": {0: 80.0}})
        tex = r"""
\section{Introduction}
Following the work of Smith et al. (2025), we propose...
"""
        result = verify_paper(tex, reg)
        assert result.severity == "PASS"


# ---------------------------------------------------------------------------
# Unit tests — fabricated numbers
# ---------------------------------------------------------------------------


class TestFabricatedNumbers:
    def test_fabricated_in_results_rejects(self):
        reg = _make_registry(
            conditions={"Baseline": {0: 80.0}, "Proposed": {0: 90.0}},
        )
        tex = r"""
\section{Results}
Our method achieves 95.5 accuracy.
"""
        result = verify_paper(tex, reg)
        assert result.severity == "REJECT"
        assert result.strict_violations >= 1
        assert any(abs(u.value - 95.5) < 0.01 for u in result.unverified_numbers)

    def test_fabricated_in_table_rejects(self):
        reg = _make_registry(conditions={"A": {0: 80.0}})
        tex = r"""
\section{Results}
\begin{table}[h]
\begin{tabular}{lc}
A & 85.3 \\
\end{tabular}
\end{table}
"""
        result = verify_paper(tex, reg)
        assert result.severity == "REJECT"

    def test_fabricated_in_discussion_warns(self):
        reg = _make_registry(conditions={"A": {0: 80.0}})
        tex = r"""
\section{Discussion}
Compared to prior work reporting 95.5 accuracy, our result is lower.
"""
        result = verify_paper(tex, reg)
        # In Discussion → warning, not reject
        assert result.severity == "WARN"
        assert result.lenient_violations >= 1

    def test_numbers_in_cite_skipped(self):
        reg = _make_registry(conditions={"A": {0: 80.0}})
        tex = r"""
\section{Results}
As shown by \cite{smith2025deep}, our method works.
"""
        result = verify_paper(tex, reg)
        assert result.severity == "PASS"

    def test_numbers_in_comments_skipped(self):
        reg = _make_registry(conditions={"A": {0: 80.0}})
        tex = r"""
\section{Results}
% This is a comment with fake number 99.99
Our method achieves 80.0.
"""
        result = verify_paper(tex, reg)
        assert result.severity == "PASS"


# ---------------------------------------------------------------------------
# Unit tests — fabricated conditions
# ---------------------------------------------------------------------------


class TestFabricatedConditions:
    def test_unknown_condition_in_table(self):
        reg = _make_registry(conditions={"DQN": {0: 80.0}, "DQN+Abstraction": {0: 90.0}})
        tex = r"""
\section{Results}
\begin{table}[h]
\begin{tabular}{lc}
DQN & 80.0 \\
DQN+Abstraction & 90.0 \\
PPO & 75.0 \\
\end{tabular}
\end{table}
"""
        result = verify_paper(tex, reg)
        assert len(result.fabricated_conditions) >= 1
        assert any(fc.name == "PPO" for fc in result.fabricated_conditions)
        assert result.severity == "REJECT"


# ---------------------------------------------------------------------------
# Unit tests — fabrication rate
# ---------------------------------------------------------------------------


class TestFabricationRate:
    def test_rate_zero_for_clean_paper(self):
        reg = _make_registry(conditions={"A": {0: 80.0}})
        tex = r"""
\section{Results}
Accuracy is 80.0.
"""
        result = verify_paper(tex, reg)
        assert result.fabrication_rate == 0.0

    def test_rate_nonzero_for_fabricated(self):
        reg = _make_registry(conditions={"A": {0: 80.0}})
        tex = r"""
\section{Results}
Accuracy is 99.99 and loss is 45.67.
"""
        result = verify_paper(tex, reg)
        assert result.fabrication_rate > 0.0


# ---------------------------------------------------------------------------
# Integration — real fabricated papers
# ---------------------------------------------------------------------------


class TestRealPapers:
    def _load(self, run_id: str) -> tuple[str, VerifiedRegistry]:
        pattern = f"rc-*-{run_id}"
        matches = sorted(ARTIFACTS.glob(pattern))
        if not matches:
            pytest.skip(f"Artifact {run_id} not found")
        base = matches[0]
        tex_path = base / "stage-22" / "paper.tex"
        summary_path = base / "stage-14" / "experiment_summary.json"
        ref_path = base / "stage-13" / "refinement_log.json"
        if not tex_path.exists() or not summary_path.exists():
            pytest.skip(f"Missing files for {run_id}")
        tex = tex_path.read_text(encoding="utf-8")
        summary = json.loads(summary_path.read_text())
        ref_log = None
        if ref_path.exists():
            ref_log = json.loads(ref_path.read_text())
        reg = VerifiedRegistry.from_experiment(summary, ref_log)
        return tex, reg

    def test_run_e57360_severe_fabrication_detected(self):
        """Run 38 (LACE) — audit found SEVERE fabrication.
        The verifier should REJECT this paper."""
        tex, reg = self._load("e57360")
        result = verify_paper(tex, reg)
        assert result.severity == "REJECT", (
            f"Expected REJECT for severely fabricated paper, got {result.severity}. "
            f"Unverified: {len(result.unverified_numbers)}, "
            f"Fabricated conditions: {[fc.name for fc in result.fabricated_conditions]}"
        )

    def test_run_6a1ec9_severe_fabrication_detected(self):
        """Run 6a1ec9 (FAME) — audit found SEVERE fabrication."""
        tex, reg = self._load("6a1ec9")
        result = verify_paper(tex, reg)
        assert result.severity == "REJECT"

    def test_run_85fefc_fabrication_detected(self):
        """Run 85fefc (CRAFT) — audit found SEVERE fabrication."""
        tex, reg = self._load("85fefc")
        result = verify_paper(tex, reg)
        # Should detect at least some issues
        assert len(result.unverified_numbers) > 0 or len(result.fabricated_conditions) > 0

    def test_run_acbdfa_moderate_fabrication(self):
        """Run acbdfa (CTS) — audit found MODERATE fabrication."""
        tex, reg = self._load("acbdfa")
        result = verify_paper(tex, reg)
        # May or may not reject (moderate case), but should find issues
        assert len(result.unverified_numbers) > 0 or result.lenient_violations > 0
