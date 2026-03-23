"""Critic Agent — tri-modal review of rendered charts.

Reviews each chart on three dimensions (inspired by PlotGen):
1. **Numerical accuracy** — verifies plotted values match source data
2. **Text correctness** — checks labels, legends, captions are accurate
3. **Visual quality** — LLM-based assessment of academic publication standards

Outputs pass/fail per figure with specific fix suggestions.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from researchclaw.agents.base import BaseAgent, AgentStepResult

logger = logging.getLogger(__name__)


class CriticAgent(BaseAgent):
    """Reviews rendered charts for accuracy and quality."""

    name = "figure_critic"

    def __init__(
        self,
        llm: Any,
        *,
        strict_mode: bool = False,
    ) -> None:
        super().__init__(llm)
        self._strict = strict_mode

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def execute(self, context: dict[str, Any]) -> AgentStepResult:
        """Review all rendered figures.

        Context keys:
            rendered (list[dict]): From Renderer — each has 'figure_id',
                'success', 'output_path', 'script_path', 'title', 'caption'
            scripts (list[dict]): From CodeGen — has the source scripts
            condition_summaries (dict): Source data for numerical verification
            metrics_summary (dict): Source metrics
            metric_key (str): Primary metric name
        """
        try:
            rendered = context.get("rendered", [])
            scripts = context.get("scripts", [])
            condition_summaries = context.get("condition_summaries", {})
            metrics_summary = context.get("metrics_summary", {})
            metric_key = context.get("metric_key", "primary_metric")

            # Build script lookup
            script_map: dict[str, dict[str, Any]] = {}
            for s in scripts:
                # BUG-38: skip non-dict entries
                if not isinstance(s, dict):
                    self.logger.warning("Skipping non-dict script entry: %s", type(s))
                    continue
                script_map[s.get("figure_id", "")] = s

            reviews: list[dict[str, Any]] = []
            all_passed = True

            for fig in rendered:
                # BUG-38: skip non-dict entries
                if not isinstance(fig, dict):
                    self.logger.warning("Skipping non-dict rendered entry: %s", type(fig))
                    continue
                figure_id = fig.get("figure_id", "unknown")
                if not fig.get("success"):
                    reviews.append({
                        "figure_id": figure_id,
                        "passed": False,
                        "issues": [{"type": "render_failure", "message": fig.get("error", "Render failed")}],
                    })
                    all_passed = False
                    continue

                script_info = script_map.get(figure_id, {})
                script_code = script_info.get("script", "")

                review = self._review_figure(
                    figure_id=figure_id,
                    script_code=script_code,
                    fig_info=fig,
                    condition_summaries=condition_summaries,
                    metrics_summary=metrics_summary,
                    metric_key=metric_key,
                )
                reviews.append(review)
                if not review["passed"]:
                    all_passed = False

            passed_count = sum(1 for r in reviews if r["passed"])
            self.logger.info(
                "Critic review: %d/%d figures passed",
                passed_count, len(reviews),
            )

            return self._make_result(
                success=True,
                data={
                    "reviews": reviews,
                    "all_passed": all_passed,
                    "passed_count": passed_count,
                    "total_count": len(reviews),
                },
            )
        except Exception as exc:
            self.logger.error("Critic failed: %s", exc)
            return self._make_result(False, error=str(exc))

    # ------------------------------------------------------------------
    # Per-figure review
    # ------------------------------------------------------------------

    def _review_figure(
        self,
        *,
        figure_id: str,
        script_code: str,
        fig_info: dict[str, Any],
        condition_summaries: dict[str, Any],
        metrics_summary: dict[str, Any],
        metric_key: str,
    ) -> dict[str, Any]:
        """Review a single rendered figure on four dimensions."""
        issues: list[dict[str, str]] = []

        # Dimension 1: Numerical accuracy
        num_issues = self._check_numerical_accuracy(
            script_code, condition_summaries, metric_key
        )
        issues.extend(num_issues)

        # Dimension 2: Text correctness
        text_issues = self._check_text_correctness(
            script_code, fig_info
        )
        issues.extend(text_issues)

        # Dimension 3: Visual quality (LLM-based)
        quality_issues = self._check_visual_quality(
            script_code, fig_info
        )
        issues.extend(quality_issues)

        # Dimension 4: Rendered image validation (pixel-level)
        output_path = fig_info.get("output_path", "")
        if output_path:
            pixel_issues = self._check_rendered_image(output_path)
            issues.extend(pixel_issues)

        # Determine pass/fail
        critical_issues = [i for i in issues if i.get("severity") == "critical"]
        passed = len(critical_issues) == 0

        if self._strict:
            passed = len(issues) == 0

        return {
            "figure_id": figure_id,
            "passed": passed,
            "issues": issues,
            "issue_count": len(issues),
        }

    # ------------------------------------------------------------------
    # Dimension 1: Numerical accuracy
    # ------------------------------------------------------------------

    def _check_numerical_accuracy(
        self,
        script_code: str,
        condition_summaries: dict[str, Any],
        metric_key: str,
    ) -> list[dict[str, str]]:
        """Verify that data values in the script match source data."""
        issues: list[dict[str, str]] = []

        if not condition_summaries or not script_code:
            return issues

        # Extract numerical values from script
        script_numbers = set()
        for m in re.finditer(r"(\d+\.\d{2,})", script_code):
            try:
                script_numbers.add(round(float(m.group(1)), 4))
            except ValueError:
                pass

        # Extract expected values from condition summaries
        expected_values = set()
        for cond, cdata in condition_summaries.items():
            if not isinstance(cdata, dict):
                continue
            metrics = cdata.get("metrics", {})
            for key in [metric_key, f"{metric_key}_mean"]:
                val = metrics.get(key)
                if val is not None:
                    try:
                        expected_values.add(round(float(val), 4))
                    except (ValueError, TypeError):
                        pass

        if not expected_values:
            return issues

        # Check for degenerate data (all identical or all zero)
        vals_list = sorted(expected_values)
        if len(vals_list) >= 2 and len(set(round(v, 6) for v in vals_list)) <= 1:
            issues.append({
                "type": "numerical_accuracy",
                "severity": "critical",
                "message": "All expected metric values are identical — chart will be uninformative",
            })
        if all(v == 0 for v in vals_list):
            issues.append({
                "type": "numerical_accuracy",
                "severity": "critical",
                "message": "All expected metric values are zero — chart will show no meaningful data",
            })

        # Check if script contains the expected values
        found = expected_values & script_numbers
        missing = expected_values - script_numbers

        if missing and len(missing) > len(expected_values) / 2:
            issues.append({
                "type": "numerical_accuracy",
                "severity": "critical",
                "message": (
                    f"Script may not contain correct data values. "
                    f"Expected values like {list(missing)[:3]} not found in script. "
                    f"Found values: {list(found)[:5]}"
                ),
            })

        return issues

    # ------------------------------------------------------------------
    # Dimension 2: Text correctness
    # ------------------------------------------------------------------

    def _check_text_correctness(
        self,
        script_code: str,
        fig_info: dict[str, Any],
    ) -> list[dict[str, str]]:
        """Check labels, titles, and legends in the script."""
        issues: list[dict[str, str]] = []

        if not script_code:
            return issues

        # Check for axis labels
        has_xlabel = "set_xlabel" in script_code or "xlabel" in script_code
        has_ylabel = "set_ylabel" in script_code or "ylabel" in script_code
        has_title = "set_title" in script_code or ".title(" in script_code

        if not has_xlabel:
            issues.append({
                "type": "text_correctness",
                "severity": "warning",
                "message": "Missing x-axis label",
            })
        if not has_ylabel:
            issues.append({
                "type": "text_correctness",
                "severity": "warning",
                "message": "Missing y-axis label",
            })
        if not has_title:
            issues.append({
                "type": "text_correctness",
                "severity": "warning",
                "message": "Missing chart title",
            })

        # Check for savefig call
        if "savefig" not in script_code:
            issues.append({
                "type": "text_correctness",
                "severity": "critical",
                "message": "Missing fig.savefig() call — chart will not be saved",
            })

        # Check for plt.close to prevent memory leaks
        if "plt.close" not in script_code and "close()" not in script_code:
            issues.append({
                "type": "text_correctness",
                "severity": "warning",
                "message": "Missing plt.close() — may cause memory leaks",
            })

        return issues

    # ------------------------------------------------------------------
    # Dimension 3: Visual quality (LLM review)
    # ------------------------------------------------------------------

    def _check_visual_quality(
        self,
        script_code: str,
        fig_info: dict[str, Any],
    ) -> list[dict[str, str]]:
        """Use LLM to assess visual quality of the chart code."""
        if not script_code:
            return []

        system_prompt = (
            "You are an expert reviewer of scientific figures for AI conferences "
            "(NeurIPS, ICML, ICLR). Review the following matplotlib script and "
            "identify any quality issues.\n\n"
            "Check for:\n"
            "1. DPI setting (should be 300+ for publication)\n"
            "2. Font sizes (readable when printed: title ≥12pt, labels ≥10pt)\n"
            "3. Color choices (colorblind-safe, not default matplotlib)\n"
            "4. Layout (tight_layout or constrained_layout used)\n"
            "5. Grid and styling (clean, professional)\n"
            "6. Legend placement (visible, not overlapping data)\n"
            "7. Data representation (appropriate chart type for the data)\n\n"
            "Return a JSON object with:\n"
            "- quality_score: 1-10 (10 = publication ready)\n"
            "- issues: list of objects with 'type', 'severity' ('warning' or 'critical'), 'message'\n"
            "- If score >= 7 with no critical issues, the figure passes.\n"
        )

        user_prompt = (
            f"Chart title: {fig_info.get('title', 'Unknown')}\n"
            f"Chart caption: {fig_info.get('caption', '')}\n\n"
            f"Script:\n```python\n{script_code[:3000]}\n```"
        )

        result = self._chat_json(system_prompt, user_prompt, max_tokens=2048)

        issues: list[dict[str, str]] = []
        quality_score = result.get("quality_score", 5)

        for issue in result.get("issues", []):
            if isinstance(issue, dict) and issue.get("message"):
                issues.append({
                    "type": "visual_quality",
                    "severity": issue.get("severity", "warning"),
                    "message": str(issue["message"]),
                })

        if quality_score < 4:
            issues.append({
                "type": "visual_quality",
                "severity": "critical",
                "message": f"Overall quality score too low: {quality_score}/10",
            })

        return issues

    # ------------------------------------------------------------------
    # Dimension 4: Rendered image validation (pixel-level)
    # ------------------------------------------------------------------

    def _check_rendered_image(
        self, output_path: str
    ) -> list[dict[str, str]]:
        """Check the rendered PNG for visual defects via pixel analysis.

        Detects:
        - Near-blank images (>95% white) indicating degenerate/empty charts
        - Text/graphics touching image edges (possible label clipping)
        """
        issues: list[dict[str, str]] = []
        try:
            from PIL import Image
            import numpy as np

            img = Image.open(output_path).convert("RGB")
            arr = np.array(img)
            h, w, _ = arr.shape

            # Check 1: Near-blank image (>95% white pixels)
            white_mask = np.all(arr > 250, axis=2)
            white_ratio = float(np.mean(white_mask))
            if white_ratio > 0.95:
                issues.append({
                    "type": "rendered_quality",
                    "severity": "critical",
                    "message": (
                        f"Image is {white_ratio:.0%} white — likely degenerate "
                        f"or empty chart"
                    ),
                })

            # Check 2: Non-white pixels touching edges (possible clipping)
            margin = 3  # pixels
            for edge_name, edge_slice in [
                ("top", arr[:margin, :]),
                ("bottom", arr[-margin:, :]),
                ("left", arr[:, :margin]),
                ("right", arr[:, -margin:]),
            ]:
                dark_mask = np.any(edge_slice < 80, axis=-1)
                dark_ratio = float(np.mean(dark_mask))
                if dark_ratio > 0.05:
                    issues.append({
                        "type": "rendered_quality",
                        "severity": "warning",
                        "message": (
                            f"Content touching {edge_name} edge ({dark_ratio:.0%} "
                            f"dark pixels) — possible label/title clipping"
                        ),
                    })
        except ImportError:
            logger.debug("PIL not available — skipping rendered image checks")
        except Exception as exc:
            logger.debug("Rendered image check failed: %s", exc)

        return issues
