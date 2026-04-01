"""Decision Agent — decides what figures are needed and how to generate them.

Analyzes the paper draft/outline and experiment data to determine:
  - Which sections need figures
  - What TYPE of figure each section needs
  - Which generation BACKEND to use:
    * ``code``  — Code-to-Viz (Matplotlib/TikZ) for data-driven charts
    * ``image`` — Nano Banana (Gemini) for architecture/conceptual diagrams

This agent acts as the "director" before the Planner/CodeGen/NanoBanana
sub-agents execute.  It does NOT generate any figures itself.

References:
  - Visual ChatGPT (Wu et al., 2023): LLM as controller
  - Nano Banana: Gemini native image generation (google.genai)
"""

from __future__ import annotations

import json
import logging
from typing import Any

from researchclaw.agents.base import BaseAgent, AgentStepResult
from researchclaw.utils.thinking_tags import strip_thinking_tags

logger = logging.getLogger(__name__)


def _safe_priority(val: object, default: int = 2) -> int:
    """Convert priority to int, clamped to 1-3."""
    try:
        return max(1, min(3, int(val)))  # type: ignore[arg-type]
    except (ValueError, TypeError):
        return default

# ---------------------------------------------------------------------------
# Figure categories
# ---------------------------------------------------------------------------

FIGURE_CATEGORY_DATA = "code"   # data-driven → Matplotlib / TikZ
FIGURE_CATEGORY_IMAGE = "image"  # conceptual → Nano Banana (Gemini)

_DECISION_SYSTEM_PROMPT = """\
You are an expert academic paper analyst.  Your job is to analyze a research
paper's content and decide which figures are needed.

For each figure, decide:
1. **section** — Which section of the paper it belongs to (e.g. "Method",
   "Results", "Introduction", "Architecture")
2. **figure_type** — A descriptive type:
   - For data/experiment figures: "bar_comparison", "line_chart", "heatmap",
     "confusion_matrix", "training_curve", "ablation_chart", "scatter_plot"
   - For conceptual/architecture figures: "architecture_diagram",
     "method_flowchart", "pipeline_overview", "concept_illustration",
     "system_diagram", "attention_visualization", "comparison_illustration"
3. **backend** — Which generation backend:
   - "code" for data-driven charts (bar charts, line plots, heatmaps) → will
     be generated via Matplotlib/Seaborn or TikZ/PGFPlots
   - "image" for conceptual diagrams (architecture, pipeline, method) → will
     be generated via Gemini Nano Banana image generation
4. **description** — A detailed description of what the figure should show
5. **priority** — 1 (essential) to 3 (nice-to-have)

Return a JSON array of figure decisions.  Example:
```json
[
  {
    "section": "Method",
    "figure_type": "architecture_diagram",
    "backend": "image",
    "description": "Overview of the proposed model architecture showing encoder-decoder structure with attention mechanism",
    "priority": 1
  },
  {
    "section": "Results",
    "figure_type": "bar_comparison",
    "backend": "code",
    "description": "Bar chart comparing accuracy of proposed method vs baselines on CIFAR-100",
    "priority": 1
  }
]
```

RULES:
- Every research paper should have at least 1 architecture/method figure
- Every paper with experiments should have at least 2 result figures
- Prioritize figures that make the paper more convincing
- Do NOT generate duplicate or redundant figures
- Return ONLY valid JSON, no explanation
- Do NOT include <think> or </think> tags
"""


class FigureDecisionAgent(BaseAgent):
    """Decides what figures are needed and which backend generates them.

    This agent analyzes the paper context (topic, draft, experiment data)
    and produces a *figure decision plan* — a list of figure requests tagged
    with either ``"code"`` (Code-to-Viz) or ``"image"`` (Nano Banana).

    The downstream orchestrator then routes each request to the appropriate
    generation sub-agent.
    """

    name = "figure_decision"

    def __init__(
        self,
        llm: Any,
        *,
        min_figures: int = 3,
        max_figures: int = 10,
    ) -> None:
        super().__init__(llm)
        self._min_figures = min_figures
        self._max_figures = max_figures

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def execute(self, context: dict[str, Any]) -> AgentStepResult:
        """Analyze context and produce figure decisions.

        Context keys:
            topic (str): Research topic
            hypothesis (str): Research hypothesis
            paper_draft (str): Current paper draft / outline (markdown)
            experiment_results (dict): Parsed experiment data (if any)
            condition_summaries (dict): Per-condition stats (if any)
            has_experiments (bool): Whether experiments were conducted
        """
        topic = context.get("topic", "")
        hypothesis = context.get("hypothesis", "")
        paper_draft = context.get("paper_draft", "")
        has_experiments = context.get("has_experiments", True)
        experiment_results = context.get("experiment_results", {})
        condition_summaries = context.get("condition_summaries", {})

        # ── Try LLM-based decision ────────────────────────────────────
        if self._llm is not None:
            try:
                decisions = self._llm_decide(
                    topic=topic,
                    hypothesis=hypothesis,
                    paper_draft=paper_draft,
                    has_experiments=has_experiments,
                    experiment_results=experiment_results,
                    condition_summaries=condition_summaries,
                )
                # Enforce bounds
                decisions = self._enforce_bounds(decisions, has_experiments)

                return self._make_result(
                    True,
                    data={
                        "decisions": decisions,
                        "code_figures": [
                            d for d in decisions if d["backend"] == "code"
                        ],
                        "image_figures": [
                            d for d in decisions if d["backend"] == "image"
                        ],
                        "total": len(decisions),
                    },
                )
            except Exception as e:
                logger.warning("LLM decision failed, using heuristic: %s", e)

        # ── Fallback: heuristic decision ──────────────────────────────
        decisions = self._heuristic_decide(
            topic=topic,
            has_experiments=has_experiments,
            condition_summaries=condition_summaries,
        )

        return self._make_result(
            True,
            data={
                "decisions": decisions,
                "code_figures": [
                    d for d in decisions if d["backend"] == "code"
                ],
                "image_figures": [
                    d for d in decisions if d["backend"] == "image"
                ],
                "total": len(decisions),
            },
        )

    # ------------------------------------------------------------------
    # LLM-based decision
    # ------------------------------------------------------------------

    def _llm_decide(
        self,
        *,
        topic: str,
        hypothesis: str,
        paper_draft: str,
        has_experiments: bool,
        experiment_results: dict[str, Any],
        condition_summaries: dict[str, Any],
    ) -> list[dict[str, Any]]:
        """Ask LLM to analyze paper and decide on figures."""

        # Build user context
        user_parts = [
            f"Research topic: {topic}",
            f"Hypothesis: {hypothesis}",
        ]

        if paper_draft:
            # Truncate to avoid token overflow
            draft_preview = paper_draft[:4000]
            user_parts.append(f"\nPaper draft (preview):\n{draft_preview}")

        if has_experiments and condition_summaries:
            conditions_preview = json.dumps(
                {k: v for k, v in list(condition_summaries.items())[:8]},
                indent=2,
                default=str,
            )
            user_parts.append(
                f"\nExperiment conditions:\n{conditions_preview}"
            )

        if has_experiments and experiment_results:
            metrics = list(experiment_results.keys())[:20]
            user_parts.append(f"\nAvailable metrics: {metrics}")

        user_parts.append(
            f"\nConstraints: Generate between {self._min_figures} "
            f"and {self._max_figures} figures total."
        )

        user_prompt = "\n".join(user_parts)

        raw = self._chat(
            _DECISION_SYSTEM_PROMPT,
            user_prompt,
            max_tokens=2048,
            temperature=0.3,
        )

        # Strip reasoning model thinking tags before JSON parsing
        raw = strip_thinking_tags(raw)

        # Parse JSON response
        return self._parse_decisions(raw)

    def _parse_decisions(self, raw: str) -> list[dict[str, Any]]:
        """Parse LLM response into decision list."""
        import re

        # Strip markdown fences
        m = re.search(r"```(?:json)?\s*\n(.*?)```", raw, re.DOTALL)
        text = m.group(1).strip() if m else raw.strip()

        # Find JSON array
        start = text.find("[")
        end = text.rfind("]")
        if start == -1 or end == -1:
            raise ValueError("No JSON array found in LLM response")

        decisions_raw = json.loads(text[start : end + 1])

        # Validate and normalize
        decisions = []
        for d in decisions_raw:
            if not isinstance(d, dict):
                continue
            decision = {
                "section": str(d.get("section", "Results")),
                "figure_type": str(d.get("figure_type", "bar_comparison")),
                "backend": str(d.get("backend", "code")),
                "description": str(d.get("description", "")),
                "priority": _safe_priority(d.get("priority", 2)),
            }
            # Validate backend
            if decision["backend"] not in ("code", "image"):
                # Auto-assign based on figure_type
                decision["backend"] = self._infer_backend(
                    decision["figure_type"]
                )
            decisions.append(decision)

        return decisions

    # ------------------------------------------------------------------
    # Heuristic fallback
    # ------------------------------------------------------------------

    def _heuristic_decide(
        self,
        *,
        topic: str,
        has_experiments: bool,
        condition_summaries: dict[str, Any],
    ) -> list[dict[str, Any]]:
        """Generate figure decisions without LLM (rule-based fallback)."""
        decisions: list[dict[str, Any]] = []

        # Always suggest an architecture/method diagram
        decisions.append({
            "section": "Method",
            "figure_type": "architecture_diagram",
            "backend": "image",
            "description": (
                f"Architecture overview diagram for the proposed method "
                f"in the paper about: {topic[:100]}"
            ),
            "priority": 1,
        })

        if has_experiments:
            # Main results comparison
            n_conditions = len(condition_summaries)
            decisions.append({
                "section": "Results",
                "figure_type": "bar_comparison",
                "backend": "code",
                "description": (
                    f"Bar chart comparing main metric across "
                    f"{n_conditions} experimental conditions"
                ),
                "priority": 1,
            })

            # Training/convergence curve
            decisions.append({
                "section": "Results",
                "figure_type": "training_curve",
                "backend": "code",
                "description": "Training convergence curves with loss/metric over epochs",
                "priority": 2,
            })

            # Ablation study
            if n_conditions >= 4:
                decisions.append({
                    "section": "Results",
                    "figure_type": "bar_comparison",
                    "backend": "code",
                    "description": "Ablation study showing contribution of each component",
                    "priority": 2,
                })

        # Pipeline/method flowchart
        decisions.append({
            "section": "Method",
            "figure_type": "pipeline_overview",
            "backend": "image",
            "description": (
                f"Step-by-step pipeline flowchart showing the method's "
                f"workflow for: {topic[:100]}"
            ),
            "priority": 2,
        })

        return decisions[:self._max_figures]

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _infer_backend(figure_type: str) -> str:
        """Infer generation backend from figure type."""
        code_types = {
            "bar_comparison", "line_chart", "heatmap", "confusion_matrix",
            "training_curve", "ablation_chart", "scatter_plot", "line_multi",
            "grouped_bar", "loss_curve",
        }
        if figure_type in code_types:
            return "code"
        return "image"

    def _enforce_bounds(
        self,
        decisions: list[dict[str, Any]],
        has_experiments: bool,
    ) -> list[dict[str, Any]]:
        """Enforce min/max figure counts and required categories."""
        # Sort by priority (1 = highest)
        decisions.sort(key=lambda d: d.get("priority", 2))

        # Ensure at least one architecture figure
        has_image = any(d["backend"] == "image" for d in decisions)
        if not has_image:
            decisions.insert(0, {
                "section": "Method",
                "figure_type": "architecture_diagram",
                "backend": "image",
                "description": "Model architecture overview",
                "priority": 1,
            })

        # Ensure at least one data figure if experiments exist
        if has_experiments:
            has_code = any(d["backend"] == "code" for d in decisions)
            if not has_code:
                decisions.append({
                    "section": "Results",
                    "figure_type": "bar_comparison",
                    "backend": "code",
                    "description": "Main results comparison",
                    "priority": 1,
                })

        # Enforce bounds
        if len(decisions) < self._min_figures:
            # Pad with lower-priority suggestions
            while len(decisions) < self._min_figures:
                decisions.append({
                    "section": "Discussion",
                    "figure_type": "concept_illustration",
                    "backend": "image",
                    "description": "Conceptual illustration of key findings",
                    "priority": 3,
                })

        return decisions[:self._max_figures]
