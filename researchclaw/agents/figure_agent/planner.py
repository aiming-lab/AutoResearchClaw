"""Planner Agent — analyzes experiment results and determines chart plan.

Examines the experiment results data structure, research topic, and paper
idea to decide:
- How many figures to generate
- What type each figure should be (bar, line, heatmap, etc.)
- What data each figure should display
- Caption specifications for each figure
- Layout (single / subplot / multi-panel)
"""

from __future__ import annotations

import json
import logging
from typing import Any

from researchclaw.agents.base import BaseAgent, AgentStepResult

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Chart type decision matrix — maps experiment characteristics to chart types
# ---------------------------------------------------------------------------

_CHART_TYPE_MATRIX: dict[str, list[dict[str, str]]] = {
    "classification": [
        {"type": "bar_comparison", "purpose": "accuracy comparison across methods"},
        {"type": "confusion_matrix", "purpose": "per-class prediction analysis"},
        {"type": "training_curve", "purpose": "convergence behavior"},
    ],
    "generation": [
        {"type": "line_multi", "purpose": "FID/IS curves over training"},
        {"type": "bar_comparison", "purpose": "generation quality metrics comparison"},
    ],
    "reinforcement_learning": [
        {"type": "training_curve", "purpose": "reward curve with mean±std shading"},
        {"type": "bar_comparison", "purpose": "final performance comparison"},
    ],
    "knowledge_distillation": [
        {"type": "bar_comparison", "purpose": "teacher-student accuracy comparison"},
        {"type": "line_multi", "purpose": "knowledge transfer efficiency curve"},
        {"type": "heatmap", "purpose": "feature alignment heatmap"},
    ],
    "nlp": [
        {"type": "bar_comparison", "purpose": "BLEU/ROUGE metric comparison"},
        {"type": "heatmap", "purpose": "attention heatmap"},
    ],
    "graph_neural_networks": [
        {"type": "bar_comparison", "purpose": "node classification accuracy"},
        {"type": "training_curve", "purpose": "convergence on graph tasks"},
    ],
    "meta_learning": [
        {"type": "line_multi", "purpose": "few-shot accuracy vs number of shots"},
        {"type": "bar_comparison", "purpose": "cross-task performance comparison"},
    ],
    "continual_learning": [
        {"type": "line_multi", "purpose": "forgetting rate curve across tasks"},
        {"type": "heatmap", "purpose": "task accuracy matrix"},
    ],
    "optimization": [
        {"type": "training_curve", "purpose": "convergence speed comparison"},
        {"type": "line_multi", "purpose": "loss landscape analysis"},
    ],
    "default": [
        {"type": "bar_comparison", "purpose": "main results comparison across methods"},
        {"type": "training_curve", "purpose": "training convergence"},
    ],
}

# Keywords for domain detection
_DOMAIN_KEYWORDS: dict[str, list[str]] = {
    "classification": ["classif", "accuracy", "cifar", "imagenet", "image recognition"],
    "generation": ["generat", "gan", "diffusion", "vae", "fid", "inception score"],
    "reinforcement_learning": ["reinforcement", "reward", "policy", "gymnasium", "mujoco", "atari"],
    "knowledge_distillation": ["distill", "teacher", "student", "knowledge transfer"],
    "nlp": ["bleu", "rouge", "language model", "translation", "summariz"],
    "graph_neural_networks": ["graph", "node classif", "gnn", "gcn", "message passing"],
    "meta_learning": ["meta-learn", "few-shot", "maml", "prototyp"],
    "continual_learning": ["continual", "lifelong", "catastrophic forgetting", "incremental"],
    "optimization": ["optim", "convergence", "learning rate", "sgd", "adam"],
}


class PlannerAgent(BaseAgent):
    """Analyzes experiment data and generates a figure plan."""

    name = "figure_planner"

    def __init__(
        self,
        llm: Any,
        *,
        min_figures: int = 3,
        max_figures: int = 8,
    ) -> None:
        super().__init__(llm)
        self._min_figures = min_figures
        self._max_figures = max_figures

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def execute(self, context: dict[str, Any]) -> AgentStepResult:
        """Generate a figure plan from experiment results.

        Context keys:
            experiment_results (dict): Parsed results.json / experiment_summary
            topic (str): Research topic
            hypothesis (str): Research hypothesis
            conditions (list[str]): Experiment condition names
            metric_key (str): Primary metric name
            metrics_summary (dict): Per-metric aggregated statistics
            condition_summaries (dict): Per-condition aggregated statistics
        """
        try:
            results = context.get("experiment_results", {})
            topic = context.get("topic", "")
            metric_key = context.get("metric_key", "primary_metric")
            conditions = context.get("conditions", [])
            metrics_summary = context.get("metrics_summary", {})
            condition_summaries = context.get("condition_summaries", {})

            # Step 1: Detect research domain
            domain = self._detect_domain(topic)
            self.logger.info("Detected research domain: %s", domain)

            # Step 2: Analyze available data
            data_analysis = self._analyze_data(
                results, conditions, metrics_summary, condition_summaries, metric_key
            )

            # Step 3: Generate figure plan via LLM
            figure_plan = self._generate_plan(
                topic=topic,
                domain=domain,
                data_analysis=data_analysis,
                metric_key=metric_key,
                conditions=conditions,
            )

            return self._make_result(True, data={
                "figures": figure_plan,
                "domain": domain,
                "data_analysis": data_analysis,
            })
        except Exception as exc:
            self.logger.error("Planner failed: %s", exc)
            return self._make_result(False, error=str(exc))

    # ------------------------------------------------------------------
    # Domain detection
    # ------------------------------------------------------------------

    def _detect_domain(self, topic: str) -> str:
        """Detect research domain from topic string."""
        topic_lower = topic.lower()
        scores: dict[str, int] = {}
        for domain, keywords in _DOMAIN_KEYWORDS.items():
            score = sum(1 for kw in keywords if kw in topic_lower)
            if score > 0:
                scores[domain] = score
        if scores:
            return max(scores, key=scores.get)  # type: ignore[arg-type]
        return "default"

    # ------------------------------------------------------------------
    # Data analysis
    # ------------------------------------------------------------------

    def _analyze_data(
        self,
        results: dict[str, Any],
        conditions: list[str],
        metrics_summary: dict[str, Any],
        condition_summaries: dict[str, Any],
        metric_key: str,
    ) -> dict[str, Any]:
        """Analyze available experiment data to determine chart potential."""
        analysis: dict[str, Any] = {
            "num_conditions": len(conditions),
            "conditions": conditions[:10],
            "num_metrics": len(metrics_summary),
            "metric_names": list(metrics_summary.keys())[:15],
            "has_training_history": False,
            "has_per_condition_data": bool(condition_summaries),
            "has_ablation": False,
            "has_multiple_seeds": False,
            "primary_metric": metric_key,
        }

        # Check for training history data
        for key in results:
            if any(t in str(key).lower() for t in ["history", "curve", "epoch", "step"]):
                analysis["has_training_history"] = True
                break

        # Check for ablation conditions
        for cond in conditions:
            cond_lower = cond.lower()
            if any(t in cond_lower for t in ["ablat", "without", "no_", "reduced", "remove"]):
                analysis["has_ablation"] = True
                break

        # Check for multi-seed data
        for cond_data in condition_summaries.values():
            if isinstance(cond_data, dict):
                n_seeds = cond_data.get("n_seeds", 0)
                if n_seeds and int(n_seeds) > 1:
                    analysis["has_multiple_seeds"] = True
                    break

        # Extract key metric values per condition
        condition_values: dict[str, float] = {}
        for cond, cdata in condition_summaries.items():
            if isinstance(cdata, dict):
                metrics = cdata.get("metrics", {})
                val = metrics.get(f"{metric_key}_mean") or metrics.get(metric_key)
                if val is not None:
                    try:
                        condition_values[cond] = float(val)
                    except (ValueError, TypeError):
                        pass
        analysis["condition_values"] = condition_values

        return analysis

    # ------------------------------------------------------------------
    # Plan generation
    # ------------------------------------------------------------------

    def _generate_plan(
        self,
        *,
        topic: str,
        domain: str,
        data_analysis: dict[str, Any],
        metric_key: str,
        conditions: list[str],
    ) -> list[dict[str, Any]]:
        """Use LLM to generate a detailed figure plan."""
        # Get domain-specific chart suggestions
        domain_charts = _CHART_TYPE_MATRIX.get(domain, _CHART_TYPE_MATRIX["default"])

        system_prompt = (
            "You are an expert scientific visualization advisor. "
            "Given experiment data from an ML research paper, you plan which "
            "figures to include in the paper.\n\n"
            "RULES:\n"
            f"- Generate between {self._min_figures} and {self._max_figures} figures\n"
            "- Each figure must serve a distinct purpose\n"
            "- At minimum include: 1 main results comparison + 1 ablation/analysis figure\n"
            "- If training history data exists, include a training curve\n"
            "- Figures should tell a coherent story about the research contributions\n"
            "- Do NOT generate figures for data that doesn't exist\n"
            "- Caption should be precise and descriptive (not generic)\n\n"
            "Available chart types: bar_comparison, grouped_bar, training_curve, "
            "loss_curve, heatmap, scatter_plot, violin_box, ablation_grouped, "
            "line_multi, radar_chart\n\n"
            "Return a JSON object with key 'figures' containing a list of figure "
            "specifications. Each figure spec must have:\n"
            "- figure_id: string (e.g. 'fig_main_results')\n"
            "- chart_type: one of the available types\n"
            "- title: short title for the chart\n"
            "- caption: detailed caption text (1-2 sentences)\n"
            "- data_source: what data to plot (metric names, conditions)\n"
            "- x_label: x-axis label\n"
            "- y_label: y-axis label\n"
            "- width: 'single_column' or 'double_column'\n"
            "- priority: 1 (must-have) to 3 (nice-to-have)\n"
            "- section: which paper section ('method', 'results', 'analysis')\n"
        )

        user_prompt = (
            f"Research topic: {topic}\n"
            f"Domain: {domain}\n"
            f"Primary metric: {metric_key}\n"
            f"Number of conditions: {data_analysis['num_conditions']}\n"
            f"Conditions: {', '.join(data_analysis.get('conditions', []))}\n"
            f"Available metrics: {', '.join(data_analysis.get('metric_names', []))}\n"
            f"Has training history: {data_analysis.get('has_training_history', False)}\n"
            f"Has ablation conditions: {data_analysis.get('has_ablation', False)}\n"
            f"Has multiple seeds: {data_analysis.get('has_multiple_seeds', False)}\n"
            f"Condition values: {json.dumps(data_analysis.get('condition_values', {}))}\n\n"
            f"Suggested chart types for this domain:\n"
        )
        for chart in domain_charts:
            user_prompt += f"- {chart['type']}: {chart['purpose']}\n"
        user_prompt += "\nGenerate the figure plan JSON."

        result = self._chat_json(system_prompt, user_prompt, max_tokens=4096)

        figures = result.get("figures", [])
        if not figures:
            # Fallback: generate a basic plan from domain matrix
            self.logger.warning("LLM returned no figures, using domain-based fallback")
            figures = self._fallback_plan(domain, data_analysis, metric_key, conditions)

        # Ensure minimum figure count
        if len(figures) < self._min_figures:
            self.logger.info(
                "LLM returned %d figures (min %d), adding defaults",
                len(figures), self._min_figures,
            )
            figures = self._augment_plan(figures, data_analysis, metric_key, conditions)

        # Cap at max
        figures = figures[:self._max_figures]

        # BUG-36: LLM may return figures as list of strings instead of dicts
        figures = [f for f in figures if isinstance(f, dict)]

        # Assign IDs if missing
        for i, fig in enumerate(figures):
            if not fig.get("figure_id"):
                fig["figure_id"] = f"fig_{i + 1}"

        return figures

    # ------------------------------------------------------------------
    # Fallback plan (no LLM needed)
    # ------------------------------------------------------------------

    def _fallback_plan(
        self,
        domain: str,
        data_analysis: dict[str, Any],
        metric_key: str,
        conditions: list[str],
    ) -> list[dict[str, Any]]:
        """Generate a basic plan without LLM (used as fallback)."""
        figures: list[dict[str, Any]] = []

        # Always include a main results comparison
        if data_analysis["num_conditions"] >= 2:
            figures.append({
                "figure_id": "fig_main_results",
                "chart_type": "bar_comparison",
                "title": "Method Comparison",
                "caption": f"Comparison of {metric_key.replace('_', ' ')} across all evaluated methods. "
                           f"Error bars show 95% confidence intervals.",
                "data_source": {"type": "condition_comparison", "metric": metric_key},
                "x_label": "Method",
                "y_label": metric_key.replace("_", " ").title(),
                "width": "single_column",
                "priority": 1,
                "section": "results",
            })

        # Ablation grouped bar if ablation exists
        if data_analysis.get("has_ablation"):
            figures.append({
                "figure_id": "fig_ablation",
                "chart_type": "ablation_grouped",
                "title": "Ablation Study",
                "caption": "Ablation study showing the contribution of each component. "
                           "Removing each component independently reveals its importance.",
                "data_source": {"type": "ablation_comparison", "metric": metric_key},
                "x_label": "Variant",
                "y_label": metric_key.replace("_", " ").title(),
                "width": "single_column",
                "priority": 1,
                "section": "results",
            })

        # Training curve if history exists
        if data_analysis.get("has_training_history"):
            figures.append({
                "figure_id": "fig_training_curve",
                "chart_type": "training_curve",
                "title": "Training Convergence",
                "caption": "Training loss curves for all methods. "
                           "Shaded regions indicate standard deviation across seeds.",
                "data_source": {"type": "training_history"},
                "x_label": "Epoch",
                "y_label": "Loss",
                "width": "single_column",
                "priority": 2,
                "section": "results",
            })

        # Multi-metric comparison if multiple metrics
        if data_analysis["num_metrics"] > 2:
            metrics_to_show = [
                m for m in data_analysis.get("metric_names", [])
                if m != metric_key and not any(
                    t in m.lower() for t in ["time", "elapsed", "seed", "runtime"]
                )
            ][:5]
            if metrics_to_show:
                figures.append({
                    "figure_id": "fig_multi_metric",
                    "chart_type": "grouped_bar",
                    "title": "Multi-Metric Comparison",
                    "caption": "Performance comparison across multiple evaluation metrics.",
                    "data_source": {"type": "multi_metric", "metrics": metrics_to_show},
                    "x_label": "Method",
                    "y_label": "Score",
                    "width": "double_column",
                    "priority": 2,
                    "section": "analysis",
                })

        return figures

    def _augment_plan(
        self,
        existing: list[dict[str, Any]],
        data_analysis: dict[str, Any],
        metric_key: str,
        conditions: list[str],
    ) -> list[dict[str, Any]]:
        """Add default figures to meet minimum count."""
        # BUG-37: chart_type may be non-hashable (list) — force str
        existing_types = {
            f.get("chart_type") for f in existing
            if isinstance(f.get("chart_type"), str)
        }
        augmented = list(existing)

        # Add main comparison if missing
        if "bar_comparison" not in existing_types and data_analysis["num_conditions"] >= 2:
            augmented.append({
                "figure_id": "fig_main_results",
                "chart_type": "bar_comparison",
                "title": "Method Comparison",
                "caption": f"Comparison of {metric_key.replace('_', ' ')} across all methods.",
                "data_source": {"type": "condition_comparison", "metric": metric_key},
                "x_label": "Method",
                "y_label": metric_key.replace("_", " ").title(),
                "width": "single_column",
                "priority": 1,
                "section": "results",
            })

        # Add ablation if applicable and missing
        if (
            "ablation_grouped" not in existing_types
            and data_analysis.get("has_ablation")
        ):
            augmented.append({
                "figure_id": "fig_ablation",
                "chart_type": "ablation_grouped",
                "title": "Ablation Study",
                "caption": "Ablation analysis showing component contributions.",
                "data_source": {"type": "ablation_comparison", "metric": metric_key},
                "x_label": "Variant",
                "y_label": metric_key.replace("_", " ").title(),
                "width": "single_column",
                "priority": 1,
                "section": "results",
            })

        return augmented
