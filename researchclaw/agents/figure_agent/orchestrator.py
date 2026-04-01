"""FigureAgent Orchestrator — coordinates the figure generation sub-agents.

Flow:
  Decision Agent → analyzes paper → decides what figures are needed
    ├── code figures  → Planner → CodeGen → Renderer → Critic → retry
    └── image figures → Nano Banana (Gemini image generation)
  → Integrator (combines all figures into manifest)

Produces a ``FigurePlan`` consumed by paper draft and export stages.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from researchclaw.agents.base import AgentOrchestrator
from researchclaw.agents.figure_agent.codegen import CodeGenAgent
from researchclaw.agents.figure_agent.critic import CriticAgent
from researchclaw.agents.figure_agent.decision import FigureDecisionAgent
from researchclaw.agents.figure_agent.integrator import IntegratorAgent
from researchclaw.agents.figure_agent.nano_banana import NanoBananaAgent
from researchclaw.agents.figure_agent.planner import PlannerAgent
from researchclaw.agents.figure_agent.renderer import RendererAgent

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FigureAgentConfig:
    """Configuration for the FigureAgent system."""

    enabled: bool = True
    # Planner
    min_figures: int = 3
    max_figures: int = 8
    # Orchestrator
    max_iterations: int = 3   # max CodeGen→Renderer→Critic retry loops
    # Renderer security
    render_timeout_sec: int = 30
    use_docker: bool | None = None  # None = auto-detect
    docker_image: str = "researchclaw/experiment:latest"
    # Code generation
    output_format: str = "python"  # "python" or "latex"
    # Nano Banana (Gemini image generation)
    gemini_api_key: str = ""  # or set GEMINI_API_KEY env var
    gemini_model: str = "gemini-2.5-flash-image"
    nano_banana_enabled: bool = True  # enable/disable image generation
    # Critic
    strict_mode: bool = False  # if True, any issue = fail
    # Output
    dpi: int = 300


# ---------------------------------------------------------------------------
# Output data structure
# ---------------------------------------------------------------------------


@dataclass
class FigurePlan:
    """Final output from the FigureAgent system.

    Consumed by:
    - Paper draft stage (figure_descriptions for writing prompt)
    - Paper export stage (manifest for LaTeX figure embedding)
    - Charts directory (scripts + rendered images)
    """

    # Figure manifest (list of figure metadata dicts)
    manifest: list[dict[str, Any]] = field(default_factory=list)

    # Generated references
    markdown_refs: str = ""
    figure_descriptions: str = ""

    # Paths
    output_dir: str = ""
    manifest_path: str = ""

    # Stats
    figure_count: int = 0
    passed_count: int = 0
    total_llm_calls: int = 0
    total_tokens: int = 0
    elapsed_sec: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-safe dict."""
        return {
            "manifest": self.manifest,
            "markdown_refs": self.markdown_refs,
            "figure_descriptions": self.figure_descriptions,
            "output_dir": self.output_dir,
            "manifest_path": self.manifest_path,
            "figure_count": self.figure_count,
            "passed_count": self.passed_count,
            "total_llm_calls": self.total_llm_calls,
            "total_tokens": self.total_tokens,
            "elapsed_sec": self.elapsed_sec,
        }

    def get_chart_files(self) -> list[str]:
        """Return list of chart filenames from manifest."""
        return [
            Path(entry["file_path"]).name
            for entry in self.manifest
            if entry.get("file_path")
        ]


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


class FigureOrchestrator(AgentOrchestrator):
    """Coordinates Decision → (Code-to-Viz | Nano Banana) → Integrator."""

    def __init__(
        self,
        llm: Any,
        config: FigureAgentConfig | None = None,
        *,
        stage_dir: Path | None = None,
    ) -> None:
        cfg = config or FigureAgentConfig()
        super().__init__(llm, max_iterations=cfg.max_iterations)

        self._config = cfg
        self._stage_dir = stage_dir

        # Decision agent
        self._decision = FigureDecisionAgent(
            llm,
            min_figures=cfg.min_figures,
            max_figures=cfg.max_figures,
        )

        # Code-to-Viz sub-agents (for data-driven charts)
        self._planner = PlannerAgent(
            llm,
            min_figures=cfg.min_figures,
            max_figures=cfg.max_figures,
        )
        # BUG-60: Pass use_docker so CodeGen generates container-aware paths
        self._codegen = CodeGenAgent(
            llm, output_format=cfg.output_format,
            use_docker=bool(cfg.use_docker) if cfg.use_docker is not None else False,
        )
        self._renderer = RendererAgent(
            llm,
            timeout_sec=cfg.render_timeout_sec,
            use_docker=cfg.use_docker,
            docker_image=cfg.docker_image,
        )
        self._critic = CriticAgent(
            llm,
            strict_mode=cfg.strict_mode,
        )

        # Nano Banana agent (for conceptual/architectural images)
        self._nano_banana: NanoBananaAgent | None = None
        if cfg.nano_banana_enabled:
            self._nano_banana = NanoBananaAgent(
                llm,
                gemini_api_key=cfg.gemini_api_key or None,
                model=cfg.gemini_model,
            )

        self._integrator = IntegratorAgent(llm)

    def _save_artifact(self, name: str, data: Any) -> None:
        """Save intermediate artifact to stage directory."""
        if self._stage_dir is None:
            return
        self._stage_dir.mkdir(parents=True, exist_ok=True)
        path = self._stage_dir / name
        if isinstance(data, str):
            path.write_text(data, encoding="utf-8")
        else:
            path.write_text(
                json.dumps(data, indent=2, ensure_ascii=False, default=str),
                encoding="utf-8",
            )

    def orchestrate(self, context: dict[str, Any]) -> FigurePlan:
        """Run the full figure generation pipeline.

        Context keys:
            experiment_results (dict): Parsed results.json
            condition_summaries (dict): Per-condition aggregated stats
            metrics_summary (dict): Per-metric aggregated stats
            metric_key (str): Primary metric name
            topic (str): Research topic
            hypothesis (str): Research hypothesis
            paper_draft (str): Current paper draft (for decision agent)
            output_dir (str|Path): Directory for chart output
        """
        t0 = time.monotonic()
        topic = context.get("topic", "")
        output_dir = Path(context.get("output_dir", "charts"))
        output_dir.mkdir(parents=True, exist_ok=True)

        self.logger.info("FigureAgent starting for: %s", topic[:80])

        plan = FigurePlan(output_dir=str(output_dir))

        # ── Phase 0: Decision — what figures are needed? ──────────────
        self.logger.info("Phase 0: Deciding what figures are needed")
        decision_result = self._decision.execute({
            "topic": topic,
            "hypothesis": context.get("hypothesis", ""),
            "paper_draft": context.get("paper_draft", ""),
            "has_experiments": bool(context.get("experiment_results")),
            "experiment_results": context.get("experiment_results", {}),
            "condition_summaries": context.get("condition_summaries", {}),
        })
        self._accumulate(decision_result)
        self._save_artifact("figure_decisions.json", decision_result.data)

        code_figures = decision_result.data.get("code_figures", [])
        image_figures = decision_result.data.get("image_figures", [])

        self.logger.info(
            "Decision: %d code figures, %d image figures",
            len(code_figures), len(image_figures),
        )

        # Track all rendered figures (from both backends)
        all_rendered: list[dict[str, Any]] = []

        # ── Phase A: Code-to-Viz for data figures ─────────────────────
        if code_figures:
            rendered_code = self._run_code_pipeline(
                code_figures=code_figures,
                context=context,
                output_dir=output_dir,
            )
            all_rendered.extend(rendered_code)

        # ── Phase B: Nano Banana for image figures ────────────────────
        if image_figures and self._nano_banana is not None:
            rendered_images = self._run_nano_banana(
                image_figures=image_figures,
                context=context,
                output_dir=output_dir,
            )
            all_rendered.extend(rendered_images)
        elif image_figures:
            self.logger.warning(
                "Nano Banana disabled — skipping %d image figures",
                len(image_figures),
            )

        # ── Phase C: Integrate all figures ────────────────────────────
        self.logger.info(
            "Phase C: Integrating %d figures into paper", len(all_rendered)
        )
        integrate_result = self._integrator.execute({
            "rendered": all_rendered,
            "topic": topic,
            "output_dir": str(output_dir),
        })
        self._accumulate(integrate_result)

        # ── Finalize ─────────────────────────────────────────────────
        plan.manifest = integrate_result.data.get("manifest", [])
        plan.markdown_refs = integrate_result.data.get("markdown_refs", "")
        plan.figure_descriptions = integrate_result.data.get("figure_descriptions", "")
        plan.manifest_path = integrate_result.data.get("manifest_path", "")
        plan.figure_count = integrate_result.data.get("figure_count", 0)
        plan.passed_count = sum(
            1 for r in all_rendered if r.get("success")
        )
        plan.total_llm_calls = self.total_llm_calls
        plan.total_tokens = self.total_tokens
        plan.elapsed_sec = time.monotonic() - t0

        # Save final plan
        self._save_artifact("figure_plan_final.json", plan.to_dict())

        self.logger.info(
            "FigureAgent complete: %d figures (%d code + %d image), "
            "%d passed, %d LLM calls, %.1fs",
            plan.figure_count,
            len(code_figures),
            len(image_figures),
            plan.passed_count,
            plan.total_llm_calls,
            plan.elapsed_sec,
        )

        return plan

    # ------------------------------------------------------------------
    # Code-to-Viz pipeline (data-driven charts)
    # ------------------------------------------------------------------

    def _run_code_pipeline(
        self,
        code_figures: list[dict[str, Any]],
        context: dict[str, Any],
        output_dir: Path,
    ) -> list[dict[str, Any]]:
        """Run Planner → CodeGen → Renderer → Critic for data figures."""

        # Phase 1: Plan (uses experiment data)
        self.logger.info("Phase A1: Planning data figures")
        plan_result = self._planner.execute({
            "experiment_results": context.get("experiment_results", {}),
            "topic": context.get("topic", ""),
            "hypothesis": context.get("hypothesis", ""),
            "conditions": context.get("conditions", []),
            "metric_key": context.get("metric_key", "primary_metric"),
            "metrics_summary": context.get("metrics_summary", {}),
            "condition_summaries": context.get("condition_summaries", {}),
        })
        self._accumulate(plan_result)

        if not plan_result.success:
            self.logger.warning("Planning failed: %s", plan_result.error)
            return []

        figures = plan_result.data.get("figures", [])
        self._save_artifact("figure_plan_code.json", figures)
        self.logger.info("Planned %d data figures", len(figures))

        # Phase 2+3+4: CodeGen → Render → Critic (with retry)
        critic_feedback: list[dict[str, Any]] = []
        final_rendered: list[dict[str, Any]] = []

        for iteration in range(self.max_iterations):
            self.logger.info(
                "Phase A2: CodeGen (iteration %d/%d)",
                iteration + 1, self.max_iterations,
            )

            # CodeGen
            codegen_result = self._codegen.execute({
                "figures": figures,
                "experiment_results": context.get("experiment_results", {}),
                "condition_summaries": context.get("condition_summaries", {}),
                "metrics_summary": context.get("metrics_summary", {}),
                "metric_key": context.get("metric_key", "primary_metric"),
                "output_dir": str(output_dir),
                "critic_feedback": critic_feedback,
            })
            self._accumulate(codegen_result)

            if not codegen_result.success:
                self.logger.warning("CodeGen failed: %s", codegen_result.error)
                continue

            scripts = codegen_result.data.get("scripts", [])
            self._save_artifact(f"scripts_{iteration}.json", [
                {k: v for k, v in s.items() if k != "script"}
                for s in scripts
            ])

            # Render
            self.logger.info(
                "Phase A3: Rendering (iteration %d/%d)",
                iteration + 1, self.max_iterations,
            )
            render_result = self._renderer.execute({
                "scripts": scripts,
                "output_dir": str(output_dir),
            })
            self._accumulate(render_result)

            if not render_result.success:
                self.logger.warning("Rendering failed: %s", render_result.error)
                continue

            rendered = render_result.data.get("rendered", [])
            # Merge newly rendered figures with previously passed figures
            # (on retries, only failed figures are re-rendered)
            if iteration == 0:
                final_rendered = rendered
            else:
                # Replace entries for re-rendered figures, keep previously passed ones
                re_rendered_ids = {r.get("figure_id") for r in rendered}
                final_rendered = [
                    r for r in final_rendered if r.get("figure_id") not in re_rendered_ids
                ] + rendered

            # Critic
            self.logger.info(
                "Phase A4: Critic review (iteration %d/%d)",
                iteration + 1, self.max_iterations,
            )
            critic_result = self._critic.execute({
                "rendered": rendered,
                "scripts": scripts,
                "condition_summaries": context.get("condition_summaries", {}),
                "metrics_summary": context.get("metrics_summary", {}),
                "metric_key": context.get("metric_key", "primary_metric"),
            })
            self._accumulate(critic_result)

            reviews = critic_result.data.get("reviews", [])
            all_passed = critic_result.data.get("all_passed", False)
            self._save_artifact(f"reviews_{iteration}.json", reviews)

            if all_passed:
                self.logger.info(
                    "All data figures passed review on iteration %d",
                    iteration + 1,
                )
                break

            # Collect feedback for failed figures
            critic_feedback = [
                r for r in reviews if not r.get("passed")
            ]

            # Only retry figures that failed
            # BUG-37: figure_id may be non-hashable (list) — force str
            failed_ids = set()
            for r in critic_feedback:
                _fid = r.get("figure_id")
                if isinstance(_fid, str):
                    failed_ids.add(_fid)
                elif isinstance(_fid, list) and _fid:
                    failed_ids.add(str(_fid[0]))
            figures = [f for f in figures if f.get("figure_id") in failed_ids]

            self.logger.warning(
                "Critic: %d/%d figures need revision",
                len(failed_ids), len(rendered),
            )

        return final_rendered

    # ------------------------------------------------------------------
    # Nano Banana pipeline (conceptual/architectural images)
    # ------------------------------------------------------------------

    def _run_nano_banana(
        self,
        image_figures: list[dict[str, Any]],
        context: dict[str, Any],
        output_dir: Path,
    ) -> list[dict[str, Any]]:
        """Run Nano Banana for conceptual/architectural figures."""
        if self._nano_banana is None:
            return []

        self.logger.info(
            "Phase B: Generating %d image figures via Nano Banana",
            len(image_figures),
        )

        # Assign figure IDs
        for i, fig in enumerate(image_figures):
            if "figure_id" not in fig:
                fig["figure_id"] = (
                    f"{fig.get('figure_type', 'conceptual')}_{i + 1}"
                )

        nb_result = self._nano_banana.execute({
            "image_figures": image_figures,
            "topic": context.get("topic", ""),
            "output_dir": str(output_dir),
        })
        self._accumulate(nb_result)
        self._save_artifact("nano_banana_results.json", nb_result.data)

        generated = nb_result.data.get("generated", [])
        success_count = nb_result.data.get("count", 0)

        self.logger.info(
            "Nano Banana: %d/%d images generated successfully",
            success_count, len(image_figures),
        )

        return generated
