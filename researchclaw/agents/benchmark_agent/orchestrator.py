"""BenchmarkAgent Orchestrator — coordinates the four sub-agents.

Flow: Surveyor → Selector → Acquirer → Validator (→ retry if failed)

Produces a ``BenchmarkPlan`` consumed by experiment design and code
generation stages.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from researchclaw.agents.base import AgentOrchestrator
from researchclaw.agents.benchmark_agent.acquirer import AcquirerAgent
from researchclaw.agents.benchmark_agent.selector import SelectorAgent
from researchclaw.agents.benchmark_agent.surveyor import SurveyorAgent
from researchclaw.agents.benchmark_agent.validator import ValidatorAgent

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BenchmarkAgentConfig:
    """Configuration for the BenchmarkAgent system."""

    enabled: bool = True
    # Surveyor
    enable_hf_search: bool = True
    max_hf_results: int = 10
    enable_web_search: bool = False
    max_web_results: int = 5
    web_search_min_local: int = 3
    # Selector
    tier_limit: int = 2
    min_benchmarks: int = 1
    min_baselines: int = 2
    prefer_cached: bool = True
    # Orchestrator
    max_iterations: int = 2  # max Acquirer→Validator retry loops


# ---------------------------------------------------------------------------
# Output data structure
# ---------------------------------------------------------------------------


@dataclass
class BenchmarkPlan:
    """Final output from the BenchmarkAgent system.

    Consumed by:
    - Experiment design stage (selected benchmarks/baselines for plan)
    - Code generation stage (data_loader_code, baseline_code)
    - Docker sandbox (setup_code, requirements)
    """

    # Selected items
    selected_benchmarks: list[dict[str, Any]] = field(default_factory=list)
    selected_baselines: list[dict[str, Any]] = field(default_factory=list)
    matched_domains: list[str] = field(default_factory=list)

    # Generated code
    data_loader_code: str = ""
    baseline_code: str = ""
    setup_code: str = ""
    requirements: str = ""

    # Metadata
    rationale: str = ""
    experiment_notes: str = ""
    validation_passed: bool = False
    validation_warnings: list[str] = field(default_factory=list)

    # Stats
    total_llm_calls: int = 0
    total_tokens: int = 0
    elapsed_sec: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-safe dict."""
        return {
            "selected_benchmarks": self.selected_benchmarks,
            "selected_baselines": self.selected_baselines,
            "matched_domains": self.matched_domains,
            "data_loader_code": self.data_loader_code,
            "baseline_code": self.baseline_code,
            "setup_code": self.setup_code,
            "requirements": self.requirements,
            "rationale": self.rationale,
            "experiment_notes": self.experiment_notes,
            "validation_passed": self.validation_passed,
            "validation_warnings": self.validation_warnings,
            "total_llm_calls": self.total_llm_calls,
            "total_tokens": self.total_tokens,
            "elapsed_sec": self.elapsed_sec,
        }

    def to_prompt_block(self) -> str:
        """Format as a prompt block for injection into code generation."""
        parts = []

        # Benchmark summary
        if self.selected_benchmarks:
            parts.append("## Selected Benchmarks")
            for b in self.selected_benchmarks:
                role = b.get("role", "secondary")
                metrics = b.get("metrics", [])
                parts.append(
                    f"- **{b.get('name', 'Unknown')}** ({role}) — "
                    f"metrics: {', '.join(str(m) for m in metrics)}"
                )
                if b.get("api"):
                    parts.append(f"  API: `{b['api']}`")
                if b.get("note"):
                    parts.append(f"  Note: {b['note']}")

        # Baseline summary
        if self.selected_baselines:
            parts.append("\n## Selected Baselines")
            for bl in self.selected_baselines:
                parts.append(
                    f"- **{bl.get('name', 'Unknown')}**: {bl.get('paper', 'N/A')}"
                )
                if bl.get("source"):
                    parts.append(f"  Code: `{bl['source']}`")

        # Data loading code
        if self.data_loader_code:
            parts.append("\n## Data Loading Code (READY TO USE)")
            parts.append("```python")
            parts.append(self.data_loader_code)
            parts.append("```")

        # Baseline code
        if self.baseline_code:
            parts.append("\n## Baseline Methods Code (READY TO USE)")
            parts.append("```python")
            parts.append(self.baseline_code)
            parts.append("```")

        # Experiment notes
        if self.experiment_notes:
            parts.append(f"\n## Experiment Notes\n{self.experiment_notes}")

        return "\n".join(parts)


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


class BenchmarkOrchestrator(AgentOrchestrator):
    """Coordinates Surveyor → Selector → Acquirer → Validator pipeline."""

    def __init__(
        self,
        llm: Any,
        config: BenchmarkAgentConfig | None = None,
        *,
        gpu_memory_mb: int = 49000,
        time_budget_sec: int = 300,
        network_policy: str = "setup_only",
        stage_dir: Path | None = None,
    ) -> None:
        cfg = config or BenchmarkAgentConfig()
        super().__init__(llm, max_iterations=cfg.max_iterations)

        self._config = cfg
        self._stage_dir = stage_dir

        # Initialize sub-agents
        self._surveyor = SurveyorAgent(
            llm,
            enable_hf_search=cfg.enable_hf_search,
            max_hf_results=cfg.max_hf_results,
        )
        self._selector = SelectorAgent(
            llm,
            gpu_memory_mb=gpu_memory_mb,
            time_budget_sec=time_budget_sec,
            network_policy=network_policy,
            tier_limit=cfg.tier_limit,
            min_benchmarks=cfg.min_benchmarks,
            min_baselines=cfg.min_baselines,
            prefer_cached=cfg.prefer_cached,
        )
        self._acquirer = AcquirerAgent(llm)
        self._validator = ValidatorAgent(llm)

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

    def orchestrate(self, context: dict[str, Any]) -> BenchmarkPlan:
        """Run the full benchmark selection pipeline.

        Context keys:
            topic (str): Research topic/title
            hypothesis (str): Research hypothesis
            experiment_plan (str): Experiment plan text
        """
        t0 = time.monotonic()
        topic = context.get("topic", "")
        hypothesis = context.get("hypothesis", "")

        self.logger.info("BenchmarkAgent starting for: %s", topic[:80])

        plan = BenchmarkPlan()

        # ── Phase 1: Survey ───────────────────────────────────────
        self.logger.info("Phase 1: Surveying benchmarks")
        survey_result = self._surveyor.execute({
            "topic": topic,
            "hypothesis": hypothesis,
            "experiment_plan": context.get("experiment_plan", ""),
        })
        self._accumulate(survey_result)

        if not survey_result.success:
            self.logger.warning("Survey failed: %s", survey_result.error)
            plan.elapsed_sec = time.monotonic() - t0
            plan.total_llm_calls = self.total_llm_calls
            plan.total_tokens = self.total_tokens
            return plan

        survey = survey_result.data
        plan.matched_domains = survey.get("matched_domains", [])
        self._save_artifact("survey_results.json", survey)

        # ── Phase 2: Select ───────────────────────────────────────
        self.logger.info("Phase 2: Selecting benchmarks and baselines")
        select_result = self._selector.execute({
            "topic": topic,
            "survey": survey,
        })
        self._accumulate(select_result)

        if not select_result.success:
            self.logger.warning("Selection failed: %s", select_result.error)
            plan.elapsed_sec = time.monotonic() - t0
            plan.total_llm_calls = self.total_llm_calls
            plan.total_tokens = self.total_tokens
            return plan

        selection = select_result.data
        plan.selected_benchmarks = selection.get("selected_benchmarks", [])
        plan.selected_baselines = selection.get("selected_baselines", [])
        plan.rationale = selection.get("rationale", "")
        plan.experiment_notes = selection.get("experiment_notes", "")
        self._save_artifact("selection_results.json", selection)

        # ── Phase 3+4: Acquire + Validate (with retry) ───────────
        for iteration in range(self.max_iterations):
            self.logger.info(
                "Phase 3: Acquiring code (iteration %d/%d)",
                iteration + 1, self.max_iterations,
            )

            # Acquire
            acq_result = self._acquirer.execute({
                "topic": topic,
                "selection": selection,
            })
            self._accumulate(acq_result)

            if not acq_result.success:
                self.logger.warning("Acquisition failed: %s", acq_result.error)
                continue

            acquisition = acq_result.data
            self._save_artifact(
                f"acquisition_{iteration}.json",
                {k: v for k, v in acquisition.items()
                 if k not in ("data_loader_code", "baseline_code", "setup_code")},
            )

            # Validate
            self.logger.info("Phase 4: Validating code (iteration %d/%d)",
                             iteration + 1, self.max_iterations)
            val_result = self._validator.execute({
                "acquisition": acquisition,
            })
            self._accumulate(val_result)

            validation = val_result.data
            self._save_artifact(f"validation_{iteration}.json", validation)

            # Store results
            plan.data_loader_code = acquisition.get("data_loader_code", "")
            plan.baseline_code = acquisition.get("baseline_code", "")
            plan.setup_code = acquisition.get("setup_code", "")
            plan.requirements = acquisition.get("requirements", "")
            plan.validation_passed = validation.get("passed", False)
            plan.validation_warnings = validation.get("warnings", [])

            if plan.validation_passed:
                self.logger.info("Validation passed on iteration %d", iteration + 1)
                break

            self.logger.warning(
                "Validation failed (iteration %d): %s",
                iteration + 1,
                validation.get("errors", []),
            )

        # ── Finalize ──────────────────────────────────────────────
        plan.total_llm_calls = self.total_llm_calls
        plan.total_tokens = self.total_tokens
        plan.elapsed_sec = time.monotonic() - t0

        # Save final plan
        self._save_artifact("benchmark_plan.json", plan.to_dict())

        self.logger.info(
            "BenchmarkAgent complete: %d benchmarks, %d baselines, "
            "validation=%s, %d LLM calls, %.1fs",
            len(plan.selected_benchmarks),
            len(plan.selected_baselines),
            "PASS" if plan.validation_passed else "FAIL",
            plan.total_llm_calls,
            plan.elapsed_sec,
        )

        return plan
