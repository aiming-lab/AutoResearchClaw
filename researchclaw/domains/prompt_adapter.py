"""Domain-aware prompt adaptation layer.

Instead of rewriting ``prompts.py`` (2395+ lines of battle-tested code),
this module wraps existing prompt blocks with domain-specific overrides
via the **adapter pattern**.

Usage::

    adapter = get_adapter(domain_profile)
    blocks = adapter.get_code_generation_blocks(context)
    # blocks dict can be injected into the existing prompt system
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

from researchclaw.domains.detector import DomainProfile, is_ml_domain

logger = logging.getLogger(__name__)


@dataclass
class PromptBlocks:
    """Collection of prompt blocks for a specific pipeline stage.

    Each field is a string block that gets injected into the prompt
    template. Empty strings mean "use the default from prompts.py".
    """

    compute_budget: str = ""
    dataset_guidance: str = ""
    hp_reporting: str = ""
    code_generation_hints: str = ""
    result_analysis_hints: str = ""
    experiment_design_context: str = ""
    statistical_test_guidance: str = ""
    output_format_guidance: str = ""


class PromptAdapter(ABC):
    """Base class for domain-specific prompt adapters.

    Subclasses override methods to provide domain-specific prompt blocks.
    The ML adapter returns empty strings for everything (meaning: use the
    existing hardcoded behavior in prompts.py unchanged).
    """

    def __init__(self, domain: DomainProfile) -> None:
        self.domain = domain

    @abstractmethod
    def get_code_generation_blocks(self, context: dict[str, Any]) -> PromptBlocks:
        """Return prompt blocks for the code generation stage."""

    @abstractmethod
    def get_experiment_design_blocks(self, context: dict[str, Any]) -> PromptBlocks:
        """Return prompt blocks for the experiment design stage."""

    @abstractmethod
    def get_result_analysis_blocks(self, context: dict[str, Any]) -> PromptBlocks:
        """Return prompt blocks for the result analysis stage."""

    def get_blueprint_context(self) -> str:
        """Extra context injected into the blueprint generation prompt.

        Includes domain-specific file structure guidance, library hints, etc.
        """
        parts: list[str] = []

        if self.domain.typical_file_structure:
            parts.append("## Recommended File Structure")
            for fname, desc in self.domain.typical_file_structure.items():
                parts.append(f"- `{fname}`: {desc}")

        if self.domain.core_libraries:
            parts.append(f"\n## Core Libraries: {', '.join(self.domain.core_libraries)}")

        if self.domain.code_generation_hints:
            parts.append(f"\n## Domain-Specific Hints\n{self.domain.code_generation_hints}")

        return "\n".join(parts)

    def get_condition_terminology(self) -> dict[str, str]:
        """Return the domain's terminology mapping."""
        return self.domain.condition_terminology


# ---------------------------------------------------------------------------
# ML Adapter — wraps ALL current behavior unchanged
# ---------------------------------------------------------------------------


class MLPromptAdapter(PromptAdapter):
    """ML adapter: returns empty blocks so the existing prompts.py behavior
    is used verbatim. This is the zero-regression guarantee.
    """

    def get_code_generation_blocks(self, context: dict[str, Any]) -> PromptBlocks:
        # Empty = use existing hardcoded ML blocks in prompts.py
        return PromptBlocks()

    def get_experiment_design_blocks(self, context: dict[str, Any]) -> PromptBlocks:
        return PromptBlocks()

    def get_result_analysis_blocks(self, context: dict[str, Any]) -> PromptBlocks:
        return PromptBlocks()


# ---------------------------------------------------------------------------
# Generic Adapter — LLM-knowledge-only fallback for unknown domains
# ---------------------------------------------------------------------------


class GenericPromptAdapter(PromptAdapter):
    """Generic adapter for domains without a specialized adapter.

    Uses the DomainProfile's guidance fields (loaded from YAML) to
    construct prompt blocks. Falls back to sensible generic guidance.
    """

    def get_code_generation_blocks(self, context: dict[str, Any]) -> PromptBlocks:
        domain = self.domain
        paradigm = domain.experiment_paradigm
        libs = ", ".join(domain.core_libraries) if domain.core_libraries else "numpy, scipy"

        code_hints = domain.code_generation_hints or self._default_code_hints(paradigm, libs)
        dataset_guidance = domain.dataset_guidance or self._default_dataset_guidance(paradigm)
        hp_guidance = domain.hp_reporting_guidance or self._default_hp_guidance()

        return PromptBlocks(
            compute_budget=domain.compute_budget_guidance,
            dataset_guidance=dataset_guidance,
            hp_reporting=hp_guidance,
            code_generation_hints=code_hints,
            output_format_guidance=self._output_format_guidance(),
        )

    def get_experiment_design_blocks(self, context: dict[str, Any]) -> PromptBlocks:
        domain = self.domain
        terminology = domain.condition_terminology
        paradigm = domain.experiment_paradigm

        design_context = (
            f"This is a {domain.display_name} experiment.\n"
            f"Experiment paradigm: {paradigm}\n"
        )
        if terminology:
            design_context += "Terminology:\n"
            for key, term in terminology.items():
                design_context += f"  - {key}: {term}\n"

        if domain.standard_baselines:
            design_context += f"Standard baselines in this domain: {', '.join(domain.standard_baselines)}\n"

        stats = ", ".join(domain.statistical_tests) if domain.statistical_tests else "appropriate statistical tests"
        stat_guidance = f"Use {stats} for result significance testing."

        return PromptBlocks(
            experiment_design_context=design_context,
            statistical_test_guidance=stat_guidance,
        )

    def get_result_analysis_blocks(self, context: dict[str, Any]) -> PromptBlocks:
        domain = self.domain
        analysis_hints = domain.result_analysis_hints or ""

        if domain.statistical_tests:
            stat_guidance = (
                "Statistical tests to use for this domain:\n"
                + "\n".join(f"  - {t}" for t in domain.statistical_tests)
            )
        else:
            stat_guidance = ""

        return PromptBlocks(
            result_analysis_hints=analysis_hints,
            statistical_test_guidance=stat_guidance,
        )

    def _default_code_hints(self, paradigm: str, libs: str) -> str:
        hints = f"Core libraries for this domain: {libs}\n"
        if paradigm == "convergence":
            hints += (
                "This is a convergence study. The code should:\n"
                "1. Run the method at multiple refinement levels (e.g., grid sizes, timesteps)\n"
                "2. Compute error norms at each level\n"
                "3. Report results in a format suitable for convergence analysis\n"
                "4. Output results as JSON to results.json\n"
            )
        elif paradigm == "progressive_spec":
            hints += (
                "This uses progressive specification (common in economics):\n"
                "1. Start with a simple model (e.g., OLS)\n"
                "2. Progressively add complexity (controls, fixed effects, IV)\n"
                "3. Present results as a regression table\n"
                "4. Output results as JSON to results.json\n"
            )
        elif paradigm == "simulation":
            hints += (
                "This is a simulation study. The code should:\n"
                "1. Set up the physical/computational system\n"
                "2. Run the simulation\n"
                "3. Compute observables from simulation data\n"
                "4. Output results as JSON to results.json\n"
            )
        else:
            hints += (
                "Output all results as JSON to results.json with the structure:\n"
                '{"conditions": {"method_name": {"seed_X": {"metric": value}}}}\n'
            )
        return hints

    def _default_dataset_guidance(self, paradigm: str) -> str:
        if paradigm in ("convergence", "simulation"):
            return (
                "Data/input for this experiment should be generated programmatically.\n"
                "Define initial conditions, parameters, or test problems in code.\n"
                "Do NOT attempt to download external datasets."
            )
        return ""

    def _default_hp_guidance(self) -> str:
        return (
            "Report all experiment parameters in a dictionary printed to stdout:\n"
            "HYPERPARAMETERS: {'param1': value1, 'param2': value2, ...}"
        )

    def _output_format_guidance(self) -> str:
        domain = self.domain
        if "convergence" in domain.metric_types:
            return (
                "Output results as JSON to results.json with convergence data:\n"
                '{"convergence": {"method": [{"h": 0.1, "error": 0.05}, ...]}}'
            )
        if "table" in domain.metric_types:
            return (
                "Output results as JSON to results.json with table data:\n"
                '{"regression_table": {"spec_1": {"coeff": 0.15, "se": 0.03, ...}}}'
            )
        return (
            "Output results as JSON to results.json:\n"
            '{"conditions": {"method": {"seed_X": {"metric": value}}}}'
        )


# ---------------------------------------------------------------------------
# Adapter registry
# ---------------------------------------------------------------------------

# Maps domain_id prefixes to adapter classes.
# If a domain_id starts with "ml_", the ML adapter is used.
def _build_adapter_registry() -> dict[str, type[PromptAdapter]]:
    """Build the adapter registry with lazy imports for domain adapters."""
    registry: dict[str, type[PromptAdapter]] = {
        "ml_": MLPromptAdapter,
        "generic": GenericPromptAdapter,
    }
    try:
        from researchclaw.domains.adapters.physics import PhysicsPromptAdapter
        registry["physics_"] = PhysicsPromptAdapter
    except ImportError:
        pass
    try:
        from researchclaw.domains.adapters.economics import EconomicsPromptAdapter
        registry["economics_"] = EconomicsPromptAdapter
    except ImportError:
        pass
    try:
        from researchclaw.domains.adapters.biology import BiologyPromptAdapter
        registry["biology_"] = BiologyPromptAdapter
    except ImportError:
        pass
    try:
        from researchclaw.domains.adapters.chemistry import ChemistryPromptAdapter
        registry["chemistry_"] = ChemistryPromptAdapter
    except ImportError:
        pass
    try:
        from researchclaw.domains.adapters.security import SecurityPromptAdapter
        registry["security_"] = SecurityPromptAdapter
    except ImportError:
        pass
    try:
        from researchclaw.domains.adapters.math import MathPromptAdapter
        registry["mathematics_"] = MathPromptAdapter
    except ImportError:
        pass
    try:
        from researchclaw.domains.adapters.neuroscience import NeurosciencePromptAdapter
        registry["neuroscience_"] = NeurosciencePromptAdapter
    except ImportError:
        pass
    try:
        from researchclaw.domains.adapters.robotics import RoboticsPromptAdapter
        registry["robotics_"] = RoboticsPromptAdapter
    except ImportError:
        pass
    return registry


_ADAPTER_REGISTRY: dict[str, type[PromptAdapter]] = _build_adapter_registry()


def register_adapter(domain_prefix: str, adapter_cls: type[PromptAdapter]) -> None:
    """Register a custom adapter for a domain prefix."""
    _ADAPTER_REGISTRY[domain_prefix] = adapter_cls


def get_adapter(domain: DomainProfile) -> PromptAdapter:
    """Get the appropriate PromptAdapter for a given domain.

    Lookup order:
    1. Exact domain_id match
    2. Prefix match (e.g., "ml_" for all ML domains)
    3. Generic fallback
    """
    # Exact match
    if domain.domain_id in _ADAPTER_REGISTRY:
        return _ADAPTER_REGISTRY[domain.domain_id](domain)

    # Prefix match
    for prefix, adapter_cls in _ADAPTER_REGISTRY.items():
        if prefix.endswith("_") and domain.domain_id.startswith(prefix):
            return adapter_cls(domain)

    # ML domain check
    if is_ml_domain(domain):
        return MLPromptAdapter(domain)

    # Generic fallback
    return GenericPromptAdapter(domain)
