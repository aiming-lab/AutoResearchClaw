"""Physics domain prompt adapter.

Provides domain-specific prompt blocks for computational physics
experiments (simulations, PDE solvers, convergence studies).
"""

from __future__ import annotations

from typing import Any

from researchclaw.domains.prompt_adapter import PromptAdapter, PromptBlocks


class PhysicsPromptAdapter(PromptAdapter):
    """Adapter for physics domains (simulation, PDE, quantum)."""

    def get_code_generation_blocks(self, context: dict[str, Any]) -> PromptBlocks:
        domain = self.domain
        paradigm = domain.experiment_paradigm
        libs = ", ".join(domain.core_libraries) if domain.core_libraries else "numpy, scipy"

        code_hints = domain.code_generation_hints or self._default_code_hints(paradigm)

        return PromptBlocks(
            compute_budget=domain.compute_budget_guidance or self._default_compute_budget(),
            dataset_guidance=domain.dataset_guidance or self._default_dataset_guidance(),
            hp_reporting=domain.hp_reporting_guidance or self._default_hp_reporting(),
            code_generation_hints=code_hints,
            output_format_guidance=self._output_format(paradigm),
        )

    def get_experiment_design_blocks(self, context: dict[str, Any]) -> PromptBlocks:
        domain = self.domain

        design_context = (
            f"This is a **{domain.display_name}** experiment.\n"
            f"Paradigm: {domain.experiment_paradigm}\n\n"
            "Key principles for physics experiments:\n"
            "1. Conservation laws must be respected (energy, momentum, etc.)\n"
            "2. Use appropriate units (reduced units for MD, SI otherwise)\n"
            "3. Validate against known analytical solutions when possible\n"
            "4. For convergence: vary grid size/timestep systematically\n"
        )

        if domain.standard_baselines:
            design_context += f"\nStandard reference methods: {', '.join(domain.standard_baselines)}\n"

        stats = ", ".join(domain.statistical_tests) if domain.statistical_tests else "convergence order analysis"

        return PromptBlocks(
            experiment_design_context=design_context,
            statistical_test_guidance=f"Use {stats} for result analysis.",
        )

    def get_result_analysis_blocks(self, context: dict[str, Any]) -> PromptBlocks:
        return PromptBlocks(
            result_analysis_hints=self.domain.result_analysis_hints or (
                "Physics result analysis:\n"
                "- Convergence: fit log(error) vs log(h) for order\n"
                "- Conservation: track energy/momentum drift\n"
                "- Accuracy: compare with analytical solutions\n"
                "- Use log-log plots for convergence studies"
            ),
            statistical_test_guidance="Use convergence order fitting and relative error analysis.",
        )

    def _default_code_hints(self, paradigm: str) -> str:
        if paradigm == "convergence":
            return (
                "This is a convergence study:\n"
                "1. Implement the numerical method(s)\n"
                "2. Run at 5+ refinement levels (e.g., h, h/2, h/4, h/8, h/16)\n"
                "3. Compute error norms at each level (L2, L-inf)\n"
                "4. Output results.json with convergence data\n"
                "5. Expected format:\n"
                '   {"convergence": {"method_name": [{"h": 0.1, "l2_error": 0.05}]}}\n'
            )
        return (
            "Physics simulation code:\n"
            "1. Define the physical system (particles, fields, etc.)\n"
            "2. Implement integrator(s)\n"
            "3. Run simulation, track observables\n"
            "4. Compare methods on the same system\n"
            "5. Output results.json with comparison data\n"
        )

    def _default_compute_budget(self) -> str:
        return (
            "Time budget for physics computations:\n"
            "- Keep simulation sizes manageable\n"
            "- Use small test systems for validation\n"
            "- Scale up only if time permits\n"
            "- Focus on accuracy, not scale"
        )

    def _default_dataset_guidance(self) -> str:
        return (
            "Physics experiments generate data programmatically:\n"
            "- Define initial conditions in code\n"
            "- Use standard test problems with known solutions\n"
            "- Do NOT download external datasets\n"
            "- Generate particle positions, velocities, or grid values in code"
        )

    def _default_hp_reporting(self) -> str:
        return (
            "Report simulation parameters:\n"
            "HYPERPARAMETERS: {'dt': ..., 'N_particles': ..., 'grid_size': ..., "
            "'num_steps': ..., 'method': ...}"
        )

    def _output_format(self, paradigm: str) -> str:
        if paradigm == "convergence":
            return (
                "Output convergence results to results.json:\n"
                '{"convergence": {"method": [{"h": 0.1, "error": 0.05}, ...]},\n'
                ' "metadata": {"domain": "...", "total_runtime_sec": ...}}'
            )
        return (
            "Output simulation results to results.json:\n"
            '{"conditions": {"method": {"metric_name": value}},\n'
            ' "metadata": {"domain": "...", "total_runtime_sec": ...}}'
        )
