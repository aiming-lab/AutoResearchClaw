"""Computational neuroscience domain prompt adapter.

Provides domain-specific prompt blocks for neural simulation
experiments (spiking networks, neural dynamics, population coding,
brain imaging analysis).
"""

from __future__ import annotations

from typing import Any

from researchclaw.domains.prompt_adapter import PromptAdapter, PromptBlocks


class NeurosciencePromptAdapter(PromptAdapter):
    """Adapter for computational neuroscience domains."""

    def get_code_generation_blocks(self, context: dict[str, Any]) -> PromptBlocks:
        domain = self.domain
        paradigm = domain.experiment_paradigm
        libs = (
            ", ".join(domain.core_libraries)
            if domain.core_libraries
            else "numpy, scipy, brian2"
        )

        return PromptBlocks(
            compute_budget=domain.compute_budget_guidance
            or self._default_compute_budget(),
            dataset_guidance=domain.dataset_guidance
            or self._default_dataset_guidance(),
            hp_reporting=domain.hp_reporting_guidance
            or self._default_hp_reporting(),
            code_generation_hints=domain.code_generation_hints
            or self._default_code_hints(paradigm),
            output_format_guidance=self._output_format(paradigm),
        )

    def get_experiment_design_blocks(self, context: dict[str, Any]) -> PromptBlocks:
        domain = self.domain

        design_context = (
            f"This is a **{domain.display_name}** experiment.\n"
            f"Paradigm: {domain.experiment_paradigm}\n\n"
            "Key principles for neuroscience simulations:\n"
            "1. Use biologically plausible parameters (membrane time constants, "
            "synaptic weights, firing rates)\n"
            "2. Validate single-neuron dynamics before scaling to networks\n"
            "3. Report spike statistics: firing rate, CV of ISI, Fano factor\n"
            "4. For network models, specify connectivity (E/I ratio, sparsity)\n"
            "5. Compare against established neuron models as baselines\n"
        )

        if domain.standard_baselines:
            design_context += (
                f"\nStandard reference models: "
                f"{', '.join(domain.standard_baselines)}\n"
            )

        stats = (
            ", ".join(domain.statistical_tests)
            if domain.statistical_tests
            else "paired t-test, KS test"
        )

        return PromptBlocks(
            experiment_design_context=design_context,
            statistical_test_guidance=(
                f"Use {stats} for result analysis. "
                "For spike train comparison, use van Rossum distance "
                "or SPIKE-distance when appropriate."
            ),
        )

    def get_result_analysis_blocks(self, context: dict[str, Any]) -> PromptBlocks:
        return PromptBlocks(
            result_analysis_hints=self.domain.result_analysis_hints
            or (
                "Neuroscience result analysis:\n"
                "- Firing rate: mean ± std across neurons and trials\n"
                "- Regularity: CV of inter-spike intervals (CV < 1 regular, "
                "CV ≈ 1 Poisson-like)\n"
                "- Synchrony: pairwise spike-count correlation\n"
                "- Population: Fano factor, dimensionality\n"
                "- Decoding: accuracy with cross-validation, information (bits)\n"
                "- Use raster plots and PSTHs for visualization"
            ),
            statistical_test_guidance=(
                "Use paired t-test or permutation test for firing rate "
                "comparisons. Use KS test for ISI distribution comparisons."
            ),
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _default_code_hints(self, paradigm: str) -> str:
        if paradigm == "simulation":
            return (
                "Neural simulation code:\n"
                "1. Define neuron model (LIF, Izhikevich, or Hodgkin-Huxley)\n"
                "2. Set biologically plausible parameters "
                "(tau_m=20ms, V_thresh=-50mV, V_reset=-65mV for LIF)\n"
                "3. Generate input stimulus (Poisson spikes or step current)\n"
                "4. Run simulation, record spikes and membrane potential\n"
                "5. Compute spike statistics: rate, CV ISI, Fano factor\n"
                "6. Compare multiple models on the same stimulus\n"
                "7. Output results.json with comparison data\n"
            )
        return (
            "Neural analysis code:\n"
            "1. Generate or load neural activity data\n"
            "2. Preprocess: spike sorting / binning / filtering\n"
            "3. Compute relevant metrics\n"
            "4. Compare methods on the same data\n"
            "5. Output results to results.json\n"
        )

    def _default_compute_budget(self) -> str:
        return (
            "Time budget for neural simulations:\n"
            "- Single neuron models: very fast, run many trials\n"
            "- Small networks (< 1000 neurons): seconds per run\n"
            "- Large networks: use vectorized code or Brian2\n"
            "- Keep biological time reasonable (100ms–10s)"
        )

    def _default_dataset_guidance(self) -> str:
        return (
            "Neuroscience experiments generate data programmatically:\n"
            "- Define neuron parameters and connectivity in code\n"
            "- Generate Poisson spike trains for inputs\n"
            "- Use standard test circuits (E/I balanced, WTA)\n"
            "- Do NOT download external neural datasets\n"
            "- For brain imaging: generate synthetic fMRI/EEG signals"
        )

    def _default_hp_reporting(self) -> str:
        return (
            "Report simulation parameters:\n"
            "HYPERPARAMETERS: {'n_neurons': ..., 'tau_m_ms': ..., "
            "'v_thresh_mV': ..., 'sim_duration_ms': ..., 'dt_ms': ..., "
            "'connectivity': ...}"
        )

    def _output_format(self, paradigm: str) -> str:
        return (
            "Output neural simulation results to results.json:\n"
            '{"conditions": {"model_name": {"firing_rate_hz": 12.5, '
            '"cv_isi": 0.85, "fano_factor": 1.02}},\n'
            ' "metadata": {"domain": "neuroscience_computational", '
            '"sim_duration_ms": ..., "n_neurons": ...}}'
        )
