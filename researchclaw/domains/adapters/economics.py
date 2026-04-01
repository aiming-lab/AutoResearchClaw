"""Economics domain prompt adapter.

Provides domain-specific prompt blocks for empirical economics
experiments (regression analysis, causal inference, panel data).
"""

from __future__ import annotations

from typing import Any

from researchclaw.domains.prompt_adapter import PromptAdapter, PromptBlocks


class EconomicsPromptAdapter(PromptAdapter):
    """Adapter for economics domains."""

    def get_code_generation_blocks(self, context: dict[str, Any]) -> PromptBlocks:
        domain = self.domain

        return PromptBlocks(
            compute_budget=domain.compute_budget_guidance or (
                "Economics regressions are fast. Focus on:\n"
                "- Multiple specifications (4-6 columns)\n"
                "- Bootstrap SE if needed (100-500 reps)\n"
                "- Cluster-robust SE for panel data"
            ),
            dataset_guidance=domain.dataset_guidance or (
                "Generate synthetic data with known treatment effect (DGP):\n"
                "- Include treatment, outcome, controls, fixed effects\n"
                "- Simulate realistic correlations and confounders\n"
                "- Do NOT download external datasets"
            ),
            hp_reporting=domain.hp_reporting_guidance or (
                "Report specification details:\n"
                "HYPERPARAMETERS: {'n_obs': ..., 'n_controls': ..., "
                "'true_effect': ..., 'fe_groups': ..., 'cluster_var': ...}"
            ),
            code_generation_hints=domain.code_generation_hints or self._default_hints(),
            output_format_guidance=self._output_format(),
        )

    def get_experiment_design_blocks(self, context: dict[str, Any]) -> PromptBlocks:
        domain = self.domain

        design_context = (
            f"This is an **{domain.display_name}** experiment.\n"
            f"Paradigm: progressive specification\n\n"
            "Key principles for economics experiments:\n"
            "1. Start simple (OLS), add complexity progressively\n"
            "2. Report each specification as a column in a regression table\n"
            "3. Use robust/clustered standard errors\n"
            "4. Include at least one robustness check\n"
            "5. Data should be generated with a known DGP for validation\n"
        )

        return PromptBlocks(
            experiment_design_context=design_context,
            statistical_test_guidance=(
                "Use Hausman test for FE vs RE choice, "
                "F-test for joint significance, "
                "robust/clustered SE for inference."
            ),
        )

    def get_result_analysis_blocks(self, context: dict[str, Any]) -> PromptBlocks:
        return PromptBlocks(
            result_analysis_hints=(
                "Economics result analysis:\n"
                "- Compare coefficient estimates across specifications\n"
                "- Check if treatment effect is robust to controls/FE\n"
                "- Report significance levels (*/**/***)\n"
                "- Discuss economic magnitude, not just statistical significance"
            ),
            statistical_test_guidance=(
                "Use Hausman test, robust SE, cluster SE. "
                "Report R², N, F-statistic for each specification."
            ),
        )

    def _default_hints(self) -> str:
        return (
            "Economics code requirements:\n"
            "1. Generate synthetic data with statsmodels/numpy\n"
            "2. Implement progressive specifications:\n"
            "   - Spec 1: Simple OLS (Y ~ treatment)\n"
            "   - Spec 2: OLS + controls (Y ~ treatment + X1 + X2)\n"
            "   - Spec 3: Fixed effects (Y ~ treatment + X + entity FE)\n"
            "   - Spec 4: IV / 2SLS if applicable\n"
            "3. Use robust/clustered standard errors\n"
            "4. Output regression table to results.json\n"
            "5. Use linearmodels for panel FE, statsmodels for OLS/IV\n"
        )

    def _output_format(self) -> str:
        return (
            "Output regression table to results.json:\n"
            '{"regression_table": {\n'
            '    "spec_1_ols": {"coeff": 0.15, "se": 0.03, "p": 0.001, "n": 5000, "r2": 0.12},\n'
            '    "spec_2_controls": {"coeff": 0.12, "se": 0.02, "p": 0.001, "n": 5000, "r2": 0.25}\n'
            '},\n'
            ' "metadata": {"domain": "economics_empirical", "total_runtime_sec": ...}}'
        )
