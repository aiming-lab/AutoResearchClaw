"""Medical observational research prompt adapter.

This adapter adds safeguards for retrospective clinical and epidemiology
studies without changing the generic 23-stage pipeline.
"""

from __future__ import annotations

from typing import Any

from researchclaw.domains.prompt_adapter import PromptAdapter, PromptBlocks


class MedicalObservationalPromptAdapter(PromptAdapter):
    """Adapter for de-identified medical observational research."""

    def get_code_generation_blocks(self, context: dict[str, Any]) -> PromptBlocks:
        domain = self.domain

        return PromptBlocks(
            compute_budget=domain.compute_budget_guidance or (
                "Medical observational analyses are usually CPU-bound. Keep "
                "demo datasets small, transparent, and reproducible."
            ),
            dataset_guidance=domain.dataset_guidance or (
                "Use synthetic data or a de-identified schema for examples. "
                "Do not use real patient data in generated demos or tests. "
                "Require explicit de-identification and secondary-use approval "
                "before handling user-provided clinical data."
            ),
            hp_reporting=domain.hp_reporting_guidance or (
                "Report study-design parameters: study design, sample counts, "
                "exposure, outcome, confounders, missing-data strategy, and "
                "ethics status."
            ),
            code_generation_hints=domain.code_generation_hints or self._default_code_hints(),
            output_format_guidance=self._output_format(),
        )

    def get_experiment_design_blocks(self, context: dict[str, Any]) -> PromptBlocks:
        design_context = (
            f"This is a **{self.domain.display_name}** study.\n\n"
            "Required planning gates before data analysis:\n"
            "1. Ethics/IRB status: approved, exempt/waived, pending, or not yet "
            "available. Never fabricate an IRB number or approval date.\n"
            "2. Privacy gate: confirm de-identification, remove direct identifiers, "
            "and avoid institution-sensitive or patient-identifiable details.\n"
            "3. Data access plan: document source type (EHR/HIS export, trauma "
            "registry, CSV/Excel, SPSS output), date range, linkage keys, and "
            "whether data are synthetic or real.\n"
            "4. Variable definition plan: define exposure, outcome, confounders, "
            "effect modifiers, follow-up window, and clinical coding rules before "
            "modelling.\n"
            "5. Missingness plan: quantify missingness, discuss likely mechanism, "
            "choose complete-case analysis by default, and add imputation only "
            "when assumptions are justified.\n"
            "6. STROBE alignment: preserve participant flow counts, eligibility "
            "criteria, statistical methods, limitations, and bias discussion.\n"
        )

        return PromptBlocks(
            experiment_design_context=design_context,
            statistical_test_guidance=(
                "Use STROBE-aligned observational statistics: Table 1 with "
                "standardized mean differences, chi-square or Fisher exact tests "
                "for categorical variables, t-test or Mann-Whitney U for continuous "
                "variables, adjusted logistic or linear regression for primary "
                "associations, Cox regression only for time-to-event outcomes, "
                "and sensitivity analyses for missing data and residual confounding."
            ),
        )

    def get_result_analysis_blocks(self, context: dict[str, Any]) -> PromptBlocks:
        return PromptBlocks(
            result_analysis_hints=self.domain.result_analysis_hints or (
                "Medical observational result analysis:\n"
                "- Interpret associations cautiously; do not imply causation from "
                "retrospective observational data alone.\n"
                "- Report exclusions, missingness, model covariates, effect estimates, "
                "95% confidence intervals, and sensitivity analyses.\n"
                "- Tie every claim to a table, coefficient, confidence interval, or "
                "STROBE flow count.\n"
                "- Explicitly discuss residual confounding, selection bias, and data "
                "quality limitations."
            ),
            statistical_test_guidance=(
                "Report effect estimates with 95% confidence intervals, p values "
                "where appropriate, standardized mean differences for baseline "
                "balance, and model diagnostics or proportional hazards checks "
                "when applicable."
            ),
        )

    def get_export_publish_blocks(self, context: dict[str, Any]) -> PromptBlocks:
        return PromptBlocks(
            export_publish_guidance=(
                "Export as a generic medical manuscript. Include an ethics "
                "statement, de-identification statement, STROBE checklist note, "
                "participant-flow summary, Table 1, primary effect estimates, "
                "limitations, and a clear statement that the generated content is "
                "research-methodology assistance, not clinical advice."
            ),
            preferred_template="generic",
        )

    def _default_code_hints(self) -> str:
        return (
            "Medical observational code requirements:\n"
            "1. Create an ethics/privacy gate before analysis.\n"
            "2. Define eligibility, exposure, outcome, confounders, and follow-up "
            "before modelling.\n"
            "3. Implement cleaning and exclusion logs.\n"
            "4. Generate Table 1 and adjusted primary models.\n"
            "5. Output results.json with ethics, data_plan, table_1, strobe_flow, "
            "primary_model, sensitivity_analyses, and limitations.\n"
            "6. Do not use real patient data in examples or tests.\n"
        )

    def _output_format(self) -> str:
        return (
            "Output results to results.json:\n"
            '{"ethics": {"irb_status": "approved|exempt|pending|unknown", '
            '"de_identified": true},\n'
            ' "data_plan": {"source_type": "synthetic|ehr|his|registry|csv", '
            '"exposure": "...", "outcome": "...", "confounders": [...]},\n'
            ' "table_1": {"groups": {...}, "smd": {...}},\n'
            ' "strobe_flow": {"initial": 1000, "excluded": [], "analyzed": 1000},\n'
            ' "primary_model": {"effect": "...", "estimate": 1.23, '
            '"ci95": [1.01, 1.49]},\n'
            ' "sensitivity_analyses": [], "limitations": []}'
        )
