"""Chemistry domain prompt adapter."""

from __future__ import annotations

from typing import Any

from researchclaw.domains.prompt_adapter import PromptAdapter, PromptBlocks


class ChemistryPromptAdapter(PromptAdapter):
    """Adapter for chemistry domains (quantum chemistry, molecular property)."""

    def get_code_generation_blocks(self, context: dict[str, Any]) -> PromptBlocks:
        domain = self.domain

        return PromptBlocks(
            compute_budget=domain.compute_budget_guidance or (
                "Quantum chemistry calculations can be slow:\n"
                "- Use small basis sets for testing (STO-3G)\n"
                "- Limit molecule size (< 20 atoms)\n"
                "- DFT is faster than post-HF methods"
            ),
            dataset_guidance=domain.dataset_guidance or (
                "Define molecular systems in code:\n"
                "- Atomic coordinates and basis sets\n"
                "- Standard test molecules (H2, H2O, CH4)\n"
                "- Do NOT download external datasets"
            ),
            code_generation_hints=domain.code_generation_hints or self._default_hints(),
            output_format_guidance=(
                "Output results to results.json:\n"
                '{"conditions": {"method": {"energy_hartree": -1.13, "error_kcal_mol": 0.5}},\n'
                ' "metadata": {"domain": "chemistry_qm"}}'
            ),
        )

    def get_experiment_design_blocks(self, context: dict[str, Any]) -> PromptBlocks:
        domain = self.domain

        design_context = (
            f"This is a **{domain.display_name}** experiment.\n\n"
            "Key principles:\n"
            "1. Use well-defined molecular test sets\n"
            "2. Compare against high-level reference (CCSD(T) or experimental)\n"
            "3. Report energies in Hartree, errors in kcal/mol\n"
            "4. Vary basis set for convergence if applicable\n"
        )

        return PromptBlocks(
            experiment_design_context=design_context,
            statistical_test_guidance="Use MAE, RMSE, max error for method comparison.",
        )

    def get_result_analysis_blocks(self, context: dict[str, Any]) -> PromptBlocks:
        return PromptBlocks(
            result_analysis_hints=(
                "Chemistry result analysis:\n"
                "- Report MAE, RMSE in kcal/mol against reference\n"
                "- 'Chemical accuracy' = MAE < 1 kcal/mol\n"
                "- Compare computation time vs accuracy trade-off"
            ),
        )

    def _default_hints(self) -> str:
        if self.domain.domain_id == "chemistry_qm":
            return (
                "Quantum chemistry code with PySCF:\n"
                "1. mol = gto.M(atom='...', basis='sto-3g')\n"
                "2. mf = scf.RHF(mol); mf.kernel()\n"
                "3. For post-HF: mp2 = mp.MP2(mf); mp2.kernel()\n"
                "4. Compare methods on same molecule set\n"
                "5. Energy conversion: 1 Ha = 627.509 kcal/mol\n"
                "6. Output results.json\n"
            )
        return (
            "Molecular property prediction:\n"
            "1. Define molecules via SMILES strings\n"
            "2. Use RDKit for featurization\n"
            "3. Train/test split on molecular data\n"
            "4. Compare ML models (RF, XGBoost, GCN)\n"
            "5. Output results.json with MAE, RMSE\n"
        )
