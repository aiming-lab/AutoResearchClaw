"""Biology domain prompt adapter.

Provides domain-specific prompt blocks for bioinformatics
experiments (single-cell analysis, genomics, protein science).
"""

from __future__ import annotations

from typing import Any

from researchclaw.domains.prompt_adapter import PromptAdapter, PromptBlocks


class BiologyPromptAdapter(PromptAdapter):
    """Adapter for biology/bioinformatics domains."""

    def get_code_generation_blocks(self, context: dict[str, Any]) -> PromptBlocks:
        domain = self.domain

        return PromptBlocks(
            compute_budget=domain.compute_budget_guidance or (
                "Bioinformatics analyses can be memory-intensive:\n"
                "- Use small/subsampled datasets for testing\n"
                "- Single-cell: cap at 5000 cells for benchmarks\n"
                "- Genomics: use small chromosomes/regions"
            ),
            dataset_guidance=domain.dataset_guidance or (
                "Generate synthetic biological data in code:\n"
                "- Single-cell: use scanpy.datasets or simulate with splatter\n"
                "- Genomics: generate synthetic sequences\n"
                "- Do NOT download external datasets"
            ),
            hp_reporting=domain.hp_reporting_guidance or (
                "Report analysis parameters:\n"
                "HYPERPARAMETERS: {'n_cells': ..., 'n_genes': ..., "
                "'n_hvg': ..., 'n_pcs': ..., 'resolution': ...}"
            ),
            code_generation_hints=domain.code_generation_hints or self._default_hints(),
            output_format_guidance=(
                "Output results to results.json:\n"
                '{"conditions": {"method": {"ARI": 0.85, "NMI": 0.82}},\n'
                ' "metadata": {"domain": "biology_singlecell"}}'
            ),
        )

    def get_experiment_design_blocks(self, context: dict[str, Any]) -> PromptBlocks:
        domain = self.domain

        design_context = (
            f"This is a **{domain.display_name}** experiment.\n\n"
            "Key principles:\n"
            "1. Proper preprocessing is critical (QC, normalization)\n"
            "2. Use standard evaluation metrics (ARI, NMI for clustering)\n"
            "3. Compare against established methods in the field\n"
            "4. Include sensitivity analysis for key parameters\n"
        )

        return PromptBlocks(
            experiment_design_context=design_context,
            statistical_test_guidance=(
                "Use Wilcoxon rank-sum test with FDR correction "
                "for differential expression. Use ARI/NMI for clustering."
            ),
        )

    def get_result_analysis_blocks(self, context: dict[str, Any]) -> PromptBlocks:
        return PromptBlocks(
            result_analysis_hints=(
                "Biology result analysis:\n"
                "- Clustering: ARI, NMI, silhouette score\n"
                "- DE: number of DEGs at FDR < 0.05\n"
                "- Trajectory: pseudotime correlation\n"
                "- Report runtime alongside quality metrics"
            ),
        )

    def _default_hints(self) -> str:
        return (
            "Bioinformatics code requirements:\n"
            "1. Use scanpy for single-cell analysis\n"
            "2. Standard pipeline: load → QC → normalize → log1p → HVG → PCA → neighbors\n"
            "3. Compare clustering methods (Leiden, Louvain, K-means)\n"
            "4. Evaluate with ARI against known cell types\n"
            "5. Output results to results.json\n"
        )
