"""Security domain prompt adapter."""

from __future__ import annotations

from typing import Any

from researchclaw.domains.prompt_adapter import PromptAdapter, PromptBlocks


class SecurityPromptAdapter(PromptAdapter):
    """Adapter for security/intrusion detection domains."""

    def get_code_generation_blocks(self, context: dict[str, Any]) -> PromptBlocks:
        domain = self.domain
        return PromptBlocks(
            dataset_guidance=domain.dataset_guidance or (
                "Generate synthetic network/security data in code:\n"
                "- Normal traffic patterns + attack patterns\n"
                "- Class-imbalanced (realistic: ~5% attacks)\n"
                "- Do NOT download external datasets"
            ),
            code_generation_hints=domain.code_generation_hints or (
                "Security detection code:\n"
                "1. Generate synthetic tabular features (packet size, duration, etc.)\n"
                "2. Train classifiers (RF, XGBoost, SVM)\n"
                "3. Evaluate with TPR, FPR, F1, per-class metrics\n"
                "4. Report confusion matrix\n"
                "5. Output results to results.json\n"
            ),
            output_format_guidance=(
                "Output results to results.json:\n"
                '{"conditions": {"detector": {"TPR": 0.95, "FPR": 0.02, "F1": 0.93}}}'
            ),
        )

    def get_experiment_design_blocks(self, context: dict[str, Any]) -> PromptBlocks:
        return PromptBlocks(
            experiment_design_context=(
                "This is a **security/intrusion detection** experiment.\n"
                "Key: class imbalance, low false positive rate is critical.\n"
                "Compare detectors on same data splits.\n"
            ),
        )

    def get_result_analysis_blocks(self, context: dict[str, Any]) -> PromptBlocks:
        return PromptBlocks(
            result_analysis_hints=(
                "Security analysis: focus on FPR (false alarm rate) alongside TPR.\n"
                "Per-class F1 is important for multi-class attack detection."
            ),
        )
