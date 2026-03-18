"""Mathematics domain prompt adapter."""

from __future__ import annotations

from typing import Any

from researchclaw.domains.prompt_adapter import PromptAdapter, PromptBlocks


class MathPromptAdapter(PromptAdapter):
    """Adapter for numerical mathematics and optimization domains."""

    def get_code_generation_blocks(self, context: dict[str, Any]) -> PromptBlocks:
        domain = self.domain
        paradigm = domain.experiment_paradigm

        return PromptBlocks(
            compute_budget=domain.compute_budget_guidance or (
                "Numerical methods are typically fast.\n"
                "Use 5-8 refinement levels for convergence plots.\n"
                "Step sizes: geometric sequence (h, h/2, h/4, ...)"
            ),
            dataset_guidance=domain.dataset_guidance or (
                "Use standard test problems with known solutions:\n"
                "- ODE: Lotka-Volterra, Van der Pol, stiff systems\n"
                "- Quadrature: smooth, oscillatory, singular integrands\n"
                "- Linear algebra: Hilbert matrix, tridiagonal\n"
                "- Do NOT download external datasets"
            ),
            code_generation_hints=domain.code_generation_hints or self._hints(paradigm),
            output_format_guidance=self._output_format(paradigm),
        )

    def get_experiment_design_blocks(self, context: dict[str, Any]) -> PromptBlocks:
        return PromptBlocks(
            experiment_design_context=(
                f"This is a **{self.domain.display_name}** experiment.\n"
                "Focus on:\n"
                "1. Correctness (verify against known solutions)\n"
                "2. Convergence order (expected vs observed)\n"
                "3. Efficiency (operations count, wall time)\n"
            ),
            statistical_test_guidance="Use convergence order fitting for accuracy analysis.",
        )

    def get_result_analysis_blocks(self, context: dict[str, Any]) -> PromptBlocks:
        return PromptBlocks(
            result_analysis_hints=(
                "Numerical methods analysis:\n"
                "- Convergence: fit log(error) vs log(h)\n"
                "- Stability: check for growth in error over long runs\n"
                "- Efficiency: compare accuracy per unit computation"
            ),
        )

    def _hints(self, paradigm: str) -> str:
        if paradigm == "convergence":
            return (
                "Numerical methods convergence study:\n"
                "1. Implement methods from scratch (not just scipy wrappers)\n"
                "2. Use test problems with KNOWN exact solutions\n"
                "3. Run at 5+ refinement levels\n"
                "4. Compute error: ||u_h - u_exact||_2\n"
                "5. Report convergence order: p = log(e_h / e_{h/2}) / log(2)\n"
                "6. Output results.json with convergence data"
            )
        return (
            "Numerical/optimization code:\n"
            "1. Implement algorithms from scratch\n"
            "2. Test on standard benchmark functions\n"
            "3. Compare accuracy and efficiency\n"
            "4. Output results.json"
        )

    def _output_format(self, paradigm: str) -> str:
        if paradigm == "convergence":
            return (
                "Output convergence results to results.json:\n"
                '{"convergence": {"method": [{"h": 0.1, "error": 0.05}, ...]}}'
            )
        return (
            "Output results to results.json:\n"
            '{"conditions": {"optimizer": {"iterations": 100, "final_value": 0.001}}}'
        )
