"""ML domain prompt adapter — preserves existing behavior exactly.

This adapter returns empty PromptBlocks for all stages, which signals
the pipeline to use the existing hardcoded ML behavior in prompts.py.
This is the **zero-regression guarantee** for ML functionality.
"""

from __future__ import annotations

from typing import Any

from researchclaw.domains.prompt_adapter import PromptAdapter, PromptBlocks


class MLPromptAdapter(PromptAdapter):
    """ML adapter: delegates to existing prompts.py behavior unchanged."""

    def get_code_generation_blocks(self, context: dict[str, Any]) -> PromptBlocks:
        return PromptBlocks()

    def get_experiment_design_blocks(self, context: dict[str, Any]) -> PromptBlocks:
        return PromptBlocks()

    def get_result_analysis_blocks(self, context: dict[str, Any]) -> PromptBlocks:
        return PromptBlocks()
