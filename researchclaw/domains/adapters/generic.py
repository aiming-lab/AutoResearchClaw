"""Generic domain adapter — fallback for unknown/new domains.

Re-exports GenericPromptAdapter from prompt_adapter.py so that the
adapters package has a consistent interface.
"""

from researchclaw.domains.prompt_adapter import GenericPromptAdapter

__all__ = ["GenericPromptAdapter"]
