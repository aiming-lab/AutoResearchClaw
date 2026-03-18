"""Code Searcher agent — searches GitHub for reference code before generation.

This agent searches GitHub repositories and code to find relevant examples
that inform the blueprint generation process, especially for domains where
the LLM's internal knowledge may be insufficient.
"""

from researchclaw.agents.code_searcher.agent import CodeSearchAgent, CodeSearchResult

__all__ = ["CodeSearchAgent", "CodeSearchResult"]
