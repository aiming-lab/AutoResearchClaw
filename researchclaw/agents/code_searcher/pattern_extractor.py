"""Extract reusable code patterns from GitHub search results.

Uses LLM to analyze reference code and extract:
  - API call patterns (how to use a specific library)
  - File organization patterns (project structure)
  - Data processing patterns (data loading / preprocessing)
  - Evaluation patterns (how to compute and report metrics)
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class CodePatterns:
    """Extracted patterns from reference code."""
    api_patterns: list[str] = field(default_factory=list)
    file_structure: dict[str, str] = field(default_factory=dict)
    data_patterns: list[str] = field(default_factory=list)
    evaluation_patterns: list[str] = field(default_factory=list)
    library_versions: dict[str, str] = field(default_factory=dict)
    raw_snippets: list[str] = field(default_factory=list)

    def to_prompt_context(self) -> str:
        """Format patterns as context for code generation prompts."""
        parts: list[str] = []

        if self.api_patterns:
            parts.append("## Reference API Usage Patterns")
            for i, pattern in enumerate(self.api_patterns[:5], 1):
                parts.append(f"### Pattern {i}")
                parts.append(f"```python\n{pattern}\n```")

        if self.file_structure:
            parts.append("\n## Reference Project Structure")
            for fname, desc in self.file_structure.items():
                parts.append(f"- `{fname}`: {desc}")

        if self.evaluation_patterns:
            parts.append("\n## Reference Evaluation Patterns")
            for pattern in self.evaluation_patterns[:3]:
                parts.append(f"```python\n{pattern}\n```")

        return "\n".join(parts)

    @property
    def has_content(self) -> bool:
        return bool(self.api_patterns or self.file_structure or self.evaluation_patterns)


_EXTRACT_PROMPT = """\
You are analyzing reference code to extract reusable patterns for a research project.

Research topic: {topic}
Domain: {domain_name}

Here are code snippets from relevant GitHub repositories:

{code_snippets}

Extract the following patterns as JSON:

{{
    "api_patterns": [
        "# Short, self-contained code snippet showing key API usage",
        "# Each should be 3-10 lines showing one specific API call pattern"
    ],
    "file_structure": {{
        "filename.py": "what this file does"
    }},
    "evaluation_patterns": [
        "# How results are computed and reported"
    ],
    "library_versions": {{
        "library_name": "recommended version"
    }}
}}

Focus on:
1. How the core libraries are imported and used
2. Common data loading / preprocessing patterns
3. How experiments are structured
4. How results are computed and reported

Return ONLY valid JSON."""


def extract_patterns(
    code_snippets: list[str],
    topic: str,
    domain_name: str,
    llm: Any | None = None,
) -> CodePatterns:
    """Extract code patterns from reference snippets.

    Parameters
    ----------
    code_snippets : list[str]
        Code content from GitHub repos.
    topic : str
        Research topic for context.
    domain_name : str
        Domain name for context.
    llm : LLMClient, optional
        LLM for pattern extraction. Falls back to heuristic if not provided.

    Returns
    -------
    CodePatterns
    """
    if not code_snippets:
        return CodePatterns()

    if llm is not None:
        return _llm_extract(code_snippets, topic, domain_name, llm)

    return _heuristic_extract(code_snippets)


def _llm_extract(
    snippets: list[str],
    topic: str,
    domain_name: str,
    llm: Any,
) -> CodePatterns:
    """Extract patterns using LLM analysis."""
    try:
        # Truncate snippets to fit context
        combined = ""
        for i, snippet in enumerate(snippets[:5]):
            truncated = snippet[:2000] if len(snippet) > 2000 else snippet
            combined += f"\n--- Snippet {i+1} ---\n{truncated}\n"

        prompt = _EXTRACT_PROMPT.format(
            topic=topic,
            domain_name=domain_name,
            code_snippets=combined,
        )

        if hasattr(llm, "chat"):
            import asyncio
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                loop = None
            if loop and loop.is_running():
                return _heuristic_extract(snippets)
            resp = llm.chat(
                [{"role": "user", "content": prompt}],
                system="You extract code patterns as JSON.",
                max_tokens=1500,
            )
        else:
            return _heuristic_extract(snippets)

        content = resp.content if hasattr(resp, "content") else str(resp)

        # Parse JSON from response
        json_match = re.search(r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", content, re.DOTALL)
        if json_match:
            data = json.loads(json_match.group())
            return CodePatterns(
                api_patterns=data.get("api_patterns", []),
                file_structure=data.get("file_structure", {}),
                evaluation_patterns=data.get("evaluation_patterns", []),
                library_versions=data.get("library_versions", {}),
                raw_snippets=snippets[:5],
            )

    except Exception:
        logger.warning("LLM pattern extraction failed", exc_info=True)

    return _heuristic_extract(snippets)


def _heuristic_extract(snippets: list[str]) -> CodePatterns:
    """Extract patterns using regex heuristics (no LLM needed)."""
    patterns = CodePatterns(raw_snippets=snippets[:5])

    for snippet in snippets:
        # Extract import statements as API patterns
        imports = re.findall(r"^(?:from|import)\s+.+$", snippet, re.MULTILINE)
        for imp in imports[:10]:
            if imp not in patterns.api_patterns:
                patterns.api_patterns.append(imp)

        # Extract function/class definitions for structure hints
        defs = re.findall(r"^(?:def|class)\s+(\w+)", snippet, re.MULTILINE)
        for d in defs[:5]:
            if d not in patterns.file_structure:
                patterns.file_structure[d] = "detected function/class"

    # Deduplicate
    patterns.api_patterns = list(dict.fromkeys(patterns.api_patterns))[:10]

    return patterns
