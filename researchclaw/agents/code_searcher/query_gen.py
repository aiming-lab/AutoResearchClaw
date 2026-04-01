"""LLM-based search query generation for code search.

Given a research topic and domain, generates targeted search queries
for GitHub repository and code search.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

logger = logging.getLogger(__name__)

_QUERY_GEN_PROMPT = """\
You are generating GitHub search queries to find reference code for a research experiment.

Research topic: {topic}
Domain: {domain_name}
Core libraries: {libraries}
Specific needs: {needs}

Generate 3-5 search queries that will help find:
1. Example implementations using the domain's core libraries
2. Similar research projects or experiments
3. Specific API usage patterns needed for this experiment

Rules:
- Each query should be 3-8 words (GitHub search works best with short queries)
- Include library names when searching for API usage
- Include domain-specific terms
- Focus on FINDING CODE, not documentation

Respond as a JSON array of strings. Example:
["pyscf DFT hartree fock example", "molecular energy calculation python"]

Queries:"""


def generate_search_queries(
    topic: str,
    domain_name: str,
    core_libraries: list[str],
    specific_needs: list[str] | None = None,
    llm: Any | None = None,
) -> list[str]:
    """Generate search queries for GitHub code search.

    If no LLM is provided, generates queries from topic keywords and
    library names using heuristic rules.

    Parameters
    ----------
    topic : str
        Research topic.
    domain_name : str
        Domain display name.
    core_libraries : list[str]
        Domain's core libraries.
    specific_needs : list[str], optional
        Specific API/library needs.
    llm : LLMClient, optional
        LLM for query generation.

    Returns
    -------
    list[str]
        3-5 search queries.
    """
    if llm is not None:
        return _llm_generate(topic, domain_name, core_libraries, specific_needs or [], llm)

    return _heuristic_generate(topic, domain_name, core_libraries, specific_needs or [])


def _heuristic_generate(
    topic: str,
    domain_name: str,
    libraries: list[str],
    needs: list[str],
) -> list[str]:
    """Generate queries without LLM using keyword extraction."""
    queries: list[str] = []

    # Clean topic: extract key phrases
    topic_words = _extract_key_phrases(topic)

    # Query 1: Topic + main library
    if libraries:
        queries.append(f"{topic_words} {libraries[0]}")

    # Query 2: Domain + "python example"
    queries.append(f"{domain_name.lower()} python example")

    # Query 3: Specific library usage
    for lib in libraries[:2]:
        queries.append(f"{lib} example tutorial python")

    # Query 4: Specific needs
    for need in needs[:2]:
        queries.append(f"{need} python")

    # Deduplicate and limit
    seen: set[str] = set()
    unique: list[str] = []
    for q in queries:
        q_norm = q.lower().strip()
        if q_norm not in seen:
            seen.add(q_norm)
            unique.append(q)

    return unique[:5]


def _llm_generate(
    topic: str,
    domain_name: str,
    libraries: list[str],
    needs: list[str],
    llm: Any,
) -> list[str]:
    """Generate queries using LLM."""
    try:
        prompt = _QUERY_GEN_PROMPT.format(
            topic=topic,
            domain_name=domain_name,
            libraries=", ".join(libraries),
            needs=", ".join(needs) if needs else "general usage",
        )

        # Synchronous LLM call — LLMClient.chat() is sync and takes
        # (messages, *, system=, max_tokens=) signature.
        if hasattr(llm, "chat"):
            resp = llm.chat(
                [{"role": "user", "content": prompt}],
                system="You generate concise GitHub search queries.",
                max_tokens=200,
            )
        else:
            return _heuristic_generate(topic, domain_name, libraries, needs)

        content = resp.content if hasattr(resp, "content") else str(resp)

        # Parse JSON array from response
        json_match = re.search(r"\[.*\]", content, re.DOTALL)
        if json_match:
            queries = json.loads(json_match.group())
            if isinstance(queries, list) and all(isinstance(q, str) for q in queries):
                return queries[:5]

        logger.warning("Failed to parse LLM query response, using heuristic")
        return _heuristic_generate(topic, domain_name, libraries, needs)

    except Exception:
        logger.warning("LLM query generation failed", exc_info=True)
        return _heuristic_generate(topic, domain_name, libraries, needs)


def _extract_key_phrases(text: str, max_words: int = 5) -> str:
    """Extract key phrases from a research topic."""
    # Remove common filler words
    stop_words = {
        "a", "an", "the", "of", "for", "in", "on", "with", "and", "or",
        "to", "by", "is", "are", "using", "based", "via", "through",
        "novel", "new", "improved", "efficient", "towards",
    }
    words = text.lower().split()
    key_words = [w for w in words if w not in stop_words and len(w) > 2]
    return " ".join(key_words[:max_words])
