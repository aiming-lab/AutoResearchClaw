"""Novelty checker — detects similar existing work before paper generation.

Searches real academic APIs (Semantic Scholar + arXiv) for papers that may
overlap with the proposed research hypotheses.  Produces a structured report
with similarity scores and a go/differentiate/abort recommendation.

Usage
-----
::

    from researchclaw.literature.novelty import check_novelty

    report = check_novelty(
        topic="Adaptive learning rate schedules",
        hypotheses_text=hypotheses_md,
    )
    print(report["novelty_score"])  # 0.72
"""

from __future__ import annotations

import json
import logging
import re
from datetime import datetime, timezone
from difflib import SequenceMatcher
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Stop words for keyword extraction (overlap with executor's but standalone)
# ---------------------------------------------------------------------------

_STOP_WORDS = frozenset(
    {
        "a",
        "an",
        "the",
        "and",
        "or",
        "but",
        "in",
        "on",
        "of",
        "for",
        "to",
        "with",
        "by",
        "at",
        "from",
        "as",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "have",
        "has",
        "had",
        "do",
        "does",
        "did",
        "will",
        "would",
        "could",
        "should",
        "may",
        "might",
        "can",
        "shall",
        "not",
        "no",
        "nor",
        "so",
        "yet",
        "both",
        "each",
        "every",
        "all",
        "any",
        "few",
        "more",
        "most",
        "other",
        "some",
        "such",
        "than",
        "too",
        "very",
        "just",
        "about",
        "above",
        "after",
        "again",
        "between",
        "into",
        "through",
        "during",
        "before",
        "under",
        "over",
        "using",
        "based",
        "via",
        "toward",
        "towards",
        "new",
        "novel",
        "approach",
        "method",
        "study",
        "research",
        "paper",
        "work",
        "propose",
        "proposed",
        "show",
        "results",
        "performance",
        "evaluation",
    }
)


# ---------------------------------------------------------------------------
# Keyword extraction
# ---------------------------------------------------------------------------


def _extract_keywords(text: str) -> list[str]:
    """Extract meaningful keywords from text (lowercased, 3+ chars, no stops)."""
    tokens = re.findall(r"[a-zA-Z][a-zA-Z0-9_-]+", text.lower())
    seen: set[str] = set()
    result: list[str] = []
    for t in tokens:
        if t not in _STOP_WORDS and len(t) >= 3 and t not in seen:
            seen.add(t)
            result.append(t)
    return result


# ---------------------------------------------------------------------------
# Similarity metrics
# ---------------------------------------------------------------------------


def _jaccard_keywords(keywords_a: list[str], keywords_b: list[str]) -> float:
    """Jaccard similarity between two keyword lists."""
    set_a = set(keywords_a)
    set_b = set(keywords_b)
    if not set_a or not set_b:
        return 0.0
    return len(set_a & set_b) / len(set_a | set_b)


def _title_similarity(title_a: str, title_b: str) -> float:
    """Sequence-based similarity between two titles (0-1)."""
    return SequenceMatcher(None, title_a.lower(), title_b.lower()).ratio()


def _compute_similarity(
    hypothesis_keywords: list[str],
    paper_title: str,
    paper_abstract: str,
    hypothesis_title: str = "",
) -> float:
    """Combined similarity score between hypotheses keywords and a paper."""
    paper_keywords = _extract_keywords(f"{paper_title} {paper_abstract}")
    kw_sim = _jaccard_keywords(hypothesis_keywords, paper_keywords)
    # Blend keyword overlap with title similarity when available
    if hypothesis_title and paper_title:
        t_sim = _title_similarity(hypothesis_title, paper_title)
        return round(0.7 * kw_sim + 0.3 * t_sim, 4)
    return round(kw_sim, 4)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def check_novelty(
    topic: str,
    hypotheses_text: str,
    *,
    papers_already_seen: list[dict[str, Any]] | None = None,
    max_search_results: int = 30,
    similarity_threshold: float = 0.25,
    s2_api_key: str = "",
) -> dict[str, Any]:
    """Check whether the proposed research has significant overlap with existing work.

    Parameters
    ----------
    topic:
        Research topic string.
    hypotheses_text:
        Full text of generated hypotheses (markdown).
    papers_already_seen:
        Papers already collected by the pipeline (from candidates.jsonl).
        If provided, these are checked for overlap too.
    max_search_results:
        Max papers to retrieve from academic APIs.
    similarity_threshold:
        Minimum similarity to flag a paper as potentially overlapping.
    s2_api_key:
        Optional Semantic Scholar API key.

    Returns
    -------
    dict with keys: topic, hypotheses_checked, similar_papers_found,
    novelty_score, assessment, similar_papers, recommendation, generated.
    """
    # Extract keywords from topic + hypotheses
    combined_text = f"{topic}\n{hypotheses_text}"
    hyp_keywords = _extract_keywords(combined_text)

    # --- Search for similar existing work ---
    similar_papers: list[dict[str, Any]] = []
    total_papers_retrieved = 0  # Track total API results (even below threshold)

    # Build search queries from hypotheses
    queries = _build_novelty_queries(topic, hypotheses_text)

    # Try real API search
    try:
        from researchclaw.literature.search import search_papers_multi_query

        found = search_papers_multi_query(
            queries,
            limit_per_query=min(15, max_search_results),
            s2_api_key=s2_api_key,
        )
        total_papers_retrieved = len(found)
        for paper in found[:max_search_results]:
            sim = _compute_similarity(hyp_keywords, paper.title, paper.abstract)
            if sim >= similarity_threshold:
                similar_papers.append(
                    {
                        "title": paper.title,
                        "paper_id": paper.paper_id,
                        "year": paper.year,
                        "venue": paper.venue,
                        "citation_count": paper.citation_count,
                        "similarity": sim,
                        "url": paper.url,
                        "cite_key": paper.cite_key,
                    }
                )
        logger.info(
            "Novelty search: %d papers found, %d above threshold %.2f",
            len(found),
            len(similar_papers),
            similarity_threshold,
        )
    except Exception:  # noqa: BLE001
        logger.warning(
            "Real novelty search failed, checking pipeline papers only", exc_info=True
        )

    # Also check papers already collected by the pipeline
    if papers_already_seen:
        for p in papers_already_seen:
            if not isinstance(p, dict):
                continue
            title = str(p.get("title", ""))
            abstract = str(p.get("abstract", ""))
            sim = _compute_similarity(hyp_keywords, title, abstract)
            if sim >= similarity_threshold:
                # Avoid duplicates
                existing_titles = {sp["title"].lower() for sp in similar_papers}
                if title.lower() not in existing_titles:
                    similar_papers.append(
                        {
                            "title": title,
                            "paper_id": str(p.get("paper_id", "")),
                            "year": p.get("year", 0),
                            "venue": str(p.get("venue", "")),
                            "citation_count": p.get("citation_count", 0),
                            "similarity": sim,
                            "url": str(p.get("url", "")),
                            "cite_key": str(p.get("cite_key", "")),
                        }
                    )

    # Sort by similarity descending
    similar_papers.sort(key=lambda x: x["similarity"], reverse=True)

    # --- Compute novelty score ---
    novelty_score, assessment = _assess_novelty(similar_papers, similarity_threshold)

    # --- Determine search coverage quality ---
    # If API returned very few papers or none at all, the novelty score is unreliable.
    if total_papers_retrieved == 0 and not papers_already_seen:
        search_coverage = "insufficient"
    elif total_papers_retrieved < 5:
        search_coverage = "partial"
    else:
        search_coverage = "full"

    # When search coverage is insufficient, flag the assessment as unreliable
    # instead of reporting a misleading perfect novelty score.
    if search_coverage == "insufficient" and not similar_papers:
        assessment = "insufficient_data"
        recommendation = "proceed_with_caution"
    elif assessment == "critical":
        recommendation = "abort"
    elif assessment == "low":
        recommendation = "differentiate"
    else:
        recommendation = "proceed"

    # Count hypotheses
    hyp_count = len(re.findall(r"^##\s+H\d+", hypotheses_text, re.MULTILINE))
    if hyp_count == 0:
        hyp_count = len(re.findall(r"hypothesis", hypotheses_text, re.IGNORECASE))
    hyp_count = max(1, hyp_count)

    return {
        "topic": topic,
        "hypotheses_checked": hyp_count,
        "search_queries": queries,
        "similar_papers_found": len(similar_papers),
        "novelty_score": novelty_score,
        "assessment": assessment,
        "similar_papers": similar_papers[:20],  # cap output size
        "recommendation": recommendation,
        "similarity_threshold": similarity_threshold,
        "search_coverage": search_coverage,
        "total_papers_retrieved": total_papers_retrieved,
        "generated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


def _build_novelty_queries(topic: str, hypotheses_text: str) -> list[str]:
    """Build targeted search queries from topic and hypotheses."""
    queries = [topic]

    # Extract hypothesis titles (## H1, ## H2, etc.)
    for match in re.finditer(r"^##\s+H\d+[:\s]*(.+)", hypotheses_text, re.MULTILINE):
        hyp_title = match.group(1).strip()
        if hyp_title and len(hyp_title) > 10:
            queries.append(hyp_title[:200])

    # Extract key phrases from the hypotheses
    keywords = _extract_keywords(hypotheses_text)[:10]
    if keywords:
        # Build a query from top keywords
        kw_query = " ".join(keywords[:5])
        if kw_query not in queries:
            queries.append(kw_query)

    return queries[:5]  # Cap at 5 queries


def _assess_novelty(
    similar_papers: list[dict[str, Any]],
    threshold: float,
) -> tuple[float, str]:
    """Compute overall novelty score and assessment.

    Returns (score, assessment) where score is 0-1 (higher = more novel)
    and assessment is 'high' | 'moderate' | 'low' | 'critical'.
    """
    if not similar_papers:
        return 1.0, "high"

    # Take top-5 most similar
    top = similar_papers[:5]
    max_sim = max(p["similarity"] for p in top)
    avg_sim = sum(p["similarity"] for p in top) / len(top)

    # High-citation papers with high similarity are more concerning
    high_cite_overlap = sum(
        1 for p in top if p["similarity"] >= 0.4 and p.get("citation_count", 0) >= 50
    )

    # Novelty score: inverse of max similarity, adjusted
    raw_score = 1.0 - max_sim
    if high_cite_overlap >= 2:
        raw_score *= 0.7  # penalty for multiple high-impact overlaps

    novelty_score = round(max(0.0, min(1.0, raw_score)), 3)

    # Assessment thresholds
    if novelty_score >= 0.7:
        assessment = "high"
    elif novelty_score >= 0.45:
        assessment = "moderate"
    elif novelty_score >= 0.25:
        assessment = "low"
    else:
        assessment = "critical"

    return novelty_score, assessment
