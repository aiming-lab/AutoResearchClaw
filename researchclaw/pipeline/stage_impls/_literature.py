"""Stages 3-6: Search strategy, literature collection, screening, and knowledge extraction."""

from __future__ import annotations

import json
import logging
import re
import shutil
from pathlib import Path
from typing import Any

import yaml

from researchclaw.adapters import AdapterBundle
from researchclaw.config import RCConfig
from researchclaw.llm.client import LLMClient
from researchclaw.literature.citation_identity import (
    CitationIdentityError,
    clean_title_text,
    parse_cite_key_registry,
    seal_citation_collection,
    validate_registry_artifacts,
)
from researchclaw.literature.evidence_cards import (
    EXTRACTION_BATCH_SIZE,
    MIN_EXCERPT_CHARS,
    CardProposal,
    EvidenceCardContractError,
    build_cards_manifest,
    build_evidence_card,
    canonical_json_text,
    load_validated_card_inputs,
    parse_card_batch_response,
    render_evidence_card_markdown,
    validate_cards_artifacts,
)
from researchclaw.literature.citation_policy import (
    CitationPolicyContractError,
    build_citation_allowlist,
    validate_citation_allowlist,
)
from researchclaw.literature.screening import (
    MAX_SCREEN_CANDIDATES,
    MAX_SCREEN_REASON_CHARS,
    MIN_RELEVANCE_SCORE,
    SCREEN_BATCH_SIZE,
    SCREENING_POLICY_VERSION,
    ScreeningContractError,
    ScreeningDecision,
    build_screening_report,
    normalize_quality_threshold,
    parse_screening_candidates,
    parse_screening_response,
    sha256_text,
)
from researchclaw.pipeline._helpers import (
    StageResult,
    _build_fallback_queries,
    _chat_with_prompt,
    _extract_topic_keywords,
    _extract_yaml_block,
    _get_evolution_overlay,
    _read_prior_artifact,
    _safe_filename,
    _safe_json_loads,
    _utcnow_iso,
    _write_jsonl,
)
from researchclaw.pipeline.stages import Stage, StageStatus
from researchclaw.prompts import PromptManager

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Local helpers
# ---------------------------------------------------------------------------


def _expand_search_queries(queries: list[str], topic: str) -> list[str]:
    """Expand search queries for broader literature coverage.

    Generates additional queries by extracting key phrases from the topic
    and creating focused sub-queries. This ensures we find papers even when
    the original queries are too narrow or specific for arXiv.
    """
    expanded = list(queries)  # keep originals
    seen = {q.lower().strip() for q in queries}

    # Extract key phrases from topic by splitting on common delimiters
    # e.g. "Comparing A, B, and C on X with Y" → ["A", "B", "C", "X", "Y"]
    topic_words = topic.split()

    # Generate shorter, broader queries from the topic
    if len(topic_words) > 5:
        # First 5 words as a broader query
        broad = " ".join(topic_words[:5])
        if broad.lower().strip() not in seen:
            expanded.append(broad)
            seen.add(broad.lower().strip())

        # Last 5 words as another perspective
        tail = " ".join(topic_words[-5:])
        if tail.lower().strip() not in seen:
            expanded.append(tail)
            seen.add(tail.lower().strip())

    # Add "survey" and "benchmark" variants of the topic
    for suffix in ("survey", "benchmark", "comparison"):
        # Take first 4 content words + suffix
        short_topic = " ".join(topic_words[:4])
        variant = f"{short_topic} {suffix}"
        if variant.lower().strip() not in seen:
            expanded.append(variant)
            seen.add(variant.lower().strip())

    return expanded


# ---------------------------------------------------------------------------
# Stage executors
# ---------------------------------------------------------------------------


def _execute_search_strategy(
    stage_dir: Path,
    run_dir: Path,
    config: RCConfig,
    adapters: AdapterBundle,
    *,
    llm: LLMClient | None = None,
    prompts: PromptManager | None = None,
) -> StageResult:
    problem_tree = _read_prior_artifact(run_dir, "problem_tree.md") or ""
    topic = config.research.topic
    plan: dict[str, Any] | None = None
    sources: list[dict[str, Any]] | None = None
    if llm is not None:
        _pm = prompts or PromptManager()
        _overlay = _get_evolution_overlay(run_dir, "search_strategy")
        sp = _pm.for_stage("search_strategy", evolution_overlay=_overlay, topic=topic, problem_tree=problem_tree)
        resp = _chat_with_prompt(
            llm,
            sp.system,
            sp.user,
            json_mode=sp.json_mode,
            max_tokens=sp.max_tokens,
        )
        payload = _safe_json_loads(resp.content, {})
        if isinstance(payload, dict):
            yaml_text = str(payload.get("search_plan_yaml", "")).strip()
            if yaml_text:
                try:
                    parsed = yaml.safe_load(_extract_yaml_block(yaml_text))
                except yaml.YAMLError:
                    parsed = None
                if isinstance(parsed, dict):
                    plan = parsed
            src = payload.get("sources", [])
            if isinstance(src, list):
                sources = [item for item in src if isinstance(item, dict)]
    model_plan_parsed = plan is not None
    if plan is None:
        # Build smart fallback queries by extracting key terms from topic
        # instead of using the raw (often very long) topic string.
        _fallback_queries = _build_fallback_queries(topic)
        plan = {
            "topic": topic,
            "generated": _utcnow_iso(),
            "search_strategies": [
                {
                    "name": "keyword_core",
                    "queries": _fallback_queries[:5],
                    "sources": ["arxiv", "semantic_scholar", "openreview"],
                    "max_results_per_query": 60,
                },
                {
                    "name": "backward_forward_citation",
                    "queries": _fallback_queries[5:10] or _fallback_queries[:3],
                    "sources": ["semantic_scholar", "google_scholar"],
                    "depth": 1,
                },
            ],
            "filters": {
                "min_year": 2020,
                "language": ["en"],
                "peer_review_preferred": True,
            },
            "deduplication": {"method": "title_doi_hash", "fuzzy_threshold": 0.9},
        }
    if not sources:
        sources = [
            {
                "id": "arxiv",
                "name": "arXiv",
                "type": "api",
                "url": "https://export.arxiv.org/api/query",
                "status": "available",
                "query": topic,
                "verified_at": _utcnow_iso(),
            },
            {
                "id": "semantic_scholar",
                "name": "Semantic Scholar",
                "type": "api",
                "url": "https://api.semanticscholar.org/graph/v1/paper/search",
                "status": "available",
                "query": topic,
                "verified_at": _utcnow_iso(),
            },
        ]
    if config.openclaw_bridge.use_web_fetch:
        for src in sources:
            try:
                response = adapters.web_fetch.fetch(str(src.get("url", "")))
                src["status"] = (
                    "verified"
                    if response.status_code in (200, 301, 302, 405)
                    else "unreachable"
                )
                src["http_status"] = response.status_code
            except Exception:  # noqa: BLE001
                src["status"] = "unknown"
    (stage_dir / "search_plan.yaml").write_text(
        yaml.dump(plan, default_flow_style=False, allow_unicode=True),
        encoding="utf-8",
    )
    (stage_dir / "sources.json").write_text(
        json.dumps(
            {"sources": sources, "count": len(sources), "generated": _utcnow_iso()},
            indent=2,
        ),
        encoding="utf-8",
    )

    # F1.5: Extract queries from plan for Stage 4 real literature search
    queries_list: list[str] = []
    year_min = 2020
    if isinstance(plan, dict):
        strategies = plan.get("search_strategies", [])
        if isinstance(strategies, list):
            for strat in strategies:
                if isinstance(strat, dict):
                    qs = strat.get("queries", [])
                    if isinstance(qs, list):
                        queries_list.extend(str(q) for q in qs if q)
        # Also accept the alternate schema where queries live under
        # query_strategies.<sub_question>.{boolean_seeds, queries}.
        if not queries_list:
            qstrats = plan.get("query_strategies", {})
            if isinstance(qstrats, dict):
                for sub in qstrats.values():
                    if not isinstance(sub, dict):
                        continue
                    for key in ("boolean_seeds", "queries"):
                        qs = sub.get(key, [])
                        if isinstance(qs, list):
                            queries_list.extend(str(q) for q in qs if q)
        filters = plan.get("filters", {})
        if isinstance(filters, dict) and filters.get("min_year"):
            try:
                year_min = int(filters["min_year"])
            except (ValueError, TypeError):
                pass

    # --- Sanitize queries: shorten overly long queries ---
    # LLMs often produce the full topic title as a query, which is too long for
    # arXiv and Semantic Scholar (they work best with 3-8 keyword queries).
    _stop = {
        "a", "an", "the", "of", "for", "in", "on", "and", "or", "with",
        "to", "by", "from", "its", "is", "are", "was", "be", "as", "at",
        "via", "using", "based", "study", "analysis", "empirical",
        "towards", "toward", "into", "exploring", "comparison", "tasks",
        "effectiveness", "investigation", "comprehensive", "novel",
        "challenge", "challenges", "gaps", "gap", "critical", "survey", "review",
    }

    def _extract_search_terms(text: str) -> list[str]:
        """Extract meaningful search terms from text, removing stop words."""
        return [
            w for w in re.split(r"[^a-zA-Z0-9]+", text)
            if w.lower() not in _stop and len(w) > 1
        ]

    _MAX_QUERY_LEN = 60  # characters — beyond this, shorten to keywords
    _SEARCH_SUFFIXES = ["benchmark", "survey", "seminal", "state of the art"]

    def _shorten_query(q: str, max_kw: int = 6) -> str:
        """Shorten a query to *max_kw* keywords, preserving any trailing suffix."""
        q_stripped = q.strip()
        # Check if query ends with a known search suffix
        suffix = ""
        q_core = q_stripped
        for sfx in _SEARCH_SUFFIXES:
            if q_stripped.lower().endswith(sfx):
                suffix = sfx
                q_core = q_stripped[: -len(sfx)].strip()
                break
        # Extract keywords from the core part
        kws = _extract_search_terms(q_core)
        shortened = " ".join(kws[:max_kw])
        if suffix:
            shortened = f"{shortened} {suffix}"
        return shortened

    if queries_list:
        sanitized: list[str] = []
        for q in queries_list:
            if len(q) > _MAX_QUERY_LEN:
                shortened = _shorten_query(q)
                if shortened.strip():
                    sanitized.append(shortened)
            else:
                sanitized.append(q)
        queries_list = sanitized

    def _build_default_search_queries(topic_text: str) -> list[str]:
        """Generate concept-style search queries from the topic instead of copying the title."""
        _words = _extract_search_terms(topic_text)
        if not _words:
            return [topic_text[:60]]
        kw_primary = " ".join(_words[:6])
        kw_short = " ".join(_words[:4])
        kw_alt = " ".join(_words[1:5]) if len(_words) > 4 else kw_short
        return [
            kw_primary,
            f"{kw_short} benchmark",
            f"{kw_short} survey",
            kw_alt,
            f"{kw_short} recent advances",
        ]

    fell_back_to_defaults = False
    if not queries_list:
        queries_list = _build_default_search_queries(topic)
        fell_back_to_defaults = True

    # Ensure minimum query diversity — if dedup leaves too few, add variants
    _all_kw = _extract_search_terms(topic)
    _seen_q: set[str] = set()
    unique_queries: list[str] = []
    for q in queries_list:
        q_lower = q.strip().lower()
        if q_lower and q_lower not in _seen_q:
            _seen_q.add(q_lower)
            unique_queries.append(q.strip())
    # If we have fewer than 5 unique queries, generate supplemental keyword variants
    if len(unique_queries) < 5 and len(_all_kw) >= 3:
        supplements = [
            " ".join(_all_kw[:4]) + " survey",
            " ".join(_all_kw[:4]) + " benchmark",
            " ".join(_all_kw[1:5]),  # shifted window for diversity
            " ".join(_all_kw[:3]) + " comparison",
            " ".join(_all_kw[:3]) + " deep learning",
            " ".join(_all_kw[2:6]),  # another shifted window
        ]
        for s in supplements:
            s_lower = s.strip().lower()
            if s_lower not in _seen_q:
                _seen_q.add(s_lower)
                unique_queries.append(s.strip())
            if len(unique_queries) >= 8:
                break
    queries_list = unique_queries
    silent_fallback = fell_back_to_defaults and model_plan_parsed
    if silent_fallback:
        logger.warning(
            "Stage 3: model plan parsed but no queries harvested; "
            "queries.json fell back to topic-derived defaults"
        )
    queries_meta = {
        "queries": queries_list,
        "year_min": year_min,
        "model_queries_extracted": model_plan_parsed and not fell_back_to_defaults,
        "fallback_reason": (
            "model_plan_used_unknown_schema" if silent_fallback else None
        ),
    }
    (stage_dir / "queries.json").write_text(
        json.dumps(queries_meta, indent=2),
        encoding="utf-8",
    )
    return StageResult(
        stage=Stage.SEARCH_STRATEGY,
        status=StageStatus.DONE,
        artifacts=("search_plan.yaml", "sources.json", "queries.json"),
        evidence_refs=(
            "stage-03/search_plan.yaml",
            "stage-03/sources.json",
            "stage-03/queries.json",
        ),
    )


def _execute_literature_collect(
    stage_dir: Path,
    run_dir: Path,
    config: RCConfig,
    adapters: AdapterBundle,
    *,
    llm: LLMClient | None = None,
    prompts: PromptManager | None = None,
) -> StageResult:
    """Stage 4: Collect literature — prefer real APIs, fallback to LLM."""
    topic = config.research.topic

    # Read queries.json from Stage 3 (F1.5 output)
    queries_text = _read_prior_artifact(run_dir, "queries.json")
    queries_data = _safe_json_loads(queries_text or "{}", {})
    queries: list[str] = queries_data.get("queries", [topic])
    year_min: int = queries_data.get("year_min", 2020)

    # --- Try real API search first ---
    candidates: list[dict[str, Any]] = []
    real_search_succeeded = False

    try:
        from researchclaw.literature.search import (
            search_papers_multi_query,
        )

        # Expand queries for broader coverage
        expanded_queries = _expand_search_queries(queries, config.research.topic)
        logger.info(
            "[literature] Searching %d queries (expanded from %d) "
            "across OpenAlex → S2 → arXiv…",
            len(expanded_queries),
            len(queries),
        )
        papers = search_papers_multi_query(
            expanded_queries,
            limit_per_query=40,
            year_min=year_min,
            s2_api_key=config.llm.s2_api_key,
        )
        if papers:
            real_search_succeeded = True
            # Count by source
            src_counts: dict[str, int] = {}
            for p in papers:
                src_counts[p.source] = src_counts.get(p.source, 0) + 1
                d = p.to_dict()
                d["collected_at"] = _utcnow_iso()
                candidates.append(d)
            src_str = ", ".join(f"{s}: {n}" for s, n in src_counts.items())
            logger.info(
                "[literature] Found %d papers (%s)", len(papers), src_str
            )
    except Exception:  # noqa: BLE001
        logger.warning(
            "[rate-limit] Literature search failed — falling back to LLM",
            exc_info=True,
        )

    # --- Inject foundational/seminal papers ---
    try:
        from researchclaw.data import load_seminal_papers
        seminal = load_seminal_papers(topic)
        if seminal:
            _existing_titles = {c.get("title", "").lower() for c in candidates}
            _injected = 0
            for sp in seminal:
                if sp.get("title", "").lower() not in _existing_titles:
                    candidates.append({
                        "id": f"seminal-{sp.get('cite_key', '')}",
                        "title": sp.get("title", ""),
                        "source": "seminal_library",
                        "url": "",
                        "year": sp.get("year", 2020),
                        "abstract": f"Foundational paper on {', '.join(sp.get('keywords', [])[:3])}.",
                        "authors": [{"name": sp.get("authors", "")}],
                        "cite_key": sp.get("cite_key", ""),
                        "venue": sp.get("venue", ""),
                        "collected_at": _utcnow_iso(),
                    })
                    _injected += 1
            if _injected:
                logger.info("Stage 4: Injected %d seminal papers from seed library", _injected)
    except Exception:  # noqa: BLE001
        logger.debug("Seminal paper injection skipped", exc_info=True)

    # --- Fallback: LLM-generated candidates ---
    if not candidates and llm is not None:
        plan_text = _read_prior_artifact(run_dir, "search_plan.yaml") or ""
        _pm = prompts or PromptManager()
        _overlay = _get_evolution_overlay(run_dir, "literature_collect")
        sp = _pm.for_stage("literature_collect", evolution_overlay=_overlay, topic=topic, plan_text=plan_text)
        resp = _chat_with_prompt(
            llm,
            sp.system,
            sp.user,
            json_mode=sp.json_mode,
            max_tokens=sp.max_tokens,
        )
        payload = _safe_json_loads(resp.content, {})
        if isinstance(payload, dict) and isinstance(payload.get("candidates"), list):
            candidates = [row for row in payload["candidates"] if isinstance(row, dict)]

    # --- Web search augmentation (Tavily/DDG + Google Scholar + Crawl4AI) ---
    web_context_parts: list[str] = []
    if config.web_search.enabled:
        try:
            from researchclaw.web.agent import WebSearchAgent
            import os

            tavily_key = config.web_search.tavily_api_key or os.environ.get(
                config.web_search.tavily_api_key_env, ""
            )
            web_agent = WebSearchAgent(
                tavily_api_key=tavily_key,
                enable_scholar=config.web_search.enable_scholar,
                enable_crawling=config.web_search.enable_crawling,
                enable_pdf=config.web_search.enable_pdf_extraction,
                max_web_results=config.web_search.max_web_results,
                max_scholar_results=config.web_search.max_scholar_results,
                max_crawl_urls=config.web_search.max_crawl_urls,
            )
            web_result = web_agent.search_and_extract(
                topic, search_queries=queries,
            )

            # Convert Google Scholar papers into candidates
            for sp in web_result.scholar_papers:
                _existing_titles = {
                    str(c.get("title", "")).lower().strip() for c in candidates
                }
                if sp.title.lower().strip() not in _existing_titles:
                    lit_paper = sp.to_literature_paper()
                    d = lit_paper.to_dict()
                    d["collected_at"] = _utcnow_iso()
                    candidates.append(d)

            # Save web search context for downstream stages
            web_context = web_result.to_context_string(max_length=20_000)
            if web_context.strip():
                (stage_dir / "web_context.md").write_text(
                    web_context, encoding="utf-8"
                )
                web_context_parts.append(web_context)

            # Save full web search metadata
            (stage_dir / "web_search_result.json").write_text(
                json.dumps(web_result.to_dict(), indent=2, default=str),
                encoding="utf-8",
            )

            logger.info(
                "[web-search] Added %d scholar papers, %d web results, %d crawled pages",
                len(web_result.scholar_papers),
                len(web_result.web_results),
                len(web_result.crawled_pages),
            )
        except Exception:  # noqa: BLE001
            logger.warning(
                "[web-search] Web search augmentation failed — continuing with academic APIs only",
                exc_info=True,
            )

    # --- Ultimate fallback: placeholder data ---
    # BUG-L2: Do NOT overwrite real_search_succeeded here — it was already
    # set correctly in the search block above. Overwriting would mislabel
    # LLM-hallucinated or seminal papers as "real search" results.
    if not candidates:
        logger.warning("Stage 4: All literature searches failed — using placeholder papers")
        candidates = [
            {
                "id": f"candidate-{idx + 1}",
                "title": f"[Placeholder] Study {idx + 1} on {topic}",
                "source": "arxiv" if idx % 2 == 0 else "semantic_scholar",
                "url": f"https://example.org/{_safe_filename(topic.lower())}/{idx + 1}",
                "year": 2024,
                "abstract": f"This candidate investigates {topic} and reports preliminary findings.",
                "collected_at": _utcnow_iso(),
                "is_placeholder": True,
            }
            for idx in range(max(20, config.research.daily_paper_count or 20))
        ]

    # Seal citation identities after all providers and injected sources have
    # contributed. Remove prior-attempt identity outputs first so a failed seal
    # cannot coexist with a stale successful registry or bibliography.
    try:
        for owned_name in (
            "candidates.jsonl",
            "references.bib",
            "cite_key_registry.json",
            "search_meta.json",
        ):
            (stage_dir / owned_name).unlink(missing_ok=True)
    except OSError as exc:
        return StageResult(
            stage=Stage.LITERATURE_COLLECT,
            status=StageStatus.FAILED,
            error=f"Could not clear stale Stage 4 citation artifacts: {exc}",
            decision="retry",
        )

    # Candidates and BibTeX are projections of this one registry.
    try:
        sealed = seal_citation_collection(candidates)
    except CitationIdentityError as exc:
        _write_jsonl(stage_dir / "candidates.jsonl", candidates)
        (stage_dir / "search_meta.json").write_text(
            json.dumps(
                {
                    "real_search": real_search_succeeded,
                    "queries_used": queries,
                    "year_min": year_min,
                    "total_candidates": len(candidates),
                    "identity_error": str(exc),
                    "ts": _utcnow_iso(),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        return StageResult(
            stage=Stage.LITERATURE_COLLECT,
            status=StageStatus.FAILED,
            artifacts=("candidates.jsonl", "search_meta.json"),
            error=f"Citation identity registry could not be sealed: {exc}",
            decision="retry",
        )

    (stage_dir / "candidates.jsonl").write_text(
        sealed.candidates_jsonl, encoding="utf-8"
    )
    (stage_dir / "references.bib").write_text(
        sealed.bibliography, encoding="utf-8"
    )
    (stage_dir / "cite_key_registry.json").write_text(
        json.dumps(sealed.registry, ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )

    artifacts = ["candidates.jsonl", "references.bib", "cite_key_registry.json"]
    if web_context_parts:
        artifacts.append("web_context.md")
    if (stage_dir / "web_search_result.json").exists():
        artifacts.append("web_search_result.json")
    logger.info(
        "Stage 4: Sealed %d identities into candidates, bibliography, and registry",
        len(sealed.candidates),
    )

    # Write search metadata
    (stage_dir / "search_meta.json").write_text(
        json.dumps(
            {
                "real_search": real_search_succeeded,
                "queries_used": queries,
                "year_min": year_min,
                "total_candidates": len(sealed.candidates),
                "raw_candidates": len(candidates),
                "bibtex_entries": len(sealed.candidates),
                "ts": _utcnow_iso(),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    artifacts.append("search_meta.json")

    return StageResult(
        stage=Stage.LITERATURE_COLLECT,
        status=StageStatus.DONE,
        artifacts=tuple(artifacts),
        evidence_refs=tuple(f"stage-04/{a}" for a in artifacts),
    )


_MAX_ABSTRACT_LEN = 800  # Truncate long abstracts to reduce token usage


def _candidate_screen_score(row: dict[str, Any], topic_keywords: list[str]) -> float:
    """Rank candidates before LLM screening so truncation preserves relevance.

    Stage 4 search providers can return globally high-citation but off-topic
    papers.  Stage 5 still performs the strict semantic decision; this score
    only determines which filtered candidates fit in the screening prompt.
    """
    title = str(row.get("title", "")).lower()
    abstract = str(row.get("abstract", "")).lower()
    text_blob = f"{title} {abstract}"
    tokens = set(re.findall(r"[a-zA-Z][a-zA-Z0-9_-]+", text_blob))

    score = 0.0
    for kw in topic_keywords:
        parts = [p for p in re.split(r"[-_]+", kw.lower()) if p]
        if kw in title:
            score += 3.0
        elif kw in abstract:
            score += 1.5
        for part in parts:
            if part in tokens:
                score += 0.8 if part in title else 0.25

    # If the topic itself is about hardware-security detection, prefer
    # candidates that connect the attack, measurement, and detection axes.
    security_terms = (
        "spectre",
        "meltdown",
        "transient execution",
        "side-channel",
        "side channel",
        "flush+reload",
        "flush reload",
        "cache attack",
    )
    counter_terms = (
        "hardware performance counter",
        "performance counter",
        "hpc",
        "pmu",
        "perf_event",
        "microarchitectural",
        "microarchitecture",
    )
    detection_terms = (
        "detection",
        "detect",
        "anomaly",
        "change-point",
        "change point",
        "cusum",
        "page-hinkley",
        "runtime monitor",
    )
    if any(term in text_blob for term in security_terms):
        score += 4.0
    if any(term in text_blob for term in counter_terms):
        score += 3.0
    if any(term in text_blob for term in detection_terms):
        score += 2.0
    if any(term in text_blob for term in security_terms) and any(
        term in text_blob for term in counter_terms
    ):
        score += 5.0
    if any(term in text_blob for term in security_terms) and any(
        term in text_blob for term in detection_terms
    ):
        score += 3.0

    return score


def _execute_literature_screen(
    stage_dir: Path,
    run_dir: Path,
    config: RCConfig,
    adapters: AdapterBundle,
    *,
    llm: LLMClient | None = None,
    prompts: PromptManager | None = None,
) -> StageResult:
    try:
        for artifact_name in (
            "shortlist.jsonl",
            "screening_partial.jsonl",
            "screening_report.json",
            "screen_meta.json",
        ):
            (stage_dir / artifact_name).unlink(missing_ok=True)
    except OSError as exc:
        return StageResult(
            stage=Stage.LITERATURE_SCREEN,
            status=StageStatus.FAILED,
            artifacts=(),
            error=f"Failed to clear stale Stage 5 artifacts: {exc}",
            decision="retry",
        )

    try:
        stage4_dir = run_dir / "stage-04"
        canonical_paths = {
            "candidates": stage4_dir / "candidates.jsonl",
            "registry": stage4_dir / "cite_key_registry.json",
            "bibliography": stage4_dir / "references.bib",
        }
        for label, path in canonical_paths.items():
            if path.is_symlink() or not path.is_file():
                raise OSError(f"canonical Stage 4 {label} is missing or not a file")
        candidates_text = canonical_paths["candidates"].read_text(encoding="utf-8")
        registry_text = canonical_paths["registry"].read_text(encoding="utf-8")
        bibliography_text = canonical_paths["bibliography"].read_text(
            encoding="utf-8"
        )
        registry = parse_cite_key_registry(registry_text)
        validate_registry_artifacts(registry, candidates_text, bibliography_text)
        candidate_rows = list(parse_screening_candidates(candidates_text))
    except (
        CitationIdentityError,
        OSError,
        ScreeningContractError,
        UnicodeDecodeError,
    ) as exc:
        return StageResult(
            stage=Stage.LITERATURE_SCREEN,
            status=StageStatus.FAILED,
            artifacts=(),
            error=f"Stage 4 sealed citation collection is invalid: {exc}",
            decision="retry",
        )
    try:
        minimum_quality_score = normalize_quality_threshold(
            config.research.quality_threshold
        )
    except ScreeningContractError as exc:
        return StageResult(
            stage=Stage.LITERATURE_SCREEN,
            status=StageStatus.FAILED,
            artifacts=(),
            error=f"Invalid Stage 5 screening policy: {exc}",
            decision="retry",
        )

    # Deterministic prefilter and rank. Rows excluded by this policy are final
    # Stage 5 rejections, not unscreened candidates and not candidates for
    # minimum-count backfill.
    topic_keywords = _extract_topic_keywords(
        config.research.topic, config.research.domains
    )
    ranked_rows: list[dict[str, Any]] = []
    prefilter_rejected_ids: list[str] = []
    for source_row in candidate_rows:
        row = dict(source_row)
        source_identity = str(row["source_identity"])
        title = str(row.get("title", "")).lower()
        abstract = str(row.get("abstract", "")).lower()
        text_blob = f"{title} {abstract}"
        overlap = sum(1 for kw in topic_keywords if kw in text_blob)
        if overlap >= 1:
            row["keyword_overlap"] = overlap
            row["screen_rank_score"] = round(
                _candidate_screen_score(row, topic_keywords), 3
            )
            ranked_rows.append(row)
        else:
            prefilter_rejected_ids.append(source_identity)
    ranked_rows.sort(
        key=lambda r: (
            float(r.get("screen_rank_score", 0.0) or 0.0),
            int(r.get("keyword_overlap", 0) or 0),
            int(r.get("citation_count", 0) or 0),
            str(r.get("source_identity", "")),
        ),
        reverse=True,
    )
    admitted_rows = ranked_rows[:MAX_SCREEN_CANDIDATES]
    prefilter_rejected_ids.extend(
        str(row["source_identity"]) for row in ranked_rows[MAX_SCREEN_CANDIDATES:]
    )
    logger.info(
        "Stage 5 deterministic prefilter: admitted %d/%d candidates "
        "(batch_size=%d, keywords=%s)",
        len(admitted_rows),
        len(candidate_rows),
        SCREEN_BATCH_SIZE,
        topic_keywords[:8],
    )

    claim_scope = config.experiment.claim_scope
    allows_degraded_screening = claim_scope == "pipeline_validation"
    selected_rows: list[dict[str, Any]] = []
    screened_ids: list[str] = []
    unscreened_ids: list[str] = []
    failed_batches: list[dict[str, str]] = []
    batches = [
        admitted_rows[index:index + SCREEN_BATCH_SIZE]
        for index in range(0, len(admitted_rows), SCREEN_BATCH_SIZE)
    ]
    if llm is None and admitted_rows:
        unscreened_ids.extend(str(row["source_identity"]) for row in admitted_rows)
        failed_batches.extend(
            {
                "batch_id": f"screen-batch-{index:03d}",
                "error": "LLM client unavailable",
            }
            for index in range(1, len(batches) + 1)
        )
    elif llm is not None:
        for batch_index, batch_rows in enumerate(batches, start=1):
            batch_id = f"screen-batch-{batch_index:03d}"
            expected_ids = [str(row["source_identity"]) for row in batch_rows]
            try:
                decisions = _screen_candidate_batch(
                    llm=llm,
                    prompts=prompts,
                    run_dir=run_dir,
                    config=config,
                    batch_id=batch_id,
                    rows=batch_rows,
                    minimum_quality_score=minimum_quality_score,
                )
            except (RuntimeError, ScreeningContractError) as exc:
                failed_batches.append(
                    {"batch_id": batch_id, "error": str(exc)[:1000]}
                )
                unscreened_ids.extend(expected_ids)
                if not allows_degraded_screening:
                    for remaining in batches[batch_index:]:
                        unscreened_ids.extend(
                            str(row["source_identity"]) for row in remaining
                        )
                    break
                continue

            screened_ids.extend(expected_ids)
            row_by_id = {str(row["source_identity"]): row for row in batch_rows}
            for decision in decisions:
                if not decision.keep:
                    continue
                selected = dict(row_by_id[decision.source_identity])
                selected["screening_policy_version"] = SCREENING_POLICY_VERSION
                selected["relevance_score"] = decision.relevance_score
                selected["quality_score"] = decision.quality_score
                selected["keep_reason"] = decision.reason
                selected_rows.append(selected)

    shortlist, semantic_duplicate_ids = _deduplicate_screened_candidates(
        selected_rows, admitted_rows
    )
    shortlist_text = "".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
        for row in shortlist
    )
    degradation_codes: list[str] = []
    if failed_batches:
        degradation_codes.extend(["screen_batch_failed", "screening_incomplete"])
    failure_error: str | None = None
    if not shortlist:
        failure_error = (
            "Strict screening produced no valid shortlist. Refine Stage 3 "
            "queries; Stage 5 does not backfill rejected or malformed rows."
        )
    elif not allows_degraded_screening and failed_batches:
        failure_error = (
            f"{claim_scope} screening is incomplete; failed batches are fatal"
        )

    output_name = "screening_partial.jsonl" if failure_error else "shortlist.jsonl"
    output_path = f"stage-05/{output_name}"
    report = build_screening_report(
        candidates_sha256=sha256_text(candidates_text),
        registry_sha256=sha256_text(registry_text),
        references_sha256=sha256_text(bibliography_text),
        screening_output_path=output_path,
        screening_output_sha256=sha256_text(shortlist_text),
        minimum_quality_score=minimum_quality_score,
        claim_scope=claim_scope,
        candidate_ids=[str(row["source_identity"]) for row in candidate_rows],
        prefilter_rejected_ids=prefilter_rejected_ids,
        screened_ids=screened_ids,
        selected_ids=[str(row["source_identity"]) for row in shortlist],
        semantic_duplicate_ids=semantic_duplicate_ids,
        unscreened_ids=unscreened_ids,
        batch_count=len(batches),
        failed_batches=failed_batches,
        degraded=bool(degradation_codes),
        degradation_codes=degradation_codes,
    )
    (stage_dir / "screening_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    # Publish the canonical shortlist last. If report persistence fails, the
    # stage fails without leaving a success-named artifact that resume can use.
    (stage_dir / output_name).write_text(shortlist_text, encoding="utf-8")

    artifacts = (output_name, "screening_report.json")
    evidence_refs = tuple(f"stage-05/{name}" for name in artifacts)
    if failure_error:
        return StageResult(
            stage=Stage.LITERATURE_SCREEN,
            status=StageStatus.FAILED,
            artifacts=artifacts,
            evidence_refs=evidence_refs,
            error=failure_error,
            decision="retry",
        )
    return StageResult(
        stage=Stage.LITERATURE_SCREEN,
        status=StageStatus.DONE,
        artifacts=artifacts,
        evidence_refs=evidence_refs,
        decision="degraded" if failed_batches else None,
    )


def _screen_candidate_batch(
    *,
    llm: LLMClient,
    prompts: PromptManager | None,
    run_dir: Path,
    config: RCConfig,
    batch_id: str,
    rows: list[dict[str, Any]],
    minimum_quality_score: float,
) -> tuple[ScreeningDecision, ...]:
    prompt_rows = []
    for row in rows:
        abstract = str(row.get("abstract", ""))
        prompt_rows.append(
            {
                "source_identity": row["source_identity"],
                "title": row.get("title", ""),
                "abstract": (
                    abstract[:_MAX_ABSTRACT_LEN] + "..."
                    if len(abstract) > _MAX_ABSTRACT_LEN
                    else abstract
                ),
                "year": row.get("year", 0),
                "venue": row.get("venue", ""),
                "citation_count": row.get("citation_count", 0),
                "source": row.get("source", ""),
                "doi": row.get("doi", ""),
                "arxiv_id": row.get("arxiv_id", ""),
                "screen_rank_score": row.get("screen_rank_score", 0.0),
                "keyword_overlap": row.get("keyword_overlap", 0),
            }
        )
    batch_text = "\n".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True) for row in prompt_rows
    )
    expected_ids = [str(row["source_identity"]) for row in rows]
    contract = (
        "\n\nSTAGE 5 BATCH OUTPUT CONTRACT (OVERRIDES ANY EARLIER RETURN SCHEMA):\n"
        f"- batch_id must be exactly {batch_id}.\n"
        "- Return exactly one decision for every source_identity below.\n"
        "- Do not return or modify candidate metadata.\n"
        "- decision is exactly keep or reject. Scores are numbers in [0,1].\n"
        f"- reason is a nonempty screening explanation of at most "
        f"{MAX_SCREEN_REASON_CHARS} Unicode code points.\n"
        f"- keep requires relevance_score >= {MIN_RELEVANCE_SCORE:.2f} and "
        f"quality_score >= {minimum_quality_score:.2f}; reject otherwise.\n"
        "- Return ONLY this JSON object shape:\n"
        '{"schema_version":1,"batch_id":"...","decisions":['
        '{"source_identity":"...","decision":"keep|reject",'
        '"relevance_score":0.0,"quality_score":0.0,"reason":"..."}]}\n'
        f"EXPECTED SOURCE IDENTITIES: {json.dumps(expected_ids)}\n"
    )
    manager = prompts or PromptManager()
    overlay = _get_evolution_overlay(run_dir, "literature_screen")
    stage_prompt = manager.for_stage(
        "literature_screen",
        evolution_overlay=overlay,
        topic=config.research.topic,
        domains=", ".join(config.research.domains)
        if config.research.domains
        else "general",
        quality_threshold=f"{minimum_quality_score:.2f} on a 0-1 scale",
        candidates_text=batch_text,
    )
    user_prompt = stage_prompt.user + contract
    response = _chat_with_prompt(
        llm,
        stage_prompt.system,
        user_prompt,
        json_mode=True,
        max_tokens=stage_prompt.max_tokens,
    )
    try:
        return parse_screening_response(
            response.content,
            expected_batch_id=batch_id,
            expected_source_ids=expected_ids,
            minimum_quality_score=minimum_quality_score,
        )
    except ScreeningContractError as initial_error:
        repair_prompt = (
            user_prompt
            + "\n\nTHE PREVIOUS RESPONSE VIOLATED THE CONTRACT:\n"
            + str(initial_error)
            + "\nRegenerate the complete batch once. Do not omit any ID."
        )
        repaired = _chat_with_prompt(
            llm,
            stage_prompt.system,
            repair_prompt,
            json_mode=True,
            max_tokens=stage_prompt.max_tokens,
            retries=0,
        )
        try:
            return parse_screening_response(
                repaired.content,
                expected_batch_id=batch_id,
                expected_source_ids=expected_ids,
                minimum_quality_score=minimum_quality_score,
            )
        except ScreeningContractError as repair_error:
            raise ScreeningContractError(
                f"initial_error={initial_error}; repair_error={repair_error}"
            ) from repair_error


def _deduplicate_screened_candidates(
    selected_rows: list[dict[str, Any]],
    admitted_rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[str]]:
    rank = {
        str(row["source_identity"]): index for index, row in enumerate(admitted_rows)
    }
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in selected_rows:
        groups.setdefault(_semantic_screen_key(row), []).append(row)

    winners: list[dict[str, Any]] = []
    duplicates: list[str] = []
    for rows in groups.values():
        winner = max(
            rows,
            key=lambda row: (
                float(row.get("relevance_score", 0.0)),
                float(row.get("quality_score", 0.0)),
                int(row.get("citation_count", 0) or 0),
                str(row.get("source_identity", "")),
            ),
        )
        winners.append(winner)
        duplicates.extend(
            str(row["source_identity"]) for row in rows if row is not winner
        )
    winners.sort(key=lambda row: rank[str(row["source_identity"])])
    return winners, sorted(duplicates)


def _semantic_screen_key(row: dict[str, Any]) -> str:
    title = clean_title_text(row.get("title", "")).casefold()
    raw_authors = row.get("authors", [])
    first_author = ""
    if isinstance(raw_authors, list) and raw_authors:
        first = raw_authors[0]
        if isinstance(first, dict):
            first_author = str(first.get("name", ""))
        elif isinstance(first, str):
            first_author = first
    author_key = re.sub(r"\W+", "", first_author.casefold())
    try:
        year = int(row.get("year", 0) or 0)
    except (TypeError, ValueError):
        year = 0
    return f"{title}\n{author_key}\n{year}"


def _execute_knowledge_extract(
    stage_dir: Path,
    run_dir: Path,
    config: RCConfig,
    adapters: AdapterBundle,
    *,
    llm: LLMClient | None = None,
    prompts: PromptManager | None = None,
) -> StageResult:
    cards_dir = stage_dir / "cards"
    manifest_path = stage_dir / "cards_manifest.json"
    allowlist_path = stage_dir / "citation_allowlist.json"
    failure_path = stage_dir / "card_extraction_failures.json"
    try:
        if cards_dir.is_symlink():
            cards_dir.unlink()
        elif cards_dir.exists():
            shutil.rmtree(cards_dir)
        manifest_path.unlink(missing_ok=True)
        allowlist_path.unlink(missing_ok=True)
        failure_path.unlink(missing_ok=True)
    except OSError as exc:
        return StageResult(
            stage=Stage.KNOWLEDGE_EXTRACT,
            status=StageStatus.FAILED,
            artifacts=(),
            error=f"Cannot clean stale Stage 6 outputs: {exc}",
            decision="retry",
        )

    try:
        inputs = load_validated_card_inputs(run_dir, config)
    except EvidenceCardContractError as exc:
        return StageResult(
            stage=Stage.KNOWLEDGE_EXTRACT,
            status=StageStatus.FAILED,
            artifacts=(),
            error=f"Stage 6 canonical inputs are invalid: {exc}",
            decision="retry",
        )

    proposals: dict[str, CardProposal] = {}
    failures: dict[str, str] = {}
    batches = [
        inputs.shortlist[index:index + EXTRACTION_BATCH_SIZE]
        for index in range(0, len(inputs.shortlist), EXTRACTION_BATCH_SIZE)
    ]
    for batch_index, batch in enumerate(batches, start=1):
        expected_ids = [str(row["source_identity"]) for row in batch]
        if llm is None:
            failures.update({identity: "LLM client unavailable" for identity in expected_ids})
            continue
        try:
            batch_proposals = _extract_evidence_card_batch(
                llm=llm,
                prompts=prompts,
                run_dir=run_dir,
                config=config,
                batch_id=f"card-batch-{batch_index:03d}",
                rows=list(batch),
            )
            proposals.update(
                {proposal.source_identity: proposal for proposal in batch_proposals}
            )
        except (RuntimeError, EvidenceCardContractError) as exc:
            failures.update({identity: str(exc)[:1000] for identity in expected_ids})

    candidates_sha256 = sha256_text(inputs.candidates_text)
    cards: list[dict[str, Any]] = []
    for index, candidate in enumerate(inputs.shortlist, start=1):
        identity = str(candidate["source_identity"])
        cards.append(
            build_evidence_card(
                card_id=f"card-{index:03d}",
                candidate=candidate,
                candidates_sha256=candidates_sha256,
                proposal=proposals.get(identity),
                failure_reason=failures.get(identity),
            )
        )

    successful_cards = [
        card for card in cards if card["extraction_status"] == "success"
    ]
    if not successful_cards:
        diagnostic = {
            "schema_version": 1,
            "status": "failed",
            "reason": "zero_eligible_evidence_cards",
            "cards": cards,
        }
        try:
            failure_path.write_text(canonical_json_text(diagnostic), encoding="utf-8")
        except OSError as exc:
            return StageResult(
                stage=Stage.KNOWLEDGE_EXTRACT,
                status=StageStatus.FAILED,
                artifacts=(),
                error=f"Stage 6 produced zero eligible evidence and diagnostics failed: {exc}",
                decision="retry",
            )
        return StageResult(
            stage=Stage.KNOWLEDGE_EXTRACT,
            status=StageStatus.FAILED,
            artifacts=(failure_path.name,),
            evidence_refs=(f"stage-06/{failure_path.name}",),
            error="Stage 6 produced zero eligible abstract evidence cards",
            decision="retry",
        )

    serialized_cards: list[tuple[dict[str, Any], str, str]] = []
    try:
        cards_dir.mkdir(parents=True, exist_ok=False)
        for card in cards:
            json_text = canonical_json_text(card)
            markdown_text = render_evidence_card_markdown(card)
            card_id = str(card["card_id"])
            (cards_dir / f"{card_id}.json").write_text(json_text, encoding="utf-8")
            (cards_dir / f"{card_id}.md").write_text(markdown_text, encoding="utf-8")
            serialized_cards.append((card, json_text, markdown_text))
        manifest = build_cards_manifest(
            shortlist_sha256=sha256_text(inputs.shortlist_text),
            screening_report_sha256=sha256_text(inputs.screening_report_text),
            cards=serialized_cards,
        )
        manifest_text = canonical_json_text(manifest)
        manifest_path.write_text(manifest_text, encoding="utf-8")
        validate_cards_artifacts(
            stage_dir=stage_dir,
            manifest=manifest,
            shortlist_text=inputs.shortlist_text,
            screening_report_text=inputs.screening_report_text,
            candidates_sha256=candidates_sha256,
            shortlist=inputs.shortlist,
        )
        allowlist = build_citation_allowlist(run_dir, config)
        allowlist_text = canonical_json_text(allowlist)
        allowlist_path.write_text(allowlist_text, encoding="utf-8")
        validate_citation_allowlist(run_dir, config, allowlist_text)
    except (
        CitationPolicyContractError,
        EvidenceCardContractError,
        OSError,
        UnicodeDecodeError,
    ) as exc:
        manifest_path.unlink(missing_ok=True)
        allowlist_path.unlink(missing_ok=True)
        if cards_dir.is_symlink():
            cards_dir.unlink(missing_ok=True)
        elif cards_dir.exists():
            shutil.rmtree(cards_dir, ignore_errors=True)
        return StageResult(
            stage=Stage.KNOWLEDGE_EXTRACT,
            status=StageStatus.FAILED,
            artifacts=(),
            error=f"Stage 6 evidence-card persistence failed: {exc}",
            decision="retry",
        )
    return StageResult(
        stage=Stage.KNOWLEDGE_EXTRACT,
        status=StageStatus.DONE,
        artifacts=("cards/", "cards_manifest.json", "citation_allowlist.json"),
        evidence_refs=(
            "stage-06/cards/",
            "stage-06/cards_manifest.json",
            "stage-06/citation_allowlist.json",
        ),
    )


def _extract_evidence_card_batch(
    *,
    llm: LLMClient,
    prompts: PromptManager | None,
    run_dir: Path,
    config: RCConfig,
    batch_id: str,
    rows: list[dict[str, Any]],
) -> tuple[CardProposal, ...]:
    expected_ids = [str(row["source_identity"]) for row in rows]
    source_rows = [
        {
            "source_identity": row["source_identity"],
            "title": row["title"],
            "abstract": row["abstract"],
        }
        for row in rows
    ]
    contract = (
        "Extract structured summaries only from each retained abstract. "
        "Return one JSON object and no prose.\n"
        f"schema_version must be 1; batch_id must be {batch_id}.\n"
        "cards must contain exactly one object for every source_identity. "
        "Each card has exactly source_identity, summary_text, and "
        "evidence_excerpt_texts. summary_text has exactly problem, method, "
        "data, metrics, findings, limitations, each a nonempty string. "
        "evidence_excerpt_texts has 1-4 unique, verbatim substrings copied from "
        f"that source's abstract, each at least {MIN_EXCERPT_CHARS} Unicode "
        "code points long. Do not infer missing evidence.\n"
        f"SOURCES:\n{json.dumps(source_rows, ensure_ascii=False, sort_keys=True)}"
    )
    manager = prompts or PromptManager()
    stage_prompt = manager.for_stage(
        "knowledge_extract",
        evolution_overlay=_get_evolution_overlay(run_dir, "knowledge_extract"),
        shortlist="",
    )
    response = _chat_with_prompt(
        llm,
        stage_prompt.system,
        contract,
        json_mode=True,
        max_tokens=stage_prompt.max_tokens,
        retries=1,
    )
    try:
        return parse_card_batch_response(
            response.content,
            expected_batch_id=batch_id,
            expected_source_ids=expected_ids,
        )
    except EvidenceCardContractError as initial_error:
        repair_prompt = (
            contract
            + "\n\nThe previous response violated the exact contract: "
            + str(initial_error)
            + "\nRegenerate the complete batch once."
        )
        repaired = _chat_with_prompt(
            llm,
            stage_prompt.system,
            repair_prompt,
            json_mode=True,
            max_tokens=stage_prompt.max_tokens,
            retries=0,
        )
        try:
            return parse_card_batch_response(
                repaired.content,
                expected_batch_id=batch_id,
                expected_source_ids=expected_ids,
            )
        except EvidenceCardContractError as repair_error:
            raise EvidenceCardContractError(
                f"initial_error={initial_error}; repair_error={repair_error}"
            ) from repair_error
