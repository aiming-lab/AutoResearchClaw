"""Stages 16-17: Paper outline and paper draft generation."""

from __future__ import annotations

import json
import logging
import math
import re
from pathlib import Path
from typing import Any

import yaml

from researchclaw.adapters import AdapterBundle
from researchclaw.config import RCConfig
from researchclaw.llm.client import LLMClient
from researchclaw.pipeline._domain import _detect_domain, _is_ml_domain
from researchclaw.pipeline._helpers import (
    StageResult,
    _build_context_preamble,
    _chat_with_prompt,
    _collect_experiment_results,
    _default_paper_outline,
    _extract_paper_title,
    _generate_framework_diagram_prompt,
    _generate_neurips_checklist,
    _get_evolution_overlay,
    _read_best_analysis,
    _read_prior_artifact,
    _safe_json_loads,
    _topic_constraint_block,
    _utcnow_iso,
)
from researchclaw.pipeline.stages import Stage, StageStatus
from researchclaw.prompts import PromptManager

logger = logging.getLogger(__name__)


def _execute_paper_outline(
    stage_dir: Path,
    run_dir: Path,
    config: RCConfig,
    adapters: AdapterBundle,
    *,
    llm: LLMClient | None = None,
    prompts: PromptManager | None = None,
) -> StageResult:
    analysis = _read_best_analysis(run_dir)
    decision = _read_prior_artifact(run_dir, "decision.md") or ""
    preamble = _build_context_preamble(
        config,
        run_dir,
        include_analysis=True,
        include_decision=True,
        include_experiment_data=True,
    )

    # WS-5.2: Read iteration feedback if available (multi-round iteration)
    feedback = ""
    iter_ctx_path = run_dir / "iteration_context.json"
    if iter_ctx_path.exists():
        try:
            ctx = json.loads(iter_ctx_path.read_text(encoding="utf-8"))
            iteration = ctx.get("iteration", 1)
            prev_score = ctx.get("quality_score")
            reviews_excerpt = ctx.get("reviews_excerpt", "")
            if iteration > 1 and reviews_excerpt:
                feedback = (
                    f"\n\n## Iteration {iteration} Feedback\n"
                    f"Previous quality score: {prev_score}/10\n"
                    f"Reviewer feedback to address:\n{reviews_excerpt[:2000]}\n"
                    f"\nYou MUST address these reviewer concerns in this revision.\n"
                )
        except (json.JSONDecodeError, KeyError):
            pass

    if llm is not None:
        _pm = prompts or PromptManager()
        # IMP-20: Pass academic style guide block for outline stage
        try:
            _asg = _pm.block("academic_style_guide")
        except (KeyError, Exception):
            _asg = ""
        _overlay = _get_evolution_overlay(run_dir, "paper_outline")
        sp = _pm.for_stage(
            "paper_outline",
            evolution_overlay=_overlay,
            preamble=preamble,
            topic_constraint=_pm.block("topic_constraint", topic=config.research.topic),
            feedback=feedback,
            analysis=analysis,
            decision=decision,
            academic_style_guide=_asg,
        )
        resp = _chat_with_prompt(
            llm,
            sp.system,
            sp.user,
            json_mode=sp.json_mode,
            max_tokens=sp.max_tokens,
        )
        outline = resp.content
        # Reasoning models may consume all tokens on CoT — retry with more
        if not outline.strip() and sp.max_tokens:
            logger.warning("Empty outline from LLM — retrying with 2x tokens")
            resp = _chat_with_prompt(
                llm,
                sp.system,
                sp.user,
                json_mode=sp.json_mode,
                max_tokens=sp.max_tokens * 2,
            )
            outline = resp.content
        if not outline.strip():
            logger.warning("LLM returned empty outline — using default")
            outline = _default_paper_outline(config.research.topic)
    else:
        outline = _default_paper_outline(config.research.topic)
    (stage_dir / "outline.md").write_text(outline, encoding="utf-8")
    return StageResult(
        stage=Stage.PAPER_OUTLINE,
        status=StageStatus.DONE,
        artifacts=("outline.md",),
        evidence_refs=("stage-16/outline.md",),
    )


def _collect_raw_experiment_metrics(run_dir: Path) -> tuple[str, bool]:
    """Collect raw experiment metric lines from stdout for paper writing.

    Returns a tuple of (formatted block, has_parsed_metrics).
    ``has_parsed_metrics`` is True when at least one run had a non-empty
    ``metrics`` dict in its JSON payload — a reliable signal of real data.
    """
    metric_lines: list[str] = []
    run_count = 0
    has_parsed_metrics = False

    for stage_subdir in sorted(run_dir.glob("stage-*/runs")):
        for run_file in sorted(stage_subdir.glob("*.json")):
            if run_file.name == "results.json":
                continue
            try:
                payload = json.loads(run_file.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                continue
            if not isinstance(payload, dict):
                continue

            # R10: Skip simulated data — only collect real experiment results
            if payload.get("status") == "simulated":
                continue

            run_count += 1

            # Extract from parsed metrics (check both 'metrics' and 'key_metrics')
            metrics = payload.get("metrics", {}) or payload.get("key_metrics", {})
            if isinstance(metrics, dict) and metrics:
                has_parsed_metrics = True
                for k, v in metrics.items():
                    metric_lines.append(f"  {k}: {v}")

            # Also extract from stdout for full detail
            # BUG-23: Filter out infrastructure lines that are NOT experiment results
            _INFRA_KEYS = {
                "SEED_COUNT", "TIME_ESTIMATE", "TRAINING_STEPS",
                "REGISTERED_CONDITIONS", "METRIC_DEF", "GPU_MEMORY",
                "BATCH_SIZE", "NUM_WORKERS", "TOTAL_PARAMS",
                "time_budget_sec", "max_epochs", "num_seeds",
            }
            stdout = payload.get("stdout", "")
            if stdout:
                for line in stdout.splitlines():
                    line = line.strip()
                    if ":" in line:
                        parts = line.rsplit(":", 1)
                        try:
                            float(parts[1].strip())
                            key_part = parts[0].strip().split("/")[-1]  # last segment
                            if key_part in _INFRA_KEYS:
                                continue  # skip infrastructure lines
                            metric_lines.append(f"  {line}")
                        except (ValueError, TypeError, IndexError):
                            pass

    # R19-4 + R23-1: Collect metrics from refinement_log.json (Stage 13).
    # If refinement has richer data than Stage 12 runs/, REPLACE Stage 12 data
    # to avoid confusing the paper writer with conflicting sources.
    _refine_lines: list[str] = []
    _refine_run_count = 0
    # Scan ALL refinement logs across versions, pick by quality (primary
    # metric) then richness (metric count).  BUG-207: Previous logic picked
    # the sandbox entry with the most metric keys regardless of whether it
    # represented a regression (e.g. sandbox_after_fix with 1.29% accuracy
    # winning over sandbox with 78.93% because it had 6 more keys).
    _best_refine_metrics: dict[str, Any] = {}
    _best_refine_stdout = ""
    _best_refine_primary: float | None = None
    for _rl_path in sorted(run_dir.glob("stage-13*/refinement_log.json")):
        try:
            _rlog = json.loads(_rl_path.read_text(encoding="utf-8"))
            for _it in _rlog.get("iterations", []):
                for _sbx_key in ("sandbox", "sandbox_after_fix"):
                    _sbx = _it.get(_sbx_key, {})
                    if not isinstance(_sbx, dict):
                        continue
                    _sbx_metrics = _sbx.get("metrics", {})
                    if not isinstance(_sbx_metrics, dict) or not _sbx_metrics:
                        continue
                    # Extract primary metric value for quality comparison
                    _sbx_primary: float | None = None
                    for _pm_key in ("primary_metric", "best_metric"):
                        if _pm_key in _sbx_metrics:
                            try:
                                _sbx_primary = float(_sbx_metrics[_pm_key])
                            except (ValueError, TypeError):
                                pass
                            break
                    # Prefer higher primary metric; fall back to count
                    _dominated = False
                    if _best_refine_primary is not None and _sbx_primary is not None:
                        if _sbx_primary > _best_refine_primary:
                            _dominated = True  # new is better
                        elif _sbx_primary < _best_refine_primary * 0.5:
                            continue  # skip: regression (>50% worse)
                    # Accept if quality-dominant or richer-with-no-regression
                    if _dominated or len(_sbx_metrics) > len(_best_refine_metrics):
                        _best_refine_metrics = _sbx_metrics
                        _best_refine_stdout = _sbx.get("stdout", "")
                        _best_refine_primary = _sbx_primary
        except (json.JSONDecodeError, OSError):
            pass

    if _best_refine_metrics and len(_best_refine_metrics) > len(metric_lines) // 2:
        # Refinement has richer data — REPLACE Stage 12 data to avoid conflicts
        metric_lines = []
        run_count = 1
        for k, v in _best_refine_metrics.items():
            metric_lines.append(f"  {k}: {v}")
        # Also extract PAIRED and metric lines from stdout
        if _best_refine_stdout:
            for _line in _best_refine_stdout.splitlines():
                _line = _line.strip()
                if _line.startswith("PAIRED:"):
                    metric_lines.append(f"  {_line}")
                elif ":" in _line:
                    parts = _line.rsplit(":", 1)
                    try:
                        float(parts[1].strip())
                        metric_lines.append(f"  {_line}")
                    except (ValueError, TypeError, IndexError):
                        pass
    elif _best_refine_metrics:
        # Refinement has some data but not richer — append to existing
        run_count += 1
        for k, v in _best_refine_metrics.items():
            metric_lines.append(f"  {k}: {v}")
        if _best_refine_stdout:
            for _line in _best_refine_stdout.splitlines():
                _line = _line.strip()
                if _line.startswith("PAIRED:"):
                    metric_lines.append(f"  {_line}")

    if not metric_lines:
        return "", has_parsed_metrics

    # Deduplicate while preserving order
    seen: set[str] = set()
    unique: list[str] = []
    for line in metric_lines:
        if line not in seen:
            seen.add(line)
            unique.append(line)

    # BUG-29: Reformat raw metric lines into human-readable condition summaries
    # to prevent LLM from pasting raw path-style lines into the paper
    _grouped: dict[str, list[str]] = {}
    _ungrouped: list[str] = []
    for line in unique[:200]:
        stripped = line.strip()
        # Match pattern: condition/env/step/metric: value
        parts = stripped.split("/")
        if len(parts) >= 3 and ":" in parts[-1]:
            cond = parts[0]
            detail = "/".join(parts[1:])
            _grouped.setdefault(cond, []).append(f"  - {detail}")
        else:
            _ungrouped.append(stripped)

    formatted_lines: list[str] = []
    if _grouped:
        for cond, details in sorted(_grouped.items()):
            formatted_lines.append(f"## Condition: {cond}")
            formatted_lines.extend(details[:30])
    if _ungrouped:
        formatted_lines.extend(_ungrouped)

    return (
        f"\n\nACTUAL EXPERIMENT DATA (from {run_count} run(s) — use ONLY these numbers):\n"
        "```\n"
        + "\n".join(formatted_lines[:200])
        + "\n```\n"
        "CRITICAL: Every number in the Results table MUST come from the data above. "
        "Do NOT round excessively, do NOT invent numbers, do NOT change values. "
        f"The experiment ran {run_count} time(s) — state this accurately in the methodology.\n"
        "NEVER paste raw metric paths (like 'condition/env/step/metric: value') "
        "into the paper. Always convert to formatted LaTeX tables or inline prose.\n"
    ), has_parsed_metrics


def _write_paper_sections(
    *,
    llm: LLMClient,
    pm: PromptManager,
    run_dir: Path | None = None,
    preamble: str,
    topic_constraint: str,
    exp_metrics_instruction: str,
    citation_instruction: str,
    outline: str,
    model_name: str = "",
) -> str:
    """Write a conference-grade paper in 3 sequential LLM calls.

    Call 1: Title + Abstract + Introduction + Related Work
    Call 2: Method + Experiments (with full experiment data)
    Call 3: Results + Discussion + Limitations + Conclusion

    Each call receives prior sections for coherence.
    """
    # Render writing_structure block for injection
    try:
        _writing_structure = pm.block("writing_structure")
    except (KeyError, Exception):  # noqa: BLE001
        _writing_structure = ""

    _overlay = _get_evolution_overlay(run_dir, "paper_draft")
    system = pm.for_stage(
        "paper_draft",
        evolution_overlay=_overlay,
        preamble=preamble,
        topic_constraint=topic_constraint,
        exp_metrics_instruction=exp_metrics_instruction,
        citation_instruction=citation_instruction,
        writing_structure=_writing_structure,
        outline=outline,
    ).system

    sections: list[str] = []

    # --- R4-3: Title guidelines and abstract structure ---
    try:
        title_guidelines = pm.block("title_guidelines")
    except (KeyError, Exception):  # noqa: BLE001
        title_guidelines = ""
    try:
        abstract_structure = pm.block("abstract_structure")
    except (KeyError, Exception):  # noqa: BLE001
        abstract_structure = ""

    # IMP-20/25/31/24: Academic style, narrative, anti-hedging, anti-repetition
    try:
        academic_style_guide = pm.block("academic_style_guide")
    except (KeyError, Exception):  # noqa: BLE001
        academic_style_guide = ""
    try:
        narrative_writing_rules = pm.block("narrative_writing_rules")
    except (KeyError, Exception):  # noqa: BLE001
        narrative_writing_rules = ""
    try:
        anti_hedging_rules = pm.block("anti_hedging_rules")
    except (KeyError, Exception):  # noqa: BLE001
        anti_hedging_rules = ""
    try:
        anti_repetition_rules = pm.block("anti_repetition_rules")
    except (KeyError, Exception):  # noqa: BLE001
        anti_repetition_rules = ""

    # --- Call 1: Title + Abstract + Introduction + Related Work ---
    call1_user = (
        f"{preamble}\n\n"
        f"{topic_constraint}"
        f"{citation_instruction}\n\n"
        f"{title_guidelines}\n\n"
        f"{academic_style_guide}\n"
        f"{narrative_writing_rules}\n"
        f"{anti_hedging_rules}\n"
        f"{anti_repetition_rules}\n\n"
        "Write the following sections of a NeurIPS/ICML-quality paper in markdown. "
        "Follow the LENGTH REQUIREMENTS strictly:\n\n"
        "1. **Title** (HARD RULE: MUST be 14 words or fewer. Create a catchy method name "
        "first, then build the title: 'MethodName: Subtitle'. If your title exceeds 14 words, "
        "it will be automatically rejected. NEVER use 'Untitled Paper'.)\n"
        f"2. **Abstract** (150-220 words — HARD LIMIT. Do NOT exceed 220 words. "
        f"Do NOT include raw metric paths or 16-digit decimals.){abstract_structure}\n"
        "3. **Introduction** (800-1000 words): real-world motivation, problem statement, "
        "research gap analysis with citations, method overview, 3-4 contributions as bullet points, "
        "paper organization paragraph. MUST cite 8-12 references.\n"
        "4. **Related Work** (600-800 words): organized into 3-4 thematic subsections, each discussing "
        "4-5 papers with proper citations. Compare approaches, identify limitations, position this work.\n\n"
        f"Outline:\n{outline}\n\n"
        "Output markdown with ## headers. Do NOT include a References section.\n"
        "IMPORTANT: Start DIRECTLY with '## Title'. Do NOT include any preamble, "
        "data verification, condition listing, or metric enumeration before the title. "
        "The paper should read like a published manuscript, not a data report."
    )
    # R14-1: Higher token limit for reasoning models
    _paper_max_tokens = 12000
    if any(model_name.startswith(p) for p in ("gpt-5", "o3", "o4")):
        _paper_max_tokens = 24000

    # T3.5: Retry once on failure, use placeholder if still fails
    try:
        resp1 = _chat_with_prompt(llm, system, call1_user, max_tokens=_paper_max_tokens, retries=1)
        part1 = resp1.content.strip()
    except Exception:  # noqa: BLE001
        logger.error("Stage 17: Part 1 LLM call failed after retry — using placeholder")
        part1 = (
            "## Title\n[PLACEHOLDER — LLM call failed]\n\n"
            "## Abstract\n[This section could not be generated due to an LLM error. "
            "Please regenerate this stage.]\n\n"
            "## Introduction\n[PLACEHOLDER]\n\n"
            "## Related Work\n[PLACEHOLDER]"
        )
    sections.append(part1)
    logger.info("Stage 17: Part 1 (Title+Abstract+Intro+Related Work) — %d chars", len(part1))

    # --- Call 2: Method + Experiments ---
    call2_user = (
        f"{preamble}\n\n"
        f"{topic_constraint}"
        f"{exp_metrics_instruction}\n\n"
        f"{narrative_writing_rules}\n"
        f"{anti_hedging_rules}\n\n"
        # IMP-21: Citation instruction for Method + Experiments
        "CITATION REQUIREMENT: The Method section MUST cite at least 3-5 related "
        "technical papers (foundations your method builds on). The Experiments section "
        "MUST cite baseline method papers. Use [cite_key] syntax.\n"
        f"{citation_instruction}\n\n"
        "You are continuing a paper. The sections written so far are:\n\n"
        f"---\n{part1}\n---\n\n"
        "Now write the next sections, maintaining consistency with the above:\n\n"
        "5. **Method** (1000-1500 words): formal problem definition with mathematical notation "
        "($x$, $\\theta$, etc.), detailed algorithm description with equations, step-by-step procedure, "
        "complexity analysis, design rationale for key choices. Include algorithm pseudocode if applicable. "
        "Write as FLOWING PROSE — do NOT use bullet-point lists for method components.\n"
        "6. **Experiments** (800-1200 words): detailed experimental setup, datasets with statistics "
        "(size, splits, features), all baselines and their implementations, hyperparameter settings "
        "in a markdown table, evaluation metrics with mathematical definitions, hardware and runtime info.\n"
        "METHOD NAMES IN TABLES: Use SHORT abbreviations (4-8 chars) for method names "
        "in tables. Define abbreviation mappings in a footnote. "
        "NEVER put method names longer than 20 characters in table cells.\n\n"
        f"Outline:\n{outline}\n\n"
        "Output markdown with ## headers. Continue from where Part 1 ended."
    )
    try:
        resp2 = _chat_with_prompt(llm, system, call2_user, max_tokens=_paper_max_tokens, retries=1)
        part2 = resp2.content.strip()
    except Exception:  # noqa: BLE001
        logger.error("Stage 17: Part 2 LLM call failed after retry — using placeholder")
        part2 = (
            "## Method\n[PLACEHOLDER — LLM call failed. Please regenerate this stage.]\n\n"
            "## Experiments\n[PLACEHOLDER]"
        )
    sections.append(part2)
    logger.info("Stage 17: Part 2 (Method+Experiments) — %d chars", len(part2))

    # --- Call 3: Results + Discussion + Limitations + Conclusion ---
    call3_user = (
        f"{preamble}\n\n"
        f"{topic_constraint}"
        f"{exp_metrics_instruction}\n\n"
        f"{narrative_writing_rules}\n"
        f"{anti_hedging_rules}\n"
        f"{anti_repetition_rules}\n\n"
        # IMP-21: Citation instruction for Results + Discussion + Conclusion
        "CITATION REQUIREMENT: The Discussion section MUST cite at least 3-5 papers "
        "when comparing findings with prior work. The Conclusion may cite 1-2 "
        "foundational references.\n"
        f"{citation_instruction}\n\n"
        "You are completing a paper. The sections written so far are:\n\n"
        f"---\n{part1}\n\n{part2}\n---\n\n"
        "Now write the final sections, maintaining consistency:\n\n"
        "7. **Results** (600-800 words):\n"
        "   - START with an AGGREGATED results table (Table 1): rows = methods, columns = metrics.\n"
        "     Each cell = mean \u00b1 std across seeds. Bold the best value per column.\n"
        "     EVERY table MUST have a descriptive caption that allows understanding without "
        "     reading the main text. NEVER use just 'Table 1' as a caption.\n"
        "   - Follow with a PER-REGIME table (Table 2) breaking down by easy/hard regimes.\n"
        "   - Include a STATISTICAL COMPARISON table (Table 3): paired t-tests between key methods.\n"
        "   - NEVER dump raw per-seed numbers in the main text. Aggregate first, then discuss.\n"
        "   - MUST include at least 2 figures using markdown image syntax: ![Caption](charts/filename.png)\n"
        "     One figure MUST be a performance comparison chart. Figures MUST be referenced "
        "     in text: 'As shown in Figure 1, ...'\n"
        "8. **Discussion** (400-600 words): interpretation of key findings, unexpected results, "
        "comparison with prior work (CITE 3-5 papers here!), practical implications.\n"
        "9. **Limitations** (200-300 words): honest assessment of scope, dataset, methodology. "
        "ALL caveats consolidated HERE — nowhere else in the paper.\n"
        "10. **Conclusion** (100-200 words MAXIMUM — this is a HARD LIMIT): "
        "Summarize contributions in 2-3 sentences. State main finding in 1 sentence. "
        "Suggest 2-3 concrete future directions in 1-2 sentences. "
        "Do NOT repeat any specific numbers from Results. Do NOT restate the abstract. "
        "A good conclusion is SHORT and forward-looking.\n\n"
        "CRITICAL FORMATTING RULES FOR ALL SECTIONS:\n"
        "- Write as FLOWING PROSE paragraphs, NOT bullet-point lists\n"
        "- NEVER dump raw metric paths like 'config/method_name/seed_3/primary_metric'\n"
        "- All numbers must be rounded to 4 decimal places maximum\n"
        "- Every table MUST have a descriptive caption (not just 'Table 1')\n"
        "- Use \\begin{algorithm} or pseudocode notation, NOT \\begin{verbatim}\n\n"
        "Output markdown with ## headers. Do NOT include a References section."
    )
    try:
        resp3 = _chat_with_prompt(llm, system, call3_user, max_tokens=_paper_max_tokens, retries=1)
        part3 = resp3.content.strip()
    except Exception:  # noqa: BLE001
        logger.error("Stage 17: Part 3 LLM call failed after retry — using placeholder")
        part3 = (
            "## Results\n[PLACEHOLDER — LLM call failed. Please regenerate this stage.]\n\n"
            "## Discussion\n[PLACEHOLDER]\n\n"
            "## Limitations\n[PLACEHOLDER]\n\n"
            "## Conclusion\n[PLACEHOLDER]"
        )
    sections.append(part3)
    logger.info("Stage 17: Part 3 (Results+Discussion+Limitations+Conclusion) — %d chars", len(part3))

    # Combine all sections
    draft = "\n\n".join(sections)

    # R32: Strip data verification preamble that LLMs sometimes emit before
    # the actual paper.  The preamble typically starts with "## Tested Conditions"
    # or similar headings and ends before "## Title".
    import re as _re_strip
    _title_match = _re_strip.search(r"^## Title\b", draft, _re_strip.MULTILINE)
    if _title_match and _title_match.start() > 200:
        _stripped = draft[_title_match.start():]
        logger.info(
            "R32: Stripped %d-char preamble before '## Title'",
            _title_match.start(),
        )
        draft = _stripped

    total_words = len(draft.split())
    logger.info("Stage 17: Full draft — %d chars, ~%d words", len(draft), total_words)

    return draft


# ---------------------------------------------------------------------------
# Draft quality validation (section balance + bullet-point density)
# ---------------------------------------------------------------------------

# Sections where bullets/numbered lists are acceptable.
_BULLET_LENIENT_SECTIONS = frozenset({
    "introduction", "limitations", "limitation",
    "limitations and future work", "abstract",
})

# Main body sections used for balance ratio check.
_BALANCE_SECTIONS = frozenset({
    "introduction", "related work", "method", "experiments", "results",
    "discussion",
})


def _validate_draft_quality(
    draft: str,
    stage_dir: Path | None = None,
) -> dict[str, Any]:
    """Validate a paper draft for section balance and prose quality.

    Checks:
    1. Per-section word count vs ``SECTION_WORD_TARGETS``.
    2. Bullet-point / numbered-list density per section.
    3. Largest-to-smallest main-section word-count ratio.

    Returns a dict with ``section_analysis``, ``overall_warnings``, and
    ``revision_directives``.  Optionally writes ``draft_quality.json`` to
    *stage_dir*.
    """
    from researchclaw.prompts import SECTION_WORD_TARGETS, _SECTION_TARGET_ALIASES

    _heading_re = re.compile(r"^(#{1,4})\s+(.+)$", re.MULTILINE)
    matches = list(_heading_re.finditer(draft))

    sections_data: list[dict[str, Any]] = []
    for i, m in enumerate(matches):
        level = len(m.group(1))
        heading = m.group(2).strip()
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(draft)
        body = draft[start:end].strip()
        sections_data.append({
            "heading": heading,
            "heading_lower": heading.strip().lower(),
            "level": level,
            "body": body,
        })

    section_analysis: list[dict[str, Any]] = []
    overall_warnings: list[str] = []
    revision_directives: list[str] = []
    main_section_words: dict[str, int] = {}

    _bullet_re = re.compile(r"^\s*[-*]\s+", re.MULTILINE)
    _numbered_re = re.compile(r"^\s*\d+\.\s+", re.MULTILINE)

    # BUG-24: Accumulate subsection (H3+) word counts into parent H2 sections
    _subsection_words: dict[str, int] = {}
    _current_parent = ""
    for sec in sections_data:
        if sec["level"] <= 2:
            _current_parent = sec["heading_lower"]
            _subsection_words.setdefault(_current_parent, 0)
        else:
            # Add subsection words to parent
            _subsection_words[_current_parent] = (
                _subsection_words.get(_current_parent, 0) + len(sec["body"].split())
            )

    for sec in sections_data:
        if sec["level"] > 2:
            continue
        heading_lower: str = sec["heading_lower"]
        body: str = sec["body"]
        # BUG-24: Include subsection words in the parent's word count
        word_count = len(body.split()) + _subsection_words.get(heading_lower, 0)
        canon = heading_lower
        if canon not in SECTION_WORD_TARGETS:
            canon = _SECTION_TARGET_ALIASES.get(heading_lower, "")
        entry: dict[str, Any] = {
            "heading": sec["heading"],
            "word_count": word_count,
            "canonical": canon,
        }
        if canon and canon in SECTION_WORD_TARGETS:
            lo, hi = SECTION_WORD_TARGETS[canon]
            entry["target"] = [lo, hi]
            if word_count < int(lo * 0.7):
                overall_warnings.append(
                    f"{sec['heading']} is severely under target "
                    f"({word_count} words, target {lo}-{hi})"
                )
                revision_directives.append(
                    f"EXPAND {sec['heading']} from {word_count} to {lo}+ words. "
                    f"Add substantive content \u2014 do NOT pad with filler."
                )
                entry["status"] = "severely_short"
            elif word_count < lo:
                overall_warnings.append(
                    f"{sec['heading']} is under target "
                    f"({word_count} words, target {lo}-{hi})"
                )
                revision_directives.append(
                    f"Expand {sec['heading']} from {word_count} to {lo}+ words."
                )
                entry["status"] = "short"
            elif word_count > int(hi * 1.3):
                overall_warnings.append(
                    f"{sec['heading']} exceeds target "
                    f"({word_count} words, target {lo}-{hi})"
                )
                revision_directives.append(
                    f"Compress {sec['heading']} from {word_count} to {hi} words or fewer."
                )
                entry["status"] = "long"
            else:
                entry["status"] = "ok"
        if body:
            total_lines = len([ln for ln in body.splitlines() if ln.strip()])
            bullet_lines = len(_bullet_re.findall(body)) + len(_numbered_re.findall(body))
            density = bullet_lines / total_lines if total_lines > 0 else 0.0
            entry["bullet_density"] = round(density, 2)
            threshold = 0.50 if heading_lower in _BULLET_LENIENT_SECTIONS else 0.25
            if density > threshold and total_lines >= 4:
                overall_warnings.append(
                    f"{sec['heading']} has {bullet_lines}/{total_lines} "
                    f"bullet/numbered lines ({density:.0%} density, "
                    f"threshold {threshold:.0%})"
                )
                revision_directives.append(
                    f"REWRITE {sec['heading']} as flowing academic prose. "
                    f"Convert bullet points to narrative paragraphs."
                )
                entry["bullet_status"] = "high"
            else:
                entry["bullet_status"] = "ok"
        canon_balance = canon or heading_lower
        if canon_balance in _BALANCE_SECTIONS:
            main_section_words[canon_balance] = word_count
        section_analysis.append(entry)

    if len(main_section_words) >= 2:
        wc_values = list(main_section_words.values())
        max_wc = max(wc_values)
        min_wc = min(wc_values)
        if min_wc > 0 and max_wc / min_wc > 3.0:
            largest = max(main_section_words, key=main_section_words.get)  # type: ignore[arg-type]
            smallest = min(main_section_words, key=main_section_words.get)  # type: ignore[arg-type]
            overall_warnings.append(
                f"Section imbalance: {largest} ({max_wc} words) vs "
                f"{smallest} ({min_wc} words) \u2014 ratio {max_wc / min_wc:.1f}x"
            )
            revision_directives.append(
                f"Rebalance sections: expand {smallest} and/or compress {largest} "
                f"to achieve more even section lengths."
            )

    # --- C-4/C-5: Citation count and recency checks ---
    _cite_pattern = re.compile(r"\[([a-zA-Z][a-zA-Z0-9_-]*\d{4}[a-zA-Z0-9]*)\]")
    cited_keys = set(_cite_pattern.findall(draft))
    if cited_keys:
        n_citations = len(cited_keys)
        if n_citations < 15:
            overall_warnings.append(
                f"Only {n_citations} unique citations found (target: >=15 for a full paper)"
            )
            revision_directives.append(
                f"Add more references — a top-venue paper typically cites 25-40 works. "
                f"Currently only {n_citations} unique citations."
            )
        # Check recency: count citations with year >= current_year - 2
        _year_pat = re.compile(r"(\d{4})")
        import datetime as _dt_cit
        _cur_year = _dt_cit.datetime.now().year
        recent_count = sum(
            1 for k in cited_keys
            for m in [_year_pat.search(k)]
            if m and int(m.group(1)) >= _cur_year - 2
        )
        recency_ratio = recent_count / n_citations if n_citations > 0 else 0.0
        if recency_ratio < 0.3 and n_citations >= 10:
            overall_warnings.append(
                f"Citation recency low: only {recent_count}/{n_citations} "
                f"({recency_ratio:.0%}) from last 3 years (target: >=30%%)"
            )

    # --- Abstract and Conclusion length enforcement ---
    for sec in sections_data:
        hl = sec["heading_lower"]
        body_text: str = sec["body"]
        wc = len(body_text.split())
        if hl == "abstract" and wc > 250:
            overall_warnings.append(
                f"Abstract is too long: {wc} words (target: 150-220 words)"
            )
            revision_directives.append(
                f"COMPRESS the Abstract from {wc} to 150-220 words. "
                f"Remove raw metric values, redundant context, and self-references."
            )
        if hl in ("conclusion", "conclusions", "conclusion and future work"):
            if wc > 300:
                overall_warnings.append(
                    f"Conclusion is too long: {wc} words (target: 100-200 words)"
                )
                revision_directives.append(
                    f"COMPRESS the Conclusion from {wc} to 100-200 words. "
                    f"Do NOT repeat specific metric values from Results. "
                    f"Summarize findings in 2-3 sentences, then 2-3 future directions."
                )

    # --- Raw metric path detection (log dumps in prose) ---
    _raw_path_re = re.compile(
        r"\\texttt\{[a-zA-Z0-9_/.-]+(?:/[a-zA-Z0-9_/.-]+){2,}",
    )
    raw_path_count = len(_raw_path_re.findall(draft))
    if raw_path_count > 3:
        overall_warnings.append(
            f"Raw metric paths in prose: {raw_path_count} instances of "
            f"\\texttt{{config/path/metric}} style dumps"
        )
        revision_directives.append(
            "REMOVE raw experiment log paths from prose. Replace "
            "\\texttt{config/metric/path} with human-readable metric names "
            "and summarize values in tables, not inline text."
        )

    # --- Writing quality lint ---
    _weasel_words = re.compile(
        r"\b(various|many|several|quite|fairly|really|very|rather|"
        r"somewhat|relatively|arguably|interestingly|importantly|"
        r"it is well known that|it is obvious that|clearly)\b",
        re.IGNORECASE,
    )
    _duplicate_words = re.compile(r"\b(\w+)\s+\1\b", re.IGNORECASE)
    weasel_count = len(_weasel_words.findall(draft))
    dup_matches = _duplicate_words.findall(draft)
    dup_count = len([d for d in dup_matches if d.lower() not in ("that", "had")])
    if weasel_count > 20:
        overall_warnings.append(
            f"High weasel-word count: {weasel_count} instances "
            f"(consider replacing vague words with precise language)"
        )
        revision_directives.append(
            "Replace vague hedging words (various, several, quite, fairly, "
            "rather, somewhat) with precise quantities or remove them."
        )
    if dup_count > 0:
        overall_warnings.append(
            f"Duplicate adjacent words found: {dup_count} instance(s) "
            f"(e.g., 'the the', 'is is')"
        )
        revision_directives.append(
            "Fix duplicate adjacent words (likely typos)."
        )

    # --- AI-slop / boilerplate detection ---
    _BOILERPLATE_PHRASES = [
        "delves into", "delve into", "it is worth noting",
        "it should be noted", "it is important to note",
        "leverage the power of", "leverages the power of",
        "in this paper, we propose", "in this work, we propose",
        "to the best of our knowledge",
        "in the realm of", "in the landscape of",
        "plays a crucial role", "plays a pivotal role",
        "groundbreaking", "cutting-edge", "state-of-the-art",
        "game-changing", "paradigm shift",
        "a myriad of", "a plethora of",
        "aims to bridge the gap", "bridge the gap",
        "shed light on", "sheds light on",
        "pave the way", "paves the way",
        "the advent of", "with the advent of",
        "in recent years", "in recent times",
        "has gained significant attention",
        "has attracted considerable interest",
        "has emerged as a promising",
        "a comprehensive overview",
        "a holistic approach", "holistic understanding",
        "showcasing the efficacy", "demonstrate the efficacy",
        "multifaceted", "underscores the importance",
        "navigate the complexities",
        "harness the potential", "harnessing the power",
        "it is imperative to", "it is crucial to",
        "a nuanced understanding", "nuanced approach",
        "robust and scalable", "seamlessly integrates",
        "the intricacies of", "intricate interplay",
        "facilitate a deeper understanding",
        "a testament to",
    ]
    draft_lower = draft.lower()
    boilerplate_hits: list[str] = []
    for phrase in _BOILERPLATE_PHRASES:
        count = draft_lower.count(phrase)
        if count > 0:
            boilerplate_hits.extend([phrase] * count)
    if len(boilerplate_hits) > 5:
        unique_phrases = sorted(set(boilerplate_hits))[:5]
        overall_warnings.append(
            f"AI boilerplate detected: {len(boilerplate_hits)} instances "
            f"of generic LLM phrases (e.g., {', '.join(repr(p) for p in unique_phrases[:3])})"
        )
        revision_directives.append(
            "REWRITE sentences containing AI-generated boilerplate phrases. "
            "Replace generic language (e.g., 'delves into', 'it is worth noting', "
            "'leverages the power of', 'plays a crucial role', 'paves the way') "
            "with precise, specific academic language."
        )

    # --- Related work depth check ---
    _rw_headings = {"related work", "related works", "background", "literature review"}
    rw_body = ""
    for sec in sections_data:
        if sec["heading_lower"] in _rw_headings and sec["level"] <= 2:
            rw_body = sec["body"]
            break
    if rw_body and len(rw_body.split()) > 50:
        _comparative_pats = re.compile(
            r"\b(unlike|in contrast|whereas|while .+ focus|"
            r"however|differ(?:s|ent)|our (?:method|approach) .+ instead|"
            r"we (?:instead|differ)|compared to|as opposed to|"
            r"goes beyond|extends|improves upon|addresses the limitation)\b",
            re.IGNORECASE,
        )
        sentences = [s.strip() for s in re.split(r"[.!?]+", rw_body) if s.strip()]
        comparative_sents = sum(1 for s in sentences if _comparative_pats.search(s))
        ratio = comparative_sents / len(sentences) if sentences else 0.0
        if ratio < 0.15 and len(sentences) >= 5:
            overall_warnings.append(
                f"Related Work is purely descriptive: only {comparative_sents}/{len(sentences)} "
                f"sentences ({ratio:.0%}) contain comparative language (target: >=15%)"
            )
            revision_directives.append(
                "REWRITE Related Work to critically compare with prior methods. "
                "Use phrases like 'unlike X, our approach...', 'in contrast to...', "
                "'while X focuses on... we address...' for at least 20% of sentences."
            )

    # --- Statistical rigor check (result sections) ---
    _results_headings = {"results", "experiments", "experimental results", "evaluation"}
    results_body = ""
    for sec in sections_data:
        if sec["heading_lower"] in _results_headings and sec["level"] <= 2:
            results_body += sec["body"] + "\n"
    if results_body and len(results_body.split()) > 100:
        has_std = bool(re.search(r"\u00b1|\\pm|\bstd\b|\\std\b|standard deviation", results_body, re.IGNORECASE))
        has_ci = bool(re.search(r"confidence interval|\bCI\b|95%|p-value|p\s*<", results_body, re.IGNORECASE))
        has_seeds = bool(re.search(r"(?:seed|run|trial)s?\s*[:=]\s*\d|averaged?\s+over\s+\d+\s+(?:seed|run|trial)", results_body, re.IGNORECASE))
        if not has_std and not has_ci and not has_seeds:
            overall_warnings.append(
                "No statistical measures found in results (no std, CI, p-values, or multi-seed reporting)"
            )
            revision_directives.append(
                "ADD error bars (\u00b1std), confidence intervals, or note the number of "
                "random seeds used. Single-run results without variance reporting "
                "are insufficient for top venues."
            )

    result: dict[str, Any] = {
        "section_analysis": section_analysis,
        "overall_warnings": overall_warnings,
        "revision_directives": revision_directives,
    }
    if stage_dir is not None:
        (stage_dir / "draft_quality.json").write_text(
            json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        if overall_warnings:
            logger.warning(
                "Draft quality: %d warning(s) \u2014 %s",
                len(overall_warnings),
                "; ".join(overall_warnings[:3]),
            )
        else:
            logger.info("Draft quality: all checks passed")
    return result


def _review_compiled_pdf(
    pdf_path: Path,
    llm: LLMClient,
    topic: str,
) -> dict[str, Any]:
    """Multi-dimensional LLM review of compiled paper (AI-Scientist style).

    Scores the paper on 7 academic review dimensions (1-10 each),
    identifies specific strengths/weaknesses, and provides an overall
    accept/reject recommendation with confidence.

    Returns a dict with dimensional scores, issues, and decision.
    """
    if not pdf_path.exists():
        return {}

    # Use source-based review since not all models support vision
    tex_path = pdf_path.with_suffix(".tex")
    if not tex_path.exists():
        return {}

    tex_content = tex_path.read_text(encoding="utf-8")[:12000]

    review_prompt = (
        "You are a senior Area Chair at a top AI conference (NeurIPS/ICML/ICLR) "
        "reviewing a paper submission. Provide a rigorous, structured review.\n\n"
        f"PAPER TOPIC: {topic}\n\n"
        f"LaTeX source:\n```latex\n{tex_content}\n```\n\n"
        "REVIEW INSTRUCTIONS:\n"
        "Score each dimension 1-10 (1=unacceptable, 5=borderline, 8=strong accept, "
        "10=best paper candidate). Be critical but fair.\n\n"
        "DIMENSIONS:\n"
        "1. SOUNDNESS: Are claims well-supported? Is methodology correct? "
        "Are there logical gaps or unsupported claims?\n"
        "2. PRESENTATION: Is the writing clear, flowing, and professional? "
        "Are there grammar errors, bullet lists in prose sections, or "
        "boilerplate phrases? Is it free of AI-generated slop?\n"
        "3. CONTRIBUTION: Is the contribution significant? Does it advance "
        "the field beyond incremental improvement?\n"
        "4. ORIGINALITY: Is the approach novel? Does it differentiate clearly "
        "from prior work?\n"
        "5. CLARITY: Are the method and results easy to understand? Are figures "
        "and tables well-designed with descriptive captions?\n"
        "6. SIGNIFICANCE: Would the community benefit from this work? Does it "
        "open new research directions?\n"
        "7. REPRODUCIBILITY: Are experimental details sufficient to reproduce "
        "results? Are hyperparameters, datasets, and metrics clearly stated?\n\n"
        "Also evaluate:\n"
        "- Are all figures referenced in the text?\n"
        "- Are tables properly formatted (booktabs style, no vertical rules)?\n"
        "- Does the related work critically compare, not just list papers?\n"
        "- Are statistical measures (std, CI, multiple seeds) reported?\n"
        "- Is there a clear limitations section?\n\n"
        "Return a JSON object:\n"
        "{\n"
        '  "soundness": N,\n'
        '  "presentation": N,\n'
        '  "contribution": N,\n'
        '  "originality": N,\n'
        '  "clarity": N,\n'
        '  "significance": N,\n'
        '  "reproducibility": N,\n'
        '  "overall_score": N,\n'
        '  "confidence": N,\n'
        '  "decision": "accept" or "reject",\n'
        '  "strengths": ["strength1", "strength2", ...],\n'
        '  "weaknesses": ["weakness1", "weakness2", ...],\n'
        '  "critical_issues": ["issue requiring revision", ...],\n'
        '  "minor_issues": ["formatting/typo issues", ...],\n'
        '  "summary": "2-3 sentence overall assessment"\n'
        "}\n"
    )

    try:
        resp = llm.chat(
            messages=[{"role": "user", "content": review_prompt}],
            system=(
                "You are a meticulous, critical academic reviewer. "
                "You have reviewed 100+ papers at top venues. "
                "Score honestly — most papers deserve 4-6, not 7-9. "
                "Flag any sign of AI-generated boilerplate."
            ),
        )
        review_data = _safe_json_loads(resp.content, {})
        if isinstance(review_data, dict) and "overall_score" in review_data:
            # Compute weighted aggregate if individual scores present
            dim_scores = {
                k: review_data.get(k, 0)
                for k in (
                    "soundness", "presentation", "contribution",
                    "originality", "clarity", "significance",
                    "reproducibility",
                )
            }
            valid = {k: v for k, v in dim_scores.items() if isinstance(v, (int, float)) and v > 0}
            if valid:
                review_data["mean_score"] = round(sum(valid.values()) / len(valid), 2)
            return review_data
    except Exception as exc:  # noqa: BLE001
        logger.debug("PDF review LLM call failed: %s", exc)

    return {}


def _check_ablation_effectiveness(
    exp_summary: dict[str, Any],
    threshold: float = 0.02,
) -> list[str]:
    """P7: Check if ablation results are within *threshold* of baseline.

    Returns a list of warning strings for ineffective ablations.
    Threshold tightened from 5% to 2% (Improvement C) — ablations with
    < 2% relative difference AND < 1pp absolute difference are flagged
    as TRIVIAL.
    """
    warnings: list[str] = []
    cond_summaries = exp_summary.get("condition_summaries", {})
    if not isinstance(cond_summaries, dict) or not cond_summaries:
        return warnings

    # Find baseline/control condition
    baseline_name = None
    baseline_mean = None
    for name, data in cond_summaries.items():
        if not isinstance(data, dict):
            continue
        name_lower = name.lower()
        if any(tag in name_lower for tag in ("baseline", "control", "vanilla", "standard")):
            metrics = data.get("metrics") or {}
            if not isinstance(metrics, dict):
                metrics = {}
            # Use the first metric that has a _mean suffix or the first available
            for mk, mv in metrics.items():
                if mk.endswith("_mean"):
                    baseline_name = name
                    baseline_mean = float(mv)
                    break
            if baseline_mean is None:
                for mk, mv in metrics.items():
                    try:
                        baseline_name = name
                        baseline_mean = float(mv)
                        break
                    except (TypeError, ValueError):
                        continue
            if baseline_name:
                break

    if baseline_name is None or baseline_mean is None:
        return warnings

    # Check each ablation condition
    for name, data in cond_summaries.items():
        if not isinstance(data, dict):
            continue
        name_lower = name.lower()
        if name == baseline_name:
            continue
        if not any(tag in name_lower for tag in ("ablation", "no_", "without", "reduced")):
            continue
        metrics = data.get("metrics") or {}
        if not isinstance(metrics, dict):
            metrics = {}
        for mk, mv in metrics.items():
            if not mk.endswith("_mean"):
                continue
            try:
                abl_val = float(mv)
            except (TypeError, ValueError):
                continue
            if baseline_mean != 0:
                rel_diff = abs(abl_val - baseline_mean) / abs(baseline_mean)
            else:
                rel_diff = abs(abl_val - baseline_mean)
            abs_diff = abs(abl_val - baseline_mean)
            # Improvement C: Tighter check — both relative < threshold
            # AND absolute < 1pp → TRIVIAL
            if rel_diff < threshold and abs_diff < 1.0:
                warnings.append(
                    f"TRIVIAL: Ablation '{name}' {mk}={abl_val:.4f} is within "
                    f"{rel_diff:.1%} (abs {abs_diff:.4f}pp) of baseline "
                    f"'{baseline_name}' {mk}={baseline_mean:.4f} — "
                    f"ablation is ineffective"
                )
            elif rel_diff < threshold:
                warnings.append(
                    f"Ablation '{name}' {mk}={abl_val:.4f} is within "
                    f"{rel_diff:.1%} of baseline '{baseline_name}' "
                    f"{mk}={baseline_mean:.4f} — ablation may be ineffective"
                )
            break  # Only check the first _mean metric per condition

    # Improvement C: Prepend CRITICAL summary if >50% trivial
    trivial_count = sum(1 for w in warnings if w.startswith("TRIVIAL:"))
    if trivial_count > 0 and len(warnings) > 0 and trivial_count / len(warnings) > 0.5:
        warnings.insert(0, (
            f"CRITICAL: {trivial_count}/{len(warnings)} ablations are trivially "
            f"similar to baseline (<{threshold:.0%} relative, <1pp absolute). "
            f"The ablation design is likely broken — components are not effectively removed."
        ))

    return warnings


def _detect_result_contradictions(
    exp_summary: dict[str, Any],
    metric_direction: str = "maximize",
) -> list[str]:
    """P10: Detect contradictions in experiment results before paper writing.

    Returns a list of advisory strings to inject into paper writing prompt.
    """
    advisories: list[str] = []
    cond_summaries = exp_summary.get("condition_summaries", {})
    if not isinstance(cond_summaries, dict) or not cond_summaries:
        return advisories

    # Collect primary metric means per condition
    means: dict[str, float] = {}
    for name, data in cond_summaries.items():
        if not isinstance(data, dict):
            continue
        metrics = data.get("metrics", {})
        for mk, mv in metrics.items():
            if mk.endswith("_mean"):
                try:
                    means[name] = float(mv)
                except (TypeError, ValueError):
                    pass
                break

    if len(means) < 2:
        return advisories

    # Check 1: All methods within noise margin (2% relative spread)
    vals = list(means.values())
    val_range = max(vals) - min(vals)
    val_mean = sum(vals) / len(vals)
    if val_mean != 0 and (val_range / abs(val_mean)) < 0.02:
        advisories.append(
            "NULL RESULT: All methods produce nearly identical primary metric values "
            f"(range={val_range:.4f}, mean={val_mean:.4f}). Frame this as a null result — "
            "the methods are statistically indistinguishable. Do NOT claim any method "
            "is superior. Discuss possible explanations (task too easy/hard, metric "
            "insensitive, insufficient differentiation in methods)."
        )

    # Check 2: Control/simple baseline outperforms proposed method
    # BUG-P1: Respect metric_direction — "higher is better" vs "lower is better"
    _maximize = metric_direction == "maximize"
    baseline_val = None
    baseline_name = None
    proposed_val = None
    proposed_name = None
    for name, val in means.items():
        name_lower = name.lower()
        if any(tag in name_lower for tag in ("baseline", "control", "random", "vanilla")):
            if baseline_val is None or (_maximize and val > baseline_val) or (not _maximize and val < baseline_val):
                baseline_val = val
                baseline_name = name
        elif any(tag in name_lower for tag in ("proposed", "our", "novel", "method")):
            if proposed_val is None or (_maximize and val > proposed_val) or (not _maximize and val < proposed_val):
                proposed_val = val
                proposed_name = name

    if baseline_val is not None and proposed_val is not None:
        _baseline_wins = (baseline_val > proposed_val) if _maximize else (baseline_val < proposed_val)
        if _baseline_wins:
            advisories.append(
                f"NEGATIVE RESULT: Baseline '{baseline_name}' ({baseline_val:.4f}) "
                f"outperforms proposed method '{proposed_name}' ({proposed_val:.4f}). "
                "This is a NEGATIVE result. Do NOT claim the proposed method is superior. "
                "Frame as 'An Empirical Study of...' or 'When X Falls Short'. "
                "Discuss why the baseline won and what this implies for future work."
            )

    return advisories


def _execute_paper_draft(
    stage_dir: Path,
    run_dir: Path,
    config: RCConfig,
    adapters: AdapterBundle,
    *,
    llm: LLMClient | None = None,
    prompts: PromptManager | None = None,
) -> StageResult:
    outline = _read_prior_artifact(run_dir, "outline.md") or ""
    preamble = _build_context_preamble(
        config,
        run_dir,
        include_goal=True,
        include_hypotheses=True,
        include_analysis=True,
        include_experiment_data=True,  # WS-5.1: inject real experiment data
    )

    # BUG-222: Read PROMOTED BEST experiment_summary for the paper prompt.
    # Previous code (R21-1) picked the "richest" experiment_summary across
    # all stage-14* dirs.  After REFINE regression, a later iteration with
    # more conditions but worse quality could win, feeding the LLM regressed
    # data.  Now: prefer experiment_summary_best.json (written by
    # _promote_best_stage14()), fall back to richest stage-14* for
    # non-REFINE runs.
    exp_summary_text = None
    _best_path = run_dir / "experiment_summary_best.json"
    if _best_path.is_file():
        try:
            _text = _best_path.read_text(encoding="utf-8")
            _parsed = _safe_json_loads(_text, {})
            if isinstance(_parsed, dict) and (
                _parsed.get("condition_summaries") or _parsed.get("metrics_summary")
            ):
                exp_summary_text = _text
                logger.info("BUG-222: Using promoted experiment_summary_best.json")
        except OSError:
            pass
    if exp_summary_text is None:
        # Fallback: pick richest stage-14* (pre-BUG-222 behavior)
        _best_metric_count = 0
        for _s14_dir in sorted(run_dir.glob("stage-14*")):
            _candidate = _s14_dir / "experiment_summary.json"
            if _candidate.is_file():
                _text = _candidate.read_text(encoding="utf-8")
                _parsed = _safe_json_loads(_text, {})
                if isinstance(_parsed, dict):
                    _mcount = _parsed.get("total_metric_keys", 0) or len(
                        _parsed.get("metrics_summary", {})
                    )
                    _paired_count = len(_parsed.get("paired_comparisons", []))
                    _cond_count = len(_parsed.get("condition_summaries", {}))
                    _score = _mcount + _paired_count * 10 + _cond_count * 5
                    if _score > _best_metric_count:
                        _best_metric_count = _score
                        exp_summary_text = _text
                        logger.info(
                            "R21-1 fallback: Selected %s (score=%d)",
                            _s14_dir.name, _score,
                        )
        if exp_summary_text is None:
            exp_summary_text = _read_prior_artifact(run_dir, "experiment_summary.json")
    exp_metrics_instruction = ""
    has_real_metrics = False
    _verified_registry = None  # Phase 1: anti-fabrication verified data registry
    # BUG-108: Load refinement_log so VerifiedRegistry has per-iteration metrics
    _refinement_log_for_vr: dict | None = None
    _rl_candidates = sorted(run_dir.glob("stage-13*/refinement_log.json"), reverse=True)
    _rl_path = _rl_candidates[0] if _rl_candidates else None
    if _rl_path and _rl_path.is_file():
        try:
            _refinement_log_for_vr = json.loads(_rl_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            pass
    if exp_summary_text:
        exp_summary = _safe_json_loads(exp_summary_text, {})
        # Phase 1: Build VerifiedRegistry from experiment data
        if isinstance(exp_summary, dict):
            try:
                from researchclaw.pipeline.verified_registry import VerifiedRegistry
                # BUG-222: Use best_only=True to ensure paper tables reflect
                # only the promoted best iteration, not regressed data
                _verified_registry = VerifiedRegistry.from_run_dir(
                    run_dir,
                    metric_direction=config.experiment.metric_direction,
                    best_only=True,
                )
                logger.info(
                    "Stage 17: VerifiedRegistry — %d verified values, %d conditions",
                    len(_verified_registry.values),
                    len(_verified_registry.condition_names),
                )
            except Exception as _vr_exc:
                logger.warning("Stage 17: Failed to build VerifiedRegistry: %s", _vr_exc)
        if isinstance(exp_summary, dict) and exp_summary.get("metrics_summary"):
            has_real_metrics = True
            exp_metrics_instruction = (
                "\n\nIMPORTANT: Use the ACTUAL experiment results provided in the context. "
                "All numbers in the Results and Experiments sections MUST reference real data. "
                "Do NOT write 'no quantitative results yet' or use placeholder numbers. "
                "Cite specific metrics with their actual values.\n"
            )

    # Collect raw experiment stdout metrics as hard constraint for the paper
    raw_metrics_block, _has_parsed_metrics = _collect_raw_experiment_metrics(run_dir)
    if raw_metrics_block:
        # BUG-23: Raw stdout alone is not sufficient — require either
        # metrics_summary data, parsed metrics from run JSONs,
        # OR at least 3 condition= patterns in raw block
        _has_condition_pattern = len(re.findall(
            r"condition[=:]", raw_metrics_block, re.IGNORECASE
        )) >= 3
        if has_real_metrics or _has_parsed_metrics or _has_condition_pattern:
            has_real_metrics = True
        exp_metrics_instruction += raw_metrics_block

    # R18-1 + R19-6: Inject paired statistical comparisons AND condition summaries
    if exp_summary_text:
        exp_summary_parsed = _safe_json_loads(exp_summary_text, {})
        if isinstance(exp_summary_parsed, dict):
            # R19-6: Inject experiment scale header so LLM knows the data richness
            _total_conds = exp_summary_parsed.get("total_conditions")
            _total_mkeys = exp_summary_parsed.get("total_metric_keys")
            if _total_conds or _total_mkeys:
                scale_block = "\n\n## EXPERIMENT SCALE\n"
                if _total_conds:
                    scale_block += f"- Total conditions tested: {_total_conds}\n"
                if _total_mkeys:
                    scale_block += f"- Total metric keys collected: {_total_mkeys}\n"
                scale_block += (
                    "- This is a MULTI-SEED experiment. Report mean +/- std across seeds.\n"
                    "- Do NOT describe results as 'single run' or 'preliminary'.\n"
                )
                exp_metrics_instruction += scale_block

            # Improvement B: Inject seed insufficiency warnings
            _seed_warns = exp_summary_parsed.get("seed_insufficiency_warnings", [])
            if _seed_warns:
                _sw_block = (
                    "\n\n## SEED INSUFFICIENCY WARNINGS\n"
                    "Some conditions were run with fewer than 3 seeds. "
                    "Results for these conditions MUST be footnoted as preliminary.\n"
                    "All tables MUST show mean ± std format. Single-run values "
                    "MUST be footnoted with '†single seed — interpret with caution'.\n"
                )
                for _sw in _seed_warns:
                    _sw_block += f"- {_sw}\n"
                exp_metrics_instruction += _sw_block

            # R19-6 + R33: Inject condition summaries with CIs
            cond_summaries = exp_summary_parsed.get("condition_summaries", {})
            if isinstance(cond_summaries, dict) and cond_summaries:
                cond_block = "\n\n## PER-CONDITION SUMMARY (use in Results tables)\n"
                for cname, cdata in sorted(cond_summaries.items()):
                    cond_block += f"\n### {cname}\n"
                    if not isinstance(cdata, dict):
                        continue
                    sr = cdata.get("success_rate")
                    if sr is not None:
                        try:
                            cond_block += f"- Success rate: {float(sr):.1%}\n"
                        except (ValueError, TypeError):
                            cond_block += f"- Success rate: {sr}\n"
                    ns = cdata.get("n_seeds") or cdata.get("n_seed_metrics")
                    if ns:
                        cond_block += f"- Seeds: {ns}\n"
                    ci_lo = cdata.get("ci95_low")
                    ci_hi = cdata.get("ci95_high")
                    if ci_lo is not None and ci_hi is not None:
                        try:
                            cond_block += f"- Bootstrap 95% CI: [{float(ci_lo):.4f}, {float(ci_hi):.4f}]\n"
                        except (ValueError, TypeError):
                            cond_block += f"- Bootstrap 95% CI: [{ci_lo}, {ci_hi}]\n"
                    cm = cdata.get("metrics") or {}
                    if isinstance(cm, dict) and cm:
                        for mk, mv in sorted(cm.items()):
                            if isinstance(mv, (int, float)):
                                cond_block += f"- {mk}: {mv:.4f}\n"
                            else:
                                cond_block += f"- {mk}: {mv}\n"
                exp_metrics_instruction += cond_block

            # R18-1: Inject paired statistical comparisons
            paired = exp_summary_parsed.get("paired_comparisons", [])
            if paired:
                paired_block = "\n\n## PAIRED STATISTICAL COMPARISONS (use these in Results)\n"
                paired_block += f"Total: {len(paired)} paired tests computed.\n"
                for pc in paired:
                    if not isinstance(pc, dict):
                        continue
                    method = pc.get("method", "?")
                    baseline = pc.get("baseline", "?")
                    regime = pc.get("regime", "all")
                    md = pc.get("mean_diff", "?")
                    sd = pc.get("std_diff", "?")
                    ts = pc.get("t_stat", "?")
                    pv = pc.get("p_value", "?")
                    ci_lo = pc.get("ci95_low")
                    ci_hi = pc.get("ci95_high")
                    ci_str = ""
                    if ci_lo is not None and ci_hi is not None:
                        try:
                            ci_str = f", 95% CI [{float(ci_lo):.3f}, {float(ci_hi):.3f}]"
                        except (ValueError, TypeError):
                            ci_str = f", 95% CI [{ci_lo}, {ci_hi}]"
                    paired_block += (
                        f"- {method} vs {baseline} (regime={regime}): "
                        f"mean_diff={md}, std_diff={sd}, "
                        f"t={ts}, p={pv}{ci_str}\n"
                    )
                exp_metrics_instruction += paired_block

            # R24: Method naming map — translate generic condition labels
            _cond_names = list(cond_summaries.keys()) if isinstance(cond_summaries, dict) and cond_summaries else []
            if _cond_names:
                naming_block = (
                    "\n\n## METHOD NAMING (CRITICAL — do NOT use generic labels in the paper)\n"
                    "The condition labels below come from the experiment code. In the paper, "
                    "you MUST use DESCRIPTIVE algorithm names, not generic labels.\n"
                    "- If a condition name is already descriptive (e.g., 'random_search', "
                    "'bayesian_optimization', 'ppo_policy'), use it directly as a proper name.\n"
                    "- If a condition name is generic (e.g., 'baseline_1', 'method_variant_1'), "
                    "you MUST infer the algorithm from the experiment code/context and use the "
                    "real algorithm name (e.g., 'Random Search', 'Bayesian Optimization', "
                    "'PPO', 'Curiosity-Driven RL').\n"
                    "- NEVER write `baseline_1` or `method_variant_1` in the paper text.\n"
                    f"- Conditions to name: {_cond_names}\n"
                )
                exp_metrics_instruction += naming_block

            # IMP-8: Inject broken ablation warnings
            abl_warnings = exp_summary_parsed.get("ablation_warnings", [])
            if abl_warnings:
                broken_block = (
                    "\n\n## BROKEN ABLATIONS (DO NOT discuss as valid results)\n"
                    "The following ablation conditions produced IDENTICAL outputs, "
                    "indicating implementation bugs. Do NOT present their differences "
                    "as findings. Mention them ONLY in a 'Limitations' sub-section "
                    "as known implementation issues:\n"
                )
                for _aw in abl_warnings:
                    broken_block += f"- {_aw}\n"
                broken_block += (
                    "\nIf you reference these conditions, state explicitly: "
                    "'Due to an implementation defect, conditions X and Y produced "
                    "identical outputs; their comparison is therefore uninformative.'\n"
                )
                exp_metrics_instruction += broken_block

            # R25: Statistical table format requirement
            if paired:
                stat_table_block = (
                    "\n\n## STATISTICAL TABLE REQUIREMENT (MANDATORY in Results section)\n"
                    "The Results section MUST include a statistical comparison table with columns:\n"
                    "| Comparison | Mean Diff | Std Diff | t-statistic | p-value | Significance |\n"
                    "Use the PAIRED STATISTICAL COMPARISONS data above to fill this table.\n"
                    "Mark significance: *** (p<0.001), ** (p<0.01), * (p<0.05), n.s.\n"
                    "This is non-negotiable — a top-venue paper MUST have statistical tests.\n"
                )
                exp_metrics_instruction += stat_table_block

            # R26: Metric definition requirement
            exp_metrics_instruction += (
                "\n\n## METRIC DEFINITIONS (MANDATORY in Experiments section)\n"
                "The Experiments section MUST define each metric:\n"
                "- **Primary metric**: what it measures, how it is computed, range, direction "
                "(higher/lower is better), and units if applicable.\n"
                "- **Secondary metric**: same details.\n"
                "- For time-to-event metrics: explain the horizon, what constitutes success, "
                "and how failures are handled (e.g., set to max horizon).\n"
                "- These definitions MUST appear BEFORE any results tables.\n"
            )

            # R27: Multi-seed framing enforcement
            _any_seeds = any(
                (cond_summaries.get(c) or {}).get("n_seed_metrics", 0) > 1
                for c in _cond_names
            ) if _cond_names else False
            if _any_seeds:
                exp_metrics_instruction += (
                    "\n\n## MULTI-SEED EXPERIMENT FRAMING (CRITICAL)\n"
                    "This experiment uses MULTIPLE independent random seeds per condition.\n"
                    "- Report mean +/- std (or SE) for all metrics.\n"
                    "- NEVER describe this as 'a single run' or '1 benchmark-artifact run'.\n"
                    "- Frame as: 'We evaluate each method across N seeds per regime.'\n"
                    "- The seed-level data IS the evidence base — it is NOT a single observation.\n"
                    "- Include per-regime breakdowns (easy vs hard) as separate rows in tables.\n"
                )

    # BUG-003: Inject actual evaluated datasets as a hard constraint
    if exp_summary_text:
        _ds_parsed = _safe_json_loads(exp_summary_text, {})
        if isinstance(_ds_parsed, dict):
            _datasets: set[str] = set()
            # Extract from condition names (often contain dataset info)
            for _cname in (_ds_parsed.get("condition_summaries") or {}).keys():
                _datasets.add(str(_cname))
            # Extract from explicit "datasets" field if present
            for _ds in (_ds_parsed.get("datasets") or []):
                if isinstance(_ds, str):
                    _datasets.add(_ds)
            # Extract from "benchmark" or "dataset" fields
            for _key in ("benchmark", "dataset", "dataset_name"):
                _dv = _ds_parsed.get(_key)
                if isinstance(_dv, str) and _dv:
                    _datasets.add(_dv)
            if _datasets:
                exp_metrics_instruction += (
                    "\n\n## ACTUAL EVALUATED DATASETS (HARD CONSTRAINT)\n"
                    "The following datasets/conditions were ACTUALLY tested in experiments:\n"
                    + "".join(f"- {d}\n" for d in sorted(_datasets))
                    + "\nCRITICAL: Do NOT claim evaluation on any dataset not listed above.\n"
                    "Do NOT fabricate results for datasets you did not run experiments on.\n"
                    "If you reference other datasets, clearly state they are 'not evaluated "
                    "in this work' or are 'left for future work'.\n"
                )

    # P7: Ablation effectiveness check
    if exp_summary_text:
        _exp_parsed_p7 = _safe_json_loads(exp_summary_text, {})
        if isinstance(_exp_parsed_p7, dict):
            _abl_warnings = _check_ablation_effectiveness(_exp_parsed_p7)
            if _abl_warnings:
                _abl_block = (
                    "\n\n## ABLATION EFFECTIVENESS WARNINGS\n"
                    "The following ablations showed minimal effect (within 5% of baseline). "
                    "Discuss this honestly — it may indicate the ablated component is not "
                    "important, or the ablation was not properly implemented:\n"
                )
                for _aw in _abl_warnings:
                    _abl_block += f"- {_aw}\n"
                exp_metrics_instruction += _abl_block
                logger.warning("P7: Ablation effectiveness warnings: %s", _abl_warnings)

    # P10: Contradiction detection
    if exp_summary_text:
        _exp_parsed_p10 = _safe_json_loads(exp_summary_text, {})
        if isinstance(_exp_parsed_p10, dict):
            _contradictions = _detect_result_contradictions(
                _exp_parsed_p10, metric_direction=config.experiment.metric_direction
            )
            if _contradictions:
                _contra_block = (
                    "\n\n## RESULT INTERPRETATION ADVISORIES (CRITICAL — read before writing)\n"
                )
                for _ca in _contradictions:
                    _contra_block += f"- {_ca}\n"
                exp_metrics_instruction += _contra_block
                logger.warning("P10: Contradiction advisories: %s", _contradictions)

    # R10: HARD BLOCK — refuse to write paper when all data is simulated
    all_simulated = True
    for stage_subdir in sorted(run_dir.glob("stage-*/runs")):
        for run_file in sorted(stage_subdir.glob("*.json")):
            if run_file.name == "results.json":
                continue
            try:
                _payload = json.loads(run_file.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                continue
            if isinstance(_payload, dict) and _payload.get("status") != "simulated":
                all_simulated = False
                break
        if not all_simulated:
            break

    if all_simulated:
        logger.error(
            "BLOCKED: All experiment data is simulated (mode='simulated'). "
            "Cannot write a paper based on formulaic fake data. "
            "Switch to experiment.mode='sandbox' and re-run."
        )
        (stage_dir / "paper_draft.md").write_text(
            "# Paper Draft Blocked\n\n"
            "**Reason**: All experiment results are from simulated mode "
            "(formulaic data: `0.3 + idx * 0.03`). "
            "These are not real experimental results.\n\n"
            "**Action Required**: Set `experiment.mode: 'sandbox'` in "
            "config.arc.yaml and re-run the pipeline.",
            encoding="utf-8",
        )
        return StageResult(
            stage=Stage.PAPER_DRAFT,
            status=StageStatus.FAILED,
            artifacts=("paper_draft.md",),
            evidence_refs=(),
        )

    # R4-2: HARD BLOCK — refuse to write paper with no real data (ML/empirical domains)
    # For non-empirical domains (math proofs, theoretical economics), allow proceeding
    _domain_id, _domain_name, _domain_venues = _detect_domain(
        config.research.topic, config.research.domains
    )
    _empirical_domains = {"ml", "engineering", "biology", "chemistry"}
    if not has_real_metrics:
        if _domain_id in _empirical_domains:
            logger.error(
                "BLOCKED: Cannot write paper — experiment produced NO metrics. "
                "The pipeline will not fabricate results."
            )
            (stage_dir / "paper_draft.md").write_text(
                "# Paper Draft Blocked\n\n"
                "**Reason**: Experiment stage produced no metrics (status: failed/timeout). "
                "Cannot write a paper without real experimental data.\n\n"
                "**Action Required**: Fix experiment execution or increase time_budget_sec.",
                encoding="utf-8",
            )
            return StageResult(
                stage=Stage.PAPER_DRAFT,
                status=StageStatus.FAILED,
                artifacts=("paper_draft.md",),
                evidence_refs=(),
            )
        else:
            logger.warning(
                "No experiment metrics found, but domain '%s' may be non-empirical "
                "(theoretical/mathematical). Proceeding with paper draft.",
                _domain_name,
            )

    # R11-5: Experiment quality minimum threshold before paper writing
    # Parse analysis.md for quality rating and condition completeness
    analysis_text = _read_best_analysis(run_dir)
    _quality_warnings: list[str] = []

    # Check 1: Was the analysis quality rating very low?
    import re as _re_q
    _rating_match = _re_q.search(
        r"(?:quality\s+rating|result\s+quality)[:\s]*\**(\d+)\s*/\s*10",
        analysis_text,
        _re_q.IGNORECASE,
    )
    if _rating_match:
        _analysis_rating = int(_rating_match.group(1))
        if _analysis_rating <= 3:
            _quality_warnings.append(
                f"Analysis rated experiment quality {_analysis_rating}/10"
            )
        # BUG-23: If quality rating is ≤ 2, force has_real_metrics = False
        # to prevent fabricated results even if stdout had stray numbers.
        # R5-BUG-05: Skip override when _has_parsed_metrics is True — the
        # analysis.md may be stale (from pre-refinement Stage 14) while
        # Stage 13 refinement produced real parsed metrics.
        if _analysis_rating <= 2 and has_real_metrics and not _has_parsed_metrics:
            logger.warning(
                "BUG-23 guard: Analysis quality %d/10 \u2264 2 — "
                "overriding has_real_metrics to False (experiment likely failed)",
                _analysis_rating,
            )
            has_real_metrics = False

    # Check 2: Are baselines missing?
    _analysis_lower = analysis_text.lower()
    if "no" in _analysis_lower and "baseline" in _analysis_lower:
        if any(phrase in _analysis_lower for phrase in [
            "no baseline", "no bo", "no random", "baselines are missing",
            "missing baselines", "baseline coverage is missing",
        ]):
            _quality_warnings.append("Baselines appear to be missing from results")

    # Check 3: Is the metric undefined?
    if any(phrase in _analysis_lower for phrase in [
        "metric is undefined", "primary_metric is undefined",
        "undefined metric", "metric undefined",
    ]):
        _quality_warnings.append("Primary metric is undefined (direction/units/formula unknown)")

    # Check 4: Very few conditions completed
    _condition_count = len(_re_q.findall(
        r"condition[=:\s]+\w+.*?(?:mean|primary_metric)",
        raw_metrics_block or "",
        _re_q.IGNORECASE,
    ))

    if _quality_warnings:
        _warning_block = "\n".join(f"  - {w}" for w in _quality_warnings)
        logger.warning(
            "Stage 17: Experiment quality concerns detected before paper writing:\n%s",
            _warning_block,
        )
        # Inject quality warnings into the paper writing prompt so the LLM
        # writes an appropriately hedged paper
        exp_metrics_instruction += (
            "\n\n## EXPERIMENT QUALITY WARNINGS (address these honestly in the paper)\n"
            + "\n".join(f"- {w}" for w in _quality_warnings)
            + "\n\nBecause of these issues, the paper MUST:\n"
            "- Use hedged language ('preliminary', 'pilot', 'initial exploration')\n"
            "- NOT claim definitive comparisons between methods\n"
            "- Dedicate a substantial Limitations section to these gaps\n"
            "- Frame the contribution as methodology/framework, not empirical findings\n"
        )
        # Save warnings for tracking
        (stage_dir / "quality_warnings.json").write_text(
            json.dumps(_quality_warnings, indent=2), encoding="utf-8"
        )

    # Phase 1: Inject pre-built results tables from VerifiedRegistry
    if _verified_registry is not None:
        try:
            from researchclaw.templates.results_table_builder import (
                build_results_tables,
                build_condition_whitelist,
            )
            _prebuilt_tables = build_results_tables(
                _verified_registry,
                metric_direction=_verified_registry.metric_direction,
            )
            _condition_whitelist = build_condition_whitelist(_verified_registry)
            if _prebuilt_tables:
                _tables_block = "\n\n".join(t.latex_code for t in _prebuilt_tables)
                exp_metrics_instruction += (
                    "\n\n## PRE-BUILT RESULTS TABLES (MANDATORY — copy verbatim)\n"
                    "The tables below were AUTO-GENERATED from verified experiment data.\n"
                    "You MUST include these tables in the Results section EXACTLY as shown.\n"
                    "Do NOT modify any numbers. Do NOT add rows with fabricated data.\n"
                    "You MAY adjust formatting (bold, alignment) but NOT numerical values.\n\n"
                    + _tables_block
                )
                logger.info("Stage 17: Injected pre-built results tables into prompt")
            if _condition_whitelist:
                exp_metrics_instruction += (
                    "\n\n## VERIFIED CONDITIONS (ONLY mention these in the paper)\n"
                    + _condition_whitelist
                    + "\nDo NOT discuss conditions not in this list. Do NOT invent new conditions.\n"
                )
        except Exception as _tb_exc:
            logger.warning("Stage 17: Failed to build pre-built tables: %s", _tb_exc)

    # R4-2: Anti-fabrication data integrity instruction
    exp_metrics_instruction += (
        "\n\n## CRITICAL: Data Integrity Rules\n"
        "- You may ONLY report numbers that appear in the experiment data above\n"
        "- If the experiment data is incomplete (fewer conditions than planned), report\n"
        "  ONLY the conditions that were actually run\n"
        "- Do NOT extrapolate, interpolate, or 'fill in' missing cells in tables\n"
        "- Do NOT invent confidence intervals, p-values, or statistical tests unless\n"
        "  the actual data supports them\n"
        "- If only N conditions completed, simply report results for those N conditions\n"
        "  without repeating apologies or disclaimers about missing conditions\n"
        "- Any table cell without real data must show '\u2014' (not a plausible number)\n"
        "- FORBIDDEN: generating numbers that 'look right' based on your training data\n"
    )

    # IMP-6 + FA: Inject chart references into paper draft prompt
    # Prefer FigureAgent's figure_plan.json (rich descriptions) over raw file scan
    # BUG-FIX: figure_plan.json may be a list (from FigureAgent planner) or a dict
    # (from executor overwrite).  The orchestrator writes a list at planning time;
    # the executor overwrites with a dict only when figure_count > 0.  If the
    # FigureAgent renders 0 charts the list persists, and calling .get() on it
    # raises AttributeError.
    _fa_descriptions = ""
    # BUG-178: Iterate in reverse order so we read the LATEST stage-14
    # iteration's figure plan, matching Stage 22 which copies charts
    # from the newest iteration.
    for _s14_dir in sorted(run_dir.glob("stage-14*"), reverse=True):
        # Prefer the final plan (dict with figure_descriptions) if it exists
        for _fp_name in ("figure_plan_final.json", "figure_plan.json"):
            _fp_path = _s14_dir / _fp_name
            if not _fp_path.exists():
                continue
            try:
                _fp_data = json.loads(_fp_path.read_text(encoding="utf-8"))
                if isinstance(_fp_data, dict):
                    _fa_descriptions = _fp_data.get("figure_descriptions", "")
                elif isinstance(_fp_data, list) and _fp_data:
                    # List format from FigureAgent planner — synthesize descriptions
                    _desc_parts = ["## PLANNED FIGURES (from figure plan)\n"]
                    for _fig in _fp_data:
                        if isinstance(_fig, dict):
                            _fid = _fig.get("figure_id", "unnamed")
                            _ftitle = _fig.get("title", "")
                            _fcap = _fig.get("caption", "")
                            _fsec = _fig.get("section", "results")
                            _desc_parts.append(
                                f"- **{_fid}** ({_fsec}): {_ftitle}\n  {_fcap}"
                            )
                    if len(_desc_parts) > 1:
                        _fa_descriptions = "\n".join(_desc_parts)
            except (json.JSONDecodeError, OSError):
                pass
            if _fa_descriptions:
                break
        if _fa_descriptions:
            break

    if _fa_descriptions:
        exp_metrics_instruction += "\n\n" + _fa_descriptions
        logger.info("Stage 17: Injected FigureAgent figure descriptions into paper draft prompt")
    else:
        # Fallback: scan for chart files from the LATEST stage-14 iteration
        # BUG-178: Must use reverse order to match Stage 22 chart copy behavior
        _chart_files: list[str] = []
        for _s14_dir in sorted(run_dir.glob("stage-14*"), reverse=True):
            _charts_path = _s14_dir / "charts"
            if _charts_path.is_dir():
                _found = sorted(_charts_path.glob("*.png"))
                if _found:
                    _chart_files = [f.name for f in _found]
                    break  # Use only the latest iteration's charts
        if _chart_files:
            _chart_block = (
                "\n\n## AVAILABLE FIGURES (embed in the paper)\n"
                "The following figures were generated from actual experiment data. "
                "You MUST reference at least 1-2 of these in the Results section "
                "using markdown image syntax: `![Caption](charts/filename.png)`\n\n"
            )
            for _cf_name in _chart_files:
                _label = _cf_name.replace("_", " ").replace(".png", "").title()
                _chart_block += f"- `charts/{_cf_name}` \u2014 {_label}\n"
            _chart_block += (
                "\nFor each figure referenced, write a descriptive caption and "
                "discuss what the figure shows in 2-3 sentences.\n"
            )
            exp_metrics_instruction += _chart_block
            logger.info(
                "Stage 17: Injected %d chart references into paper draft prompt",
                len(_chart_files),
            )

    # WS-5.5: Framework diagram placeholder instruction
    exp_metrics_instruction += (
        "\n\n## FRAMEWORK DIAGRAM PLACEHOLDER\n"
        "In the Method/Approach section, include a placeholder for the methodology "
        "framework overview figure. Insert this exactly:\n\n"
        "```\n"
        "![Framework Overview](charts/framework_diagram.png)\n"
        "**Figure N.** Overview of the proposed methodology. "
        "[A detailed framework diagram will be generated separately and inserted here.]\n"
        "```\n\n"
        "This figure should be referenced in the text as 'Figure N' and discussed briefly "
        "(1-2 sentences describing the overall pipeline/architecture flow). "
        "The actual image will be generated post-hoc using a text-to-image model.\n"
    )

    # P5: Extract hyperparameters from results.json for paper Method section
    _hp_table = ""
    for _s14_dir in sorted(run_dir.glob("stage-14*")):
        for _run_file in sorted(_s14_dir.glob("runs/*.json")):
            try:
                _run_data = json.loads(_run_file.read_text(encoding="utf-8"))
                if isinstance(_run_data, dict) and _run_data.get("hyperparameters"):
                    _hp = _run_data["hyperparameters"]
                    if isinstance(_hp, dict) and _hp:
                        _hp_table = "\n\n## HYPERPARAMETERS (include as a table in the Method section)\n"
                        _hp_table += "| Hyperparameter | Value |\n|---|---|\n"
                        for _hk, _hv in sorted(_hp.items()):
                            _hp_table += f"| {_hk} | {_hv} |\n"
                        _hp_table += (
                            "\nThis table MUST appear in the Method/Experiments section. "
                            "Include ALL hyperparameters used, with justification for key choices.\n"
                        )
                        break
            except (json.JSONDecodeError, OSError):
                continue
        if _hp_table:
            break
    # Also check staging dirs for results.json
    if not _hp_table:
        for _staging_dir in sorted(run_dir.glob("stage-*/runs/_docker_*")):
            _rjson = _staging_dir / "results.json"
            if _rjson.is_file():
                try:
                    _rdata = json.loads(_rjson.read_text(encoding="utf-8"))
                    if isinstance(_rdata, dict) and _rdata.get("hyperparameters"):
                        _hp = _rdata["hyperparameters"]
                        if isinstance(_hp, dict) and _hp:
                            _hp_table = "\n\n## HYPERPARAMETERS (include as a table in the Method section)\n"
                            _hp_table += "| Hyperparameter | Value |\n|---|---|\n"
                            for _hk, _hv in sorted(_hp.items()):
                                _hp_table += f"| {_hk} | {_hv} |\n"
                            _hp_table += (
                                "\nThis table MUST appear in the Method/Experiments section. "
                                "Include ALL hyperparameters used, with justification for key choices.\n"
                            )
                            break
                except (json.JSONDecodeError, OSError):
                    continue
    if _hp_table:
        exp_metrics_instruction += _hp_table

    # F2.6: Build citation list from references.bib / candidates with cite_keys
    citation_instruction = ""
    bib_text = _read_prior_artifact(run_dir, "references.bib")

    # P3: Pre-verify citations before paper draft — remove hallucinated refs
    if bib_text and bib_text.strip():
        from researchclaw.literature.verify import (
            filter_verified_bibtex,
            verify_citations as _verify_cit,
        )
        try:
            _pre_report = _verify_cit(bib_text, inter_verify_delay=0.5)
            _kept = _pre_report.verified + _pre_report.suspicious
            _removed = _pre_report.hallucinated
            if _removed > 0:
                bib_text = filter_verified_bibtex(
                    bib_text, _pre_report, include_suspicious=True
                )
                (stage_dir / "references_preverified.bib").write_text(
                    bib_text, encoding="utf-8"
                )
                logger.info(
                    "P3: Pre-verification kept %d/%d citations (removed %d hallucinated)",
                    _kept, _pre_report.total, _removed,
                )
        except Exception as exc:
            logger.warning("P3: Pre-verification failed, using original bib: %s", exc)

    candidates_text = _read_prior_artifact(run_dir, "candidates.jsonl")
    if candidates_text:
        cite_lines: list[str] = []
        for row_text in candidates_text.strip().splitlines():
            row = _safe_json_loads(row_text, {})
            if isinstance(row, dict) and row.get("cite_key"):
                authors_info = ""
                if isinstance(row.get("authors"), list) and row["authors"]:
                    first_author = row["authors"][0]
                    if isinstance(first_author, dict):
                        # BUG-38: name may be non-str (tuple/list) — force str
                        _name = first_author.get("name", "")
                        authors_info = _name if isinstance(_name, str) else str(_name)
                    elif isinstance(first_author, str):
                        authors_info = first_author
                    if len(row["authors"]) > 1:
                        authors_info += " et al."
                title = row.get("title", "")
                cite_lines.append(
                    f"- [{row['cite_key']}] \u2192 TITLE: \"{title}\" "
                    f"| {authors_info} "
                    f"({row.get('venue', '')}, {row.get('year', '')}, "
                    f"cited {row.get('citation_count', 0)} times) "
                    f"| ONLY cite this key when discussing: {title}"
                )
        if cite_lines:
            citation_instruction = (
                "\n\nAVAILABLE REFERENCES (use [cite_key] to cite in the text):\n"
                + "\n".join(cite_lines)
                + "\n\nCRITICAL CITATION RULES:\n"
                "- In the body text, cite using [cite_key] format, e.g. [smith2024transformer].\n"
                "- Do NOT write a References section \u2014 it will be auto-generated from the bibliography file.\n"
                "- Do NOT invent any references or arXiv IDs not in the above list.\n"
                "- You may cite a subset, but NEVER fabricate citations or change arXiv IDs.\n"
                "- SEMANTIC MATCHING: Before citing a reference, verify that its TITLE matches\n"
                "  the concept you are discussing. Do NOT use an unrelated cite_key just\n"
                "  because it sounds similar.\n"
                "- If no reference in the list matches the concept you want to cite,\n"
                "  write 'prior work has shown...' WITHOUT a citation, rather than using\n"
                "  a mismatched reference.\n"
                "- Each [cite_key] MUST correspond to the paper whose title is shown\n"
                "  next to that key in the list above. Cross-check before citing.\n"
                "\nCITATION QUANTITY & QUALITY CONSTRAINTS:\n"
                "- Cite 25-40 unique references in the paper body. The Related Work\n"
                "  section alone should cite at least 15 references.\n"
                "- Every citation MUST be directly relevant to the paper's topic.\n"
                "- DO NOT cite papers from unrelated domains (wireless communication, "
                "manufacturing, UAV, etc.).\n"
                "- Prefer well-known, highly-cited papers over obscure ones.\n"
                "- If unsure whether a paper exists or is relevant, DO NOT cite it.\n"
            )

    if llm is not None:
        _pm = prompts or PromptManager()
        topic_constraint = _pm.block("topic_constraint", topic=config.research.topic)

        # --- Section-by-section writing (3 calls) for conference-grade depth ---
        draft = _write_paper_sections(
            llm=llm,
            pm=_pm,
            run_dir=run_dir,
            preamble=preamble,
            topic_constraint=topic_constraint,
            exp_metrics_instruction=exp_metrics_instruction,
            citation_instruction=citation_instruction,
            outline=outline,
            model_name=config.llm.primary_model,
        )

        # R7: Strip LLM-generated References section — it often fabricates arXiv IDs.
        import re as _re_r7
        ref_pattern = _re_r7.compile(
            r'^(#{1,2}\s*References.*)', _re_r7.MULTILINE | _re_r7.DOTALL
        )
        ref_match = ref_pattern.search(draft)
        if ref_match:
            draft = draft[:ref_match.start()].rstrip()
            logger.info("Stage 17: Stripped LLM-generated References section (R7 fix)")
    else:
        # Build template with real data if available
        results_section = "Template results summary."
        if exp_summary_text:
            exp_summary = _safe_json_loads(exp_summary_text, {})
            if isinstance(exp_summary, dict) and exp_summary.get("metrics_summary"):
                lines = ["Experiment results:"]
                for mk, mv in exp_summary["metrics_summary"].items():
                    if isinstance(mv, dict):
                        lines.append(
                            f"- {mk}: mean={mv.get('mean')}, min={mv.get('min')}, "
                            f"max={mv.get('max')}, n={mv.get('count')}"
                        )
                results_section = "\n".join(lines)

        draft = f"""# Draft Title

## Abstract
Template draft abstract.

## Introduction
Template introduction for {config.research.topic}.

## Related Work
Template related work.

## Method
Template method description.

## Experiments
Template experimental setup.

## Results
{results_section}

## Limitations
Template limitations.

## Conclusion
Template conclusion.

## References
Template references.

Generated: {_utcnow_iso()}
"""
    (stage_dir / "paper_draft.md").write_text(draft, encoding="utf-8")

    # Validate draft quality (section balance + bullet density)
    _validate_draft_quality(draft, stage_dir=stage_dir)

    return StageResult(
        stage=Stage.PAPER_DRAFT,
        status=StageStatus.DONE,
        artifacts=("paper_draft.md",),
        evidence_refs=("stage-17/paper_draft.md",),
    )
