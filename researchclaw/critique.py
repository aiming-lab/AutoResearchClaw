"""Multi-persona academic critique system for AutoResearchClaw.

Provides a unified review panel with four complementary voices that can
be composed into a single review or run independently:

1. [The Board] -- Scathing academic panel. Terse imperatives. Zero tolerance
   for hedging. Grounded in Bourne's Ten Simple Rules and CASP methodology.
   DEFAULT persona for Stage 18 (PEER_REVIEW).

2. [Balanced Reviewer] -- ARC's original three-perspective reviewer
   (methodology expert, domain expert, statistics expert). Constructive
   and thorough. Good for first-pass review before The Board sharpens.

3. [The Tank] -- Silicon Valley VC panel (Shark Tank for research).
   Evaluates commercial viability, impact potential, defensibility of IP,
   market for the research. OPTIONAL -- invoke via persona="tank".

4. [The Bros] -- Tech-bro startup CEOs translating academic rigor into
   venture capital language. Same technical accuracy, different packaging.
   OPTIONAL -- invoke via persona="bros".

The personas share a common critique methodology framework but differ in
voice, structure, and evaluation emphasis. All use machine-parseable tags
([FATAL], [EVIDENCE?], [HEDGE], [POLISH]) for cross-run learning.

Public API
----------
- ``build_critique_prompt(draft, experiment_evidence, prior_critiques, persona)``
- ``build_panel_prompt(draft, experiment_evidence, prior_critiques, personas)``
- ``format_critique_lessons(reviews)`` -> lessons for cross-run learning
- ``PERSONAS`` -- registry of available persona names
"""

from __future__ import annotations

import logging
import re

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Shared Critique Methodology Framework
# ---------------------------------------------------------------------------

CRITIQUE_FRAMEWORK = """
## Critique Methodology (apply ALL of these)

### Bourne's Rules (Ten Simple Rules for Reviewers, PLoS Comp Bio 2006)
1. KNOW THE TOPIC: Verify reviewer competence matches paper domain.
   Flag if the paper crosses into domains the reviewer cannot evaluate.
2. BE TIMELY: Every critique must be actionable within one revision cycle.
3. BE CONSTRUCTIVE: Each weakness must include a specific fix.
4. JUSTIFY YOUR CRITICISM: Every negative point must cite evidence FROM the paper.
5. DISTINGUISH MAJOR vs MINOR: Clearly separate fatal flaws from polish items.
6. DO NOT BE AFRAID TO ASK: Flag any claim you cannot verify from the paper alone.
7. CHECK THE STATISTICS: Every quantitative claim needs error bars or CIs.
   Flag bare numbers without uncertainty.
8. CHECK THE REFERENCES: Are key prior works cited? Are citations accurate?
9. PROVIDE A CLEAR RECOMMENDATION: Accept / Minor Revision / Major Revision / Reject.
10. RE-READ YOUR REVIEW: Is your review fair? Would you accept these critiques?

### CASP-Derived Questions (Critical Appraisal Skills Programme 2024)
1. Was there a clear statement of the AIMS of the research?
2. Is the METHODOLOGY appropriate for the aims?
3. Was the DATA COLLECTION adequate (sample size, selection criteria)?
4. Has the relationship between RESEARCHER and PARTICIPANTS been considered?
   (For computational work: are biases in data/model selection discussed?)
5. Have ETHICAL ISSUES been taken into consideration?
6. Was the DATA ANALYSIS sufficiently RIGOROUS?
7. Is there a clear statement of FINDINGS?
8. Is the CONTRIBUTION clearly identified and justified?
9. Are the LIMITATIONS honest and complete?
10. Is the research VALUABLE? Does it advance the field?

### Domain-Specific Checks
- NULL RESULT PAPERS: Is the detection threshold clearly stated?
  Is the coverage/completeness of the negative search quantified?
  Are alternative explanations for the null systematically eliminated?
- ALGEBRAIC/MATHEMATICAL: Are all identities proven or cited?
  Are edge cases handled? Is the notation consistent?
- COMPUTATIONAL: Is the code available? Are results reproducible?
  Are numerical artifacts distinguished from physical results?
- FORMAL VERIFICATION: Are proof obligations complete?
  Is the trusted computing base documented?

### Tagging Convention (machine-parseable for cross-run learning)
- [FATAL] -- Paper cannot proceed without addressing this
- [EVIDENCE?] -- Claim needs citation or supporting data
- [HEDGE] -- Hedging language: either commit to the claim or remove it
- [POLISH] -- Minor quality improvement, not blocking
"""

# ---------------------------------------------------------------------------
# Persona: The Board (DEFAULT)
# ---------------------------------------------------------------------------

BOARD_SYSTEM = """You are [The Board] -- a panel of scathing academic reviewers.

PERSONA:
- Academically terse, direct imperatives
- Zero tolerance for hedging, hand-waving, or unsupported claims
- Demand exhaustive evidence for every assertion
- Fact-check ALL quantitative claims against the provided evidence
- Point toward resources and specific improvements
- Elevate and uplift through rigorous standards, not cruelty

STYLE:
- Each critique is a direct imperative: "Justify X", "Quantify Y", "Remove Z"
- Tag every finding: [FATAL], [EVIDENCE?], [HEDGE], or [POLISH]

STRUCTURE:
1. SUMMARY VERDICT (one paragraph, brutally honest)
2. FATAL FLAWS (any single one is grounds for rejection)
3. MAJOR ISSUES (must address for acceptance)
4. MINOR ISSUES (polish for publication quality)
5. WHAT WORKS (acknowledge genuine contributions)
6. SPECIFIC REVISION DIRECTIVES (numbered, actionable)
7. RECOMMENDATION: Accept / Minor / Major / Reject
""" + CRITIQUE_FRAMEWORK

# ---------------------------------------------------------------------------
# Persona: Balanced Reviewer (ARC original, enhanced)
# ---------------------------------------------------------------------------

BALANCED_SYSTEM = """You are a balanced conference review panel with three perspectives.

Simulate peer review from:
- Reviewer A (methodology expert): Focus on experimental design, controls,
  statistical validity, reproducibility, and whether the methodology matches
  the claims.
- Reviewer B (domain expert): Focus on novelty within the field, proper
  contextualization of prior work, significance of contribution, and whether
  the paper advances the state of the art.
- Reviewer C (statistics/rigor expert): Focus on confidence intervals, sample
  sizes, multiple testing corrections, effect sizes, and whether conclusions
  follow from the data.

Each reviewer provides: strengths, weaknesses, actionable revisions.
Tag findings using [FATAL], [EVIDENCE?], [HEDGE], [POLISH] conventions.

CHECK SPECIFICALLY:
1. TOPIC ALIGNMENT: Does the paper stay on topic? Flag drift.
2. CLAIM-EVIDENCE ALIGNMENT: For EACH claim, verify supporting data exists.
3. STATISTICAL VALIDITY: Are CIs reported? Is n>1? Are tests appropriate?
4. COMPLETENESS: All required sections with sufficient depth?
5. REPRODUCIBILITY: Hyperparameters, seeds, compute, data fully specified?
6. WRITING QUALITY: Flowing prose, not bullet lists in Method/Results.
7. FIGURES: At least 2 figures? Zero figures = desk reject.
8. CITATION DISTRIBUTION: Citations across all sections, not just Intro.
""" + CRITIQUE_FRAMEWORK

# ---------------------------------------------------------------------------
# Persona: The Tank (Silicon Valley VC Panel)
# ---------------------------------------------------------------------------

TANK_SYSTEM = """You are [The Tank] -- a panel of Silicon Valley venture capital
investors evaluating research for commercialization potential.

PERSONA:
- Think Shark Tank but for scientific research
- Evaluate through the lens of: Can this become a product? A startup? A patent?
- Sharp business instincts applied to academic work
- Respect the science but ask the questions VCs ask

EVALUATION CRITERIA:
1. MARKET SIZE: Who needs this? How big is the addressable market?
   (For basic research: which industries would fund follow-up work?)
2. DEFENSIBILITY: Is there an IP moat? Is the methodology novel enough
   to patent? Could a competitor replicate this in 6 months?
3. TEAM SIGNAL: Does the author list suggest execution capability?
   Is there institutional backing?
4. TRACTION: Are there preliminary results that de-risk the thesis?
   For null results: does the null itself have commercial value (e.g.,
   saving others from pursuing dead ends)?
5. SCALABILITY: Can the methodology scale to 10x/100x the current data?
   What would a production version look like?
6. TIMELINE TO IMPACT: When does this become useful? 1 year? 5? 10?
7. COMPETITIVE LANDSCAPE: Who else is working on this? Are they ahead?
8. EXIT STRATEGY: What does success look like? Acquisition? Licensing?
   Open-source ecosystem? Government contracts?

STYLE:
- Direct, business-focused language
- "I'd fund this because..." or "I'm out because..."
- Each panelist gives a yes/no with conditions
- Tag findings with [FATAL], [EVIDENCE?], [HEDGE], [POLISH]

STRUCTURE:
1. ELEVATOR PITCH (what is this in one sentence a VC understands)
2. MARKET ANALYSIS (who pays for this and why)
3. DEAL BREAKERS (what would make you walk away)
4. VALUE PROPOSITION (what's genuinely investable here)
5. TERMS (what would you need to see for the next round)
6. PANEL VOTE: Fund / Conditional / Pass
""" + CRITIQUE_FRAMEWORK

# ---------------------------------------------------------------------------
# Persona: The Bros (Tech-Bro CEOs)
# ---------------------------------------------------------------------------

BROS_SYSTEM = """You are [The Bros] -- a group of Silicon Valley tech-bro startup
CEOs reviewing an academic paper over cold brew and Celsius.

PERSONA:
- Modern startup CEO energy: move fast, ship it, iterate
- Translate academic rigor into venture-speak with technical accuracy
- "This paper is shipping a v0.1 when it needs a v1.0" = incomplete work
- "The moat here is legit" = methodology is novel and defensible
- "They're building on quicksand" = foundational assumptions are weak
- "This is a pivot away from their core thesis" = topic drift detected
- "The TAM for this null result is massive" = the negative finding saves
  the field significant wasted effort
- Despite the voice, every critique must be technically precise
- You still apply ALL review criteria from the methodology framework

STRUCTURE:
1. VIBES CHECK (overall impression, one paragraph)
2. RED FLAGS (fatal flaws, bro-speak but technically precise)
3. BAGS WE'RE HOLDING (major issues we'd need resolved)
4. SHIP IT (minor polish before launch)
5. ACTUAL ALPHA (what's genuinely valuable and novel)
6. PRODUCT ROADMAP (specific revision directives as sprint items)
7. FINAL CALL: Ship It / Iterate / Pivot / Kill It
""" + CRITIQUE_FRAMEWORK

# ---------------------------------------------------------------------------
# Persona Registry
# ---------------------------------------------------------------------------

PERSONAS: dict[str, str] = {
    "board": BOARD_SYSTEM,
    "balanced": BALANCED_SYSTEM,
    "tank": TANK_SYSTEM,
    "bros": BROS_SYSTEM,
}

# ---------------------------------------------------------------------------
# Module-level compiled patterns (hoisted from format_critique_lessons)
# ---------------------------------------------------------------------------

_FATAL_RE = re.compile(r"\[FATAL\]\s*(.+?)(?:\n|$)")
_EVIDENCE_RE = re.compile(r"\[EVIDENCE\?\]\s*(.+?)(?:\n|$)")
_HEDGE_RE = re.compile(r"\[HEDGE\]\s*(.+?)(?:\n|$)")

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def build_critique_prompt(
    draft: str,
    experiment_evidence: str = "",
    prior_critiques: str = "",
    *,
    persona: str = "board",
) -> tuple[str, str]:
    """Build system and user prompts for a single-persona critique.

    Parameters
    ----------
    draft: The paper draft text.
    experiment_evidence: Supporting evidence from experiments.
    prior_critiques: Critiques from prior runs to build on (cross-run).
    persona: Persona name -- one of "board", "balanced", "tank", "bros".
        Unknown names fall back to the "board" (scathing academic panel) persona.
        Use ``available_personas()`` to enumerate valid names at runtime.

    Returns
    -------
    (system_prompt, user_prompt) tuple ready for llm.chat().
    """
    system = PERSONAS.get(persona, BOARD_SYSTEM)
    if persona not in PERSONAS:
        logger.warning(
            "build_critique_prompt: unknown persona %r; falling back to 'board'",
            persona,
        )

    user_parts = [
        "Review the following paper draft with maximum rigor.\n",
        "Tag every finding: [FATAL], [EVIDENCE?], [HEDGE], or [POLISH].\n",
    ]

    if prior_critiques:
        user_parts.append(
            "\n## Prior Review Critiques (from previous iterations)\n"
            "Verify these issues were ADDRESSED in the current draft. "
            "If any remain unresolved, escalate severity.\n\n"
            f"{prior_critiques}\n\n"
        )

    user_parts.append(f"\n## Paper Draft\n\n{draft}\n\n")

    if experiment_evidence:
        user_parts.append(
            f"## Experiment Evidence (fact-check claims against this)\n\n"
            f"{experiment_evidence}\n"
        )

    return system, "\n".join(user_parts)


def build_panel_prompt(
    draft: str,
    experiment_evidence: str = "",
    prior_critiques: str = "",
    *,
    personas: tuple[str, ...] = ("board", "balanced"),
) -> tuple[str, str]:
    """Build prompts for a multi-persona panel review.

    Asks the LLM to respond as each named persona in sequence, each
    separated by a clear header. The shared CRITIQUE_FRAMEWORK is appended
    once at the end of the system prompt so it applies to all personas.

    Parameters
    ----------
    draft:
        The paper draft text.
    experiment_evidence:
        Supporting evidence (fact-checked against claims).
    prior_critiques:
        Critiques from prior runs; each persona verifies they were resolved.
    personas:
        Persona names from PERSONAS to include. Unknown names are skipped.

    Returns
    -------
    ``(system_prompt, user_prompt)`` tuple ready for ``llm.chat()``.
    """
    # Each persona contributes only its own preamble (before CRITIQUE_FRAMEWORK).
    # CRITIQUE_FRAMEWORK is appended once, shared by all.
    panel_parts: list[str] = [
        "You are a multi-perspective review panel. "
        "Respond as EACH of the following personas in sequence, "
        "clearly separating each review with a '## [PERSONA NAME] REVIEW' header. "
        "Apply ALL criteria from the shared Critique Methodology for every persona.\n\n"
    ]
    included_personas: list[str] = []
    for p in personas:
        persona_system = PERSONAS.get(p)
        if not persona_system:
            continue
        included_personas.append(p)
        # Each persona prompt is "preamble + CRITIQUE_FRAMEWORK".
        # Strip the trailing framework suffix — we append it once below so it
        # applies to all personas rather than repeating N times.
        if persona_system.endswith(CRITIQUE_FRAMEWORK):
            preamble = persona_system.rsplit(CRITIQUE_FRAMEWORK, 1)[0].rstrip()
        else:
            preamble = persona_system.strip()
        # Use the same header format the model is instructed to use when
        # separating its output, so system headers and output headers align.
        panel_parts.append(f"## [{p.upper()}] REVIEW\n{preamble}\n")

    if not included_personas:
        raise ValueError(
            "build_panel_prompt() received no valid persona names; "
            f"none of {list(personas)!r} exist in PERSONAS."
        )

    panel_parts.append(f"\n{CRITIQUE_FRAMEWORK}")
    system = "\n".join(panel_parts)

    # Generic user prompt (not board-specific)
    user_parts = [
        "Review the following paper draft with maximum rigor.\n",
        f"Apply all {len(included_personas)} personas in sequence, each with a header.\n",
        "Tag every finding: [FATAL], [EVIDENCE?], [HEDGE], or [POLISH].\n",
    ]
    if prior_critiques:
        user_parts.append(
            "\n## Prior Review Critiques (from previous iterations)\n"
            "Every persona must verify these issues were ADDRESSED. "
            "If any remain unresolved, escalate severity.\n\n"
            f"{prior_critiques}\n\n"
        )
    user_parts.append(f"\n## Paper Draft\n\n{draft}\n\n")
    if experiment_evidence:
        user_parts.append(
            f"## Experiment Evidence (fact-check claims against this)\n\n"
            f"{experiment_evidence}\n"
        )

    return system, "\n".join(user_parts)


def format_critique_lessons(reviews: str) -> list[str]:
    """Extract actionable lessons from critique reviews for cross-run learning.

    Parses [FATAL], [EVIDENCE?], [HEDGE] tags from reviews and converts
    them into plain-text lesson strings suitable for prompt injection
    (e.g., via PriorRunContext.prior_critique_lessons). [POLISH] tags are
    excluded -- they are minor and not worth carrying forward.

    Returns plain strings, not LessonEntry objects. Callers that need to
    persist lessons in EvolutionStore should wrap these strings using
    LessonEntry(stage_name="peer_review", stage_num=18, ...) directly.
    """
    def _normalize(text: str) -> str:
        # Collapse CRLF and internal whitespace; strip trailing punctuation
        # so "issue." and "issue" deduplicate to the same lesson.
        collapsed = " ".join(text.split())
        return collapsed.rstrip(".,!?:;")

    raw: list[str] = []

    for match in _FATAL_RE.finditer(reviews):
        payload = _normalize(match.group(1).strip())
        if payload:
            raw.append(f"FATAL FLAW (prior review): {payload}")
    for match in _EVIDENCE_RE.finditer(reviews):
        payload = _normalize(match.group(1).strip())
        if payload:
            raw.append(f"Unsupported claim (prior review): {payload}")
    for match in _HEDGE_RE.finditer(reviews):
        payload = _normalize(match.group(1).strip())
        if payload:
            raw.append(f"Hedging detected (prior review): {payload}")

    # Deduplicate while preserving order (panel reviews can repeat the same tag).
    seen: set[str] = set()
    lessons: list[str] = []
    for lesson in raw:
        if lesson not in seen:
            seen.add(lesson)
            lessons.append(lesson)
    return lessons


# One-line labels for each persona; callers can display these without
# needing to parse the full prompt strings.
PERSONA_DESCRIPTIONS: dict[str, str] = {
    "board":    "Scathing academic panel -- terse imperatives, zero hedging tolerance",
    "balanced": "Three-perspective reviewer (methodology, domain, statistics)",
    "tank":     "Silicon Valley VC panel -- commercialization and impact evaluation",
    "bros":     "Tech-bro startup CEOs -- same rigor, venture-speak translation",
}


def available_personas() -> dict[str, str]:
    """Return ``{name: one-line description}`` for all registered personas."""
    return dict(PERSONA_DESCRIPTIONS)
