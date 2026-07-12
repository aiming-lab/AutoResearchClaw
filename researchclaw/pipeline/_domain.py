"""Compatibility wrapper around the canonical domain profile detector.

Historically the pipeline used a coarse keyword-based detector that returned
``(domain_id, display_name, top_venues)`` tuples such as ``("ml", ...)``.
The newer domain system in :mod:`researchclaw.domains.detector` is more
expressive (for example ``ml_vision`` vs ``ml_tabular``) and should be treated
as the source of truth.

This module now exists only to preserve the legacy tuple API for older call
sites while routing every decision through the newer detector.
"""

from __future__ import annotations

from researchclaw.domains.detector import (
    DomainProfile,
    detect_domain as _detect_profile,
    detect_domain_unforced as _detect_profile_unforced,
)

_TOP_VENUES_BY_PARENT: dict[str, str] = {
    "ml": "NeurIPS, ICML, ICLR",
    "hep_ph": "JHEP, PRD, PRL, EPJC, Phys.Lett.B",
    "physics": "Physical Review Letters, Nature Physics, JHEP",
    "chemistry": "JACS, Nature Chemistry, Angewandte Chemie",
    "economics": "AER, Econometrica, QJE, Review of Economic Studies",
    "mathematics": "Annals of Mathematics, Inventiones Mathematicae, JAMS",
    "engineering": "IEEE Transactions, ASME journals, AIAA",
    "biology": "Nature, Science, Cell, PNAS",
    "security": "IEEE S&P, USENIX Security, CCS",
    "neuroscience": "Neuron, Nature Neuroscience, eLife",
    "robotics": "ICRA, RSS, CoRL",
    "generic": "ArXiv, workshop venues",
}

_DISPLAY_NAME_BY_PARENT: dict[str, str] = {
    "ml": "machine learning",
    "hep_ph": "high energy physics phenomenology",
    "physics": "physics",
    "chemistry": "chemistry",
    "economics": "economics",
    "mathematics": "mathematics",
    "engineering": "engineering",
    "biology": "biology",
    "security": "security",
    "neuroscience": "neuroscience",
    "robotics": "robotics",
    "generic": "generic research",
}

_COARSE_DOMAIN_ALIASES: tuple[tuple[str, str], ...] = (
    ("ml_", "ml"),
    ("hep_ph", "hep_ph"),   # exact prefix match before generic physics
    ("physics_", "physics"),
    ("chemistry_", "chemistry"),
    ("biology_", "biology"),
    ("economics_", "economics"),
    ("mathematics_", "mathematics"),
    ("security_", "security"),
    ("neuroscience_", "neuroscience"),
    ("robotics_", "robotics"),
)


def _coarse_domain_id(profile: DomainProfile) -> str:
    """Map a detailed domain profile back to the legacy coarse ID space."""
    domain_id = str(profile.domain_id).strip()
    if not domain_id:
        return "generic"

    for prefix, coarse in _COARSE_DOMAIN_ALIASES:
        if domain_id.startswith(prefix):
            return coarse

    parent = str(profile.parent_domain or "").strip().lower()
    if parent:
        return parent

    return domain_id


def _coarse_display_name(profile: DomainProfile, coarse_domain_id: str) -> str:
    """Choose a stable human-facing display name for legacy call sites."""
    parent = str(profile.parent_domain or "").strip()
    if parent:
        return _DISPLAY_NAME_BY_PARENT.get(parent, parent.replace("_", " "))
    return _DISPLAY_NAME_BY_PARENT.get(
        coarse_domain_id,
        coarse_domain_id.replace("_", " "),
    )


def _top_venues_for_profile(profile: DomainProfile, coarse_domain_id: str) -> str:
    """Best-effort venue context for old tuple-based consumers."""
    parent = str(profile.parent_domain or "").strip().lower()
    if parent in _TOP_VENUES_BY_PARENT:
        return _TOP_VENUES_BY_PARENT[parent]
    if coarse_domain_id in _TOP_VENUES_BY_PARENT:
        return _TOP_VENUES_BY_PARENT[coarse_domain_id]
    return _TOP_VENUES_BY_PARENT["generic"]


def _detect_domain(topic: str, domains: tuple[str, ...] = ()) -> tuple[str, str, str]:
    """Detect domain via the canonical domain-profile system.

    Returns the historical ``(domain_id, display_name, top_venues)`` tuple so
    older pipeline stages can remain unchanged while sharing one detector.
    """
    domain_hints = ", ".join(
        str(d).strip().replace("-", " ").replace("_", " ")
        for d in domains
        if str(d).strip()
    )
    hypotheses = (
        f"Configured domains: {domain_hints}."
        if domain_hints
        else ""
    )
    profile = _detect_profile(topic=topic, hypotheses=hypotheses)
    coarse_domain_id = _coarse_domain_id(profile)
    return (
        coarse_domain_id,
        _coarse_display_name(profile, coarse_domain_id),
        _top_venues_for_profile(profile, coarse_domain_id),
    )


def _is_ml_domain(domain_id: str) -> bool:
    """Check if the detected legacy coarse domain is ML/AI."""
    return domain_id == "ml"


# ---------------------------------------------------------------------------
# Prompt-bank domain selection
# ---------------------------------------------------------------------------

# Coarse IDs that map to the HEP-phenomenology prompt bank. Anything else
# falls back to the ML bank — that is the safe default because the ML bank
# uses general research vocabulary that works for most domains. Future
# forks (chemistry, economics, …) are added here as their prompt banks land.
_HEP_PROMPT_BANK_IDS: frozenset[str] = frozenset({"hep_ph"})


def _prompt_bank_domain_from_config(config: object) -> str:
    """Pick the prompt bank (``"ml"`` or ``"hep_ph"``) for this run.

    Resolution order:
    1. ``config.project.profile`` (set by ``--profile`` or ``project.profile``)
       — the explicit user choice wins.
    2. Topic-based detection via :func:`_detect_domain`. This catches runs
       where the user did not set a profile but the topic is unmistakably
       HEP-phenomenology (dark matter, Z-prime, BSM, …).
    3. Fallback to ``"ml"``.
    """
    profile_id = ""
    try:
        profile_id = str(getattr(getattr(config, "project", object()), "profile", "") or "").strip()
    except Exception:  # noqa: BLE001
        profile_id = ""
    if profile_id in _HEP_PROMPT_BANK_IDS:
        return "hep_ph"
    if profile_id:
        return "ml"

    topic = ""
    domains: tuple[str, ...] = ()
    try:
        research = getattr(config, "research", None)
        if research is not None:
            topic = str(getattr(research, "topic", "") or "")
            domains_raw = getattr(research, "domains", ()) or ()
            domains = tuple(str(d) for d in domains_raw)
    except Exception:  # noqa: BLE001
        pass

    # Explicit domain list takes precedence over keyword detection, which
    # has a known limitation (it strips hyphens before matching, so the
    # literal arXiv category "hep-ph" gets mangled). Trusting an explicitly
    # configured ``hep-ph`` / ``hep-ex`` avoids that bug.
    for d in domains:
        tag = str(d).strip().lower()
        if tag in {"hep-ph", "hep_ph", "hep-ex", "hep_ex"}:
            return "hep_ph"

    if not topic:
        return "ml"

    domain_hints = ", ".join(
        str(domain).strip().replace("-", " ").replace("_", " ")
        for domain in domains
        if str(domain).strip()
    )
    hypotheses = f"Configured domains: {domain_hints}." if domain_hints else ""
    try:
        profile = _detect_profile_unforced(topic, hypotheses=hypotheses)
        coarse_id = _coarse_domain_id(profile)
    except Exception:  # noqa: BLE001
        return "ml"
    if coarse_id in _HEP_PROMPT_BANK_IDS:
        return "hep_ph"
    return "ml"
