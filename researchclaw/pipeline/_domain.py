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

from researchclaw.domains.detector import DomainProfile, detect_domain as _detect_profile

_TOP_VENUES_BY_PARENT: dict[str, str] = {
    "ml": "NeurIPS, ICML, ICLR",
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
