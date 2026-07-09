from __future__ import annotations

from researchclaw.pipeline.stage_impls._literature import _candidate_screen_score


def test_candidate_screen_score_prefers_topic_relevance_over_global_citations() -> None:
    keywords = [
        "hardware-performance-counter",
        "runtime",
        "detection",
        "spectre",
        "meltdown",
        "transient-execution",
        "side-channel",
    ]
    off_topic_high_citation = {
        "title": "SciPy 1.0: fundamental algorithms for scientific computing in Python",
        "abstract": (
            "SciPy is an open-source scientific computing library with many "
            "algorithms and broad development practices."
        ),
        "citation_count": 38000,
    }
    on_topic_low_citation = {
        "title": "Detecting Spectre Attacks with Hardware Performance Counters",
        "abstract": (
            "We use performance counter traces and lightweight runtime anomaly "
            "detection to identify transient execution side-channel attacks."
        ),
        "citation_count": 12,
    }

    assert _candidate_screen_score(on_topic_low_citation, keywords) > (
        _candidate_screen_score(off_topic_high_citation, keywords)
    )
