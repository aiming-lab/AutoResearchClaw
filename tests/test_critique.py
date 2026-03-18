"""Tests for researchclaw.critique -- multi-persona critique system."""

from __future__ import annotations

import pytest

from researchclaw.critique import (
    CRITIQUE_FRAMEWORK,
    PERSONAS,
    available_personas,
    build_critique_prompt,
    build_panel_prompt,
    format_critique_lessons,
)


# ---------------------------------------------------------------------------
# available_personas
# ---------------------------------------------------------------------------


class TestAvailablePersonas:
    def test_returns_all_four(self) -> None:
        descs = available_personas()
        assert set(descs) == {"board", "balanced", "tank", "bros"}

    def test_values_are_strings(self) -> None:
        for name, desc in available_personas().items():
            assert isinstance(desc, str), f"description for {name!r} is not a string"
            assert desc, f"description for {name!r} is empty"


# ---------------------------------------------------------------------------
# build_critique_prompt
# ---------------------------------------------------------------------------


class TestBuildCritiquePrompt:
    def test_board_persona_default(self) -> None:
        system, user = build_critique_prompt("A draft paper.")
        assert CRITIQUE_FRAMEWORK in system
        assert "A draft paper." in user

    def test_known_persona_selects_correct_system(self) -> None:
        _, _ = build_critique_prompt("Draft.", persona="balanced")
        system_balanced, _ = build_critique_prompt("Draft.", persona="balanced")
        system_board, _ = build_critique_prompt("Draft.", persona="board")
        assert system_balanced != system_board

    def test_unknown_persona_falls_back_to_board(self) -> None:
        """Unknown persona falls back silently to board; caller gets valid output."""
        board_system, _ = build_critique_prompt("Draft.", persona="board")
        unknown_system, _ = build_critique_prompt("Draft.", persona="nonexistent")
        assert unknown_system == board_system

    def test_experiment_evidence_included_in_user(self) -> None:
        _, user = build_critique_prompt(
            "Draft.", experiment_evidence="R-squared = 0.97"
        )
        assert "R-squared = 0.97" in user

    def test_prior_critiques_included_in_user(self) -> None:
        _, user = build_critique_prompt(
            "Draft.", prior_critiques="[FATAL] Missing error bars."
        )
        assert "[FATAL] Missing error bars." in user

    def test_tags_instruction_in_user(self) -> None:
        _, user = build_critique_prompt("Draft.")
        assert "[FATAL]" in user and "[EVIDENCE?]" in user


# ---------------------------------------------------------------------------
# build_panel_prompt -- framework deduplication
# ---------------------------------------------------------------------------


class TestBuildPanelPrompt:
    def test_framework_appears_exactly_once(self) -> None:
        """CRITIQUE_FRAMEWORK must appear once in system prompt, not N times."""
        system, _ = build_panel_prompt("Draft.", personas=("board", "balanced", "tank"))
        count = system.count(CRITIQUE_FRAMEWORK.strip()[:60])
        assert count == 1, (
            f"CRITIQUE_FRAMEWORK appeared {count} times; expected exactly 1"
        )

    def test_all_persona_headers_present(self) -> None:
        system, _ = build_panel_prompt("Draft.", personas=("board", "balanced"))
        assert "## [BOARD] REVIEW" in system
        assert "## [BALANCED] REVIEW" in system

    def test_output_header_matches_separator_instruction(self) -> None:
        """System prompt must tell the model to use '## [PERSONA NAME] REVIEW' separators
        and the persona sections must use the same format."""
        system, user = build_panel_prompt("Draft.", personas=("board",))
        # The system prompt instructs the model to use this separator format
        assert "## [PERSONA NAME] REVIEW" in system or "## [BOARD] REVIEW" in system
        # The user prompt says "apply all N personas with a header"
        assert "header" in user.lower()

    def test_invalid_persona_skipped(self) -> None:
        """Unknown persona names are silently skipped; valid ones still included."""
        system, _ = build_panel_prompt("Draft.", personas=("board", "nonexistent"))
        assert "## [BOARD] REVIEW" in system
        assert "NONEXISTENT" not in system

    def test_all_invalid_personas_raises(self) -> None:
        with pytest.raises(ValueError, match="no valid persona names"):
            build_panel_prompt("Draft.", personas=("ghost", "phantom"))

    def test_single_persona_panel(self) -> None:
        system, user = build_panel_prompt("Draft.", personas=("tank",))
        assert "## [TANK] REVIEW" in system
        assert CRITIQUE_FRAMEWORK.strip()[:60] in system

    def test_experiment_evidence_in_user(self) -> None:
        _, user = build_panel_prompt(
            "Draft.", experiment_evidence="p=0.001", personas=("board",)
        )
        assert "p=0.001" in user

    def test_prior_critiques_in_user(self) -> None:
        _, user = build_panel_prompt(
            "Draft.", prior_critiques="[FATAL] Bad stats.", personas=("board",)
        )
        assert "[FATAL] Bad stats." in user


# ---------------------------------------------------------------------------
# format_critique_lessons -- tag extraction and deduplication
# ---------------------------------------------------------------------------


class TestFormatCritiqueLessons:
    def test_fatal_extracted(self) -> None:
        reviews = "[FATAL] Missing confidence intervals in Table 2.\n"
        lessons = format_critique_lessons(reviews)
        assert any("FATAL FLAW" in l and "Missing confidence intervals" in l for l in lessons)

    def test_evidence_extracted(self) -> None:
        reviews = "[EVIDENCE?] The claim that accuracy is 99% lacks a citation.\n"
        lessons = format_critique_lessons(reviews)
        assert any("Unsupported claim" in l and "99%" in l for l in lessons)

    def test_hedge_extracted(self) -> None:
        reviews = "[HEDGE] The results may suggest a correlation.\n"
        lessons = format_critique_lessons(reviews)
        assert any("Hedging detected" in l and "may suggest" in l for l in lessons)

    def test_polish_not_extracted(self) -> None:
        """[POLISH] tags are minor; they must NOT appear in lessons."""
        reviews = "[POLISH] Consider rewording the abstract.\n"
        lessons = format_critique_lessons(reviews)
        assert not lessons

    def test_deduplication_across_personas(self) -> None:
        """Same tag from two personas -> one lesson, not two."""
        reviews = (
            "[FATAL] Missing error bars.\n"
            "[FATAL] Missing error bars.\n"
        )
        lessons = format_critique_lessons(reviews)
        fatal_lessons = [l for l in lessons if "Missing error bars" in l]
        assert len(fatal_lessons) == 1

    def test_order_preserved(self) -> None:
        """FATAL extracted before EVIDENCE? before HEDGE."""
        reviews = (
            "[HEDGE] Hedging here.\n"
            "[EVIDENCE?] No citation.\n"
            "[FATAL] Fatal flaw.\n"
        )
        lessons = format_critique_lessons(reviews)
        # FATAL must come before EVIDENCE? must come before HEDGE
        fatals = [i for i, l in enumerate(lessons) if "FATAL FLAW" in l]
        evidences = [i for i, l in enumerate(lessons) if "Unsupported claim" in l]
        hedges = [i for i, l in enumerate(lessons) if "Hedging detected" in l]
        assert fatals and evidences and hedges
        assert max(fatals) < min(evidences)
        assert max(evidences) < min(hedges)

    def test_trailing_punctuation_normalised(self) -> None:
        """Lessons that differ only by trailing period deduplicate correctly."""
        reviews = (
            "[FATAL] Missing confidence intervals.\n"
            "[FATAL] Missing confidence intervals\n"
        )
        lessons = format_critique_lessons(reviews)
        fatal = [l for l in lessons if "Missing confidence intervals" in l]
        assert len(fatal) == 1

    def test_empty_string_returns_empty(self) -> None:
        assert format_critique_lessons("") == []

    def test_no_tags_returns_empty(self) -> None:
        assert format_critique_lessons("This paper is great!") == []

    def test_whitespace_collapse(self) -> None:
        """Multi-space payloads collapse to single spaces before deduplication."""
        reviews = "[FATAL]  Inconsistent   notation.\n"
        lessons = format_critique_lessons(reviews)
        assert len(lessons) == 1
        assert "  " not in lessons[0]
