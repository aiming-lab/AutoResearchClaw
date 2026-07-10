from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from researchclaw.pipeline.manuscript_sections import (
    ManuscriptStructureError,
    merge_manuscript,
    parse_manuscript,
    parse_manuscript_file,
)


def test_round_trip_preserves_commonmark_source_byte_for_byte() -> None:
    source = (
        "---\r\n"
        "title: Example\r\n"
        "---\r\n"
        "Lead text.\r\n\r\n"
        "# Paper\r\n\r\n"
        "## Method\r\n\r\n"
        "| A | B |\r\n"
        "|---|---|\r\n"
        "| 1 | 2 |\r\n\r\n"
        "## Results\r\n\r\n"
        "Final line without newline"
    )

    document = parse_manuscript(source)

    assert merge_manuscript(document) == source
    assert document.preamble.startswith("---\r\n")
    assert [section.title for section in document.sections] == [
        "Paper",
        "Method",
        "Results",
    ]


def test_fenced_heading_is_not_split_into_a_section() -> None:
    source = (
        "# Paper\n\n"
        "## Method\n\n"
        "```markdown\n"
        "# Not a real heading\n"
        "## Also not a heading\n"
        "```\n\n"
        "## Results\n\n"
        "Done.\n"
    )

    document = parse_manuscript(source)

    assert [section.title for section in document.sections] == [
        "Paper",
        "Method",
        "Results",
    ]
    assert "# Not a real heading" in document.sections[1].body
    assert merge_manuscript(document) == source


@pytest.mark.parametrize("separator", ("\u2028", "\f", "\x85"))
def test_non_commonmark_line_separator_does_not_shift_boundaries(
    separator: str,
) -> None:
    source = (
        "# Paper\n\n"
        f"First paragraph{separator}same CommonMark line.\n\n"
        "## Method\n\n"
        "Method body.\n"
    )

    document = parse_manuscript(source)

    method = document.sections[1]
    assert method.heading_source == "## Method\n"
    assert method.body == "\nMethod body.\n"
    assert separator in document.sections[0].body
    assert merge_manuscript(document) == source


def test_setext_heading_and_html_comment_round_trip() -> None:
    source = (
        "<!-- generated metadata -->\n"
        "Paper Title\n"
        "===========\n\n"
        "Abstract text.\n\n"
        "Method\n"
        "------\n\n"
        "$$x = y$$\n"
    )

    document = parse_manuscript(source)

    assert [section.title for section in document.sections] == [
        "Paper Title",
        "Method",
    ]
    assert merge_manuscript(document) == source


def test_front_matter_ellipsis_closer_round_trip() -> None:
    source = "---\ntitle: Paper\n...\n# Paper\n\nText.\n"

    document = parse_manuscript(source)

    assert document.preamble == "---\ntitle: Paper\n...\n"
    assert merge_manuscript(document) == source


def test_unclosed_front_matter_delimiter_remains_commonmark_content() -> None:
    source = "---\n\n# Paper\n\nText.\n"

    document = parse_manuscript(source)

    assert document.preamble == "---\n\n"
    assert [section.title for section in document.sections] == ["Paper"]
    assert merge_manuscript(document) == source


def test_thematic_breaks_do_not_swallow_markdown_as_front_matter() -> None:
    source = "---\n\n# Paper\n\nProse, not YAML.\n\n---\n\n## Results\n\nDone.\n"

    document = parse_manuscript(source)

    assert document.preamble == "---\n\n"
    assert [section.title for section in document.sections] == ["Paper", "Results"]
    assert "---\n\n" in document.sections[0].body
    assert merge_manuscript(document) == source


def test_duplicate_canonical_heading_path_fails_strict_mode() -> None:
    source = "# Paper\n\n## Method\nA\n\n## Method\nB\n"

    with pytest.raises(ManuscriptStructureError) as exc_info:
        parse_manuscript(source)

    assert any(
        issue.code == "duplicate_heading_path" for issue in exc_info.value.issues
    )

    diagnostic = parse_manuscript(source, strict=False)
    assert merge_manuscript(diagnostic) == source
    assert len({section.section_id for section in diagnostic.sections}) == 3
    with pytest.raises(ManuscriptStructureError) as merge_error:
        merge_manuscript(
            diagnostic,
            {diagnostic.sections[1].section_id: "Replacement.\n"},
        )
    assert any(
        issue.code == "duplicate_heading_path" for issue in merge_error.value.issues
    )


def test_same_child_heading_under_different_parents_is_not_ambiguous() -> None:
    source = (
        "# Paper\n\n"
        "## Experiment A\n\n"
        "### Results\n\nA.\n\n"
        "## Experiment B\n\n"
        "### Results\n\nB.\n"
    )

    document = parse_manuscript(source)

    result_paths = [
        section.path for section in document.sections if section.title == "Results"
    ]
    assert result_paths == [
        ("Paper", "Experiment A", "Results"),
        ("Paper", "Experiment B", "Results"),
    ]
    assert merge_manuscript(document) == source


def test_empty_heading_and_level_jump_fail_strict_mode() -> None:
    source = "# Paper\n\n### Jumped\n\n##\n"

    with pytest.raises(ManuscriptStructureError) as exc_info:
        parse_manuscript(source)

    codes = {issue.code for issue in exc_info.value.issues}
    assert codes == {"empty_heading", "heading_level_jump"}


def test_nested_heading_is_reported_and_cannot_be_replaced() -> None:
    source = "# Paper\n\n> ## Quoted heading\n> Quoted body.\n\n## Results\nDone.\n"

    with pytest.raises(ManuscriptStructureError) as exc_info:
        parse_manuscript(source)
    assert any(issue.code == "nested_heading" for issue in exc_info.value.issues)

    diagnostic = parse_manuscript(source, strict=False)
    assert merge_manuscript(diagnostic) == source
    with pytest.raises(ManuscriptStructureError):
        merge_manuscript(
            diagnostic,
            {diagnostic.sections[1].section_id: "Unsafe replacement.\n"},
        )


def test_h2_can_be_the_first_heading_without_level_jump() -> None:
    source = "## Abstract\n\nText.\n\n### Detail\n\nMore.\n"

    document = parse_manuscript(source)

    assert document.structure_issues == ()
    assert merge_manuscript(document) == source


def test_non_ascii_heading_ids_remain_unique() -> None:
    source = "## 方法\n\n内容。\n\n## 结果\n\n结果。\n"

    document = parse_manuscript(source)

    ids = [section.section_id for section in document.sections]
    assert ids[0].startswith("s000-section-")
    assert ids[1].startswith("s001-section-")
    assert len(set(ids)) == 2


def test_merge_replaces_only_requested_body() -> None:
    source = "# Paper\n\nIntro.\n\n## Results\n\nOld result.\n"
    document = parse_manuscript(source)
    results = document.sections[1]

    merged = merge_manuscript(
        document,
        {results.section_id: "\nNew grounded result.\n"},
    )

    assert merged == "# Paper\n\nIntro.\n\n## Results\n\nNew grounded result.\n"
    assert document.sections[0].source == "# Paper\n\nIntro.\n\n"


def test_merge_rejects_unknown_section_and_non_string_body() -> None:
    document = parse_manuscript("# Paper\n\nText.\n")
    section_id = document.sections[0].section_id

    with pytest.raises(KeyError, match="unknown section IDs"):
        merge_manuscript(document, {"missing": "body"})
    with pytest.raises(TypeError, match="must be a string"):
        merge_manuscript(document, {section_id: 42})  # type: ignore[dict-item]


def test_merge_rejects_tampered_preamble_even_with_replacements() -> None:
    document = parse_manuscript("Preface.\n\n# Paper\n\nText.\n")
    tampered = replace(document, preamble="Changed preface.\n\n")

    with pytest.raises(ManuscriptStructureError) as exc_info:
        merge_manuscript(
            tampered,
            {tampered.sections[0].section_id: "Replacement.\n"},
        )

    assert exc_info.value.issues[0].code == "round_trip_mismatch"


def test_no_heading_is_lossless_only_in_diagnostic_mode() -> None:
    source = "Plain manuscript without structure.\n"

    with pytest.raises(ManuscriptStructureError) as exc_info:
        parse_manuscript(source)
    assert exc_info.value.issues[0].code == "no_headings"

    document = parse_manuscript(source, strict=False)
    assert merge_manuscript(document) == source


def test_parse_file_preserves_crlf(tmp_path: Path) -> None:
    path = tmp_path / "paper.md"
    path.write_bytes(b"# Paper\r\n\r\nText.\r\n")

    document = parse_manuscript_file(path)

    assert merge_manuscript(document).encode("utf-8") == path.read_bytes()


def test_parse_file_preserves_cr_only_line_endings(tmp_path: Path) -> None:
    path = tmp_path / "paper.md"
    path.write_bytes(b"# Paper\r\r## Method\r\rText.\r")

    document = parse_manuscript_file(path)

    assert document.sections[1].heading_source == "## Method\r"
    assert merge_manuscript(document).encode("utf-8") == path.read_bytes()


def test_real_stage17_manuscript_round_trip_when_available() -> None:
    path = (
        Path(__file__).resolve().parents[1]
        / "runs"
        / "hwsec-scaffold-v2"
        / "stage-17"
        / "paper_draft.md"
    )
    if not path.is_file():
        pytest.skip("local real Stage 17 manuscript is not available")

    with pytest.raises(ManuscriptStructureError) as exc_info:
        parse_manuscript_file(path)
    assert any(
        issue.code == "duplicate_heading_path" for issue in exc_info.value.issues
    )

    document = parse_manuscript_file(path, strict=False)

    with path.open("r", encoding="utf-8", newline="") as handle:
        source = handle.read()
    assert merge_manuscript(document) == source
    assert any(
        issue.code == "duplicate_heading_path"
        for issue in document.structure_issues
    )
