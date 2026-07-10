from __future__ import annotations

import copy
from dataclasses import replace
from pathlib import Path

import pytest

from researchclaw.pipeline.manuscript_sections import parse_manuscript
from researchclaw.pipeline.sectional_revision import (
    ReviewLedger,
    SectionalRevisionContractError,
    extract_review_ledger,
    make_attempt_id,
    validate_review_ledger,
    validate_revision_plan,
)


FIXTURE_DIR = Path(__file__).parent / "fixtures"


def _codes(exc: SectionalRevisionContractError) -> set[str]:
    return {issue.code for issue in exc.issues}


def _sanitized_reviews() -> str:
    return (FIXTURE_DIR / "stage18_reviews_sanitized.md").read_text(
        encoding="utf-8"
    )


def _ledger() -> ReviewLedger:
    return extract_review_ledger(
        _sanitized_reviews(), source_path="stage-18/reviews.md"
    )


def _document():
    return parse_manuscript(
        "# Paper\n\nIntro.\n\n## Method\n\nMethod body.\n\n## Results\n\nResults.\n"
    )


def _valid_plan() -> dict:
    ledger = _ledger()
    document = _document()
    target = document.sections[1].section_id
    return {
        "schema_version": 1,
        "planner_version": 1,
        "source_paper_sha256": document.source_sha256,
        "source_reviews_sha256": ledger.source_reviews_sha256,
        "section_model_version": 1,
        "assignments": [
            {
                "comment_id": comment.comment_id,
                "target_section_ids": [target],
                "disposition": "assigned",
                "reason": None,
            }
            for comment in ledger.comments
        ],
    }


def _validate_plan(plan: object, ledger=None, document=None):
    return validate_revision_plan(
        plan,
        ledger or _ledger(),
        document or _document(),
        reviews=_sanitized_reviews(),
    )


def test_sanitized_stage18_fixture_extracts_exactly_17_comments() -> None:
    ledger = _ledger()

    assert len(ledger.comments) == 17
    assert sum(c.category == "actionable_revision" for c in ledger.comments) == 13
    assert sum(c.category == "general_comment" for c in ledger.comments) == 4
    assert all(c.required for c in ledger.comments[:13])
    assert all(not c.required for c in ledger.comments[13:])


def test_general_comments_suffix_normalizes_and_reviewer_is_all() -> None:
    comments = [c for c in _ledger().comments if c.category == "general_comment"]

    assert len(comments) == 4
    assert {comment.reviewer for comment in comments} == {"all"}


def test_unknown_content_bearing_subsection_fails_closed() -> None:
    reviews = """## Reviewer C

### Additional Rigor Issues
- This issue must not disappear.

### Actionable Revisions
1. Fix the supported issue.
"""

    with pytest.raises(SectionalRevisionContractError) as caught:
        extract_review_ledger(reviews, source_path="stage-18/reviews.md")

    assert "unknown_review_subsection" in _codes(caught.value)


def test_unknown_nested_heading_inside_actionable_item_fails_closed() -> None:
    reviews = """## Reviewer A

### Actionable Revisions
1. Address the main issue.

   #### Hidden Extra Section
   - This nested issue must not hide inside the accepted list span.
"""

    with pytest.raises(SectionalRevisionContractError) as caught:
        extract_review_ledger(reviews, source_path="stage-18/reviews.md")

    assert "unknown_review_subsection" in _codes(caught.value)


def test_live_reviews_are_an_explicit_negative_fixture_when_available() -> None:
    path = Path("runs/hwsec-scaffold-v2/stage-18/reviews.md")
    if not path.is_file():
        pytest.skip("ignored live run is not available")

    with pytest.raises(SectionalRevisionContractError) as caught:
        extract_review_ledger(
            path.read_text(encoding="utf-8"), source_path="stage-18/reviews.md"
        )

    issues = caught.value.issues
    assert [(issue.code, issue.line) for issue in issues] == [
        ("unknown_review_subsection", 59)
    ]


def test_thematic_breaks_are_not_comments() -> None:
    reviews = """## Reviewer A

### Actionable Revisions
1. First item.

---

## Reviewer B

### Actionable Revisions
1. Second item.
"""

    ledger = extract_review_ledger(reviews, source_path="stage-18/reviews.md")

    assert len(ledger.comments) == 2


def test_nested_list_and_continuation_remain_in_parent_comment() -> None:
    reviews = """## Reviewer A

### Actionable Revisions
1. First line.
   Continuation line.
   - Nested supporting detail.
2. Second item.
"""

    ledger = extract_review_ledger(reviews, source_path="stage-18/reviews.md")

    assert len(ledger.comments) == 2
    assert "Continuation line" in ledger.comments[0].exact_text
    assert "Nested supporting detail" in ledger.comments[0].exact_text


def test_duplicate_comment_text_at_distinct_spans_gets_distinct_ids() -> None:
    reviews = """## Reviewer A

### Actionable Revisions
1. Clarify the metric.
2. Clarify the metric.
"""

    ledger = extract_review_ledger(reviews, source_path="stage-18/reviews.md")

    assert len(ledger.comments) == 2
    assert ledger.comments[0].comment_id != ledger.comments[1].comment_id


def test_nonempty_reviews_with_zero_actionable_comments_fail() -> None:
    reviews = """## Reviewer A

### Strengths
- Clear writing.
"""

    with pytest.raises(SectionalRevisionContractError) as caught:
        extract_review_ledger(reviews, source_path="stage-18/reviews.md")

    assert "review_comments_empty" in _codes(caught.value)


@pytest.mark.parametrize(
    "source_path",
    [
        "/stage-18/reviews.md",
        "../stage-18/reviews.md",
        "./stage-18/reviews.md",
        "stage-18//reviews.md",
        "stage-18\\reviews.md",
    ],
)
def test_review_source_path_must_be_canonical_and_relative(source_path: str) -> None:
    with pytest.raises(SectionalRevisionContractError) as caught:
        extract_review_ledger(_sanitized_reviews(), source_path=source_path)

    assert "artifact_path_invalid" in _codes(caught.value)


def test_prose_outside_subsection_fails_closed() -> None:
    reviews = """## Reviewer A

This prose is outside a subsection.

### Actionable Revisions
1. Fix it.
"""

    with pytest.raises(SectionalRevisionContractError) as caught:
        extract_review_ledger(reviews, source_path="stage-18/reviews.md")

    assert "review_content_outside_subsection" in _codes(caught.value)


def test_review_ledger_round_trips_strict_schema() -> None:
    ledger = _ledger()

    restored = ReviewLedger.from_dict(ledger.to_dict())

    assert restored == ledger


def test_review_ledger_rejects_unknown_root_and_comment_fields() -> None:
    root_extra = _ledger().to_dict()
    root_extra["unexpected"] = True
    with pytest.raises(SectionalRevisionContractError) as root_error:
        ReviewLedger.from_dict(root_extra)
    assert "unknown_fields" in _codes(root_error.value)

    comment_extra = _ledger().to_dict()
    comment_extra["comments"][0]["unexpected"] = True
    with pytest.raises(SectionalRevisionContractError) as comment_error:
        ReviewLedger.from_dict(comment_extra)
    assert "unknown_fields" in _codes(comment_error.value)


def test_review_ledger_binds_hash_path_and_exact_source_span() -> None:
    reviews = _sanitized_reviews()
    ledger = _ledger()

    with pytest.raises(SectionalRevisionContractError) as hash_error:
        validate_review_ledger(ledger, reviews=reviews + "\n")
    assert "source_reviews_hash_mismatch" in _codes(hash_error.value)

    with pytest.raises(SectionalRevisionContractError) as path_error:
        validate_review_ledger(
            ledger, reviews=reviews, source_path="stage-18_v2/reviews.md"
        )
    assert "source_reviews_path_mismatch" in _codes(path_error.value)

    changed = replace(ledger.comments[0], exact_text="1. Rewritten.\n")
    tampered = replace(ledger, comments=(changed,) + ledger.comments[1:])
    with pytest.raises(SectionalRevisionContractError) as source_error:
        validate_review_ledger(tampered, reviews=reviews)
    assert {
        "comment_id_mismatch",
        "comment_hash_mismatch",
        "comment_source_mismatch",
    } <= _codes(source_error.value)


def test_review_ledger_source_closure_rejects_omitted_or_duplicated_comment() -> None:
    reviews = _sanitized_reviews()
    ledger = _ledger()

    omitted = replace(ledger, comments=ledger.comments[:-1])
    with pytest.raises(SectionalRevisionContractError) as omitted_error:
        validate_review_ledger(omitted, reviews=reviews)
    assert "ledger_source_closure_mismatch" in _codes(omitted_error.value)

    duplicated = replace(ledger, comments=ledger.comments + (ledger.comments[-1],))
    with pytest.raises(SectionalRevisionContractError) as duplicated_error:
        validate_review_ledger(duplicated, reviews=reviews)
    assert "ledger_source_closure_mismatch" in _codes(duplicated_error.value)


def test_review_ledger_required_policy_is_not_user_controlled() -> None:
    ledger = _ledger()
    changed = replace(ledger.comments[0], required=False)

    with pytest.raises(SectionalRevisionContractError) as caught:
        validate_review_ledger(replace(ledger, comments=(changed,) + ledger.comments[1:]))

    assert "required_policy_mismatch" in _codes(caught.value)


def test_review_ledger_closure_requires_every_final_status() -> None:
    with pytest.raises(SectionalRevisionContractError) as caught:
        validate_review_ledger(_ledger(), require_final=True)

    assert "ledger_not_closed" in _codes(caught.value)


def test_valid_revision_plan_closes_all_comments_and_sections() -> None:
    plan = _validate_plan(_valid_plan())

    assert len(plan.assignments) == 17
    assert all(assignment.disposition == "assigned" for assignment in plan.assignments)


def test_revision_plan_rejects_invalid_ledger_and_permissive_document() -> None:
    ledger = _ledger()
    changed = replace(ledger.comments[0], required=False)
    invalid_ledger = replace(ledger, comments=(changed,) + ledger.comments[1:])
    with pytest.raises(SectionalRevisionContractError) as ledger_error:
        _validate_plan(_valid_plan(), invalid_ledger)
    assert "required_policy_mismatch" in _codes(ledger_error.value)

    ambiguous = parse_manuscript(
        "# Paper\n\n## Method\n\nOne.\n\n## Method\n\nTwo.\n",
        strict=False,
    )
    plan = _valid_plan()
    plan["source_paper_sha256"] = ambiguous.source_sha256
    target = ambiguous.sections[1].section_id
    for assignment in plan["assignments"]:
        assignment["target_section_ids"] = [target]
    with pytest.raises(SectionalRevisionContractError) as document_error:
        _validate_plan(plan, ledger, ambiguous)
    assert "manuscript_structure_ambiguous" in _codes(document_error.value)


def test_revision_plan_rebinds_ledger_to_raw_reviews_before_closure() -> None:
    ledger = _ledger()
    omitted_ledger = replace(ledger, comments=ledger.comments[:-1])
    plan = _valid_plan()
    plan["assignments"].pop()

    with pytest.raises(SectionalRevisionContractError) as caught:
        _validate_plan(plan, omitted_ledger)

    assert "ledger_source_closure_mismatch" in _codes(caught.value)


def test_revision_plan_rejects_unknown_fields_at_every_level() -> None:
    root_extra = _valid_plan()
    root_extra["unexpected"] = True
    with pytest.raises(SectionalRevisionContractError) as root_error:
        _validate_plan(root_extra)
    assert "unknown_fields" in _codes(root_error.value)

    assignment_extra = _valid_plan()
    assignment_extra["assignments"][0]["unexpected"] = True
    with pytest.raises(SectionalRevisionContractError) as assignment_error:
        _validate_plan(assignment_extra)
    assert "unknown_fields" in _codes(assignment_error.value)


def test_revision_plan_rejects_missing_and_duplicate_comments() -> None:
    missing = _valid_plan()
    missing["assignments"].pop()
    with pytest.raises(SectionalRevisionContractError) as missing_error:
        _validate_plan(missing)
    assert "plan_comments_missing" in _codes(missing_error.value)

    duplicate = _valid_plan()
    duplicate["assignments"][-1] = copy.deepcopy(duplicate["assignments"][0])
    with pytest.raises(SectionalRevisionContractError) as duplicate_error:
        _validate_plan(duplicate)
    assert {
        "duplicate_plan_comment",
        "plan_comments_missing",
    } <= _codes(duplicate_error.value)


def test_revision_plan_rejects_unknown_comment_and_section() -> None:
    unknown_comment = _valid_plan()
    unknown_comment["assignments"][0]["comment_id"] = "rc-999-unknown"
    with pytest.raises(SectionalRevisionContractError) as comment_error:
        _validate_plan(unknown_comment)
    assert "unknown_plan_comment" in _codes(comment_error.value)

    unknown_section = _valid_plan()
    unknown_section["assignments"][0]["target_section_ids"] = ["s999-missing"]
    with pytest.raises(SectionalRevisionContractError) as section_error:
        _validate_plan(unknown_section)
    assert "unknown_section_id" in _codes(section_error.value)


def test_revision_plan_rejects_global_and_duplicate_targets() -> None:
    global_plan = _valid_plan()
    global_plan["assignments"][0]["target_section_ids"] = ["global"]
    with pytest.raises(SectionalRevisionContractError) as global_error:
        _validate_plan(global_plan)
    assert "global_target_not_executable" in _codes(global_error.value)

    duplicate = _valid_plan()
    target = duplicate["assignments"][0]["target_section_ids"][0]
    duplicate["assignments"][0]["target_section_ids"] = [target, target]
    with pytest.raises(SectionalRevisionContractError) as duplicate_error:
        _validate_plan(duplicate)
    assert "duplicate_plan_target" in _codes(duplicate_error.value)


@pytest.mark.parametrize("disposition", ["unresolved", "not_actionable_with_reason"])
def test_terminal_plan_disposition_requires_reason_and_no_targets(
    disposition: str,
) -> None:
    plan = _valid_plan()
    assignment = plan["assignments"][0]
    assignment["disposition"] = disposition
    assignment["reason"] = None

    with pytest.raises(SectionalRevisionContractError) as reason_error:
        _validate_plan(plan)
    assert "plan_reason_missing" in _codes(reason_error.value)

    assignment["reason"] = "No supporting experiment artifact exists."
    with pytest.raises(SectionalRevisionContractError) as target_error:
        _validate_plan(plan)
    assert "terminal_disposition_has_targets" in _codes(target_error.value)


def test_assigned_plan_requires_null_reason() -> None:
    plan = _valid_plan()
    plan["assignments"][0]["reason"] = "Writer supplied a reason."

    with pytest.raises(SectionalRevisionContractError) as caught:
        _validate_plan(plan)

    assert "assigned_has_reason" in _codes(caught.value)


def test_revision_plan_binds_paper_and_review_hashes() -> None:
    bad_paper = _valid_plan()
    bad_paper["source_paper_sha256"] = "0" * 64
    with pytest.raises(SectionalRevisionContractError) as paper_error:
        _validate_plan(bad_paper)
    assert "source_paper_hash_mismatch" in _codes(paper_error.value)

    bad_reviews = _valid_plan()
    bad_reviews["source_reviews_sha256"] = "0" * 64
    with pytest.raises(SectionalRevisionContractError) as reviews_error:
        _validate_plan(bad_reviews)
    assert "source_reviews_hash_mismatch" in _codes(reviews_error.value)


def test_attempt_id_uses_complete_section_id() -> None:
    first = make_attempt_id("s001-shared-prefix-aaaaaaaaaaaa", 1)
    second = make_attempt_id("s001-shared-prefix-bbbbbbbbbbbb", 1)

    assert first == "sec-s001-shared-prefix-aaaaaaaaaaaa-a1"
    assert second == "sec-s001-shared-prefix-bbbbbbbbbbbb-a1"
    assert first != second
