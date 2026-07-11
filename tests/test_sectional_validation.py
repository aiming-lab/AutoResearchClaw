from __future__ import annotations

import json
import random
import re
from dataclasses import replace

import pytest
from markdown_it.rules_core.normalize import NEWLINES_RE

from researchclaw.pipeline.manuscript_sections import (
    merge_manuscript,
    parse_manuscript,
    split_commonmark_lines_keepends,
)
from researchclaw.pipeline.sectional_revision import SectionalRevisionContractError
from researchclaw.pipeline.sectional_revision import (
    extract_review_ledger,
    validate_review_ledger,
    validate_revision_plan,
)
from researchclaw.pipeline.sectional_validation import (
    NUMERIC_REL_TOL,
    QUANTITATIVE_UNIT_LEXICON_VERSION,
    QUANTITATIVE_UNIT_LEXICON_V1,
    SectionValidationContext,
    SectionValidationResult,
    SectionAttemptRecord,
    SectionManifestMetadata,
    SectionRevisionManifest,
    ResolutionAssessmentRecord,
    ValidatedSectionReplacement,
    build_section_revision_manifest,
    build_unresolved_comments_artifact,
    canonical_json_sha256,
    extract_citation_keys,
    extract_declared_reference_targets,
    extract_quantitative_values,
    extract_reference_mentions,
    merge_validated_sections,
    parse_resolution_assessments_jsonl,
    parse_section_attempts_jsonl,
    validate_section_revision_manifest,
    validate_section_candidate,
)


SOURCE = (
    "# Paper\n\n"
    "Lead.\n\n"
    "## Method\n\n"
    "The detector scored 0.475 in the recorded run and used three seeds. "
    "See Table 1 for details \\cite{smith2024}.\n\n"
    "## Results\n\n"
    "**Table 1.** Recorded metrics.\n\n"
    "| Metric | Value |\n"
    "|---|---:|\n"
    "| F1 | 0.475 |\n"
)


def _document():
    return parse_manuscript(SOURCE)


def _method():
    return _document().sections[1]


def _candidate(extra: str = "") -> str:
    return (
        "\nThe recorded detector scored 0.475 and used three seeds. "
        "See Table 1 for details \\cite{smith2024}. "
        f"{extra}\n\n"
    )


def _context(**overrides):
    values = {
        "document": _document(),
        "section_id": _method().section_id,
        "attempt": 1,
        "allowed_citation_keys": frozenset({"smith2024", "jones2023"}),
        "grounded_numeric_values": (0.475, 0.85, 2.0, 3.0),
        "required_comment_ids": ("rc-001",),
        "resolution_comment_ids": ("rc-001",),
        "min_length_ratio": 0.50,
        "max_length_ratio": 2.50,
    }
    values.update(overrides)
    return SectionValidationContext(**values)


def _failed_codes(result: SectionValidationResult) -> set[str]:
    return {check.code for check in result.checks if check.status == "failed"}


def _manifest_inputs(*, final_status: str = "resolved", changed: bool = True):
    reviews = """## Reviewer A

### Actionable Revisions
1. Clarify the metric wording.
"""
    ledger = extract_review_ledger(reviews, source_path="stage-18/reviews.md")
    document = _document()
    section_id = document.sections[1].section_id
    plan = validate_revision_plan(
        {
            "schema_version": 1,
            "planner_version": 1,
            "source_paper_sha256": document.source_sha256,
            "source_reviews_sha256": ledger.source_reviews_sha256,
            "section_model_version": 1,
            "assignments": [
                {
                    "comment_id": ledger.comments[0].comment_id,
                    "target_section_ids": [section_id],
                    "disposition": "assigned",
                    "reason": None,
                }
            ],
        },
        ledger,
        document,
        reviews=reviews,
    )
    attempt_id = f"sec-{section_id}-a1"
    final_comment = replace(
        ledger.comments[0],
        working_status="assigned",
        target_section_ids=(section_id,),
        final_status=final_status,
        resolution_reason="Deterministic test outcome.",
        attempt_ids=(attempt_id,),
    )
    final_ledger = replace(ledger, comments=(final_comment,))
    validate_review_ledger(final_ledger, reviews=reviews, require_final=True)

    candidate = _candidate("The metric wording is now explicit.")
    context = _context(
        document=document,
        section_id=section_id,
        required_comment_ids=(final_comment.comment_id,),
        resolution_comment_ids=(final_comment.comment_id,),
    )
    validation = validate_section_candidate(context, candidate)
    if changed:
        replacement = ValidatedSectionReplacement(
            section_id=section_id,
            body=candidate,
            validation=validation,
            context=context,
        )
        merge_result = merge_validated_sections(document, {section_id: replacement})
        metadata = {
            section_id: SectionManifestMetadata(
                comment_ids=(final_comment.comment_id,),
                attempt_ids=(attempt_id,),
                final_status="accepted",
                validation_result=validation,
                validation_context=context,
            )
        }
    else:
        merge_result = merge_validated_sections(document, {})
        metadata = {
            section_id: SectionManifestMetadata(
                comment_ids=(final_comment.comment_id,),
                attempt_ids=(attempt_id,),
                final_status="unresolved_original_preserved",
                validation_result=None,
                validation_context=None,
            )
        }
    config_payload = {
        "max_section_retries": 1,
        "min_length_ratio": 0.50,
        "max_length_ratio": 2.50,
    }
    validation_context_text = json.dumps(
        {
            "schema_version": 1,
            "source_paper_sha256": document.source_sha256,
            "allowed_citation_keys": ["jones2023", "smith2024"],
            "grounded_numeric_values": [0.475, 0.85, 2.0, 3.0],
            **config_payload,
            "sources": [
                {
                    "kind": "citations",
                    "path": "stage-04/references.bib",
                    "sha256": "a" * 64,
                },
                {
                    "kind": "metrics",
                    "path": "stage-12/runs/results.json",
                    "sha256": "b" * 64,
                },
                {
                    "kind": "config",
                    "path": "config.paper_revision",
                    "sha256": canonical_json_sha256(config_payload),
                },
            ],
        },
        sort_keys=True,
        indent=2,
    ) + "\n"
    attempt_status = "accepted" if changed else "rejected"
    attempt_record = SectionAttemptRecord.from_dict(
        SectionAttemptRecord(
            schema_version=1,
            attempt_id=attempt_id,
            section_id=section_id,
            source_section_sha256=document.sections[1].original_sha256,
            comment_ids=(final_comment.comment_id,),
            resolution_comment_ids=(final_comment.comment_id,),
            writer_model="writer-model",
            attempt=1,
            status=attempt_status,
            candidate_path=f"stage-19/sections/{section_id}.attempt-1.md",
            candidate_body_sha256=validation.candidate_sha256,
            validation_report_path=(
                f"stage-19/section_validation/{section_id}.attempt-1.json"
            ),
            validation_report_sha256=canonical_json_sha256(validation.to_dict()),
            validator_codes=(),
            error_type=None,
            error=None if changed else "candidate not accepted",
            timestamp="2026-07-11T00:00:00+00:00",
        ).to_dict()
    )
    attempts_text = json.dumps(attempt_record.to_dict(), sort_keys=True) + "\n"
    assessment_records = ()
    if changed:
        assessment_records = (
            ResolutionAssessmentRecord.from_dict(
                ResolutionAssessmentRecord(
                    schema_version=1,
                    assessment_id=f"ra-{final_comment.comment_id}-{attempt_id}",
                    comment_id=final_comment.comment_id,
                    section_id=section_id,
                    attempt_id=attempt_id,
                    critic_model="critic-model",
                    context_isolated=True,
                    verdict="resolved",
                    reason="The deterministic fixture resolves the comment.",
                    timestamp="2026-07-11T00:00:00+00:00",
                ).to_dict()
            ),
        )
    assessments_text = "".join(
        json.dumps(record.to_dict(), sort_keys=True) + "\n"
        for record in assessment_records
    )
    return {
        "document": document,
        "merge_result": merge_result,
        "ledger": final_ledger,
        "plan": plan,
        "reviews": reviews,
        "experiment_contract_path": "stage-09/experiment_contract.yaml",
        "experiment_contract_sha256": "c" * 64,
        "writer_model": "writer-model",
        "critic_model": "critic-model",
        "source_paper_path": "stage-17/paper_draft.md",
        "section_metadata": metadata,
        "attempts_text": attempts_text,
        "assessments_text": assessments_text,
        "unresolved_comments_text": json.dumps(
            build_unresolved_comments_artifact(final_ledger),
            sort_keys=True,
        ),
        "completed": True,
        "validation_context_text": validation_context_text,
    }


def test_valid_candidate_passes_every_hard_check() -> None:
    result = validate_section_candidate(_context(), _candidate("The wording is clearer."))

    assert result.accepted is True
    assert _failed_codes(result) == set()
    assert SectionValidationResult.from_dict(result.to_dict()) == result


@pytest.mark.parametrize(
    ("candidate", "code"),
    [
        ("", "empty_section_body"),
        (_candidate("\n### Hidden Heading\n"), "new_heading_introduced"),
        (_candidate("\nHidden\n------\n"), "new_heading_introduced"),
        (_candidate("<H2 class='x'>Hidden</H2>"), "html_heading_introduced"),
        (_candidate("\n```python\nprint('open')\n"), "markdown_structure_unbalanced"),
        (_candidate("$$ x = y"), "markdown_structure_unbalanced"),
    ],
)
def test_structural_mutations_fail(candidate: str, code: str) -> None:
    result = validate_section_candidate(_context(), candidate)

    assert result.accepted is False
    assert code in _failed_codes(result)


def test_code_fence_contents_do_not_trigger_html_reference_or_math_checks() -> None:
    candidate = _candidate(
        "\n```html\n<h1>Example</h1>\nFigure 99\n$$\n```\n"
    )

    result = validate_section_candidate(_context(), candidate)

    assert {
        "html_heading_introduced",
        "unknown_reference_introduced",
        "markdown_structure_unbalanced",
        "unknown_numeric_value",
    }.isdisjoint(_failed_codes(result))

    inline = validate_section_candidate(
        _context(),
        _candidate("Literal `<h1>Figure 99 scored 0.99</h1>` is code."),
    )
    assert {
        "html_heading_introduced",
        "unknown_reference_introduced",
        "unknown_numeric_value",
    }.isdisjoint(_failed_codes(inline))


def test_candidate_without_final_newline_cannot_swallow_next_heading() -> None:
    result = validate_section_candidate(
        _context(min_length_ratio=0.01),
        "The detector scored 0.475 \\cite{smith2024}. See Table 1",
    )

    assert {
        "post_merge_heading_mismatch",
        "section_order_mismatch",
    } <= _failed_codes(result)


def test_citation_introduction_and_removal_fail_closed() -> None:
    unknown = validate_section_candidate(_context(), _candidate("\\cite{fake2025}"))
    assert "unknown_citation_key" in _failed_codes(unknown)

    removed = validate_section_candidate(
        _context(),
        _candidate().replace(r" \cite{smith2024}", ""),
    )
    assert "required_citation_removed" in _failed_codes(removed)

    allowed = validate_section_candidate(_context(), _candidate("\\cite{jones2023}"))
    assert "unknown_citation_key" not in _failed_codes(allowed)


def test_reference_introduction_and_removal_fail_closed() -> None:
    unknown = validate_section_candidate(_context(), _candidate("See Figure 2."))
    assert "unknown_reference_introduced" in _failed_codes(unknown)

    removed = validate_section_candidate(
        _context(),
        _candidate().replace("See Table 1 for details ", "Details appear below "),
    )
    assert "required_reference_removed" in _failed_codes(removed)

    declared = validate_section_candidate(_context(), _candidate("Table 1 remains grounded."))
    assert "unknown_reference_introduced" not in _failed_codes(declared)


def test_plural_reference_mentions_are_not_invisible() -> None:
    mentions = extract_reference_mentions(
        "See Figures 2, 3, and 8; Tables 4 & 5; and Eqs. 6, 7."
    )
    assert mentions == frozenset(
        {
            ("figure", "2"),
            ("figure", "3"),
            ("figure", "8"),
            ("table", "4"),
            ("table", "5"),
            ("equation", "6"),
            ("equation", "7"),
        }
    )

    result = validate_section_candidate(_context(), _candidate("See Figures 2 and 3."))
    assert "unknown_reference_introduced" in _failed_codes(result)


@pytest.mark.parametrize("form", ["0.85", "85%", "8.5e-1"])
def test_numeric_equivalent_forms_match_grounded_value(form: str) -> None:
    result = validate_section_candidate(_context(), _candidate(f"Recall was {form}."))

    assert "unknown_numeric_value" not in _failed_codes(result)


def test_digit_percent_word_normalizes_to_fraction() -> None:
    result = validate_section_candidate(_context(), _candidate("Recall was 85 percent."))

    assert "unknown_numeric_value" not in _failed_codes(result)


def test_material_rounding_change_is_rejected_at_fixed_tolerance() -> None:
    assert NUMERIC_REL_TOL == 1e-3

    result = validate_section_candidate(_context(), _candidate("A second score was 0.48."))

    assert "unknown_numeric_value" in _failed_codes(result)


def test_number_words_parse_or_fail_explicitly() -> None:
    assert QUANTITATIVE_UNIT_LEXICON_VERSION == 1
    assert {"trials", "seeds", "percent"} <= QUANTITATIVE_UNIT_LEXICON_V1

    parsed = validate_section_candidate(_context(), _candidate("We ran two trials."))
    assert "unknown_numeric_value" not in _failed_codes(parsed)
    assert "unparsed_quantitative_expression" not in _failed_codes(parsed)

    ambiguous = validate_section_candidate(_context(), _candidate("We ran several trials."))
    assert "unparsed_quantitative_expression" in _failed_codes(ambiguous)


def test_years_and_reference_labels_are_not_treated_as_metrics() -> None:
    result = validate_section_candidate(
        _context(), _candidate("Smith (2024) also discusses Table 1."),
    )

    assert "unknown_numeric_value" not in _failed_codes(result)


@pytest.mark.parametrize(
    "phrase",
    ["2000 samples", "2048 windows", "1000 trials", "1899 samples", "2100 samples"],
)
def test_plain_four_digit_quantities_are_not_exempt_as_years(phrase: str) -> None:
    result = validate_section_candidate(_context(), _candidate(f"We processed {phrase}."))

    assert "unknown_numeric_value" in _failed_codes(result)


def test_math_decimal_metric_is_grounded_but_structural_integers_are_ignored() -> None:
    fabricated = validate_section_candidate(
        _context(), _candidate(r"The result was $F_1 = 0.97$."),
    )
    structural = validate_section_candidate(
        _context(), _candidate(r"We use $x^2$ and $\frac{1}{2}$ in the definition."),
    )

    assert "unknown_numeric_value" in _failed_codes(fabricated)
    assert "unknown_numeric_value" not in _failed_codes(structural)


def test_length_and_resolution_guards_fail_closed() -> None:
    shrink = validate_section_candidate(
        _context(min_length_ratio=0.95),
        "\n0.475 three seeds Table 1 \\cite{smith2024}.\n",
    )
    assert "abnormal_section_shrink" in _failed_codes(shrink)

    growth = validate_section_candidate(
        _context(max_length_ratio=1.01),
        _candidate(" ".join(["extra"] * 100)),
    )
    assert "abnormal_section_growth" in _failed_codes(growth)

    unresolved = validate_section_candidate(
        _context(resolution_comment_ids=()), _candidate()
    )
    assert "unaddressed_required_comment" in _failed_codes(unresolved)

    unknown_resolution = validate_section_candidate(
        _context(resolution_comment_ids=("rc-001", "rc-unknown")), _candidate()
    )
    assert "unaddressed_required_comment" in _failed_codes(unknown_resolution)


def test_validation_context_rejects_nonfinite_grounded_values() -> None:
    with pytest.raises(SectionalRevisionContractError) as caught:
        validate_section_candidate(
            _context(grounded_numeric_values=(float("nan"),)), _candidate()
        )

    assert any(
        issue.code == "grounded_numeric_value_invalid" for issue in caught.value.issues
    )


def test_validation_result_rejects_unknown_fields_and_forged_acceptance() -> None:
    result = validate_section_candidate(_context(), _candidate())
    payload = result.to_dict()
    payload["unexpected"] = True
    with pytest.raises(SectionalRevisionContractError):
        SectionValidationResult.from_dict(payload)

    forged = result.to_dict()
    forged["accepted"] = False
    with pytest.raises(SectionalRevisionContractError) as caught:
        SectionValidationResult.from_dict(forged)
    assert any(
        issue.code == "validation_acceptance_mismatch" for issue in caught.value.issues
    )

    bad_attempt = result.to_dict()
    bad_attempt["attempt_id"] = "sec-truncated-a1"
    with pytest.raises(SectionalRevisionContractError) as attempt_error:
        SectionValidationResult.from_dict(bad_attempt)
    assert any(issue.code == "attempt_id_invalid" for issue in attempt_error.value.issues)


def test_merge_accepts_only_hash_bound_validation_results() -> None:
    candidate = _candidate("The wording is clearer.")
    validation = validate_section_candidate(_context(), candidate)
    replacement = ValidatedSectionReplacement(
        section_id=_method().section_id,
        body=candidate,
        validation=validation,
        context=_context(),
    )

    merged = merge_validated_sections(_document(), {_method().section_id: replacement})

    assert candidate in merged.merged_text
    assert merged.source_paper_sha256 == _document().source_sha256
    assert merged.merged_paper_sha256 != merged.source_paper_sha256
    assert [item.section_id for item in merged.sections] == [
        item.section_id for item in _document().sections
    ]
    assert merge_manuscript(parse_manuscript(merged.merged_text)) == merged.merged_text

    tampered = replace(replacement, body=candidate + "tampered")
    with pytest.raises(SectionalRevisionContractError) as caught:
        merge_validated_sections(_document(), {_method().section_id: tampered})
    assert any(
        issue.code == "validation_candidate_hash_mismatch"
        for issue in caught.value.issues
    )


def test_merge_recomputes_validation_instead_of_trusting_forged_green_artifact() -> None:
    body = _candidate("Unsupported score 0.99.")
    failed = validate_section_candidate(_context(), body)
    assert failed.accepted is False
    forged_payload = failed.to_dict()
    forged_payload["accepted"] = True
    for check in forged_payload["checks"]:
        check["status"] = "passed"
        check["details"] = []
    forged = SectionValidationResult.from_dict(forged_payload)

    with pytest.raises(SectionalRevisionContractError) as caught:
        merge_validated_sections(
            _document(),
            {
                _method().section_id: ValidatedSectionReplacement(
                    _method().section_id,
                    body,
                    forged,
                    _context(),
                )
            },
        )

    assert any(
        issue.code == "validation_recompute_mismatch" for issue in caught.value.issues
    )


def test_merge_rejects_failed_validation_and_preserves_unchanged_bytes() -> None:
    failed_body = _candidate("An unsupported score was 0.99.")
    failed_validation = validate_section_candidate(_context(), failed_body)
    assert failed_validation.accepted is False
    with pytest.raises(SectionalRevisionContractError) as caught:
        merge_validated_sections(
            _document(),
            {
                _method().section_id: ValidatedSectionReplacement(
                    _method().section_id, failed_body, failed_validation, _context()
                )
            },
        )
    assert any(issue.code == "unaccepted_section_revision" for issue in caught.value.issues)

    unchanged = merge_validated_sections(_document(), {})
    assert unchanged.merged_text == SOURCE
    assert all(not item.changed for item in unchanged.sections)


def test_manifest_is_derived_from_authoritative_inputs_and_round_trips() -> None:
    inputs = _manifest_inputs()

    manifest = build_section_revision_manifest(
        claim_scope="pipeline_validation", **inputs
    )
    restored = SectionRevisionManifest.from_dict(manifest.to_dict())
    validated = validate_section_revision_manifest(
        restored,
        claim_scope="pipeline_validation",
        **inputs,
    )

    assert validated == manifest
    assert manifest.completed is True
    assert dict(manifest.comment_counts) == {
        "input": 1,
        "resolved": 1,
        "unresolved": 0,
        "not_actionable_with_reason": 0,
    }
    assert manifest.ledger_sha256 == canonical_json_sha256(
        inputs["ledger"].to_dict()
    )
    assert manifest.plan_sha256 == canonical_json_sha256(inputs["plan"].to_dict())
    assert manifest.experiment_contract_path == "stage-09/experiment_contract.yaml"
    assert manifest.experiment_contract_sha256 == "c" * 64
    assert manifest.writer_model == "writer-model"
    assert manifest.critic_model == "critic-model"
    assert [entry.section_id for entry in manifest.sections] == [
        section.section_id for section in inputs["document"].sections
    ]


def test_strict_attempt_and_assessment_loaders_round_trip() -> None:
    inputs = _manifest_inputs()
    attempts = parse_section_attempts_jsonl(inputs["attempts_text"])
    assessments = parse_resolution_assessments_jsonl(inputs["assessments_text"])

    assert len(attempts) == 1
    assert attempts[0].candidate_path and attempts[0].candidate_path.startswith(
        "stage-19/sections/"
    )
    assert attempts[0].resolution_comment_ids == attempts[0].comment_ids
    assert len(assessments) == 1
    assert assessments[0].context_isolated is True


def test_attempt_loader_rejects_unknown_fields_and_invalid_transport_state() -> None:
    inputs = _manifest_inputs()
    payload = json.loads(inputs["attempts_text"])
    payload["unexpected"] = True
    with pytest.raises(SectionalRevisionContractError):
        parse_section_attempts_jsonl(json.dumps(payload) + "\n")

    payload.pop("unexpected")
    payload["status"] = "transport_failed"
    payload["error_type"] = "RuntimeError"
    payload["error"] = "network failed"
    with pytest.raises(SectionalRevisionContractError) as caught:
        parse_section_attempts_jsonl(json.dumps(payload) + "\n")
    assert any(issue.code == "attempt_state_invalid" for issue in caught.value.issues)


def test_jsonl_loaders_reject_duplicate_keys_and_duplicate_ids() -> None:
    with pytest.raises(SectionalRevisionContractError) as duplicate_key:
        parse_section_attempts_jsonl('{"schema_version":1,"schema_version":1}\n')
    assert any(issue.code == "jsonl_invalid" for issue in duplicate_key.value.issues)

    inputs = _manifest_inputs()
    with pytest.raises(SectionalRevisionContractError) as duplicate_id:
        parse_resolution_assessments_jsonl(
            inputs["assessments_text"] + inputs["assessments_text"]
        )
    assert any(
        issue.code == "duplicate_assessment_id" for issue in duplicate_id.value.issues
    )


def test_jsonl_loader_preserves_unicode_line_separator_inside_string() -> None:
    inputs = _manifest_inputs()
    assessment = json.loads(inputs["assessments_text"])
    assessment["reason"] = "First clause.\u2028Second clause."
    text = json.dumps(assessment, ensure_ascii=False, sort_keys=True) + "\n"

    records = parse_resolution_assessments_jsonl(text)

    assert records[0].reason == "First clause.\u2028Second clause."


@pytest.mark.parametrize(
    ("artifact_key", "model_key", "model_value", "expected_code"),
    (
        (
            "attempts_text",
            "writer_model",
            "third-writer-model",
            "manifest_writer_model_mismatch",
        ),
        (
            "assessments_text",
            "critic_model",
            "third-critic-model",
            "manifest_critic_model_mismatch",
        ),
    ),
)
def test_manifest_rejects_bundle_model_identity_drift(
    artifact_key: str,
    model_key: str,
    model_value: str,
    expected_code: str,
) -> None:
    inputs = _manifest_inputs()
    record = json.loads(inputs[artifact_key])
    record[model_key] = model_value
    inputs[artifact_key] = json.dumps(record, sort_keys=True) + "\n"

    with pytest.raises(SectionalRevisionContractError) as caught:
        build_section_revision_manifest(claim_scope="pipeline_validation", **inputs)
    assert any(issue.code == expected_code for issue in caught.value.issues)


@pytest.mark.parametrize("attempt", (0, 5))
def test_attempt_loader_rejects_unbounded_attempt_ordinal(attempt: int) -> None:
    inputs = _manifest_inputs()
    payload = json.loads(inputs["attempts_text"])
    payload["attempt"] = attempt

    with pytest.raises(SectionalRevisionContractError) as caught:
        parse_section_attempts_jsonl(json.dumps(payload) + "\n")

    assert any(issue.code == "attempt_id_invalid" for issue in caught.value.issues)


@pytest.mark.parametrize(
    "field",
    [
        "source_paper_sha256",
        "source_reviews_sha256",
        "ledger_sha256",
        "plan_sha256",
        "attempts_sha256",
        "experiment_contract_sha256",
        "assessments_sha256",
        "unresolved_comments_sha256",
        "validation_context_sha256",
        "merged_paper_sha256",
    ],
)
def test_manifest_recompute_rejects_tampered_hashes(field: str) -> None:
    inputs = _manifest_inputs()
    manifest = build_section_revision_manifest(
        claim_scope="pipeline_validation", **inputs
    ).to_dict()
    manifest[field] = "0" * 64

    with pytest.raises(SectionalRevisionContractError) as caught:
        validate_section_revision_manifest(
            manifest,
            claim_scope="pipeline_validation",
            **inputs,
        )

    assert any(issue.code == "manifest_recompute_mismatch" for issue in caught.value.issues)


def test_manifest_rejects_section_order_and_nested_unknown_fields() -> None:
    inputs = _manifest_inputs()
    payload = build_section_revision_manifest(
        claim_scope="pipeline_validation", **inputs
    ).to_dict()
    payload["sections"] = list(reversed(payload["sections"]))
    with pytest.raises(SectionalRevisionContractError) as order_error:
        validate_section_revision_manifest(
            payload,
            claim_scope="pipeline_validation",
            **inputs,
        )
    assert any(
        issue.code == "manifest_recompute_mismatch" for issue in order_error.value.issues
    )

    payload = build_section_revision_manifest(
        claim_scope="pipeline_validation", **inputs
    ).to_dict()
    payload["sections"][0]["unexpected"] = True
    with pytest.raises(SectionalRevisionContractError) as field_error:
        SectionRevisionManifest.from_dict(payload)
    assert any(issue.code == "unknown_fields" for issue in field_error.value.issues)


def test_manifest_rejects_noncanonical_validation_context_path() -> None:
    inputs = _manifest_inputs()
    payload = build_section_revision_manifest(
        claim_scope="pipeline_validation", **inputs
    ).to_dict()
    payload["validation_context_path"] = "stage-18/validation_context.json"

    with pytest.raises(SectionalRevisionContractError) as caught:
        SectionRevisionManifest.from_dict(payload)

    assert any(
        issue.code == "validation_context_path_invalid"
        for issue in caught.value.issues
    )


@pytest.mark.parametrize(
    ("mutate", "expected_code"),
    (
        (lambda payload: payload.clear(), "missing_fields"),
        (
            lambda payload: payload["sources"].append(
                {
                    "kind": "metrics",
                    "path": "stage-10/smoke/smoke_results.json",
                    "sha256": "c" * 64,
                }
            ),
            "validation_context_stage10_source",
        ),
        (
            lambda payload: payload.__setitem__(
                "grounded_numeric_values", [0.999]
            ),
            "validation_context_numeric_mismatch",
        ),
    ),
)
def test_manifest_rejects_invalid_or_unbound_validation_context(
    mutate,
    expected_code: str,
) -> None:
    inputs = _manifest_inputs()
    payload = json.loads(inputs["validation_context_text"])
    mutate(payload)
    inputs["validation_context_text"] = json.dumps(
        payload, sort_keys=True, indent=2
    ) + "\n"

    with pytest.raises(SectionalRevisionContractError) as caught:
        build_section_revision_manifest(
            claim_scope="pipeline_validation", **inputs
        )

    assert any(issue.code == expected_code for issue in caught.value.issues)


def test_manifest_recomputes_assessment_and_unresolved_artifact_hashes() -> None:
    inputs = _manifest_inputs()
    manifest = build_section_revision_manifest(
        claim_scope="pipeline_validation", **inputs
    )
    changed = dict(inputs)
    assessment = json.loads(changed["assessments_text"])
    assessment["reason"] = "Tampered but schema-valid reason."
    changed["assessments_text"] = json.dumps(assessment, sort_keys=True) + "\n"

    with pytest.raises(SectionalRevisionContractError) as caught:
        validate_section_revision_manifest(
            manifest,
            claim_scope="pipeline_validation",
            **changed,
        )

    assert any(issue.code == "manifest_recompute_mismatch" for issue in caught.value.issues)

    changed = dict(inputs)
    unresolved = json.loads(changed["unresolved_comments_text"])
    unresolved["comments"].append(
        {
            "comment_id": "rc-fake",
            "final_status": "unresolved",
            "reason": "fabricated",
        }
    )
    changed["unresolved_comments_text"] = json.dumps(unresolved, sort_keys=True)
    with pytest.raises(SectionalRevisionContractError) as unresolved_error:
        validate_section_revision_manifest(
            manifest,
            claim_scope="pipeline_validation",
            **changed,
        )
    assert any(
        issue.code == "unresolved_artifact_mismatch"
        for issue in unresolved_error.value.issues
    )


def test_manifest_rejects_forged_merge_result_fields() -> None:
    inputs = _manifest_inputs()
    records = list(inputs["merge_result"].sections)
    method_index = 1
    records[method_index] = replace(records[method_index], changed=False)
    inputs["merge_result"] = replace(
        inputs["merge_result"], sections=tuple(records)
    )

    with pytest.raises(SectionalRevisionContractError) as caught:
        build_section_revision_manifest(
            claim_scope="pipeline_validation", **inputs
        )

    assert any(issue.code == "merge_changed_flag_mismatch" for issue in caught.value.issues)


def test_incomplete_manifest_allows_assigned_comment_without_attempt_but_not_done() -> None:
    inputs = _manifest_inputs()
    initial_ledger = extract_review_ledger(
        inputs["reviews"], source_path="stage-18/reviews.md"
    )
    incomplete = dict(inputs)
    incomplete.update(
        {
            "ledger": initial_ledger,
            "merge_result": merge_validated_sections(inputs["document"], {}),
            "section_metadata": {},
            "unresolved_comments_text": json.dumps(
                build_unresolved_comments_artifact(initial_ledger),
                sort_keys=True,
            ),
            "completed": False,
        }
    )

    manifest = build_section_revision_manifest(
        claim_scope="pipeline_validation", **incomplete
    )

    assert manifest.completed is False
    assert dict(manifest.comment_counts)["resolved"] == 0

    incomplete["completed"] = True
    with pytest.raises(SectionalRevisionContractError):
        build_section_revision_manifest(
            claim_scope="pipeline_validation", **incomplete
        )


def test_pipeline_validation_can_record_unresolved_original_but_strict_scope_cannot() -> None:
    inputs = _manifest_inputs(final_status="unresolved", changed=False)

    manifest = build_section_revision_manifest(
        claim_scope="pipeline_validation", **inputs
    )
    assert manifest.completed is True
    assert dict(manifest.comment_counts)["unresolved"] == 1

    with pytest.raises(SectionalRevisionContractError) as caught:
        build_section_revision_manifest(claim_scope="research_release", **inputs)
    assert any(
        issue.code == "manifest_required_comment_unresolved"
        for issue in caught.value.issues
    )


def test_manifest_rejects_accepted_state_without_change_or_validation_hash() -> None:
    inputs = _manifest_inputs(final_status="resolved", changed=False)
    section_id = inputs["document"].sections[1].section_id
    inputs["section_metadata"] = {
        section_id: SectionManifestMetadata(
            comment_ids=(inputs["ledger"].comments[0].comment_id,),
            attempt_ids=(inputs["ledger"].comments[0].attempt_ids[0],),
            final_status="accepted",
            validation_result=None,
            validation_context=None,
        )
    }

    with pytest.raises(SectionalRevisionContractError) as caught:
        build_section_revision_manifest(
            claim_scope="pipeline_validation", **inputs
        )

    assert any(issue.code == "manifest_accepted_state_invalid" for issue in caught.value.issues)


def test_manifest_rejects_resolved_comment_with_unresolved_preserved_section() -> None:
    inputs = _manifest_inputs(final_status="resolved", changed=False)

    with pytest.raises(SectionalRevisionContractError) as caught:
        build_section_revision_manifest(
            claim_scope="pipeline_validation", **inputs
        )

    assert any(
        issue.code == "manifest_resolved_section_not_accepted"
        for issue in caught.value.issues
    )


def test_manifest_attempt_id_must_bind_complete_section_id() -> None:
    inputs = _manifest_inputs()
    section_id = inputs["document"].sections[1].section_id
    metadata = inputs["section_metadata"][section_id]
    inputs["section_metadata"] = {
        section_id: replace(metadata, attempt_ids=("sec-truncated-a1",))
    }

    with pytest.raises(SectionalRevisionContractError) as caught:
        build_section_revision_manifest(
            claim_scope="pipeline_validation", **inputs
        )

    assert any(issue.code == "manifest_attempt_id_invalid" for issue in caught.value.issues)


def test_manifest_validation_context_cannot_omit_required_plan_comments() -> None:
    inputs = _manifest_inputs()
    section_id = inputs["document"].sections[1].section_id
    metadata = inputs["section_metadata"][section_id]
    assert metadata.validation_context is not None
    context = replace(
        metadata.validation_context,
        required_comment_ids=(),
        resolution_comment_ids=(),
    )
    final_body = parse_manuscript(inputs["merge_result"].merged_text).sections[1].body
    validation = validate_section_candidate(context, final_body)
    assert validation.accepted is True
    inputs["section_metadata"] = {
        section_id: replace(
            metadata,
            validation_context=context,
            validation_result=validation,
        )
    }

    with pytest.raises(SectionalRevisionContractError) as caught:
        build_section_revision_manifest(
            claim_scope="pipeline_validation", **inputs
        )

    assert any(
        issue.code == "manifest_validation_comment_mismatch"
        for issue in caught.value.issues
    )


def test_extractors_ignore_code_but_scan_metric_literals_in_math() -> None:
    text = (
        r"See Table 2 and \cite{smith2024}. "
        "```markdown\nTable 9 [fake2025] 0.99\n```\n"
        "Inline $2025 + 0.88$ math."
    )

    assert extract_citation_keys(text) == frozenset({"smith2024"})
    assert extract_reference_mentions(text) >= frozenset({("table", "2")})
    values, unparsed = extract_quantitative_values(text)
    assert values == (0.88,)
    assert unparsed == ()


def test_pandoc_at_citations_are_explicitly_unsupported() -> None:
    text = "Contact author@example.com; Pandoc-style @smith2024 is not a v1 citation."

    assert extract_citation_keys(text) == frozenset()


def test_declared_reference_targets_require_caption_like_syntax() -> None:
    text = "Table 1 shows data.\n\n**Table 2.** Caption.\n\nFigure 3: Caption.\n"

    assert extract_declared_reference_targets(text) == frozenset(
        {("table", "2"), ("figure", "3")}
    )


def test_commonmark_line_splitter_matches_markdown_it_protocol_property() -> None:
    rng = random.Random(20260710)
    alphabet = ["a", "b", " ", "\n", "\r", "\u2028", "\f", "\x85"]
    for _ in range(1000):
        source = "".join(rng.choice(alphabet) for _ in range(rng.randrange(0, 100)))
        lines = split_commonmark_lines_keepends(source)
        normalized = NEWLINES_RE.sub("\n", source)
        expected_count = normalized.count("\n")
        if normalized and not normalized.endswith("\n"):
            expected_count += 1
        assert "".join(lines) == source
        assert len(lines) == expected_count
        reference = re.findall(r".*?(?:\r\n|\r|\n)|.+\Z", source, flags=re.DOTALL)
        assert lines == reference
