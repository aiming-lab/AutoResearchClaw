from __future__ import annotations

import json
from typing import Any, Callable

import pytest

from researchclaw.llm.client import LLMResponse
from researchclaw.pipeline.manuscript_sections import parse_manuscript
from researchclaw.pipeline.sectional_llm import (
    LLMSectionalRevisionProvider,
    _parse_json_object,
)
from researchclaw.pipeline.sectional_revision import extract_review_ledger
from researchclaw.pipeline.sectional_validation import SectionValidationContext


DRAFT = """## Method

The baseline score was 0.475 using \\cite{smith2024}.

## Results

SECRET_OTHER_SECTION must never enter the Method writer prompt.
"""

REVIEWS = """## Reviewer A

### Actionable Revisions
1. Clarify how the baseline score is reported.
"""


class _FakeLLM:
    def __init__(self, responder: Callable[[dict[str, Any], dict[str, Any]], object]):
        self.responder = responder
        self.calls: list[tuple[list[dict[str, str]], dict[str, Any]]] = []

    def chat(self, messages: list[dict[str, str]], **kwargs: Any) -> LLMResponse:
        self.calls.append((messages, kwargs))
        payload = json.loads(messages[0]["content"])
        content = self.responder(payload, kwargs)
        if isinstance(content, LLMResponse):
            return content
        if not isinstance(content, str):
            content = json.dumps(content)
        return LLMResponse(content=content, model=str(kwargs["model"]))


def _inputs():
    document = parse_manuscript(DRAFT)
    ledger = extract_review_ledger(REVIEWS, source_path="stage-18/reviews.md")
    method = next(section for section in document.sections if section.title == "Method")
    comment = ledger.comments[0]
    context = SectionValidationContext(
        document=document,
        section_id=method.section_id,
        attempt=1,
        allowed_citation_keys=frozenset({"smith2024"}),
        grounded_numeric_values=(0.475,),
        required_comment_ids=(comment.comment_id,),
    )
    return document, ledger, method, comment, context


def _provider(fake: _FakeLLM) -> LLMSectionalRevisionProvider:
    return LLMSectionalRevisionProvider(
        llm=fake,  # type: ignore[arg-type]
        writer_model="writer-model",
        critic_model="critic-model",
    )


def test_planner_uses_explicit_writer_model_and_strict_json() -> None:
    document, ledger, method, comment, _ = _inputs()

    def respond(payload: dict[str, Any], kwargs: dict[str, Any]) -> object:
        assert kwargs["model"] == "writer-model"
        assert kwargs["json_mode"] is True
        assert kwargs["temperature"] == 0
        assert len(payload["sections"]) == 2
        return {
            "schema_version": 1,
            "planner_version": 1,
            "source_paper_sha256": document.source_sha256,
            "source_reviews_sha256": ledger.source_reviews_sha256,
            "section_model_version": 1,
            "assignments": [
                {
                    "comment_id": comment.comment_id,
                    "target_section_ids": [method.section_id],
                    "disposition": "assigned",
                    "reason": None,
                }
            ],
        }

    plan = _provider(_FakeLLM(respond)).build_plan(ledger=ledger, document=document)
    assert plan["assignments"][0]["comment_id"] == comment.comment_id


def test_planner_rejects_unknown_top_level_fields() -> None:
    document, ledger, _, _, _ = _inputs()
    fake = _FakeLLM(
        lambda payload, kwargs: {
            "schema_version": 1,
            "planner_version": 1,
            "source_paper_sha256": document.source_sha256,
            "source_reviews_sha256": ledger.source_reviews_sha256,
            "section_model_version": 1,
            "assignments": [],
            "unexpected": True,
        }
    )
    with pytest.raises(RuntimeError, match="fields mismatch"):
        _provider(fake).build_plan(ledger=ledger, document=document)


@pytest.mark.parametrize(
    "response",
    (
        '{"schema_version": 1} trailing prose',
        "```json\n{\"schema_version\": 1}\n```junk",
        "[]",
    ),
)
def test_json_parser_rejects_non_bounded_responses(response: str) -> None:
    with pytest.raises(RuntimeError):
        _parse_json_object(response)


def test_json_parser_accepts_one_complete_fenced_object() -> None:
    assert _parse_json_object("```json\n{\"schema_version\": 1}\n```") == {
        "schema_version": 1
    }


def test_provider_rejects_truncated_but_valid_json_response() -> None:
    document, ledger, _, _, _ = _inputs()
    fake = _FakeLLM(
        lambda payload, kwargs: LLMResponse(
            content=json.dumps(
                {
                    "schema_version": 1,
                    "planner_version": 1,
                    "source_paper_sha256": document.source_sha256,
                    "source_reviews_sha256": ledger.source_reviews_sha256,
                    "section_model_version": 1,
                    "assignments": [],
                }
            ),
            model="writer-model",
            truncated=True,
        )
    )
    with pytest.raises(RuntimeError, match="truncated"):
        _provider(fake).build_plan(ledger=ledger, document=document)


def test_writer_receives_only_one_section_and_accounts_for_every_comment() -> None:
    _, _, method, comment, context = _inputs()

    def respond(payload: dict[str, Any], kwargs: dict[str, Any]) -> object:
        encoded = json.dumps(payload)
        assert "SECRET_OTHER_SECTION" not in encoded
        assert kwargs["model"] == "writer-model"
        return {
            "schema_version": 1,
            "section_id": method.section_id,
            "revised_body": (
                "\nThe baseline score was 0.475 using \\cite{smith2024}; "
                "this wording clarifies its reporting.\n"
            ),
            "resolutions": [
                {
                    "comment_id": comment.comment_id,
                    "writer_status": "addressed",
                    "reason": "The requested reporting clarification was added.",
                }
            ],
        }

    proposal = _provider(_FakeLLM(respond)).propose(
        section=method,
        comments=(comment,),
        attempt=1,
        context=context,
    )
    assert proposal.resolution_comment_ids == (comment.comment_id,)


def test_writer_rejects_missing_comment_resolution() -> None:
    _, _, method, comment, context = _inputs()
    fake = _FakeLLM(
        lambda payload, kwargs: {
            "schema_version": 1,
            "section_id": method.section_id,
            "revised_body": method.body,
            "resolutions": [],
        }
    )
    with pytest.raises(RuntimeError, match="every assigned comment"):
        _provider(fake).propose(
            section=method,
            comments=(comment,),
            attempt=1,
            context=context,
        )


def test_critic_uses_isolated_model_and_exact_identity() -> None:
    _, _, method, comment, _ = _inputs()

    def respond(payload: dict[str, Any], kwargs: dict[str, Any]) -> object:
        assert kwargs["model"] == "critic-model"
        assert len(payload["comment"]["exact_text"]) > 0
        return {
            "schema_version": 1,
            "comment_id": comment.comment_id,
            "section_id": method.section_id,
            "attempt_id": "attempt-1",
            "verdict": "resolved",
            "reason": "The revised sentence directly addresses the comment.",
        }

    fake = _FakeLLM(respond)
    assessment = _provider(fake).assess(
        comment=comment,
        section=method,
        original_body=method.body,
        revised_body=method.body + "\nClarified.\n",
        attempt_id="attempt-1",
        validator_codes=(),
    )
    assert assessment.context_isolated is True
    assert assessment.critic_model == "critic-model"
    assert fake.calls[0][0][0]["role"] == "user"


def test_critic_rejects_identity_claim_from_another_attempt() -> None:
    _, _, method, comment, _ = _inputs()
    fake = _FakeLLM(
        lambda payload, kwargs: {
            "schema_version": 1,
            "comment_id": comment.comment_id,
            "section_id": method.section_id,
            "attempt_id": "wrong-attempt",
            "verdict": "resolved",
            "reason": "Invalid identity claim.",
        }
    )
    with pytest.raises(RuntimeError, match="identity mismatches"):
        _provider(fake).assess(
            comment=comment,
            section=method,
            original_body=method.body,
            revised_body=method.body + "\nClarified.\n",
            attempt_id="attempt-1",
            validator_codes=(),
        )


def test_provider_rejects_same_writer_and_critic_model() -> None:
    fake = _FakeLLM(lambda payload, kwargs: {})
    with pytest.raises(ValueError, match="must differ"):
        LLMSectionalRevisionProvider(
            llm=fake,  # type: ignore[arg-type]
            writer_model="same-model",
            critic_model="same-model",
        )
