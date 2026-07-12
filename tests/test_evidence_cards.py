"""Stage 6 strict evidence-card and deterministic-renderer tests."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from researchclaw.adapters import AdapterBundle
from researchclaw.literature.citation_identity import seal_citation_collection
from researchclaw.literature.evidence_cards import (
    EvidenceCardContractError,
    build_cards_manifest,
    build_evidence_card,
    canonical_json_text,
    parse_card_batch_response,
    parse_evidence_card,
    render_evidence_card_markdown,
    validate_cards_artifacts,
)
from researchclaw.literature.screening import sha256_text
from researchclaw.pipeline.stage_impls._literature import (
    _execute_knowledge_extract,
    _execute_literature_screen,
)
from researchclaw.pipeline.stage_impls._synthesis import _execute_synthesis
from researchclaw.pipeline.stages import StageStatus


class _SequenceLLM:
    def __init__(self, responses: list[str]) -> None:
        self.responses = list(responses)
        self.calls: list[str] = []

    def chat(
        self, messages: list[dict[str, str]], **_kwargs: object
    ) -> SimpleNamespace:
        self.calls.append(messages[0]["content"])
        if not self.responses:
            raise RuntimeError("unexpected extra LLM call")
        return SimpleNamespace(content=self.responses.pop(0))


def _config(claim_scope: str = "pipeline_validation") -> SimpleNamespace:
    return SimpleNamespace(
        research=SimpleNamespace(
            topic="hardware runtime detection",
            domains=("hardware security",),
            quality_threshold=6.0,
        ),
        experiment=SimpleNamespace(claim_scope=claim_scope),
    )


def _candidate(index: int, *, abstract: str | None = None) -> dict[str, Any]:
    return {
        "paper_id": f"provider-{index}",
        "title": f"Hardware Detection Study {index}",
        "authors": [{"name": "Jane Smith"}],
        "year": 2024,
        "abstract": abstract
        or "Hardware runtime detection uses performance counters for attacks.",
        "venue": "Security Conference",
        "citation_count": 100 - index,
        "doi": f"10.1000/{index:03d}",
        "arxiv_id": "",
        "url": "",
        "source": "semantic_scholar",
    }


def _screen_response(source_ids: list[str]) -> str:
    return json.dumps(
        {
            "schema_version": 1,
            "batch_id": "screen-batch-001",
            "decisions": [
                {
                    "source_identity": source_id,
                    "decision": "keep",
                    "relevance_score": 0.9,
                    "quality_score": 0.8,
                    "reason": "directly relevant",
                }
                for source_id in source_ids
            ],
        }
    )


def _card_response(rows: list[dict[str, Any]]) -> str:
    return json.dumps(
        {
            "schema_version": 1,
            "batch_id": "card-batch-001",
            "cards": [
                {
                    "source_identity": row["source_identity"],
                    "summary_text": {
                        "problem": "Detect attacks at runtime.",
                        "method": "Use hardware counters.",
                        "data": "Retained abstract evidence.",
                        "metrics": "Detection performance.",
                        "findings": "Counters expose attack behavior.",
                        "limitations": "The abstract reports limited detail.",
                    },
                    "evidence_excerpt_texts": [row["abstract"]],
                }
                for row in rows
            ],
        },
        ensure_ascii=False,
    )


def _prepare_stage5(run_dir: Path, candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    sealed = seal_citation_collection(candidates)
    stage4 = run_dir / "stage-04"
    stage4.mkdir(parents=True)
    (stage4 / "candidates.jsonl").write_text(sealed.candidates_jsonl, encoding="utf-8")
    (stage4 / "references.bib").write_text(sealed.bibliography, encoding="utf-8")
    (stage4 / "cite_key_registry.json").write_text(
        canonical_json_text(sealed.registry), encoding="utf-8"
    )
    source_ids = [str(row["source_identity"]) for row in sealed.candidates]
    stage5 = run_dir / "stage-05"
    stage5.mkdir()
    result = _execute_literature_screen(
        stage5,
        run_dir,
        _config(),  # type: ignore[arg-type]
        AdapterBundle(),
        llm=_SequenceLLM([_screen_response(source_ids)]),  # type: ignore[arg-type]
    )
    assert result.status is StageStatus.DONE
    return [
        json.loads(line)
        for line in (stage5 / "shortlist.jsonl").read_text(encoding="utf-8").splitlines()
    ]


def test_card_batch_requires_exact_identity_closure() -> None:
    response = _card_response(
        [
            {
                "source_identity": "doi:one",
                "abstract": "Substantive retained abstract evidence.",
            }
        ]
    )
    with pytest.raises(EvidenceCardContractError, match="closure mismatch"):
        parse_card_batch_response(
            response,
            expected_batch_id="card-batch-001",
            expected_source_ids=("doi:one", "doi:two"),
        )


def test_card_batch_rejects_duplicate_json_keys() -> None:
    response = (
        '{"schema_version":1,"schema_version":1,'
        '"batch_id":"card-batch-001","cards":[]}'
    )
    with pytest.raises(EvidenceCardContractError, match="duplicate JSON key"):
        parse_card_batch_response(
            response,
            expected_batch_id="card-batch-001",
            expected_source_ids=(),
        )


@pytest.mark.parametrize(
    ("excerpt", "accepted"),
    [("x" * 24, False), ("x" * 25, True)],
)
def test_card_batch_enforces_minimum_excerpt_length(
    excerpt: str, accepted: bool
) -> None:
    candidate = {"source_identity": "doi:one", "abstract": excerpt}
    response = _card_response([candidate])
    if accepted:
        proposals = parse_card_batch_response(
            response,
            expected_batch_id="card-batch-001",
            expected_source_ids=("doi:one",),
        )
        assert proposals[0].excerpt_texts == (excerpt,)
    else:
        with pytest.raises(EvidenceCardContractError, match="at least 25"):
            parse_card_batch_response(
                response,
                expected_batch_id="card-batch-001",
                expected_source_ids=("doi:one",),
            )


def test_evidence_span_is_exact_for_unicode_text() -> None:
    candidate = _candidate(
        1, abstract="Alpha\u2028beta U0001f680 gamma provides retained evidence."
    )
    candidate.update(
        {"source_identity": "doi:10.1000/001", "cite_key": "smith2024alpha"}
    )
    proposal = parse_card_batch_response(
        _card_response([candidate]),
        expected_batch_id="card-batch-001",
        expected_source_ids=(candidate["source_identity"],),
    )[0]
    card = build_evidence_card(
        card_id="card-001",
        candidate=candidate,
        candidates_sha256="a" * 64,
        proposal=proposal,
    )
    excerpt = card["evidence_excerpts"][0]
    assert candidate["abstract"][excerpt["char_start"]:excerpt["char_end"]] == excerpt[
        "excerpt_text"
    ]
    parse_evidence_card(
        canonical_json_text(card),
        candidate=candidate,
        candidates_sha256="a" * 64,
    )


def test_evidence_card_replay_rejects_shortened_excerpt() -> None:
    candidate = _candidate(1, abstract="x" * 25)
    candidate.update(
        {"source_identity": "doi:10.1000/001", "cite_key": "smith2024alpha"}
    )
    proposal = parse_card_batch_response(
        _card_response([candidate]),
        expected_batch_id="card-batch-001",
        expected_source_ids=(candidate["source_identity"],),
    )[0]
    card = build_evidence_card(
        card_id="card-001",
        candidate=candidate,
        candidates_sha256="a" * 64,
        proposal=proposal,
    )
    card["evidence_excerpts"][0]["excerpt_text"] = "x" * 24
    card["evidence_excerpts"][0]["char_end"] = 24
    with pytest.raises(EvidenceCardContractError, match="shorter than 25"):
        parse_evidence_card(
            canonical_json_text(card),
            candidate=candidate,
            candidates_sha256="a" * 64,
        )


def test_fabricated_excerpt_produces_no_evidence() -> None:
    candidate = _candidate(1)
    candidate.update(
        {"source_identity": "doi:10.1000/001", "cite_key": "smith2024hardware"}
    )
    response = json.loads(_card_response([candidate]))
    response["cards"][0]["evidence_excerpt_texts"] = [
        "Fabricated evidence that is not retained."
    ]
    proposal = parse_card_batch_response(
        json.dumps(response),
        expected_batch_id="card-batch-001",
        expected_source_ids=(candidate["source_identity"],),
    )[0]
    card = build_evidence_card(
        card_id="card-001",
        candidate=candidate,
        candidates_sha256="a" * 64,
        proposal=proposal,
    )
    assert card["extraction_status"] == "fallback"
    assert card["evidence_excerpts"] == []


def test_stage6_writes_json_authority_and_deterministic_markdown(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    shortlist = _prepare_stage5(run_dir, [_candidate(1), _candidate(2)])
    stage6 = run_dir / "stage-06"
    stage6.mkdir()
    result = _execute_knowledge_extract(
        stage6,
        run_dir,
        _config(),  # type: ignore[arg-type]
        AdapterBundle(),
        llm=_SequenceLLM([_card_response(shortlist)]),  # type: ignore[arg-type]
    )
    assert result.status is StageStatus.DONE
    assert result.artifacts == ("cards/", "cards_manifest.json")
    manifest = json.loads((stage6 / "cards_manifest.json").read_text())
    assert [entry["source_identity"] for entry in manifest["cards"]] == [
        row["source_identity"] for row in shortlist
    ]
    first_json = (stage6 / "cards" / "card-001.json").read_text()
    first_card = parse_evidence_card(
        first_json,
        candidate=shortlist[0],
        candidates_sha256=sha256_text(
            (run_dir / "stage-04" / "candidates.jsonl").read_text()
        ),
    )
    assert (stage6 / "cards" / "card-001.md").read_text() == render_evidence_card_markdown(
        first_card
    )
    assert "Template" not in first_json


def test_stage6_zero_evidence_fails_without_canonical_cards(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    _prepare_stage5(run_dir, [_candidate(1)])
    stage6 = run_dir / "stage-06"
    stage6.mkdir()
    (stage6 / "cards").mkdir()
    (stage6 / "cards" / "stale.md").write_text("stale")
    (stage6 / "cards_manifest.json").write_text("stale")
    result = _execute_knowledge_extract(
        stage6,
        run_dir,
        _config(),  # type: ignore[arg-type]
        AdapterBundle(),
        llm=None,
    )
    assert result.status is StageStatus.FAILED
    assert not (stage6 / "cards").exists()
    assert not (stage6 / "cards_manifest.json").exists()
    diagnostic = json.loads((stage6 / "card_extraction_failures.json").read_text())
    assert diagnostic["reason"] == "zero_eligible_evidence_cards"
    assert diagnostic["cards"][0]["evidence_excerpts"] == []


def test_stage6_mixed_success_keeps_failed_card_non_evidentiary(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    shortlist = _prepare_stage5(run_dir, [_candidate(1), _candidate(2)])
    response = json.loads(_card_response(shortlist))
    response["cards"][1]["evidence_excerpt_texts"] = [
        "This substantive sentence is not in the abstract."
    ]
    stage6 = run_dir / "stage-06"
    stage6.mkdir()
    result = _execute_knowledge_extract(
        stage6,
        run_dir,
        _config(),  # type: ignore[arg-type]
        AdapterBundle(),
        llm=_SequenceLLM([json.dumps(response)]),  # type: ignore[arg-type]
    )
    assert result.status is StageStatus.DONE
    second = json.loads((stage6 / "cards" / "card-002.json").read_text())
    assert second["extraction_status"] == "fallback"
    assert second["evidence_excerpts"] == []


def test_stage6_rejects_tampered_screening_report_before_llm(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    shortlist = _prepare_stage5(run_dir, [_candidate(1)])
    report_path = run_dir / "stage-05" / "screening_report.json"
    report = json.loads(report_path.read_text())
    report["selected_candidate_ids"] = []
    report_path.write_text(canonical_json_text(report), encoding="utf-8")
    stage6 = run_dir / "stage-06"
    stage6.mkdir()
    llm = _SequenceLLM([_card_response(shortlist)])
    result = _execute_knowledge_extract(
        stage6,
        run_dir,
        _config(),  # type: ignore[arg-type]
        AdapterBundle(),
        llm=llm,  # type: ignore[arg-type]
    )
    assert result.status is StageStatus.FAILED
    assert "screening report" in (result.error or "")
    assert llm.calls == []
    assert not (stage6 / "cards").exists()


def test_cards_manifest_rejects_reordered_shortlist_identity(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    shortlist = _prepare_stage5(run_dir, [_candidate(1), _candidate(2)])
    stage6 = run_dir / "stage-06"
    stage6.mkdir()
    result = _execute_knowledge_extract(
        stage6,
        run_dir,
        _config(),  # type: ignore[arg-type]
        AdapterBundle(),
        llm=_SequenceLLM([_card_response(shortlist)]),  # type: ignore[arg-type]
    )
    assert result.status is StageStatus.DONE
    manifest = json.loads((stage6 / "cards_manifest.json").read_text())
    manifest["cards"].reverse()
    with pytest.raises(EvidenceCardContractError, match="shortlist-order"):
        validate_cards_artifacts(
            stage_dir=stage6,
            manifest=manifest,
            shortlist_text=(run_dir / "stage-05" / "shortlist.jsonl").read_text(),
            screening_report_text=(
                run_dir / "stage-05" / "screening_report.json"
            ).read_text(),
            candidates_sha256=sha256_text(
                (run_dir / "stage-04" / "candidates.jsonl").read_text()
            ),
            shortlist=shortlist,
        )


def test_cards_artifacts_default_deny_extra_file(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    shortlist = _prepare_stage5(run_dir, [_candidate(1)])
    stage6 = run_dir / "stage-06"
    stage6.mkdir()
    result = _execute_knowledge_extract(
        stage6,
        run_dir,
        _config(),  # type: ignore[arg-type]
        AdapterBundle(),
        llm=_SequenceLLM([_card_response(shortlist)]),  # type: ignore[arg-type]
    )
    assert result.status is StageStatus.DONE
    (stage6 / "cards" / "unmanifested.json").write_text("{}")
    manifest = json.loads((stage6 / "cards_manifest.json").read_text())
    with pytest.raises(EvidenceCardContractError, match="manifest closure"):
        validate_cards_artifacts(
            stage_dir=stage6,
            manifest=manifest,
            shortlist_text=(run_dir / "stage-05" / "shortlist.jsonl").read_text(),
            screening_report_text=(
                run_dir / "stage-05" / "screening_report.json"
            ).read_text(),
            candidates_sha256=sha256_text(
                (run_dir / "stage-04" / "candidates.jsonl").read_text()
            ),
            shortlist=shortlist,
        )


def test_stage7_replays_manifest_before_consuming_markdown(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    shortlist = _prepare_stage5(run_dir, [_candidate(1)])
    stage6 = run_dir / "stage-06"
    stage6.mkdir()
    result = _execute_knowledge_extract(
        stage6,
        run_dir,
        _config(),  # type: ignore[arg-type]
        AdapterBundle(),
        llm=_SequenceLLM([_card_response(shortlist)]),  # type: ignore[arg-type]
    )
    assert result.status is StageStatus.DONE
    (stage6 / "cards" / "card-001.md").write_text("tampered\n")
    stage7 = run_dir / "stage-07"
    stage7.mkdir()
    (stage7 / "synthesis.md").write_text("stale\n")

    synthesis = _execute_synthesis(
        stage7,
        run_dir,
        _config(),  # type: ignore[arg-type]
        AdapterBundle(),
        llm=None,
    )

    assert synthesis.status is StageStatus.FAILED
    assert "Markdown hash mismatch" in (synthesis.error or "")
    assert not (stage7 / "synthesis.md").exists()
