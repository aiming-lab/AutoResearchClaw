"""Stage 6 strict evidence-card and deterministic-renderer tests."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import yaml

from researchclaw.adapters import AdapterBundle
from researchclaw.config import RCConfig
from researchclaw.experiment_runtime.contract import derive_contract, dump_contract
from researchclaw.experiment_runtime.contract import find_stage09_contract, load_contract
from researchclaw.literature.citation_policy import (
    CitationPolicyContractError,
    load_effective_citation_policy,
    parse_citation_allowlist,
    parse_effective_citation_policy,
    resolve_active_config_snapshot,
    validate_citation_allowlist,
    write_active_config_binding,
)
from researchclaw.literature.citation_identity import seal_citation_collection
from researchclaw.literature.citation_plan import (
    CitationPlanContractError,
    build_citation_closure_report,
    build_citation_writer_instruction,
    load_final_citation_plan,
    parse_citation_plan,
    validate_paper_citation_minimum,
)
from researchclaw.literature.citation_support import (
    CitationSupportContractError,
    parse_citation_support_closure,
)
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
from researchclaw.literature.experiment_fact_closure import (
    build_experiment_fact_closure_report,
    parse_experiment_fact_closure_report,
)
from researchclaw.literature.screening import sha256_text
from researchclaw.literature.verify import (
    CitationResult,
    VerificationReport,
    VerifyStatus,
    parse_bibtex_entries,
)
from researchclaw.pipeline.stage_impls._literature import (
    _execute_knowledge_extract,
    _execute_literature_screen,
)
from researchclaw.pipeline.stage_impls._paper_writing import _execute_paper_outline
from researchclaw.pipeline.stage_impls._paper_writing import _execute_paper_draft
from researchclaw.pipeline.stage_impls._review_publish import (
    _execute_citation_verify,
    _execute_export_publish,
    _execute_peer_review,
    _execute_quality_gate,
)
from researchclaw.pipeline.stage_impls._release_audit import _execute_truth_audit
from researchclaw.pipeline.citation_release_audit import (
    CitationAuditError,
    audit_citation_evidence,
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


def _prepare_stage5(
    run_dir: Path,
    candidates: list[dict[str, Any]],
    config: Any | None = None,
) -> list[dict[str, Any]]:
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
        config or _config(),  # type: ignore[arg-type]
        AdapterBundle(),
        llm=_SequenceLLM([_screen_response(source_ids)]),  # type: ignore[arg-type]
    )
    assert result.status is StageStatus.DONE
    return [
        json.loads(line)
        for line in (stage5 / "shortlist.jsonl").read_text(encoding="utf-8").splitlines()
    ]


def _real_config_snapshot(
    run_dir: Path, *, claim_scope: str = "pipeline_validation"
) -> RCConfig:
    run_dir.mkdir(parents=True, exist_ok=True)
    source = Path("config.deepseek.sectional-dry-run.yaml")
    raw = yaml.safe_load(source.read_text(encoding="utf-8"))
    raw["experiment"]["claim_scope"] = claim_scope
    snapshot = run_dir / "config.yaml"
    snapshot.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    config = RCConfig.from_dict(raw, project_root=run_dir, check_paths=False)
    write_active_config_binding(run_dir, snapshot)
    return config


def _prepare_stage23_fixture(
    run_dir: Path,
    *,
    claim_scope: str = "pipeline_validation",
    fail_last_card: bool = False,
) -> tuple[RCConfig, str, tuple[str, ...]]:
    config = _real_config_snapshot(run_dir, claim_scope=claim_scope)
    stage9 = run_dir / "stage-09"
    stage9.mkdir()
    dump_contract(
        derive_contract(config, {"datasets": [config.experiment.dataset_origin]}),
        stage9 / "experiment_contract.yaml",
    )
    candidate_count = 15 if claim_scope != "pipeline_validation" else 5
    shortlist = _prepare_stage5(
        run_dir,
        [_candidate(i) for i in range(1, candidate_count + 1)],
        config,
    )
    stage6 = run_dir / "stage-06"
    stage6.mkdir()
    card_responses = [
        _card_response(shortlist[start:start + 4])
        for start in range(0, len(shortlist), 4)
    ]
    if fail_last_card:
        card_responses[-1:] = ["{}", "{}"]
    extracted = _execute_knowledge_extract(
        stage6,
        run_dir,
        config,
        AdapterBundle(),
        llm=_SequenceLLM(card_responses),  # type: ignore[arg-type]
    )
    assert extracted.status is StageStatus.DONE, extracted.error
    stage16 = run_dir / "stage-16"
    stage16.mkdir()
    outlined = _execute_paper_outline(
        stage16, run_dir, config, AdapterBundle(), llm=None
    )
    assert outlined.status is StageStatus.DONE, outlined.error
    plan = load_final_citation_plan(run_dir, config)
    planned_keys = tuple(
        citation["cite_key"]
        for claim in plan["claims"]
        for citation in claim["planned_citations"]
    )
    paper_text = "## Introduction\n\n" + " ".join(
        f"Evidence [{key}]." for key in planned_keys
    )
    stage22 = run_dir / "stage-22"
    stage22.mkdir()
    (stage22 / "paper_final.md").write_text(paper_text, encoding="utf-8")
    return config, paper_text, planned_keys


def _verification_report(
    keys: tuple[str, ...], *, status: VerifyStatus = VerifyStatus.VERIFIED
) -> VerificationReport:
    results = [
        CitationResult(
            cite_key=key,
            title=f"Verified title for {key}",
            status=status,
            confidence=0.95 if status is VerifyStatus.VERIFIED else 0.0,
            method="title_search" if status is not VerifyStatus.SKIPPED else "skipped",
        )
        for key in keys
    ]
    return VerificationReport(
        total=len(results),
        verified=sum(result.status is VerifyStatus.VERIFIED for result in results),
        suspicious=sum(result.status is VerifyStatus.SUSPICIOUS for result in results),
        hallucinated=sum(result.status is VerifyStatus.HALLUCINATED for result in results),
        skipped=sum(result.status is VerifyStatus.SKIPPED for result in results),
        results=results,
    )


def _run_stage23_verified(
    run_dir: Path,
    config: RCConfig,
    planned_keys: tuple[str, ...],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "researchclaw.literature.verify.verify_citations",
        lambda *_args, **_kwargs: _verification_report(planned_keys),
    )
    stage23 = run_dir / "stage-23"
    stage23.mkdir()
    result = _execute_citation_verify(
        stage23,
        run_dir,
        config,
        AdapterBundle(),
        llm=_SequenceLLM([json.dumps({key: 0.9 for key in planned_keys})]),  # type: ignore[arg-type]
    )
    assert result.status is StageStatus.DONE, result.error


def _prepare_e9_run(
    run_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[RCConfig, tuple[str, ...]]:
    config, paper_text, planned_keys = _prepare_stage23_fixture(run_dir)
    metric_path = run_dir / "stage-12" / "runs" / "results.json"
    metric_path.parent.mkdir(parents=True)
    metric_path.write_text(
        json.dumps({"metrics": {"detection_f1": 0.95}}), encoding="utf-8"
    )
    stage17 = run_dir / "stage-17"
    stage17.mkdir()
    (stage17 / "paper_draft.md").write_text(paper_text, encoding="utf-8")
    structure_text = canonical_json_text(
        {
            "schema_version": 1,
            "valid": True,
            "source_sha256": sha256_text(paper_text),
            "section_count": 1,
            "issues": [],
        }
    )
    (stage17 / "paper_structure_report.json").write_text(structure_text)
    experiment_text = canonical_json_text(
        build_experiment_fact_closure_report(run_dir, paper_text=paper_text)
    )
    (stage17 / "experiment_fact_closure_report.json").write_text(experiment_text)
    closure = build_citation_closure_report(
        run_dir,
        config,
        paper_text=paper_text,
        structure_report_text=structure_text,
        experiment_fact_report_text=experiment_text,
    )
    assert closure["valid"] is True
    (stage17 / "citation_closure_report.json").write_text(
        canonical_json_text(closure), encoding="utf-8"
    )
    _run_stage23_verified(run_dir, config, planned_keys, monkeypatch)
    support_responses = [
        json.dumps({"verdict": "supported", "reason": "The excerpt supports the claim."})
        for _key in planned_keys
    ]
    support_responses.append(
        json.dumps(
            {
                "claims": [
                    {
                        "text": "The experiment reports a bounded result.",
                        "type": "result",
                        "values": [],
                        "cited_keys": [],
                    }
                ]
            }
        )
    )
    stage24 = run_dir / "stage-24"
    stage24.mkdir()
    result = _execute_truth_audit(
        stage24,
        run_dir,
        config,
        AdapterBundle(),
        llm=_SequenceLLM(support_responses),  # type: ignore[arg-type]
    )
    assert result.status is StageStatus.DONE, result.error
    return config, planned_keys


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
    assert result.artifacts == (
        "cards/",
        "cards_manifest.json",
        "citation_allowlist.json",
    )
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


def test_stage6_allowlist_is_recomputed_from_success_cards(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    shortlist = _prepare_stage5(run_dir, [_candidate(1), _candidate(2)])
    response = json.loads(_card_response(shortlist))
    response["cards"][1]["evidence_excerpt_texts"] = [
        "This substantive sentence is absent from the retained abstract."
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
    allowlist_text = (stage6 / "citation_allowlist.json").read_text()
    allowlist = json.loads(allowlist_text)
    assert allowlist["eligible_keys"] == [shortlist[0]["cite_key"]]
    assert allowlist["ineligible"] == [
        {"cite_key": shortlist[1]["cite_key"], "reason_code": "card_fallback"}
    ]

    allowlist["eligible_keys"].append(shortlist[1]["cite_key"])
    allowlist["ineligible"] = []
    with pytest.raises(CitationPolicyContractError, match="replay mismatch"):
        validate_citation_allowlist(
            run_dir,
            _config(),  # type: ignore[arg-type]
            canonical_json_text(allowlist),
        )


def test_stage16_effective_policy_binds_run_local_config(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    config = _real_config_snapshot(run_dir)
    shortlist = _prepare_stage5(run_dir, [_candidate(1)], config)
    stage6 = run_dir / "stage-06"
    stage6.mkdir()
    result = _execute_knowledge_extract(
        stage6,
        run_dir,
        config,
        AdapterBundle(),
        llm=_SequenceLLM([_card_response(shortlist)]),  # type: ignore[arg-type]
    )
    assert result.status is StageStatus.DONE
    stage16 = run_dir / "stage-16"
    stage16.mkdir()
    outline = _execute_paper_outline(
        stage16, run_dir, config, AdapterBundle(), llm=None
    )
    assert outline.status is StageStatus.DONE
    policy = load_effective_citation_policy(run_dir, config)
    assert policy["eligible_count"] == 1
    assert policy["effective_min_unique_sources"] == 1
    assert policy["effective_target_unique_sources"] == 1
    assert policy["config_source_path"] == "config.yaml"
    final_plan = load_final_citation_plan(run_dir, config)
    assert final_plan["plan_status"] == "final"
    assert [
        claim["planned_citations"][0]["cite_key"]
        for claim in final_plan["claims"]
    ] == [shortlist[0]["cite_key"]]
    instruction = build_citation_writer_instruction(run_dir, config)
    assert shortlist[0]["cite_key"] in instruction
    assert shortlist[0]["abstract"] in instruction
    assert "AVAILABLE REFERENCES" not in instruction

    history_path = run_dir / "config_snapshot_history.jsonl"
    history_text = history_path.read_text(encoding="utf-8")
    history_path.write_text(history_text + "{}\n", encoding="utf-8")
    with pytest.raises(CitationPolicyContractError, match="history hash mismatch"):
        load_effective_citation_policy(run_dir, config)
    history_path.write_text(history_text, encoding="utf-8")

    (run_dir / "config.yaml").write_text("tampered: true\n", encoding="utf-8")
    with pytest.raises(CitationPolicyContractError, match="hash mismatch"):
        load_effective_citation_policy(run_dir, config)


def test_research_release_fails_below_citation_minimum(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    config = _real_config_snapshot(run_dir, claim_scope="research_release")
    shortlist = _prepare_stage5(run_dir, [_candidate(1)], config)
    stage6 = run_dir / "stage-06"
    stage6.mkdir()
    result = _execute_knowledge_extract(
        stage6,
        run_dir,
        config,
        AdapterBundle(),
        llm=_SequenceLLM([_card_response(shortlist)]),  # type: ignore[arg-type]
    )
    assert result.status is StageStatus.DONE
    stage16 = run_dir / "stage-16"
    stage16.mkdir()
    outline = _execute_paper_outline(
        stage16, run_dir, config, AdapterBundle(), llm=None
    )
    assert outline.status is StageStatus.FAILED
    assert "below required minimum 15" in (outline.error or "")
    assert not (stage16 / "citation_policy_effective.json").exists()


def test_citation_policy_loaders_reject_duplicate_keys_and_boolean_counts() -> None:
    with pytest.raises(CitationPolicyContractError, match="duplicate JSON key"):
        parse_citation_allowlist(
            '{"schema_version":1,"schema_version":1}'
        )
    payload = {
        "schema_version": 1,
        "policy_version": 1,
        "claim_scope": "pipeline_validation",
        "eligible_count": True,
        "effective_min_unique_sources": 1,
        "effective_target_unique_sources": 1,
        "citation_allowlist_path": "stage-06/citation_allowlist.json",
        "citation_allowlist_sha256": "a" * 64,
        "config_source_path": "config.yaml",
        "config_source_sha256": "b" * 64,
    }
    with pytest.raises(CitationPolicyContractError, match="nonnegative integer"):
        parse_effective_citation_policy(canonical_json_text(payload))


def test_citation_plan_loader_rejects_unknown_fields_and_tampering(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    config = _real_config_snapshot(run_dir)
    shortlist = _prepare_stage5(run_dir, [_candidate(1)], config)
    stage6 = run_dir / "stage-06"
    stage6.mkdir()
    assert _execute_knowledge_extract(
        stage6,
        run_dir,
        config,
        AdapterBundle(),
        llm=_SequenceLLM([_card_response(shortlist)]),  # type: ignore[arg-type]
    ).status is StageStatus.DONE
    stage16 = run_dir / "stage-16"
    stage16.mkdir()
    assert _execute_paper_outline(
        stage16, run_dir, config, AdapterBundle(), llm=None
    ).status is StageStatus.DONE
    plan_path = stage16 / "citation_plan.json"
    payload = json.loads(plan_path.read_text(encoding="utf-8"))
    payload["claims"][0]["planned_citations"][0]["cite_key"] = "fake2024key"
    plan_path.write_text(canonical_json_text(payload), encoding="utf-8")
    with pytest.raises(CitationPlanContractError, match="replay mismatch"):
        load_final_citation_plan(run_dir, config)
    payload["unexpected"] = True
    with pytest.raises(CitationPlanContractError, match="fields mismatch"):
        parse_citation_plan(canonical_json_text(payload))


def test_citation_closure_rejects_key_outside_assigned_section(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    config = _real_config_snapshot(run_dir)
    shortlist = _prepare_stage5(run_dir, [_candidate(1)], config)
    stage6 = run_dir / "stage-06"
    stage6.mkdir()
    assert _execute_knowledge_extract(
        stage6,
        run_dir,
        config,
        AdapterBundle(),
        llm=_SequenceLLM([_card_response(shortlist)]),  # type: ignore[arg-type]
    ).status is StageStatus.DONE
    stage16 = run_dir / "stage-16"
    stage16.mkdir()
    assert _execute_paper_outline(
        stage16, run_dir, config, AdapterBundle(), llm=None
    ).status is StageStatus.DONE
    key = shortlist[0]["cite_key"]
    paper = f"## Introduction\n\nBackground.\n\n## Results\n\nResult [{key}].\n"
    stage9 = run_dir / "stage-09"
    stage9.mkdir()
    dump_contract(derive_contract(config, None), stage9 / "experiment_contract.yaml")
    metric_path = run_dir / "stage-12" / "runs" / "results.json"
    metric_path.parent.mkdir(parents=True)
    metric_path.write_text(
        json.dumps({"metrics": {"detection_f1": 0.95}}), encoding="utf-8"
    )
    structure = canonical_json_text(
        {
            "schema_version": 1,
            "valid": True,
            "source_sha256": sha256_text(paper),
            "section_count": 2,
            "issues": [],
        }
    )
    experiment = canonical_json_text(
        build_experiment_fact_closure_report(run_dir, paper_text=paper)
    )
    report = build_citation_closure_report(
        run_dir,
        config,
        paper_text=paper,
        structure_report_text=structure,
        experiment_fact_report_text=experiment,
    )
    assert report["misplaced_planned_keys"] == [key]
    assert report["valid"] is False


def test_experiment_fact_closure_binds_metrics_and_synthetic_origin(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    config = _real_config_snapshot(run_dir)
    stage9 = run_dir / "stage-09"
    stage9.mkdir()
    dump_contract(derive_contract(config, None), stage9 / "experiment_contract.yaml")
    run_path = run_dir / "stage-12" / "runs" / "run-001.json"
    run_path.parent.mkdir(parents=True)
    run_path.write_text(
        json.dumps({"metrics": {"detection_rate": 0.95}}), encoding="utf-8"
    )
    paper = (
        "## Introduction\n\nPrior work reported 0.42 in another setting.\n\n"
        "## Results\n\nThe measured detection rate was 95%.\n"
    )
    report = build_experiment_fact_closure_report(run_dir, paper_text=paper)
    assert report["valid"] is True
    assert report["manuscript_numeric_values"] == [0.95]
    assert report["unknown_numeric_values"] == []

    contradicted = paper.replace(
        "The measured", "Using real-hardware measurements, the measured"
    )
    report = build_experiment_fact_closure_report(run_dir, paper_text=contradicted)
    assert report["valid"] is False
    assert report["dataset_claim_violations"]


def test_experiment_fact_closure_rejects_unknown_metric_and_duplicate_json() -> None:
    payload = {
        "schema_version": 1,
        "paper_path": "stage-17/paper_draft.md",
        "paper_sha256": "a" * 64,
        "experiment_contract_path": "stage-09/experiment_contract.yaml",
        "experiment_contract_sha256": "b" * 64,
        "dataset_origin": "synthetic",
        "metric_sources": [{"path": "stage-12/runs/run.json", "sha256": "c" * 64}],
        "grounded_numeric_values": [0.95],
        "manuscript_numeric_values": [0.97],
        "unknown_numeric_values": [0.97],
        "dataset_claim_violations": [],
        "valid": False,
    }
    assert parse_experiment_fact_closure_report(canonical_json_text(payload))["valid"] is False
    with pytest.raises(ValueError, match="duplicate JSON key"):
        parse_experiment_fact_closure_report('{"schema_version":1,"schema_version":1}')


def test_experiment_fact_closure_detects_integer_and_non_results_metrics(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    config = _real_config_snapshot(run_dir)
    stage9 = run_dir / "stage-09"
    stage9.mkdir()
    dump_contract(derive_contract(config, None), stage9 / "experiment_contract.yaml")
    metric_path = run_dir / "stage-12" / "runs" / "results.json"
    metric_path.parent.mkdir(parents=True)
    metric_path.write_text(
        json.dumps({"metrics": {"fps": 90, "latency_cycles": 64, "f1": 0.5}}),
        encoding="utf-8",
    )
    paper = (
        "## Abstract\n\nThe system achieves 92 FPS and 0.7 F1.\n\n"
        "## Results\n\nLatency was 128 cycles.\n\n"
        "## Discussion\n\nThe gain remained 3x.\n"
    )
    report = build_experiment_fact_closure_report(run_dir, paper_text=paper)
    assert report["valid"] is False
    assert report["unknown_numeric_values"] == [92.0, 0.7, 128.0, 3.0]


def test_experiment_fact_closure_percent_normalization_is_token_explicit(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    config = _real_config_snapshot(run_dir)
    stage9 = run_dir / "stage-09"
    stage9.mkdir()
    dump_contract(derive_contract(config, None), stage9 / "experiment_contract.yaml")
    metric_path = run_dir / "stage-12" / "runs" / "results.json"
    metric_path.parent.mkdir(parents=True)
    metric_path.write_text(json.dumps({"metrics": {"rate": 0.5}}), encoding="utf-8")
    percent = build_experiment_fact_closure_report(
        run_dir, paper_text="## Results\n\nThe rate was 50%.\n"
    )
    assert percent["valid"] is True
    unmarked = build_experiment_fact_closure_report(
        run_dir, paper_text="## Results\n\nThe rate was 50.0.\n"
    )
    assert unmarked["unknown_numeric_values"] == [50.0]


def test_experiment_fact_closure_ignores_shadow_metric_stage_and_flags_hardware_claim(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    config = _real_config_snapshot(run_dir)
    stage9 = run_dir / "stage-09"
    stage9.mkdir()
    dump_contract(derive_contract(config, None), stage9 / "experiment_contract.yaml")
    direct = run_dir / "stage-12" / "runs" / "results.json"
    direct.parent.mkdir(parents=True)
    direct.write_text(json.dumps({"metrics": {"rate": 0.5}}), encoding="utf-8")
    shadow = run_dir / "stage-12b" / "runs" / "fake.json"
    shadow.parent.mkdir(parents=True)
    shadow.write_text(json.dumps({"metrics": {"rate": 0.97}}), encoding="utf-8")
    paper = (
        "## Abstract\n\nWe captured on our FPGA prototype board.\n\n"
        "## Results\n\nThe rate was 0.97.\n"
    )
    report = build_experiment_fact_closure_report(run_dir, paper_text=paper)
    assert report["unknown_numeric_values"] == [0.97]
    assert report["dataset_claim_violations"]


def test_stage17_uses_final_plan_only_and_writes_replayable_closure(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    config = _real_config_snapshot(run_dir)
    shortlist = _prepare_stage5(run_dir, [_candidate(1)], config)
    stage6 = run_dir / "stage-06"
    stage6.mkdir()
    assert _execute_knowledge_extract(
        stage6,
        run_dir,
        config,
        AdapterBundle(),
        llm=_SequenceLLM([_card_response(shortlist)]),  # type: ignore[arg-type]
    ).status is StageStatus.DONE
    stage9 = run_dir / "stage-09"
    stage9.mkdir()
    dump_contract(derive_contract(config, None), stage9 / "experiment_contract.yaml")
    run_path = run_dir / "stage-12" / "runs" / "results.json"
    run_path.parent.mkdir(parents=True)
    run_path.write_text(
        json.dumps(
            {
                "claim_scope": "pipeline_validation",
                "dataset_origin": "synthetic",
                "evaluator_owner": "scaffold",
                "metrics": {"detection_f1": 0.95},
            }
        ),
        encoding="utf-8",
    )
    stage16 = run_dir / "stage-16"
    stage16.mkdir()
    assert _execute_paper_outline(
        stage16, run_dir, config, AdapterBundle(), llm=None
    ).status is StageStatus.DONE
    key = shortlist[0]["cite_key"]
    llm = _SequenceLLM(
        [
            (
                "## Title\n\nBounded Study\n\n## Abstract\n\nAbstract.\n\n"
                f"## Introduction\n\nEvidence-backed context [{key}].\n\n"
                "## Related Work\n\nRelated work."
            ),
            "## Method\n\nMethod.\n\n## Experiments\n\nExperiment setup.",
            "## Results\n\nDetection F1 was 95%.\n\n## Conclusion\n\nConclusion.",
        ]
    )
    stage17 = run_dir / "stage-17"
    stage17.mkdir()
    result = _execute_paper_draft(
        stage17,
        run_dir,
        config,
        AdapterBundle(),
        llm=llm,  # type: ignore[arg-type]
    )
    assert result.status is StageStatus.DONE, result.error
    prompts = "\n".join(llm.calls)
    assert "FINAL CITATION PLAN" in prompts
    assert "AVAILABLE REFERENCES" not in prompts
    assert "Hardware Detection Study" not in prompts
    assert not (stage17 / "references_preverified.bib").exists()
    closure = json.loads((stage17 / "citation_closure_report.json").read_text())
    assert closure["valid"] is True
    assert closure["experiment_fact_closure_valid"] is True
    draft_text = (stage17 / "paper_draft.md").read_text(encoding="utf-8")
    assert validate_paper_citation_minimum(
        run_dir, config, draft_text, minimum=1
    ) == (key,)
    stage19 = run_dir / "stage-19"
    stage19.mkdir()
    stage20 = run_dir / "stage-20"
    stage20.mkdir()
    (stage19 / "paper_revised.md").write_text(
        draft_text.replace(f"[{key}]", ""), encoding="utf-8"
    )
    quality = _execute_quality_gate(
        stage20, run_dir, config, AdapterBundle(), llm=None
    )
    assert quality.status is StageStatus.FAILED
    assert "effective citation policy" in (quality.error or "").lower()
    assert "minimum=1" in (quality.error or "")
    closure["paper_sha256"] = "0" * 64
    (stage17 / "citation_closure_report.json").write_text(
        canonical_json_text(closure), encoding="utf-8"
    )
    stage18 = run_dir / "stage-18"
    stage18.mkdir()
    review = _execute_peer_review(
        stage18, run_dir, config, AdapterBundle(), llm=None
    )
    assert review.status is StageStatus.FAILED
    assert "closure" in (review.error or "").lower()


def test_stage20_22_and_23_reject_bibliography_key_outside_allowlist(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    config = _real_config_snapshot(run_dir)
    shortlist = _prepare_stage5(run_dir, [_candidate(i) for i in range(1, 6)], config)
    stage6 = run_dir / "stage-06"
    stage6.mkdir()
    llm = _SequenceLLM(
        [_card_response(shortlist[:4]), "{}", "{}"]
    )
    assert _execute_knowledge_extract(
        stage6, run_dir, config, AdapterBundle(), llm=llm  # type: ignore[arg-type]
    ).status is StageStatus.DONE
    stage16 = run_dir / "stage-16"
    stage16.mkdir()
    assert _execute_paper_outline(
        stage16, run_dir, config, AdapterBundle(), llm=None
    ).status is StageStatus.DONE
    planned = load_final_citation_plan(run_dir, config)
    planned_keys = [
        citation["cite_key"]
        for claim in planned["claims"]
        for citation in claim["planned_citations"]
    ]
    ineligible_key = shortlist[4]["cite_key"]
    final_text = "## Introduction\n\n" + " ".join(
        f"Evidence [{key}]." for key in planned_keys + [ineligible_key]
    )
    stage19 = run_dir / "stage-19"
    stage19.mkdir()
    (stage19 / "paper_revised.md").write_text(final_text, encoding="utf-8")
    stage20 = run_dir / "stage-20"
    stage20.mkdir()
    quality = _execute_quality_gate(
        stage20, run_dir, config, AdapterBundle(), llm=None
    )
    assert quality.status is StageStatus.FAILED
    assert "invalid=" in (quality.error or "")

    stage22 = run_dir / "stage-22"
    stage22.mkdir()
    exported = _execute_export_publish(
        stage22, run_dir, config, AdapterBundle(), llm=None
    )
    assert exported.status is StageStatus.FAILED
    assert "Evidence-bound" in (exported.error or "")

    (stage22 / "paper_final.md").write_text(final_text, encoding="utf-8")
    stage23 = run_dir / "stage-23"
    stage23.mkdir()
    verified = _execute_citation_verify(
        stage23, run_dir, config, AdapterBundle(), llm=None
    )
    assert verified.status is StageStatus.FAILED
    assert "Evidence-bound" in (verified.error or "")


def test_stage23_verifies_only_final_cited_keys(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_dir = tmp_path / "run"
    config, paper_text, planned_keys = _prepare_stage23_fixture(
        run_dir, fail_last_card=True
    )
    shadow = run_dir / "stage-99"
    shadow.mkdir()
    (shadow / "paper_final.md").write_text(
        "## Introduction\n\nShadow [fake2024].\n", encoding="utf-8"
    )
    captured_keys: set[str] = set()

    def _verify(bib_text: str, **_kwargs: object) -> VerificationReport:
        captured_keys.update(
            str(entry["key"]) for entry in parse_bibtex_entries(bib_text)
        )
        return _verification_report(planned_keys)

    monkeypatch.setattr("researchclaw.literature.verify.verify_citations", _verify)
    relevance = json.dumps({key: 0.9 for key in planned_keys})
    stage23 = run_dir / "stage-23"
    stage23.mkdir()
    result = _execute_citation_verify(
        stage23,
        run_dir,
        config,
        AdapterBundle(),
        llm=_SequenceLLM([relevance]),  # type: ignore[arg-type]
    )
    assert result.status is StageStatus.DONE, result.error
    assert result.decision is None
    assert captured_keys == set(planned_keys)
    assert len(captured_keys) < 5
    assert (stage23 / "paper_final_verified.md").read_text() == paper_text
    report = json.loads((stage23 / "verification_report.json").read_text())
    assert report["summary"]["verification_complete"] is True
    assert report["summary"]["relevance_complete"] is True


def test_stage23_pipeline_validation_degrades_without_relevance_scores(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_dir = tmp_path / "run"
    config, paper_text, planned_keys = _prepare_stage23_fixture(run_dir)
    monkeypatch.setattr(
        "researchclaw.literature.verify.verify_citations",
        lambda *_args, **_kwargs: _verification_report(planned_keys),
    )
    stage23 = run_dir / "stage-23"
    stage23.mkdir()
    result = _execute_citation_verify(
        stage23, run_dir, config, AdapterBundle(), llm=None
    )
    assert result.status is StageStatus.DONE, result.error
    assert result.decision == "degraded"
    assert (stage23 / "paper_final_verified.md").read_text() == paper_text
    report = json.loads((stage23 / "verification_report.json").read_text())
    assert report["summary"]["unscored_keys"] == sorted(planned_keys)
    assert report["summary"]["fatal"] is False


def test_stage23_pipeline_validation_still_fails_hallucinated_citation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_dir = tmp_path / "run"
    config, paper_text, planned_keys = _prepare_stage23_fixture(run_dir)
    monkeypatch.setattr(
        "researchclaw.literature.verify.verify_citations",
        lambda *_args, **_kwargs: _verification_report(
            planned_keys, status=VerifyStatus.HALLUCINATED
        ),
    )
    relevance = json.dumps({key: 0.9 for key in planned_keys})
    stage23 = run_dir / "stage-23"
    stage23.mkdir()
    result = _execute_citation_verify(
        stage23,
        run_dir,
        config,
        AdapterBundle(),
        llm=_SequenceLLM([relevance]),  # type: ignore[arg-type]
    )
    assert result.status is StageStatus.FAILED
    assert (stage23 / "paper_final_verified.md").read_text() == paper_text
    report = json.loads((stage23 / "verification_report.json").read_text())
    assert report["summary"]["fatal"] is True
    assert report["summary"]["degraded"] is False


@pytest.mark.parametrize(
    "status",
    [VerifyStatus.SKIPPED, VerifyStatus.SUSPICIOUS, VerifyStatus.HALLUCINATED],
)
def test_stage23_strict_scope_rejects_incomplete_verification_without_editing_paper(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    status: VerifyStatus,
) -> None:
    run_dir = tmp_path / "run"
    config, paper_text, planned_keys = _prepare_stage23_fixture(
        run_dir, claim_scope="exploratory"
    )
    monkeypatch.setattr(
        "researchclaw.literature.verify.verify_citations",
        lambda *_args, **_kwargs: _verification_report(
            planned_keys, status=status
        ),
    )
    relevance = json.dumps({key: 0.9 for key in planned_keys})
    stage23 = run_dir / "stage-23"
    stage23.mkdir()
    result = _execute_citation_verify(
        stage23,
        run_dir,
        config,
        AdapterBundle(),
        llm=_SequenceLLM([relevance]),  # type: ignore[arg-type]
    )
    assert result.status is StageStatus.FAILED
    assert result.decision == "retry"
    assert (stage23 / "paper_final_verified.md").read_text() == paper_text
    assert all(f"[{key}]" in paper_text for key in planned_keys)


def test_stage23_strict_scope_rejects_malformed_relevance_response(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_dir = tmp_path / "run"
    config, _paper_text, planned_keys = _prepare_stage23_fixture(
        run_dir, claim_scope="exploratory"
    )
    monkeypatch.setattr(
        "researchclaw.literature.verify.verify_citations",
        lambda *_args, **_kwargs: _verification_report(planned_keys),
    )
    stage23 = run_dir / "stage-23"
    stage23.mkdir()
    result = _execute_citation_verify(
        stage23,
        run_dir,
        config,
        AdapterBundle(),
        llm=_SequenceLLM([json.dumps({planned_keys[0]: True})]),  # type: ignore[arg-type]
    )
    assert result.status is StageStatus.FAILED
    report = json.loads((stage23 / "verification_report.json").read_text())
    assert report["summary"]["relevance_complete"] is False
    assert report["summary"]["relevance_error"]


def test_stage23_rejects_verification_result_closure_mismatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_dir = tmp_path / "run"
    config, _paper_text, planned_keys = _prepare_stage23_fixture(run_dir)
    monkeypatch.setattr(
        "researchclaw.literature.verify.verify_citations",
        lambda *_args, **_kwargs: _verification_report(planned_keys[:-1]),
    )
    stage23 = run_dir / "stage-23"
    stage23.mkdir()
    result = _execute_citation_verify(
        stage23, run_dir, config, AdapterBundle(), llm=None
    )
    assert result.status is StageStatus.FAILED
    assert "closure mismatch" in (result.error or "").lower()


def test_stage23_rejects_bounded_bibliography_closure_mismatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_dir = tmp_path / "run"
    config, _paper_text, planned_keys = _prepare_stage23_fixture(run_dir)
    canonical_bib = (run_dir / "stage-04" / "references.bib").read_text()
    truncated_bib = canonical_bib.replace(
        next(
            entry
            for entry in canonical_bib.split("\n\n")
            if planned_keys[-1] in entry
        ),
        "",
    )
    monkeypatch.setattr(
        "researchclaw.pipeline.stage_impls._review_publish.load_canonical_bibliography",
        lambda _run_dir: truncated_bib,
    )
    stage23 = run_dir / "stage-23"
    stage23.mkdir()
    result = _execute_citation_verify(
        stage23, run_dir, config, AdapterBundle(), llm=None
    )
    assert result.status is StageStatus.FAILED
    assert "bounded bibliography" in (result.error or "").lower()


def test_stage23_rejects_cited_paper_without_canonical_bibliography(
    tmp_path: Path
) -> None:
    run_dir = tmp_path / "run"
    config = _real_config_snapshot(run_dir)
    stage22 = run_dir / "stage-22"
    stage22.mkdir()
    (stage22 / "paper_final.md").write_text(
        "## Introduction\n\nUnsupported [fake2024].\n", encoding="utf-8"
    )
    stage23 = run_dir / "stage-23"
    stage23.mkdir()
    result = _execute_citation_verify(
        stage23, run_dir, config, AdapterBundle(), llm=None
    )
    assert result.status is StageStatus.FAILED
    assert "canonical bibliography is invalid" in (result.error or "").lower()
    assert not (stage23 / "verification_report.json").exists()


def test_stage23_rejects_symlinked_canonical_paper(
    tmp_path: Path
) -> None:
    run_dir = tmp_path / "run"
    config = _real_config_snapshot(run_dir)
    outside = tmp_path / "outside.md"
    outside.write_text("## Introduction\n", encoding="utf-8")
    stage22 = run_dir / "stage-22"
    stage22.mkdir()
    (stage22 / "paper_final.md").symlink_to(outside)
    stage23 = run_dir / "stage-23"
    stage23.mkdir()
    result = _execute_citation_verify(
        stage23, run_dir, config, AdapterBundle(), llm=None
    )
    assert result.status is StageStatus.FAILED
    assert "missing or unsafe" in (result.error or "").lower()


def test_stage23_cleans_stale_verified_outputs_before_early_failure(
    tmp_path: Path
) -> None:
    run_dir = tmp_path / "run"
    config = _real_config_snapshot(run_dir)
    stage22 = run_dir / "stage-22"
    stage22.mkdir()
    (stage22 / "paper_final.md").write_text(
        "## Introduction\n\nUnknown [fake2024].\n", encoding="utf-8"
    )
    stage23 = run_dir / "stage-23"
    stage23.mkdir()
    for name in (
        "verification_report.json",
        "references_verified.bib",
        "paper_final_verified.md",
    ):
        (stage23 / name).write_text("stale", encoding="utf-8")
    result = _execute_citation_verify(
        stage23, run_dir, config, AdapterBundle(), llm=None
    )
    assert result.status is StageStatus.FAILED
    assert not any(
        (stage23 / name).exists()
        for name in (
            "verification_report.json",
            "references_verified.bib",
            "paper_final_verified.md",
        )
    )


def test_stage24_builds_evidence_bound_support_closure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_dir = tmp_path / "run"
    config, paper_text, planned_keys = _prepare_stage23_fixture(run_dir)
    _run_stage23_verified(run_dir, config, planned_keys, monkeypatch)
    support_responses = [
        json.dumps({"verdict": "supported", "reason": "The excerpt supports the claim."})
        for _key in planned_keys
    ]
    support_responses.append(
        json.dumps(
            {
                "claims": [
                    {
                        "text": "The experiment reports a bounded result.",
                        "type": "result",
                        "values": [],
                        "cited_keys": [],
                    }
                ]
            }
        )
    )
    stage24 = run_dir / "stage-24"
    stage24.mkdir()
    result = _execute_truth_audit(
        stage24,
        run_dir,
        config,
        AdapterBundle(),
        llm=_SequenceLLM(support_responses),  # type: ignore[arg-type]
    )
    assert result.status is StageStatus.DONE, result.error
    support = parse_citation_support_closure(
        (stage24 / "citation_support.json").read_text(encoding="utf-8")
    )
    assert support["valid"] is True
    assert support["counts"]["supported"] == len(planned_keys)
    assert all(row["evidence_excerpts"] for row in support["instances"])
    assert all(
        row["assessment"]["critic_model"] == config.paper_revision.critic_model
        for row in support["instances"]
    )
    citations = json.loads((stage24 / "citations.json").read_text())
    assert citations["counts"]["claim_support"] == len(planned_keys)
    truth = json.loads((stage24 / "truth_audit.json").read_text())
    assert truth["citation_support_valid"] is True
    assert (stage24 / "citation_support.json").is_file()
    assert (stage23_text := run_dir / "stage-23" / "paper_final_verified.md").read_text() == paper_text


def test_stage24_fails_unsupported_citation_without_fabricating_mapping(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_dir = tmp_path / "run"
    config, _paper_text, planned_keys = _prepare_stage23_fixture(run_dir)
    _run_stage23_verified(run_dir, config, planned_keys, monkeypatch)
    responses = [
        json.dumps({"verdict": "unsupported", "reason": "The excerpt is insufficient."})
        for _key in planned_keys
    ]
    responses.append('{"claims": []}')
    stage24 = run_dir / "stage-24"
    stage24.mkdir()
    result = _execute_truth_audit(
        stage24,
        run_dir,
        config,
        AdapterBundle(),
        llm=_SequenceLLM(responses),  # type: ignore[arg-type]
    )
    assert result.status is StageStatus.FAILED
    assert set(result.artifacts) == {
        "claims.json",
        "citations.json",
        "citation_support.json",
        "critique_resolution.json",
        "truth_audit.json",
    }
    support = json.loads((stage24 / "citation_support.json").read_text())
    assert support["valid"] is False
    assert support["counts"]["unsupported"] == len(planned_keys)
    citations = json.loads((stage24 / "citations.json").read_text())
    assert citations["counts"]["claim_support"] == 0
    assert citations["counts"]["unmapped"] == len(planned_keys)


def test_stage24_fails_synthetic_claim_of_real_hardware_measurement(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_dir = tmp_path / "run"
    config, _paper_text, planned_keys = _prepare_stage23_fixture(run_dir)
    _run_stage23_verified(run_dir, config, planned_keys, monkeypatch)
    violating_text = "## Introduction\n\n" + " ".join(
        f"We measured on real hardware [{key}]." for key in planned_keys
    )
    (run_dir / "stage-22" / "paper_final.md").write_text(violating_text)
    (run_dir / "stage-23" / "paper_final_verified.md").write_text(violating_text)
    responses = [
        json.dumps({"verdict": "supported", "reason": "The excerpt supports the claim."})
        for _key in planned_keys
    ]
    responses.append('{"claims": []}')
    stage24 = run_dir / "stage-24"
    stage24.mkdir()
    result = _execute_truth_audit(
        stage24,
        run_dir,
        config,
        AdapterBundle(),
        llm=_SequenceLLM(responses),  # type: ignore[arg-type]
    )
    assert result.status is StageStatus.FAILED
    support = json.loads((stage24 / "citation_support.json").read_text())
    assert support["dataset_origin"] == "synthetic"
    assert support["dataset_claim_violations"]
    assert support["valid"] is False


def test_stage24_rejects_malformed_support_critic_response(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_dir = tmp_path / "run"
    config, _paper_text, planned_keys = _prepare_stage23_fixture(run_dir)
    _run_stage23_verified(run_dir, config, planned_keys, monkeypatch)
    stage24 = run_dir / "stage-24"
    stage24.mkdir()
    result = _execute_truth_audit(
        stage24,
        run_dir,
        config,
        AdapterBundle(),
        llm=_SequenceLLM(["{}"]),  # type: ignore[arg-type]
    )
    assert result.status is StageStatus.FAILED
    assert "support critic response" in (result.error or "")
    assert not (stage24 / "citation_support.json").exists()


def test_stage24_without_support_critic_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_dir = tmp_path / "run"
    config, _paper_text, planned_keys = _prepare_stage23_fixture(run_dir)
    _run_stage23_verified(run_dir, config, planned_keys, monkeypatch)
    stage24 = run_dir / "stage-24"
    stage24.mkdir()
    result = _execute_truth_audit(
        stage24, run_dir, config, AdapterBundle(), llm=None
    )
    assert result.status is StageStatus.FAILED
    support = json.loads((stage24 / "citation_support.json").read_text())
    assert support["counts"]["unsupported"] == len(planned_keys)
    assert support["valid"] is False


def test_stage24_support_replay_rejects_excerpt_tampering(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_dir = tmp_path / "run"
    config, _paper_text, planned_keys = _prepare_stage23_fixture(run_dir)
    _run_stage23_verified(run_dir, config, planned_keys, monkeypatch)
    responses = [
        json.dumps({"verdict": "supported", "reason": "The excerpt supports the claim."})
        for _key in planned_keys
    ]
    responses.append('{"claims": []}')
    stage24 = run_dir / "stage-24"
    stage24.mkdir()
    result = _execute_truth_audit(
        stage24,
        run_dir,
        config,
        AdapterBundle(),
        llm=_SequenceLLM(responses),  # type: ignore[arg-type]
    )
    assert result.status is StageStatus.DONE
    support = json.loads((stage24 / "citation_support.json").read_text())
    support["instances"][0]["evidence_excerpts"][0]["excerpt_text"] += " tampered"
    with pytest.raises(CitationSupportContractError, match="hash mismatch"):
        parse_citation_support_closure(json.dumps(support))


def test_stage24_rejects_support_critic_equal_to_writer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_dir = tmp_path / "run"
    config, _paper_text, planned_keys = _prepare_stage23_fixture(run_dir)
    _run_stage23_verified(run_dir, config, planned_keys, monkeypatch)
    unsafe_config = replace(
        config,
        paper_revision=replace(
            config.paper_revision, critic_model=config.llm.primary_model
        ),
    )
    stage24 = run_dir / "stage-24"
    stage24.mkdir()
    result = _execute_truth_audit(
        stage24, run_dir, unsafe_config, AdapterBundle(), llm=_SequenceLLM([])  # type: ignore[arg-type]
    )
    assert result.status is StageStatus.FAILED
    assert "must differ" in (result.error or "")
    assert not (stage24 / "citation_support.json").exists()


@pytest.mark.parametrize(
    "status",
    ["suspicious", "hallucinated", "skipped"],
)
def test_stage24_existence_status_overrides_supported_critic(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    status: str,
) -> None:
    run_dir = tmp_path / "run"
    config, _paper_text, planned_keys = _prepare_stage23_fixture(run_dir)
    _run_stage23_verified(run_dir, config, planned_keys, monkeypatch)
    report_path = run_dir / "stage-23" / "verification_report.json"
    report = json.loads(report_path.read_text())
    report["results"][0]["status"] = status
    report_path.write_text(json.dumps(report), encoding="utf-8")
    responses = [
        json.dumps({"verdict": "supported", "reason": "The excerpt supports the claim."})
        for _key in planned_keys
    ]
    responses.append('{"claims": []}')
    stage24 = run_dir / "stage-24"
    stage24.mkdir()
    result = _execute_truth_audit(
        stage24,
        run_dir,
        config,
        AdapterBundle(),
        llm=_SequenceLLM(responses),  # type: ignore[arg-type]
    )
    assert result.status is StageStatus.FAILED
    support = json.loads((stage24 / "citation_support.json").read_text())
    assert support["instances"][0]["existence_status"] == status
    assert support["instances"][0]["assessment"]["verdict"] == "unsupported"


def test_stage24_rejects_stage22_stage23_paper_divergence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_dir = tmp_path / "run"
    config, _paper_text, planned_keys = _prepare_stage23_fixture(run_dir)
    _run_stage23_verified(run_dir, config, planned_keys, monkeypatch)
    (run_dir / "stage-23" / "paper_final_verified.md").write_text(
        "## Introduction\n\nTampered paper.\n", encoding="utf-8"
    )
    stage24 = run_dir / "stage-24"
    stage24.mkdir()
    result = _execute_truth_audit(
        stage24, run_dir, config, AdapterBundle(), llm=_SequenceLLM([])  # type: ignore[arg-type]
    )
    assert result.status is StageStatus.FAILED
    assert "byte-identical" in (result.error or "")
    assert not (stage24 / "citation_support.json").exists()


@pytest.mark.parametrize("mutation", ["missing", "extra", "duplicate"])
def test_stage24_rejects_verification_result_key_closure_tampering(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
) -> None:
    run_dir = tmp_path / "run"
    config, _paper_text, planned_keys = _prepare_stage23_fixture(run_dir)
    _run_stage23_verified(run_dir, config, planned_keys, monkeypatch)
    report_path = run_dir / "stage-23" / "verification_report.json"
    report = json.loads(report_path.read_text())
    if mutation == "missing":
        report["results"].pop()
    elif mutation == "extra":
        report["results"].append(
            {
                "cite_key": "extra2024",
                "title": "Extra",
                "status": "verified",
                "confidence": 1.0,
                "method": "title_search",
                "details": "",
            }
        )
    else:
        report["results"].append(dict(report["results"][0]))
    report_path.write_text(json.dumps(report), encoding="utf-8")
    stage24 = run_dir / "stage-24"
    stage24.mkdir()
    result = _execute_truth_audit(
        stage24, run_dir, config, AdapterBundle(), llm=_SequenceLLM([])  # type: ignore[arg-type]
    )
    assert result.status is StageStatus.FAILED
    assert (
        "verification result key closure" in (result.error or "")
        or "duplicate verification cite_key" in (result.error or "")
    )


def test_e9_replays_complete_citation_evidence_chain(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_dir = tmp_path / "run"
    _config_value, _planned_keys = _prepare_e9_run(run_dir, monkeypatch)
    contract_path = find_stage09_contract(run_dir)
    assert contract_path is not None
    audit_citation_evidence(
        run_dir,
        contract_path=contract_path,
        contract=load_contract(contract_path),
    )


def test_e9_rejects_support_artifact_single_point_tampering(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_dir = tmp_path / "run"
    _config_value, _planned_keys = _prepare_e9_run(run_dir, monkeypatch)
    support_path = run_dir / "stage-24" / "citation_support.json"
    support = json.loads(support_path.read_text())
    support["instances"][0]["claim_text"] += " tampered"
    support_path.write_text(json.dumps(support), encoding="utf-8")
    contract_path = find_stage09_contract(run_dir)
    assert contract_path is not None
    with pytest.raises(CitationAuditError) as exc_info:
        audit_citation_evidence(
            run_dir,
            contract_path=contract_path,
            contract=load_contract(contract_path),
        )
    assert exc_info.value.code == "citation_support_replay_failed"


def test_e9_replays_stage5_prefilter_instead_of_trusting_report_partition(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_dir = tmp_path / "run"
    _config_value, _planned_keys = _prepare_e9_run(run_dir, monkeypatch)
    shortlist_path = run_dir / "stage-05" / "shortlist.jsonl"
    shortlist_lines = shortlist_path.read_text().splitlines()
    removed = json.loads(shortlist_lines.pop())
    shortlist_text = "\n".join(shortlist_lines) + "\n"
    shortlist_path.write_text(shortlist_text, encoding="utf-8")
    report_path = run_dir / "stage-05" / "screening_report.json"
    report = json.loads(report_path.read_text())
    identity = removed["source_identity"]
    report["screening_output_sha256"] = sha256_text(shortlist_text)
    report["selected_candidate_ids"].remove(identity)
    report["screened_candidate_ids"].remove(identity)
    report["prefilter_rejected_candidate_ids"].append(identity)
    report_path.write_text(json.dumps(report), encoding="utf-8")
    contract_path = find_stage09_contract(run_dir)
    assert contract_path is not None
    with pytest.raises(CitationAuditError) as exc_info:
        audit_citation_evidence(
            run_dir,
            contract_path=contract_path,
            contract=load_contract(contract_path),
        )
    assert exc_info.value.code == "literature_screening_replay_failed"
    assert "prefilter partition replay" in exc_info.value.message


def test_e9_rejects_active_config_contract_scope_divergence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_dir = tmp_path / "run"
    _config_value, _planned_keys = _prepare_e9_run(run_dir, monkeypatch)
    contract_path = find_stage09_contract(run_dir)
    assert contract_path is not None
    contract_text = contract_path.read_text().replace(
        "claim_scope: pipeline_validation", "claim_scope: exploratory"
    )
    contract_path.write_text(contract_text, encoding="utf-8")
    with pytest.raises(CitationAuditError) as exc_info:
        audit_citation_evidence(
            run_dir,
            contract_path=contract_path,
            contract=load_contract(contract_path),
        )
    assert exc_info.value.code == "citation_contract_scope_mismatch"


def test_e9_rejects_verification_summary_self_report_tampering(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_dir = tmp_path / "run"
    _config_value, _planned_keys = _prepare_e9_run(run_dir, monkeypatch)
    report_path = run_dir / "stage-23" / "verification_report.json"
    report = json.loads(report_path.read_text())
    report["summary"]["verified"] -= 1
    report_path.write_text(json.dumps(report), encoding="utf-8")
    contract_path = find_stage09_contract(run_dir)
    assert contract_path is not None
    with pytest.raises(CitationAuditError) as exc_info:
        audit_citation_evidence(
            run_dir,
            contract_path=contract_path,
            contract=load_contract(contract_path),
        )
    assert exc_info.value.code == "citation_verification_replay_failed"


@pytest.mark.parametrize("artifact", ["claims", "citations", "truth"])
def test_e9_rejects_stage24_cross_artifact_tampering(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    artifact: str,
) -> None:
    run_dir = tmp_path / "run"
    _config_value, _planned_keys = _prepare_e9_run(run_dir, monkeypatch)
    if artifact == "claims":
        path = run_dir / "stage-24" / "claims.json"
        payload = json.loads(path.read_text())
        support_claim = next(
            row for row in payload["claims"] if row["id"].startswith("support-")
        )
        support_claim["status"] = "unsupported"
    elif artifact == "citations":
        path = run_dir / "stage-24" / "citations.json"
        payload = json.loads(path.read_text())
        payload["instances"] = list(reversed(payload["instances"]))
    else:
        path = run_dir / "stage-24" / "truth_audit.json"
        payload = json.loads(path.read_text())
        payload["citation_support_sha256"] = "0" * 64
    path.write_text(json.dumps(payload), encoding="utf-8")
    contract_path = find_stage09_contract(run_dir)
    assert contract_path is not None
    with pytest.raises(CitationAuditError) as exc_info:
        audit_citation_evidence(
            run_dir,
            contract_path=contract_path,
            contract=load_contract(contract_path),
        )
    assert exc_info.value.code == "citation_support_replay_failed"


def test_stage22_rejects_multi_key_markers_when_canonical_bib_is_missing(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    config = _real_config_snapshot(run_dir)
    stage19 = run_dir / "stage-19"
    stage19.mkdir(parents=True)
    (stage19 / "paper_revised.md").write_text(
        "## Introduction\n\nPrior work [alpha2020, beta2021].\n",
        encoding="utf-8",
    )
    stage22 = run_dir / "stage-22"
    stage22.mkdir()
    result = _execute_export_publish(
        stage22, run_dir, config, AdapterBundle(), llm=None
    )
    assert result.status is StageStatus.FAILED
    assert "Canonical bibliography" in (result.error or "")


def test_resumed_config_without_active_pointer_fails_closed(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    config = _real_config_snapshot(run_dir)
    (run_dir / "active_config_snapshot.json").unlink()
    (run_dir / "config_snapshot_history.jsonl").unlink()
    (run_dir / "config.resumed-20260711-120000.yaml").write_text(
        (run_dir / "config.yaml").read_text(), encoding="utf-8"
    )
    with pytest.raises(CitationPolicyContractError, match="without active config pointer"):
        resolve_active_config_snapshot(run_dir, config)


def test_config_history_cannot_be_truncated_before_append(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    _real_config_snapshot(run_dir)
    base_text = (run_dir / "config.yaml").read_text(encoding="utf-8")
    resumed = run_dir / "config.resumed-20260711-120000.yaml"
    resumed.write_text(base_text, encoding="utf-8")
    write_active_config_binding(run_dir, resumed)
    history_path = run_dir / "config_snapshot_history.jsonl"
    history_lines = history_path.read_text(encoding="utf-8").splitlines()
    assert len(history_lines) == 2
    history_path.write_text(history_lines[0] + "\n", encoding="utf-8")
    third = run_dir / "config.resumed-20260711-120001.yaml"
    third.write_text(base_text, encoding="utf-8")
    with pytest.raises(CitationPolicyContractError, match="history hash mismatch"):
        write_active_config_binding(run_dir, third)


def test_active_config_binding_updates_and_replays_checkpoint(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "checkpoint.json").write_text(
        canonical_json_text({"last_completed_stage": 15, "run_id": "test"}),
        encoding="utf-8",
    )
    source = Path("config.deepseek.sectional-dry-run.yaml")
    raw = yaml.safe_load(source.read_text(encoding="utf-8"))
    snapshot = run_dir / "config.yaml"
    snapshot.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    config = RCConfig.from_dict(raw, project_root=run_dir, check_paths=False)
    write_active_config_binding(run_dir, snapshot)
    checkpoint_path = run_dir / "checkpoint.json"
    checkpoint = json.loads(checkpoint_path.read_text())
    assert checkpoint["active_config_snapshot_path"] == "config.yaml"
    resolve_active_config_snapshot(run_dir, config)

    checkpoint["active_config_snapshot_sha256"] = "0" * 64
    checkpoint_path.write_text(canonical_json_text(checkpoint), encoding="utf-8")
    with pytest.raises(CitationPolicyContractError, match="checkpoint config binding"):
        resolve_active_config_snapshot(run_dir, config)


@pytest.mark.parametrize(
    ("stage_name", "executor"),
    [
        ("stage-17", _execute_paper_draft),
        ("stage-18", _execute_peer_review),
        ("stage-20", _execute_quality_gate),
    ],
)
def test_paper_consumers_reject_tampered_effective_policy(
    tmp_path: Path, stage_name: str, executor: Any
) -> None:
    run_dir = tmp_path / "run"
    config = _real_config_snapshot(run_dir)
    shortlist = _prepare_stage5(run_dir, [_candidate(1)], config)
    stage6 = run_dir / "stage-06"
    stage6.mkdir()
    assert _execute_knowledge_extract(
        stage6,
        run_dir,
        config,
        AdapterBundle(),
        llm=_SequenceLLM([_card_response(shortlist)]),  # type: ignore[arg-type]
    ).status is StageStatus.DONE
    stage16 = run_dir / "stage-16"
    stage16.mkdir()
    assert _execute_paper_outline(
        stage16, run_dir, config, AdapterBundle(), llm=None
    ).status is StageStatus.DONE
    policy_path = stage16 / "citation_policy_effective.json"
    policy = json.loads(policy_path.read_text())
    policy["eligible_count"] = 2
    policy_path.write_text(canonical_json_text(policy), encoding="utf-8")
    stage_dir = run_dir / stage_name
    stage_dir.mkdir(parents=True, exist_ok=True)
    result = executor(stage_dir, run_dir, config, AdapterBundle(), llm=None)
    assert result.status is StageStatus.FAILED
    assert "Effective citation policy is invalid" in (result.error or "")
