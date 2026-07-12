"""Stage 5 strict batched-screening contract tests."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from researchclaw.adapters import AdapterBundle
from researchclaw.literature.citation_identity import seal_citation_collection
from researchclaw.literature.screening import (
    MAX_SCREEN_CANDIDATES,
    SCREEN_BATCH_SIZE,
    ScreeningContractError,
    build_screening_report,
    parse_screening_report,
    parse_screening_response,
    sha256_text,
)
from researchclaw.pipeline.stage_impls._literature import _execute_literature_screen
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
            raise RuntimeError("unexpected extra screening call")
        return SimpleNamespace(content=self.responses.pop(0))


def _candidate(
    index: int,
    *,
    title: str | None = None,
    doi: str | None = None,
    arxiv_id: str = "",
) -> dict[str, Any]:
    return {
        "paper_id": f"provider-{index}",
        "title": title or f"Hardware Detection Study {index}",
        "authors": [{"name": "Jane Smith"}],
        "year": 2024,
        "abstract": "Hardware runtime detection using performance counters.",
        "venue": "Security Conference",
        "citation_count": 1000 - index,
        "doi": doi if doi is not None else f"10.1000/{index:03d}",
        "arxiv_id": arxiv_id,
        "url": "",
        "source": "semantic_scholar",
    }


def _response(
    batch_id: str,
    source_ids: list[str],
    *,
    keep_ids: set[str] | None = None,
) -> str:
    keep = keep_ids if keep_ids is not None else set(source_ids)
    return json.dumps(
        {
            "schema_version": 1,
            "batch_id": batch_id,
            "decisions": [
                {
                    "source_identity": source_id,
                    "decision": "keep" if source_id in keep else "reject",
                    "relevance_score": 0.9 if source_id in keep else 0.1,
                    "quality_score": 0.8 if source_id in keep else 0.2,
                    "reason": "directly relevant" if source_id in keep else "off topic",
                }
                for source_id in source_ids
            ],
        }
    )


def _config(claim_scope: str = "pipeline_validation") -> SimpleNamespace:
    return SimpleNamespace(
        research=SimpleNamespace(
            topic="hardware runtime detection",
            domains=("hardware security",),
            quality_threshold=6.0,
        ),
        experiment=SimpleNamespace(claim_scope=claim_scope),
    )


def _write_candidates(run_dir: Path, rows: list[dict[str, Any]]) -> list[str]:
    sealed = seal_citation_collection(rows)
    stage4 = run_dir / "stage-04"
    stage4.mkdir(parents=True)
    (stage4 / "candidates.jsonl").write_text(
        sealed.candidates_jsonl, encoding="utf-8"
    )
    (stage4 / "references.bib").write_text(
        sealed.bibliography, encoding="utf-8"
    )
    (stage4 / "cite_key_registry.json").write_text(
        json.dumps(sealed.registry, ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    return [str(row["source_identity"]) for row in sealed.candidates]


def test_screening_response_requires_exact_candidate_id_closure() -> None:
    text = _response("screen-batch-001", ["doi:10.1000/a"])
    with pytest.raises(ScreeningContractError, match="closure mismatch"):
        parse_screening_response(
            text,
            expected_batch_id="screen-batch-001",
            expected_source_ids=["doi:10.1000/a", "doi:10.1000/b"],
        )


def test_screening_response_rejects_duplicate_json_key() -> None:
    text = (
        '{"schema_version":1,"schema_version":1,'
        '"batch_id":"screen-batch-001","decisions":[]}'
    )
    with pytest.raises(ScreeningContractError, match="duplicate JSON key"):
        parse_screening_response(
            text,
            expected_batch_id="screen-batch-001",
            expected_source_ids=[],
        )


def test_screening_response_rejects_boolean_score() -> None:
    payload = json.loads(_response("screen-batch-001", ["doi:10.1000/a"]))
    payload["decisions"][0]["relevance_score"] = True
    with pytest.raises(ScreeningContractError, match="must be numeric"):
        parse_screening_response(
            json.dumps(payload),
            expected_batch_id="screen-batch-001",
            expected_source_ids=["doi:10.1000/a"],
        )


def test_screening_response_rejects_decision_score_contradiction() -> None:
    payload = json.loads(_response("screen-batch-001", ["doi:10.1000/a"]))
    payload["decisions"][0]["quality_score"] = 0.4
    with pytest.raises(ScreeningContractError, match="contradicts"):
        parse_screening_response(
            json.dumps(payload),
            expected_batch_id="screen-batch-001",
            expected_source_ids=["doi:10.1000/a"],
            minimum_quality_score=0.6,
        )


def test_stage5_strict_screen_does_not_backfill_to_fifteen(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    ids = _write_candidates(run_dir, [_candidate(1), _candidate(2), _candidate(3)])
    llm = _SequenceLLM(
        [_response("screen-batch-001", ids, keep_ids={ids[0]})]
    )
    stage_dir = run_dir / "stage-05"
    stage_dir.mkdir()

    result = _execute_literature_screen(
        stage_dir,
        run_dir,
        _config(),  # type: ignore[arg-type]
        AdapterBundle(),
        llm=llm,  # type: ignore[arg-type]
    )

    assert result.status is StageStatus.DONE
    shortlist = (stage_dir / "shortlist.jsonl").read_text(encoding="utf-8")
    rows = [json.loads(line) for line in shortlist.splitlines()]
    assert [row["source_identity"] for row in rows] == [ids[0]]
    assert "Template fallback" not in shortlist
    report = json.loads((stage_dir / "screening_report.json").read_text())
    assert report["selected_candidate_ids"] == [ids[0]]
    assert report["screening_complete"] is True
    assert report["degraded"] is False


def test_stage5_repairs_one_malformed_batch_once(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    ids = _write_candidates(run_dir, [_candidate(1), _candidate(2)])
    malformed = _response("screen-batch-001", [ids[0]])
    llm = _SequenceLLM(
        [malformed, _response("screen-batch-001", ids, keep_ids=set(ids))]
    )
    stage_dir = run_dir / "stage-05"
    stage_dir.mkdir()

    result = _execute_literature_screen(
        stage_dir,
        run_dir,
        _config(),  # type: ignore[arg-type]
        AdapterBundle(),
        llm=llm,  # type: ignore[arg-type]
    )

    assert result.status is StageStatus.DONE
    assert len(llm.calls) == 2
    assert "PREVIOUS RESPONSE VIOLATED" in llm.calls[1]


def test_pipeline_validation_can_continue_only_verified_batches_degraded(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    ids = _write_candidates(
        run_dir, [_candidate(index) for index in range(1, SCREEN_BATCH_SIZE + 2)]
    )
    first_ids = ids[:SCREEN_BATCH_SIZE]
    second_ids = ids[SCREEN_BATCH_SIZE:]
    malformed = _response("screen-batch-002", [])
    llm = _SequenceLLM(
        [
            _response("screen-batch-001", first_ids, keep_ids={first_ids[0]}),
            malformed,
            malformed,
        ]
    )
    stage_dir = run_dir / "stage-05"
    stage_dir.mkdir()

    result = _execute_literature_screen(
        stage_dir,
        run_dir,
        _config("pipeline_validation"),  # type: ignore[arg-type]
        AdapterBundle(),
        llm=llm,  # type: ignore[arg-type]
    )

    assert result.status is StageStatus.DONE
    assert result.decision == "degraded"
    report = json.loads((stage_dir / "screening_report.json").read_text())
    assert report["selected_candidate_ids"] == [first_ids[0]]
    assert report["unscreened_candidate_ids"] == second_ids
    assert report["screening_complete"] is False
    assert report["degraded"] is True


def test_research_release_rejects_any_failed_batch(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    ids = _write_candidates(
        run_dir, [_candidate(index) for index in range(1, SCREEN_BATCH_SIZE + 2)]
    )
    first_ids = ids[:SCREEN_BATCH_SIZE]
    malformed = _response("screen-batch-002", [])
    llm = _SequenceLLM(
        [
            _response("screen-batch-001", first_ids, keep_ids={first_ids[0]}),
            malformed,
            malformed,
        ]
    )
    stage_dir = run_dir / "stage-05"
    stage_dir.mkdir()

    result = _execute_literature_screen(
        stage_dir,
        run_dir,
        _config("research_release"),  # type: ignore[arg-type]
        AdapterBundle(),
        llm=llm,  # type: ignore[arg-type]
    )

    assert result.status is StageStatus.FAILED
    assert "incomplete" in (result.error or "")
    assert not (stage_dir / "shortlist.jsonl").exists()
    assert (stage_dir / "screening_partial.jsonl").exists()
    report = json.loads((stage_dir / "screening_report.json").read_text())
    assert report["screening_complete"] is False
    assert report["screening_output_path"] == "stage-05/screening_partial.jsonl"


def test_exploratory_rejects_failed_batch_instead_of_degrading(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    ids = _write_candidates(
        run_dir, [_candidate(index) for index in range(1, SCREEN_BATCH_SIZE + 2)]
    )
    first_ids = ids[:SCREEN_BATCH_SIZE]
    malformed = _response("screen-batch-002", [])
    llm = _SequenceLLM(
        [
            _response("screen-batch-001", first_ids, keep_ids={first_ids[0]}),
            malformed,
            malformed,
        ]
    )
    stage_dir = run_dir / "stage-05"
    stage_dir.mkdir()

    result = _execute_literature_screen(
        stage_dir,
        run_dir,
        _config("exploratory"),  # type: ignore[arg-type]
        AdapterBundle(),
        llm=llm,  # type: ignore[arg-type]
    )

    assert result.status is StageStatus.FAILED
    assert "exploratory screening is incomplete" in (result.error or "")
    assert not (stage_dir / "shortlist.jsonl").exists()
    assert (stage_dir / "screening_partial.jsonl").exists()


def test_research_release_marks_unattempted_batches_unscreened(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    ids = _write_candidates(
        run_dir,
        [_candidate(index) for index in range(1, 2 * SCREEN_BATCH_SIZE + 2)],
    )
    first_ids = ids[:SCREEN_BATCH_SIZE]
    malformed = _response("screen-batch-002", [])
    llm = _SequenceLLM(
        [
            _response("screen-batch-001", first_ids, keep_ids={first_ids[0]}),
            malformed,
            malformed,
        ]
    )
    stage_dir = run_dir / "stage-05"
    stage_dir.mkdir()

    result = _execute_literature_screen(
        stage_dir,
        run_dir,
        _config("research_release"),  # type: ignore[arg-type]
        AdapterBundle(),
        llm=llm,  # type: ignore[arg-type]
    )

    assert result.status is StageStatus.FAILED
    report = json.loads((stage_dir / "screening_report.json").read_text())
    assert report["batch_count"] == 3
    assert report["unscreened_candidate_ids"] == ids[SCREEN_BATCH_SIZE:]
    assert [item["batch_id"] for item in report["failed_batches"]] == [
        "screen-batch-002"
    ]


def test_semantic_duplicate_identities_produce_one_shortlist_row(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    common_title = "Hardware Detection with Performance Counters"
    rows = [
        _candidate(1, title=common_title, doi="10.1000/record"),
        _candidate(2, title=common_title, doi="", arxiv_id="2401.00001"),
    ]
    ids = _write_candidates(run_dir, rows)
    llm = _SequenceLLM([_response("screen-batch-001", ids)])
    stage_dir = run_dir / "stage-05"
    stage_dir.mkdir()

    result = _execute_literature_screen(
        stage_dir,
        run_dir,
        _config(),  # type: ignore[arg-type]
        AdapterBundle(),
        llm=llm,  # type: ignore[arg-type]
    )

    assert result.status is StageStatus.DONE
    shortlist = (stage_dir / "shortlist.jsonl").read_text().splitlines()
    report = json.loads((stage_dir / "screening_report.json").read_text())
    assert len(shortlist) == 1
    assert len(report["semantic_duplicate_candidate_ids"]) == 1


def test_no_llm_fails_without_template_shortlist(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    _write_candidates(run_dir, [_candidate(1)])
    stage_dir = run_dir / "stage-05"
    stage_dir.mkdir()

    result = _execute_literature_screen(
        stage_dir,
        run_dir,
        _config(),  # type: ignore[arg-type]
        AdapterBundle(),
        llm=None,
    )

    assert result.status is StageStatus.FAILED
    assert not (stage_dir / "shortlist.jsonl").exists()
    assert (stage_dir / "screening_partial.jsonl").read_text() == ""
    report = json.loads((stage_dir / "screening_report.json").read_text())
    assert report["failed_batches"][0]["batch_id"] == "screen-batch-001"
    assert report["failed_batches"][0]["error"] == "LLM client unavailable"


def test_stage5_failure_clears_stale_owned_artifacts(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    stage4 = run_dir / "stage-04"
    stage4.mkdir(parents=True)
    (stage4 / "candidates.jsonl").write_text("not-json\n", encoding="utf-8")
    stage_dir = run_dir / "stage-05"
    stage_dir.mkdir()
    for artifact_name in (
        "shortlist.jsonl",
        "screening_partial.jsonl",
        "screening_report.json",
        "screen_meta.json",
    ):
        (stage_dir / artifact_name).write_text("stale", encoding="utf-8")

    result = _execute_literature_screen(
        stage_dir,
        run_dir,
        _config(),  # type: ignore[arg-type]
        AdapterBundle(),
        llm=None,
    )

    assert result.status is StageStatus.FAILED
    assert "sealed citation collection is invalid" in (result.error or "")
    for artifact_name in (
        "shortlist.jsonl",
        "screening_partial.jsonl",
        "screening_report.json",
        "screen_meta.json",
    ):
        assert not (stage_dir / artifact_name).exists()


def test_stage5_rejects_malformed_candidate_field_types(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    _write_candidates(run_dir, [_candidate(1)])
    candidate_path = run_dir / "stage-04" / "candidates.jsonl"
    row = json.loads(candidate_path.read_text(encoding="utf-8"))
    row["citation_count"] = True
    mutated_candidates = json.dumps(row) + "\n"
    candidate_path.write_text(mutated_candidates, encoding="utf-8")
    registry_path = run_dir / "stage-04" / "cite_key_registry.json"
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    registry["candidates_sha256"] = sha256_text(mutated_candidates)
    registry_path.write_text(
        json.dumps(registry, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    stage_dir = run_dir / "stage-05"
    stage_dir.mkdir()

    result = _execute_literature_screen(
        stage_dir,
        run_dir,
        _config(),  # type: ignore[arg-type]
        AdapterBundle(),
        llm=None,
    )

    assert result.status is StageStatus.FAILED
    assert "citation_count must be a nonnegative integer" in (result.error or "")
    assert result.artifacts == ()


def test_stage5_rejects_candidate_mutation_against_registry(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    _write_candidates(run_dir, [_candidate(1)])
    candidate_path = run_dir / "stage-04" / "candidates.jsonl"
    row = json.loads(candidate_path.read_text(encoding="utf-8"))
    row["title"] = "Mutated after Stage 4 sealing"
    candidate_path.write_text(json.dumps(row) + "\n", encoding="utf-8")
    stage_dir = run_dir / "stage-05"
    stage_dir.mkdir()

    result = _execute_literature_screen(
        stage_dir,
        run_dir,
        _config(),  # type: ignore[arg-type]
        AdapterBundle(),
        llm=None,
    )

    assert result.status is StageStatus.FAILED
    assert "candidates_sha256 mismatch" in (result.error or "")
    assert result.artifacts == ()


def test_stage5_ignores_self_consistent_shadow_stage5_inputs(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    canonical_ids = _write_candidates(run_dir, [_candidate(1)])
    shadow = seal_citation_collection(
        [_candidate(2, title="Hardware Runtime Detection Shadow")]
    )
    stage_dir = run_dir / "stage-05"
    stage_dir.mkdir()
    (stage_dir / "candidates.jsonl").write_text(
        shadow.candidates_jsonl, encoding="utf-8"
    )
    (stage_dir / "references.bib").write_text(
        shadow.bibliography, encoding="utf-8"
    )
    (stage_dir / "cite_key_registry.json").write_text(
        json.dumps(shadow.registry, ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    llm = _SequenceLLM([_response("screen-batch-001", canonical_ids)])

    result = _execute_literature_screen(
        stage_dir,
        run_dir,
        _config(),  # type: ignore[arg-type]
        AdapterBundle(),
        llm=llm,  # type: ignore[arg-type]
    )

    assert result.status is StageStatus.DONE
    shortlist = [
        json.loads(line)
        for line in (stage_dir / "shortlist.jsonl").read_text().splitlines()
    ]
    assert [row["source_identity"] for row in shortlist] == canonical_ids


def test_stage5_report_write_failure_does_not_publish_shortlist(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_dir = tmp_path / "run"
    ids = _write_candidates(run_dir, [_candidate(1)])
    stage_dir = run_dir / "stage-05"
    stage_dir.mkdir()
    llm = _SequenceLLM([_response("screen-batch-001", ids)])
    original_write_text = Path.write_text

    def fail_report_write(
        path: Path, data: str, *args: object, **kwargs: object
    ) -> int:
        if path.name == "screening_report.json":
            raise OSError("simulated report persistence failure")
        return original_write_text(path, data, *args, **kwargs)

    monkeypatch.setattr(Path, "write_text", fail_report_write)

    with pytest.raises(OSError, match="simulated report persistence failure"):
        _execute_literature_screen(
            stage_dir,
            run_dir,
            _config(),  # type: ignore[arg-type]
            AdapterBundle(),
            llm=llm,  # type: ignore[arg-type]
        )

    assert not (stage_dir / "shortlist.jsonl").exists()


def test_stage5_rejects_invalid_quality_threshold(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    _write_candidates(run_dir, [_candidate(1)])
    config = _config()
    config.research.quality_threshold = float("nan")
    stage_dir = run_dir / "stage-05"
    stage_dir.mkdir()

    result = _execute_literature_screen(
        stage_dir,
        run_dir,
        config,  # type: ignore[arg-type]
        AdapterBundle(),
        llm=None,
    )

    assert result.status is StageStatus.FAILED
    assert "quality threshold must be between 0 and 10" in (result.error or "")
    assert result.artifacts == ()


def test_stage5_caps_model_screening_without_backfill(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    ids = _write_candidates(
        run_dir,
        [_candidate(index) for index in range(1, MAX_SCREEN_CANDIDATES + 2)],
    )
    admitted_ids = ids[:MAX_SCREEN_CANDIDATES]
    responses = [
        _response(
            f"screen-batch-{index // SCREEN_BATCH_SIZE + 1:03d}",
            admitted_ids[index:index + SCREEN_BATCH_SIZE],
        )
        for index in range(0, len(admitted_ids), SCREEN_BATCH_SIZE)
    ]
    llm = _SequenceLLM(responses)
    stage_dir = run_dir / "stage-05"
    stage_dir.mkdir()

    result = _execute_literature_screen(
        stage_dir,
        run_dir,
        _config(),  # type: ignore[arg-type]
        AdapterBundle(),
        llm=llm,  # type: ignore[arg-type]
    )

    assert result.status is StageStatus.DONE
    report = json.loads((stage_dir / "screening_report.json").read_text())
    assert report["batch_count"] == MAX_SCREEN_CANDIDATES // SCREEN_BATCH_SIZE
    assert report["screened_candidate_ids"] == admitted_ids
    assert report["prefilter_rejected_candidate_ids"] == ids[MAX_SCREEN_CANDIDATES:]
    assert report["unscreened_candidate_ids"] == []
    assert len(llm.calls) == MAX_SCREEN_CANDIDATES // SCREEN_BATCH_SIZE


def test_screening_report_rejects_hash_mutation() -> None:
    candidates_text = '{"source_identity":"doi:10.1000/a"}\n'
    shortlist_text = candidates_text
    report = build_screening_report(
        candidates_sha256=sha256_text(candidates_text),
        registry_sha256=sha256_text("registry"),
        references_sha256=sha256_text("references"),
        screening_output_path="stage-05/shortlist.jsonl",
        screening_output_sha256=sha256_text(shortlist_text),
        minimum_quality_score=0.6,
        claim_scope="pipeline_validation",
        candidate_ids=["doi:10.1000/a"],
        prefilter_rejected_ids=[],
        screened_ids=["doi:10.1000/a"],
        selected_ids=["doi:10.1000/a"],
        semantic_duplicate_ids=[],
        unscreened_ids=[],
        batch_count=1,
        failed_batches=[],
        degraded=False,
        degradation_codes=[],
    )
    with pytest.raises(ScreeningContractError, match="output sha256 mismatch"):
        parse_screening_report(
            json.dumps(report),
            candidates_text_sha256=sha256_text(candidates_text),
            registry_text_sha256=sha256_text("registry"),
            references_text_sha256=sha256_text("references"),
            expected_screening_output_path="stage-05/shortlist.jsonl",
            screening_output_text_sha256=sha256_text(shortlist_text + "mutation"),
            expected_minimum_quality_score=0.6,
            expected_claim_scope="pipeline_validation",
            expected_candidate_ids=["doi:10.1000/a"],
            expected_selected_ids=["doi:10.1000/a"],
        )


def test_screening_report_rejects_selected_id_self_assertion() -> None:
    candidates_text = '{"source_identity":"doi:10.1000/a"}\n'
    shortlist_text = candidates_text
    report = build_screening_report(
        candidates_sha256=sha256_text(candidates_text),
        registry_sha256=sha256_text("registry"),
        references_sha256=sha256_text("references"),
        screening_output_path="stage-05/shortlist.jsonl",
        screening_output_sha256=sha256_text(shortlist_text),
        minimum_quality_score=0.6,
        claim_scope="pipeline_validation",
        candidate_ids=["doi:10.1000/a"],
        prefilter_rejected_ids=[],
        screened_ids=["doi:10.1000/a"],
        selected_ids=["doi:10.1000/a"],
        semantic_duplicate_ids=[],
        unscreened_ids=[],
        batch_count=1,
        failed_batches=[],
        degraded=False,
        degradation_codes=[],
    )
    report["selected_candidate_ids"] = []
    with pytest.raises(ScreeningContractError, match="selected_candidate_ids mismatch"):
        parse_screening_report(
            json.dumps(report),
            candidates_text_sha256=sha256_text(candidates_text),
            registry_text_sha256=sha256_text("registry"),
            references_text_sha256=sha256_text("references"),
            expected_screening_output_path="stage-05/shortlist.jsonl",
            screening_output_text_sha256=sha256_text(shortlist_text),
            expected_minimum_quality_score=0.6,
            expected_claim_scope="pipeline_validation",
            expected_candidate_ids=["doi:10.1000/a"],
            expected_selected_ids=["doi:10.1000/a"],
        )


@pytest.mark.parametrize(
    ("section_heading", "required_fields"),
    [
        (
            "### 7.1 Stage 4 cite-key registry",
            {"candidates_path", "references_path", "entries"},
        ),
        (
            "### 7.2 Stage 5 screening report",
            {
                "registry_path",
                "references_path",
                "screening_output_path",
                "screening_output_sha256",
            },
        ),
    ],
)
def test_citation_spec_json_examples_have_no_duplicate_keys(
    section_heading: str, required_fields: set[str]
) -> None:
    spec = (
        Path(__file__).parents[1] / "docs" / "CITATION_EVIDENCE_PIPELINE_SPEC.md"
    ).read_text(encoding="utf-8")
    section_start = spec.index(section_heading)
    fence_start = spec.index("```json\n", section_start) + len("```json\n")
    fence_end = spec.index("\n```", fence_start)

    def reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise AssertionError(f"duplicate spec JSON key: {key}")
            result[key] = value
        return result

    payload = json.loads(
        spec[fence_start:fence_end], object_pairs_hook=reject_duplicate_keys
    )
    assert required_fields <= set(payload)
