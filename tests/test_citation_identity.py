"""Citation-key v2 identity registry and Stage 4 integration tests."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from researchclaw.adapters import AdapterBundle
from researchclaw.literature.citation_identity import (
    CITE_KEY_VERSION,
    CitationIdentityError,
    base_cite_key_for_candidate,
    normalize_arxiv_id,
    normalize_arxiv_url,
    parse_cite_key_registry,
    seal_citation_collection,
    source_identity_for_candidate,
    validate_registry_artifacts,
)
from researchclaw.literature.models import Author, Paper
from researchclaw.pipeline.stage_impls._literature import _execute_literature_collect
from researchclaw.pipeline.stages import StageStatus


def _candidate(
    *,
    paper_id: str,
    title: str = "Runtime Detection for Transient Execution",
    doi: str = "",
    arxiv_id: str = "",
    citation_count: int = 10,
) -> dict[str, object]:
    return {
        "paper_id": paper_id,
        "title": title,
        "authors": [{"name": "Jane Smith"}],
        "year": 2024,
        "abstract": "We study runtime detection using hardware counters.",
        "venue": "Security Conference",
        "citation_count": citation_count,
        "doi": doi,
        "arxiv_id": arxiv_id,
        "url": "",
        "source": "semantic_scholar",
    }


def test_cite_key_v2_removes_html_and_nfkc_markup() -> None:
    row = _candidate(
        paper_id="s2-markup",
        title="＜scp＞EC＜/scp＞ Runtime Detection",
        doi="10.1000/markup",
    )
    assert base_cite_key_for_candidate(row) == "smith2024runtime"
    paper = Paper(
        paper_id="s2-markup",
        title="<scp>EC</scp> Runtime Detection",
        authors=(Author(name="Jane Smith"),),
        year=2024,
        doi="10.1000/markup",
    )
    assert paper.cite_key == "smith2024runtime"
    assert paper.to_dict()["cite_key_version"] == CITE_KEY_VERSION


def test_source_identity_normalizes_doi_before_provider_id() -> None:
    row = _candidate(paper_id="s2-1", doi="HTTPS://doi.org/10.1000/ABC ")
    assert source_identity_for_candidate(row) == "doi:10.1000/abc"


@pytest.mark.parametrize(
    "url",
    [
        "https://portal.example/papers/1234567",
        "https://aclanthology.org/2023.10577.pdf",
    ],
)
def test_non_arxiv_urls_cannot_create_arxiv_identity(url: str) -> None:
    row = _candidate(paper_id="", doi="")
    row["url"] = url
    assert normalize_arxiv_url(url) == ""
    assert source_identity_for_candidate(row).startswith("metadata:")


def test_arxiv_field_and_canonical_url_normalize_versions() -> None:
    assert normalize_arxiv_id("arXiv:2301.00001v3") == "2301.00001"
    assert (
        normalize_arxiv_url("https://arxiv.org/pdf/2301.00001v3.pdf")
        == "2301.00001"
    )


def test_same_doi_is_deduplicated_deterministically() -> None:
    lower_quality = _candidate(
        paper_id="s2-low",
        doi="10.1000/shared",
        citation_count=1,
    )
    higher_quality = _candidate(
        paper_id="oalex-high",
        doi="https://doi.org/10.1000/SHARED",
        citation_count=100,
    )
    sealed = seal_citation_collection([lower_quality, higher_quality])
    assert len(sealed.candidates) == 1
    assert sealed.candidates[0]["citation_count"] == 100
    assert sealed.registry["entries"] == [
        {
            "source_identity": "doi:10.1000/shared",
            "cite_key": "smith2024runtime",
            "base_key": "smith2024runtime",
            "collision_suffix": None,
        }
    ]


def test_same_identity_with_conflicting_titles_is_rejected() -> None:
    first = _candidate(paper_id="s2-a", title="Paper Alpha", doi="10.1000/shared")
    second = _candidate(paper_id="s2-b", title="Paper Beta", doi="10.1000/shared")
    with pytest.raises(CitationIdentityError, match="conflicting titles"):
        seal_citation_collection([first, second])


def test_distinct_identity_collision_suffixes_are_stable_across_input_order() -> None:
    first = _candidate(paper_id="s2-a", doi="10.1000/a")
    second = _candidate(paper_id="s2-b", doi="10.1000/b")
    forward = seal_citation_collection([first, second])
    reverse = seal_citation_collection([second, first])
    forward_map = {
        entry["source_identity"]: entry["cite_key"]
        for entry in forward.registry["entries"]
    }
    reverse_map = {
        entry["source_identity"]: entry["cite_key"]
        for entry in reverse.registry["entries"]
    }
    assert forward_map == reverse_map
    assert len(set(forward_map.values())) == 2
    assert all(key.startswith("smith2024runtime") for key in forward_map.values())
    assert all(len(key) == len("smith2024runtime") + 8 for key in forward_map.values())


def test_registry_hashes_exact_candidates_and_bibliography() -> None:
    sealed = seal_citation_collection(
        [
            _candidate(paper_id="s2-a", doi="10.1000/a"),
            _candidate(
                paper_id="s2-b",
                title="Counter Measurement for Detection",
                doi="10.1000/b",
            ),
        ]
    )
    assert sealed.registry["candidates_sha256"] == hashlib.sha256(
        sealed.candidates_jsonl.encode("utf-8")
    ).hexdigest()
    assert sealed.registry["references_sha256"] == hashlib.sha256(
        sealed.bibliography.encode("utf-8")
    ).hexdigest()
    for row in sealed.candidates:
        assert row["cite_key_version"] == CITE_KEY_VERSION
        assert f"{{{row['cite_key']}," in sealed.bibliography


def test_registry_strict_loader_rejects_duplicate_fields() -> None:
    sealed = seal_citation_collection(
        [_candidate(paper_id="s2-a", doi="10.1000/a")]
    )
    registry_text = json.dumps(sealed.registry)
    duplicate = registry_text.replace(
        '"schema_version": 1',
        '"schema_version": 1, "schema_version": 1',
        1,
    )
    with pytest.raises(CitationIdentityError, match="duplicate JSON key"):
        parse_cite_key_registry(duplicate)


def test_registry_artifact_mutation_is_rejected() -> None:
    sealed = seal_citation_collection(
        [_candidate(paper_id="s2-a", doi="10.1000/a")]
    )
    with pytest.raises(CitationIdentityError, match="candidates_sha256 mismatch"):
        validate_registry_artifacts(
            sealed.registry,
            sealed.candidates_jsonl + " ",
            sealed.bibliography,
        )


def test_bibtex_metadata_cannot_inject_entry_or_unbalanced_brace() -> None:
    row = _candidate(
        paper_id="s2-injection",
        title="Safe } Title\n@article{evil,",
        doi="10.1000/safe",
    )
    sealed = seal_citation_collection([row])
    assert sealed.bibliography.startswith("@inproceedings{smith2024safe,")
    assert "\n@article{" not in sealed.bibliography
    assert "title = {Safe  Title @articleevil,}," in sealed.bibliography


def test_no_author_uses_anon_cite_key_prefix() -> None:
    row = _candidate(paper_id="s2-anon", doi="10.1000/anon")
    row["authors"] = []
    sealed = seal_citation_collection([row])
    assert sealed.candidates[0]["cite_key"].startswith("anon2024runtime")


def test_placeholder_candidate_cannot_enter_registry() -> None:
    row = _candidate(paper_id="placeholder")
    row["is_placeholder"] = True
    with pytest.raises(CitationIdentityError, match="placeholder"):
        seal_citation_collection([row])


def test_stage4_writes_registry_candidates_and_bib_from_same_keys(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_dir = tmp_path / "run"
    stage_dir = run_dir / "stage-04"
    (run_dir / "stage-03").mkdir(parents=True)
    stage_dir.mkdir(parents=True)
    (run_dir / "stage-03" / "queries.json").write_text(
        json.dumps({"queries": ["hardware counter detection"], "year_min": 2020}),
        encoding="utf-8",
    )

    papers = [
        Paper(
            paper_id="s2-a",
            title="<scp>EC</scp> Runtime Detection",
            authors=(Author(name="Jane Smith"),),
            year=2024,
            abstract="Runtime detector evidence.",
            venue="Security Conference",
            doi="10.1000/a",
            source="semantic_scholar",
        ),
        Paper(
            paper_id="s2-b",
            title="<scp>EC</scp> Runtime Detection",
            authors=(Author(name="Jane Smith"),),
            year=2024,
            abstract="A second detector.",
            venue="Security Conference",
            doi="10.1000/b",
            source="semantic_scholar",
        ),
    ]
    monkeypatch.setattr(
        "researchclaw.literature.search.search_papers_multi_query",
        lambda *_args, **_kwargs: papers,
    )
    monkeypatch.setattr("researchclaw.data.load_seminal_papers", lambda _topic: [])

    config = SimpleNamespace(
        research=SimpleNamespace(
            topic="hardware counter detection",
            daily_paper_count=20,
        ),
        llm=SimpleNamespace(s2_api_key=""),
        web_search=SimpleNamespace(enabled=False),
    )
    result = _execute_literature_collect(
        stage_dir,
        run_dir,
        config,  # type: ignore[arg-type]
        AdapterBundle(),
        llm=None,
    )

    assert result.status is StageStatus.DONE
    assert {
        "candidates.jsonl",
        "references.bib",
        "cite_key_registry.json",
    }.issubset(result.artifacts)
    rows = [
        json.loads(line)
        for line in (stage_dir / "candidates.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    bib = (stage_dir / "references.bib").read_text(encoding="utf-8")
    registry = json.loads(
        (stage_dir / "cite_key_registry.json").read_text(encoding="utf-8")
    )
    assert len(rows) == 2
    assert len({row["cite_key"] for row in rows}) == 2
    assert {row["source_identity"] for row in rows} == {
        entry["source_identity"] for entry in registry["entries"]
    }
    assert all(f"{{{row['cite_key']}," in bib for row in rows)
    assert registry["candidates_sha256"] == hashlib.sha256(
        (stage_dir / "candidates.jsonl").read_bytes()
    ).hexdigest()
    assert registry["references_sha256"] == hashlib.sha256(
        (stage_dir / "references.bib").read_bytes()
    ).hexdigest()


def test_stage4_placeholder_fallback_fails_before_registry_seal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_dir = tmp_path / "run"
    stage_dir = run_dir / "stage-04"
    (run_dir / "stage-03").mkdir(parents=True)
    stage_dir.mkdir(parents=True)
    (run_dir / "stage-03" / "queries.json").write_text(
        json.dumps({"queries": ["unavailable topic"], "year_min": 2020}),
        encoding="utf-8",
    )
    (stage_dir / "candidates.jsonl").write_text("old candidates\n", encoding="utf-8")
    (stage_dir / "references.bib").write_text("old bib\n", encoding="utf-8")
    (stage_dir / "cite_key_registry.json").write_text(
        '{"old": true}\n', encoding="utf-8"
    )
    monkeypatch.setattr(
        "researchclaw.literature.search.search_papers_multi_query",
        lambda *_args, **_kwargs: [],
    )
    monkeypatch.setattr("researchclaw.data.load_seminal_papers", lambda _topic: [])
    config = SimpleNamespace(
        research=SimpleNamespace(topic="unavailable topic", daily_paper_count=20),
        llm=SimpleNamespace(s2_api_key=""),
        web_search=SimpleNamespace(enabled=False),
    )

    result = _execute_literature_collect(
        stage_dir,
        run_dir,
        config,  # type: ignore[arg-type]
        AdapterBundle(),
        llm=None,
    )

    assert result.status is StageStatus.FAILED
    assert "placeholder candidates" in (result.error or "")
    assert not (stage_dir / "cite_key_registry.json").exists()
    assert not (stage_dir / "references.bib").exists()
    candidates_text = (stage_dir / "candidates.jsonl").read_text(encoding="utf-8")
    assert "old candidates" not in candidates_text
    assert '"is_placeholder": true' in candidates_text
