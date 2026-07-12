# pyright: reportPrivateUsage=false, reportUnknownParameterType=false
from __future__ import annotations

import json
import textwrap
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from researchclaw.literature.verify import (
    CitationResult,
    VerificationReport,
    VerifyStatus,
    filter_verified_bibtex,
    parse_bibtex_entries,
    title_similarity,
    verify_by_arxiv_id,
    verify_by_doi,
    verify_by_title_search,
    verify_citations,
)
from researchclaw.literature.models import Author, Paper


SAMPLE_BIB = textwrap.dedent("""\
    @article{vaswani2017attention,
      title = {Attention Is All You Need},
      author = {Ashish Vaswani and Noam Shazeer},
      year = {2017},
      eprint = {1706.03762},
      archiveprefix = {arXiv},
    }

    @inproceedings{devlin2019bert,
      title = {BERT: Pre-training of Deep Bidirectional Transformers},
      author = {Jacob Devlin},
      year = {2019},
      doi = {10.18653/v1/N19-1423},
      booktitle = {NAACL},
    }

    @article{fakepaper2025hallucinated,
      title = {A Completely Made Up Paper That Does Not Exist},
      author = {Imaginary Author},
      year = {2025},
    }
""")

SAMPLE_ARXIV_VERIFY_RESPONSE = textwrap.dedent("""\
    <?xml version="1.0" encoding="UTF-8"?>
    <feed xmlns="http://www.w3.org/2005/Atom">
      <entry>
        <id>http://arxiv.org/abs/1706.03762v5</id>
        <title>Attention Is All You Need</title>
        <summary>The dominant sequence transduction models...</summary>
        <author><name>Ashish Vaswani</name></author>
      </entry>
    </feed>
""")

SAMPLE_ARXIV_EMPTY_RESPONSE = textwrap.dedent("""\
    <?xml version="1.0" encoding="UTF-8"?>
    <feed xmlns="http://www.w3.org/2005/Atom">
      <entry>
        <id>http://arxiv.org/api/errors#incorrect_id_format_for_9999.99999</id>
        <title>Error</title>
        <summary>incorrect id format for 9999.99999</summary>
      </entry>
    </feed>
""")

SAMPLE_CROSSREF_RESPONSE = {
    "status": "ok",
    "message": {
        "DOI": "10.18653/v1/N19-1423",
        "title": [
            "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding"
        ],
        "author": [{"given": "Jacob", "family": "Devlin"}],
    },
}


class TestParseBibtexEntries:
    def test_parses_three_entries(self) -> None:
        entries = parse_bibtex_entries(SAMPLE_BIB)
        assert len(entries) == 3

    def test_entry_keys(self) -> None:
        entries = parse_bibtex_entries(SAMPLE_BIB)
        keys = [e["key"] for e in entries]
        assert "vaswani2017attention" in keys
        assert "devlin2019bert" in keys
        assert "fakepaper2025hallucinated" in keys

    def test_entry_fields(self) -> None:
        entries = parse_bibtex_entries(SAMPLE_BIB)
        vaswani = next(e for e in entries if e["key"] == "vaswani2017attention")
        assert vaswani["title"] == "Attention Is All You Need"
        assert vaswani["eprint"] == "1706.03762"
        assert vaswani["type"] == "article"

    def test_entry_type(self) -> None:
        entries = parse_bibtex_entries(SAMPLE_BIB)
        devlin = next(e for e in entries if e["key"] == "devlin2019bert")
        assert devlin["type"] == "inproceedings"
        assert devlin["doi"] == "10.18653/v1/N19-1423"

    def test_empty_bib(self) -> None:
        assert parse_bibtex_entries("") == []

    def test_malformed_bib(self) -> None:
        assert parse_bibtex_entries("not bibtex at all") == []


class TestTitleSimilarity:
    def test_identical(self) -> None:
        assert (
            title_similarity("Attention Is All You Need", "Attention Is All You Need")
            == 1.0
        )

    def test_case_insensitive(self) -> None:
        assert (
            title_similarity("attention is all you need", "ATTENTION IS ALL YOU NEED")
            == 1.0
        )

    def test_high_similarity(self) -> None:
        sim = title_similarity(
            "Attention Is All You Need",
            "Attention Is All You Need: A Transformer Architecture",
        )
        assert sim >= 0.5

    def test_low_similarity(self) -> None:
        sim = title_similarity(
            "Attention Is All You Need",
            "Protein Folding with AlphaFold",
        )
        assert sim < 0.3

    def test_empty_strings(self) -> None:
        assert title_similarity("", "") == 0.0
        assert title_similarity("something", "") == 0.0


class TestVerifyByArxivId:
    def test_verified_match(self) -> None:
        mock_resp = MagicMock()
        mock_resp.read.return_value = SAMPLE_ARXIV_VERIFY_RESPONSE.encode("utf-8")
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)

        with patch("urllib.request.urlopen", return_value=mock_resp):
            result = verify_by_arxiv_id("1706.03762", "Attention Is All You Need")

        assert result is not None
        assert result.status == VerifyStatus.VERIFIED
        assert result.method == "arxiv_id"
        assert result.confidence >= 0.80

    def test_hallucinated_error_response(self) -> None:
        mock_resp = MagicMock()
        mock_resp.read.return_value = SAMPLE_ARXIV_EMPTY_RESPONSE.encode("utf-8")
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)

        with patch("urllib.request.urlopen", return_value=mock_resp):
            result = verify_by_arxiv_id("9999.99999", "Fake Paper")

        assert result is not None
        assert result.status == VerifyStatus.HALLUCINATED

    def test_network_failure_returns_none(self) -> None:
        with patch("urllib.request.urlopen", side_effect=OSError("connection refused")):
            result = verify_by_arxiv_id("1706.03762", "Attention Is All You Need")
        assert result is None

    def test_title_mismatch_suspicious(self) -> None:
        different_title_response = textwrap.dedent("""\
            <?xml version="1.0" encoding="UTF-8"?>
            <feed xmlns="http://www.w3.org/2005/Atom">
              <entry>
                <id>http://arxiv.org/abs/1706.03762v5</id>
                <title>A Completely Different Paper Title About Quantum Computing</title>
                <summary>Summary</summary>
              </entry>
            </feed>
        """)
        mock_resp = MagicMock()
        mock_resp.read.return_value = different_title_response.encode("utf-8")
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)

        with patch("urllib.request.urlopen", return_value=mock_resp):
            result = verify_by_arxiv_id("1706.03762", "Attention Is All You Need")

        assert result is not None
        assert result.status == VerifyStatus.SUSPICIOUS


class TestVerifyByDoi:
    def test_verified_crossref(self) -> None:
        mock_resp = MagicMock()
        mock_resp.read.return_value = json.dumps(SAMPLE_CROSSREF_RESPONSE).encode(
            "utf-8"
        )
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)

        with patch("urllib.request.urlopen", return_value=mock_resp):
            result = verify_by_doi(
                "10.18653/v1/N19-1423",
                "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding",
            )

        assert result is not None
        assert result.status == VerifyStatus.VERIFIED
        assert result.method == "doi"

    def test_doi_404_hallucinated(self) -> None:
        import urllib.error

        with patch(
            "urllib.request.urlopen",
            side_effect=urllib.error.HTTPError(
                "https://api.crossref.org/works/10.fake/doi",
                404,
                "Not Found",
                {},
                None,  # type: ignore[arg-type]
            ),
        ):
            result = verify_by_doi("10.fake/doi", "Nonexistent Paper")

        assert result is not None
        assert result.status == VerifyStatus.HALLUCINATED

    def test_network_error_returns_none(self) -> None:
        with patch("urllib.request.urlopen", side_effect=OSError("timeout")):
            result = verify_by_doi("10.1234/test", "Test Paper")
        assert result is None

    def test_doi_exists_no_title(self) -> None:
        no_title_resp = {"status": "ok", "message": {"DOI": "10.1234/test"}}
        mock_resp = MagicMock()
        mock_resp.read.return_value = json.dumps(no_title_resp).encode("utf-8")
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)

        with patch("urllib.request.urlopen", return_value=mock_resp):
            result = verify_by_doi("10.1234/test", "Some Paper")

        assert result is not None
        assert result.status == VerifyStatus.VERIFIED
        assert "no title comparison" in result.details.lower()


class TestVerifyByTitleSearch:
    def test_verified_via_search(self) -> None:
        mock_paper = Paper(
            paper_id="s2-abc",
            title="Attention Is All You Need",
            authors=(Author(name="Vaswani"),),
            year=2017,
            source="semantic_scholar",
        )
        with patch(
            "researchclaw.literature.search.search_papers",
            return_value=[mock_paper],
        ):
            result = verify_by_title_search("Attention Is All You Need")

        assert result is not None
        assert result.status == VerifyStatus.VERIFIED
        assert result.matched_paper is not None

    def test_no_results_hallucinated(self) -> None:
        with patch("researchclaw.literature.search.search_papers", return_value=[]):
            result = verify_by_title_search("A Completely Made Up Paper")

        assert result is not None
        assert result.status == VerifyStatus.HALLUCINATED

    def test_weak_match_hallucinated(self) -> None:
        mock_paper = Paper(
            paper_id="s2-xyz",
            title="Quantum Computing for Protein Folding",
            year=2023,
            source="arxiv",
        )
        with patch(
            "researchclaw.literature.search.search_papers",
            return_value=[mock_paper],
        ):
            result = verify_by_title_search("A Completely Made Up Paper About Nothing")

        assert result is not None
        assert result.status == VerifyStatus.HALLUCINATED

    def test_partial_match_suspicious(self) -> None:
        mock_paper = Paper(
            paper_id="s2-partial",
            title="Attention Mechanisms in Neural Networks",
            year=2019,
            source="semantic_scholar",
        )
        with patch(
            "researchclaw.literature.search.search_papers",
            return_value=[mock_paper],
        ):
            result = verify_by_title_search("Attention Neural Networks Survey Overview")

        assert result is not None
        assert result.status in (VerifyStatus.SUSPICIOUS, VerifyStatus.HALLUCINATED)

    def test_network_failure_returns_none(self) -> None:
        with patch(
            "researchclaw.literature.search.search_papers",
            side_effect=OSError("network down"),
        ):
            result = verify_by_title_search("Any Paper")
        assert result is None


class TestVerifyCitations:
    def test_full_pipeline_mocked(self) -> None:
        arxiv_resp = MagicMock()
        arxiv_resp.read.return_value = SAMPLE_ARXIV_VERIFY_RESPONSE.encode("utf-8")
        arxiv_resp.__enter__ = lambda s: s
        arxiv_resp.__exit__ = MagicMock(return_value=False)

        crossref_resp = MagicMock()
        crossref_resp.read.return_value = json.dumps(SAMPLE_CROSSREF_RESPONSE).encode(
            "utf-8"
        )
        crossref_resp.__enter__ = lambda s: s
        crossref_resp.__exit__ = MagicMock(return_value=False)

        call_count = {"n": 0}

        def mock_urlopen(req: Any, **kwargs: Any) -> MagicMock:
            call_count["n"] += 1
            url = req.full_url if hasattr(req, "full_url") else str(req)
            if "arxiv.org" in url:
                return arxiv_resp
            if "crossref.org" in url:
                return crossref_resp
            raise OSError("unexpected URL")

        with (
            patch("researchclaw.literature.verify.time.sleep"),
            patch("urllib.request.urlopen", side_effect=mock_urlopen),
            patch("researchclaw.literature.search.search_papers", return_value=[]),
        ):
            report = verify_citations(SAMPLE_BIB, inter_verify_delay=0)

        assert report.total == 3
        assert report.verified >= 1
        assert report.hallucinated >= 1

        report_dict = report.to_dict()
        assert "summary" in report_dict
        assert "results" in report_dict
        assert report_dict["summary"]["total"] == 3

    def test_empty_bib(self) -> None:
        report = verify_citations("")
        assert report.total == 0
        assert report.integrity_score == 1.0

    def test_no_title_entry_skipped(self) -> None:
        bib = textwrap.dedent("""\
            @article{noauthor2025,
              author = {Some Author},
              year = {2025},
            }
        """)
        report = verify_citations(bib)
        assert report.total == 1
        assert report.skipped == 1


class TestVerificationReport:
    def test_integrity_score(self) -> None:
        report = VerificationReport(
            total=10, verified=7, suspicious=1, hallucinated=2, skipped=0
        )
        assert report.integrity_score == 0.7

    def test_integrity_score_with_skips(self) -> None:
        report = VerificationReport(
            total=10, verified=6, suspicious=0, hallucinated=2, skipped=2
        )
        assert report.integrity_score == 0.75

    def test_integrity_score_all_skipped(self) -> None:
        report = VerificationReport(
            total=3, verified=0, suspicious=0, hallucinated=0, skipped=3
        )
        assert report.integrity_score == 1.0

    def test_to_dict(self) -> None:
        report = VerificationReport(total=2, verified=1, hallucinated=1)
        d = report.to_dict()
        assert d["summary"]["total"] == 2
        assert d["summary"]["integrity_score"] == 0.5


class TestFilterVerifiedBibtex:
    def _make_report(self) -> VerificationReport:
        return VerificationReport(
            total=3,
            verified=1,
            suspicious=1,
            hallucinated=1,
            results=[
                CitationResult(
                    cite_key="vaswani2017attention",
                    title="Attention Is All You Need",
                    status=VerifyStatus.VERIFIED,
                    confidence=1.0,
                    method="arxiv_id",
                ),
                CitationResult(
                    cite_key="devlin2019bert",
                    title="BERT",
                    status=VerifyStatus.SUSPICIOUS,
                    confidence=0.6,
                    method="doi",
                ),
                CitationResult(
                    cite_key="fakepaper2025hallucinated",
                    title="Fake Paper",
                    status=VerifyStatus.HALLUCINATED,
                    confidence=0.9,
                    method="title_search",
                ),
            ],
        )

    def test_includes_verified_and_suspicious(self) -> None:
        report = self._make_report()
        filtered = filter_verified_bibtex(SAMPLE_BIB, report, include_suspicious=True)
        assert "vaswani2017attention" in filtered
        assert "devlin2019bert" in filtered
        assert "fakepaper2025hallucinated" not in filtered

    def test_excludes_suspicious(self) -> None:
        report = self._make_report()
        filtered = filter_verified_bibtex(SAMPLE_BIB, report, include_suspicious=False)
        assert "vaswani2017attention" in filtered
        assert "devlin2019bert" not in filtered
        assert "fakepaper2025hallucinated" not in filtered

    def test_empty_bib(self) -> None:
        report = VerificationReport()
        assert filter_verified_bibtex("", report) == ""


class TestCitationResultSerialization:
    def test_to_dict_basic(self) -> None:
        result = CitationResult(
            cite_key="smith2024test",
            title="Test Paper",
            status=VerifyStatus.VERIFIED,
            confidence=0.95,
            method="arxiv_id",
            details="Confirmed",
        )
        d = result.to_dict()
        assert d["cite_key"] == "smith2024test"
        assert d["status"] == "verified"
        assert d["confidence"] == 0.95

    def test_to_dict_with_matched_paper(self) -> None:
        paper = Paper(
            paper_id="s2-abc",
            title="Found Paper",
            authors=(Author(name="Smith"),),
            year=2024,
            source="semantic_scholar",
        )
        result = CitationResult(
            cite_key="smith2024test",
            title="Test",
            status=VerifyStatus.VERIFIED,
            confidence=0.9,
            method="title_search",
            matched_paper=paper,
        )
        d = result.to_dict()
        assert "matched_paper" in d
        assert d["matched_paper"]["title"] == "Found Paper"


class TestStage23Integration:
    def test_stage_exists_in_enum(self) -> None:
        from researchclaw.pipeline.stages import Stage

        assert hasattr(Stage, "CITATION_VERIFY")
        assert Stage.CITATION_VERIFY == 23

    def test_stage_in_sequence(self) -> None:
        from researchclaw.pipeline.stages import Stage, STAGE_SEQUENCE, NEXT_STAGE

        assert Stage.CITATION_VERIFY in STAGE_SEQUENCE
        assert NEXT_STAGE[Stage.EXPORT_PUBLISH] == Stage.CITATION_VERIFY
        # v2: citation verify feeds the release-audit phase
        assert NEXT_STAGE[Stage.CITATION_VERIFY] == Stage.TRUTH_AUDIT
        assert NEXT_STAGE[Stage.DEAI_AUDIT] is None

    def test_contract_exists(self) -> None:
        from researchclaw.pipeline.contracts import CONTRACTS
        from researchclaw.pipeline.stages import Stage

        assert Stage.CITATION_VERIFY in CONTRACTS
        contract = CONTRACTS[Stage.CITATION_VERIFY]
        assert "verification_report.json" in contract.output_files
        assert "references_verified.bib" in contract.output_files

    def test_executor_registered(self) -> None:
        from researchclaw.pipeline.executor import _STAGE_EXECUTORS
        from researchclaw.pipeline.stages import Stage

        assert Stage.CITATION_VERIFY in _STAGE_EXECUTORS

    def test_phase_map(self) -> None:
        from researchclaw.pipeline.stages import PHASE_MAP, Stage

        finalization_stages = PHASE_MAP["H: Finalization"]
        assert Stage.CITATION_VERIFY in finalization_stages

    def test_total_stages_is_25(self) -> None:
        from researchclaw.pipeline.stages import STAGE_SEQUENCE

        assert len(STAGE_SEQUENCE) == 25
