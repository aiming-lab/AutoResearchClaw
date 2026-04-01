"""Tests for researchclaw.web.pdf_extractor — PDFExtractor."""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from researchclaw.web.pdf_extractor import PDFContent, PDFExtractor


# ---------------------------------------------------------------------------
# PDFContent dataclass
# ---------------------------------------------------------------------------


class TestPDFContent:
    def test_has_content_true(self):
        c = PDFContent(path="test.pdf", text="x" * 200, success=True)
        assert c.has_content

    def test_has_content_false_empty(self):
        c = PDFContent(path="test.pdf", text="", success=True)
        assert not c.has_content

    def test_has_content_false_short(self):
        c = PDFContent(path="test.pdf", text="short", success=True)
        assert not c.has_content


# ---------------------------------------------------------------------------
# PDFExtractor
# ---------------------------------------------------------------------------


class TestPDFExtractor:
    def test_backend_detection(self):
        extractor = PDFExtractor()
        assert extractor.backend == "pymupdf"  # PyMuPDF is now installed

    def test_extract_nonexistent_file(self, tmp_path):
        extractor = PDFExtractor()
        result = extractor.extract(tmp_path / "does_not_exist.pdf")
        assert not result.success or "not found" in result.error.lower() or result.error

    def test_extract_abstract_pattern(self):
        text = """
Some header text

Abstract
This paper presents a novel approach to knowledge distillation
that achieves state-of-the-art results on ImageNet.

1 Introduction
We begin by motivating our approach...
"""
        abstract = PDFExtractor._extract_abstract(text)
        assert "knowledge distillation" in abstract

    def test_extract_abstract_no_match(self):
        text = "No abstract section here, just random text."
        abstract = PDFExtractor._extract_abstract(text)
        assert abstract == ""

    def test_detect_sections(self):
        text = """
1. Introduction
This is the introduction section with some content.

2. Related Work
This covers prior work in the field.

3. Method
Our proposed approach works as follows.

4. Experiments
We evaluate on several benchmarks.
"""
        sections = PDFExtractor._detect_sections(text)
        assert len(sections) >= 3
        headings = [s["heading"] for s in sections]
        assert any("Introduction" in h for h in headings)
        assert any("Related" in h or "Method" in h for h in headings)

    def test_detect_sections_empty(self):
        text = "No numbered sections here at all."
        sections = PDFExtractor._detect_sections(text)
        assert sections == []

    @patch("researchclaw.web.pdf_extractor.urlopen")
    def test_extract_from_url_failure(self, mock_urlopen):
        mock_urlopen.side_effect = Exception("404 Not Found")
        extractor = PDFExtractor()
        result = extractor.extract_from_url("https://example.com/paper.pdf")
        assert not result.success or result.error
