"""PDF full-text extraction powered by PyMuPDF (fitz).

PyMuPDF is installed as a dependency and provides fast, high-quality
PDF text extraction with metadata, section detection, and table support.

Usage::

    extractor = PDFExtractor()
    result = extractor.extract("/path/to/paper.pdf")
    print(result.text[:1000])
"""

from __future__ import annotations

import logging
import re
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.request import Request, urlopen

try:
    import fitz  # PyMuPDF
    HAS_FITZ = True
except ImportError:
    fitz = None  # type: ignore[assignment]
    HAS_FITZ = False

logger = logging.getLogger(__name__)


@dataclass
class PDFContent:
    """Extracted content from a PDF file."""

    path: str
    text: str = ""
    title: str = ""
    authors: list[str] = field(default_factory=list)
    abstract: str = ""
    sections: list[dict[str, str]] = field(default_factory=list)
    page_count: int = 0
    success: bool = False
    error: str = ""
    backend: str = "pymupdf"
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def has_content(self) -> bool:
        return bool(self.text and len(self.text.strip()) > 100)


class PDFExtractor:
    """PDF text extraction using PyMuPDF.

    Parameters
    ----------
    max_pages:
        Maximum pages to extract (0 = all).
    extract_sections:
        Whether to attempt section boundary detection.
    """

    def __init__(
        self,
        *,
        max_pages: int = 0,
        extract_sections: bool = True,
    ) -> None:
        self.max_pages = max_pages
        self.extract_sections = extract_sections

    @property
    def backend(self) -> str:
        return "pymupdf"

    def extract(self, path: str | Path) -> PDFContent:
        """Extract text from a local PDF file using PyMuPDF."""
        if not HAS_FITZ:
            return PDFContent(
                path=str(path),
                error="PyMuPDF not installed. Install: pip install 'researchclaw[pdf]'",
            )
        path = Path(path)
        try:
            _exists = path.exists()
        except (PermissionError, OSError):
            _exists = False
        if not _exists:
            return PDFContent(path=str(path), error=f"File not found: {path}")

        try:
            with fitz.open(str(path)) as doc:
                pages_to_read = doc.page_count
                if self.max_pages > 0:
                    pages_to_read = min(pages_to_read, self.max_pages)

                all_text = []
                for i in range(pages_to_read):
                    page = doc[i]
                    all_text.append(page.get_text())

                full_text = "\n".join(all_text)

                meta = doc.metadata or {}
                title = meta.get("title", "")
                author = meta.get("author", "")
                authors = [a.strip() for a in author.split(",")] if author else []

                abstract = self._extract_abstract(full_text)
                sections = self._detect_sections(full_text) if self.extract_sections else []

                page_count = doc.page_count

            return PDFContent(
                path=str(path),
                text=full_text,
                title=title,
                authors=authors,
                abstract=abstract,
                sections=sections,
                page_count=page_count,
                success=True,
                metadata=meta,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("PDF extraction failed for %s: %s", path, exc)
            return PDFContent(path=str(path), error=str(exc))

    def extract_from_url(self, url: str) -> PDFContent:
        """Download a PDF from URL and extract text."""
        # Validate URL scheme to prevent SSRF (file://, internal IPs, etc.)
        from urllib.parse import urlparse
        parsed = urlparse(url)
        if parsed.scheme not in ("http", "https"):
            return PDFContent(path=url, error=f"Unsupported URL scheme: {parsed.scheme}")
        # Block private/internal IPs
        hostname = parsed.hostname or ""
        if hostname in ("localhost", "127.0.0.1", "0.0.0.0") or hostname.startswith("169.254."):
            return PDFContent(path=url, error=f"Blocked internal URL: {hostname}")

        tmp_path = None
        try:
            req = Request(url, headers={
                "User-Agent": "ResearchClaw/0.5 (Academic Research Bot)"
            })
            resp = urlopen(req, timeout=30)  # noqa: S310
            data = resp.read()

            with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
                f.write(data)
                tmp_path = f.name

            result = self.extract(tmp_path)
            result.path = url
            return result
        except Exception as exc:  # noqa: BLE001
            logger.warning("PDF download failed for %s: %s", url, exc)
            return PDFContent(path=url, error=str(exc))
        finally:
            if tmp_path:
                Path(tmp_path).unlink(missing_ok=True)

    # ------------------------------------------------------------------
    # Section detection
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_abstract(text: str) -> str:
        """Extract abstract from paper text."""
        match = re.search(
            r"(?:^|\n)\s*Abstract\s*\n(.*?)(?=\n\s*(?:\d+\.?\s+)?(?:Introduction|1\s))",
            text, re.DOTALL | re.IGNORECASE,
        )
        if match:
            return match.group(1).strip()
        match = re.search(
            r"(?:^|\n)\s*Abstract[:\s]*\n?(.*?)(?:\n\n|\n\s*\n)",
            text, re.DOTALL | re.IGNORECASE,
        )
        if match:
            return match.group(1).strip()
        return ""

    @staticmethod
    def _detect_sections(text: str) -> list[dict[str, str]]:
        """Detect section boundaries in paper text."""
        sections: list[dict[str, str]] = []
        pattern = re.compile(r"(?:^|\n)\s*(\d+\.?\s+[A-Z][^\n]{2,50})\s*\n", re.MULTILINE)
        matches = list(pattern.finditer(text))
        for i, match in enumerate(matches):
            heading = match.group(1).strip()
            start = match.end()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            body = text[start:end].strip()
            sections.append({"heading": heading, "text": body[:5000]})
        return sections
