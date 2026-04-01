"""Web search, crawling, and content extraction layer.

Provides unified access to:
- **Crawl4AI**: Web page → Markdown extraction
- **Tavily**: AI-native web search API
- **scholarly**: Google Scholar search
- **PDF extraction**: Full-text from PDF files

Public API
----------
- ``WebSearchAgent`` — orchestrates all web capabilities
- ``WebCrawler`` — Crawl4AI wrapper
- ``WebSearchClient`` — Tavily search wrapper
- ``GoogleScholarClient`` — scholarly wrapper
- ``PDFExtractor`` — PDF text extraction
- ``check_url_ssrf`` — SSRF validation for URLs
"""

from researchclaw.web._ssrf import check_url_ssrf
from researchclaw.web.crawler import WebCrawler
from researchclaw.web.search import WebSearchClient
from researchclaw.web.scholar import GoogleScholarClient
from researchclaw.web.pdf_extractor import PDFExtractor
from researchclaw.web.agent import WebSearchAgent

__all__ = [
    "check_url_ssrf",
    "WebCrawler",
    "WebSearchClient",
    "GoogleScholarClient",
    "PDFExtractor",
    "WebSearchAgent",
]
