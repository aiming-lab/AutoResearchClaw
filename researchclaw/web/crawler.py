"""Web page → Markdown extraction powered by Crawl4AI.

Crawl4AI is the primary extraction engine (installed as a dependency).
A lightweight urllib fallback exists for environments where Crawl4AI's
browser dependency is not set up.

Usage::

    crawler = WebCrawler()
    result = await crawler.crawl("https://arxiv.org/abs/2301.00001")
    print(result.markdown)
"""

from __future__ import annotations

import asyncio
import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any
from urllib.request import Request, urlopen

from researchclaw.web._ssrf import check_url_ssrf

logger = logging.getLogger(__name__)


@dataclass
class CrawlResult:
    """Result of crawling a single URL."""

    url: str
    markdown: str = ""
    title: str = ""
    success: bool = False
    error: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    elapsed_seconds: float = 0.0

    @property
    def has_content(self) -> bool:
        return bool(self.markdown and len(self.markdown.strip()) > 50)


class WebCrawler:
    """Web page → Markdown crawler powered by Crawl4AI.

    Parameters
    ----------
    timeout:
        Request timeout in seconds.
    max_content_length:
        Maximum content length in characters (truncate beyond this).
    """

    def __init__(
        self,
        *,
        timeout: int = 30,
        max_content_length: int = 50_000,
        user_agent: str = "ResearchClaw/0.5 (Academic Research Bot)",
    ) -> None:
        self.timeout = timeout
        self.max_content_length = max_content_length
        self.user_agent = user_agent

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def crawl(self, url: str) -> CrawlResult:
        """Crawl a URL and return Markdown content (async)."""
        err = check_url_ssrf(url)
        if err:
            return CrawlResult(url=url, success=False, error=err, elapsed_seconds=0.0)
        t0 = time.monotonic()
        try:
            return await self._crawl_with_crawl4ai(url, t0)
        except Exception as exc:  # noqa: BLE001
            logger.debug("Crawl4AI failed for %s (%s), trying urllib fallback", url, exc)
            try:
                return self._crawl_with_urllib(url, t0)
            except Exception as exc2:  # noqa: BLE001
                elapsed = time.monotonic() - t0
                logger.warning("All crawl backends failed for %s: %s", url, exc2)
                return CrawlResult(url=url, success=False, error=str(exc2), elapsed_seconds=elapsed)

    def crawl_sync(self, url: str) -> CrawlResult:
        """Synchronous crawl — tries Crawl4AI via asyncio.run, falls back to urllib."""
        err = check_url_ssrf(url)
        if err:
            return CrawlResult(url=url, success=False, error=err, elapsed_seconds=0.0)
        t0 = time.monotonic()
        try:
            return asyncio.run(self._crawl_with_crawl4ai(url, t0))
        except Exception:  # noqa: BLE001
            try:
                return self._crawl_with_urllib(url, t0)
            except Exception as exc:  # noqa: BLE001
                elapsed = time.monotonic() - t0
                return CrawlResult(url=url, success=False, error=str(exc), elapsed_seconds=elapsed)

    async def crawl_many(self, urls: list[str]) -> list[CrawlResult]:
        """Crawl multiple URLs using Crawl4AI's async engine."""
        results = []
        try:
            from crawl4ai import AsyncWebCrawler, CrawlerRunConfig, BrowserConfig

            browser_config = BrowserConfig(headless=True)
            run_config = CrawlerRunConfig(
                word_count_threshold=10,
                excluded_tags=["nav", "footer", "header", "sidebar"],
                remove_overlay_elements=True,
            )

            async with AsyncWebCrawler(config=browser_config) as crawler:
                for url in urls:
                    err = check_url_ssrf(url)
                    if err:
                        results.append(CrawlResult(url=url, success=False, error=err, elapsed_seconds=0.0))
                        continue
                    t0 = time.monotonic()
                    try:
                        raw = await crawler.arun(url=url, config=run_config)
                        elapsed = time.monotonic() - t0
                        if raw.success:
                            md = self._extract_markdown(raw)
                            results.append(CrawlResult(
                                url=url, markdown=md,
                                title=getattr(raw, "title", "") or "",
                                success=True, elapsed_seconds=elapsed,
                                metadata=raw.metadata if hasattr(raw, "metadata") and raw.metadata else {},
                            ))
                        else:
                            results.append(CrawlResult(
                                url=url, success=False,
                                error=getattr(raw, "error_message", "crawl failed"),
                                elapsed_seconds=elapsed,
                            ))
                    except Exception as exc:  # noqa: BLE001
                        elapsed = time.monotonic() - t0
                        results.append(CrawlResult(url=url, success=False, error=str(exc), elapsed_seconds=elapsed))
        except ImportError:
            # Crawl4AI browser not set up — use urllib for each
            for url in urls:
                err = check_url_ssrf(url)
                if err:
                    results.append(CrawlResult(url=url, success=False, error=err, elapsed_seconds=0.0))
                    continue
                t0 = time.monotonic()
                try:
                    results.append(self._crawl_with_urllib(url, t0))
                except Exception as exc:  # noqa: BLE001
                    elapsed = time.monotonic() - t0
                    results.append(CrawlResult(url=url, success=False, error=str(exc), elapsed_seconds=elapsed))
        return results

    # ------------------------------------------------------------------
    # Crawl4AI backend (primary)
    # ------------------------------------------------------------------

    async def _crawl_with_crawl4ai(self, url: str, t0: float) -> CrawlResult:
        """Use Crawl4AI for high-quality extraction."""
        from crawl4ai import AsyncWebCrawler, CrawlerRunConfig, BrowserConfig

        browser_config = BrowserConfig(headless=True)
        run_config = CrawlerRunConfig(
            word_count_threshold=10,
            excluded_tags=["nav", "footer", "header", "sidebar"],
            remove_overlay_elements=True,
        )

        async with AsyncWebCrawler(config=browser_config) as crawler:
            raw = await crawler.arun(url=url, config=run_config)

        elapsed = time.monotonic() - t0
        if raw.success:
            md = self._extract_markdown(raw)
            return CrawlResult(
                url=url, markdown=md,
                title=getattr(raw, "title", "") or "",
                success=True, elapsed_seconds=elapsed,
                metadata=raw.metadata if hasattr(raw, "metadata") and raw.metadata else {},
            )
        return CrawlResult(
            url=url, success=False,
            error=getattr(raw, "error_message", "Unknown crawl4ai error"),
            elapsed_seconds=elapsed,
        )

    def _extract_markdown(self, raw: Any) -> str:
        """Extract markdown from a Crawl4AI result object."""
        # Crawl4AI v0.8+ uses markdown_v2.raw_markdown
        md = ""
        if hasattr(raw, "markdown_v2") and raw.markdown_v2:
            md = getattr(raw.markdown_v2, "raw_markdown", "") or ""
        if not md and hasattr(raw, "markdown"):
            md = raw.markdown or ""
        if len(md) > self.max_content_length:
            md = md[: self.max_content_length] + "\n\n[... truncated]"
        return md

    # ------------------------------------------------------------------
    # urllib fallback (lightweight, no browser needed)
    # ------------------------------------------------------------------

    def _crawl_with_urllib(self, url: str, t0: float) -> CrawlResult:
        """Lightweight fallback: fetch HTML and strip tags."""
        req = Request(url, headers={"User-Agent": self.user_agent})
        resp = urlopen(req, timeout=self.timeout)  # noqa: S310
        content_type = resp.headers.get("Content-Type", "")
        raw = resp.read()

        encoding = "utf-8"
        if "charset=" in content_type:
            encoding = content_type.split("charset=")[-1].split(";")[0].strip()
        html = raw.decode(encoding, errors="replace")

        title_match = re.search(r"<title[^>]*>(.*?)</title>", html, re.DOTALL | re.IGNORECASE)
        title = title_match.group(1).strip() if title_match else ""

        markdown = self._html_to_markdown(html)
        if len(markdown) > self.max_content_length:
            markdown = markdown[: self.max_content_length] + "\n\n[... truncated]"

        elapsed = time.monotonic() - t0
        return CrawlResult(
            url=url, markdown=markdown, title=title,
            success=bool(markdown.strip()), elapsed_seconds=elapsed,
        )

    @staticmethod
    def _html_to_markdown(html: str) -> str:
        """Best-effort HTML → Markdown conversion via regex."""
        text = re.sub(r"<(script|style|noscript)[^>]*>.*?</\1>", "", html, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r"<h1[^>]*>(.*?)</h1>", r"\n# \1\n", text, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r"<h2[^>]*>(.*?)</h2>", r"\n## \1\n", text, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r"<h3[^>]*>(.*?)</h3>", r"\n### \1\n", text, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r"<li[^>]*>(.*?)</li>", r"\n- \1", text, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r"<p[^>]*>(.*?)</p>", r"\n\1\n", text, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r"<br\s*/?>", "\n", text, flags=re.IGNORECASE)
        text = re.sub(r"<a[^>]*href=[\"']([^\"']*)[\"'][^>]*>(.*?)</a>", r"[\2](\1)", text, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r"<[^>]+>", "", text)
        import html as _html
        text = _html.unescape(text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        text = re.sub(r" {2,}", " ", text)
        return text.strip()
