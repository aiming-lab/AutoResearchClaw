"""Daily paper digest generation."""

from __future__ import annotations

import logging
from datetime import date
from pathlib import Path
from typing import Any

from researchclaw.trends.feeds import FeedManager

logger = logging.getLogger(__name__)


class DailyDigest:
    """Generate daily paper digest reports."""

    def __init__(
        self,
        feed_manager: FeedManager,
        llm_client: Any = None,
    ):
        self.feeds = feed_manager
        self.llm = llm_client

    async def generate(
        self,
        domains: list[str] | None = None,
        max_papers: int = 20,
        target_date: date | None = None,
    ) -> str:
        """Generate a daily paper digest as Markdown."""
        effective_domains = domains or ["machine learning"]
        today = target_date or date.today()

        papers = self.feeds.fetch_recent_papers(
            domains=effective_domains,
            max_papers=max_papers,
            since_date=today,
        )

        if not papers:
            return (
                f"## Daily Paper Digest ({today})\n\n"
                f"No new papers found for domains: {', '.join(effective_domains)}\n"
            )

        if self.llm is not None:
            return await self._generate_with_llm(papers, effective_domains, today)
        return self._generate_basic(papers, effective_domains, today)

    async def _generate_with_llm(
        self,
        papers: list[dict[str, Any]],
        domains: list[str],
        today: date,
    ) -> str:
        """Generate digest with LLM-enhanced summaries."""
        lines = [
            f"## Daily Paper Digest ({today})",
            f"Domains: {', '.join(domains)}",
            f"Papers found: {len(papers)}",
            "",
        ]

        for i, paper in enumerate(papers, 1):
            title = paper.get("title", "Untitled")
            url = paper.get("url", "")
            abstract = paper.get("abstract", "")[:500]
            authors = paper.get("authors", [])

            if isinstance(authors, list):
                author_str = ", ".join(
                    a if isinstance(a, str) else a.get("name", "")
                    for a in authors[:3]
                )
                if len(authors) > 3:
                    author_str += " et al."
            else:
                author_str = str(authors)

            # Get LLM summary
            try:
                prompt = (
                    f"Summarize this paper in 2 sentences and rate its relevance "
                    f"to {', '.join(domains)} on a scale of 1-5 stars.\n\n"
                    f"Title: {title}\nAbstract: {abstract}\n\n"
                    f"Format: SUMMARY: <text> | RELEVANCE: <1-5>"
                )
                response = await self.llm.chat_async(prompt)
                summary, relevance = self._parse_summary(response)
            except Exception:
                summary = abstract[:200] + "..." if len(abstract) > 200 else abstract
                relevance = 3

            stars = "*" * relevance
            link = f"[{title}]({url})" if url else title
            lines.extend([
                f"### {i}. {link}",
                f"**Authors**: {author_str}",
                f"**Relevance**: {stars}",
                f"**Summary**: {summary}",
                "",
            ])

        return "\n".join(lines)

    def _generate_basic(
        self,
        papers: list[dict[str, Any]],
        domains: list[str],
        today: date,
    ) -> str:
        """Generate basic digest without LLM."""
        lines = [
            f"## Daily Paper Digest ({today})",
            f"Domains: {', '.join(domains)}",
            f"Papers found: {len(papers)}",
            "",
        ]

        for i, paper in enumerate(papers, 1):
            title = paper.get("title", "Untitled")
            url = paper.get("url", "")
            abstract = paper.get("abstract", "")
            authors = paper.get("authors", [])

            if isinstance(authors, list):
                author_str = ", ".join(
                    a if isinstance(a, str) else a.get("name", "")
                    for a in authors[:3]
                )
                if len(authors) > 3:
                    author_str += " et al."
            else:
                author_str = str(authors)

            short_abstract = (
                abstract[:200] + "..." if len(abstract) > 200 else abstract
            )
            link = f"[{title}]({url})" if url else title
            lines.extend([
                f"### {i}. {link}",
                f"**Authors**: {author_str}",
                f"**Abstract**: {short_abstract}",
                "",
            ])

        return "\n".join(lines)

    @staticmethod
    def _parse_summary(response: str) -> tuple[str, int]:
        """Parse LLM summary response."""
        summary = response
        relevance = 3

        if "SUMMARY:" in response:
            parts = response.split("|")
            summary = parts[0].split("SUMMARY:", 1)[-1].strip()
            if len(parts) > 1 and "RELEVANCE:" in parts[1]:
                try:
                    rel_str = parts[1].split("RELEVANCE:", 1)[-1].strip()
                    relevance = int(rel_str.strip("* "))
                    relevance = max(1, min(5, relevance))
                except (ValueError, IndexError):
                    pass

        return summary, relevance

    async def generate_and_save(
        self,
        output_dir: Path,
        domains: list[str] | None = None,
        max_papers: int = 20,
    ) -> Path:
        """Generate digest and save to a file."""
        today = date.today()
        content = await self.generate(domains, max_papers, today)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / f"digest_{today}.md"
        output_file.write_text(content, encoding="utf-8")
        return output_file
