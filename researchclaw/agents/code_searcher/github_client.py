"""GitHub REST API client for code and repository search.

Handles rate limiting, authentication, and response parsing for:
  - Repository search (``/search/repositories``)
  - Code search (``/search/code``)
  - File content retrieval (``/repos/{owner}/{repo}/contents/{path}``)
  - README retrieval

Rate limits:
  - Authenticated: 30 req/min for search, 5000 req/hr for core
  - Code search: 10 req/min
  - Unauthenticated: 10 req/min for search
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import quote

logger = logging.getLogger(__name__)

_GITHUB_API = "https://api.github.com"


@dataclass
class RepoInfo:
    """Summary of a GitHub repository."""
    full_name: str  # "owner/repo"
    description: str = ""
    stars: int = 0
    language: str = ""
    updated_at: str = ""
    html_url: str = ""
    default_branch: str = "main"
    topics: list[str] = field(default_factory=list)


@dataclass
class CodeSnippet:
    """A code snippet found via GitHub code search."""
    repo_full_name: str
    file_path: str
    file_url: str = ""
    content: str = ""  # populated after fetching
    score: float = 0.0


@dataclass
class RepoAnalysis:
    """Analysis of a repository's structure and content."""
    repo: RepoInfo
    readme: str = ""
    requirements: list[str] = field(default_factory=list)
    key_files: dict[str, str] = field(default_factory=dict)  # path -> content
    file_tree: list[str] = field(default_factory=list)


class GitHubClient:
    """GitHub REST API client with rate limiting and caching.

    Uses ``GITHUB_TOKEN`` env var for authentication (strongly recommended).
    Falls back to unauthenticated access (much lower rate limits).
    """

    def __init__(self, token: str | None = None) -> None:
        self._token = token or os.environ.get("GITHUB_TOKEN", "")
        self._last_search_time: float = 0
        self._search_interval: float = 6.0  # 10 req/min → 6s between requests
        self._request_count: int = 0

    def _headers(self) -> dict[str, str]:
        headers = {
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        }
        if self._token:
            headers["Authorization"] = f"Bearer {self._token}"
        return headers

    def _rate_limit_wait(self) -> None:
        """Enforce rate limiting between search requests."""
        elapsed = time.time() - self._last_search_time
        if elapsed < self._search_interval:
            wait = self._search_interval - elapsed
            logger.debug("Rate limit: waiting %.1fs", wait)
            time.sleep(wait)
        self._last_search_time = time.time()

    def _get(self, url: str, params: dict[str, str] | None = None) -> dict[str, Any] | None:
        """Make a GET request to the GitHub API."""
        import urllib.request
        import urllib.error
        import json

        if params:
            query_str = "&".join(f"{k}={quote(str(v))}" for k, v in params.items())
            url = f"{url}?{query_str}"

        req = urllib.request.Request(url, headers=self._headers())
        self._request_count += 1

        try:
            with urllib.request.urlopen(req, timeout=15) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            if e.code == 403:
                logger.warning("GitHub API rate limited (403). Skipping.")
                return None
            if e.code == 422:
                logger.warning("GitHub API validation error (422): %s", url)
                return None
            logger.warning("GitHub API error %d: %s", e.code, url)
            return None
        except Exception:
            logger.warning("GitHub API request failed: %s", url, exc_info=True)
            return None

    def search_repos(
        self,
        query: str,
        language: str = "Python",
        sort: str = "stars",
        max_results: int = 10,
    ) -> list[RepoInfo]:
        """Search for repositories matching a query.

        Parameters
        ----------
        query : str
            Search query (e.g., "PDE solver finite element").
        language : str
            Filter by programming language.
        sort : str
            Sort order: "stars", "updated", "best-match".
        max_results : int
            Maximum number of results to return.

        Returns
        -------
        list[RepoInfo]
        """
        self._rate_limit_wait()

        search_q = f"{query} language:{language}"
        params = {
            "q": search_q,
            "sort": sort,
            "order": "desc",
            "per_page": str(min(max_results, 30)),
        }

        data = self._get(f"{_GITHUB_API}/search/repositories", params)
        if data is None:
            return []

        repos: list[RepoInfo] = []
        for item in data.get("items", [])[:max_results]:
            repos.append(RepoInfo(
                full_name=item.get("full_name", ""),
                description=item.get("description", "") or "",
                stars=item.get("stargazers_count", 0),
                language=item.get("language", "") or "",
                updated_at=item.get("updated_at", ""),
                html_url=item.get("html_url", ""),
                default_branch=item.get("default_branch", "main"),
                topics=item.get("topics", []),
            ))

        logger.info("Found %d repos for query: %.60s", len(repos), query)
        return repos

    def search_code(
        self,
        query: str,
        language: str = "Python",
        max_results: int = 10,
    ) -> list[CodeSnippet]:
        """Search for code snippets matching a query.

        Note: Code search has stricter rate limits (10 req/min).

        Parameters
        ----------
        query : str
            Search query (e.g., "from pyscf import gto scf").
        language : str
            Filter by programming language.
        max_results : int
            Maximum results.

        Returns
        -------
        list[CodeSnippet]
        """
        self._rate_limit_wait()

        search_q = f"{query} language:{language}"
        params = {
            "q": search_q,
            "per_page": str(min(max_results, 30)),
        }

        data = self._get(f"{_GITHUB_API}/search/code", params)
        if data is None:
            return []

        snippets: list[CodeSnippet] = []
        for item in data.get("items", [])[:max_results]:
            repo = item.get("repository", {})
            snippets.append(CodeSnippet(
                repo_full_name=repo.get("full_name", ""),
                file_path=item.get("path", ""),
                file_url=item.get("html_url", ""),
                score=item.get("score", 0.0),
            ))

        logger.info("Found %d code snippets for query: %.60s", len(snippets), query)
        return snippets

    def get_file_content(
        self,
        repo_full_name: str,
        path: str,
        max_size_kb: int = 100,
    ) -> str | None:
        """Get the content of a file from a repository.

        Parameters
        ----------
        repo_full_name : str
            Repository in "owner/repo" format.
        path : str
            File path within the repository.
        max_size_kb : int
            Skip files larger than this.

        Returns
        -------
        str or None
            File content, or None if not found/too large.
        """
        import base64

        url = f"{_GITHUB_API}/repos/{repo_full_name}/contents/{quote(path, safe='/')}"
        data = self._get(url)
        if data is None:
            return None

        size = data.get("size", 0)
        if size > max_size_kb * 1024:
            logger.debug("File too large (%d KB): %s/%s", size // 1024, repo_full_name, path)
            return None

        content = data.get("content", "")
        encoding = data.get("encoding", "")

        if encoding == "base64":
            try:
                return base64.b64decode(content).decode("utf-8", errors="replace")
            except Exception:
                return None
        return content

    def get_readme(self, repo_full_name: str) -> str | None:
        """Get the README content of a repository."""
        import base64

        url = f"{_GITHUB_API}/repos/{repo_full_name}/readme"
        data = self._get(url)
        if data is None:
            return None

        content = data.get("content", "")
        encoding = data.get("encoding", "")
        if encoding == "base64":
            try:
                return base64.b64decode(content).decode("utf-8", errors="replace")
            except Exception:
                return None
        return content

    def get_repo_tree(
        self,
        repo_full_name: str,
        branch: str = "main",
    ) -> list[str]:
        """Get the file tree of a repository (flat list of paths)."""
        url = f"{_GITHUB_API}/repos/{repo_full_name}/git/trees/{branch}"
        params = {"recursive": "1"}
        data = self._get(url, params)
        if data is None:
            return []

        tree = data.get("tree", [])
        return [item["path"] for item in tree if item.get("type") == "blob"]

    @property
    def request_count(self) -> int:
        return self._request_count

    @property
    def has_token(self) -> bool:
        return bool(self._token)
