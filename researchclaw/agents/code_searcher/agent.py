"""Code Search Agent — orchestrates GitHub search, pattern extraction, and caching.

This is the main entry point for code search. It:
1. Checks cache for existing results
2. Generates search queries (LLM or heuristic)
3. Searches GitHub for repos and code
4. Reads key files from top repos
5. Extracts patterns using LLM
6. Caches results for future use
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from researchclaw.agents.code_searcher.cache import SearchCache
from researchclaw.agents.code_searcher.github_client import (
    CodeSnippet,
    GitHubClient,
    RepoAnalysis,
    RepoInfo,
)
from researchclaw.agents.code_searcher.pattern_extractor import CodePatterns, extract_patterns
from researchclaw.agents.code_searcher.query_gen import generate_search_queries
from researchclaw.domains.detector import DomainProfile

logger = logging.getLogger(__name__)


@dataclass
class CodeSearchResult:
    """Complete result from a code search operation."""
    patterns: CodePatterns = field(default_factory=CodePatterns)
    repos_found: list[RepoInfo] = field(default_factory=list)
    snippets_found: list[CodeSnippet] = field(default_factory=list)
    repo_analyses: list[RepoAnalysis] = field(default_factory=list)
    queries_used: list[str] = field(default_factory=list)
    from_cache: bool = False
    github_requests: int = 0

    def to_prompt_context(self) -> str:
        """Format as context block for injection into code generation prompts."""
        if not self.patterns.has_content:
            return ""
        return self.patterns.to_prompt_context()

    def to_cache_dict(self) -> dict[str, Any]:
        """Serialize for caching."""
        return {
            "api_patterns": self.patterns.api_patterns,
            "file_structure": self.patterns.file_structure,
            "evaluation_patterns": self.patterns.evaluation_patterns,
            "library_versions": self.patterns.library_versions,
            "repos": [
                {
                    "full_name": r.full_name,
                    "description": r.description,
                    "stars": r.stars,
                    "html_url": r.html_url,
                }
                for r in self.repos_found[:5]
            ],
            "queries": self.queries_used,
        }

    @classmethod
    def from_cache_dict(cls, data: dict[str, Any]) -> CodeSearchResult:
        """Deserialize from cache."""
        patterns = CodePatterns(
            api_patterns=data.get("api_patterns", []),
            file_structure=data.get("file_structure", {}),
            evaluation_patterns=data.get("evaluation_patterns", []),
            library_versions=data.get("library_versions", {}),
        )
        repos = [
            RepoInfo(
                full_name=r.get("full_name", ""),
                description=r.get("description", ""),
                stars=r.get("stars", 0),
                html_url=r.get("html_url", ""),
            )
            for r in data.get("repos", [])
        ]
        return cls(
            patterns=patterns,
            repos_found=repos,
            queries_used=data.get("queries", []),
            from_cache=True,
        )


class CodeSearchAgent:
    """Orchestrates code search for reference material before code generation.

    Usage::

        agent = CodeSearchAgent(llm=llm_client)
        result = agent.search(
            topic="PDE solver comparison",
            domain=domain_profile,
            specific_needs=["finite element method", "convergence test"],
        )
        context = result.to_prompt_context()
    """

    def __init__(
        self,
        llm: Any | None = None,
        github_token: str | None = None,
        cache: SearchCache | None = None,
        max_repos_to_analyze: int = 3,
        max_code_searches: int = 3,
    ) -> None:
        self._llm = llm
        self._github = GitHubClient(token=github_token)
        self._cache = cache or SearchCache()
        self._max_repos = max_repos_to_analyze
        self._max_code_searches = max_code_searches

    def search(
        self,
        topic: str,
        domain: DomainProfile,
        specific_needs: list[str] | None = None,
    ) -> CodeSearchResult:
        """Execute a complete code search for a research topic.

        Flow:
        1. Check cache
        2. Generate search queries
        3. Search GitHub repos + code
        4. Read key files from top repos
        5. Extract patterns
        6. Cache results

        Parameters
        ----------
        topic : str
            Research topic.
        domain : DomainProfile
            Detected domain profile.
        specific_needs : list[str], optional
            Specific library/API needs.

        Returns
        -------
        CodeSearchResult
        """
        logger.info("Code search started for: %.60s (domain=%s)", topic, domain.domain_id)

        # 1. Check cache
        cached = self._cache.get(domain.domain_id, topic)
        if cached:
            logger.info("Using cached code search results")
            return CodeSearchResult.from_cache_dict(cached)

        # 2. Generate search queries
        queries = generate_search_queries(
            topic=topic,
            domain_name=domain.display_name,
            core_libraries=domain.core_libraries,
            specific_needs=specific_needs,
            llm=self._llm,
        )

        # Add domain-specific search terms from profile
        if domain.github_search_terms:
            for term in domain.github_search_terms[:2]:
                if term not in queries:
                    queries.append(term)

        result = CodeSearchResult(queries_used=queries)

        # 3. Search GitHub repos (use first query)
        if queries:
            try:
                repos = self._github.search_repos(queries[0], max_results=10)
                # Filter: recent, well-starred
                repos = [
                    r for r in repos
                    if r.stars >= 10  # minimum quality threshold
                ]
                result.repos_found = repos[:self._max_repos * 2]
            except Exception:
                logger.warning("Repo search failed, continuing", exc_info=True)

        # 4. Search GitHub code (use remaining queries)
        code_snippets: list[str] = []
        for query in queries[1:self._max_code_searches + 1]:
            try:
                snippets = self._github.search_code(query, max_results=5)
                result.snippets_found.extend(snippets)
            except Exception:
                logger.warning("Code search failed for query: %s", query)

        # 5. Read key files from top repos
        for repo in result.repos_found[:self._max_repos]:
            try:
                analysis = self._analyze_repo(repo)
                if analysis:
                    result.repo_analyses.append(analysis)
                    # Collect code snippets
                    for content in analysis.key_files.values():
                        if content:
                            code_snippets.append(content)
            except Exception:
                logger.warning("Failed to analyze repo: %s", repo.full_name)

        # Also fetch content for code search results
        for snippet in result.snippets_found[:5]:
            try:
                content = self._github.get_file_content(
                    snippet.repo_full_name,
                    snippet.file_path,
                )
                if content:
                    snippet.content = content
                    code_snippets.append(content)
            except Exception:
                pass

        # 6. Extract patterns
        if code_snippets:
            result.patterns = extract_patterns(
                code_snippets=code_snippets,
                topic=topic,
                domain_name=domain.display_name,
                llm=self._llm,
            )

        result.github_requests = self._github.request_count

        # 7. Cache results
        if result.patterns.has_content:
            self._cache.put(domain.domain_id, topic, result.to_cache_dict())

        logger.info(
            "Code search complete: %d repos, %d snippets, %d patterns, %d API calls",
            len(result.repos_found),
            len(result.snippets_found),
            len(result.patterns.api_patterns),
            result.github_requests,
        )

        return result

    def _analyze_repo(self, repo: RepoInfo) -> RepoAnalysis | None:
        """Analyze a repository by reading key files."""
        analysis = RepoAnalysis(repo=repo)

        # Get README
        readme = self._github.get_readme(repo.full_name)
        if readme:
            analysis.readme = readme[:3000]  # truncate

        # Get file tree
        file_tree = self._github.get_repo_tree(
            repo.full_name,
            repo.default_branch,
        )
        analysis.file_tree = file_tree

        # Identify and read key files
        key_patterns = [
            "main.py", "run.py", "train.py", "experiment.py",
            "requirements.txt", "setup.py", "pyproject.toml",
        ]
        for pattern in key_patterns:
            matches = [f for f in file_tree if f.endswith(pattern)]
            for match in matches[:1]:  # first match only
                content = self._github.get_file_content(
                    repo.full_name, match, max_size_kb=50,
                )
                if content:
                    analysis.key_files[match] = content

        # Parse requirements
        req_content = analysis.key_files.get("requirements.txt", "")
        if req_content:
            analysis.requirements = [
                line.strip().split("==")[0].split(">=")[0]
                for line in req_content.splitlines()
                if line.strip() and not line.startswith("#")
            ]

        return analysis
