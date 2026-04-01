"""Tests for the Code Searcher agent."""

from __future__ import annotations

import json
import time
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch

from researchclaw.agents.code_searcher.agent import CodeSearchAgent, CodeSearchResult
from researchclaw.agents.code_searcher.cache import SearchCache
from researchclaw.agents.code_searcher.github_client import (
    CodeSnippet,
    GitHubClient,
    RepoAnalysis,
    RepoInfo,
)
from researchclaw.agents.code_searcher.pattern_extractor import (
    CodePatterns,
    extract_patterns,
    _heuristic_extract,
)
from researchclaw.agents.code_searcher.query_gen import (
    generate_search_queries,
    _heuristic_generate,
    _extract_key_phrases,
)
from researchclaw.domains.detector import DomainProfile, get_profile


# ---------------------------------------------------------------------------
# Query Generation tests
# ---------------------------------------------------------------------------


class TestQueryGeneration:
    def test_heuristic_generates_queries(self):
        queries = _heuristic_generate(
            topic="finite element method for Poisson equation",
            domain_name="PDE Solvers",
            libraries=["numpy", "scipy", "fenics"],
            needs=["FEM assembly", "mesh generation"],
        )
        assert len(queries) >= 3
        assert len(queries) <= 5
        # Should include library names
        any_lib = any("numpy" in q or "scipy" in q or "fenics" in q for q in queries)
        assert any_lib

    def test_heuristic_no_duplicates(self):
        queries = _heuristic_generate(
            topic="simple test",
            domain_name="Test",
            libraries=["numpy"],
            needs=[],
        )
        # No exact duplicates
        assert len(queries) == len(set(q.lower().strip() for q in queries))

    def test_extract_key_phrases(self):
        result = _extract_key_phrases("A Novel Approach for Image Classification Using Deep Learning")
        # Should remove filler words
        assert "novel" not in result.lower()
        assert "using" not in result.lower()

    def test_generate_without_llm(self):
        queries = generate_search_queries(
            topic="molecular dynamics simulation",
            domain_name="Computational Physics",
            core_libraries=["jax", "numpy"],
            llm=None,
        )
        assert isinstance(queries, list)
        assert len(queries) >= 2


# ---------------------------------------------------------------------------
# Pattern Extractor tests
# ---------------------------------------------------------------------------


class TestPatternExtractor:
    def test_heuristic_extract_imports(self):
        snippets = [
            "import numpy as np\nimport scipy.sparse as sp\n\ndef solve():\n    pass",
            "from pyscf import gto, scf\nmol = gto.M(atom='H 0 0 0')",
        ]
        patterns = _heuristic_extract(snippets)
        assert len(patterns.api_patterns) > 0
        assert any("numpy" in p for p in patterns.api_patterns)

    def test_heuristic_extract_functions(self):
        snippets = [
            "class Solver:\n    pass\ndef solve_pde():\n    pass\ndef analyze():\n    pass",
        ]
        patterns = _heuristic_extract(snippets)
        assert len(patterns.file_structure) > 0

    def test_empty_snippets(self):
        patterns = extract_patterns([], topic="test", domain_name="test")
        assert not patterns.has_content

    def test_code_patterns_to_prompt(self):
        patterns = CodePatterns(
            api_patterns=["import numpy as np\nresult = np.linalg.solve(A, b)"],
            file_structure={"solver.py": "Main solver implementation"},
            evaluation_patterns=["error = np.linalg.norm(x - x_exact)"],
        )
        ctx = patterns.to_prompt_context()
        assert "numpy" in ctx
        assert "solver.py" in ctx
        assert "error" in ctx

    def test_code_patterns_has_content(self):
        empty = CodePatterns()
        assert not empty.has_content

        with_data = CodePatterns(api_patterns=["import x"])
        assert with_data.has_content


# ---------------------------------------------------------------------------
# Search Cache tests
# ---------------------------------------------------------------------------


class TestSearchCache:
    def test_put_and_get(self, tmp_path):
        cache = SearchCache(cache_dir=tmp_path, ttl_days=30)
        data = {"api_patterns": ["import numpy"], "repos": []}
        cache.put("ml_vision", "image classification", data)

        result = cache.get("ml_vision", "image classification")
        assert result is not None
        assert result["api_patterns"] == ["import numpy"]

    def test_cache_miss(self, tmp_path):
        cache = SearchCache(cache_dir=tmp_path)
        result = cache.get("unknown", "unknown topic")
        assert result is None

    def test_cache_expiry(self, tmp_path):
        cache = SearchCache(cache_dir=tmp_path, ttl_days=0)  # immediate expiry
        data = {"test": True}
        cache.put("test", "topic", data)

        # Manually set old timestamp
        cache_path = tmp_path / "test"
        for f in cache_path.glob("*.json"):
            content = json.loads(f.read_text())
            content["_cached_at"] = time.time() - 86400  # 1 day ago
            f.write_text(json.dumps(content))

        result = cache.get("test", "topic")
        assert result is None  # expired

    def test_clear_domain(self, tmp_path):
        cache = SearchCache(cache_dir=tmp_path)
        cache.put("ml_vision", "topic1", {"data": 1})
        cache.put("ml_vision", "topic2", {"data": 2})
        cache.put("physics", "topic3", {"data": 3})

        count = cache.clear("ml_vision")
        assert count == 2
        assert cache.get("ml_vision", "topic1") is None
        assert cache.get("physics", "topic3") is not None

    def test_clear_all(self, tmp_path):
        cache = SearchCache(cache_dir=tmp_path)
        cache.put("a", "t1", {"x": 1})
        cache.put("b", "t2", {"x": 2})

        count = cache.clear()
        assert count == 2

    def test_stats(self, tmp_path):
        cache = SearchCache(cache_dir=tmp_path)
        cache.put("ml_vision", "t1", {"x": 1})
        cache.put("ml_vision", "t2", {"x": 2})
        cache.put("physics", "t3", {"x": 3})

        stats = cache.stats()
        assert stats["total"] == 3
        assert stats.get("ml_vision", 0) == 2

    def test_topic_hash_deterministic(self):
        h1 = SearchCache._topic_hash("test topic")
        h2 = SearchCache._topic_hash("test topic")
        assert h1 == h2

    def test_topic_hash_case_insensitive(self):
        h1 = SearchCache._topic_hash("Test Topic")
        h2 = SearchCache._topic_hash("test topic")
        assert h1 == h2


# ---------------------------------------------------------------------------
# GitHubClient tests (mocked)
# ---------------------------------------------------------------------------


class TestGitHubClient:
    def test_has_token_false(self):
        with patch.dict("os.environ", {}, clear=True):
            client = GitHubClient(token="")
            # Can't easily clear env, but token="" means no token
            assert not client.has_token

    def test_has_token_true(self):
        client = GitHubClient(token="ghp_test123")
        assert client.has_token

    def test_headers_with_token(self):
        client = GitHubClient(token="ghp_test123")
        headers = client._headers()
        assert "Authorization" in headers
        assert "Bearer" in headers["Authorization"]

    def test_headers_without_token(self):
        client = GitHubClient(token="")
        headers = client._headers()
        assert "Authorization" not in headers


# ---------------------------------------------------------------------------
# RepoInfo / CodeSnippet data class tests
# ---------------------------------------------------------------------------


class TestDataClasses:
    def test_repo_info_defaults(self):
        repo = RepoInfo(full_name="owner/repo")
        assert repo.stars == 0
        assert repo.default_branch == "main"

    def test_code_snippet(self):
        snippet = CodeSnippet(
            repo_full_name="owner/repo",
            file_path="src/main.py",
        )
        assert snippet.content == ""

    def test_repo_analysis(self):
        analysis = RepoAnalysis(
            repo=RepoInfo(full_name="test/repo"),
            readme="# Test Repo",
            requirements=["numpy", "scipy"],
        )
        assert len(analysis.requirements) == 2


# ---------------------------------------------------------------------------
# CodeSearchResult tests
# ---------------------------------------------------------------------------


class TestCodeSearchResult:
    def test_empty_result(self):
        result = CodeSearchResult()
        assert result.to_prompt_context() == ""
        assert not result.from_cache

    def test_result_with_patterns(self):
        result = CodeSearchResult(
            patterns=CodePatterns(
                api_patterns=["import numpy as np"],
                file_structure={"main.py": "Entry point"},
            ),
        )
        ctx = result.to_prompt_context()
        assert "numpy" in ctx

    def test_cache_roundtrip(self):
        result = CodeSearchResult(
            patterns=CodePatterns(
                api_patterns=["import numpy"],
                file_structure={"main.py": "Entry"},
                evaluation_patterns=["error = norm(diff)"],
            ),
            repos_found=[
                RepoInfo(full_name="test/repo", stars=100, html_url="https://example.com"),
            ],
            queries_used=["test query"],
        )
        cache_dict = result.to_cache_dict()
        restored = CodeSearchResult.from_cache_dict(cache_dict)
        assert restored.from_cache
        assert restored.patterns.api_patterns == ["import numpy"]
        assert len(restored.repos_found) == 1
        assert restored.queries_used == ["test query"]


# ---------------------------------------------------------------------------
# CodeSearchAgent tests (mocked GitHub)
# ---------------------------------------------------------------------------


class TestCodeSearchAgent:
    def _mock_github(self):
        """Create a mock GitHub client."""
        mock = MagicMock(spec=GitHubClient)
        mock.search_repos.return_value = [
            RepoInfo(
                full_name="user/physics-sim",
                description="Physics simulation framework",
                stars=500,
                html_url="https://github.com/user/physics-sim",
            ),
        ]
        mock.search_code.return_value = [
            CodeSnippet(
                repo_full_name="user/physics-sim",
                file_path="main.py",
                score=10.0,
            ),
        ]
        mock.get_readme.return_value = "# Physics Simulation\nA framework for physics sims."
        mock.get_repo_tree.return_value = ["main.py", "solver.py", "requirements.txt"]
        mock.get_file_content.return_value = "import numpy as np\ndef solve(): pass"
        mock.request_count = 5
        return mock

    def test_search_uses_cache(self, tmp_path):
        cache = SearchCache(cache_dir=tmp_path)
        cache.put("physics_simulation", "N-body sim", {
            "api_patterns": ["cached pattern"],
            "file_structure": {},
            "evaluation_patterns": [],
            "library_versions": {},
            "repos": [],
            "queries": ["cached query"],
        })

        agent = CodeSearchAgent(cache=cache)
        profile = DomainProfile(
            domain_id="physics_simulation",
            display_name="Physics",
            core_libraries=["numpy"],
        )
        result = agent.search("N-body sim", profile)
        assert result.from_cache
        assert result.patterns.api_patterns == ["cached pattern"]

    def test_search_with_mock_github(self, tmp_path):
        mock_github = self._mock_github()
        cache = SearchCache(cache_dir=tmp_path)

        agent = CodeSearchAgent(cache=cache)
        agent._github = mock_github

        profile = DomainProfile(
            domain_id="physics_simulation",
            display_name="Computational Physics",
            core_libraries=["numpy", "scipy"],
            github_search_terms=["physics simulation python"],
        )
        result = agent.search("molecular dynamics simulation", profile)

        assert not result.from_cache
        assert len(result.queries_used) >= 2
        mock_github.search_repos.assert_called_once()

    def test_search_graceful_failure(self, tmp_path):
        """If GitHub fails, should still return empty result without crashing."""
        mock_github = MagicMock(spec=GitHubClient)
        mock_github.search_repos.side_effect = Exception("Network error")
        mock_github.search_code.side_effect = Exception("Network error")
        mock_github.request_count = 0

        cache = SearchCache(cache_dir=tmp_path)
        agent = CodeSearchAgent(cache=cache)
        agent._github = mock_github

        profile = DomainProfile(
            domain_id="test",
            display_name="Test",
            core_libraries=["numpy"],
        )
        result = agent.search("test topic", profile)
        # Should not crash
        assert isinstance(result, CodeSearchResult)
