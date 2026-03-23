# pyright: reportMissingImports=false, reportUnknownParameterType=false, reportMissingParameterType=false, reportUnknownVariableType=false, reportUnknownMemberType=false, reportUnknownArgumentType=false, reportPrivateUsage=false, reportUnknownLambdaType=false
from __future__ import annotations

import json
import urllib.error
from email.message import Message
from pathlib import Path
from unittest.mock import patch

import pytest


class _DummyResponse:
    def __init__(self, payload: bytes) -> None:
        self._payload: bytes = payload

    def read(self) -> bytes:
        return self._payload

    def __enter__(self) -> _DummyResponse:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        _ = exc_type, exc, tb
        return None


class TestRateLimitRetry:
    def test_s2_429_retries_and_succeeds(self) -> None:
        from researchclaw.literature.semantic_scholar import (
            _reset_circuit_breaker,
            search_semantic_scholar,
        )

        _reset_circuit_breaker()  # ensure clean CB state from prior tests

        call_count = 0

        def mock_urlopen(req, **kwargs):
            _ = kwargs
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise urllib.error.HTTPError(
                    req.full_url if hasattr(req, "full_url") else str(req),
                    429,
                    "Too Many Requests",
                    Message(),
                    None,
                )

            payload = json.dumps(
                {
                    "data": [
                        {
                            "paperId": "abc123",
                            "title": "Test Paper",
                            "authors": [{"name": "Smith"}],
                            "year": 2024,
                            "abstract": "test abstract",
                            "venue": "NeurIPS",
                            "citationCount": 10,
                            "externalIds": {"DOI": "10.1234/test"},
                            "url": "https://example.com",
                        }
                    ]
                }
            ).encode("utf-8")
            return _DummyResponse(payload)

        with patch("urllib.request.urlopen", side_effect=mock_urlopen):
            with patch("time.sleep"):
                papers = search_semantic_scholar("test query", limit=5)

        assert call_count >= 2
        assert len(papers) == 1

    def test_s2_persistent_429_exhausts_retries_and_returns_empty(self) -> None:
        from researchclaw.literature.semantic_scholar import (
            _MAX_RETRIES,
            _reset_circuit_breaker,
            search_semantic_scholar,
        )

        _reset_circuit_breaker()  # ensure clean CB state from prior tests
        call_count = 0

        def mock_urlopen(req, **kwargs):
            _ = kwargs
            nonlocal call_count
            call_count += 1
            raise urllib.error.HTTPError(
                req.full_url if hasattr(req, "full_url") else str(req),
                429,
                "Too Many Requests",
                Message(),
                None,
            )

        with patch("urllib.request.urlopen", side_effect=mock_urlopen):
            with patch("time.sleep"):
                papers = search_semantic_scholar("test query", limit=5)

        assert papers == []
        assert call_count == _MAX_RETRIES


class TestDegradationChain:
    def test_search_degrades_to_cache_on_api_failure(self, tmp_path: Path) -> None:
        from researchclaw.literature.cache import put_cache
        from researchclaw.literature.search import search_papers

        cached = [
            {
                "paper_id": "cached-1",
                "title": "Cached Paper",
                "authors": [],
                "year": 2024,
                "abstract": "cached",
                "venue": "",
                "citation_count": 5,
                "doi": "",
                "arxiv_id": "",
                "url": "",
                "source": "semantic_scholar",
            }
        ]
        put_cache(
            "test degradation", "semantic_scholar", 20, cached, cache_base=tmp_path
        )

        with patch(
            "researchclaw.literature.search.search_semantic_scholar",
            side_effect=RuntimeError("API down"),
        ):
            with patch(
                "researchclaw.literature.search.search_arxiv",
                side_effect=RuntimeError("API down"),
            ):
                with patch(
                    "researchclaw.literature.cache._DEFAULT_CACHE_DIR", tmp_path
                ):
                    papers = search_papers("test degradation", limit=20)

        assert len(papers) >= 1
        assert any(p.title == "Cached Paper" for p in papers)

    def test_search_empty_on_total_failure(self, tmp_path: Path) -> None:
        from researchclaw.literature.search import search_papers

        with patch(
            "researchclaw.literature.search.search_openalex",
            side_effect=RuntimeError("API down"),
        ):
            with patch(
                "researchclaw.literature.search.search_semantic_scholar",
                side_effect=RuntimeError("API down"),
            ):
                with patch(
                    "researchclaw.literature.search.search_arxiv",
                    side_effect=RuntimeError("API down"),
                ):
                    with patch(
                        "researchclaw.literature.cache._DEFAULT_CACHE_DIR",
                        tmp_path / "empty-cache",
                    ):
                        papers = search_papers("no results query", limit=20)

        assert papers == []


class TestLLMFallback:
    def test_primary_403_forbidden_fallback_succeeds(self) -> None:
        from researchclaw.llm.client import LLMClient, LLMConfig, LLMResponse

        client = LLMClient(
            LLMConfig(
                base_url="https://api.example.com/v1",
                api_key="test-key",
                primary_model="gpt-blocked",
                fallback_models=["gpt-fallback"],
                max_retries=1,
            )
        )

        call_models: list[str] = []

        def mock_raw_call(model, messages, max_tokens, temperature, json_mode):
            _ = messages, max_tokens, temperature, json_mode
            call_models.append(model)
            if model == "gpt-blocked":
                raise urllib.error.HTTPError(
                    "url", 403, "not allowed to use model", Message(), None
                )
            return LLMResponse(content="ok", model=model)

        with patch.object(client, "_raw_call", side_effect=mock_raw_call):
            resp = client.chat([{"role": "user", "content": "test"}])

        assert resp.content == "ok"
        assert "gpt-blocked" in call_models
        assert "gpt-fallback" in call_models

    def test_preflight_detects_401(self) -> None:
        from researchclaw.llm.client import LLMClient, LLMConfig

        client = LLMClient(
            LLMConfig(
                base_url="https://api.example.com/v1",
                api_key="bad-key",
                primary_model="gpt-test",
                fallback_models=[],
                max_retries=1,
            )
        )

        if not hasattr(client, "preflight"):
            pytest.skip("preflight() not yet implemented")

        err = urllib.error.HTTPError("url", 401, "Unauthorized", Message(), None)
        with patch.object(client, "chat", side_effect=err):
            ok, msg = client.preflight()

        assert ok is False
        assert "Invalid API key" in msg


class TestNoncriticalStageSkip:
    @staticmethod
    def _make_rc_config(tmp_path: Path):
        from researchclaw.config import RCConfig

        data = {
            "project": {"name": "rc-e2e-regression", "mode": "docs-first"},
            "research": {"topic": "pipeline regression"},
            "runtime": {"timezone": "UTC"},
            "notifications": {"channel": "local"},
            "knowledge_base": {"backend": "markdown", "root": str(tmp_path / "kb")},
            "openclaw_bridge": {},
            "llm": {
                "provider": "openai-compatible",
                "base_url": "http://localhost:1234/v1",
                "api_key_env": "RC_TEST_KEY",
                "api_key": "inline",
            },
        }
        return RCConfig.from_dict(data, project_root=tmp_path, check_paths=False)

    def test_noncritical_stage_failure_is_skipped(self, tmp_path: Path) -> None:
        from researchclaw.adapters import AdapterBundle
        from researchclaw.pipeline import runner as rc_runner
        from researchclaw.pipeline.executor import StageResult
        from researchclaw.pipeline.stages import STAGE_SEQUENCE, Stage, StageStatus

        run_dir = tmp_path / "run"
        run_dir.mkdir()
        config = self._make_rc_config(tmp_path)
        adapters = AdapterBundle()

        def mock_execute_stage(stage: Stage, **kwargs) -> StageResult:
            _ = kwargs
            if stage is Stage.KNOWLEDGE_ARCHIVE:
                return StageResult(
                    stage=stage,
                    status=StageStatus.FAILED,
                    artifacts=(),
                    error="archive error",
                )
            return StageResult(
                stage=stage, status=StageStatus.DONE, artifacts=("ok.md",)
            )

        with patch.object(rc_runner, "execute_stage", side_effect=mock_execute_stage):
            results = rc_runner.execute_pipeline(
                run_dir=run_dir,
                run_id="run-skip-noncritical",
                config=config,
                adapters=adapters,
                skip_noncritical=True,
            )

        assert len(results) == len(STAGE_SEQUENCE)
        assert results[-1].stage is Stage.CITATION_VERIFY
        assert any(
            r.stage is Stage.KNOWLEDGE_ARCHIVE and r.status is StageStatus.FAILED
            for r in results
        )

    def test_critical_stage_failure_still_aborts(self, tmp_path: Path) -> None:
        from researchclaw.adapters import AdapterBundle
        from researchclaw.pipeline import runner as rc_runner
        from researchclaw.pipeline.executor import StageResult
        from researchclaw.pipeline.stages import Stage, StageStatus

        run_dir = tmp_path / "run-critical"
        run_dir.mkdir()
        config = self._make_rc_config(tmp_path)
        adapters = AdapterBundle()

        def mock_execute_stage(stage: Stage, **kwargs) -> StageResult:
            _ = kwargs
            if stage is Stage.PAPER_DRAFT:
                return StageResult(
                    stage=stage,
                    status=StageStatus.FAILED,
                    artifacts=(),
                    error="draft error",
                )
            return StageResult(
                stage=stage, status=StageStatus.DONE, artifacts=("ok.md",)
            )

        with patch.object(rc_runner, "execute_stage", side_effect=mock_execute_stage):
            results = rc_runner.execute_pipeline(
                run_dir=run_dir,
                run_id="run-fail-critical",
                config=config,
                adapters=adapters,
                skip_noncritical=True,
            )

        assert results[-1].stage is Stage.PAPER_DRAFT
        assert results[-1].status is StageStatus.FAILED
