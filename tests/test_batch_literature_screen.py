# pyright: reportPrivateUsage=false, reportUnknownParameterType=false, reportMissingParameterType=false, reportUnknownMemberType=false, reportUnknownArgumentType=false, reportUnknownVariableType=false, reportUnusedCallResult=false, reportAttributeAccessIssue=false
"""Tests for batch LITERATURE_SCREEN, _opt_int, and new config fields."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from researchclaw.adapters import AdapterBundle
from researchclaw.config import RCConfig, _opt_int
from researchclaw.pipeline import executor as rc_executor
from researchclaw.pipeline.stages import StageStatus


# ── helpers ──────────────────────────────────────────────────────────


class FakeLLMClient:
    """Minimal LLM stub that returns a canned JSON response."""

    def __init__(self, responses: list[str] | str = "{}"):
        if isinstance(responses, str):
            responses = [responses]
        self._responses = list(responses)
        self._call_idx = 0
        self.calls: list[list[dict[str, str]]] = []

    def chat(self, messages: list[dict[str, str]], **kwargs: object):
        _ = kwargs
        self.calls.append(messages)
        from researchclaw.llm.client import LLMResponse

        text = self._responses[min(self._call_idx, len(self._responses) - 1)]
        self._call_idx += 1
        return LLMResponse(content=text, model="fake-model")


def _make_config(tmp_path: Path, **research_overrides: Any) -> RCConfig:
    research: dict[str, Any] = {
        "topic": "batch screening test",
        "domains": ["ml"],
        "daily_paper_count": 2,
        "quality_threshold": 5.0,
    }
    research.update(research_overrides)
    data: dict[str, Any] = {
        "project": {"name": "rc-test", "mode": "docs-first"},
        "research": research,
        "runtime": {"timezone": "UTC"},
        "notifications": {"channel": "local"},
        "knowledge_base": {"backend": "markdown", "root": str(tmp_path / "kb")},
        "openclaw_bridge": {"use_memory": True, "use_message": True},
        "llm": {
            "provider": "openai-compatible",
            "base_url": "http://localhost:1234/v1",
            "api_key_env": "RC_TEST_KEY",
            "api_key": "test-key",
            "primary_model": "fake-model",
            "fallback_models": [],
        },
        "security": {"hitl_required_stages": [5, 9, 20]},
        "experiment": {"mode": "sandbox"},
    }
    return RCConfig.from_dict(data, project_root=tmp_path, check_paths=False)


def _make_candidates(n: int) -> str:
    """Generate n JSONL candidate lines with ml-related titles."""
    lines = []
    for i in range(n):
        paper = {
            "title": f"ML Paper {i}",
            "abstract": f"This paper studies machine learning approach {i}.",
            "authors": ["Author A"],
            "year": 2024,
        }
        lines.append(json.dumps(paper, ensure_ascii=False))
    return "\n".join(lines)


def _write_candidates(run_dir: Path, n: int) -> None:
    stage_dir = run_dir / "stage-04"
    stage_dir.mkdir(parents=True, exist_ok=True)
    (stage_dir / "candidates.jsonl").write_text(
        _make_candidates(n), encoding="utf-8"
    )


def _shortlist_response(titles: list[str]) -> str:
    return json.dumps(
        {"shortlist": [{"title": t, "relevance_score": 0.9} for t in titles]}
    )


# ── _opt_int tests ───────────────────────────────────────────────────


class TestOptInt:
    def test_none_returns_none(self) -> None:
        assert _opt_int(None) is None

    def test_nan_returns_none(self) -> None:
        assert _opt_int(float("nan")) is None

    def test_int_value(self) -> None:
        assert _opt_int(10) == 10

    def test_float_value_truncated(self) -> None:
        assert _opt_int(3.7) == 3

    def test_string_numeric(self) -> None:
        assert _opt_int("25") == 25

    def test_string_nan_returns_none(self) -> None:
        assert _opt_int("nan") is None

    def test_non_numeric_raises(self) -> None:
        with pytest.raises(ValueError):
            _opt_int("abc")


# ── config field parsing tests ───────────────────────────────────────


class TestConfigScreenFields:
    def test_defaults_when_missing(self, tmp_path: Path) -> None:
        cfg = _make_config(tmp_path)
        assert cfg.research.screen_batch_size is None
        assert cfg.research.max_shortlist is None
        assert cfg.research.screen_max_workers == 4

    def test_explicit_values_parsed(self, tmp_path: Path) -> None:
        cfg = _make_config(
            tmp_path,
            screen_batch_size=20,
            max_shortlist=50,
            screen_max_workers=2,
        )
        assert cfg.research.screen_batch_size == 20
        assert cfg.research.max_shortlist == 50
        assert cfg.research.screen_max_workers == 2

    def test_null_batch_size_parses_as_none(self, tmp_path: Path) -> None:
        cfg = _make_config(tmp_path, screen_batch_size=None)
        assert cfg.research.screen_batch_size is None

    def test_nan_batch_size_parses_as_none(self, tmp_path: Path) -> None:
        cfg = _make_config(tmp_path, screen_batch_size=float("nan"))
        assert cfg.research.screen_batch_size is None


# ── batch literature screen tests ────────────────────────────────────


class TestBatchLiteratureScreen:
    def test_no_batching_when_batch_size_none(self, tmp_path: Path) -> None:
        """Non-batch path: single LLM call, shortlist returned."""
        run_dir = tmp_path / "run"
        run_dir.mkdir()
        _write_candidates(run_dir, 5)
        stage_dir = run_dir / "stage-05"
        stage_dir.mkdir(parents=True, exist_ok=True)

        titles = [f"ML Paper {i}" for i in range(5)]
        llm = FakeLLMClient(_shortlist_response(titles))
        cfg = _make_config(tmp_path, screen_batch_size=None)

        result = rc_executor._execute_literature_screen(
            stage_dir, run_dir, cfg, AdapterBundle(), llm=llm
        )
        assert result.status == StageStatus.DONE
        # Only 1 LLM call (no batching)
        assert len(llm.calls) == 1

    def test_batching_splits_candidates(self, tmp_path: Path) -> None:
        """5 candidates with batch_size=2 → 3 batches + 1 final ranking."""
        run_dir = tmp_path / "run"
        run_dir.mkdir()
        _write_candidates(run_dir, 5)
        stage_dir = run_dir / "stage-05"
        stage_dir.mkdir(parents=True, exist_ok=True)

        titles = [f"ML Paper {i}" for i in range(5)]
        # Each batch call returns its papers; final ranking returns all
        batch_resp = _shortlist_response(titles)
        llm = FakeLLMClient([batch_resp] * 4)  # 3 batches + 1 final
        cfg = _make_config(
            tmp_path, screen_batch_size=2, max_shortlist=50, screen_max_workers=1
        )

        result = rc_executor._execute_literature_screen(
            stage_dir, run_dir, cfg, AdapterBundle(), llm=llm
        )
        assert result.status == StageStatus.DONE
        # 3 batch calls + 1 final ranking call = 4
        assert len(llm.calls) == 4

    def test_dedup_removes_duplicate_titles(self, tmp_path: Path) -> None:
        """Batch results with duplicate titles are deduplicated."""
        run_dir = tmp_path / "run"
        run_dir.mkdir()
        _write_candidates(run_dir, 4)
        stage_dir = run_dir / "stage-05"
        stage_dir.mkdir(parents=True, exist_ok=True)

        # Both batches return the same paper
        dup_resp = _shortlist_response(["ML Paper 0", "ML Paper 1"])
        # Final ranking returns deduped set
        final_resp = _shortlist_response(["ML Paper 0", "ML Paper 1"])
        llm = FakeLLMClient([dup_resp, dup_resp, final_resp])
        cfg = _make_config(
            tmp_path, screen_batch_size=2, max_shortlist=50, screen_max_workers=1
        )

        result = rc_executor._execute_literature_screen(
            stage_dir, run_dir, cfg, AdapterBundle(), llm=llm
        )
        assert result.status == StageStatus.DONE

        shortlist_path = stage_dir / "shortlist.jsonl"
        lines = shortlist_path.read_text().strip().splitlines()
        titles = [json.loads(l)["title"] for l in lines]
        # No duplicates
        assert len(titles) == len(set(t.lower() for t in titles))

    def test_max_shortlist_cap_enforced(self, tmp_path: Path) -> None:
        """max_shortlist caps the output even when LLM returns more."""
        run_dir = tmp_path / "run"
        run_dir.mkdir()
        _write_candidates(run_dir, 30)
        stage_dir = run_dir / "stage-05"
        stage_dir.mkdir(parents=True, exist_ok=True)

        all_titles = [f"ML Paper {i}" for i in range(30)]
        resp = _shortlist_response(all_titles)
        llm = FakeLLMClient(resp)
        # max_shortlist=20, but _MIN_SHORTLIST=15 so effective cap is max(15,20)=20
        cfg = _make_config(tmp_path, screen_batch_size=None, max_shortlist=20)

        result = rc_executor._execute_literature_screen(
            stage_dir, run_dir, cfg, AdapterBundle(), llm=llm
        )
        assert result.status == StageStatus.DONE
        shortlist_path = stage_dir / "shortlist.jsonl"
        lines = shortlist_path.read_text().strip().splitlines()
        assert len(lines) <= 20

    def test_max_shortlist_floor_is_min_shortlist(self, tmp_path: Path) -> None:
        """max_shortlist < 15 is clamped to _MIN_SHORTLIST (15)."""
        run_dir = tmp_path / "run"
        run_dir.mkdir()
        _write_candidates(run_dir, 20)
        stage_dir = run_dir / "stage-05"
        stage_dir.mkdir(parents=True, exist_ok=True)

        all_titles = [f"ML Paper {i}" for i in range(20)]
        resp = _shortlist_response(all_titles)
        llm = FakeLLMClient(resp)
        cfg = _make_config(tmp_path, screen_batch_size=None, max_shortlist=5)

        result = rc_executor._execute_literature_screen(
            stage_dir, run_dir, cfg, AdapterBundle(), llm=llm
        )
        assert result.status == StageStatus.DONE
        shortlist_path = stage_dir / "shortlist.jsonl"
        lines = shortlist_path.read_text().strip().splitlines()
        # Effective cap is max(15, 5) = 15
        assert len(lines) == 15

    def test_batch_failure_continues_with_remaining(self, tmp_path: Path) -> None:
        """If one batch fails, others still contribute to shortlist."""
        run_dir = tmp_path / "run"
        run_dir.mkdir()
        _write_candidates(run_dir, 4)
        stage_dir = run_dir / "stage-05"
        stage_dir.mkdir(parents=True, exist_ok=True)

        # First batch returns valid, second returns garbage
        good_resp = _shortlist_response(["ML Paper 0", "ML Paper 1"])
        bad_resp = "not json at all"
        final_resp = _shortlist_response(["ML Paper 0", "ML Paper 1"])
        llm = FakeLLMClient([good_resp, bad_resp, final_resp])
        cfg = _make_config(
            tmp_path, screen_batch_size=2, max_shortlist=50, screen_max_workers=1
        )

        result = rc_executor._execute_literature_screen(
            stage_dir, run_dir, cfg, AdapterBundle(), llm=llm
        )
        # Should complete without error
        assert result.status == StageStatus.DONE

    def test_no_llm_uses_template_shortlist(self, tmp_path: Path) -> None:
        """Without LLM, falls back to template shortlist from candidates."""
        run_dir = tmp_path / "run"
        run_dir.mkdir()
        _write_candidates(run_dir, 20)
        stage_dir = run_dir / "stage-05"
        stage_dir.mkdir(parents=True, exist_ok=True)

        cfg = _make_config(tmp_path, screen_batch_size=2)

        result = rc_executor._execute_literature_screen(
            stage_dir, run_dir, cfg, AdapterBundle(), llm=None
        )
        assert result.status == StageStatus.DONE
        shortlist_path = stage_dir / "shortlist.jsonl"
        lines = shortlist_path.read_text().strip().splitlines()
        assert len(lines) == 15  # _MIN_SHORTLIST
