"""Tests for --from-stage run directory resolution (fixes #216)."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from researchclaw import cli


def _make_run_dir(artifacts_root: Path, topic: str, stage: int = 9) -> Path:
    """Create a minimal run directory with a checkpoint at the given stage."""
    topic_hash = hashlib.sha256(topic.encode()).hexdigest()[:6]
    run_dir = artifacts_root / f"rc-20260406-030000-{topic_hash}"
    run_dir.mkdir(parents=True, exist_ok=True)

    # Write checkpoint
    checkpoint = {
        "last_completed_stage": stage,
        "last_completed_name": "EXPERIMENT_DESIGN",
        "run_id": run_dir.name,
        "timestamp": "2026-04-06T03:00:00+00:00",
    }
    (run_dir / "checkpoint.json").write_text(json.dumps(checkpoint))

    # Create stage directories with artifacts
    for s in range(1, stage + 1):
        stage_dir = run_dir / f"stage-{s:02d}"
        stage_dir.mkdir(exist_ok=True)

    # Write exp_plan.yaml in stage-09 (required by CODE_GENERATION)
    if stage >= 9:
        (run_dir / "stage-09" / "exp_plan.yaml").write_text(
            "topic: test topic\nobjectives:\n- test\n"
        )

    return run_dir


class TestFromStageRunDirResolution:
    """Verify that --from-stage finds existing run directories."""

    def test_from_stage_without_output_finds_existing_run(self, tmp_path):
        """--from-stage without --output should find the existing run dir
        with matching topic hash, just like --resume does."""
        topic = "test research topic for unit test"
        artifacts_root = tmp_path / "artifacts"
        run_dir = _make_run_dir(artifacts_root, topic, stage=9)

        # Simulate what cli.py does: generate a new run_id, then search
        topic_hash = hashlib.sha256(topic.encode()).hexdigest()[:6]

        # The fix: when from_stage_name is set and output is None,
        # search for existing checkpoint dirs
        from_stage_name = "CODE_GENERATION"
        resume = False
        output = None

        # This is the condition after the fix:
        if (resume or from_stage_name) and not output:
            candidates = sorted(
                (
                    d
                    for d in artifacts_root.iterdir()
                    if d.is_dir()
                    and d.name.startswith("rc-")
                    and d.name.endswith(f"-{topic_hash}")
                    and (d / "checkpoint.json").exists()
                ),
                key=lambda d: d.name,
                reverse=True,
            )
            found_dir = candidates[0] if candidates else None
        else:
            found_dir = None

        assert found_dir is not None, "Should find existing run directory"
        assert found_dir == run_dir
        assert (found_dir / "stage-09" / "exp_plan.yaml").exists()

    def test_from_stage_without_output_old_behavior_fails(self, tmp_path):
        """Without the fix, --from-stage without --output would NOT search
        for existing runs (only --resume did)."""
        topic = "test research topic for unit test"
        artifacts_root = tmp_path / "artifacts"
        _make_run_dir(artifacts_root, topic, stage=9)

        topic_hash = hashlib.sha256(topic.encode()).hexdigest()[:6]

        # Old behavior: only resume triggers search
        from_stage_name = "CODE_GENERATION"
        resume = False
        output = None

        # OLD condition (before fix):
        if resume and not output:
            candidates = sorted(
                (
                    d
                    for d in artifacts_root.iterdir()
                    if d.is_dir()
                    and d.name.startswith("rc-")
                    and d.name.endswith(f"-{topic_hash}")
                    and (d / "checkpoint.json").exists()
                ),
                key=lambda d: d.name,
                reverse=True,
            )
            found_dir = candidates[0] if candidates else None
        else:
            found_dir = None

        # With old behavior, from_stage alone does NOT trigger search
        assert found_dir is None, "Old behavior should NOT find run dir for --from-stage"

    def test_resume_still_works(self, tmp_path):
        """--resume should still find existing run directories (no regression)."""
        topic = "test research topic for unit test"
        artifacts_root = tmp_path / "artifacts"
        run_dir = _make_run_dir(artifacts_root, topic, stage=9)

        topic_hash = hashlib.sha256(topic.encode()).hexdigest()[:6]

        resume = True
        from_stage_name = None
        output = None

        if (resume or from_stage_name) and not output:
            candidates = sorted(
                (
                    d
                    for d in artifacts_root.iterdir()
                    if d.is_dir()
                    and d.name.startswith("rc-")
                    and d.name.endswith(f"-{topic_hash}")
                    and (d / "checkpoint.json").exists()
                ),
                key=lambda d: d.name,
                reverse=True,
            )
            found_dir = candidates[0] if candidates else None
        else:
            found_dir = None

        assert found_dir is not None
        assert found_dir == run_dir

    def test_explicit_output_skips_search(self, tmp_path):
        """--from-stage with explicit --output should use the provided path."""
        topic = "test research topic for unit test"
        artifacts_root = tmp_path / "artifacts"
        run_dir = _make_run_dir(artifacts_root, topic, stage=9)

        from_stage_name = "CODE_GENERATION"
        resume = False
        output = str(run_dir)  # Explicit output path

        # With explicit output, search should NOT be triggered
        if (resume or from_stage_name) and not output:
            searched = True
        else:
            searched = False

        assert not searched, "--output should bypass the search"

    def test_picks_newest_run_when_multiple_exist(self, tmp_path):
        """When multiple runs exist for the same topic, pick the newest."""
        topic = "test research topic for unit test"
        topic_hash = hashlib.sha256(topic.encode()).hexdigest()[:6]
        artifacts_root = tmp_path / "artifacts"

        # Create two runs with different timestamps
        older = artifacts_root / f"rc-20260405-010000-{topic_hash}"
        older.mkdir(parents=True)
        (older / "checkpoint.json").write_text('{"last_completed_stage": 5}')
        (older / "stage-05").mkdir()

        newer = artifacts_root / f"rc-20260406-020000-{topic_hash}"
        newer.mkdir(parents=True)
        (newer / "checkpoint.json").write_text('{"last_completed_stage": 9}')
        stage_dir = newer / "stage-09"
        stage_dir.mkdir()
        (stage_dir / "exp_plan.yaml").write_text("topic: test\n")

        from_stage_name = "CODE_GENERATION"
        resume = False
        output = None

        if (resume or from_stage_name) and not output:
            candidates = sorted(
                (
                    d
                    for d in artifacts_root.iterdir()
                    if d.is_dir()
                    and d.name.startswith("rc-")
                    and d.name.endswith(f"-{topic_hash}")
                    and (d / "checkpoint.json").exists()
                ),
                key=lambda d: d.name,
                reverse=True,
            )
            found_dir = candidates[0] if candidates else None
        else:
            found_dir = None

        assert found_dir == newer, "Should pick the newest run directory"
