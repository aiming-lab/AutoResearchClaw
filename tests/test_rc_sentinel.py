# pyright: reportPrivateUsage=false
"""Tests for the sentinel watchdog and heartbeat system."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pytest

from researchclaw.pipeline import runner as rc_runner
from researchclaw.pipeline.stages import Stage


# ── Heartbeat writing tests ──


class TestHeartbeatWriting:
    def test_write_heartbeat_creates_file(self, tmp_path: Path) -> None:
        rc_runner._write_heartbeat(tmp_path, Stage.TOPIC_INIT, "run-hb-1")
        hb_path = tmp_path / "heartbeat.json"
        assert hb_path.exists()

    def test_heartbeat_contains_required_fields(self, tmp_path: Path) -> None:
        rc_runner._write_heartbeat(tmp_path, Stage.HYPOTHESIS_GEN, "run-hb-2")
        data = json.loads((tmp_path / "heartbeat.json").read_text())
        assert data["pid"] == os.getpid()
        assert data["last_stage"] == 8
        assert data["last_stage_name"] == "HYPOTHESIS_GEN"
        assert data["run_id"] == "run-hb-2"
        assert "timestamp" in data

    def test_heartbeat_updates_on_each_stage(self, tmp_path: Path) -> None:
        rc_runner._write_heartbeat(tmp_path, Stage.TOPIC_INIT, "run-1")
        data1 = json.loads((tmp_path / "heartbeat.json").read_text())
        rc_runner._write_heartbeat(tmp_path, Stage.PAPER_DRAFT, "run-1")
        data2 = json.loads((tmp_path / "heartbeat.json").read_text())
        assert data2["last_stage"] == 17
        assert data1["last_stage"] == 1


class TestHeartbeatInPipeline:
    def test_pipeline_writes_heartbeat_after_each_stage(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        from researchclaw.adapters import AdapterBundle
        from researchclaw.config import RCConfig
        from researchclaw.pipeline.executor import StageResult
        from researchclaw.pipeline.stages import StageStatus

        data = {
            "project": {"name": "hb-test", "mode": "docs-first"},
            "research": {"topic": "heartbeat testing"},
            "runtime": {"timezone": "UTC"},
            "notifications": {"channel": "local"},
            "knowledge_base": {"backend": "markdown", "root": str(tmp_path / "kb")},
            "openclaw_bridge": {},
            "llm": {
                "provider": "openai-compatible",
                "base_url": "http://localhost/v1",
                "api_key_env": "K",
                "api_key": "k",
            },
        }
        config = RCConfig.from_dict(data, project_root=tmp_path, check_paths=False)
        run_dir = tmp_path / "run"
        run_dir.mkdir()

        call_count = 0

        def mock_execute_stage(stage: Stage, **kwargs) -> StageResult:
            nonlocal call_count
            call_count += 1
            if call_count >= 3:
                return StageResult(
                    stage=stage, status=StageStatus.FAILED, artifacts=(), error="stop"
                )
            return StageResult(stage=stage, status=StageStatus.DONE, artifacts=("x.md",))

        monkeypatch.setattr(rc_runner, "execute_stage", mock_execute_stage)
        rc_runner.execute_pipeline(
            run_dir=run_dir,
            run_id="hb-test",
            config=config,
            adapters=AdapterBundle(),
        )
        hb_path = run_dir / "heartbeat.json"
        assert hb_path.exists()
        data_out = json.loads(hb_path.read_text())
        assert data_out["run_id"] == "hb-test"


# ── Sentinel script syntax check ──


class TestSentinelScript:
    def test_sentinel_script_exists(self) -> None:
        script = Path(__file__).parent.parent / "sentinel.sh"
        assert script.exists()

    def test_sentinel_script_is_valid_bash(self) -> None:
        script = Path(__file__).parent.parent / "sentinel.sh"
        result = subprocess.run(
            ["bash", "-n", str(script)],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, f"Bash syntax error: {result.stderr}"

    def test_sentinel_script_is_executable(self) -> None:
        script = Path(__file__).parent.parent / "sentinel.sh"
        assert os.access(script, os.X_OK)

    def test_sentinel_script_has_shebang(self) -> None:
        script = Path(__file__).parent.parent / "sentinel.sh"
        first_line = script.read_text(encoding="utf-8").splitlines()[0]
        assert first_line.startswith("#!/")

    def test_sentinel_prints_usage_on_no_args(self) -> None:
        script = Path(__file__).parent.parent / "sentinel.sh"
        result = subprocess.run(
            ["bash", str(script)],
            capture_output=True,
            text=True,
        )
        # Should fail because no run_dir argument provided
        assert result.returncode != 0
